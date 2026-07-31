/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA graph construction, capture, and executable graph management.
//!
//! [`CudaStream::begin_capture`] records subsequent stream work instead of
//! executing it. Finishing the returned [`CudaStreamCapture`] yields a
//! [`CudaGraph`], which can be instantiated once and launched repeatedly
//! through [`CudaGraphExec`]. Empty graphs created with [`CudaGraph::new`] can
//! also contain conditional WHILE nodes built with
//! [`CudaGraph::add_while_node`].

use crate::context::CudaContext;
use crate::error::{DriverError, IntoResult};
use crate::stream::CudaStream;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::rc::Rc;
use std::sync::Arc;

/// CUDA's device-visible representation of a graph conditional handle.
///
/// This is an unsigned 64-bit token in both the Driver API and device code, so
/// values returned by [`CudaGraphWhileNode::conditional_handle`] can be passed
/// directly to a kernel parameter of type
/// [`cuda_device::graph::CudaGraphConditionalHandle`](https://docs.rs/cuda-device/latest/cuda_device/graph/type.CudaGraphConditionalHandle.html).
pub type CudaGraphConditionalHandle = u64;

/// Options used when creating a graph conditional handle.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaGraphConditionalOptions {
    default_launch_value: u32,
    assign_default_on_launch: bool,
}

impl CudaGraphConditionalOptions {
    /// Create options with the value CUDA receives as `defaultLaunchValue`.
    pub const fn new(default_launch_value: u32) -> Self {
        Self {
            default_launch_value,
            assign_default_on_launch: false,
        }
    }

    /// Reassign `default_launch_value` at the beginning of every graph launch.
    ///
    /// A WHILE loop normally uses a nonzero default so every replay enters the
    /// loop before a body kernel decides whether another iteration is needed.
    pub const fn assign_default_on_launch(mut self) -> Self {
        self.assign_default_on_launch = true;
        self
    }

    /// Return the value passed to CUDA as `defaultLaunchValue`.
    pub const fn default_launch_value(self) -> u32 {
        self.default_launch_value
    }

    /// Whether CUDA resets the condition to the default on every graph launch.
    pub const fn assigns_default_on_launch(self) -> bool {
        self.assign_default_on_launch
    }

    fn flags(self) -> u32 {
        if self.assign_default_on_launch {
            cuda_bindings::CU_GRAPH_COND_ASSIGN_DEFAULT
        } else {
            0
        }
    }
}

/// A graph-owned conditional WHILE node and its CUDA-owned body graph.
///
/// The body graph may be populated with raw graph node APIs through
/// [`crate::sys`] or with `cuStreamBeginCaptureToGraph`. CUDA owns the body
/// graph; it must not be destroyed separately. This view mutably borrows the
/// parent graph so the parent cannot be instantiated or dropped while its raw
/// body handle is being used.
#[derive(Debug)]
pub struct CudaGraphWhileNode<'graph> {
    cu_node: cuda_bindings::CUgraphNode,
    body_graph: cuda_bindings::CUgraph,
    conditional_handle: CudaGraphConditionalHandle,
    ctx: Arc<CudaContext>,
    _graph: PhantomData<&'graph mut CudaGraph>,
}

impl<'graph> CudaGraphWhileNode<'graph> {
    /// Return the raw conditional-node handle.
    pub fn cu_node(&self) -> cuda_bindings::CUgraphNode {
        self.cu_node
    }

    /// Return the CUDA-owned graph that executes for each WHILE iteration.
    pub fn body_graph(&self) -> cuda_bindings::CUgraph {
        self.body_graph
    }

    /// Return the token that body kernels use to update the loop condition.
    pub fn conditional_handle(&self) -> CudaGraphConditionalHandle {
        self.conditional_handle
    }

    /// Begin thread-local stream capture directly into this node's body graph.
    ///
    /// Operations subsequently submitted to `stream` become nodes in the WHILE
    /// body; the first captured operations have no external dependencies. Call
    /// [`CudaGraphBodyCapture::finish`] after submitting one loop iteration.
    /// The guard borrows this node until capture ends and never destroys the
    /// CUDA-owned body graph, including during unwinding.
    ///
    /// CUDA rejects the legacy/default stream and a stream from another
    /// context. This API requires CUDA Toolkit 12.3 or newer.
    pub fn begin_body_capture<'capture>(
        &'capture mut self,
        stream: &'capture CudaStream,
    ) -> Result<CudaGraphBodyCapture<'capture, 'graph>, DriverError> {
        if self.ctx.as_ref() != stream.context().as_ref() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        self.ctx.bind_to_thread()?;
        unsafe {
            cuda_bindings::cuStreamBeginCaptureToGraph(
                stream.cu_stream(),
                self.body_graph,
                std::ptr::null(),
                std::ptr::null(),
                0,
                cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        }
        .result()?;
        Ok(CudaGraphBodyCapture {
            stream,
            node: self,
            active: true,
            _same_thread: PhantomData,
        })
    }
}

/// A CUDA graph captured from a stream or created for manual construction.
#[derive(Debug)]
pub struct CudaGraph {
    cu_graph: cuda_bindings::CUgraph,
    ctx: Arc<CudaContext>,
}

/// A reusable executable instance of a CUDA graph.
#[derive(Debug)]
pub struct CudaGraphExec {
    cu_graph_exec: cuda_bindings::CUgraphExec,
    ctx: Arc<CudaContext>,
}

/// Active thread-local capture on one CUDA stream.
///
/// Call [`finish`](Self::finish) to retain the captured graph. Dropping an
/// unfinished capture terminates capture and destroys any graph returned by
/// the driver, leaving the stream usable for ordinary submissions.
#[derive(Debug)]
pub struct CudaStreamCapture<'a> {
    stream: &'a CudaStream,
    active: bool,
    _same_thread: PhantomData<Rc<()>>,
}

/// Active thread-local capture into a CUDA-owned conditional body graph.
///
/// Finishing or dropping this guard ends stream capture but never destroys the
/// body graph. Captured nodes remain owned by the parent conditional node.
#[derive(Debug)]
pub struct CudaGraphBodyCapture<'capture, 'graph> {
    stream: &'capture CudaStream,
    node: &'capture mut CudaGraphWhileNode<'graph>,
    active: bool,
    _same_thread: PhantomData<Rc<()>>,
}

impl CudaStream {
    /// Begin thread-local graph capture on this non-legacy stream.
    ///
    /// All operations subsequently submitted to this stream are captured
    /// until the returned guard is finished or dropped. CUDA requires capture
    /// to end on the same host thread; the guard is deliberately neither
    /// `Send` nor `Sync`.
    pub fn begin_capture(&self) -> Result<CudaStreamCapture<'_>, DriverError> {
        self.context().bind_to_thread()?;
        unsafe {
            cuda_bindings::cuStreamBeginCapture_v2(
                self.cu_stream(),
                cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        }
        .result()?;
        Ok(CudaStreamCapture {
            stream: self,
            active: true,
            _same_thread: PhantomData,
        })
    }
}

impl CudaStreamCapture<'_> {
    /// End capture and return the recorded graph.
    pub fn finish(mut self) -> Result<CudaGraph, DriverError> {
        self.stream.context().bind_to_thread()?;
        let mut cu_graph = MaybeUninit::uninit();
        let result = unsafe {
            cuda_bindings::cuStreamEndCapture(self.stream.cu_stream(), cu_graph.as_mut_ptr())
        };
        self.active = false;
        result.result()?;
        let cu_graph = unsafe { cu_graph.assume_init() };
        if cu_graph.is_null() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_HANDLE,
            ));
        }
        Ok(CudaGraph {
            cu_graph,
            ctx: self.stream.context().clone(),
        })
    }
}

impl Drop for CudaStreamCapture<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        self.stream
            .context()
            .record_err(self.stream.context().bind_to_thread());
        let mut cu_graph = MaybeUninit::uninit();
        let end = unsafe {
            cuda_bindings::cuStreamEndCapture(self.stream.cu_stream(), cu_graph.as_mut_ptr())
        };
        self.stream.context().record_err(end.result());
        if end == cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            let cu_graph = unsafe { cu_graph.assume_init() };
            if !cu_graph.is_null() {
                self.stream
                    .context()
                    .record_err(unsafe { cuda_bindings::cuGraphDestroy(cu_graph).result() });
            }
        }
    }
}

impl CudaGraphBodyCapture<'_, '_> {
    /// End capture, leaving the captured nodes in the WHILE body graph.
    pub fn finish(mut self) -> Result<(), DriverError> {
        self.stream.context().bind_to_thread()?;
        let mut captured_graph = std::ptr::null_mut();
        let result = unsafe {
            cuda_bindings::cuStreamEndCapture(self.stream.cu_stream(), &mut captured_graph)
        };
        self.active = false;
        result.result()?;
        validate_captured_body_graph(self.node.body_graph, captured_graph)
    }
}

impl Drop for CudaGraphBodyCapture<'_, '_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        self.stream
            .context()
            .record_err(self.stream.context().bind_to_thread());
        let mut captured_graph = std::ptr::null_mut();
        let end = unsafe {
            cuda_bindings::cuStreamEndCapture(self.stream.cu_stream(), &mut captured_graph)
        };
        self.stream.context().record_err(end.result());
        if end == cuda_bindings::cudaError_enum_CUDA_SUCCESS {
            self.stream
                .context()
                .record_err(validate_captured_body_graph(
                    self.node.body_graph,
                    captured_graph,
                ));
        }
    }
}

impl CudaGraph {
    /// Create an empty graph for manual node construction.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let mut cu_graph = MaybeUninit::uninit();
        unsafe { cuda_bindings::cuGraphCreate(cu_graph.as_mut_ptr(), 0) }.result()?;
        Ok(Self {
            cu_graph: unsafe { cu_graph.assume_init() },
            ctx: Arc::clone(ctx),
        })
    }

    /// Return the raw CUDA graph handle.
    pub fn cu_graph(&self) -> cuda_bindings::CUgraph {
        self.cu_graph
    }

    /// Return the context that owns every node in this graph.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Add a conditional WHILE node and return its empty body graph.
    ///
    /// `dependencies` contains raw nodes in this parent graph that must finish
    /// before the WHILE node begins. The returned body graph can contain kernel,
    /// empty, child-graph, memset, memcpy, and nested conditional nodes, subject
    /// to CUDA's conditional-graph restrictions.
    ///
    /// The conditional-handle and arbitrary-node APIs used here require CUDA
    /// Toolkit 12.3 or newer. CUDA does not provide a way to destroy a
    /// conditional handle independently: if handle creation succeeds but node
    /// creation fails (for example because a dependency belongs to another
    /// graph), this graph may no longer be instantiable and should be discarded.
    pub fn add_while_node(
        &mut self,
        dependencies: &[cuda_bindings::CUgraphNode],
        options: CudaGraphConditionalOptions,
    ) -> Result<CudaGraphWhileNode<'_>, DriverError> {
        self.ctx.bind_to_thread()?;

        let mut conditional_handle = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGraphConditionalHandleCreate(
                conditional_handle.as_mut_ptr(),
                self.cu_graph,
                self.ctx.cu_ctx(),
                options.default_launch_value(),
                options.flags(),
            )
        }
        .result()?;
        let conditional_handle = unsafe { conditional_handle.assume_init() };

        let mut body_graph = std::ptr::null_mut();
        let mut params =
            conditional_while_node_params(conditional_handle, self.ctx.cu_ctx(), &mut body_graph);
        let mut cu_node = MaybeUninit::uninit();
        let dependency_ptr = if dependencies.is_empty() {
            std::ptr::null()
        } else {
            dependencies.as_ptr()
        };
        unsafe {
            cuda_bindings::cuGraphAddNode(
                cu_node.as_mut_ptr(),
                self.cu_graph,
                dependency_ptr,
                dependencies.len(),
                &mut params,
            )
        }
        .result()?;

        Ok(CudaGraphWhileNode {
            cu_node: unsafe { cu_node.assume_init() },
            body_graph,
            conditional_handle,
            ctx: Arc::clone(&self.ctx),
            _graph: PhantomData,
        })
    }

    /// Instantiate this graph for repeated host launches.
    ///
    /// The source graph is consumed because the executable graph is
    /// self-contained after successful instantiation.
    pub fn instantiate(self) -> Result<CudaGraphExec, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut cu_graph_exec = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGraphInstantiateWithFlags(cu_graph_exec.as_mut_ptr(), self.cu_graph, 0)
        }
        .result()?;
        Ok(CudaGraphExec {
            cu_graph_exec: unsafe { cu_graph_exec.assume_init() },
            ctx: self.ctx.clone(),
        })
    }
}

fn conditional_while_node_params(
    handle: cuda_bindings::CUgraphConditionalHandle,
    ctx: cuda_bindings::CUcontext,
    body_graph: &mut cuda_bindings::CUgraph,
) -> cuda_bindings::CUgraphNodeParams {
    // CUDA requires every reserved byte and every byte after the selected
    // union member to be zero. Initialize the complete tagged union before
    // installing the conditional member.
    let mut params =
        unsafe { MaybeUninit::<cuda_bindings::CUgraphNodeParams>::zeroed().assume_init() };
    params.type_ = cuda_bindings::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_CONDITIONAL;
    params.__bindgen_anon_1.conditional = cuda_bindings::CUDA_CONDITIONAL_NODE_PARAMS {
        handle,
        type_: cuda_bindings::CUgraphConditionalNodeType_enum_CU_GRAPH_COND_TYPE_WHILE,
        size: 1,
        phGraph_out: body_graph,
        ctx,
    };
    params
}

fn validate_captured_body_graph(
    expected: cuda_bindings::CUgraph,
    captured: cuda_bindings::CUgraph,
) -> Result<(), DriverError> {
    if captured == expected {
        Ok(())
    } else {
        Err(DriverError(
            cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_HANDLE,
        ))
    }
}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { cuda_bindings::cuGraphDestroy(self.cu_graph).result() });
    }
}

impl CudaGraphExec {
    /// Return the raw executable graph handle.
    pub fn cu_graph_exec(&self) -> cuda_bindings::CUgraphExec {
        self.cu_graph_exec
    }

    /// Return the CUDA context that owns this executable graph.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Upload graph resources without executing the graph.
    pub fn upload(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.validate_stream_context(stream)?;
        unsafe { cuda_bindings::cuGraphUpload(self.cu_graph_exec, stream.cu_stream()) }.result()
    }

    /// Enqueue one execution of this graph on `stream`.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.validate_stream_context(stream)?;
        unsafe { cuda_bindings::cuGraphLaunch(self.cu_graph_exec, stream.cu_stream()) }.result()
    }

    fn validate_stream_context(&self, stream: &CudaStream) -> Result<(), DriverError> {
        if self.ctx.as_ref() != stream.context().as_ref() {
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_CONTEXT,
            ));
        }
        self.ctx.bind_to_thread()
    }
}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        self.ctx.record_err(self.ctx.bind_to_thread());
        self.ctx
            .record_err(unsafe { cuda_bindings::cuGraphExecDestroy(self.cu_graph_exec).result() });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conditional_options_map_only_the_supported_driver_flag() {
        let persistent = CudaGraphConditionalOptions::new(7);
        assert_eq!(persistent.default_launch_value(), 7);
        assert!(!persistent.assigns_default_on_launch());
        assert_eq!(persistent.flags(), 0);

        let reset = persistent.assign_default_on_launch();
        assert!(reset.assigns_default_on_launch());
        assert_eq!(reset.flags(), cuda_bindings::CU_GRAPH_COND_ASSIGN_DEFAULT);
    }

    #[test]
    fn while_node_params_select_one_body_and_zero_reserved_storage() {
        let handle = 0x0123_4567_89ab_cdef;
        let ctx = 0x1357usize as cuda_bindings::CUcontext;
        let mut body_graph = std::ptr::null_mut();
        let body_graph_out = std::ptr::addr_of_mut!(body_graph);
        let params = conditional_while_node_params(handle, ctx, &mut body_graph);

        assert_eq!(
            params.type_,
            cuda_bindings::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_CONDITIONAL
        );
        assert_eq!(params.reserved0, [0; 3]);
        assert_eq!(params.reserved2, 0);

        let conditional = unsafe { params.__bindgen_anon_1.conditional };
        assert_eq!(conditional.handle, handle);
        assert_eq!(
            conditional.type_,
            cuda_bindings::CUgraphConditionalNodeType_enum_CU_GRAPH_COND_TYPE_WHILE
        );
        assert_eq!(conditional.size, 1);
        assert_eq!(conditional.phGraph_out, body_graph_out);
        assert_eq!(conditional.ctx, ctx);

        let storage = unsafe { params.__bindgen_anon_1.reserved1 };
        let used_words = std::mem::size_of::<cuda_bindings::CUDA_CONDITIONAL_NODE_PARAMS>()
            .div_ceil(std::mem::size_of::<i64>());
        assert!(storage[used_words..].iter().all(|word| *word == 0));
    }

    #[test]
    fn body_capture_accepts_only_the_cuda_owned_body_graph() {
        let body = 0x2468usize as cuda_bindings::CUgraph;
        let other = 0x1357usize as cuda_bindings::CUgraph;

        assert!(validate_captured_body_graph(body, body).is_ok());
        assert_eq!(
            validate_captured_body_graph(body, other),
            Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_HANDLE
            ))
        );
        assert_eq!(
            validate_captured_body_graph(body, std::ptr::null_mut()),
            Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_HANDLE
            ))
        );
    }
}
