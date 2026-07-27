/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA graph capture and executable graph management.
//!
//! [`CudaStream::begin_capture`] records subsequent stream work instead of
//! executing it. Finishing the returned [`CudaStreamCapture`] yields a
//! [`CudaGraph`], which can be instantiated once and launched repeatedly
//! through [`CudaGraphExec`].

use crate::context::CudaContext;
use crate::error::{DriverError, IntoResult};
use crate::stream::CudaStream;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::rc::Rc;
use std::sync::Arc;

/// An immutable CUDA graph captured from one stream.
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

impl CudaGraph {
    /// Return the raw CUDA graph handle.
    pub fn cu_graph(&self) -> cuda_bindings::CUgraph {
        self.cu_graph
    }

    /// Return the context that owns every node in this graph.
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
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
