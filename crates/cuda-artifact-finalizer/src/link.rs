/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::nvvm::{loaded_tool_digest, report_changed_tool};
use crate::options::{FinalizationOptions, FinalizerOutput, NamedInput};
use crate::provenance::{
    StableDigest, digest_file_handle, linker_provenance_digest, recipe_digest,
    with_revalidated_tool_identity,
};
use crate::validation::is_valid_cubin;
use crate::{FinalizerError, validate_name};
use nvjitlink_sys::{InputType, LibNvJitLink, Linker};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

struct LoadedLinkerTool {
    library: Arc<LibNvJitLink>,
    digest: Option<[u8; 32]>,
}

static LINKER_TOOL: OnceLock<Arc<LoadedLinkerTool>> = OnceLock::new();
static LINKER_TOOL_LOAD: OnceLock<Mutex<()>> = OnceLock::new();

/// Driver-independent ordered LTOIR/PTX linker.
#[derive(Clone)]
pub struct LtoLinker {
    tool: Arc<LoadedLinkerTool>,
}

impl LtoLinker {
    /// Discover and pin nvJitLink without loading libNVVM or the CUDA Driver.
    pub fn discover() -> Result<Self, FinalizerError> {
        Ok(Self {
            tool: load_linker_tool()?,
        })
    }

    /// Digest of the exact loaded nvJitLink file, when its identity is known.
    pub fn nvjitlink_digest(&self) -> Option<[u8; 32]> {
        let digest = self.tool.digest?;
        if self.tool.library.loaded_file_if_unchanged().is_some() {
            Some(digest)
        } else {
            report_changed_tool("nvJitLink");
            None
        }
    }

    /// Exact route provenance, or `None` when the loaded DSO is unidentifiable.
    pub fn provenance_digest(&self) -> Option<[u8; 32]> {
        self.nvjitlink_digest()
            .map(|digest| linker_provenance_digest(&digest))
    }

    /// Link one or more LTOIR modules in the exact supplied order.
    pub fn link_ltoir(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Result<Vec<u8>, FinalizerError> {
        self.link_inputs(inputs, LinkInputKind::Ltoir, options, output)
    }

    /// Link one or more PTX modules to a cubin in the exact supplied order.
    pub fn link_ptx(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        self.link_inputs(inputs, LinkInputKind::Ptx, options, FinalizerOutput::Cubin)
    }

    /// Link one or more ptxas relocatable CUDA objects as cubin inputs.
    ///
    /// nvJitLink's `Object` input kind means a host object. CUDA device
    /// relocatables emitted by `ptxas -c` use its `Cubin` input kind.
    pub fn link_cubin_inputs(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        self.link_inputs(
            inputs,
            LinkInputKind::Cubin,
            options,
            FinalizerOutput::Cubin,
        )
    }

    pub(crate) fn link_ptx_streaming<F>(
        &self,
        options: &FinalizationOptions,
        add_inputs: F,
    ) -> Result<(Vec<u8>, Duration), FinalizerError>
    where
        F: FnOnce(&mut StreamingLinkSink<'_, '_>) -> Result<(), FinalizerError>,
    {
        self.link_streaming(LinkInputKind::Ptx, options, add_inputs)
    }

    pub(crate) fn link_cubin_streaming<F>(
        &self,
        options: &FinalizationOptions,
        add_inputs: F,
    ) -> Result<(Vec<u8>, Duration), FinalizerError>
    where
        F: FnOnce(&mut StreamingLinkSink<'_, '_>) -> Result<(), FinalizerError>,
    {
        self.link_streaming(LinkInputKind::Cubin, options, add_inputs)
    }

    fn link_streaming<F>(
        &self,
        input_kind: LinkInputKind,
        options: &FinalizationOptions,
        add_inputs: F,
    ) -> Result<(Vec<u8>, Duration), FinalizerError>
    where
        F: FnOnce(&mut StreamingLinkSink<'_, '_>) -> Result<(), FinalizerError>,
    {
        debug_assert!(matches!(
            input_kind,
            LinkInputKind::Ptx | LinkInputKind::Cubin
        ));
        with_revalidated_tool_identity(
            "nvJitLink",
            self.tool.digest,
            || current_linker_tool_digest(&self.tool),
            || {
                let option_storage = match input_kind {
                    LinkInputKind::Ptx => options.nvjitlink_ptx_options(),
                    LinkInputKind::Cubin => options.nvjitlink_cubin_options(),
                    LinkInputKind::Ltoir => unreachable!("LTOIR does not use streaming linking"),
                };
                let option_refs = option_storage
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                let mut linker = Linker::new(&self.tool.library, &option_refs)?;
                let input_count = {
                    let mut sink = StreamingLinkSink {
                        linker: &mut linker,
                        input_count: 0,
                        input_kind,
                    };
                    add_inputs(&mut sink)?;
                    sink.input_count
                };
                if input_count == 0 {
                    return Err(FinalizerError::NoLinkInputs);
                }
                let started = std::time::Instant::now();
                let image = linker.finish()?;
                let elapsed = started.elapsed();
                if !is_valid_cubin(&image) {
                    return Err(FinalizerError::InvalidCubin);
                }
                Ok((image, elapsed))
            },
        )
    }

    fn link_inputs(
        &self,
        inputs: &[NamedInput<'_>],
        input_kind: LinkInputKind,
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Result<Vec<u8>, FinalizerError> {
        validate_inputs(inputs, input_kind)?;
        with_revalidated_tool_identity(
            "nvJitLink",
            self.tool.digest,
            || current_linker_tool_digest(&self.tool),
            || {
                let option_storage = match input_kind {
                    LinkInputKind::Ltoir => options.nvjitlink_ltoir_options(output),
                    LinkInputKind::Ptx => options.nvjitlink_ptx_options(),
                    LinkInputKind::Cubin => options.nvjitlink_cubin_options(),
                };
                let option_refs = option_storage
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>();
                let mut linker = Linker::new(&self.tool.library, &option_refs)?;
                for input in inputs {
                    linker.add(input_kind.nvjitlink_type(), input.bytes, input.name)?;
                }
                let image = match output {
                    FinalizerOutput::Cubin => linker.finish()?,
                    FinalizerOutput::Ptx => linker.finish_ptx()?,
                };
                if output == FinalizerOutput::Cubin && !is_valid_cubin(&image) {
                    return Err(FinalizerError::InvalidCubin);
                }
                if output == FinalizerOutput::Ptx && image.is_empty() {
                    return Err(FinalizerError::EmptyPtx);
                }
                Ok(image)
            },
        )
    }

    /// Digest every semantic input to an ordered LTOIR link.
    pub fn artifact_digest(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
        output: FinalizerOutput,
    ) -> Option<[u8; 32]> {
        let nvjitlink = self.nvjitlink_digest()?;
        Some(ltoir_artifact_digest_parts(
            inputs, options, output, &nvjitlink,
        ))
    }

    /// Digest every semantic input to an ordered PTX-to-cubin link.
    pub fn ptx_artifact_digest(
        &self,
        inputs: &[NamedInput<'_>],
        options: &FinalizationOptions,
    ) -> Option<[u8; 32]> {
        let nvjitlink = self.nvjitlink_digest()?;
        Some(ptx_artifact_digest_parts(inputs, options, &nvjitlink))
    }
}

pub(crate) struct StreamingLinkSink<'linker, 'tool> {
    linker: &'linker mut Linker<'tool>,
    input_count: usize,
    input_kind: LinkInputKind,
}

impl StreamingLinkSink<'_, '_> {
    pub(crate) fn add(&mut self, name: &str, bytes: &[u8]) -> Result<(), FinalizerError> {
        validate_input(&NamedInput::new(name, bytes), self.input_kind)?;
        self.linker
            .add(self.input_kind.nvjitlink_type(), bytes, name)?;
        self.input_count += 1;
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum LinkInputKind {
    Ltoir,
    Ptx,
    Cubin,
}

impl LinkInputKind {
    fn nvjitlink_type(self) -> InputType {
        match self {
            Self::Ltoir => InputType::Ltoir,
            Self::Ptx => InputType::Ptx,
            Self::Cubin => InputType::Cubin,
        }
    }
}

fn current_linker_tool_digest(tool: &LoadedLinkerTool) -> Option<[u8; 32]> {
    let file = tool.library.loaded_file_if_unchanged()?;
    digest_file_handle(file).ok()
}

fn validate_inputs(
    inputs: &[NamedInput<'_>],
    input_kind: LinkInputKind,
) -> Result<(), FinalizerError> {
    if inputs.is_empty() {
        return Err(FinalizerError::NoLinkInputs);
    }
    for input in inputs {
        validate_input(input, input_kind)?;
    }
    Ok(())
}

fn validate_input(input: &NamedInput<'_>, input_kind: LinkInputKind) -> Result<(), FinalizerError> {
    validate_name(input.name)?;
    if input.bytes.is_empty() {
        return Err(FinalizerError::EmptyInput {
            name: input.name.to_string(),
        });
    }
    if matches!(input_kind, LinkInputKind::Ptx)
        && let Some(offset) = input.bytes.iter().position(|byte| *byte == 0)
        && offset + 1 != input.bytes.len()
    {
        return Err(FinalizerError::InteriorNulPtx {
            name: input.name.to_string(),
            offset,
        });
    }
    Ok(())
}

fn load_linker_tool() -> Result<Arc<LoadedLinkerTool>, FinalizerError> {
    if let Some(loaded) = LINKER_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }
    let _guard = LINKER_TOOL_LOAD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(loaded) = LINKER_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }

    let library = LibNvJitLink::load_for_cache()?;
    let digest = loaded_tool_digest("nvJitLink", library.loaded_file_if_unchanged());
    let digest = if digest.is_some() && library.loaded_file_if_unchanged().is_none() {
        report_changed_tool("nvJitLink");
        None
    } else {
        digest
    };
    let loaded = Arc::new(LoadedLinkerTool {
        library: Arc::new(library),
        digest,
    });
    let _ = LINKER_TOOL.set(Arc::clone(&loaded));
    Ok(loaded)
}

pub(crate) fn ltoir_artifact_digest_parts(
    inputs: &[NamedInput<'_>],
    options: &FinalizationOptions,
    output: FinalizerOutput,
    nvjitlink_digest: &[u8; 32],
) -> [u8; 32] {
    artifact_digest_parts(
        inputs,
        LinkInputKind::Ltoir,
        options,
        output,
        nvjitlink_digest,
    )
}

pub(crate) fn ptx_artifact_digest_parts(
    inputs: &[NamedInput<'_>],
    options: &FinalizationOptions,
    nvjitlink_digest: &[u8; 32],
) -> [u8; 32] {
    artifact_digest_parts(
        inputs,
        LinkInputKind::Ptx,
        options,
        FinalizerOutput::Cubin,
        nvjitlink_digest,
    )
}

fn artifact_digest_parts(
    inputs: &[NamedInput<'_>],
    input_kind: LinkInputKind,
    options: &FinalizationOptions,
    output: FinalizerOutput,
    nvjitlink_digest: &[u8; 32],
) -> [u8; 32] {
    let output_name = match output {
        FinalizerOutput::Cubin => b"elf-cubin".as_slice(),
        FinalizerOutput::Ptx => b"ptx".as_slice(),
    };
    let (route, input_name_field, input_bytes_field) = match input_kind {
        LinkInputKind::Ltoir => ("ltoir-to-output", "ltoir-name", "ltoir"),
        LinkInputKind::Ptx => ("ptx-to-output", "ptx-name", "ptx"),
        LinkInputKind::Cubin => ("cubin-to-output", "cubin-name", "cubin"),
    };
    let mut digest = StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", route.as_bytes())
        .field("output", output_name);
    for input in inputs {
        digest = digest
            .field(input_name_field, input.name.as_bytes())
            .field(input_bytes_field, input.bytes);
    }
    let link_options = match input_kind {
        LinkInputKind::Ltoir => options.nvjitlink_ltoir_options(output),
        LinkInputKind::Ptx => options.nvjitlink_ptx_options(),
        LinkInputKind::Cubin => options.nvjitlink_cubin_options(),
    };
    for option in link_options {
        digest = digest.field("nvjitlink-option", option.as_bytes());
    }
    digest
        .field("libnvjitlink-sha256", nvjitlink_digest)
        .finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_digest_preserves_input_order_names_output_and_policy() {
        let options = FinalizationOptions::new("sm_120".parse().unwrap());
        let a = NamedInput::new("a.ltoir", b"a");
        let b = NamedInput::new("b.ltoir", b"b");
        let baseline =
            ltoir_artifact_digest_parts(&[a, b], &options, FinalizerOutput::Cubin, &[7; 32]);
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(&[b, a], &options, FinalizerOutput::Cubin, &[7; 32])
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[NamedInput::new("renamed.ltoir", b"a"), b],
                &options,
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[a, b],
                &FinalizationOptions::new("sm_90".parse().unwrap()),
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[a, b],
                &options
                    .clone()
                    .with_debug_policy(crate::DebugPolicy::LineTables),
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(&[a, b], &options, FinalizerOutput::Cubin, &[8; 32])
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(&[a, b], &options, FinalizerOutput::Ptx, &[7; 32])
        );
        assert_ne!(
            baseline,
            ltoir_artifact_digest_parts(
                &[a, b],
                &options.clone().with_fma_contraction(false),
                FinalizerOutput::Cubin,
                &[7; 32]
            )
        );
        assert_ne!(
            baseline,
            ptx_artifact_digest_parts(&[a, b], &options, &[7; 32]),
            "LTOIR and PTX inputs must never share a cache identity"
        );
    }

    #[test]
    fn input_validation_rejects_zero_inputs_empty_data_and_nul_names() {
        assert!(matches!(
            validate_inputs(&[], LinkInputKind::Ltoir),
            Err(FinalizerError::NoLinkInputs)
        ));
        assert!(matches!(
            validate_inputs(&[NamedInput::new("empty", b"")], LinkInputKind::Ltoir),
            Err(FinalizerError::EmptyInput { .. })
        ));
        assert!(matches!(
            validate_inputs(&[NamedInput::new("bad\0name", b"x")], LinkInputKind::Ltoir),
            Err(FinalizerError::InvalidInputName { .. })
        ));
    }

    #[test]
    fn ptx_validation_allows_only_an_optional_trailing_nul() {
        for bytes in [&b"ptx"[..], &b"ptx\0"[..]] {
            validate_inputs(&[NamedInput::new("kernel.ptx", bytes)], LinkInputKind::Ptx).unwrap();
        }

        for (bytes, expected_offset) in [
            (&b"ptx\0ignored"[..], 3),
            (&b"ptx\0\0"[..], 3),
            (&b"\0ptx"[..], 0),
        ] {
            let error =
                validate_inputs(&[NamedInput::new("kernel.ptx", bytes)], LinkInputKind::Ptx)
                    .unwrap_err();
            assert!(matches!(
                error,
                FinalizerError::InteriorNulPtx {
                    ref name,
                    offset
                } if name == "kernel.ptx" && offset == expected_offset
            ));
        }

        validate_inputs(
            &[NamedInput::new("kernel.ltoir", b"lto\0ir")],
            LinkInputKind::Ltoir,
        )
        .unwrap();
    }
}
