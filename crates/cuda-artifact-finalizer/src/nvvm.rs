/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::options::FinalizationOptions;
use crate::provenance::{
    StableDigest, compiler_provenance_digest, digest_bytes, digest_file_handle, recipe_digest,
    with_revalidated_tool_identity,
};
use crate::{FinalizerError, validate_name};
use libnvvm_sys::{LibNvvm, Program, find_libdevice};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, OnceLock};

const DEFERRED_INLINE_CANDIDATE_PREFIX: &str = "; cuda-oxide-device-link-inline-candidate @";
const MAX_DEFERRED_INLINE_TARGETS: usize = 16;
const MAX_DEFERRED_INLINE_CALL_SITES: usize = 64;

struct LoadedNvvmTool {
    library: Arc<LibNvvm>,
    digest: Option<[u8; 32]>,
}

static NVVM_TOOL: OnceLock<Arc<LoadedNvvmTool>> = OnceLock::new();
static NVVM_TOOL_LOAD: OnceLock<Mutex<()>> = OnceLock::new();

/// Driver-independent libNVVM compiler with exact libdevice provenance.
#[derive(Clone)]
pub struct NvvmCompiler {
    tool: Arc<LoadedNvvmTool>,
    libdevice: Arc<[u8]>,
    libdevice_digest: [u8; 32],
}

impl NvvmCompiler {
    /// Discover and pin libNVVM, then read the selected libdevice bytes.
    pub fn discover() -> Result<Self, FinalizerError> {
        let path = find_libdevice().map_err(|libnvvm_sys::LibdeviceNotFound { tried }| {
            FinalizerError::LibdeviceNotFound { tried }
        })?;
        let libdevice = std::fs::read(&path).map_err(|source| FinalizerError::Io {
            path: path.clone(),
            source,
        })?;
        let libdevice_digest = digest_bytes(&libdevice);
        Ok(Self {
            tool: load_nvvm_tool()?,
            libdevice: libdevice.into(),
            libdevice_digest,
        })
    }

    /// Digest of the exact loaded libNVVM file, when its identity is known.
    pub fn libnvvm_digest(&self) -> Option<[u8; 32]> {
        let digest = self.tool.digest?;
        if self.tool.library.loaded_file_if_unchanged().is_some() {
            Some(digest)
        } else {
            report_changed_tool("libNVVM");
            None
        }
    }

    /// Digest of the exact libdevice bytes that will be compiled.
    pub fn libdevice_digest(&self) -> [u8; 32] {
        self.libdevice_digest
    }

    /// Exact route provenance, or `None` when the loaded DSO is unidentifiable.
    pub fn provenance_digest(&self) -> Option<[u8; 32]> {
        self.libnvvm_digest()
            .map(|digest| compiler_provenance_digest(&digest, &self.libdevice_digest))
    }

    /// Compile one NVVM IR module plus libdevice into LTOIR.
    pub fn compile_nvvm_ir_to_ltoir(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        self.with_revalidated_session(options, |session| {
            session.compile_nvvm_ir(module_name, nvvm_ir, NvvmOutputKind::Ltoir, None)
        })
    }

    /// Compile one NVVM IR module plus libdevice into linkable PTX.
    ///
    /// libNVVM performs LLVM-level optimization before the final nvJitLink
    /// invocation performs target-specific optimization on the complete PTX
    /// module.
    pub fn compile_nvvm_ir_to_ptx(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Result<Vec<u8>, FinalizerError> {
        self.with_revalidated_session(options, |session| {
            session.compile_nvvm_ir_to_ptx_deferred(module_name, nvvm_ir, None)
        })
    }

    /// Run a compilation batch between one pair of exact libNVVM identity
    /// checks.
    ///
    /// The higher-ranked session cannot escape this call. Callers may share
    /// the session across scoped threads, but every compile constructs and
    /// destroys its own non-`Send` [`Program`] on the calling thread.
    pub(crate) fn with_revalidated_session<T>(
        &self,
        options: &FinalizationOptions,
        operation: impl for<'session> FnOnce(NvvmCompileSession<'session>) -> Result<T, FinalizerError>,
    ) -> Result<T, FinalizerError> {
        with_revalidated_tool_identity(
            "libNVVM",
            self.tool.digest,
            || current_nvvm_tool_digest(&self.tool),
            || {
                validate_nvvm_frontend(&self.tool.library, options)?;
                operation(NvvmCompileSession {
                    compiler: self,
                    options,
                })
            },
        )
    }

    /// Digest every semantic input to the NVVM IR to LTOIR stage.
    pub fn artifact_digest(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Option<[u8; 32]> {
        let libnvvm = self.libnvvm_digest()?;
        Some(nvvm_ir_artifact_digest_parts(
            module_name,
            nvvm_ir,
            options,
            &self.libdevice_digest,
            &libnvvm,
        ))
    }

    /// Digest every semantic input to the NVVM IR to linkable PTX stage.
    pub fn ptx_artifact_digest(
        &self,
        module_name: &str,
        nvvm_ir: &[u8],
        options: &FinalizationOptions,
    ) -> Option<[u8; 32]> {
        let libnvvm = self.libnvvm_digest()?;
        Some(nvvm_ir_ptx_artifact_digest_parts(
            module_name,
            nvvm_ir,
            options,
            &self.libdevice_digest,
            &libnvvm,
        ))
    }
}

/// A non-escaping libNVVM compilation window.
///
/// `LibNvvm` is safe to share across threads for distinct program handles.
/// This type contains no program handle; each method creates its `Program`
/// locally so the handle never crosses a thread boundary.
#[derive(Clone, Copy)]
pub(crate) struct NvvmCompileSession<'a> {
    compiler: &'a NvvmCompiler,
    options: &'a FinalizationOptions,
}

impl NvvmCompileSession<'_> {
    pub(crate) fn compile_nvvm_ir_to_ptx_capped(
        self,
        module_name: &str,
        nvvm_ir: &[u8],
        maximum_bytes: u64,
    ) -> Result<Vec<u8>, FinalizerError> {
        match self.compile_nvvm_ir_to_ptx_deferred(module_name, nvvm_ir, Some(maximum_bytes)) {
            Err(FinalizerError::Nvvm(libnvvm_sys::NvvmError::CompiledResultTooLarge {
                actual_bytes,
                maximum_bytes,
            })) => Err(FinalizerError::CompiledPtxTooLarge {
                name: module_name.to_string(),
                actual_bytes,
                maximum_bytes,
            }),
            result => result,
        }
    }

    fn compile_nvvm_ir_to_ptx_deferred(
        self,
        module_name: &str,
        nvvm_ir: &[u8],
        maximum_output_bytes: Option<u64>,
    ) -> Result<Vec<u8>, FinalizerError> {
        // Give ordinary libNVVM optimization first refusal. A bounded second
        // pass changes only marked inline-hint roots which remain real PTX
        // calls, then keeps that pass only when conservative frame, access,
        // call, and code-size measures do not regress.
        let baseline = self.compile_nvvm_ir(
            module_name,
            nvvm_ir,
            NvvmOutputKind::Ptx,
            maximum_output_bytes,
        )?;
        let Some((promoted_ir, targets, call_sites)) =
            promote_surviving_inline_candidates(nvvm_ir, &baseline)
        else {
            return Ok(baseline);
        };
        let promoted = match self.compile_nvvm_ir(
            module_name,
            &promoted_ir,
            NvvmOutputKind::Ptx,
            maximum_output_bytes,
        ) {
            Ok(promoted) => promoted,
            Err(error) if deferred_inline_can_use_baseline(&error) => {
                if std::env::var_os("CUDA_OXIDE_INLINE_STATS").is_some() {
                    eprintln!(
                        "[cuda_artifact_finalizer] deferred inline: module={module_name} \
                         targets={} call_sites={call_sites} selected=baseline \
                         promoted_error={error}",
                        targets.len(),
                    );
                }
                return Ok(baseline);
            }
            Err(error) => return Err(error),
        };
        let baseline_score = ptx_local_frame_score(&baseline);
        let promoted_score = ptx_local_frame_score(&promoted);
        let use_promoted = promoted_score.improves_on(baseline_score);
        if std::env::var_os("CUDA_OXIDE_INLINE_STATS").is_some() {
            eprintln!(
                "[cuda_artifact_finalizer] deferred inline: module={module_name} \
                 targets={} call_sites={call_sites} baseline={baseline_score:?} \
                 promoted={promoted_score:?} selected={}",
                targets.len(),
                if use_promoted { "promoted" } else { "baseline" },
            );
        }
        Ok(if use_promoted { promoted } else { baseline })
    }

    fn compile_nvvm_ir(
        self,
        module_name: &str,
        nvvm_ir: &[u8],
        output: NvvmOutputKind,
        maximum_output_bytes: Option<u64>,
    ) -> Result<Vec<u8>, FinalizerError> {
        validate_name(module_name)?;
        if nvvm_ir.is_empty() {
            return Err(FinalizerError::EmptyInput {
                name: module_name.to_string(),
            });
        }
        let mut program = Program::new(&self.compiler.tool.library)?;
        // libdevice must precede user IR so the plan, diagnostics, and
        // provenance all use one deterministic module order.
        program.add_module(&self.compiler.libdevice, "libdevice.10.bc")?;
        program.add_module(nvvm_ir, module_name)?;

        let verify = self.options.nvvm_verify_options();
        let verify_refs = verify.iter().map(String::as_str).collect::<Vec<_>>();
        program.verify(&verify_refs)?;
        let compile = match output {
            NvvmOutputKind::Ltoir => self.options.nvvm_ltoir_options(),
            NvvmOutputKind::Ptx => self.options.nvvm_ptx_options(),
        };
        let compile_refs = compile.iter().map(String::as_str).collect::<Vec<_>>();
        match maximum_output_bytes {
            Some(maximum_bytes) => {
                Ok(program.compile_with_max_output_bytes(&compile_refs, maximum_bytes)?)
            }
            None => Ok(program.compile(&compile_refs)?),
        }
    }
}

fn deferred_inline_can_use_baseline(error: &FinalizerError) -> bool {
    matches!(
        error,
        FinalizerError::Nvvm(libnvvm_sys::NvvmError::CompiledResultTooLarge { .. })
    )
}

#[derive(Clone, Copy)]
enum NvvmOutputKind {
    Ltoir,
    Ptx,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct PtxLocalFrameScore {
    maximum_depot_bytes: usize,
    total_depot_bytes: usize,
    local_access_bytes: usize,
    local_accesses: usize,
    deferred_calls: usize,
    ptx_bytes: usize,
}

impl PtxLocalFrameScore {
    fn improves_on(self, baseline: Self) -> bool {
        self.ptx_bytes <= baseline.ptx_bytes
            && self.total_depot_bytes <= baseline.total_depot_bytes
            && self.local_access_bytes <= baseline.local_access_bytes
            && self.deferred_calls <= baseline.deferred_calls
            && self < baseline
    }
}

fn ptx_call_operands(line: &str) -> Option<&str> {
    let mut line = line.trim_start();
    if line.starts_with('@') {
        let predicate_end = line.find(char::is_whitespace)?;
        line = line[predicate_end..].trim_start();
    }
    let opcode_end = line.find(char::is_whitespace).unwrap_or(line.len());
    let opcode = &line[..opcode_end];
    if opcode != "call" && !opcode.starts_with("call.") {
        return None;
    }
    Some(line[opcode_end..].trim_start())
}

fn after_parenthesized_prefix(value: &str) -> Option<&str> {
    if !value.starts_with('(') {
        return None;
    }
    let mut depth = 0_usize;
    for (offset, character) in value.char_indices() {
        match character {
            '(' => depth = depth.checked_add(1)?,
            ')' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(&value[offset + character.len_utf8()..]);
                }
            }
            _ => {}
        }
    }
    None
}

fn ptx_call_target(statement: &str) -> Option<&str> {
    let mut operands = ptx_call_operands(statement)?;
    if operands.starts_with('(') {
        operands = after_parenthesized_prefix(operands)?.trim_start();
        operands = operands.strip_prefix(',')?.trim_start();
    }
    let target_end = operands
        .find(|character: char| {
            character.is_ascii_whitespace() || matches!(character, ',' | ';' | '(' | ')')
        })
        .unwrap_or(operands.len());
    (target_end != 0).then_some(&operands[..target_end])
}

fn ptx_call_targets(ptx: &str) -> Option<Vec<String>> {
    let mut targets = Vec::new();
    let mut pending = None::<String>;
    for raw_line in ptx.lines() {
        let mut remaining = raw_line.split_once("//").map_or(raw_line, |(code, _)| code);
        loop {
            if let Some(statement) = pending.as_mut() {
                if let Some((piece, tail)) = remaining.split_once(';') {
                    statement.push(' ');
                    statement.push_str(piece);
                    targets.push(ptx_call_target(statement)?.to_string());
                    pending = None;
                    remaining = tail;
                    continue;
                }
                statement.push(' ');
                statement.push_str(remaining);
                break;
            }

            let trimmed = remaining.trim_start();
            if trimmed.is_empty() {
                break;
            }
            if ptx_call_operands(trimmed).is_some() {
                if let Some((statement, tail)) = trimmed.split_once(';') {
                    targets.push(ptx_call_target(statement)?.to_string());
                    remaining = tail;
                    continue;
                }
                pending = Some(trimmed.to_string());
                break;
            }
            if let Some((_, tail)) = trimmed.split_once(';') {
                remaining = tail;
                continue;
            }
            break;
        }
    }
    pending.is_none().then_some(targets)
}

fn ptx_local_access_width(line: &str) -> Option<usize> {
    let start = ["ld.local", "st.local"]
        .into_iter()
        .filter_map(|opcode| line.find(opcode))
        .min()?;
    let opcode = line[start..].split_ascii_whitespace().next()?;
    let mut lanes = 1_usize;
    let mut bits = None;
    for part in opcode.split('.') {
        if let Some(width) = part.strip_prefix('v').and_then(|width| width.parse().ok()) {
            lanes = width;
        } else if matches!(part.as_bytes().first(), Some(b'b' | b'u' | b's' | b'f'))
            && let Ok(width) = part[1..].parse::<usize>()
        {
            bits = Some(width);
        }
    }
    bits.map(|bits| lanes.saturating_mul(bits.div_ceil(8)))
}

fn ptx_local_frame_score(ptx: &[u8]) -> PtxLocalFrameScore {
    let Ok(ptx) = std::str::from_utf8(ptx) else {
        return PtxLocalFrameScore {
            maximum_depot_bytes: usize::MAX,
            total_depot_bytes: usize::MAX,
            local_access_bytes: usize::MAX,
            local_accesses: usize::MAX,
            deferred_calls: usize::MAX,
            ptx_bytes: ptx.len(),
        };
    };
    let depot_sizes = ptx.lines().filter_map(|line| {
        line.contains(".local")
            .then_some(line)
            .filter(|line| line.contains("__local_depot"))
            .and_then(|line| line.rsplit_once('['))
            .and_then(|(_, extent)| extent.split_once(']'))
            .and_then(|(extent, _)| extent.parse::<usize>().ok())
    });
    let (maximum_depot_bytes, total_depot_bytes) = depot_sizes
        .fold((0, 0_usize), |(maximum, total), bytes| {
            (maximum.max(bytes), total.saturating_add(bytes))
        });
    let local_accesses = ptx
        .lines()
        .filter(|line| line.contains("ld.local") || line.contains("st.local"))
        .collect::<Vec<_>>();
    let local_access_bytes = local_accesses
        .iter()
        .try_fold(0_usize, |total, line| {
            ptx_local_access_width(line).map(|bytes| total.saturating_add(bytes))
        })
        .unwrap_or(usize::MAX);
    PtxLocalFrameScore {
        maximum_depot_bytes,
        total_depot_bytes,
        local_access_bytes,
        local_accesses: local_accesses.len(),
        deferred_calls: ptx_call_targets(ptx)
            .map(|targets| targets.len())
            .unwrap_or(usize::MAX),
        ptx_bytes: ptx.len(),
    }
}

fn promote_surviving_inline_candidates(
    nvvm_ir: &[u8],
    ptx: &[u8],
) -> Option<(Vec<u8>, BTreeSet<String>, usize)> {
    let nvvm_ir_text = std::str::from_utf8(nvvm_ir).ok()?;
    let ptx_text = std::str::from_utf8(ptx).ok()?;
    let candidates = nvvm_ir_text
        .lines()
        .filter_map(|line| line.strip_prefix(DEFERRED_INLINE_CANDIDATE_PREFIX))
        .filter(|symbol| !symbol.is_empty() && !symbol.chars().any(char::is_whitespace))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if candidates.is_empty() {
        return None;
    }

    let mut targets = BTreeSet::new();
    let mut call_sites = 0_usize;
    for target in ptx_call_targets(ptx_text)? {
        if candidates.contains(&target) {
            call_sites = call_sites.checked_add(1)?;
            if call_sites > MAX_DEFERRED_INLINE_CALL_SITES {
                return None;
            }
            targets.insert(target);
        }
    }
    if targets.is_empty() || targets.len() > MAX_DEFERRED_INLINE_TARGETS {
        return None;
    }

    let mut promoted = String::with_capacity(nvvm_ir_text.len());
    let mut promoted_targets = BTreeSet::new();
    for line in nvvm_ir_text.split_inclusive('\n') {
        let target = line
            .strip_prefix("define ")
            .and_then(|_| {
                targets
                    .iter()
                    .find(|target| line.contains(&format!("@{target}(")))
            })
            .filter(|_| line.contains(" inlinehint #"));
        if let Some(target) = target {
            promoted.push_str(&line.replacen(" inlinehint #", " alwaysinline #", 1));
            promoted_targets.insert(target.clone());
        } else {
            promoted.push_str(line);
        }
    }
    (promoted_targets == targets).then(|| (promoted.into_bytes(), targets, call_sites))
}

fn current_nvvm_tool_digest(tool: &LoadedNvvmTool) -> Option<[u8; 32]> {
    let file = tool.library.loaded_file_if_unchanged()?;
    digest_file_handle(file).ok()
}

fn validate_nvvm_frontend(
    nvvm: &LibNvvm,
    options: &FinalizationOptions,
) -> Result<(), FinalizerError> {
    let ir_version = nvvm.ir_version()?;
    if (ir_version.ir_major, ir_version.ir_minor) != (2, 0) {
        return Err(FinalizerError::UnsupportedNvvmIrVersion {
            major: ir_version.ir_major,
            minor: ir_version.ir_minor,
        });
    }
    let arch = options.target();
    if let Some(llvm_major) = nvvm.llvm_version(arch)? {
        let mismatch = if arch.uses_legacy_llvm() {
            llvm_major != 7
        } else {
            llvm_major == 7
        };
        if mismatch {
            return Err(FinalizerError::DialectMismatch {
                target: arch.compute(),
                llvm_major,
                expected: if arch.uses_legacy_llvm() {
                    "legacy LLVM 7"
                } else {
                    "modern opaque-pointer"
                },
            });
        }
    }
    Ok(())
}

fn load_nvvm_tool() -> Result<Arc<LoadedNvvmTool>, FinalizerError> {
    if let Some(loaded) = NVVM_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }
    let _guard = NVVM_TOOL_LOAD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(loaded) = NVVM_TOOL.get() {
        return Ok(Arc::clone(loaded));
    }

    let library = LibNvvm::load_for_cache()?;
    let digest = loaded_tool_digest("libNVVM", library.loaded_file_if_unchanged());
    let digest = if digest.is_some() && library.loaded_file_if_unchanged().is_none() {
        report_changed_tool("libNVVM");
        None
    } else {
        digest
    };
    let loaded = Arc::new(LoadedNvvmTool {
        library: Arc::new(library),
        digest,
    });
    let _ = NVVM_TOOL.set(Arc::clone(&loaded));
    Ok(loaded)
}

pub(crate) fn loaded_tool_digest(label: &str, file: Option<&std::fs::File>) -> Option<[u8; 32]> {
    let Some(file) = file else {
        if std::env::var_os("CUDA_OXIDE_VERBOSE").is_some() {
            eprintln!(
                "cuda-oxide: {label} has no exact loaded-file identity; disabling artifact reuse"
            );
        }
        return None;
    };
    match digest_file_handle(file) {
        Ok(digest) => Some(digest),
        Err(error) => {
            if std::env::var_os("CUDA_OXIDE_VERBOSE").is_some() {
                eprintln!(
                    "cuda-oxide: could not fingerprint loaded {label} ({error}); disabling artifact reuse"
                );
            }
            None
        }
    }
}

pub(crate) fn report_changed_tool(label: &str) {
    if std::env::var_os("CUDA_OXIDE_VERBOSE").is_some() {
        eprintln!(
            "cuda-oxide: {label} changed while it was fingerprinted; disabling artifact reuse"
        );
    }
}

pub(crate) fn nvvm_ir_artifact_digest_parts(
    module_name: &str,
    nvvm_ir: &[u8],
    options: &FinalizationOptions,
    libdevice_digest: &[u8; 32],
    libnvvm_digest: &[u8; 32],
) -> [u8; 32] {
    nvvm_ir_artifact_digest_parts_for_output(
        module_name,
        nvvm_ir,
        options,
        libdevice_digest,
        libnvvm_digest,
        NvvmOutputKind::Ltoir,
    )
}

pub(crate) fn nvvm_ir_ptx_artifact_digest_parts(
    module_name: &str,
    nvvm_ir: &[u8],
    options: &FinalizationOptions,
    libdevice_digest: &[u8; 32],
    libnvvm_digest: &[u8; 32],
) -> [u8; 32] {
    nvvm_ir_artifact_digest_parts_for_output(
        module_name,
        nvvm_ir,
        options,
        libdevice_digest,
        libnvvm_digest,
        NvvmOutputKind::Ptx,
    )
}

fn nvvm_ir_artifact_digest_parts_for_output(
    module_name: &str,
    nvvm_ir: &[u8],
    options: &FinalizationOptions,
    libdevice_digest: &[u8; 32],
    libnvvm_digest: &[u8; 32],
    output: NvvmOutputKind,
) -> [u8; 32] {
    let route = match output {
        NvvmOutputKind::Ltoir => b"nvvm-ir-to-ltoir".as_slice(),
        NvvmOutputKind::Ptx => b"nvvm-ir-to-linkable-ptx".as_slice(),
    };
    let mut digest = StableDigest::new()
        .field("recipe", recipe_digest())
        .field("route", route)
        .field("module-name", module_name.as_bytes())
        .field("module", nvvm_ir)
        .field("module-order", b"libdevice.10.bc,user-nvvm-ir")
        .field("libdevice-sha256", libdevice_digest);
    for option in options.nvvm_verify_options() {
        digest = digest.field("nvvm-verify-option", option.as_bytes());
    }
    let compile_options = match output {
        NvvmOutputKind::Ltoir => options.nvvm_ltoir_options(),
        NvvmOutputKind::Ptx => options.nvvm_ptx_options(),
    };
    for option in compile_options {
        digest = digest.field("nvvm-compile-option", option.as_bytes());
    }
    digest.field("libnvvm-sha256", libnvvm_digest).finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn nvvm_digest_covers_module_name_bytes_options_and_libdevice() {
        let options = FinalizationOptions::new("sm_90".parse().unwrap());
        let baseline =
            nvvm_ir_artifact_digest_parts("kernel.ll", b"ir", &options, &[1; 32], &[2; 32]);
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts("other.ll", b"ir", &options, &[1; 32], &[2; 32])
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts("kernel.ll", b"changed", &options, &[1; 32], &[2; 32])
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts(
                "kernel.ll",
                b"ir",
                &options.clone().with_fma_contraction(false),
                &[1; 32],
                &[2; 32]
            )
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts("kernel.ll", b"ir", &options, &[3; 32], &[2; 32])
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts("kernel.ll", b"ir", &options, &[1; 32], &[4; 32])
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts(
                "kernel.ll",
                b"ir",
                &FinalizationOptions::new("sm_120".parse().unwrap()),
                &[1; 32],
                &[2; 32]
            )
        );
        assert_ne!(
            baseline,
            nvvm_ir_artifact_digest_parts(
                "kernel.ll",
                b"ir",
                &options.clone().with_debug_policy(crate::DebugPolicy::Full),
                &[1; 32],
                &[2; 32]
            )
        );
        assert_ne!(
            baseline,
            nvvm_ir_ptx_artifact_digest_parts("kernel.ll", b"ir", &options, &[1; 32], &[2; 32]),
            "LTOIR and PTX compiler routes must never share a cache identity"
        );
    }

    #[test]
    fn promotes_only_marked_candidates_that_survive_as_ptx_calls() {
        let nvvm_ir = br#"; cuda-oxide-device-link-inline-candidate @selected
define internal i32 @selected(i32 %value) inlinehint #0 {
  ret i32 %value
}
; cuda-oxide-device-link-inline-candidate @eliminated
define internal i32 @eliminated(i32 %value) inlinehint #0 {
  ret i32 %value
}
define internal i32 @unmarked(i32 %value) inlinehint #0 {
  ret i32 %value
}
"#;
        let ptx = br#"{
  call.uni (retval0),
  selected,
  (
  param0
  );
}
"#;
        let (promoted, targets, call_sites) =
            promote_surviving_inline_candidates(nvvm_ir, ptx).expect("one target survives");
        let promoted = std::str::from_utf8(&promoted).unwrap();
        assert_eq!(targets, BTreeSet::from(["selected".to_string()]));
        assert_eq!(call_sites, 1);
        assert!(promoted.contains("define internal i32 @selected(i32 %value) alwaysinline #0"));
        assert!(promoted.contains("define internal i32 @eliminated(i32 %value) inlinehint #0"));
        assert!(promoted.contains("define internal i32 @unmarked(i32 %value) inlinehint #0"));
    }

    #[test]
    fn ptx_call_parser_handles_exact_multiline_predicated_and_adjacent_calls() {
        let ptx = r#"
call first, ();
@%p1 call.uni (retval0),
second,
(
param0
);
// call ignored, ();
call.uni third, (); call fourth, ();
"#;
        assert_eq!(
            ptx_call_targets(ptx).unwrap(),
            ["first", "second", "third", "fourth"]
        );
    }

    #[test]
    fn deferred_inline_matches_complete_callee_symbols() {
        let nvvm_ir = br#"; cuda-oxide-device-link-inline-candidate @foo
define internal i32 @foo(i32 %value) inlinehint #0 {
  ret i32 %value
}
; cuda-oxide-device-link-inline-candidate @foobar
define internal i32 @foobar(i32 %value) inlinehint #0 {
  ret i32 %value
}
"#;
        let ptx = b"call.uni foobar, ();\n";
        let (promoted, targets, call_sites) =
            promote_surviving_inline_candidates(nvvm_ir, ptx).expect("foobar survives");
        let promoted = std::str::from_utf8(&promoted).unwrap();
        assert_eq!(targets, BTreeSet::from(["foobar".to_string()]));
        assert_eq!(call_sites, 1);
        assert!(promoted.contains("define internal i32 @foo(i32 %value) inlinehint #0"));
        assert!(promoted.contains("define internal i32 @foobar(i32 %value) alwaysinline #0"));
    }

    #[test]
    fn local_frame_score_accepts_bounded_depot_reduction_before_call_count() {
        let smaller = br#"
.local .align 8 .b8 __local_depot0[144];
"#;
        let larger_frame = br#"
.local .align 8 .b8 __local_depot0[512];
st.local.f64 [%rd1], %fd1;
call.uni (retval0),
selected,
(
);
"#;
        let smaller = ptx_local_frame_score(smaller);
        let larger_frame = ptx_local_frame_score(larger_frame);
        assert!(smaller.improves_on(larger_frame));
    }

    #[test]
    fn local_frame_score_rejects_total_depot_or_code_growth() {
        let baseline = PtxLocalFrameScore {
            maximum_depot_bytes: 512,
            total_depot_bytes: 512,
            local_access_bytes: 160,
            local_accesses: 20,
            deferred_calls: 1,
            ptx_bytes: 1_000,
        };
        assert!(
            !PtxLocalFrameScore {
                maximum_depot_bytes: 511,
                total_depot_bytes: 768,
                local_access_bytes: 80,
                local_accesses: 10,
                deferred_calls: 0,
                ptx_bytes: 900,
            }
            .improves_on(baseline)
        );
        assert!(
            !PtxLocalFrameScore {
                maximum_depot_bytes: 511,
                total_depot_bytes: 511,
                local_access_bytes: 80,
                local_accesses: 10,
                deferred_calls: 0,
                ptx_bytes: 1_001,
            }
            .improves_on(baseline)
        );
        assert!(
            !PtxLocalFrameScore {
                maximum_depot_bytes: 511,
                total_depot_bytes: 511,
                local_access_bytes: 161,
                local_accesses: 41,
                deferred_calls: 0,
                ptx_bytes: 900,
            }
            .improves_on(baseline)
        );
    }

    #[test]
    fn local_access_width_accounts_for_vector_lanes() {
        assert_eq!(
            ptx_local_access_width("st.local.v4.u32 [%rd1], {%r1, %r2, %r3, %r4};"),
            Some(16)
        );
        assert_eq!(
            ptx_local_access_width("@%p1 ld.local.volatile.v2.f64 {%fd1, %fd2}, [%rd1];"),
            Some(16)
        );
    }

    #[test]
    fn deferred_inline_rejects_excessive_surviving_call_sites() {
        let nvvm_ir = br#"; cuda-oxide-device-link-inline-candidate @selected
define internal i32 @selected(i32 %value) inlinehint #0 {
  ret i32 %value
}
"#;
        let ptx = "call.uni (retval0),\nselected,\n(\nparam0\n);\n"
            .repeat(MAX_DEFERRED_INLINE_CALL_SITES + 1);
        assert!(promote_surviving_inline_candidates(nvvm_ir, ptx.as_bytes()).is_none());
    }

    #[test]
    fn only_deterministic_output_caps_can_fall_back_to_baseline() {
        let output_cap = FinalizerError::Nvvm(libnvvm_sys::NvvmError::CompiledResultTooLarge {
            actual_bytes: 65,
            maximum_bytes: 64,
        });
        assert!(deferred_inline_can_use_baseline(&output_cap));

        let cancelled = FinalizerError::Nvvm(libnvvm_sys::NvvmError::Call {
            operation: "nvvmCompileProgram",
            code: 10,
            log: None,
        });
        assert!(!deferred_inline_can_use_baseline(&cancelled));
    }

    #[test]
    #[ignore = "requires discoverable CUDA Toolkit libNVVM and libdevice"]
    fn shared_libnvvm_parallel_programs_match_sequential_ptx_exactly() {
        let compiler = NvvmCompiler::discover().unwrap();
        let options = FinalizationOptions::new("sm_86".parse().unwrap());
        let modules = (0..4)
            .map(|index| {
                (
                    format!("parallel-{index}.ll"),
                    legacy_nvvm_module(&format!("parallel_kernel_{index}")),
                )
            })
            .collect::<Vec<_>>();

        let sequential = compiler
            .with_revalidated_session(&options, |session| {
                modules
                    .iter()
                    .map(|(name, module)| {
                        session.compile_nvvm_ir_to_ptx_capped(
                            name,
                            module,
                            crate::MAX_PARTITION_PTX_BYTES,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .unwrap();
        let parallel = compiler
            .with_revalidated_session(&options, |session| {
                let barrier = Arc::new(Barrier::new(modules.len()));
                std::thread::scope(|scope| {
                    let handles = modules
                        .iter()
                        .map(|(name, module)| {
                            let barrier = Arc::clone(&barrier);
                            scope.spawn(move || {
                                barrier.wait();
                                session.compile_nvvm_ir_to_ptx_capped(
                                    name,
                                    module,
                                    crate::MAX_PARTITION_PTX_BYTES,
                                )
                            })
                        })
                        .collect::<Vec<_>>();
                    handles
                        .into_iter()
                        .map(|handle| handle.join().expect("NVVM worker must not panic"))
                        .collect::<Result<Vec<_>, _>>()
                })
            })
            .unwrap();

        assert_eq!(parallel, sequential);
        for (index, ptx) in parallel.into_iter().enumerate() {
            let kernel = format!("parallel_kernel_{index}");
            assert!(
                ptx.windows(kernel.len())
                    .any(|window| window == kernel.as_bytes()),
                "compiled PTX did not retain expected kernel text for module {index}",
            );
        }
    }

    fn legacy_nvvm_module(kernel: &str) -> Vec<u8> {
        format!(
            r#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @{kernel}() {{
entry:
  ret void
}}

!nvvm.annotations = !{{!0}}
!nvvmir.version = !{{!1}}
!0 = !{{void ()* @{kernel}, !"kernel", i32 1}}
!1 = !{{i32 2, i32 0, i32 3, i32 1}}
"#,
        )
        .into_bytes()
    }
}
