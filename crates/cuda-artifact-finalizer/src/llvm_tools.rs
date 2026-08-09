/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Matched-pair resolution of the LLVM `opt` and `llc` binaries.
//!
//! The optimized middle-end (`opt`) and the backend (`llc`) must come from the
//! same LLVM major release: textual IR is not stable across majors. The
//! concrete failure that motivated this module (issue #150): LLVM 22's
//! inliner emits the new sizeless `llvm.lifetime.start(ptr)` intrinsic form
//! (the `i64` size parameter was removed in LLVM 22), which an LLVM 21
//! `llc` rejects with `Intrinsic has incorrect argument type!`. Before this
//! module existed, `optimize_ll` discovered `opt` with its own precedence
//! and never consulted the `llc` that would consume its output, so a user
//! pinning `CUDA_OXIDE_LLC` to an LLVM 21 `llc` (documented as supported)
//! still got the rustc sysroot's LLVM 22 `opt`.
//!
//! [`LlvmToolchain::resolve`] therefore picks `llc` first (it is the tool
//! users pin via `CUDA_OXIDE_LLC` and the one with the hard version floor),
//! reads its major from `llc --version`, then selects an `opt` of the SAME
//! major:
//!
//! 1. An explicit `opt_override` (historically `CUDA_OXIDE_OPT`) is always
//!    respected, but a major mismatch against the chosen `llc` records a
//!    prominent diagnostic naming both binaries.
//! 2. Otherwise the `opt` sitting next to the chosen `llc` (LLVM installs
//!    keep tools side by side) is preferred, provided its major matches.
//! 3. Otherwise the remaining candidates (sysroot llvm-tools `opt`,
//!    `opt-22` / `opt-21` / `opt` on `PATH`) are considered, filtered to
//!    the same major as `llc`.
//! 4. If no same-major `opt` exists, resolution records a diagnostic naming
//!    every rejected candidate. The experimental API treats requested
//!    optimization as strict; the legacy rustc path retains its unoptimized
//!    fallback.

use std::{
    collections::BTreeMap,
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
    process::{Command, Output},
};

/// Exact process context used while discovering and probing LLVM tools.
///
/// Cargo wrappers use this to make discovery observe the same working
/// directory and environment as the child Cargo command. The compiler backend
/// uses [`Self::ambient`] because its process environment is already the
/// effective child environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolchainProcessContext {
    current_dir: PathBuf,
    environment: BTreeMap<OsString, OsString>,
}

#[derive(Debug)]
struct ExecutableCandidate {
    invocation: PathBuf,
    canonical: PathBuf,
}

impl ToolchainProcessContext {
    /// Captures the current process environment and working directory.
    pub fn ambient() -> Self {
        Self {
            current_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            environment: std::env::vars_os().collect(),
        }
    }

    /// Builds a context from an already-computed child environment.
    pub fn new(
        current_dir: PathBuf,
        environment: impl IntoIterator<Item = (OsString, OsString)>,
    ) -> Self {
        Self {
            current_dir,
            environment: environment.into_iter().collect(),
        }
    }

    /// Reads a variable from the captured environment.
    pub fn var_os(&self, name: &str) -> Option<&OsStr> {
        self.environment
            .get(OsStr::new(name))
            .map(OsString::as_os_str)
    }

    fn command(&self, program: &Path) -> Command {
        let mut command = Command::new(program);
        command
            .current_dir(&self.current_dir)
            .env_clear()
            .envs(&self.environment);
        command
    }

    fn executable_candidates(&self, program: &OsStr) -> Vec<ExecutableCandidate> {
        let path = Path::new(program);
        let candidates = if path.is_absolute() || path.components().count() > 1 {
            vec![if path.is_absolute() {
                path.to_path_buf()
            } else {
                self.current_dir.join(path)
            }]
        } else {
            self.var_os("PATH")
                .map(std::env::split_paths)
                .into_iter()
                .flatten()
                .map(|directory| {
                    if directory.is_absolute() {
                        directory.join(path)
                    } else {
                        self.current_dir.join(directory).join(path)
                    }
                })
                .collect()
        };
        candidates
            .into_iter()
            .filter(|candidate| candidate.is_file())
            .filter_map(|invocation| {
                invocation
                    .canonicalize()
                    .ok()
                    .map(|canonical| ExecutableCandidate {
                        invocation,
                        canonical,
                    })
            })
            .collect()
    }

    /// Runs the first spawnable matching program with the captured process
    /// context.
    pub fn output(&self, program: &OsStr, arguments: &[&str]) -> Option<Output> {
        self.executable_candidates(program)
            .into_iter()
            .find_map(|candidate| {
                self.command(&candidate.invocation)
                    .args(arguments)
                    .output()
                    .ok()
            })
    }

    /// Resolves an executable using the captured working directory and PATH.
    pub fn resolve_executable(&self, program: &OsStr) -> Option<PathBuf> {
        self.executable_candidates(program)
            .into_iter()
            .find(|candidate| is_executable(&candidate.invocation))
            .map(|candidate| candidate.canonical)
    }

    fn explicit_path(&self, program: &OsStr) -> PathBuf {
        self.resolve_executable(program).unwrap_or_else(|| {
            let path = Path::new(program);
            if path.is_absolute() {
                path.to_path_buf()
            } else {
                self.current_dir.join(path)
            }
        })
    }
}

#[cfg(unix)]
fn is_executable(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt as _;

    std::fs::metadata(path).is_ok_and(|metadata| metadata.permissions().mode() & 0o111 != 0)
}

#[cfg(not(unix))]
fn is_executable(path: &Path) -> bool {
    path.is_file()
}

/// Inputs which affect selection of the LLVM programs used to emit PTX.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LlvmToolchainOptions {
    /// Skip the optimized LLVM middle-end.
    pub no_opt: bool,
    /// Explicit `llc` binary. Automatic discovery is used when absent.
    pub llc_override: Option<PathBuf>,
    /// Explicit `opt` binary. Matched automatic discovery is used when absent.
    pub opt_override: Option<PathBuf>,
    /// Explicit `llvm-link` binary. Matched automatic discovery is used when absent.
    pub llvm_link_override: Option<PathBuf>,
    /// Preserve an intentionally absent `llvm-link` selection.
    pub llvm_link_disabled: bool,
}

/// A resolved `opt` binary and the LLVM major it reported (if parseable).
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptTool {
    pub path: String,
    pub major: Option<u32>,
}

/// The `opt` / `llc` / `llvm-link` set the pipeline will use, resolved once
/// per compilation so `optimize_ll`, `generate_ptx`, and `link_libdevice`
/// agree on it. Future version-conditional IR emission should key off
/// `llc_major` here rather than re-probing binaries.
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct LlvmToolchain {
    /// The `llc` binary that will produce PTX.
    pub llc_path: String,
    /// `llc`'s LLVM major (from `llc --version`), `None` if unparseable.
    pub llc_major: Option<u32>,
    /// Whether `llc_path` came from `opts.llc_override` (historically
    /// `CUDA_OXIDE_LLC`; affects messages).
    pub llc_from_env: bool,
    /// The matched `opt` for the middle-end; `None` skips LLVM optimization.
    /// (either `opts.no_opt` or no same-major `opt` exists).
    pub opt: Option<OptTool>,
    /// The matched `llvm-link` for libdevice linking; `None` when no
    /// same-major `llvm-link` is available. The backend decision probes the
    /// same discovery ([`libdevice_ir_linking_available`]) and falls back to
    /// the NVVM IR path, so libdevice kernels never reach `llc` with this
    /// unset.
    pub llvm_link: Option<OptTool>,
    /// Tool-selection diagnostics for the caller to report or retain.
    pub diagnostics: Vec<String>,
}

impl LlvmToolchain {
    /// Resolves the `opt` / `llc` pair. Returns `None` when no `llc`
    /// candidate is runnable at all (the caller reports the existing
    /// "No working llc found" error).
    ///
    /// Selection warnings are returned in [`LlvmToolchain::diagnostics`].
    /// Library code never writes them directly to stdout or stderr.
    pub fn resolve(opts: &LlvmToolchainOptions) -> Option<Self> {
        Self::resolve_with_context(opts, &ToolchainProcessContext::ambient())
    }

    /// Resolves against an explicit process context.
    pub fn resolve_with_context(
        opts: &LlvmToolchainOptions,
        context: &ToolchainProcessContext,
    ) -> Option<Self> {
        let (llc_path, llc_major, llc_from_env) = resolve_llc(opts, context)?;

        let (opt, diagnostics) = if opts.no_opt {
            // Explicit user intent: skip the middle-end, no warning needed.
            (None, Vec::new())
        } else {
            let explicit = opts.opt_override.as_ref().map(|path| {
                let path = path.to_string_lossy().into_owned();
                probe_runnable_with_context(&path, context).unwrap_or_else(|| OptTool {
                    path: context
                        .explicit_path(path.as_ref())
                        .to_string_lossy()
                        .into_owned(),
                    major: None,
                })
            });
            let sibling = sibling_tool_candidates(&llc_path, "opt")
                .into_iter()
                .find_map(|c| probe_runnable_with_context(&c, context));
            let mut others: Vec<OptTool> = Vec::new();
            if let Some(p) = sysroot_tool("opt", context)
                && let Some(t) = probe_runnable_with_context(&p, context)
            {
                others.push(t);
            }
            for name in ["opt-22", "opt-21", "opt"] {
                if let Some(t) = probe_runnable_with_context(name, context) {
                    others.push(t);
                }
            }

            let (choice, diagnostics) = choose_opt(&llc_path, llc_major, explicit, sibling, others);
            (choice.into_opt(), diagnostics)
        };

        // Resolve llvm-link for libdevice linking. Same discovery pattern
        // as opt: env var, sibling, sysroot, versioned on PATH. Silently
        // None when absent (the backend decision then avoids the PTX path
        // for libdevice kernels).
        let llvm_link = (!opts.llvm_link_disabled)
            .then(|| {
                resolve_sibling_tool_with_context(
                    "llvm-link",
                    opts.llvm_link_override.as_deref(),
                    &llc_path,
                    llc_major,
                    context,
                )
            })
            .flatten();

        Some(LlvmToolchain {
            llc_path,
            llc_major,
            llc_from_env,
            opt,
            llvm_link,
            diagnostics,
        })
    }
}

/// Resolves the `llc` binary with the documented precedence:
/// `opts.llc_override` (historically `CUDA_OXIDE_LLC`; used exclusively,
/// even if it cannot be probed - the pinned binary's own errors must
/// surface), then the Rust toolchain's llvm-tools `llc`, then `llc-22` /
/// `llc-21` on `PATH` (first runnable wins). Returns `(path, major,
/// from_override)`.
fn resolve_llc(
    opts: &LlvmToolchainOptions,
    context: &ToolchainProcessContext,
) -> Option<(String, Option<u32>, bool)> {
    if let Some(path) = &opts.llc_override {
        let path = path.to_string_lossy().into_owned();
        return Some(match probe_runnable_with_context(&path, context) {
            Some(tool) => (tool.path, tool.major, true),
            None => (
                context
                    .explicit_path(path.as_ref())
                    .to_string_lossy()
                    .into_owned(),
                None,
                true,
            ),
        });
    }

    let mut candidates: Vec<String> = Vec::new();
    if let Some(p) = sysroot_tool("llc", context) {
        candidates.push(p);
    }
    candidates.push("llc-22".to_string());
    candidates.push("llc-21".to_string());

    candidates
        .into_iter()
        .find_map(|c| probe_runnable_with_context(&c, context))
        .map(|t| (t.path, t.major, false))
}

/// The result of [`choose_opt`]: either a usable `opt`, or skip the
/// middle-end entirely.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum OptChoice {
    Use(OptTool),
    Skip,
}

impl OptChoice {
    /// Converts the decision into the `Option<OptTool>` the toolchain
    /// stores (`Skip` = no middle-end, same as `opts.no_opt`).
    fn into_opt(self) -> Option<OptTool> {
        match self {
            OptChoice::Use(t) => Some(t),
            OptChoice::Skip => None,
        }
    }
}

/// Pure decision logic for picking an `opt` matched to the chosen `llc`.
/// Returns the choice plus any warnings the caller should print.
///
/// - `explicit` (`CUDA_OXIDE_OPT`) always wins, with a prominent warning
///   when its major differs from `llc`'s.
/// - `sibling` is the `opt` co-located with `llc`; it is accepted when its
///   major matches, or when `llc`'s major is unknown (same install
///   directory implies the same release).
/// - `others` are accepted only on an exact major match, which requires
///   `llc`'s major to be known.
/// - Otherwise: skip the middle-end, warning with every rejected candidate.
pub(crate) fn choose_opt(
    llc_path: &str,
    llc_major: Option<u32>,
    explicit: Option<OptTool>,
    sibling: Option<OptTool>,
    others: Vec<OptTool>,
) -> (OptChoice, Vec<String>) {
    let mut warnings = Vec::new();

    if let Some(t) = explicit {
        if let (Some(opt_major), Some(llc_major)) = (t.major, llc_major)
            && opt_major != llc_major
        {
            warnings.push(format!(
                "warning: LLVM version mismatch between opt and llc:\n\
                 warning:   CUDA_OXIDE_OPT = {} (LLVM {opt_major})\n\
                 warning:   llc            = {llc_path} (LLVM {llc_major})\n\
                 warning: mixing majors can produce IR the older tool rejects (e.g. LLVM 22's\n\
                 warning: sizeless llvm.lifetime.start/end form fails LLVM 21's llc verifier).\n\
                 warning: proceeding anyway because CUDA_OXIDE_OPT is an explicit override;\n\
                 warning: unset it (or point it at an LLVM {llc_major} opt) to fix the mismatch.",
                t.path
            ));
        }
        return (OptChoice::Use(t), warnings);
    }

    let mut rejected: Vec<OptTool> = Vec::new();

    if let Some(s) = sibling {
        if llc_major.is_none() || s.major == llc_major {
            return (OptChoice::Use(s), warnings);
        }
        rejected.push(s);
    }

    if llc_major.is_some() {
        for t in others {
            if t.major == llc_major {
                return (OptChoice::Use(t), warnings);
            }
            rejected.push(t);
        }
    } else {
        // llc's major is unknown and there is no co-located opt: no
        // candidate can be verified to match, so none is acceptable.
        rejected.extend(others);
    }

    let mut msg = format!(
        "LLVM optimization is unavailable: no opt matching the chosen llc.\n\
         warning:   llc: {}",
        describe_tool(llc_path, llc_major)
    );
    if rejected.is_empty() {
        msg.push_str("\nwarning:   no opt candidates were found at all.");
    } else {
        msg.push_str("\nwarning:   rejected opt candidates:");
        for t in &rejected {
            msg.push_str(&format!(
                "\nwarning:     {}",
                describe_tool(&t.path, t.major)
            ));
        }
    }
    msg.push_str(
        "\nwarning:   install an opt of the same LLVM major as llc, explicitly disable \
         optimization, or set CUDA_OXIDE_OPT.",
    );
    warnings.push(msg);

    (OptChoice::Skip, warnings)
}

/// `"path (LLVM 21)"` or `"path (unknown LLVM version)"` for messages.
pub fn describe_tool(path: &str, major: Option<u32>) -> String {
    match major {
        Some(m) => format!("{path} (LLVM {m})"),
        None => format!("{path} (unknown LLVM version)"),
    }
}

/// Extracts the LLVM major from `--version` output. Handles both distro
/// (`"Ubuntu LLVM version 21.1.8"`) and rustc llvm-tools
/// (`"LLVM version 22.1.2-rust-1.96.0-nightly"`) banners.
pub(crate) fn parse_llvm_major(version_output: &str) -> Option<u32> {
    const NEEDLE: &str = "LLVM version ";
    let idx = version_output.find(NEEDLE)?;
    let rest = &version_output[idx + NEEDLE.len()..];
    let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// Paths co-located with `llc_path` for a given tool name, most specific
/// first: the version-suffixed twin (`/usr/bin/llc-21` ->
/// `/usr/bin/<tool>-21`), then the plain name in the same directory. A bare
/// `llc` name (resolved through `PATH`) yields bare tool names.
pub(crate) fn sibling_tool_candidates(llc_path: &str, tool: &str) -> Vec<String> {
    let p = Path::new(llc_path);
    let dir = p.parent().filter(|d| !d.as_os_str().is_empty());

    let mut names: Vec<String> = Vec::new();
    if let Some(suffix) = p
        .file_name()
        .and_then(|b| b.to_str())
        .and_then(|b| b.strip_prefix("llc"))
        && !suffix.is_empty()
    {
        names.push(format!("{tool}{suffix}"));
    }
    names.push(tool.to_string());

    names
        .into_iter()
        .map(|n| match dir {
            Some(d) => d.join(&n).to_string_lossy().into_owned(),
            None => n,
        })
        .collect()
}

/// Resolve an LLVM tool (e.g., `llvm-link`) co-located with the chosen
/// `llc`, matched to the same LLVM major. Discovery order:
///
/// 1. Explicit env var (always used if set)
/// 2. Co-located binary next to `llc`
/// 3. Sysroot llvm-tools, then versioned names on PATH
///
/// Returns `None` silently when no same-major candidate exists.
pub fn resolve_sibling_tool(
    tool: &str,
    explicit: Option<&Path>,
    llc_path: &str,
    llc_major: Option<u32>,
) -> Option<OptTool> {
    resolve_sibling_tool_with_context(
        tool,
        explicit,
        llc_path,
        llc_major,
        &ToolchainProcessContext::ambient(),
    )
}

fn resolve_sibling_tool_with_context(
    tool: &str,
    explicit: Option<&Path>,
    llc_path: &str,
    llc_major: Option<u32>,
    context: &ToolchainProcessContext,
) -> Option<OptTool> {
    if let Some(path) = explicit {
        return probe_runnable_with_context(&path.to_string_lossy(), context);
    }

    // Co-located sibling next to llc.
    let sibling = sibling_tool_candidates(llc_path, tool)
        .into_iter()
        .find_map(|c| probe_runnable_with_context(&c, context));
    if let Some(ref s) = sibling
        && (llc_major.is_none() || s.major == llc_major)
    {
        return sibling;
    }

    // Sysroot and versioned on PATH.
    let mut candidates: Vec<OptTool> = Vec::new();
    if let Some(p) = sysroot_tool(tool, context)
        && let Some(t) = probe_runnable_with_context(&p, context)
    {
        candidates.push(t);
    }
    for name in [format!("{tool}-22"), format!("{tool}-21"), tool.to_string()] {
        if let Some(t) = probe_runnable_with_context(&name, context) {
            candidates.push(t);
        }
    }
    if let Some(llc_m) = llc_major {
        candidates.into_iter().find(|t| t.major == Some(llc_m))
    } else {
        candidates.into_iter().next()
    }
}

/// Decision-time capability probe for IR-level libdevice linking.
///
/// True only when the `llc` that [`LlvmToolchain::resolve`] would pick is
/// runnable AND a same-major `llvm-link` resolves for it. The PTX-vs-NVVM
/// backend decision is made before the toolchain itself is resolved, so it
/// must probe the same discovery logic: committing to the PTX path on
/// libdevice file existence alone would later emit PTX with unresolved
/// `.extern .func __nv_*` whenever `llvm-link` turns out to be missing.
pub fn libdevice_ir_linking_available(opts: &LlvmToolchainOptions) -> bool {
    let context = ToolchainProcessContext::ambient();
    let Some((llc_path, llc_major, _)) = resolve_llc(opts, &context) else {
        return false;
    };
    !opts.llvm_link_disabled
        && resolve_sibling_tool_with_context(
            "llvm-link",
            opts.llvm_link_override.as_deref(),
            &llc_path,
            llc_major,
            &context,
        )
        .is_some()
}

/// Runs `cmd --version` and, on success, returns the tool with its parsed
/// major. `None` means the binary does not exist or is not runnable.
pub fn probe_runnable(cmd: &str) -> Option<OptTool> {
    probe_runnable_with_context(cmd, &ToolchainProcessContext::ambient())
}

fn probe_runnable_with_context(
    program: &str,
    context: &ToolchainProcessContext,
) -> Option<OptTool> {
    context
        .executable_candidates(OsStr::new(program))
        .into_iter()
        .find_map(|candidate| {
            let output = context
                .command(&candidate.canonical)
                .arg("--version")
                .output()
                .ok()?;
            output.status.success().then(|| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                OptTool {
                    path: candidate.canonical.to_string_lossy().into_owned(),
                    major: parse_llvm_major(&stdout),
                }
            })
        })
}

/// Path of `tool` inside the Rust toolchain's llvm-tools component:
/// `<sysroot>/lib/rustlib/<host>/bin/<tool>`.
fn sysroot_tool(tool: &str, context: &ToolchainProcessContext) -> Option<String> {
    let rustc = context
        .var_os("RUSTC")
        .unwrap_or_else(|| OsStr::new("rustc"));
    let out = context
        .output(rustc, &["--print", "sysroot", "--print", "host-tuple"])
        .filter(|o| o.status.success())?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let mut lines = stdout.lines();
    let sysroot = lines.next()?;
    let host = lines.next()?;
    let path: std::path::PathBuf = [sysroot, "lib", "rustlib", host, "bin", tool]
        .iter()
        .collect();
    path.to_str().map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(path: &str, major: Option<u32>) -> OptTool {
        OptTool {
            path: path.to_string(),
            major,
        }
    }

    #[cfg(unix)]
    #[test]
    fn explicit_tool_selection_executes_and_retains_the_canonical_path() {
        use std::os::unix::fs::{PermissionsExt as _, symlink};

        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("llc-real");
        std::fs::write(&executable, "#!/bin/sh\nprintf 'LLVM version 22.1.0\\n'\n").unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&executable, permissions).unwrap();
        let alias = directory.path().join("llc-alias");
        symlink(&executable, &alias).unwrap();

        let selected = LlvmToolchain::resolve(&LlvmToolchainOptions {
            no_opt: true,
            llc_override: Some(alias),
            ..LlvmToolchainOptions::default()
        })
        .unwrap();

        assert_eq!(
            Path::new(&selected.llc_path),
            executable.canonicalize().unwrap()
        );
        assert_eq!(selected.llc_major, Some(22));
    }

    #[cfg(unix)]
    #[test]
    fn llvm_probe_uses_the_same_canonical_multicall_path_that_is_pinned() {
        use std::os::unix::fs::{PermissionsExt as _, symlink};

        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("llc-real");
        std::fs::write(
            &executable,
            "#!/bin/sh\ncase \"$0\" in\n  *alias*) printf 'LLVM version 21.1.0\\n' ;;\n  *) printf 'LLVM version 22.1.0\\n' ;;\nesac\n",
        )
        .unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&executable, permissions).unwrap();
        let alias = directory.path().join("llc-alias");
        symlink(&executable, &alias).unwrap();

        let selected = LlvmToolchain::resolve(&LlvmToolchainOptions {
            no_opt: true,
            llc_override: Some(alias),
            llvm_link_disabled: true,
            ..LlvmToolchainOptions::default()
        })
        .unwrap();

        assert_eq!(
            Path::new(&selected.llc_path),
            executable.canonicalize().unwrap()
        );
        assert_eq!(selected.llc_major, Some(22));
    }

    #[cfg(unix)]
    #[test]
    fn unprobeable_explicit_tool_still_retains_the_canonical_path() {
        use std::os::unix::fs::{PermissionsExt as _, symlink};

        let directory = tempfile::tempdir().unwrap();
        let executable = directory.path().join("llc-real");
        std::fs::write(&executable, "#!/bin/sh\nexit 1\n").unwrap();
        let mut permissions = std::fs::metadata(&executable).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&executable, permissions).unwrap();
        let alias = directory.path().join("llc-alias");
        symlink(&executable, &alias).unwrap();

        let selected = LlvmToolchain::resolve(&LlvmToolchainOptions {
            no_opt: true,
            llc_override: Some(alias),
            llvm_link_disabled: true,
            ..LlvmToolchainOptions::default()
        })
        .unwrap();

        assert_eq!(
            Path::new(&selected.llc_path),
            executable.canonicalize().unwrap()
        );
        assert_eq!(selected.llc_major, None);
    }

    #[cfg(unix)]
    #[test]
    fn path_resolution_skips_non_executable_shadow_and_uses_runnable_tool() {
        use std::os::unix::fs::PermissionsExt as _;

        let root = tempfile::tempdir().unwrap();
        let shadow_directory = root.path().join("shadow");
        let runnable_directory = root.path().join("runnable");
        std::fs::create_dir_all(&shadow_directory).unwrap();
        std::fs::create_dir_all(&runnable_directory).unwrap();

        let shadow = shadow_directory.join("llc");
        std::fs::write(
            &shadow,
            "#!/bin/sh\nprintf 'shadow LLVM version 21.0.0\\n'\n",
        )
        .unwrap();

        let runnable = runnable_directory.join("llc");
        std::fs::write(&runnable, "#!/bin/sh\nprintf 'LLVM version 22.1.0\\n'\n").unwrap();
        let mut permissions = std::fs::metadata(&runnable).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&runnable, permissions).unwrap();

        let search_path = std::env::join_paths([shadow_directory, runnable_directory]).unwrap();
        let mut environment = std::env::vars_os().collect::<BTreeMap<_, _>>();
        environment.insert(OsString::from("PATH"), search_path);
        let context = ToolchainProcessContext::new(root.path().to_path_buf(), environment);
        assert_eq!(
            probe_runnable_with_context("llc", &context).map(|tool| PathBuf::from(tool.path)),
            Some(runnable.canonicalize().unwrap()),
        );
    }

    #[test]
    fn parse_llvm_major_handles_distro_and_rustc_banners() {
        assert_eq!(
            parse_llvm_major("Ubuntu LLVM version 21.1.8\n  Optimized build."),
            Some(21)
        );
        assert_eq!(
            parse_llvm_major(
                "LLVM (http://llvm.org/):\n  LLVM version 22.1.2-rust-1.96.0-nightly\n"
            ),
            Some(22)
        );
        assert_eq!(parse_llvm_major("LLVM version 22.1.7"), Some(22));
        assert_eq!(parse_llvm_major("no version banner here"), None);
        assert_eq!(parse_llvm_major("LLVM version x.y.z"), None);
        assert_eq!(parse_llvm_major(""), None);
    }

    #[test]
    fn sibling_candidates_mirror_the_llc_name() {
        assert_eq!(
            sibling_tool_candidates("/usr/bin/llc-21", "opt"),
            ["/usr/bin/opt-21", "/usr/bin/opt"]
        );
        assert_eq!(
            sibling_tool_candidates("/usr/lib/llvm/21/bin/llc", "opt"),
            ["/usr/lib/llvm/21/bin/opt"]
        );
        // Bare PATH names stay bare so they resolve through PATH.
        assert_eq!(sibling_tool_candidates("llc-22", "opt"), ["opt-22", "opt"]);
        assert_eq!(sibling_tool_candidates("llc", "opt"), ["opt"]);
    }

    #[test]
    fn sibling_tool_candidates_generates_versioned_and_plain() {
        assert_eq!(
            sibling_tool_candidates("/usr/bin/llc-21", "llvm-link"),
            ["/usr/bin/llvm-link-21", "/usr/bin/llvm-link"]
        );
        assert_eq!(
            sibling_tool_candidates("/usr/lib/llvm/21/bin/llc", "llvm-link"),
            ["/usr/lib/llvm/21/bin/llvm-link"]
        );
        assert_eq!(
            sibling_tool_candidates("llc-22", "llvm-link"),
            ["llvm-link-22", "llvm-link"]
        );
        assert_eq!(sibling_tool_candidates("llc", "llvm-link"), ["llvm-link"]);
    }

    #[test]
    fn explicit_opt_is_respected_on_match_without_warning() {
        let (choice, warnings) = choose_opt(
            "/usr/bin/llc-21",
            Some(21),
            Some(tool("/usr/bin/opt-21", Some(21))),
            None,
            vec![],
        );
        assert_eq!(choice, OptChoice::Use(tool("/usr/bin/opt-21", Some(21))));
        assert!(warnings.is_empty());
    }

    #[test]
    fn explicit_opt_is_respected_on_mismatch_with_warning() {
        let (choice, warnings) = choose_opt(
            "/usr/bin/llc-21",
            Some(21),
            Some(tool("/usr/bin/opt-22", Some(22))),
            Some(tool("/usr/bin/opt-21", Some(21))),
            vec![],
        );
        // The user's explicit pin wins even over a perfectly matched sibling.
        assert_eq!(choice, OptChoice::Use(tool("/usr/bin/opt-22", Some(22))));
        assert_eq!(warnings.len(), 1);
        let w = &warnings[0];
        assert!(
            w.contains("CUDA_OXIDE_OPT = /usr/bin/opt-22 (LLVM 22)"),
            "{w}"
        );
        assert!(w.contains("/usr/bin/llc-21 (LLVM 21)"), "{w}");
        assert!(w.contains("mismatch"), "{w}");
    }

    #[test]
    fn sibling_opt_wins_when_major_matches() {
        let (choice, warnings) = choose_opt(
            "/usr/bin/llc-21",
            Some(21),
            None,
            Some(tool("/usr/bin/opt-21", Some(21))),
            vec![tool("/sysroot/bin/opt", Some(22))],
        );
        assert_eq!(choice, OptChoice::Use(tool("/usr/bin/opt-21", Some(21))));
        assert!(warnings.is_empty());
    }

    #[test]
    fn mismatched_sibling_is_rejected_and_matching_other_wins() {
        let (choice, warnings) = choose_opt(
            "/usr/bin/llc-21",
            Some(21),
            None,
            Some(tool("/usr/bin/opt", Some(22))),
            vec![tool("/sysroot/bin/opt", Some(22)), tool("opt-21", Some(21))],
        );
        assert_eq!(choice, OptChoice::Use(tool("opt-21", Some(21))));
        assert!(warnings.is_empty());
    }

    #[test]
    fn unverifiable_other_candidates_are_rejected() {
        // An opt whose --version output could not be parsed must not be
        // assumed to match.
        let (choice, warnings) =
            choose_opt("llc-21", Some(21), None, None, vec![tool("opt", None)]);
        assert_eq!(choice, OptChoice::Skip);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("opt (unknown LLVM version)"));
    }

    #[test]
    fn no_matching_opt_reports_every_rejected_candidate() {
        let (choice, warnings) = choose_opt(
            "/usr/bin/llc-21",
            Some(21),
            None,
            Some(tool("/usr/bin/opt", Some(22))),
            vec![tool("/sysroot/bin/opt", Some(22)), tool("opt-22", Some(22))],
        );
        assert_eq!(choice, OptChoice::Skip);
        assert_eq!(warnings.len(), 1);
        let w = &warnings[0];
        assert!(w.contains("LLVM optimization is unavailable"), "{w}");
        assert!(w.contains("/usr/bin/llc-21 (LLVM 21)"), "{w}");
        assert!(w.contains("/usr/bin/opt (LLVM 22)"), "{w}");
        assert!(w.contains("/sysroot/bin/opt (LLVM 22)"), "{w}");
        assert!(w.contains("opt-22 (LLVM 22)"), "{w}");
        assert!(w.contains("explicitly disable optimization"), "{w}");
    }

    #[test]
    fn no_candidates_at_all_skips_with_warning() {
        let (choice, warnings) = choose_opt("/usr/bin/llc-21", Some(21), None, None, vec![]);
        assert_eq!(choice, OptChoice::Skip);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("no opt candidates were found at all"));
    }

    #[test]
    fn unknown_llc_major_trusts_only_the_colocated_opt() {
        // Same install directory implies the same release, so the sibling
        // is accepted even when llc's banner could not be parsed...
        let (choice, warnings) = choose_opt(
            "/custom/bin/llc",
            None,
            None,
            Some(tool("/custom/bin/opt", None)),
            vec![tool("opt-22", Some(22))],
        );
        assert_eq!(choice, OptChoice::Use(tool("/custom/bin/opt", None)));
        assert!(warnings.is_empty());

        // ...but unrelated candidates cannot be verified against an unknown
        // llc major and are all rejected.
        let (choice, warnings) = choose_opt(
            "/custom/bin/llc",
            None,
            None,
            None,
            vec![tool("opt-22", Some(22)), tool("opt-21", Some(21))],
        );
        assert_eq!(choice, OptChoice::Skip);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("/custom/bin/llc (unknown LLVM version)"));
    }
}
