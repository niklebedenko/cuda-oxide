/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Thread-local `TyCtxt` scope for translator code that must use rustc internals.

use rustc_middle::ty::TyCtxt;
use std::cell::Cell;

thread_local! {
    static TCX_PTR: Cell<*const ()> = const { Cell::new(std::ptr::null()) };
}

/// Stash `tcx` in a thread-local for the duration of `f`.
pub fn set_tcx<'tcx, R>(tcx: TyCtxt<'tcx>, f: impl FnOnce() -> R) -> R {
    // SAFETY: `TyCtxt<'tcx>` is a pass-by-value wrapper around a single
    // context pointer. We erase only the lifetime while the scoped guard keeps
    // the original context active.
    let raw: *const () = unsafe { std::mem::transmute(tcx) };
    let prev = TCX_PTR.with(|cell| cell.replace(raw));

    struct Guard(*const ());

    impl Drop for Guard {
        fn drop(&mut self) {
            TCX_PTR.with(|cell| cell.set(self.0));
        }
    }

    let _guard = Guard(prev);
    f()
}

/// Run `f` with the currently stashed `TyCtxt`.
pub fn with_tcx<R>(f: impl FnOnce(TyCtxt<'_>) -> R) -> R {
    let raw = TCX_PTR.with(|cell| cell.get());
    assert!(
        !raw.is_null(),
        "mir_importer::tcx_scope::with_tcx called without an active set_tcx scope"
    );

    // SAFETY: `raw` was written by `set_tcx` on this thread and remains valid
    // until that scoped call returns. The anonymous lifetime in the closure
    // argument prevents callers from storing the context beyond this call.
    let tcx: TyCtxt<'static> = unsafe { std::mem::transmute(raw) };
    f(tcx)
}

pub fn is_active() -> bool {
    !TCX_PTR.with(|cell| cell.get()).is_null()
}
