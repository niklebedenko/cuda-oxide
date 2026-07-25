use cuda_artifact_finalizer::{CudaArch, FinalizationOptions, Finalizer};

const LEGACY_ATOMIC_RMW: &[u8] = br#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i32 @atomic_umax(i32* %ptr, i32 %value) {
entry:
  %old = atomicrmw umax i32* %ptr, i32 %value syncscope("device") monotonic
  ret i32 %old
}

define i64 @atomic_umax_i64(i64* %ptr, i64 %value) {
entry:
  %old = atomicrmw umax i64* %ptr, i64 %value syncscope("device") monotonic
  ret i64 %old
}

define i32 @atomic_or(i32* %ptr, i32 %value) {
entry:
  %old = atomicrmw or i32* %ptr, i32 %value syncscope("device") monotonic
  ret i32 %old
}

define void @kernel(i32* %ptr32, i64* %ptr64) {
entry:
  %max32 = call i32 @atomic_umax(i32* %ptr32, i32 7)
  %max64 = call i64 @atomic_umax_i64(i64* %ptr64, i64 9)
  %or32 = call i32 @atomic_or(i32* %ptr32, i32 16)
  ret void
}

!nvvm.annotations = !{!0}
!nvvmir.version = !{!1}
!0 = !{void (i32*, i64*)* @kernel, !"kernel", i32 1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
"#;

fn main() {
    let finalizer = Finalizer::discover().expect("CUDA compiler tools");
    let target: CudaArch = "sm_86".parse().unwrap();
    let options = FinalizationOptions::new(target);
    let materialized = finalizer
        .materialize_nvvm_ir_with_ptx("legacy_atomic_rmw.ll", LEGACY_ATOMIC_RMW, &options)
        .expect("legacy integer atomicrmw must compile and link");
    let ptx = String::from_utf8_lossy(&materialized.ptx_input);
    assert!(ptx.contains("atom.max.u32"), "{ptx}");
    assert!(ptx.contains("atom.max.u64"), "{ptx}");
    assert!(ptx.contains("atom.or.b32"), "{ptx}");
    println!(
        "{} {}",
        materialized.ptx_input.len(),
        materialized.cubin.len()
    );
}
