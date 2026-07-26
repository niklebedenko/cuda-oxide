use cuda_artifact_finalizer::{
    CudaArch, FinalizationOptions, Finalizer, PartitionFileInput, PartitionFileInputKind,
    is_valid_cubin,
};
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

const MODULE_A: &[u8] = br#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define linkonce_odr i32 @shared_device(i32 %value) #0 {
entry:
  %result = add i32 %value, 7
  ret i32 %result
}

define void @kernel_a(i32* %output) {
entry:
  %value = call i32 @shared_device(i32 11)
  store i32 %value, i32* %output, align 4
  ret void
}

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (i32)* @shared_device to i8*)], section "llvm.metadata"

attributes #0 = { noinline }

!nvvm.annotations = !{!0}
!nvvmir.version = !{!1}
!0 = !{void (i32*)* @kernel_a, !"kernel", i32 1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
"#;

const MODULE_B: &[u8] = br#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define linkonce_odr i32 @shared_device(i32 %value) #0 {
entry:
  %result = add i32 %value, 7
  ret i32 %result
}

define void @kernel_b(i32* %output) {
entry:
  %value = call i32 @shared_device(i32 13)
  store i32 %value, i32* %output, align 4
  ret void
}

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (i32)* @shared_device to i8*)], section "llvm.metadata"

attributes #0 = { noinline }

!nvvm.annotations = !{!0}
!nvvmir.version = !{!1}
!0 = !{void (i32*)* @kernel_b, !"kernel", i32 1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
"#;

fn main() -> Result<(), Box<dyn Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("cuda-oxide-partition-probe.cubin"));
    let output_dir = output.parent().ok_or("output has no parent")?;
    let input_a = output_dir.join("partition-probe-a.ll");
    let input_b = output_dir.join("partition-probe-b.ll");
    let ptx_bundle = output.with_extension("ptx.bundle");
    std::fs::write(&input_a, MODULE_A)?;
    std::fs::write(&input_b, MODULE_B)?;
    let target: CudaArch = "sm_86".parse()?;
    let options = FinalizationOptions::new(target);
    let finalizer = Finalizer::discover()?;

    let started = Instant::now();
    let materialized = finalizer.materialize_partition_files(
        &[
            PartitionFileInput::new("partition-a.ll", &input_a, PartitionFileInputKind::NvvmIr),
            PartitionFileInput::new("partition-b.ll", &input_b, PartitionFileInputKind::NvvmIr),
        ],
        &ptx_bundle,
        &options,
    )?;
    let elapsed = started.elapsed();
    assert!(is_valid_cubin(&materialized.cubin));
    std::fs::write(&output, &materialized.cubin)?;
    println!(
        "partitions={} ptx_bytes={} cubin_bytes={} elapsed_ms={} link_ms={} peak_rss_kib={:?} bundle={}",
        materialized.partitions.len(),
        materialized
            .partitions
            .iter()
            .map(|partition| partition.ptx_bytes)
            .sum::<usize>(),
        materialized.cubin.len(),
        elapsed.as_millis(),
        materialized.link_elapsed.as_millis(),
        materialized.peak_rss_kib,
        materialized.ptx_bundle_path.display(),
    );
    Ok(())
}
