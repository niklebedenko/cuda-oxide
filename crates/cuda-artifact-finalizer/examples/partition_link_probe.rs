use cuda_artifact_finalizer::{
    CudaArch, FinalizationOptions, Finalizer, PartitionFileInput, PartitionFileInputKind,
    is_valid_cubin,
};
use cuda_core::{CudaContext, DeviceBuffer};
use std::error::Error;
use std::ffi::OsString;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

const MODULE_A: &[u8] = br#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__device_global__ZN5probe6SHAREDE = linkonce_odr addrspace(1) global i32 11, align 4
@_ZN5probe8CONSTANTE = linkonce_odr addrspace(4) global i32 13, align 4

define linkonce_odr i32 @shared_device(i32 %value) #0 {
entry:
  %result = add i32 %value, 7
  ret i32 %result
}

define void @kernel_a(i32* %output) {
entry:
  %global_value = load i32, i32 addrspace(1)* @__device_global__ZN5probe6SHAREDE, align 4
  %constant_value = load i32, i32 addrspace(4)* @_ZN5probe8CONSTANTE, align 4
  %input = add i32 %global_value, %constant_value
  %value = call i32 @shared_device(i32 %input)
  store i32 %value, i32 addrspace(1)* @__device_global__ZN5probe6SHAREDE, align 4
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

@__device_global__ZN5probe6SHAREDE = linkonce_odr addrspace(1) global i32 11, align 4
@_ZN5probe8CONSTANTE = linkonce_odr addrspace(4) global i32 13, align 4

define linkonce_odr i32 @shared_device(i32 %value) #0 {
entry:
  %result = add i32 %value, 7
  ret i32 %result
}

define void @kernel_b(i32* %output) {
entry:
  %global_value = load i32, i32 addrspace(1)* @__device_global__ZN5probe6SHAREDE, align 4
  %constant_value = load i32, i32 addrspace(4)* @_ZN5probe8CONSTANTE, align 4
  %input = add i32 %global_value, %constant_value
  %value = call i32 @shared_device(i32 %input)
  store i32 %value, i32 addrspace(1)* @__device_global__ZN5probe6SHAREDE, align 4
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

fn parse_count(
    value: Option<OsString>,
    default: usize,
    label: &str,
) -> Result<usize, Box<dyn Error>> {
    let Some(value) = value else {
        return Ok(default);
    };
    let value = value
        .into_string()
        .map_err(|_| format!("{label} is not valid UTF-8"))?;
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {label} {value:?}: {error}"))?;
    Ok(parsed)
}

fn pressure_module(index: usize, operations: usize) -> Vec<u8> {
    let mut module = String::with_capacity(1_024 + operations.saturating_mul(180));
    writeln!(
        module,
        r#"target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16-v32:32-v64:64-v128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@__device_global__ZN5probe6SHAREDE = linkonce_odr addrspace(1) global i32 11, align 4
@_ZN5probe8CONSTANTE = linkonce_odr addrspace(4) global i32 13, align 4

define linkonce_odr i32 @shared_device(i32 %value) #0 {{
entry:
  %result = add i32 %value, 7
  ret i32 %result
}}

define void @kernel_{index}(i32* %output) {{
entry:
  %global_value = load i32, i32 addrspace(1)* @__device_global__ZN5probe6SHAREDE, align 4
  %constant_value = load i32, i32 addrspace(4)* @_ZN5probe8CONSTANTE, align 4
  %input = add i32 %global_value, %constant_value
  %seed = call i32 @shared_device(i32 %input)
  store i32 %seed, i32 addrspace(1)* @__device_global__ZN5probe6SHAREDE, align 4"#
    )
    .expect("writing a String cannot fail");
    let mut previous = "%seed".to_string();
    for operation in 0..operations {
        writeln!(
            module,
            "  store volatile i32 {previous}, i32* %output, align 4\n  \
             %load_{operation} = load volatile i32, i32* %output, align 4\n  \
             %value_{operation} = add i32 %load_{operation}, {}",
            operation % 32_749 + 1,
        )
        .expect("writing a String cannot fail");
        previous = format!("%value_{operation}");
    }
    writeln!(
        module,
        r#"  store i32 {previous}, i32* %output, align 4
  ret void
}}

@llvm.used = appending global [1 x i8*] [i8* bitcast (i32 (i32)* @shared_device to i8*)], section "llvm.metadata"

attributes #0 = {{ noinline }}

!nvvm.annotations = !{{!0}}
!nvvmir.version = !{{!1}}
!0 = !{{void (i32*)* @kernel_{index}, !"kernel", i32 1}}
!1 = !{{i32 2, i32 0, i32 3, i32 1}}"#
    )
    .expect("writing a String cannot fail");
    module.into_bytes()
}

fn validate_static_coalescence_on_gpu(cubin: &[u8]) -> Result<(), Box<dyn Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.new_stream()?;
    let module = context.load_module_from_image(cubin)?;
    let (global_pointer, global_bytes) = module.get_global("__device_global__ZN5probe6SHAREDE")?;
    let (constant_pointer, constant_bytes) = module.get_global("_ZN5probe8CONSTANTE")?;
    if global_pointer == constant_pointer || global_bytes != 4 || constant_bytes != 4 {
        return Err(format!(
            "unexpected linked storage: global=({global_pointer:#x}, {global_bytes}) \
             constant=({constant_pointer:#x}, {constant_bytes})"
        )
        .into());
    }

    let output = DeviceBuffer::from_host(&stream, &[0_i32])?;
    let mut output_pointer = output.cu_deviceptr();
    for (kernel_name, expected) in [("kernel_a", 31_i32), ("kernel_b", 51_i32)] {
        let kernel = module.load_function(kernel_name)?;
        let mut arguments = [std::ptr::addr_of_mut!(output_pointer).cast::<c_void>()];
        // SAFETY: both generated probe kernels take one writable i32 device
        // pointer, use no thread coordinates, and are launched as one thread.
        unsafe {
            cuda_core::launch_kernel_on_stream(
                &kernel,
                (1, 1, 1),
                (1, 1, 1),
                0,
                &stream,
                &mut arguments,
            )?;
        }
        let actual = output.to_host_vec(&stream)?;
        if actual != [expected] {
            return Err(format!(
                "{kernel_name} observed {actual:?}; expected [{expected}] from shared linked state"
            )
            .into());
        }
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let output = arguments
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("cuda-oxide-partition-probe.cubin"));
    let partition_count = parse_count(arguments.next(), 2, "partition count")?;
    let operations_per_partition = parse_count(arguments.next(), 0, "operations per partition")?;
    if partition_count == 0 {
        return Err("partition count must be at least one".into());
    }
    if arguments.next().is_some() {
        return Err(
            "usage: partition_link_probe [OUTPUT] [PARTITIONS] [OPERATIONS_PER_PARTITION]".into(),
        );
    }
    let output_dir = output.parent().ok_or("output has no parent")?;
    std::fs::create_dir_all(output_dir)?;
    let ptx_bundle = output.with_extension("ptx.bundle");
    let mut owned_inputs = Vec::with_capacity(partition_count);
    for index in 0..partition_count {
        let name = format!("partition-{index:04}.ll");
        let path = output_dir.join(format!("partition-probe-{index:04}.ll"));
        let module = match (partition_count, operations_per_partition, index) {
            (2, 0, 0) => MODULE_A.to_vec(),
            (2, 0, 1) => MODULE_B.to_vec(),
            _ => pressure_module(index, operations_per_partition),
        };
        std::fs::write(&path, module)?;
        owned_inputs.push((name, path));
    }
    let inputs = owned_inputs
        .iter()
        .map(|(name, path)| PartitionFileInput::new(name, path, PartitionFileInputKind::NvvmIr))
        .collect::<Vec<_>>();
    let target: CudaArch = "sm_86".parse()?;
    let options = FinalizationOptions::new(target);
    let finalizer = Finalizer::discover()?;

    let started = Instant::now();
    let materialized = finalizer.materialize_partition_files(&inputs, &ptx_bundle, &options)?;
    let elapsed = started.elapsed();
    assert!(is_valid_cubin(&materialized.cubin));
    assert_eq!(materialized.partitions.len(), partition_count);
    let ran_static_launch_probe = partition_count == 2 && operations_per_partition == 0;
    if ran_static_launch_probe {
        validate_static_coalescence_on_gpu(&materialized.cubin)?;
    }
    std::fs::write(&output, &materialized.cubin)?;
    for (_, path) in &owned_inputs {
        std::fs::remove_file(path)?;
    }

    let mut bundle_names = Vec::new();
    oxide_artifacts::ptx_bundle::visit_ptx_bundle(
        std::fs::File::open(&ptx_bundle)?,
        oxide_artifacts::ptx_bundle::PtxBundleLimits::default(),
        |header, _ptx| {
            bundle_names.push(header.name.clone());
            Ok(())
        },
    )?;
    let expected_bundle_names = owned_inputs
        .iter()
        .map(|(name, _)| format!("{}.ptx", name.trim_end_matches(".ll")))
        .collect::<Vec<_>>();
    assert_eq!(bundle_names, expected_bundle_names);

    println!(
        "partitions={} operations_per_partition={} source_bytes={} ptx_bytes={} \
         object_bytes={} cubin_bytes={} elapsed_ms={} nvvm_sum_ms={} nvvm_wall_ms={} \
         nvvm_peak_concurrency={} ptxas_ms={} ptxas_peak_concurrency={} \
         ptxas_peak_aggregate_rss_kib={:?} \
         link_add_ms={} link_complete_ms={} peak_rss_kib={:?} \
         static_launch_probe={} bundle_durable={} bundle={}",
        materialized.partitions.len(),
        operations_per_partition,
        materialized
            .partitions
            .iter()
            .map(|partition| partition.source_bytes)
            .sum::<usize>(),
        materialized
            .partitions
            .iter()
            .map(|partition| partition.ptx_bytes)
            .sum::<usize>(),
        materialized
            .partitions
            .iter()
            .map(|partition| partition.object_bytes)
            .sum::<usize>(),
        materialized.cubin.len(),
        elapsed.as_millis(),
        materialized
            .partitions
            .iter()
            .map(|partition| partition.nvvm_compile_elapsed.as_millis())
            .sum::<u128>(),
        materialized.nvvm_wall_elapsed.as_millis(),
        materialized.nvvm_peak_concurrency,
        materialized
            .partitions
            .iter()
            .map(|partition| partition.ptxas_elapsed.as_millis())
            .sum::<u128>(),
        materialized.ptxas_peak_concurrency,
        materialized.ptxas_peak_aggregate_rss_kib,
        materialized
            .partitions
            .iter()
            .map(|partition| partition.jit_link_add_elapsed.as_millis())
            .sum::<u128>(),
        materialized.link_elapsed.as_millis(),
        materialized.peak_rss_kib,
        ran_static_launch_probe,
        materialized.ptx_bundle_durability_warning.is_none(),
        materialized.ptx_bundle_path.display(),
    );
    Ok(())
}
