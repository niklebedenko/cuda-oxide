use cuda_artifact_finalizer::{CudaArch, FinalizationOptions, Finalizer};
use std::collections::HashSet;
use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

fn internalize_kernel_helpers(nvvm_ir: &str) -> Result<(String, usize), Box<dyn Error>> {
    let mut kernels = HashSet::new();
    for line in nvvm_ir.lines().filter(|line| line.contains("!\"kernel\"")) {
        let marker = line
            .find("!\"kernel\"")
            .ok_or("kernel annotation lost its marker")?;
        let at = line[..marker]
            .rfind('@')
            .ok_or("kernel annotation has no function symbol")?;
        let tail = &line[at + 1..marker];
        let end = tail
            .find(',')
            .ok_or("kernel annotation symbol has no delimiter")?;
        kernels.insert(tail[..end].to_string());
    }
    if kernels.is_empty() {
        return Err("NVVM module has no kernel annotations".into());
    }

    let mut output = String::with_capacity(nvvm_ir.len() + 64);
    let mut changed = 0;
    for line in nvvm_ir.split_inclusive('\n') {
        if let Some(rest) = line.strip_prefix("define ") {
            let at = rest.find('@').ok_or("function definition has no symbol")?;
            let symbol = &rest[at + 1..];
            let end = symbol
                .find('(')
                .ok_or("function definition symbol has no argument list")?;
            if !kernels.contains(&symbol[..end]) {
                output.push_str("define internal ");
                output.push_str(rest);
                changed += 1;
                continue;
            }
        }
        if (line.starts_with("@__device_global_") || line.starts_with("@__shared_mem_"))
            && let Some((symbol, definition)) = line.split_once(" = ")
        {
            output.push_str(symbol);
            output.push_str(" = internal ");
            output.push_str(definition);
            changed += 1;
            continue;
        }
        output.push_str(line);
    }
    Ok((output, changed))
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let input = PathBuf::from(args.next().ok_or(
        "usage: materialize_nvvm_bundle <input.ll> <output-prefix> [sm_XX] \
                 [--internalize-kernel-helpers]",
    )?);
    let output = PathBuf::from(args.next().ok_or(
        "usage: materialize_nvvm_bundle <input.ll> <output-prefix> [sm_XX] \
                 [--internalize-kernel-helpers]",
    )?);
    let target: CudaArch = args
        .next()
        .unwrap_or_else(|| "sm_86".into())
        .to_string_lossy()
        .parse()?;
    let internalize = match args.next() {
        None => false,
        Some(arg) if arg == "--internalize-kernel-helpers" => true,
        Some(arg) => {
            return Err(format!(
                "unrecognized materialization option: {}",
                arg.to_string_lossy()
            )
            .into());
        }
    };
    if args.next().is_some() {
        return Err("materialize_nvvm_bundle accepts at most four arguments".into());
    }

    let input_bytes = std::fs::read(&input)?;
    let (internalized, internalized_count) = if internalize {
        let input_text = std::str::from_utf8(&input_bytes)?;
        let (internalized, count) = internalize_kernel_helpers(input_text)?;
        (Some(internalized), count)
    } else {
        (None, 0)
    };
    let nvvm_ir = internalized
        .as_ref()
        .map_or(input_bytes.as_slice(), String::as_bytes);
    let module_name = input
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("input path needs a UTF-8 file name")?;
    let finalizer = Finalizer::discover()?;
    let started = Instant::now();
    let artifact = finalizer.materialize_nvvm_ir_with_ptx(
        module_name,
        nvvm_ir,
        &FinalizationOptions::new(target),
    )?;
    let elapsed = started.elapsed();

    std::fs::write(output.with_extension("ptx"), &artifact.ptx_input)?;
    std::fs::write(output.with_extension("cubin"), &artifact.cubin)?;
    println!(
        "input_bytes={} internalized={} ptx_bytes={} cubin_bytes={} elapsed_ms={}",
        nvvm_ir.len(),
        internalized_count,
        artifact.ptx_input.len(),
        artifact.cubin.len(),
        elapsed.as_millis()
    );
    Ok(())
}
