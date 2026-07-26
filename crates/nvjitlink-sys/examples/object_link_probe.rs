/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Time nvJitLink ingestion and completion for ordered relocatable objects.

use nvjitlink_sys::{InputType, LibNvJitLink, Linker};
use std::error::Error;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let input_kind = match arguments
        .next()
        .and_then(|value| value.into_string().ok())
        .as_deref()
    {
        Some("object") => InputType::Object,
        Some("cubin") => InputType::Cubin,
        _ => {
            return Err(
                "usage: object_link_probe object|cubin OUTPUT.cubin INPUT.o [INPUT.o ...]".into(),
            );
        }
    };
    let output = arguments
        .next()
        .ok_or("object_link_probe requires an output path")?;
    let inputs = arguments.collect::<Vec<_>>();
    if inputs.is_empty() {
        return Err("object_link_probe requires at least one input".into());
    }

    let library = LibNvJitLink::load()?;
    let mut linker = Linker::new(&library, &["-arch=sm_86"])?;
    let total_started = Instant::now();
    let mut total_input_bytes = 0_u64;
    for (index, input) in inputs.iter().enumerate() {
        let path = Path::new(input);
        let bytes = std::fs::read(path)?;
        total_input_bytes += bytes.len() as u64;
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or("object input name is not UTF-8")?;
        let started = Instant::now();
        linker.add(input_kind, &bytes, name)?;
        println!(
            "stage=add_input kind={input_kind:?} index={index} name={name} bytes={} elapsed_ms={}",
            bytes.len(),
            started.elapsed().as_millis()
        );
    }

    let finish_started = Instant::now();
    let cubin = linker.finish()?;
    let finish_elapsed = finish_started.elapsed();
    if !cubin.starts_with(b"\x7fELF") {
        return Err("nvJitLink output is not an ELF cubin".into());
    }
    std::fs::write(&output, &cubin)?;
    println!(
        "stage=complete inputs={} input_bytes={total_input_bytes} cubin_bytes={} \
         finish_ms={} total_ms={}",
        inputs.len(),
        cubin.len(),
        finish_elapsed.as_millis(),
        total_started.elapsed().as_millis()
    );
    Ok(())
}
