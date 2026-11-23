use std::{path::Path, time::Instant};

use ort::{session::Session, value::Tensor};

pub struct RunOptions<'a> {
    pub input_path: &'a Path,
    pub output_path: &'a Path,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub threads: usize,
}

pub fn full(model: &Path, options: &RunOptions<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let tensor = input_tensor(options)?;
    let environment = ort::init().build()?;
    let mut session = Session::builder(&environment)?
        .with_intra_threads(options.threads)?
        .commit_from_file(model)?;
    let loaded_ms = started.elapsed().as_millis();
    let loaded_memory = crate::memory::process_snapshot()?;
    let input_name = single_input_name(&session)?;
    let inference_started = Instant::now();
    let outputs = session.run(ort::inputs![input_name => &tensor])?;
    let inference_ms = inference_started.elapsed().as_millis();
    let run_memory = crate::memory::process_snapshot()?;
    write_output(&outputs, options.output_path)?;
    let final_memory = crate::memory::process_snapshot()?;
    println!(
        "mode=full load_ms={loaded_ms} inference_ms={inference_ms} total_ms={} loaded_commit_bytes={} run_commit_bytes={} peak_commit_bytes={} peak_working_set_bytes={}",
        started.elapsed().as_millis(),
        loaded_memory.commit_bytes,
        run_memory.commit_bytes,
        final_memory.peak_commit_bytes,
        final_memory.peak_working_set_bytes,
    );
    Ok(())
}

pub fn split(
    head_path: &Path,
    tail_path: &Path,
    options: &RunOptions<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let tensor = input_tensor(options)?;
    let environment = ort::init().build()?;
    let mut head = Session::builder(&environment)?
        .with_intra_threads(options.threads)?
        .commit_from_file(head_path)?;
    let head_load_ms = started.elapsed().as_millis();
    let head_loaded_memory = crate::memory::process_snapshot()?;
    let input_name = single_input_name(&head)?;
    let head_started = Instant::now();
    let boundary = head.run(ort::inputs![input_name.as_str() => &tensor])?;
    let head_ms = head_started.elapsed().as_millis();
    let head_run_memory = crate::memory::process_snapshot()?;
    // Materialize each boundary tensor before handing it to a different session.
    // ONNX Runtime may alias an output to the head's input or reuse its buffers.
    let copy_started = Instant::now();
    let owned_boundary: Vec<(String, ort::value::DynValue)> = boundary
        .iter()
        .map(|(name, value)| {
            let name = name.to_string();
            if name == input_name {
                return Ok((name, input_tensor(options)?.into_dyn()));
            }
            let owned = if let Ok((shape, values)) = value.try_extract_tensor::<f32>() {
                Tensor::<f32>::from_array((shape.clone(), values.to_vec()))?.into_dyn()
            } else if let Ok((shape, values)) = value.try_extract_tensor::<i64>() {
                Tensor::<i64>::from_array((shape.clone(), values.to_vec()))?.into_dyn()
            } else {
                return Err(format!("unsupported boundary tensor type for {name}").into());
            };
            Ok::<_, Box<dyn std::error::Error>>((name, owned))
        })
        .collect::<Result<_, _>>()?;
    let boundary_copy_ms = copy_started.elapsed().as_millis();
    let copied_memory = crate::memory::process_snapshot()?;
    drop(boundary);
    drop(head);
    let head_dropped_memory = crate::memory::process_snapshot()?;
    let tail_load_started = Instant::now();
    let mut tail = Session::builder(&environment)?
        .with_intra_threads(options.threads)?
        .commit_from_file(tail_path)?;
    let tail_load_ms = tail_load_started.elapsed().as_millis();
    let tail_loaded_memory = crate::memory::process_snapshot()?;
    let tail_names = tail
        .inputs()
        .iter()
        .map(|input| input.name().to_owned())
        .collect::<Vec<_>>();
    let tail_inputs = tail_names
        .iter()
        .map(|name| {
            let value = owned_boundary
                .iter()
                .find(|(candidate, _)| candidate == name)
                .ok_or_else(|| format!("head did not produce required tail input {name}"))?;
            Ok((name.as_str(), &value.1))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let tail_started = Instant::now();
    let outputs = tail.run(tail_inputs)?;
    let tail_ms = tail_started.elapsed().as_millis();
    let tail_run_memory = crate::memory::process_snapshot()?;
    write_output(&outputs, options.output_path)?;
    let final_memory = crate::memory::process_snapshot()?;
    println!(
        "mode=split head_load_ms={head_load_ms} head_ms={head_ms} boundary_copy_ms={boundary_copy_ms} tail_load_ms={tail_load_ms} tail_ms={tail_ms} total_ms={} head_loaded_commit_bytes={} head_run_commit_bytes={} copied_commit_bytes={} head_dropped_commit_bytes={} tail_loaded_commit_bytes={} tail_run_commit_bytes={} tail_run_working_set_bytes={} peak_commit_bytes={} peak_working_set_bytes={}",
        started.elapsed().as_millis(),
        head_loaded_memory.commit_bytes,
        head_run_memory.commit_bytes,
        copied_memory.commit_bytes,
        head_dropped_memory.commit_bytes,
        tail_loaded_memory.commit_bytes,
        tail_run_memory.commit_bytes,
        tail_run_memory.working_set_bytes,
        final_memory.peak_commit_bytes,
        final_memory.peak_working_set_bytes
    );
    Ok(())
}

fn input_tensor(options: &RunOptions<'_>) -> Result<Tensor<f32>, Box<dyn std::error::Error>> {
    if options.channels == 0 || options.height == 0 || options.width == 0 || options.threads == 0 {
        return Err("channels, height, width, and threads must be positive".into());
    }
    let count = options
        .channels
        .checked_mul(options.height)
        .and_then(|n| n.checked_mul(options.width))
        .ok_or("input dimensions overflow")?;
    let expected_bytes = count.checked_mul(4).ok_or("input size overflows")?;
    let bytes = std::fs::read(options.input_path)?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "input has {} bytes, expected {expected_bytes} for 1x{}x{}x{} FP32",
            bytes.len(),
            options.channels,
            options.height,
            options.width
        )
        .into());
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect::<Vec<_>>();
    Ok(Tensor::<f32>::from_array((
        vec![1, options.channels, options.height, options.width],
        values,
    ))?)
}

fn single_input_name(session: &Session) -> Result<String, String> {
    let inputs = session.inputs();
    if inputs.len() != 1 {
        return Err(format!("expected one image input, found {}", inputs.len()));
    }
    Ok(inputs[0].name().to_owned())
}

fn write_output(
    outputs: &ort::session::SessionOutputs<'_>,
    path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if outputs.len() != 1 {
        return Err(format!("expected one raster output, found {}", outputs.len()).into());
    }
    let output = outputs.values().next().ok_or("missing raster output")?;
    let (shape, values) = output.try_extract_tensor::<f32>()?;
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::fs::write(path, bytes)?;
    println!("output shape={shape:?} values={}", values.len());
    Ok(())
}
