use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use ort::{
    session::{builder::SessionBuilder, Session},
    value::{Shape, Tensor},
};
use serde::Deserialize;

pub struct RunOptions<'a> {
    pub input_path: &'a Path,
    pub output_path: &'a Path,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub threads: usize,
}

fn session_builder(
    environment: &ort::environment::Environment,
    threads: usize,
) -> ort::Result<SessionBuilder> {
    // Arena growth and cached memory patterns can reserve more than a tile
    // needs on later runs. Direct allocations avoid that extra reservation.
    Ok(Session::builder(environment)?
        .with_intra_threads(threads)?
        .with_memory_pattern(false)?
        .with_execution_providers([ort::ep::CPU::default().with_arena_allocator(false).build()])?)
}

pub fn full(model: &Path, options: &RunOptions<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let tensor = input_tensor(options)?;
    let environment = ort::init().build()?;
    let mut session = session_builder(&environment, options.threads)?.commit_from_file(model)?;
    let loaded_ms = started.elapsed().as_millis();
    let loaded_memory = crate::memory::process_snapshot()?;
    let input_name = single_input_name(&session)?;
    let inference_started = Instant::now();
    let outputs = session.run(ort::inputs![input_name => &tensor])?;
    let inference_ms = inference_started.elapsed().as_millis();
    let run_memory = crate::memory::process_snapshot()?;
    write_output(&outputs, options.output_path, true)?;
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
    spill_parent: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let tensor = input_tensor(options)?;
    let environment = ort::init().build()?;
    let mut head = session_builder(&environment, options.threads)?.commit_from_file(head_path)?;
    let head_load_ms = started.elapsed().as_millis();
    let head_loaded_memory = crate::memory::process_snapshot()?;
    let input_name = single_input_name(&head)?;
    let head_started = Instant::now();
    let boundary = head.run(ort::inputs![input_name.as_str() => &tensor])?;
    let head_ms = head_started.elapsed().as_millis();
    let head_run_memory = crate::memory::process_snapshot()?;
    let copy_started = Instant::now();
    // ONNX Runtime may alias an output to the head's input or reuse its buffers.
    // Materialize the boundary before releasing the head session, either in
    // owned tensors or in bounded-buffer temporary files.
    let mut owned_boundary = Vec::new();
    let mut spilled_boundary = Vec::new();
    let mut spill = if let Some(parent) = spill_parent {
        Some(SpillFiles::create(parent)?)
    } else {
        None
    };
    let mut spill_bytes = 0_u64;
    for (index, (name, value)) in boundary.iter().enumerate() {
        let name = name.to_string();
        if let Some(files) = spill.as_mut() {
            let path = files.new_path(index);
            let (shape, kind, bytes) = if name == input_name {
                let (shape, values) = tensor.try_extract_tensor::<f32>()?;
                (shape.clone(), BoundaryKind::F32, write_f32(&path, values)?)
            } else if let Ok((shape, values)) = value.try_extract_tensor::<f32>() {
                (shape.clone(), BoundaryKind::F32, write_f32(&path, values)?)
            } else if let Ok((shape, values)) = value.try_extract_tensor::<i64>() {
                (shape.clone(), BoundaryKind::I64, write_i64(&path, values)?)
            } else {
                return Err(format!("unsupported boundary tensor type for {name}").into());
            };
            spill_bytes = spill_bytes
                .checked_add(bytes)
                .ok_or("spill size overflow")?;
            spilled_boundary.push(SpilledBoundary {
                name,
                path,
                shape,
                kind,
            });
        } else {
            let owned = if name == input_name {
                input_tensor(options)?.into_dyn()
            } else if let Ok((shape, values)) = value.try_extract_tensor::<f32>() {
                Tensor::<f32>::from_array((shape.clone(), values.to_vec()))?.into_dyn()
            } else if let Ok((shape, values)) = value.try_extract_tensor::<i64>() {
                Tensor::<i64>::from_array((shape.clone(), values.to_vec()))?.into_dyn()
            } else {
                return Err(format!("unsupported boundary tensor type for {name}").into());
            };
            owned_boundary.push((name, owned));
        }
    }
    let boundary_copy_ms = copy_started.elapsed().as_millis();
    let copied_memory = crate::memory::process_snapshot()?;
    drop(boundary);
    drop(head);
    if spill.is_some() {
        drop(tensor);
    }
    let head_dropped_memory = crate::memory::process_snapshot()?;
    let tail_load_started = Instant::now();
    let mut tail = session_builder(&environment, options.threads)?.commit_from_file(tail_path)?;
    let tail_load_ms = tail_load_started.elapsed().as_millis();
    let tail_loaded_memory = crate::memory::process_snapshot()?;
    let boundary_read_started = Instant::now();
    for item in &spilled_boundary {
        let owned = match item.kind {
            BoundaryKind::F32 => Tensor::<f32>::from_array((
                item.shape.clone(),
                read_f32(&item.path, item.shape.num_elements())?,
            ))?
            .into_dyn(),
            BoundaryKind::I64 => Tensor::<i64>::from_array((
                item.shape.clone(),
                read_i64(&item.path, item.shape.num_elements())?,
            ))?
            .into_dyn(),
        };
        owned_boundary.push((item.name.clone(), owned));
    }
    let boundary_read_ms = boundary_read_started.elapsed().as_millis();
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
    write_output(&outputs, options.output_path, true)?;
    let final_memory = crate::memory::process_snapshot()?;
    println!(
        "mode=split boundary_storage={} spill_write_bytes={spill_bytes} spill_read_bytes={spill_bytes} spill_peak_storage_bytes={spill_bytes} boundary_read_ms={boundary_read_ms} head_load_ms={head_load_ms} head_ms={head_ms} boundary_copy_ms={boundary_copy_ms} tail_load_ms={tail_load_ms} tail_ms={tail_ms} total_ms={} head_loaded_commit_bytes={} head_run_commit_bytes={} copied_commit_bytes={} head_dropped_commit_bytes={} tail_loaded_commit_bytes={} tail_run_commit_bytes={} tail_run_working_set_bytes={} peak_commit_bytes={} peak_working_set_bytes={}",
        if spill.is_some() { "disk" } else { "memory" },
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

enum BoundaryKind {
    F32,
    I64,
}

struct SpilledBoundary {
    name: String,
    path: PathBuf,
    shape: Shape,
    kind: BoundaryKind,
}

struct SpillFiles {
    directory: PathBuf,
    paths: Vec<PathBuf>,
}

impl SpillFiles {
    fn create(parent: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        std::fs::create_dir_all(parent)?;
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        for attempt in 0..100 {
            let directory = parent.join(format!(
                ".raster-compiler-spill-{}-{nonce}-{attempt}",
                std::process::id()
            ));
            match std::fs::create_dir(&directory) {
                Ok(()) => {
                    return Ok(Self {
                        directory,
                        paths: Vec::new(),
                    });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        Err("could not create a unique spill directory".into())
    }

    fn new_path(&mut self, index: usize) -> PathBuf {
        let path = self.directory.join(format!("boundary_{index}.bin"));
        self.paths.push(path.clone());
        path
    }
}

impl Drop for SpillFiles {
    fn drop(&mut self) {
        for path in &self.paths {
            let _ = std::fs::remove_file(path);
        }
        let _ = std::fs::remove_dir(&self.directory);
    }
}

fn write_f32(path: &Path, values: &[f32]) -> Result<u64, Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    for chunk in values.chunks(4096) {
        let mut bytes = Vec::with_capacity(chunk.len() * 4);
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes)?;
    }
    writer.flush()?;
    Ok(u64::try_from(values.len())?
        .checked_mul(4)
        .ok_or("spill size overflow")?)
}

fn write_i64(path: &Path, values: &[i64]) -> Result<u64, Box<dyn std::error::Error>> {
    let mut writer = BufWriter::new(File::create(path)?);
    for chunk in values.chunks(2048) {
        let mut bytes = Vec::with_capacity(chunk.len() * 8);
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        writer.write_all(&bytes)?;
    }
    writer.flush()?;
    Ok(u64::try_from(values.len())?
        .checked_mul(8)
        .ok_or("spill size overflow")?)
}

fn read_f32(path: &Path, count: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    if std::fs::metadata(path)?.len()
        != u64::try_from(count)?
            .checked_mul(4)
            .ok_or("spill size overflow")?
    {
        return Err(format!("spill tensor has wrong size: {}", path.display()).into());
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut values = Vec::with_capacity(count);
    let mut bytes = [0_u8; 16384];
    while values.len() < count {
        let take = (count - values.len()).min(4096);
        reader.read_exact(&mut bytes[..take * 4])?;
        for chunk in bytes[..take * 4].chunks_exact(4) {
            values.push(f32::from_le_bytes(chunk.try_into()?));
        }
    }
    Ok(values)
}

fn read_i64(path: &Path, count: usize) -> Result<Vec<i64>, Box<dyn std::error::Error>> {
    if std::fs::metadata(path)?.len()
        != u64::try_from(count)?
            .checked_mul(8)
            .ok_or("spill size overflow")?
    {
        return Err(format!("spill tensor has wrong size: {}", path.display()).into());
    }
    let mut reader = BufReader::new(File::open(path)?);
    let mut values = Vec::with_capacity(count);
    let mut bytes = [0_u8; 16384];
    while values.len() < count {
        let take = (count - values.len()).min(2048);
        reader.read_exact(&mut bytes[..take * 8])?;
        for chunk in bytes[..take * 8].chunks_exact(8) {
            values.push(i64::from_le_bytes(chunk.try_into()?));
        }
    }
    Ok(values)
}

#[derive(Deserialize)]
struct TileRequest {
    input_f32: PathBuf,
    output_f32: PathBuf,
}

#[derive(Deserialize)]
struct PipeTileRequest {
    input_bytes: usize,
}

pub fn serve_split(
    head_path: &Path,
    tail_path: &Path,
    channels: usize,
    height: usize,
    width: usize,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if channels == 0 || height == 0 || width == 0 || threads == 0 {
        return Err("channels, height, width, and threads must be positive".into());
    }
    let environment = ort::init().build()?;
    let mut head = session_builder(&environment, threads)?.commit_from_file(head_path)?;
    let mut tail = session_builder(&environment, threads)?.commit_from_file(tail_path)?;
    let head_input = single_input_name(&head)?;
    let tail_names = tail
        .inputs()
        .iter()
        .map(|input| input.name().to_owned())
        .collect::<Vec<_>>();
    let stdout = std::io::stdout();
    let mut stdout = std::io::BufWriter::new(stdout.lock());
    let ready_memory = crate::memory::process_snapshot()?;
    writeln!(
        stdout,
        "{}",
        serde_json::json!({
            "ready": true,
            "loaded_commit_bytes": ready_memory.commit_bytes,
            "peak_commit_bytes": ready_memory.peak_commit_bytes
        })
    )?;
    stdout.flush()?;
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let result = (|| -> Result<serde_json::Value, Box<dyn std::error::Error>> {
            let request: TileRequest = serde_json::from_str(&line)?;
            let started = Instant::now();
            let options = RunOptions {
                input_path: &request.input_f32,
                output_path: &request.output_f32,
                channels,
                height,
                width,
                threads,
            };
            let tensor = input_tensor(&options)?;
            let head_started = Instant::now();
            let boundary = head.run(ort::inputs![head_input.as_str() => &tensor])?;
            let head_ms = head_started.elapsed().as_millis();
            let owned_boundary: Vec<(String, ort::value::DynValue)> = boundary
                .iter()
                .map(|(name, value)| {
                    let name = name.to_string();
                    let owned = if name == head_input {
                        let (shape, values) = tensor.try_extract_tensor::<f32>()?;
                        Tensor::<f32>::from_array((shape.clone(), values.to_vec()))?.into_dyn()
                    } else if let Ok((shape, values)) = value.try_extract_tensor::<f32>() {
                        Tensor::<f32>::from_array((shape.clone(), values.to_vec()))?.into_dyn()
                    } else if let Ok((shape, values)) = value.try_extract_tensor::<i64>() {
                        Tensor::<i64>::from_array((shape.clone(), values.to_vec()))?.into_dyn()
                    } else {
                        return Err(format!("unsupported boundary tensor type for {name}").into());
                    };
                    Ok::<_, Box<dyn std::error::Error>>((name, owned))
                })
                .collect::<Result<_, _>>()?;
            drop(boundary);
            let tail_inputs = tail_names
                .iter()
                .map(|name| {
                    let value = owned_boundary
                        .iter()
                        .find(|(candidate, _)| candidate == name)
                        .ok_or_else(|| {
                            format!("head did not produce required tail input {name}")
                        })?;
                    Ok((name.as_str(), &value.1))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let tail_started = Instant::now();
            let outputs = tail.run(tail_inputs)?;
            let tail_ms = tail_started.elapsed().as_millis();
            write_output(&outputs, options.output_path, false)?;
            let memory = crate::memory::process_snapshot()?;
            Ok(serde_json::json!({
                "ok": true,
                "head_ms": head_ms,
                "tail_ms": tail_ms,
                "total_ms": started.elapsed().as_millis(),
                "current_commit_bytes": memory.commit_bytes,
                "peak_commit_bytes": memory.peak_commit_bytes,
                "peak_working_set_bytes": memory.peak_working_set_bytes
            }))
        })();
        match result {
            Ok(response) => {
                writeln!(stdout, "{response}")?;
                stdout.flush()?;
            }
            Err(error) => {
                writeln!(
                    stdout,
                    "{}",
                    serde_json::json!({"ok": false, "error": error.to_string()})
                )?;
                stdout.flush()?;
                return Err(error);
            }
        }
    }
    Ok(())
}

pub fn serve_split_pipe(
    head_path: &Path,
    tail_path: &Path,
    channels: usize,
    height: usize,
    width: usize,
    threads: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if channels == 0 || height == 0 || width == 0 || threads == 0 {
        return Err("channels, height, width, and threads must be positive".into());
    }
    let expected_bytes = channels
        .checked_mul(height)
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_mul(4))
        .ok_or("input dimensions overflow")?;
    let environment = ort::init().build()?;
    let mut head = session_builder(&environment, threads)?.commit_from_file(head_path)?;
    let mut tail = session_builder(&environment, threads)?.commit_from_file(tail_path)?;
    let head_input = single_input_name(&head)?;
    let tail_names = tail
        .inputs()
        .iter()
        .map(|input| input.name().to_owned())
        .collect::<Vec<_>>();
    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();
    let stdout = std::io::stdout();
    let mut stdout = std::io::BufWriter::new(stdout.lock());
    let loaded_memory = crate::memory::process_snapshot()?;
    writeln!(
        stdout,
        "{}",
        serde_json::json!({
            "ready": true,
            "transport": "pipe",
            "input_bytes": expected_bytes,
            "loaded_commit_bytes": loaded_memory.commit_bytes
        })
    )?;
    stdout.flush()?;
    let mut line = String::new();
    loop {
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        let result = (|| -> Result<(Vec<u8>, serde_json::Value), Box<dyn std::error::Error>> {
            let request: PipeTileRequest = serde_json::from_str(&line)?;
            if request.input_bytes != expected_bytes {
                return Err(format!(
                    "input has {} bytes, expected {expected_bytes}",
                    request.input_bytes
                )
                .into());
            }
            let started = Instant::now();
            let mut input_bytes = vec![0u8; expected_bytes];
            stdin.read_exact(&mut input_bytes)?;
            let values = input_bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
                .collect::<Vec<_>>();
            let tensor = Tensor::<f32>::from_array((vec![1, channels, height, width], values))?;
            let head_started = Instant::now();
            let boundary = head.run(ort::inputs![head_input.as_str() => &tensor])?;
            let head_ms = head_started.elapsed().as_millis();
            let owned_boundary: Vec<(String, ort::value::DynValue)> = boundary
                .iter()
                .map(|(name, value)| {
                    let name = name.to_string();
                    let owned = if name == head_input {
                        let (shape, values) = tensor.try_extract_tensor::<f32>()?;
                        Tensor::<f32>::from_array((shape.clone(), values.to_vec()))?.into_dyn()
                    } else if let Ok((shape, values)) = value.try_extract_tensor::<f32>() {
                        Tensor::<f32>::from_array((shape.clone(), values.to_vec()))?.into_dyn()
                    } else if let Ok((shape, values)) = value.try_extract_tensor::<i64>() {
                        Tensor::<i64>::from_array((shape.clone(), values.to_vec()))?.into_dyn()
                    } else {
                        return Err(format!("unsupported boundary tensor type for {name}").into());
                    };
                    Ok::<_, Box<dyn std::error::Error>>((name, owned))
                })
                .collect::<Result<_, _>>()?;
            drop(boundary);
            let tail_inputs = tail_names
                .iter()
                .map(|name| {
                    let value = owned_boundary
                        .iter()
                        .find(|(candidate, _)| candidate == name)
                        .ok_or_else(|| {
                            format!("head did not produce required tail input {name}")
                        })?;
                    Ok((name.as_str(), &value.1))
                })
                .collect::<Result<Vec<_>, String>>()?;
            let tail_started = Instant::now();
            let outputs = tail.run(tail_inputs)?;
            let tail_ms = tail_started.elapsed().as_millis();
            if outputs.len() != 1 {
                return Err(format!("expected one raster output, found {}", outputs.len()).into());
            }
            let output = outputs.values().next().ok_or("missing raster output")?;
            let (_, scores) = output.try_extract_tensor::<f32>()?;
            let mut bytes = Vec::with_capacity(scores.len().saturating_mul(4));
            for score in scores {
                bytes.extend_from_slice(&score.to_le_bytes());
            }
            let memory = crate::memory::process_snapshot()?;
            let response = serde_json::json!({
                "ok": true,
                "output_bytes": bytes.len(),
                "head_ms": head_ms,
                "tail_ms": tail_ms,
                "total_ms": started.elapsed().as_millis(),
                "current_commit_bytes": memory.commit_bytes,
                "peak_commit_bytes": memory.peak_commit_bytes,
                "peak_working_set_bytes": memory.peak_working_set_bytes
            });
            Ok((bytes, response))
        })();
        match result {
            Ok((bytes, response)) => {
                writeln!(stdout, "{response}")?;
                stdout.write_all(&bytes)?;
                stdout.flush()?;
            }
            Err(error) => {
                writeln!(
                    stdout,
                    "{}",
                    serde_json::json!({"ok": false, "error": error.to_string()})
                )?;
                stdout.flush()?;
                return Err(error);
            }
        }
    }
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
    announce: bool,
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
    if announce {
        println!("output shape={shape:?} values={}", values.len());
    }
    Ok(())
}
