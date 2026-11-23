use std::path::PathBuf;

use clap::{Parser, Subcommand};
use onnx_protobuf::{Message, ModelProto};
use ort::session::Session;

mod execute;
mod graph;
mod memory;
mod split;

#[derive(Parser)]
#[command(
    version,
    about = "Inspect ONNX models for memory-bounded raster inference"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Load a model and display its input and output tensors.
    Inspect {
        /// Path to an ONNX model.
        model: PathBuf,
    },
    /// List graph operators and their tensor connections without loading the runtime.
    Graph {
        /// Path to an ONNX model.
        model: PathBuf,
        /// Image height for estimating dynamic tensor sizes.
        #[arg(long, requires = "width")]
        height: Option<usize>,
        /// Image width for estimating dynamic tensor sizes.
        #[arg(long, requires = "height")]
        width: Option<usize>,
    },
    /// Write two ONNX models at a graph cut for runtime validation.
    Split {
        /// Path to an ONNX model.
        model: PathBuf,
        /// Zero-based node index after which to split.
        #[arg(long)]
        cut_after: usize,
        /// Concrete image height for the boundary tensor types.
        #[arg(long)]
        height: usize,
        /// Concrete image width for the boundary tensor types.
        #[arg(long)]
        width: usize,
        /// Directory for head.onnx and tail.onnx.
        #[arg(long)]
        out_dir: PathBuf,
    },
    /// Execute one ONNX model on a raw NCHW FP32 input.
    Run {
        model: PathBuf,
        #[arg(long)]
        input_f32: PathBuf,
        #[arg(long)]
        output_f32: PathBuf,
        #[arg(long)]
        channels: usize,
        #[arg(long)]
        height: usize,
        #[arg(long)]
        width: usize,
        #[arg(long, default_value_t = 4)]
        threads: usize,
    },
    /// Execute a saved head/tail ONNX pair on a raw NCHW FP32 input.
    RunSplit {
        head: PathBuf,
        tail: PathBuf,
        #[arg(long)]
        input_f32: PathBuf,
        #[arg(long)]
        output_f32: PathBuf,
        #[arg(long)]
        channels: usize,
        #[arg(long)]
        height: usize,
        #[arg(long)]
        width: usize,
        #[arg(long, default_value_t = 4)]
        threads: usize,
    },
    /// Keep split-model sessions loaded and execute JSON-line tile requests.
    ServeSplit {
        head: PathBuf,
        tail: PathBuf,
        #[arg(long)]
        channels: usize,
        #[arg(long)]
        height: usize,
        #[arg(long)]
        width: usize,
        #[arg(long, default_value_t = 4)]
        threads: usize,
    },
    /// Keep split-model sessions loaded and exchange tile data over pipes.
    ServeSplitPipe {
        head: PathBuf,
        tail: PathBuf,
        #[arg(long)]
        channels: usize,
        #[arg(long)]
        height: usize,
        #[arg(long)]
        width: usize,
        #[arg(long, default_value_t = 4)]
        threads: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Inspect { model } => inspect(model)?,
        Command::Graph {
            model,
            height,
            width,
        } => graph(model, height.zip(width))?,
        Command::Split {
            model,
            cut_after,
            height,
            width,
            out_dir,
        } => split(model, cut_after, (height, width), out_dir)?,
        Command::Run {
            model,
            input_f32,
            output_f32,
            channels,
            height,
            width,
            threads,
        } => execute::full(
            &model,
            &execute::RunOptions {
                input_path: &input_f32,
                output_path: &output_f32,
                channels,
                height,
                width,
                threads,
            },
        )?,
        Command::RunSplit {
            head,
            tail,
            input_f32,
            output_f32,
            channels,
            height,
            width,
            threads,
        } => execute::split(
            &head,
            &tail,
            &execute::RunOptions {
                input_path: &input_f32,
                output_path: &output_f32,
                channels,
                height,
                width,
                threads,
            },
        )?,
        Command::ServeSplit {
            head,
            tail,
            channels,
            height,
            width,
            threads,
        } => execute::serve_split(&head, &tail, channels, height, width, threads)?,
        Command::ServeSplitPipe {
            head,
            tail,
            channels,
            height,
            width,
            threads,
        } => execute::serve_split_pipe(&head, &tail, channels, height, width, threads)?,
    }

    Ok(())
}

fn split(
    model: PathBuf,
    cut: usize,
    image_size: (usize, usize),
    out_dir: PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    if image_size.0 == 0 || image_size.1 == 0 {
        return Err("image height and width must be positive".into());
    }
    let bytes = std::fs::read(&model)?;
    let model_proto = ModelProto::parse_from_bytes(&bytes)?;
    let graph = model_proto
        .graph
        .as_ref()
        .ok_or("ONNX model has no graph")?;
    for input in &graph.input {
        if let Some(dims) = input
            .type_
            .as_ref()
            .and_then(|ty| ty.tensor_type().shape.as_ref())
            .map(|shape| &shape.dim)
            && dims.len() == 4
        {
            for (axis, supplied) in [(2, image_size.0), (3, image_size.1)] {
                if dims[axis].has_dim_value()
                    && usize::try_from(dims[axis].dim_value()).ok() != Some(supplied)
                {
                    return Err(format!(
                        "input {} has fixed dimension {} at axis {axis}, supplied {supplied}",
                        input.name,
                        dims[axis].dim_value()
                    )
                    .into());
                }
            }
        }
    }
    let analysis = graph::analyze(graph, Some(image_size))?;
    let (head, tail) = split::at(&model_proto, &analysis, cut)?;
    std::fs::create_dir_all(&out_dir)?;
    let head_path = out_dir.join("head.onnx");
    let tail_path = out_dir.join("tail.onnx");
    std::fs::write(&head_path, head.write_to_bytes()?)?;
    std::fs::write(&tail_path, tail.write_to_bytes()?)?;
    println!("head: {}", head_path.display());
    println!("tail: {}", tail_path.display());
    Ok(())
}

fn graph(
    model: PathBuf,
    image_size: Option<(usize, usize)>,
) -> Result<(), Box<dyn std::error::Error>> {
    if image_size.is_some_and(|(height, width)| height == 0 || width == 0) {
        return Err("image height and width must be positive".into());
    }
    let bytes = std::fs::read(&model)?;
    let model_proto = ModelProto::parse_from_bytes(&bytes)?;
    let graph = model_proto
        .graph
        .as_ref()
        .ok_or("ONNX model has no graph")?;

    let analysis = graph::analyze(graph, image_size)?;
    analysis.print(graph);

    Ok(())
}

fn inspect(model: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if !model.is_file() {
        return Err(format!("model file not found: {}", model.display()).into());
    }

    let environment = ort::init().build()?;
    let session = Session::builder(&environment)?.commit_from_file(&model)?;

    println!("model: {}", model.display());
    println!("inputs:");
    for input in session.inputs() {
        println!("  {}: {:?}", input.name(), input.dtype());
    }
    println!("outputs:");
    for output in session.outputs() {
        println!("  {}: {:?}", output.name(), output.dtype());
    }

    Ok(())
}
