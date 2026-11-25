use std::collections::{BTreeMap, BTreeSet};

use onnx_protobuf::{GraphProto, NodeProto, ValueInfoProto};

type Shape = Vec<usize>;

pub struct Analysis {
    producers: BTreeMap<String, String>,
    consumers: BTreeMap<String, Vec<usize>>,
    shapes: BTreeMap<String, Option<Shape>>,
    element_sizes: BTreeMap<String, Option<usize>>,
    element_types: BTreeMap<String, Option<i32>>,
    cuts: Vec<Vec<String>>,
}

pub fn analyze(graph: &GraphProto, image_size: Option<(usize, usize)>) -> Result<Analysis, String> {
    let mut producers = BTreeMap::new();
    let mut consumers: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut shapes = BTreeMap::new();
    let mut element_sizes = BTreeMap::new();
    let mut element_types = BTreeMap::new();

    for input in &graph.input {
        add_producer(&mut producers, &input.name, "graph input")?;
        shapes.insert(input.name.clone(), declared_shape(input, image_size, true));
        element_sizes.insert(
            input.name.clone(),
            onnx_element_size(input.type_.as_ref().map(|ty| ty.tensor_type().elem_type)),
        );
        element_types.insert(
            input.name.clone(),
            input.type_.as_ref().map(|ty| ty.tensor_type().elem_type),
        );
    }
    for initializer in &graph.initializer {
        if producers
            .get(&initializer.name)
            .is_some_and(|source| source == "initializer")
        {
            return Err(format!("duplicate initializer {}", initializer.name));
        }
        if producers.contains_key(&initializer.name) {
            // ONNX permits an initializer to also be declared as a graph input.
            producers.insert(initializer.name.clone(), "initializer".into());
        } else {
            add_producer(&mut producers, &initializer.name, "initializer")?;
        }
        shapes.insert(initializer.name.clone(), positive_dims(&initializer.dims));
        element_sizes.insert(
            initializer.name.clone(),
            onnx_element_size(Some(initializer.data_type)),
        );
        element_types.insert(initializer.name.clone(), Some(initializer.data_type));
    }

    for (index, node) in graph.node.iter().enumerate() {
        for input in node.input.iter().filter(|name| !name.is_empty()) {
            if !producers.contains_key(input) {
                return Err(format!(
                    "node {index} ({}) has missing input {input}",
                    node.op_type
                ));
            }
            consumers.entry(input.clone()).or_default().push(index);
        }

        let inferred = infer_shape(node, &shapes, graph);
        let element_size = if node.op_type == "Constant" {
            attr(node, "value")
                .and_then(|val| val.t.as_ref())
                .and_then(|tensor| onnx_element_size(Some(tensor.data_type)))
        } else if node.op_type == "Shape" {
            Some(8)
        } else if node.op_type == "ConstantOfShape" {
            let data_type = attr(node, "value")
                .and_then(|val| val.t.as_ref())
                .map(|t| t.data_type)
                .unwrap_or(1);
            onnx_element_size(Some(data_type))
        } else {
            node.input
                .first()
                .and_then(|name| element_sizes.get(name).copied().flatten())
        };
        let element_type = if node.op_type == "Constant" {
            attr(node, "value")
                .and_then(|val| val.t.as_ref())
                .map(|tensor| tensor.data_type)
        } else if node.op_type == "Shape" {
            Some(7)
        } else if node.op_type == "ConstantOfShape" {
            let data_type = attr(node, "value")
                .and_then(|val| val.t.as_ref())
                .map(|t| t.data_type)
                .unwrap_or(1);
            Some(data_type)
        } else {
            node.input
                .first()
                .and_then(|name| element_types.get(name).copied().flatten())
        };
        for (output_index, output) in node
            .output
            .iter()
            .enumerate()
            .filter(|(_, name)| !name.is_empty())
        {
            add_producer(
                &mut producers,
                output,
                &format!("node {index} ({})", node.op_type),
            )?;
            // Optional outputs can have a different shape and element type.
            let primary = output_index == 0;
            shapes.insert(
                output.clone(),
                if primary { inferred.clone() } else { None },
            );
            element_sizes.insert(output.clone(), if primary { element_size } else { None });
            element_types.insert(output.clone(), if primary { element_type } else { None });
        }
    }

    for output in &graph.output {
        if !producers.contains_key(&output.name) {
            return Err(format!("graph output {} has no producer", output.name));
        }
        if let (Some(inferred), Some(declared)) = (
            shapes.get(&output.name).and_then(Option::as_ref),
            declared_shape(output, image_size, false),
        ) && *inferred != declared
        {
            return Err(format!(
                "graph output {} has inferred shape {inferred:?}, declared shape {declared:?}",
                output.name
            ));
        }
    }

    let mut cuts = Vec::new();
    for cut in 0..graph.node.len().saturating_sub(1) {
        let before: BTreeSet<&str> = graph.node[..=cut]
            .iter()
            .flat_map(|node| node.output.iter().map(String::as_str))
            .chain(graph.input.iter().map(|input| input.name.as_str()))
            .filter(|name| !name.is_empty())
            .collect();
        let after: BTreeSet<&str> = graph.node[cut + 1..]
            .iter()
            .flat_map(|node| node.input.iter().map(String::as_str))
            .chain(graph.output.iter().map(|output| output.name.as_str()))
            .filter(|name| !name.is_empty())
            .collect();
        cuts.push(
            before
                .intersection(&after)
                .filter(|name| {
                    producers
                        .get(**name)
                        .is_some_and(|kind| kind != "initializer")
                })
                .map(|name| (*name).to_owned())
                .collect(),
        );
    }

    Ok(Analysis {
        producers,
        consumers,
        shapes,
        element_sizes,
        element_types,
        cuts,
    })
}

impl Analysis {
    pub fn boundary(&self, cut: usize) -> Option<&[String]> {
        self.cuts.get(cut).map(Vec::as_slice)
    }

    pub fn value_info(&self, name: &str) -> Result<ValueInfoProto, String> {
        let shape = self
            .shapes
            .get(name)
            .and_then(Option::as_ref)
            .ok_or_else(|| format!("cannot split at tensor {name}: unknown shape"))?;
        let element_type = self
            .element_types
            .get(name)
            .copied()
            .flatten()
            .ok_or_else(|| format!("cannot split at tensor {name}: unknown element type"))?;
        let mut tensor_shape = onnx_protobuf::TensorShapeProto::new();
        for size in shape {
            let mut dimension = onnx_protobuf::tensor_shape_proto::Dimension::new();
            dimension.set_dim_value(
                i64::try_from(*size).map_err(|_| format!("tensor {name}: dimension too large"))?,
            );
            tensor_shape.dim.push(dimension);
        }
        let tensor_type = onnx_protobuf::type_proto::Tensor {
            elem_type: element_type,
            shape: protobuf::MessageField::some(tensor_shape),
            ..Default::default()
        };
        let type_ = onnx_protobuf::TypeProto {
            value: Some(onnx_protobuf::type_proto::Value::TensorType(tensor_type)),
            ..Default::default()
        };
        Ok(ValueInfoProto {
            name: name.into(),
            type_: protobuf::MessageField::some(type_),
            ..Default::default()
        })
    }

    fn tensor_bytes(&self, name: &str) -> Option<usize> {
        let dims = self.shapes.get(name)?.as_ref()?;
        let element_size = self.element_sizes.get(name).copied().flatten()?;
        dims.iter()
            .try_fold(element_size, |bytes, dim| bytes.checked_mul(*dim))
    }

    pub fn print(&self, graph: &GraphProto) {
        println!("graph: {}", graph.name);
        println!("nodes: {}", graph.node.len());
        println!("initializers: {}", graph.initializer.len());
        for (index, node) in graph.node.iter().enumerate() {
            println!(
                "{:>3}  {:<20}  {} -> {}",
                index,
                node.op_type,
                node.input.join(", "),
                node.output.join(", ")
            );
        }
        println!("tensors:");
        for (name, producer) in &self.producers {
            let shape = self.shapes.get(name).and_then(Option::as_ref);
            let size = self.tensor_bytes(name);
            let consumers = self.consumers.get(name).cloned().unwrap_or_default();
            println!(
                "  {name}: source={producer}, consumers={consumers:?}, shape={}, bytes={}",
                shape.map_or_else(|| "unknown".into(), |dims| format!("{dims:?}")),
                size.map_or_else(|| "unknown".into(), |bytes| bytes.to_string())
            );
        }
        println!("live tensors at cuts:");
        for (index, tensors) in self.cuts.iter().enumerate() {
            let boundary_bytes = tensors.iter().try_fold(0usize, |total, name| {
                total.checked_add(self.tensor_bytes(name)?)
            });
            println!(
                "  after node {index}: {} (boundary bytes={})",
                tensors.join(", "),
                boundary_bytes.map_or_else(|| "unknown".into(), |bytes| bytes.to_string())
            );
        }
    }
}

fn onnx_element_size(data_type: Option<i32>) -> Option<usize> {
    match data_type? {
        1 | 6 => Some(4),   // float32 or int32
        7 | 11 => Some(8),  // int64 or float64
        9 => Some(1),       // bool
        10 | 16 => Some(2), // float16 or bfloat16
        _ => None,
    }
}

fn add_producer(
    producers: &mut BTreeMap<String, String>,
    name: &str,
    source: &str,
) -> Result<(), String> {
    if name.is_empty() {
        return Err(format!("{source} has an empty tensor name"));
    }
    if let Some(previous) = producers.insert(name.into(), source.into()) {
        return Err(format!(
            "tensor {name} has duplicate producers: {previous} and {source}"
        ));
    }
    Ok(())
}

fn positive_dims(dims: &[i64]) -> Option<Shape> {
    dims.iter().map(|dim| usize::try_from(*dim).ok()).collect()
}

fn declared_shape(
    input: &ValueInfoProto,
    image_size: Option<(usize, usize)>,
    graph_input: bool,
) -> Option<Shape> {
    let dims = &input.type_.as_ref()?.tensor_type().shape.as_ref()?.dim;
    dims.iter()
        .enumerate()
        .map(|(index, dim)| {
            if dim.has_dim_value() {
                usize::try_from(dim.dim_value()).ok()
            } else {
                match (dim.dim_param(), image_size) {
                    ("height", Some((height, _))) => Some(height),
                    ("width", Some((_, width))) => Some(width),
                    (_, Some((height, width))) if graph_input && dims.len() == 4 => {
                        match index {
                            0 => Some(1),
                            2 => Some(height),
                            3 => Some(width),
                            _ => None,
                        }
                    }
                    _ => None,
                }
            }
        })
        .collect()
}

fn attr<'a>(node: &'a NodeProto, name: &str) -> Option<&'a onnx_protobuf::AttributeProto> {
    node.attribute
        .iter()
        .find(|attribute| attribute.name == name)
}

fn int64_values(tensor: &onnx_protobuf::TensorProto) -> Option<Vec<i64>> {
    if tensor.data_type != 7 {
        return None;
    }
    let values = if !tensor.int64_data.is_empty() {
        tensor.int64_data.clone()
    } else {
        if !tensor.raw_data.len().is_multiple_of(8) {
            return None;
        }
        tensor
            .raw_data
            .chunks_exact(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().expect("eight-byte chunk")))
            .collect()
    };
    let count = tensor.dims.iter().try_fold(1usize, |total, dim| {
        total.checked_mul(usize::try_from(*dim).ok()?)
    })?;
    (values.len() == count).then_some(values)
}

fn spatial_shape(
    node: &NodeProto,
    input: &[usize],
    channels: usize,
    transpose: bool,
) -> Option<Shape> {
    if attr(node, "auto_pad").is_some_and(|value| value.s != b"NOTSET")
        || attr(node, "dilations").is_some_and(|value| value.ints != [1, 1])
        || attr(node, "group").is_some_and(|value| value.i != 1)
        || attr(node, "output_padding").is_some_and(|value| value.ints != [0, 0])
        || attr(node, "output_shape").is_some()
        || attr(node, "ceil_mode").is_some_and(|value| value.i != 0)
    {
        return None;
    }
    let kernel = &attr(node, "kernel_shape")?.ints;
    let strides = attr(node, "strides").map_or(vec![1, 1], |value| value.ints.clone());
    let pads = attr(node, "pads").map_or(vec![0, 0, 0, 0], |value| value.ints.clone());
    if input.len() != 4 || kernel.len() != 2 || strides.len() != 2 || pads.len() != 4 {
        return None;
    }
    let axis = |size: usize, k: i64, stride: i64, before: i64, after: i64| {
        if k <= 0 || stride <= 0 || before < 0 || after < 0 {
            return None;
        }
        let size = i128::try_from(size).ok()?;
        let (k, stride, before, after) = (
            i128::from(k),
            i128::from(stride),
            i128::from(before),
            i128::from(after),
        );
        let padded = size.checked_add(before)?.checked_add(after)?;
        let result = if transpose {
            size.checked_sub(1)?
                .checked_mul(stride)?
                .checked_sub(before)?
                .checked_sub(after)?
                .checked_add(k)?
        } else {
            if padded < k {
                return None;
            }
            padded.checked_sub(k)?.checked_div(stride)?.checked_add(1)?
        };
        (result > 0)
            .then_some(result)
            .and_then(|v| usize::try_from(v).ok())
    };
    Some(vec![
        input[0],
        channels,
        axis(input[2], kernel[0], strides[0], pads[0], pads[2])?,
        axis(input[3], kernel[1], strides[1], pads[1], pads[3])?,
    ])
}

fn infer_shape(
    node: &NodeProto,
    shapes: &BTreeMap<String, Option<Shape>>,
    graph: &GraphProto,
) -> Option<Shape> {
    let input = |index: usize| shapes.get(node.input.get(index)?).and_then(Option::as_ref);
    match node.op_type.as_str() {
        "Constant" => positive_dims(&attr(node, "value")?.t.as_ref()?.dims),
        "Conv" | "ConvTranspose" => {
            let image = input(0)?;
            let weight = input(1)?;
            if weight.len() != 4 {
                return None;
            }
            let channels = if node.op_type == "Conv" {
                weight[0]
            } else {
                weight[1]
            };
            // Kernel dimensions are taken from the weight tensor unless explicitly declared.
            let mut normalized = node.clone();
            if attr(node, "kernel_shape").is_none() {
                let mut attribute = onnx_protobuf::AttributeProto::new();
                attribute.name = "kernel_shape".into();
                attribute.ints = vec![weight[2] as i64, weight[3] as i64];
                normalized.attribute.push(attribute);
            }
            spatial_shape(
                &normalized,
                image,
                channels,
                node.op_type == "ConvTranspose",
            )
        }
        "MaxPool" => spatial_shape(node, input(0)?, *input(0)?.get(1)?, false),
        "Pad" => {
            let image = input(0)?;
            if image.len() != 4 || attr(node, "mode").is_some_and(|value| value.s != b"constant") {
                return None;
            }
            let pads_name = node.input.get(1)?;
            let pads = int64_values(graph.initializer.iter().find(|item| item.name == *pads_name)?)?;
            if pads.len() != 8 || pads.iter().any(|value| *value < 0)
                || pads[0] != 0 || pads[1] != 0 || pads[4] != 0 || pads[5] != 0 {
                return None;
            }
            Some(vec![image[0], image[1],
                      image[2].checked_add(usize::try_from(pads[2].checked_add(pads[6])?).ok()?)?,
                      image[3].checked_add(usize::try_from(pads[3].checked_add(pads[7])?).ok()?)?])
        }
        "Relu" | "Identity" | "Sigmoid" | "Softmax" | "BatchNormalization" => {
            input(0).cloned()
        }
        "Add" | "Sub" | "Mul" | "Div" => broadcast(input(0)?, input(1)?),
        "Concat" => {
            let first = input(0)?.clone();
            let rank = first.len();
            let raw_axis = attr(node, "axis")?.i;
            let axis = if raw_axis < 0 {
                usize::try_from(i64::try_from(rank).ok()?.checked_add(raw_axis)?).ok()?
            } else {
                usize::try_from(raw_axis).ok()?
            };
            if axis >= rank {
                return None;
            }
            let mut output = first;
            for index in 1..node.input.len() {
                let next = input(index)?;
                if next.len() != rank || (0..rank).any(|dim| dim != axis && next[dim] != output[dim]) {
                    return None;
                }
                output[axis] = output[axis].checked_add(next[axis])?;
            }
            Some(output)
        }
        "ReduceMean" => {
            if attr(node, "keepdims").is_some_and(|value| value.i != 1) {
                return None;
            }
            let mut shape = input(0)?.clone();
            let axes_name = node.input.get(1)?;
            let axes = graph
                .initializer
                .iter()
                .find(|item| item.name == *axes_name)?;
            let axes = int64_values(axes)?;
            for axis in axes {
                *shape.get_mut(usize::try_from(axis).ok()?)? = 1;
            }
            Some(shape)
        }
        "Shape" => {
            let rank = i64::try_from(input(0)?.len()).ok()?;
            let resolve = |value: i64| if value < 0 { rank.checked_add(value) } else { Some(value) };
            let start = resolve(attr(node, "start").map_or(0, |value| value.i))?;
            let end = resolve(attr(node, "end").map_or(rank, |value| value.i))?;
            if start < 0 || end < start || end > rank {
                return None;
            }
            Some(vec![usize::try_from(end - start).ok()?])
        }
        "Squeeze" => {
            let source = input(0)?;
            if let Some(axes_name) = node.input.get(1) {
                let axes = int64_values(graph.initializer.iter().find(|item| item.name == *axes_name)?)?;
                let rank = i64::try_from(source.len()).ok()?;
                let mut selected = BTreeSet::new();
                for axis in axes {
                    let axis = if axis < 0 { rank.checked_add(axis)? } else { axis };
                    let axis = usize::try_from(axis).ok()?;
                    if *source.get(axis)? != 1 || !selected.insert(axis) {
                        return None;
                    }
                }
                Some(source.iter().enumerate().filter_map(|(axis, dim)|
                    (!selected.contains(&axis)).then_some(*dim)).collect())
            } else {
                Some(source.iter().copied().filter(|dim| *dim != 1).collect())
            }
        }
        "Transpose" => {
            let permutation = &attr(node, "perm")?.ints;
            if permutation.len() != input(0)?.len()
                || permutation.iter().copied().collect::<BTreeSet<_>>().len() != permutation.len()
            {
                return None;
            }
            permutation
                .iter()
                .map(|axis| input(0)?.get(usize::try_from(*axis).ok()?).copied())
                .collect()
        }
        "MatMul" => {
            let left = input(0)?;
            let right = input(1)?;
            if left.len() == 3 && right.len() == 3 && left[0] == right[0] && left[2] == right[1] {
                Some(vec![left[0], left[1], right[2]])
            } else {
                None
            }
        }
        "Reshape" => {
            let shape_name = node.input.get(1)?;
            if attr(node, "allowzero").is_some_and(|value| value.i != 0) {
                return None;
            }
            if let Some(source) = graph
                .node
                .iter()
                .find(|previous| previous.output.contains(shape_name))
                && source.op_type == "Shape"
                && attr(source, "start").is_none()
                && attr(source, "end").is_none()
            {
                return shapes.get(source.input.first()?).and_then(Clone::clone);
            }
            let target = graph
                .initializer
                .iter()
                .find(|item| item.name == *shape_name)?;
            let values = int64_values(target)?;
            let source = input(0)?;
            let total = source
                .iter()
                .try_fold(1usize, |product, dim| product.checked_mul(*dim))?;
            let known = values
                .iter()
                .filter(|value| **value > 0)
                .try_fold(1usize, |product, value| {
                    product.checked_mul(usize::try_from(*value).ok()?)
                })?;
            if values.iter().filter(|value| **value == -1).count() > 1
                || values.iter().any(|value| *value == 0 || *value < -1)
                || (values.iter().all(|value| *value != -1) && total != known)
            {
                return None;
            }
            values
                .iter()
                .map(|value| match *value {
                    -1 if known > 0 && total % known == 0 => Some(total / known),
                    0 => None,
                    n if n > 0 => usize::try_from(n).ok(),
                    _ => None,
                })
                .collect()
        }
        _ => None,
    }
}

fn broadcast(left: &[usize], right: &[usize]) -> Option<Shape> {
    let rank = left.len().max(right.len());
    (0..rank)
        .map(|index| {
            let a = left
                .get(left.len().wrapping_sub(rank - index))
                .copied()
                .unwrap_or(1);
            let b = right
                .get(right.len().wrapping_sub(rank - index))
                .copied()
                .unwrap_or(1);
            (a == b || a == 1 || b == 1).then_some(a.max(b))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_protobuf::ValueInfoProto;

    fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: op.into(),
            input: inputs.iter().map(|name| (*name).into()).collect(),
            output: outputs.iter().map(|name| (*name).into()).collect(),
            ..Default::default()
        }
    }

    fn graph() -> GraphProto {
        GraphProto {
            input: vec![ValueInfoProto {
                name: "image".into(),
                ..Default::default()
            }],
            output: vec![ValueInfoProto {
                name: "logits".into(),
                ..Default::default()
            }],
            node: vec![
                node("Relu", &["image"], &["skip"]),
                node("Relu", &["skip"], &["deep"]),
                node("Add", &["deep", "skip"], &["logits"]),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn skip_tensor_stays_live_across_later_cut() {
        let analysis = analyze(&graph(), None).unwrap();
        assert_eq!(analysis.cuts[1], ["deep", "skip"]);
        assert_eq!(analysis.consumers["skip"], [1, 2]);
    }

    #[test]
    fn division_broadcasts_channel_constants() {
        let shapes = BTreeMap::from([
            ("image".into(), Some(vec![1, 3, 64, 64])),
            ("constant".into(), Some(vec![1, 3, 1, 1])),
        ]);
        assert_eq!(infer_shape(&node("Div", &["image", "constant"], &["scaled"]),
                               &shapes, &GraphProto::new()), Some(vec![1, 3, 64, 64]));
    }

    #[test]
    fn sliced_shape_and_squeeze_have_boundary_shapes() {
        let shapes = BTreeMap::from([("image".into(), Some(vec![1, 3, 512, 512]))]);
        let mut shape = node("Shape", &["image"], &["height_vector"]);
        for (name, value) in [("start", 2), ("end", 3)] {
            let mut attribute = onnx_protobuf::AttributeProto::new();
            attribute.name = name.into();
            attribute.i = value;
            shape.attribute.push(attribute);
        }
        assert_eq!(infer_shape(&shape, &shapes, &GraphProto::new()), Some(vec![1]));
        let shapes = BTreeMap::from([("height_vector".into(), Some(vec![1]))]);
        assert_eq!(infer_shape(&node("Squeeze", &["height_vector"], &["height"]),
                               &shapes, &GraphProto::new()), Some(vec![]));
    }

    #[test]
    fn missing_input_identifies_the_node() {
        let mut broken = graph();
        broken.node[1].input[0] = "missing".into();
        assert!(
            analyze(&broken, None)
                .err()
                .unwrap()
                .contains("node 1 (Relu) has missing input missing")
        );
    }

    #[test]
    fn duplicate_output_is_rejected() {
        let mut broken = graph();
        broken.node[1].output[0] = "skip".into();
        assert!(
            analyze(&broken, None)
                .err()
                .unwrap()
                .contains("duplicate producers")
        );
    }

    #[test]
    fn invalid_spatial_parameters_have_no_size_estimate() {
        let mut pool = node("MaxPool", &["image"], &["pooled"]);
        let mut kernel = onnx_protobuf::AttributeProto::new();
        kernel.name = "kernel_shape".into();
        kernel.ints = vec![2, 2];
        pool.attribute.push(kernel);
        let mut stride = onnx_protobuf::AttributeProto::new();
        stride.name = "strides".into();
        stride.ints = vec![0, 2];
        pool.attribute.push(stride);
        assert_eq!(spatial_shape(&pool, &[1, 3, 8, 8], 3, false), None);
    }

    #[test]
    fn malformed_int64_initializer_has_no_shape_values() {
        let mut tensor = onnx_protobuf::TensorProto::new();
        tensor.data_type = 7;
        tensor.dims = vec![2];
        tensor.raw_data = vec![0; 8];
        assert_eq!(int64_values(&tensor), None);
    }

    #[test]
    fn constant_of_shape_infers_element_type() {
        let mut g = GraphProto::new();
        let mut shape_input = ValueInfoProto::new();
        shape_input.name = "shape_tensor".into();
        let mut ty = onnx_protobuf::TypeProto::new();
        ty.mut_tensor_type().elem_type = 7;
        shape_input.type_ = protobuf::MessageField::some(ty);
        g.input.push(shape_input);

        let mut cos_node = node("ConstantOfShape", &["shape_tensor"], &["const_out"]);
        let mut attr_val = onnx_protobuf::AttributeProto::new();
        attr_val.name = "value".into();
        let mut val_tensor = onnx_protobuf::TensorProto::new();
        val_tensor.data_type = 1; // float32
        val_tensor.dims = vec![1];
        attr_val.t = protobuf::MessageField::some(val_tensor);
        cos_node.attribute.push(attr_val);
        g.node.push(cos_node);

        let analysis = analyze(&g, None).unwrap();
        assert_eq!(analysis.element_types["const_out"], Some(1));
        assert_eq!(analysis.element_sizes["const_out"], Some(4));
    }

    #[test]
    fn constant_and_static_spatial_pad_shapes_are_inferred() {
        let mut constant = node("Constant", &[], &["shape_two"]);
        let mut attribute = onnx_protobuf::AttributeProto::new();
        attribute.name = "value".into();
        let mut tensor = onnx_protobuf::TensorProto::new();
        tensor.data_type = 7;
        tensor.dims = vec![1];
        tensor.int64_data = vec![2];
        attribute.t = protobuf::MessageField::some(tensor);
        constant.attribute.push(attribute);
        assert_eq!(infer_shape(&constant, &BTreeMap::new(), &GraphProto::new()),
                   Some(vec![1]));

        let mut pads = onnx_protobuf::TensorProto::new();
        pads.name = "pads".into();
        pads.data_type = 7;
        pads.dims = vec![8];
        pads.int64_data = vec![0, 0, 1, 2, 0, 0, 1, 2];
        let mut graph = GraphProto::new();
        graph.initializer.push(pads);
        let shapes = BTreeMap::from([("image".into(), Some(vec![1, 3, 10, 12]))]);
        assert_eq!(infer_shape(&node("Pad", &["image", "pads"], &["padded"]),
                               &shapes, &graph), Some(vec![1, 3, 12, 16]));
    }
}
