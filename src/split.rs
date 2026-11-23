use std::collections::BTreeSet;

use onnx_protobuf::{GraphProto, ModelProto};

use crate::graph::Analysis;

pub fn at(
    model: &ModelProto,
    analysis: &Analysis,
    cut: usize,
) -> Result<(ModelProto, ModelProto), String> {
    let graph = model.graph.as_ref().ok_or("ONNX model has no graph")?;
    let boundary = analysis
        .boundary(cut)
        .ok_or_else(|| format!("cut {cut} is outside the graph's internal boundaries"))?;
    if boundary.is_empty() {
        return Err(format!("cut {cut} has no live boundary tensors"));
    }
    if !graph.sparse_initializer.is_empty() {
        return Err("sparse initializers are not supported for splitting".into());
    }
    if let Some((index, node)) = graph.node.iter().enumerate().find(|(_, node)| {
        node.attribute
            .iter()
            .any(|attribute| attribute.g.is_some() || !attribute.graphs.is_empty())
    }) {
        return Err(format!(
            "node {index} ({}) contains a nested graph; splitting is unsupported",
            node.op_type
        ));
    }
    if graph
        .initializer
        .iter()
        .any(|item| !item.external_data.is_empty())
    {
        return Err("external initializer data is not supported for splitting".into());
    }

    let head_nodes = &graph.node[..=cut];
    let tail_nodes = &graph.node[cut + 1..];
    let head_inputs = input_names(head_nodes);
    let tail_inputs = input_names(tail_nodes);
    let head_outputs = boundary
        .iter()
        .map(|name| analysis.value_info(name))
        .collect::<Result<Vec<_>, _>>()?;
    let tail_boundary = boundary
        .iter()
        .filter(|name| {
            tail_inputs.contains(name.as_str()) || graph.output.iter().any(|out| out.name == **name)
        })
        .map(|name| analysis.value_info(name))
        .collect::<Result<Vec<_>, _>>()?;

    let mut head = GraphProto {
        name: format!("{}_head_{cut}", graph.name),
        node: head_nodes.to_vec(),
        input: graph
            .input
            .iter()
            .filter(|item| head_inputs.contains(item.name.as_str()))
            .cloned()
            .collect(),
        output: head_outputs,
        initializer: graph
            .initializer
            .iter()
            .filter(|item| head_inputs.contains(item.name.as_str()))
            .cloned()
            .collect(),
        ..Default::default()
    };
    let mut tail = GraphProto {
        name: format!("{}_tail_{cut}", graph.name),
        node: tail_nodes.to_vec(),
        input: tail_boundary,
        output: graph.output.clone(),
        initializer: graph
            .initializer
            .iter()
            .filter(|item| tail_inputs.contains(item.name.as_str()))
            .cloned()
            .collect(),
        ..Default::default()
    };
    // Keep available intermediate metadata only for tensors present in each graph.
    let head_names = tensor_names(&head);
    head.value_info = graph
        .value_info
        .iter()
        .filter(|item| head_names.contains(item.name.as_str()))
        .cloned()
        .collect();
    let tail_names = tensor_names(&tail);
    tail.value_info = graph
        .value_info
        .iter()
        .filter(|item| tail_names.contains(item.name.as_str()))
        .cloned()
        .collect();

    let mut first = model.clone();
    first.graph = protobuf::MessageField::some(head);
    let mut second = model.clone();
    second.graph = protobuf::MessageField::some(tail);
    Ok((first, second))
}

fn input_names(nodes: &[onnx_protobuf::NodeProto]) -> BTreeSet<&str> {
    nodes
        .iter()
        .flat_map(|node| node.input.iter().map(String::as_str))
        .filter(|name| !name.is_empty())
        .collect()
}

fn tensor_names(graph: &GraphProto) -> BTreeSet<&str> {
    graph
        .node
        .iter()
        .flat_map(|node| node.input.iter().chain(&node.output).map(String::as_str))
        .chain(graph.input.iter().map(|item| item.name.as_str()))
        .chain(graph.output.iter().map(|item| item.name.as_str()))
        .collect()
}
