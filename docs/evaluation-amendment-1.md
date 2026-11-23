# Evaluation amendment 1: attention holdout exposure

Recorded 2026-11-23. The frozen [evaluation protocol](evaluation-protocol.md) designated `attention_holdout.onnx` as unseen until the planner and attention support were frozen. During Phases 2 and 3 it was used for graph inspection, subgraph validation, and Rust execution smoke tests. Its architecture and behavior are therefore exposed to the development process.

The file, generator, and SHA-256 remain unchanged for reproducibility. It is now an **attention development fixture**, despite its historical filename. Results on it may demonstrate that the implementation handles that graph, but they cannot be counted as held-out evidence or used to claim cross-model generalization.

Before running the feasibility decision, freeze a new independent held-out model, input seeds, and model-machine pair under a versioned protocol. The planner author must not tune against that pair. The numerical thresholds, committed-memory limits, manual-baseline allowance, and runtime decision rules in the original protocol remain the intended acceptance criteria. Until a valid replacement is selected and run, the held-out evaluation is incomplete.
