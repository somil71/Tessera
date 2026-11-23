# U-Net raster storage planner integration

Recorded 2026-11-23. These are development checks, not the independent held-out feasibility result.

The fixed U-Net GeoTIFF runner can execute a selected graph cut with boundary tensors stored on disk. The automatic runner profiles RAM tile/cut choices first under its Windows Job Object cap. If none passes, it profiles disk choices. When the fastest predicted RAM plan is at least 10 seconds, it also profiles larger disk tiles, where fewer tiles may compensate for spill overhead. If a probe passes but the complete raster run fails, it tries the remaining feasible plans and then profiles untried disk tiles. `--explore-spill` forces the extra disk probes for a diagnostic run.

The 512×512, three-band seeded U-Net development fixture used a 128 MiB job limit and four inference threads. `tests/fixtures/smoke_storage_planner.py` compared both fixed disk and automatic outputs with whole-image ONNX Runtime logits at 1e-4 maximum absolute tolerance. Both passed and preserved EPSG:32643 and the source tag. A separate raster comparison found zero differing FP32 values and matching output hashes between the two schedules.

| Run | Selected plan | Wall time | Raw job peak | Sampled private peak | Boundary spill |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed disk | tile 512, cut 4 | 0.59 s | 117.00 MB | 117.00 MB | 12.58 MB each way |
| Automatic with forced disk exploration | RAM tile 256, cut 1 | 19.93 s including calibration | 167.74 MB | 134.07 MB | 0 in selected execution |

The automatic run evaluated 70 candidates, including seven disk candidates; six disk candidates passed their probes. It selected RAM because its estimated execution was faster. This check demonstrates disk eligibility, numerical correctness, and cost-based rejection of slower spill. It does **not** demonstrate a case where the planner selects disk and wins against a tuned baseline. Forced exploration makes planning expensive on this small raster. Without that diagnostic flag, the short predicted run skips discretionary disk probes. The report records all candidate timings, errors, and storage modes.

The automatic run's raw Windows `PeakJobMemoryUsed` counter exceeded the 134,217,728-byte configured hard limit after failed probes. The sampled private commitment stayed below it. The raw counter is retained as observed; [evaluation amendment 2](evaluation-amendment-2.md) explains the evidence rule and its limits. A sampled value is not an exact historical peak. A hard process kill may leave spill files for manual cleanup.

Reproduce on the pinned Windows development environment:

```powershell
cargo build --locked
.\.venv\Scripts\python.exe tests\fixtures\create_eval_models.py
.\.venv\Scripts\python.exe tests\fixtures\smoke_storage_planner.py
```

The raw outputs and reports are in ignored `runs/storage_planner_smoke/`. This planner applies only to the validated U-Net graph topology; it does not provide arbitrary tensor placement, recomputation, or transformer storage planning.
