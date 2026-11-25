# SAR shoreline independent holdout preflight

Recorded 2026-11-25. [Amendment 6](evaluation-amendment-6.md), the logits-only projection algorithm, source revision, input seeds, cap, numerical thresholds, and paired benchmark rule were committed before model download or graph access. The downloaded source and generated projected model and rasters matched the hashes committed in `d80b4ca`. The complete generated doctor record is `runs/holdout/sar3/preflight.json` (ignored by Git).

The projected model declares one FP32 NCHW three-band input and one FP32 NCHW two-logit output, both with dynamic spatial dimensions. The 512×512 three-band GeoTIFF has EPSG:32643, no nodata, and the frozen SHA-256 hash. The model has 194 selected logits-ancestor nodes. The installation doctor reported **failed** before inference: first unsupported node **1**, `Squeeze`, has no validated raster schedule.

No automatic schedule, numerical oracle, manual tuning, or five-pair benchmark was run. This is a **failed independent compatibility result** and cannot establish the feasibility pass. The graph is now exposed development material. Subsequent inspection showed the `Squeeze` operations are part of shape calculations; supporting that pattern later will require another newly frozen model for an unseen decision test.

## Exposed-model development follow-up

After the failed preflight, the local analyzer gained shape-only `Squeeze`, scalar shape arithmetic, sliced `Shape`, and the model's nearest-neighbor `Resize` defaults. The Rust splitter now infers sliced `Shape` and `Squeeze` boundary shapes. Resize sizes are evaluated in an isolated shape-only subgraph and checked against the inferred spatial result. Shape inference uses initializer dimensions without copying large convolution weights into the concrete probe, avoiding a 1 GiB preflight allocation failure.

On `sar3-512.tif`, fixed schedules with 512-pixel and 256-pixel tiles both matched whole-image FP32 logits exactly (`max_absolute_error=0`, 2×512×512 values) and preserved EPSG:32643 and the affine transform. The 256-pixel run used four tiles; its enforced job peak committed memory was 847,392,768 bytes. Automatic execution selected tile 512, cut 0, memory storage and also matched exactly, with 729,960,448 peak committed bytes under the 1,073,741,824-byte job cap. Local run reports and comparison records are under ignored `runs/holdout/sar3/`.

These runs test implementation behavior on an already exposed model. They do not replace the failed independent result, the frozen paired performance benchmark, or a new unseen model and machine configuration.

The larger `sar3-1024.tif` automatic run also produced four 512-pixel tiles whose 2×1024×1024 logits matched a separate whole-image oracle exactly. The selected plan's nested Windows job reported `PeakJobMemoryUsed=2,058,858,496` bytes against the configured 1,073,741,824-byte cap, while 50 ms sampled private commitment peaked at 988,524,544 bytes and sampled working set at 829,452,288 bytes. No failed candidate was recorded. Under the [frozen memory evidence rule](evaluation-amendment-2.md), this run is **numerically correct but fails strict memory evidence**. The fact that it completed under an active job limit does not justify treating the raw counter as a pass.

Disabling ONNX Runtime's CPU arena and memory-pattern cache for Rust inference sessions resolved that development case. With the same model, raster, 1 GiB cap, and two threads, a new automatic run selected four 512-pixel tiles and matched the whole-image logits exactly. Its nested and outer raw job peaks were both 730,681,344 bytes, its sampled private-commit peak was 730,644,480 bytes, and its total wall time was 41.84 seconds. The earlier run remains recorded above as a genuine failed memory-evidence observation. A fixed 128-pixel, 64-tile schedule also matched exactly with a 980,717,568-byte raw peak, but took 204.48 seconds. The 512-pixel fixed schedule with the arena disabled took 31.23 seconds and peaked at 729,636,864 bytes. These timings are single development runs, not paired baseline evidence.

Reproduce the current automatic development run after generating the frozen model and raster artifacts per [amendment 6](evaluation-amendment-6.md):

```powershell
cargo build --locked
.\.venv\Scripts\python.exe tests\performance\whole_image_oracle.py --model models\holdout\sar3-logits.onnx --input data\holdout\sar3\sar3-1024.tif --reference-output runs\holdout\sar3\oracle-1024.tif --report runs\holdout\sar3\oracle-1024.json --threads 2
.\.venv\Scripts\python.exe scripts\run_raster_auto.py --model models\holdout\sar3-logits.onnx --input data\holdout\sar3\sar3-1024.tif --output runs\holdout\sar3\auto-1024.tif --reference-raster runs\holdout\sar3\oracle-1024.tif --memory-limit-mib 1024 --threads 2 --max-tile-size 512
```
