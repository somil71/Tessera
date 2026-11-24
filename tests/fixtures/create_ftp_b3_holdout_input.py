"""Generate amendment 5's predeclared synthetic eight-band raster."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "holdout" / "ftp-b3" / "ftp-b3-512.tif"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(5107)
image = rng.random((8, 512, 512), dtype=np.float32)
with rasterio.open(
    OUTPUT, "w", driver="GTiff", height=512, width=512, count=8,
    dtype="float32", crs="EPSG:32643",
    transform=from_origin(500000, 2000000, 3, 3),
) as raster:
    raster.write(image)
print(OUTPUT)
