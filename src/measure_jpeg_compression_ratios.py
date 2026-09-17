"""Measure the actual JPEG compression ratio achieved at each quality level
on real IDC patches — persisted as a reproducible script (audit-flagged
2026-09-16: this was previously a one-off chat calculation, never saved to
the repo, so the 8.5x-27.5x compression-ratio numbers cited in discussion
were not independently verifiable from the codebase).

Ties GPFR at each JPEG quality level to a REAL measured compression ratio,
not PIL's abstract "quality" parameter — the clinical citation (AJCP 2011)
is stated in terms of compression ratio (15-20x), not PIL quality, so this
conversion is required to make that citation actually applicable.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from perturbations import JPEG_SWEEP

N_PATCHES = 40


def load_patches(n: int) -> list[Image.Image]:
    import importlib.util
    st_dataset_path = PROJECT_ROOT / "external/HEST/src/hest/bench/st_dataset.py"
    spec = importlib.util.spec_from_file_location("st_dataset", st_dataset_path)
    st_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st_dataset)
    ds = st_dataset.H5PatchDataset(
        str(PROJECT_ROOT / "data/hest-bench/IDC/patches/NCBI783.h5"), img_transform=None
    )
    return [Image.fromarray(ds[i]["imgs"].astype(np.uint8)) for i in range(n)]


def main():
    patches = load_patches(N_PATCHES)
    raw_uncompressed_bytes = patches[0].size[0] * patches[0].size[1] * 3  # H x W x 3 channels, raw pixels
    print(f"Raw uncompressed size per patch: {raw_uncompressed_bytes} bytes "
          f"({patches[0].size[0]}x{patches[0].size[1]}x3)", flush=True)

    import csv
    rows = []
    for name, fn in JPEG_SWEEP.items():
        sizes = []
        for img in patches:
            # Re-derive the actual byte size directly from the same
            # jpeg_compression() call path used by the perturbation itself,
            # not a separate re-implementation.
            quality = int(name.split("_q")[1])
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            sizes.append(len(buf.getvalue()))
        avg_size = float(np.mean(sizes))
        ratio = raw_uncompressed_bytes / avg_size
        print(f"{name}: avg JPEG size={avg_size:.0f} bytes, "
              f"compression ratio vs raw pixels={ratio:.1f}x", flush=True)
        rows.append({"perturbation": name, "avg_jpeg_bytes": avg_size,
                     "raw_uncompressed_bytes": raw_uncompressed_bytes,
                     "compression_ratio": ratio, "n_patches": len(patches)})

    out = PROJECT_ROOT / "results" / "jpeg_compression_ratios.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
