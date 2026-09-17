"""Generate real side-by-side clean-vs-perturbed patch comparisons for
(a) visual sanity-checking the calibrated perturbation suite, and
(b) reusable figure panels for the eventual paper.

Pure CPU/PIL work — no CONCH/UNI/GPU dependency, safe to run alongside any
GPU-bound experiment.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from perturbations import PERTURBATIONS, JPEG_SWEEP, STAIN_MAGNITUDE_SWEEP

OUT_DIR = PROJECT_ROOT / "results" / "figures"
N_SAMPLE_PATCHES = 8  # 2 per IDC sample (4 samples) for broader coverage
LABEL_HEIGHT = 24


def load_patches(n: int) -> list[Image.Image]:
    """Sample across ALL 4 IDC samples, not just one — a single-sample figure
    risks visual conclusions that don't generalize across the (real, already
    documented) sample-to-sample variability in stain/tissue appearance."""
    st_dataset_path = PROJECT_ROOT / "external/HEST/src/hest/bench/st_dataset.py"
    spec = importlib.util.spec_from_file_location("st_dataset", st_dataset_path)
    st_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st_dataset)

    sample_ids = ["NCBI783", "NCBI785", "TENX95", "TENX99"]
    patches = []
    per_sample = max(1, n // len(sample_ids))
    for sid in sample_ids:
        ds = st_dataset.H5PatchDataset(
            str(PROJECT_ROOT / f"data/hest-bench/IDC/patches/{sid}.h5"), img_transform=None
        )
        step = max(1, len(ds) // (per_sample * 5))
        indices = [i * step for i in range(per_sample)]
        patches.extend(Image.fromarray(ds[i]["imgs"].astype(np.uint8)) for i in indices)
    return patches[:n] if n <= len(patches) else patches


def label_image(img: Image.Image, text: str) -> Image.Image:
    w, h = img.size
    canvas = Image.new("RGB", (w, h + LABEL_HEIGHT), (255, 255, 255))
    canvas.paste(img, (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), text, fill=(0, 0, 0))
    return canvas


def make_row(patch: Image.Image, perturbations: dict[str, callable]) -> Image.Image:
    labeled = [label_image(patch, "clean")]
    for name, fn in perturbations.items():
        labeled.append(label_image(fn(patch), name))
    total_w = sum(im.width for im in labeled)
    max_h = max(im.height for im in labeled)
    row = Image.new("RGB", (total_w, max_h), (255, 255, 255))
    x = 0
    for im in labeled:
        row.paste(im, (x, 0))
        x += im.width
    return row


def stack_rows(rows: list[Image.Image]) -> Image.Image:
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows) + 4 * (len(rows) - 1)
    canvas = Image.new("RGB", (w, h), (200, 200, 200))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height + 4
    return canvas


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    patches = load_patches(N_SAMPLE_PATCHES)
    print(f"Loaded {len(patches)} sample patches", flush=True)

    # 1. Main perturbation suite (calibrated, SSIM~0.90) — the primary figure
    rows = [make_row(p, PERTURBATIONS) for p in patches]
    grid = stack_rows(rows)
    out_path = OUT_DIR / "perturbation_suite_comparison.png"
    grid.save(out_path)
    print(f"Wrote {out_path} ({grid.width}x{grid.height})", flush=True)

    # 2. JPEG quality dose-response — clean | q30 | q50 | q70 | q90
    rows = [make_row(p, JPEG_SWEEP) for p in patches]
    grid = stack_rows(rows)
    out_path = OUT_DIR / "jpeg_dose_response_comparison.png"
    grid.save(out_path)
    print(f"Wrote {out_path} ({grid.width}x{grid.height})", flush=True)

    # 3. Stain-shift magnitude comparison — clean | crude(old) | macenko identity (null control) |
    #    macenko mild | macenko moderate (5 columns total; updated 2026-09-16 when the identity
    #    null control was added to STAIN_MAGNITUDE_SWEEP)
    from perturbations import he_stain_shift
    stain_perturbations = {"crude_old": he_stain_shift, **STAIN_MAGNITUDE_SWEEP}
    rows = [make_row(p, stain_perturbations) for p in patches]
    grid = stack_rows(rows)
    out_path = OUT_DIR / "stain_shift_comparison.png"
    grid.save(out_path)
    print(f"Wrote {out_path} ({grid.width}x{grid.height})", flush=True)

    print("\nAll figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
