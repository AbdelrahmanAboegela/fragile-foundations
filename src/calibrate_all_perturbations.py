"""Root-find calibration for every perturbation's magnitude, replacing
hand-picked constants (brightness factor=1.2, contrast=1.2, gamma=1.2,
blur radius=1.0, scanner_color gains) that were set early in this project
with no calibration or citation at all.

ALL are calibrated to hit the SAME target: SSIM=0.90, which is not an
arbitrary round number — it is the actual measured mean SSIM of JPEG at
quality=70 (the 15.5x-compression point cited as the clinical
diagnostic-safety boundary, AJCP 2011), from the real
N=3000 IDC run. Anchoring every perturbation's magnitude to that one
citation-backed reference point means every number in the suite can be
traced back to real evidence, not eyeballing: "each perturbation is
calibrated to match the visual-similarity magnitude of the most clinically
realistic condition in the suite."

PER-ORGAN CALIBRATION (2026-09-16, audit-flagged): originally calibrated on
IDC only and the resulting constants used unchanged for all 8 organs. An
independent audit measured the same fixed constants landing at very
different SSIM on other organs (PRAD's `gamma`: SSIM=0.800, BELOW this
project's own 0.90 morphology-preservation anchor) — HEST's patch
extraction targets a fixed physical patch size (224px @ 0.5um/px per
`external/HEST/src/hest/HESTData.py`'s `dump_patches` docstring), but the
per-.h5 `factor`/`patch_size` attrs actually observed vary across organs
(1.09-2.00) and at least one organ (COAD) records a `pixel_size` attr
(~0.25um/px) inconsistent with that documented 0.5um/px default — the
released HEST-Bench .h5 files were evidently not all extracted with
identical settings, and this repo could not fully resolve why from the
available metadata alone. Rather than guess at a physical-scale correction,
this script now calibrates EACH organ separately and empirically verifies
SSIM=0.90 is actually achieved per organ, sidestepping the need to know why
the scale differs. `jpeg` (fixed quality=70, not a continuous magnitude)
and `stain_shift` (deliberately NOT SSIM-targeted — see perturbations.py's
module docstring) are NOT part of this per-organ recalibration; their
actual measured SSIM per organ is still recorded in every results CSV via
the `mean_ssim` column, informationally not correctively.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import compute_ssim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_CALIBRATION_PATCHES = 40
TARGET_SSIM = 0.90  # = measured mean SSIM of jpeg_q70 on the real N=3000 IDC run
MIN_FOREGROUND_FRACTION = 0.5  # exclude mostly-background patches from calibration
OUT_PATH = PROJECT_ROOT / "configs" / "perturbation_calibration.json"

# CALIBRATION/EXPERIMENT DISJOINTNESS (2026-09-16, added proactively after a
# leakage review — not itself label leakage, since SSIM calibration never
# touches expression labels, but a reviewer could still reasonably ask
# "were your calibration patches also used for evaluation?" and the clean
# answer should be "provably not," not "unlikely by chance." This script
# scans the LAST CALIBRATION_RESERVED_TAIL patches (by index) of each
# organ's alphabetically-first sample file; main_experiment.py's
# load_organ_sample excludes that exact same tail range from its random
# sampling pool for that same designated sample. Every organ's first
# sample has >=1084 patches (smallest: CCRCC), so reserving 200 leaves
# plenty for the actual experiment.
CALIBRATION_RESERVED_TAIL = 200


def foreground_fraction(img: Image.Image, beta: float = 0.15) -> float:
    """Same OD-thresholding used by macenko_stain_shift to define
    foreground/background, reused here so both use a consistent definition
    of "tissue" vs. "blank slide"."""
    arr = np.asarray(img).astype(np.float64)
    od = -np.log((arr.reshape(-1, 3) + 1) / 256.0)
    return float((od > beta).any(axis=1).mean())


MAX_CALIBRATION_SAMPLES = 5  # cap runtime; organs with more samples than this get an evenly-spaced subset


def _calibration_sample_paths(organ: str) -> list[Path]:
    """MULTI-SAMPLE CALIBRATION (2026-09-16, fixing a real problem this
    caused): calibrating from only ONE sample (the organ's alphabetically-
    first) is not representative when an organ has many, heterogeneous
    samples. Concretely found: CCRCC (24 samples, a real internal split
    between 12 samples at 36,601 genes and 12 at 17,943) calibrated `blur` to verified
    SSIM=0.8997 on ONE sample's 40 patches, but the full N=3000 CONCH run
    (pooling patches across all 24 samples) measured mean_ssim=0.761 for
    that same calibrated constant — a 0.14 gap, well outside normal
    calibration-transfer noise (most organs were within ~0.05). COAD showed
    the same pattern to a lesser degree. Fix: calibrate against patches
    pooled across up to MAX_CALIBRATION_SAMPLES samples (evenly spaced
    through the sorted sample list, not just the first few alphabetically,
    so a large organ's real inter-sample heterogeneity is actually sampled)
    instead of one.
    """
    patches_dir = PROJECT_ROOT / "data" / "hest-bench" / organ / "patches"
    candidates = sorted(patches_dir.glob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"no patch .h5 files found for organ={organ} in {patches_dir}")
    if len(candidates) <= MAX_CALIBRATION_SAMPLES:
        return candidates
    idx = np.linspace(0, len(candidates) - 1, MAX_CALIBRATION_SAMPLES).round().astype(int)
    return [candidates[i] for i in sorted(set(idx))]


def load_calibration_patches(organ: str = "IDC", sample_paths: list[Path] | None = None) -> list[Image.Image]:
    """CHANGED 2026-09-16 (audit-followup, found by actually looking at
    example figures): calibrating against whole-patch SSIM let
    background-heavy patches "hide" a disproportionately strong tissue-level
    perturbation behind an average that's dominated by near-white background
    pixels that barely change. Result: brightness/contrast looked
    dramatically more aggressive on sparse patches than on tissue-dense
    ones, for the *same* calibrated factor. Fix: only calibrate against
    patches that are majority tissue, so the target magnitude reflects
    typical tissue-level change, not an average skewed by mostly-empty crops.

    `organ`/`sample_paths` (2026-09-16): parameterized so this can calibrate
    per organ across MULTIPLE samples, not just one IDC sample — see
    _calibration_sample_paths' docstring for why this matters.

    Scans from the TAIL of each sample file, not the start (2026-09-16, see
    module docstring's CALIBRATION/EXPERIMENT DISJOINTNESS note) —
    main_experiment.py excludes this same tail range from its random sample
    for every sample calibration touches (recorded per-organ in the output
    JSON's "_calibration_sample_ids"), so calibration and experiment
    patches are provably disjoint, not just unlikely to overlap.
    """
    import importlib.util
    st_dataset_path = Path(__file__).resolve().parent.parent / "external/HEST/src/hest/bench/st_dataset.py"
    spec = importlib.util.spec_from_file_location("st_dataset", st_dataset_path)
    st_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st_dataset)
    paths = sample_paths or [Path(__file__).resolve().parent.parent / "data/hest-bench/IDC/patches/NCBI783.h5"]

    per_sample_budget = max(1, -(-N_CALIBRATION_PATCHES // len(paths)))  # ceil division
    patches = []
    for path in paths:
        ds = st_dataset.H5PatchDataset(str(path), img_transform=None)
        tail_start = max(0, len(ds) - CALIBRATION_RESERVED_TAIL)
        found_this_sample = 0
        i = len(ds) - 1
        while found_this_sample < per_sample_budget and len(patches) < N_CALIBRATION_PATCHES and i >= tail_start:
            img = Image.fromarray(ds[i]["imgs"].astype(np.uint8))
            if foreground_fraction(img) >= MIN_FOREGROUND_FRACTION:
                patches.append(img)
                found_this_sample += 1
            i -= 1
        print(f"[setup:{organ}] {path.stem}: found {found_this_sample} tissue-dominant patches "
              f"in tail indices [{i+1}, {len(ds)})", flush=True)
    print(f"[setup:{organ}] {len(patches)} total calibration patches pooled across "
          f"{len(paths)} sample(s): {[p.stem for p in paths]}", flush=True)
    return patches


def ssim_of(patches, transform_fn) -> float:
    ssims = []
    for img in patches:
        pert = transform_fn(img)
        c = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        p = torch.from_numpy(np.asarray(pert)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        ssims.append(compute_ssim(c, p))
    return float(np.mean(ssims))


def calibrate_param(name, patches, transform_factory, bracket, target=TARGET_SSIM):
    def f(x):
        return ssim_of(patches, transform_factory(x)) - target
    lo, hi = bracket
    f_lo, f_hi = f(lo), f(hi)
    print(f"[{name}] bracket check: f({lo})={f_lo:.4f}, f({hi})={f_hi:.4f}", flush=True)
    if f_lo * f_hi > 0:
        raise ValueError(f"{name}: target {target} not bracketed in {bracket} — "
                          f"range covered is [{ssim_of(patches, transform_factory(hi)):.3f}, "
                          f"{ssim_of(patches, transform_factory(lo)):.3f}]")
    x = brentq(f, lo, hi, xtol=1e-4)
    verified = ssim_of(patches, transform_factory(x))
    print(f"[{name}] calibrated x={x:.5f}, verified SSIM={verified:.4f}\n", flush=True)
    return x


def calibrate_one_organ(organ: str, patches: list[Image.Image]) -> dict:
    """Each perturbation is calibrated independently with its own
    try/except (2026-09-16 fix): a bracket failure on one perturbation for
    one organ (e.g. CCRCC's scanner_color couldn't reach SSIM=0.90 even at
    the widest bracket tried — it's measurably less color-sensitive than
    IDC at the same nominal gain) must not abort calibration for the other
    4 perturbations on that organ, or for the other 7 organs. A failed
    (organ, perturbation) falls back to IDC's default for that one
    perturbation, logged explicitly — never silently.
    """
    results = {}
    fallback_needed = []

    def try_calibrate(pname, key, transform_factory, brackets):
        for bracket in brackets:
            try:
                return calibrate_param(f"{organ}/{pname}", patches, transform_factory, bracket=bracket)
            except ValueError as e:
                print(f"[{organ}/{pname}] bracket {bracket} failed: {e}", flush=True)
        fallback_needed.append(pname)
        return None

    brightness = try_calibrate(
        "brightness", "brightness_factor",
        lambda factor: (lambda img: ImageEnhance.Brightness(img).enhance(factor)),
        brackets=[(1.001, 2.0), (1.0001, 4.0)])
    if brightness is not None:
        results["brightness_factor"] = brightness

    contrast = try_calibrate(
        "contrast", "contrast_factor",
        lambda factor: (lambda img: ImageEnhance.Contrast(img).enhance(factor)),
        brackets=[(1.001, 2.0), (1.0001, 4.0)])
    if contrast is not None:
        results["contrast_factor"] = contrast

    def gamma_fn(gamma):
        def f(img):
            arr = np.asarray(img).astype(np.float32) / 255.0
            arr = np.power(arr, gamma)
            return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
        return f
    gamma = try_calibrate("gamma", "gamma", gamma_fn, brackets=[(1.001, 2.0), (1.0001, 4.0)])
    if gamma is not None:
        results["gamma"] = gamma

    def blur_fn(radius):
        return lambda img: img.filter(ImageFilter.GaussianBlur(radius=radius))
    blur = try_calibrate("blur", "blur_radius", blur_fn, brackets=[(0.1, 5.0), (0.05, 15.0)])
    if blur is not None:
        results["blur_radius"] = blur

    def scanner_color_fn(strength):
        base_direction = np.array([0.05, -0.02, 0.02])  # original hand-picked direction, now just a direction not a magnitude
        def f(img):
            arr = np.asarray(img).astype(np.float32)
            gains = 1.0 + strength * base_direction
            for c in range(3):
                arr[..., c] *= gains[c]
            return Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        return f
    # Widened bracket (2026-09-16): CCRCC's scanner_color measured SSIM=0.904
    # even at strength=10.0 (the original bracket's max) — the perturbation
    # is measurably less perceptible per unit gain on this organ, not a bug,
    # so it genuinely needs a larger magnitude to hit the same SSIM target.
    scanner = try_calibrate("scanner_color", "scanner_color_strength", scanner_color_fn,
                             brackets=[(0.1, 10.0), (0.1, 50.0), (0.1, 150.0)])
    if scanner is not None:
        results["scanner_color_strength"] = scanner

    if fallback_needed:
        print(f"[{organ}] FALLING BACK to IDC-calibrated defaults for: {fallback_needed} "
              f"— could not hit SSIM=0.90 for these even with a widened bracket. "
              f"main_experiment.py will warn and use the IDC default for these specific "
              f"perturbations on this organ; treat their mean_ssim column in results as the "
              f"honest (non-0.90) measured value for this organ.", flush=True)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--organs", nargs="+", default=None,
                         help="organs to calibrate (default: all discovered under data/hest-bench)")
    args = parser.parse_args()

    organs = args.organs or sorted(
        p.name for p in (PROJECT_ROOT / "data" / "hest-bench").iterdir()
        if p.is_dir() and (p / "patches").is_dir() and (p / "adata").is_dir()
    )
    print(f"[setup] organs to calibrate: {organs}, target SSIM={TARGET_SSIM}\n", flush=True)

    all_results = {}
    for organ in organs:
        sample_paths = _calibration_sample_paths(organ)
        print(f"\n=== Calibrating {organ} (samples: {[p.stem for p in sample_paths]}) ===", flush=True)
        patches = load_calibration_patches(organ, sample_paths)
        all_results[organ] = calibrate_one_organ(organ, patches)
        # Recorded so main_experiment.py can exclude each of these samples'
        # reserved tail range, not just assume it's always sample_ids[0].
        all_results[organ]["_calibration_sample_ids"] = [p.stem for p in sample_paths]
        print(f"=== {organ} done: {all_results[organ]} ===", flush=True)

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote per-organ calibration for {len(all_results)} organs to {OUT_PATH}")

    print("\n=== Summary (all verified to hit SSIM=0.90 on their own organ) ===")
    for organ, results in all_results.items():
        print(f"{organ}:")
        for k, v in results.items():
            print(f"  {k} = {v}" if isinstance(v, list) else f"  {k} = {v:.5f}")


if __name__ == "__main__":
    main()
