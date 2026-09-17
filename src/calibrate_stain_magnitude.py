"""Measures (does not target-search) the SSIM cost of the two literature-
grounded stain_shift magnitudes hardcoded in perturbations.py.

STATUS (2026-09-16): this used to root-find (scipy.optimize.brentq) a
scaling strength s to hit a target SSIM, the same way the other six
perturbations in PERTURBATIONS are calibrated. That approach is now known
to be wrong for stain_shift specifically: after an audit found and fixed a
real bug in _macenko_stain_vectors (missing mean-centering, producing a
numerically degenerate stain-vector basis), re-running the SSIM-target
search showed SSIM=0.85 is UNREACHABLE within any physically plausible
concentration-scaling range with the corrected vectors — the bracket
(0.0005, 0.05) that used to bracket SSIM=0.85 now covers only
SSIM in [0.985, 0.985], and SSIM=0.85 only becomes reachable at s~=0.80
(h_scale=1.80, e_scale=0.20), a physically extreme stain-channel imbalance
that isn't a realistic simulation of inter-lab stain variation. SSIM is
dominated by structural similarity, not color, so it under-detects color
problems — chasing an SSIM target here would only reward an unrealistic
color shift for hitting a number, defeating the "morphology-preserving"
premise of the whole perturbation suite.

So this script no longer searches for anything. It measures mean SSIM at
the two magnitudes now hardcoded in perturbations.py (s=0.10 "mild",
s=0.30 "moderate" — chosen to bracket the HED-light/HED-strong precedent
in Tellez et al. 2019, arXiv:1902.06543
perturbations.py's module docstring) and prints the result as an
observation. If you change those hardcoded magnitudes, re-run this to get
the new measured SSIM and update the docstrings that cite it — but do NOT
use this script's output to pick the magnitude itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "external/HEST/src"))

from perturbations import macenko_stain_shift, macenko_stain_shift_moderate, macenko_stain_shift_identity
from metrics import compute_ssim

N_CALIBRATION_PATCHES = 40
MIN_FOREGROUND_FRACTION = 0.5  # consistent with calibrate_all_perturbations.py — see its docstring


def foreground_fraction(img: Image.Image, beta: float = 0.15) -> float:
    arr = np.asarray(img).astype(np.float64)
    od = -np.log((arr.reshape(-1, 3) + 1) / 256.0)
    return float((od > beta).any(axis=1).mean())


def load_calibration_patches() -> list[Image.Image]:
    import importlib.util
    st_dataset_path = Path(__file__).resolve().parent.parent / "external/HEST/src/hest/bench/st_dataset.py"
    spec = importlib.util.spec_from_file_location("st_dataset", st_dataset_path)
    st_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st_dataset)
    patch_path = Path(__file__).resolve().parent.parent / "data/hest-bench/IDC/patches/NCBI783.h5"
    ds = st_dataset.H5PatchDataset(str(patch_path), img_transform=None)

    patches = []
    i = 0
    while len(patches) < N_CALIBRATION_PATCHES and i < len(ds):
        img = Image.fromarray(ds[i]["imgs"].astype(np.uint8))
        if foreground_fraction(img) >= MIN_FOREGROUND_FRACTION:
            patches.append(img)
        i += 1
    print(f"[setup] scanned {i} patches to find {len(patches)} tissue-dominant calibration patches", flush=True)
    return patches


def mean_ssim(patches: list[Image.Image], fn) -> float:
    ssims = []
    for img in patches:
        pert = fn(img)
        c = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        p = torch.from_numpy(np.asarray(pert).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        ssims.append(compute_ssim(c, p))
    return float(np.mean(ssims))


def main():
    print(f"[setup] loading {N_CALIBRATION_PATCHES} calibration patches...", flush=True)
    patches = load_calibration_patches()

    print("\n[measure] stain_shift_identity (h=1.0/e=1.0, null control)...", flush=True)
    ssim_identity = mean_ssim(patches, macenko_stain_shift_identity)
    print(f"  -> measured mean SSIM vs clean={ssim_identity:.4f}  "
          f"(this is the reconstruction-round-trip noise floor, NOT a stain effect — "
          f"see macenko_stain_shift_identity's docstring)", flush=True)

    print("\n[measure] stain_shift_mild (h=1.10/e=0.90, s=0.10)...", flush=True)
    ssim_mild = mean_ssim(patches, macenko_stain_shift)
    print(f"  -> measured mean SSIM vs clean={ssim_mild:.4f}", flush=True)

    print("\n[measure] stain_shift_moderate (h=1.30/e=0.70, s=0.30)...", flush=True)
    ssim_moderate = mean_ssim(patches, macenko_stain_shift_moderate)
    print(f"  -> measured mean SSIM vs clean={ssim_moderate:.4f}", flush=True)

    noise_floor_loss = 1.0 - ssim_identity
    mild_loss = 1.0 - ssim_mild
    moderate_loss = 1.0 - ssim_moderate
    print(f"\n[interpretation] fraction of SSIM loss (vs clean) that is already present "
          f"in the null control, i.e. reconstruction noise rather than real stain scaling:")
    print(f"  mild:     {noise_floor_loss / mild_loss:.1%}" if mild_loss > 0 else "  mild: n/a")
    print(f"  moderate: {noise_floor_loss / moderate_loss:.1%}" if moderate_loss > 0 else "  moderate: n/a")

    print("\nThese are OBSERVATIONS, not design targets. If they drift materially "
          "from the values cited in perturbations.py's docstrings after any future "
          "code change, update those docstrings to match — don't adjust h_scale/e_scale "
          "to chase a particular SSIM.")


if __name__ == "__main__":
    main()
