"""Measures train/held-out pixel overlap from HEST's patch-extraction grid.

WHY THIS EXISTS (2026-09-17, third-adversarial-review-caught): the paper
claimed "27% of held-out patches have at least one training patch that
shares physical pixel overlap" with no producing script -- a hand-typed
number in a section whose entire purpose is disclosing leakage exposure.
Independently reproducing the exact seed-42 patch selection and checking
each held-out patch's (x, y) grid coordinate against every same-slide
training patch's coordinate (two patches of side `patch_size` overlap iff
both |dx| < patch_size and |dy| < patch_size) gives 49.1%, not 27% -- this
script is the actual source of that number from here on.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "hest-bench"
CALIBRATION_RESERVED_TAIL = 200
SEED = 42
N_PATCHES = 3000
ORGAN = "IDC"


def main():
    st_dataset_path = PROJECT_ROOT / "external/HEST/src/hest/bench/st_dataset.py"
    spec = importlib.util.spec_from_file_location("st_dataset", st_dataset_path)
    st_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st_dataset)

    sample_ids = sorted(p.stem for p in (DATA_ROOT / ORGAN / "patches").glob("*.h5"))
    base, remainder = divmod(N_PATCHES, len(sample_ids))

    all_coords, all_slide, patch_size_of = [], [], {}
    for index, sid in enumerate(sample_ids):
        n_patches = base + (index < remainder)
        f = h5py.File(str(DATA_ROOT / ORGAN / "patches" / f"{sid}.h5"), "r")
        patch_size_of[sid] = int(f["img"].attrs["patch_size"])
        f.close()
        patch_ds = st_dataset.H5PatchDataset(str(DATA_ROOT / ORGAN / "patches" / f"{sid}.h5"), img_transform=None)
        adata = ad.read_h5ad(DATA_ROOT / ORGAN / "adata" / f"{sid}.h5ad")
        rng = np.random.RandomState(SEED)
        n_total = len(patch_ds)
        n_available = max(0, n_total - CALIBRATION_RESERVED_TAIL)
        chosen = rng.choice(n_available, size=min(n_patches, n_available), replace=False)
        chosen.sort()
        coords, barcodes = [], []
        for i in chosen:
            item = patch_ds[int(i)]
            coords.append(item["coords"])
            barcodes.append(item["barcodes"])
        valid = [b in adata.obs_names for b in barcodes]
        coords = [c for c, keep in zip(coords, valid) if keep]
        all_coords.extend(coords)
        all_slide.extend([sid] * len(coords))
        print(f"  {ORGAN}/{sid}: {len(coords)} patches, patch_size={patch_size_of[sid]}px", flush=True)

    all_coords = np.array(all_coords)
    all_slide = np.array(all_slide)
    n = len(all_coords)
    train_idx, heldout_idx = train_test_split(np.arange(n), test_size=0.3, random_state=SEED)

    rows = []
    any_overlap_total = rook_overlap_total = 0
    for sid in sample_ids:
        ps = patch_size_of[sid]
        heldout_this_slide = heldout_idx[all_slide[heldout_idx] == sid]
        train_this_slide = train_idx[all_slide[train_idx] == sid]
        any_overlap = rook_overlap = 0
        for hi in heldout_this_slide:
            dx = np.abs(all_coords[train_this_slide, 0] - all_coords[hi, 0])
            dy = np.abs(all_coords[train_this_slide, 1] - all_coords[hi, 1])
            overlaps = (dx < ps) & (dy < ps)
            if overlaps.any():
                any_overlap += 1
                # rook/edge-adjacent: overlap along exactly one axis (not a diagonal-only touch)
                if ((dx[overlaps] == 0) | (dy[overlaps] == 0)).any():
                    rook_overlap += 1
        rows.append({"slide": sid, "n_heldout": len(heldout_this_slide),
                     "any_overlap_frac": any_overlap / len(heldout_this_slide) if len(heldout_this_slide) else float("nan"),
                     "rook_overlap_frac": rook_overlap / len(heldout_this_slide) if len(heldout_this_slide) else float("nan")})
        any_overlap_total += any_overlap
        rook_overlap_total += rook_overlap

    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"slide": "ALL", "n_heldout": len(heldout_idx),
                        "any_overlap_frac": any_overlap_total / len(heldout_idx),
                        "rook_overlap_frac": rook_overlap_total / len(heldout_idx)}
    out = PROJECT_ROOT / "results" / "pixel_overlap_check.csv"
    df.to_csv(out, index=False)
    print(f"\n{df.to_string(index=False)}\n\nWrote {out}")


if __name__ == "__main__":
    main()
