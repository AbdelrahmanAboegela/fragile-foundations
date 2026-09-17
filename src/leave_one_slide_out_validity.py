"""Spatially-disjoint (leave-one-slide-out) predictive validity check.

WHY THIS EXISTS (2026-09-17, second-adversarial-review-caught): the paper's
primary validity number uses a RANDOM PATCH split (main_experiment.py), and
the paper called this "leakage-free". An independent review directly
inspected the HEST patch HDF5 files and found that, on every IDC sample, the
patch-extraction grid pitch is SMALLER than the patch size itself
(e.g. NCBI783: 365px grid pitch vs 409px patch size in source-image pixels),
so adjacent patches share real image pixels -- confirmed empirically here,
not just from the attrs: two NCBI783 patches one grid-step apart have
corr=0.9935 between their shared border rows. With a random patch split,
27.1% of held-out patches (seed=42) have at least one pixel-overlapping
neighbor in the training set. Calling the resulting r/R^2 "leakage-free" is
false; the number is real but has an unknown, likely non-trivial optimistic
bias from both direct pixel overlap AND ordinary spatial autocorrelation in
Xenium expression.

This script reports the number that HEST-Bench's own protocol calls for and
that the paper should have led with: predictive validity when the held-out
set is an ENTIRE SLIDE the model never saw ANY patch from during training,
for every one of IDC's 4 slides (4-fold leave-one-slide-out). This is
immune to both pixel-overlap leakage and patch-level spatial
autocorrelation by construction, since train and held-out patches come from
physically different tissue sections.

Uses the same library-size normalization, same alpha_per_target RidgeCV,
and the same Hallmark-restricted-gene reporting as main_experiment.py, for
direct comparability. The one methodological choice worth flagging
explicitly: the per-patch normalization's target (the cohort median total
count) is computed over ALL pooled patches (train+held-out combined), same
as main_experiment.py -- this is a scale-normalization constant, not label
information, and keeping it identical to the primary pipeline is what makes
the two validity numbers comparable at all. It is not gene-expression
signal, but is disclosed here rather than silently assumed innocuous.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_experiment import (
    DATA_ROOT, ENCODERS, embed_patches, load_organ_sample,
    PERTURBATION_CALIBRATION_PATH, CALIBRATION_RESERVED_TAIL,
)
from gene_program_scoring import load_hallmark_gene_sets, load_program_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORGAN = "IDC"
N_PATCHES = 3000
SEED = 42


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=list(ENCODERS.keys()), default="conch")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}, encoder={args.encoder}", flush=True)

    sample_ids = sorted(p.stem for p in (DATA_ROOT / ORGAN / "patches").glob("*.h5"))
    base, remainder = divmod(N_PATCHES, len(sample_ids))

    calibration_sample_ids = set()
    if PERTURBATION_CALIBRATION_PATH.exists():
        with open(PERTURBATION_CALIBRATION_PATH) as f:
            _cal = json.load(f).get(ORGAN, {})
        calibration_sample_ids = set(_cal.get("_calibration_sample_ids", []))
    if not calibration_sample_ids and sample_ids:
        calibration_sample_ids = {sample_ids[0]}

    all_patches, expressions, all_slide_ids = [], [], []
    for index, sample_id in enumerate(sample_ids):
        tail = CALIBRATION_RESERVED_TAIL if sample_id in calibration_sample_ids else 0
        patches, expr = load_organ_sample(ORGAN, sample_id, base + (index < remainder), seed=SEED, exclude_tail=tail)
        all_patches.extend(patches)
        all_slide_ids.extend([sample_id] * len(patches))
        expressions.append(expr)
        print(f"  {ORGAN}/{sample_id}: {len(patches)} patches, {expr.shape[1]} genes", flush=True)

    common_genes = set(expressions[0].columns)
    for e in expressions[1:]:
        common_genes &= set(e.columns)
    common_genes = sorted(common_genes)
    expr_df = pd.concat([e[common_genes] for e in expressions], axis=0)
    slide_ids = np.array(all_slide_ids)
    n_total = len(all_patches)
    print(f"[setup] {n_total} patches, {len(common_genes)} common genes, "
          f"{len(sample_ids)} slides: {dict(zip(*np.unique(slide_ids, return_counts=True)))}", flush=True)

    # Same library-size normalization as main_experiment.py, computed over
    # the full pooled set -- see module docstring for why this specific
    # choice keeps this check comparable to the primary result.
    library_size = expr_df.sum(axis=1)
    assert (library_size > 0).all(), "found a patch with zero total panel counts -- cannot normalize"
    median_library_size = float(library_size.median())
    expr_norm = (expr_df.div(library_size, axis=0) * median_library_size).values
    print(f"[normalize] median total panel count={median_library_size:.0f}", flush=True)

    hallmark, prog_cfg = load_hallmark_gene_sets(), load_program_config()
    hallmark_gene_union = set()
    for spec in prog_cfg["programs"].values():
        for hset in spec["hallmark_sets"]:
            hallmark_gene_union.update(hallmark.get(hset, []))
    hallmark_idx = [i for i, g in enumerate(common_genes) if g in hallmark_gene_union]

    print(f"[setup] loading {args.encoder} encoder...", flush=True)
    encoder, preprocess = ENCODERS[args.encoder]["loader"](device)
    forward_fn = ENCODERS[args.encoder]["forward"]
    embeddings = embed_patches(encoder, preprocess, all_patches, device, forward_fn=forward_fn)

    def r2_all_and_hallmark(true, pred):
        # MEAN vs MEDIAN per-gene R^2 (2026-09-17, third-adversarial-review-
        # caught): r2_score(..., multioutput="uniform_average") is the MEAN
        # of 280 per-gene R^2 values. A single gene with a large between-
        # slide mean offset relative to its within-slide variance can post an
        # arbitrarily large negative R^2 (unlike Pearson r, R^2 is unbounded
        # below) and dominate that mean -- confirmed directly: on the CONCH/
        # NCBI785 fold, whole-panel mean R^2=-68.9 while the median per-gene
        # R^2 is far less extreme. Reporting the median alongside the mean
        # (and per-gene raw values, so this can be re-checked) prevents a few
        # outlier genes from being read as "the model's absolute calibration
        # in general," which is what the mean alone would misleadingly imply.
        raw_all = r2_score(true, pred, multioutput="raw_values")
        r2_all = float(np.mean(raw_all))
        r2_all_median = float(np.median(raw_all))
        if hallmark_idx:
            raw_hm = r2_score(true[:, hallmark_idx], pred[:, hallmark_idx], multioutput="raw_values")
            r2_hm = float(np.mean(raw_hm))
            r2_hm_median = float(np.median(raw_hm))
        else:
            r2_hm = r2_hm_median = float("nan")
        corrs, corrs_hm = [], []
        for gi in range(true.shape[1]):
            if true[:, gi].std() > 1e-8 and pred[:, gi].std() > 1e-8:
                r, _ = pearsonr(true[:, gi], pred[:, gi])
                if not np.isnan(r):
                    corrs.append(r)
                    if gi in hallmark_idx:
                        corrs_hm.append(r)
        mean_r = float(np.mean(corrs)) if corrs else float("nan")
        mean_r_hm = float(np.mean(corrs_hm)) if corrs_hm else float("nan")
        return r2_all, r2_all_median, r2_hm, r2_hm_median, mean_r, mean_r_hm

    rows = []
    for held_out_slide in sample_ids:
        fold_start = time.time()
        train_mask = slide_ids != held_out_slide
        heldout_mask = slide_ids == held_out_slide
        train_emb, heldout_emb = embeddings[train_mask], embeddings[heldout_mask]
        train_expr, heldout_expr = expr_norm[train_mask], expr_norm[heldout_mask]

        alphas = np.logspace(-2, 4, 13)
        head = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True).fit(train_emb, train_expr)
        pred = head.predict(heldout_emb)
        r2_all, r2_all_median, r2_hm, r2_hm_median, mean_r, mean_r_hm = r2_all_and_hallmark(heldout_expr, pred)

        print(f"[fold] held-out slide={held_out_slide} (n={heldout_mask.sum()}, "
              f"train n={train_mask.sum()}): r={mean_r:.4f} (Hallmark {mean_r_hm:.4f}), "
              f"R^2 mean={r2_all:.4f} median={r2_all_median:.4f} (Hallmark mean={r2_hm:.4f} median={r2_hm_median:.4f}), "
              f"elapsed={time.time()-fold_start:.1f}s", flush=True)
        rows.append({"encoder": args.encoder, "held_out_slide": held_out_slide,
                     "n_train": int(train_mask.sum()), "n_heldout": int(heldout_mask.sum()),
                     "pearson_r": mean_r, "pearson_r_hallmark": mean_r_hm,
                     "r2": r2_all, "r2_median": r2_all_median,
                     "r2_hallmark": r2_hm, "r2_hallmark_median": r2_hm_median})

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / f"leave_one_slide_out_validity_{args.encoder}.csv"
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")
    print(f"\nMean across 4 slide-held-out folds: r={df.pearson_r.mean():.4f} (range {df.pearson_r.min():.4f}-{df.pearson_r.max():.4f}), "
          f"r_hallmark={df.pearson_r_hallmark.mean():.4f} (range {df.pearson_r_hallmark.min():.4f}-{df.pearson_r_hallmark.max():.4f}), "
          f"R^2 mean={df.r2.mean():.4f} median-of-per-fold-medians={df.r2_median.median():.4f} (range {df.r2.min():.4f}-{df.r2.max():.4f}), "
          f"R^2_hallmark mean={df.r2_hallmark.mean():.4f} median-of-per-fold-medians={df.r2_hallmark_median.median():.4f} "
          f"(range {df.r2_hallmark.min():.4f}-{df.r2_hallmark.max():.4f})")


if __name__ == "__main__":
    main()
