"""Floor baselines to contextualize the CONCH/UNI-embedding Ridge model's
held-out predictive validity (Table 1).

WHY THIS EXISTS (2026-09-17, adversarial-review-caught): "our model gets
r=0.4 / R^2=0.15 on held-out data" is meaningless without a floor
comparison -- is that actually using the histology image, or would a model
that ignores the image entirely (or uses only a single trivial per-patch
number) get nearly the same score? This script fits two such floor
baselines on the EXACT same IDC train/held-out split as main_experiment.py's
primary run (seed=42, test_size=0.3), so Table 1 can report them side by
side with the real model:

  1. training_mean: predict every held-out patch's expression as the TRAIN
     set's per-gene mean. No image, no metadata at all -- the absolute floor
     any real model must clear to be doing anything.
  2. total_counts_linear: predict expression from a single feature, the
     patch's own RAW (pre-normalization) total panel counts, via per-gene
     linear regression. This is exactly the confound the library-size
     normalization fix (2026-09-17, see main_experiment.py) was designed to
     remove from the modeling target. If this baseline now sits close to
     the training_mean floor, that is independent evidence the
     normalization fix worked; if it is still competitive with CONCH-Ridge,
     the confound persists even after normalizing the target.

Metric: R^2 (coefficient of determination), NOT Pearson r -- r is undefined
(zero variance) for a constant-prediction baseline like training_mean, so it
cannot be used for this comparison. R^2 handles a constant baseline
correctly (it evaluates to ~0 by construction on data drawn from the same
distribution as the training set). main_experiment.py's held-out R^2 for
CONCH/UNI-Ridge (added alongside this script, same reason) is directly
comparable to the numbers here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_experiment import DATA_ROOT, load_organ_sample, PERTURBATION_CALIBRATION_PATH, CALIBRATION_RESERVED_TAIL
from gene_program_scoring import load_hallmark_gene_sets, load_program_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORGAN = "IDC"
N_PATCHES = 3000
SEED = 42  # matches main_experiment.py's primary (Table 1) IDC run


def main():
    sample_ids = sorted(p.stem for p in (DATA_ROOT / ORGAN / "patches").glob("*.h5"))
    base, remainder = divmod(N_PATCHES, len(sample_ids))

    calibration_sample_ids = set()
    if PERTURBATION_CALIBRATION_PATH.exists():
        with open(PERTURBATION_CALIBRATION_PATH) as f:
            _cal = json.load(f).get(ORGAN, {})
        calibration_sample_ids = set(_cal.get("_calibration_sample_ids", []))
    if not calibration_sample_ids and sample_ids:
        calibration_sample_ids = {sample_ids[0]}

    expressions = []
    n_patches_per_sample = []
    for index, sample_id in enumerate(sample_ids):
        tail = CALIBRATION_RESERVED_TAIL if sample_id in calibration_sample_ids else 0
        patches, expr = load_organ_sample(ORGAN, sample_id, base + (index < remainder), seed=SEED, exclude_tail=tail)
        expressions.append(expr)
        n_patches_per_sample.append(len(patches))
        print(f"  {ORGAN}/{sample_id}: {len(patches)} patches, {expr.shape[1]} genes", flush=True)

    common_genes = set(expressions[0].columns)
    for e in expressions[1:]:
        common_genes &= set(e.columns)
    common_genes = sorted(common_genes)
    expr_df = pd.concat([e[common_genes] for e in expressions], axis=0)
    n_total = sum(n_patches_per_sample)
    print(f"[setup] {n_total} patches, {len(common_genes)} common genes", flush=True)

    # RAW total panel counts per patch, BEFORE normalization -- this is the
    # exact confound signal main_experiment.py's library-size fix removes
    # from the modeling target. Captured here so total_counts_linear can use
    # it as its one feature.
    raw_counts_total = expr_df.sum(axis=1).values

    # Same library-size normalization as main_experiment.py -- the modeling
    # target both CONCH-Ridge and these baselines are scored against must be
    # identical for the comparison to mean anything.
    library_size = expr_df.sum(axis=1)
    assert (library_size > 0).all(), "found a patch with zero total panel counts -- cannot normalize"
    median_library_size = float(library_size.median())
    expr_norm = (expr_df.div(library_size, axis=0) * median_library_size).values

    train_idx, heldout_idx = train_test_split(np.arange(n_total), test_size=0.3, random_state=SEED)
    train_expr, heldout_expr = expr_norm[train_idx], expr_norm[heldout_idx]
    train_counts, heldout_counts = raw_counts_total[train_idx], raw_counts_total[heldout_idx]
    print(f"[split] {len(train_idx)} train, {len(heldout_idx)} held-out patches (seed={SEED}), "
          f"matching main_experiment.py's primary IDC run", flush=True)

    hallmark, prog_cfg = load_hallmark_gene_sets(), load_program_config()
    hallmark_gene_union = set()
    for spec in prog_cfg["programs"].values():
        for hset in spec["hallmark_sets"]:
            hallmark_gene_union.update(hallmark.get(hset, []))
    hallmark_idx = [i for i, g in enumerate(common_genes) if g in hallmark_gene_union]
    print(f"[setup] {len(hallmark_idx)}/{len(common_genes)} genes are in the scored Hallmark programs", flush=True)

    def r2_all_and_hallmark(pred: np.ndarray) -> tuple[float, float]:
        r2_all = float(r2_score(heldout_expr, pred, multioutput="uniform_average"))
        r2_hm = float(r2_score(heldout_expr[:, hallmark_idx], pred[:, hallmark_idx],
                                multioutput="uniform_average")) if hallmark_idx else float("nan")
        return r2_all, r2_hm

    # Baseline 1: training_mean -- no image, no metadata, just the train set's per-gene average.
    train_mean = train_expr.mean(axis=0, keepdims=True)
    pred_mean = np.repeat(train_mean, len(heldout_idx), axis=0)
    r2_mean_all, r2_mean_hm = r2_all_and_hallmark(pred_mean)

    # Baseline 2: total_counts_linear -- one feature, this patch's own raw total panel counts.
    lr = LinearRegression().fit(train_counts.reshape(-1, 1), train_expr)
    pred_counts = lr.predict(heldout_counts.reshape(-1, 1))
    r2_counts_all, r2_counts_hm = r2_all_and_hallmark(pred_counts)

    print(f"\n[baseline] training_mean:        R^2(whole panel)={r2_mean_all:.4f}  R^2(Hallmark genes)={r2_mean_hm:.4f}")
    print(f"[baseline] total_counts_linear:  R^2(whole panel)={r2_counts_all:.4f}  R^2(Hallmark genes)={r2_counts_hm:.4f}")
    print("\nCompare against main_experiment_results_idc.csv's heldout_mean_r2 / "
          "heldout_mean_r2_hallmark_genes for CONCH-Ridge on this same split.")

    out = pd.DataFrame([
        {"model": "training_mean", "heldout_mean_r2": r2_mean_all, "heldout_mean_r2_hallmark_genes": r2_mean_hm},
        {"model": "total_counts_linear", "heldout_mean_r2": r2_counts_all, "heldout_mean_r2_hallmark_genes": r2_counts_hm},
    ])
    out_path = PROJECT_ROOT / "results" / "baseline_models_idc.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
