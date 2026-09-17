"""Direct, script-backed measurement of the library-size/cellularity confound.

WHY THIS EXISTS: an early note describing this project quoted "a patch's
predicted proliferation score correlated with its own raw total panel
count at r=0.82" before the library-size normalization fix -- but no
script or CSV ever computed this number directly. That is exactly the
kind of hand-transcribed, unverifiable number the project's own
Data-and-Code-Availability section promises never appears. This script
fits the SAME Ridge head twice on the SAME train/held-out split (seed 42,
IDC, CONCH) -- once on raw (unnormalized) target counts, once on the
library-size-normalized target -- and reports, for both, the Pearson
correlation between each held-out patch's predicted `proliferation` program
score and that patch's own raw total panel count. This directly answers
"does the fix actually reduce this specific confound," with a number that
traces to this file and to results/confound_check.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_experiment import (
    DATA_ROOT, ENCODERS, embed_patches, load_organ_sample,
    PERTURBATION_CALIBRATION_PATH, CALIBRATION_RESERVED_TAIL,
)
from gene_program_scoring import load_hallmark_gene_sets, load_program_config, score_programs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORGAN = "IDC"
N_PATCHES = 3000
SEED = 42


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_ids = sorted(p.stem for p in (DATA_ROOT / ORGAN / "patches").glob("*.h5"))
    base, remainder = divmod(N_PATCHES, len(sample_ids))
    calibration_sample_ids = set()
    if PERTURBATION_CALIBRATION_PATH.exists():
        with open(PERTURBATION_CALIBRATION_PATH) as f:
            _cal = json.load(f).get(ORGAN, {})
        calibration_sample_ids = set(_cal.get("_calibration_sample_ids", []))
    if not calibration_sample_ids and sample_ids:
        calibration_sample_ids = {sample_ids[0]}

    all_patches, expressions = [], []
    for index, sample_id in enumerate(sample_ids):
        tail = CALIBRATION_RESERVED_TAIL if sample_id in calibration_sample_ids else 0
        patches, expr = load_organ_sample(ORGAN, sample_id, base + (index < remainder), seed=SEED, exclude_tail=tail)
        all_patches.extend(patches)
        expressions.append(expr)
        print(f"  {ORGAN}/{sample_id}: {len(patches)} patches, {expr.shape[1]} genes", flush=True)

    common_genes = set(expressions[0].columns)
    for e in expressions[1:]:
        common_genes &= set(e.columns)
    common_genes = sorted(common_genes)
    expr_df = pd.concat([e[common_genes] for e in expressions], axis=0)
    raw_counts_total = expr_df.sum(axis=1).values
    print(f"[setup] {len(all_patches)} patches, {len(common_genes)} genes, "
          f"total-count range [{raw_counts_total.min():.0f}, {raw_counts_total.max():.0f}], "
          f"median={np.median(raw_counts_total):.0f}", flush=True)

    library_size = expr_df.sum(axis=1)
    median_library_size = float(library_size.median())
    expr_norm = (expr_df.div(library_size, axis=0) * median_library_size).values
    expr_raw = expr_df.values

    n = len(all_patches)
    train_idx, heldout_idx = train_test_split(np.arange(n), test_size=0.3, random_state=SEED)
    heldout_counts = raw_counts_total[heldout_idx]

    print("[setup] loading conch encoder...", flush=True)
    encoder, preprocess = ENCODERS["conch"]["loader"](device)
    embeddings = embed_patches(encoder, preprocess, all_patches, device, forward_fn=ENCODERS["conch"]["forward"])
    train_emb, heldout_emb = embeddings[train_idx], embeddings[heldout_idx]

    hallmark, prog_cfg = load_hallmark_gene_sets(), load_program_config()
    alphas = np.logspace(-2, 4, 13)
    rows = []
    for label, target in [("raw_unnormalized", expr_raw), ("library_size_normalized", expr_norm)]:
        head = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True).fit(train_emb, target[train_idx])
        pred = pd.DataFrame(head.predict(heldout_emb), columns=common_genes)
        scores = score_programs(pred, hallmark, prog_cfg)
        r, p = pearsonr(heldout_counts, scores["proliferation"].values)
        print(f"[confound] {label}: corr(raw total count, predicted proliferation score) = r={r:.4f} (p={p:.2e})", flush=True)
        rows.append({"target": label, "pearson_r_totalcount_vs_proliferation": r, "p_value": p})

    out = PROJECT_ROOT / "results" / "confound_check.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
