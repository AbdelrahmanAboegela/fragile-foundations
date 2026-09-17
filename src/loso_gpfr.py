"""Leave-one-slide-out GPFR: the perturbation-robustness analysis itself,
evaluated under the same slide-disjoint design used for LOSO predictive
validity (leave_one_slide_out_validity.py), instead of the random 70/30
patch split main_experiment.py uses for the primary GPFR result.

WHY THIS EXISTS (2026-09-17, reviewer-caught): the paper's own random-
split leakage investigation (Section 2.3) shows a random patch split lets
held-out patches share physical pixels with training patches. The paper's
headline GPFR result nonetheless uses Ridge heads fit on that same random
split. A reviewer can reasonably ask why the perturbation-robustness
analysis wasn't repeated under the leakage-free, slide-disjoint design
the paper itself established as more trustworthy for predictive validity.
This script closes that gap: for each of an organ's 4 slides in turn, fit
the Ridge head on the other 3 slides only, predict clean AND perturbed
scores for the held-out slide's own patches from that head, and compute
GPFR exactly as main_experiment.py does -- but with every prediction
(clean and perturbed) coming from a slide that was not used to fit the
head making that prediction.

Pooling all 4 folds gives one clean/perturbed score pair per patch in the
whole cohort (each patch appears in exactly one fold's held-out set), so
this uses the full ~3000-patch cohort (not a 30% held-out subset) while
every prediction is still genuinely out-of-slide. Output is written in
the same raw_score_rows schema as main_experiment.py's
*_raw_scores.csv.gz, so analyze_statistics.py's existing cluster
bootstrap can be reused unchanged.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main_experiment import (
    DATA_ROOT, ENCODERS, embed_patches, load_organ_sample, build_perturbation_set_for_organ,
    PERTURBATION_CALIBRATION_PATH, CALIBRATION_RESERVED_TAIL,
)
from perturbations import PERTURBATIONS
from metrics import gene_program_flip_rate
from gene_program_scoring import load_hallmark_gene_sets, load_program_config, score_programs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_PATCHES = 3000
SEED = 42


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=list(ENCODERS.keys()), default="conch")
    parser.add_argument("--organ", default="IDC")
    args = parser.parse_args()
    ORGAN = args.organ

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}, encoder={args.encoder}, organ={ORGAN}", flush=True)

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
    print(f"[setup] {n_total} patches, {len(common_genes)} common genes, {len(sample_ids)} slides", flush=True)

    library_size = expr_df.sum(axis=1)
    assert (library_size > 0).all(), "found a patch with zero total panel counts -- cannot normalize"
    median_library_size = float(library_size.median())
    expr_norm = (expr_df.div(library_size, axis=0) * median_library_size).values
    print(f"[normalize] median total panel count={median_library_size:.0f}", flush=True)

    hallmark, prog_cfg = load_hallmark_gene_sets(), load_program_config()

    print(f"[setup] loading {args.encoder} encoder...", flush=True)
    encoder, preprocess = ENCODERS[args.encoder]["loader"](device)
    forward_fn = ENCODERS[args.encoder]["forward"]
    embeddings = embed_patches(encoder, preprocess, all_patches, device, forward_fn=forward_fn)

    organ_perturbation_set = build_perturbation_set_for_organ(ORGAN, PERTURBATIONS)

    raw_score_rows = []
    for held_out_slide in sample_ids:
        fold_start = time.time()
        train_mask = slide_ids != held_out_slide
        heldout_mask = slide_ids == held_out_slide
        train_emb_raw, heldout_emb_raw = embeddings[train_mask], embeddings[heldout_mask]
        train_expr, heldout_expr = expr_norm[train_mask], expr_norm[heldout_mask]
        heldout_patches = [p for p, keep in zip(all_patches, heldout_mask) if keep]

        fold_scaler = StandardScaler().fit(train_emb_raw)
        train_emb = fold_scaler.transform(train_emb_raw)
        heldout_emb = fold_scaler.transform(heldout_emb_raw)

        alphas = np.logspace(-2, 6, 17)
        head = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True).fit(train_emb, train_expr)
        clean_pred = head.predict(heldout_emb)
        clean_pred_scores = score_programs(pd.DataFrame(clean_pred, columns=common_genes), hallmark, prog_cfg)
        print(f"[fold] held_out={held_out_slide}: n_train={train_mask.sum()}, n_heldout={heldout_mask.sum()}, "
              f"setup elapsed={time.time()-fold_start:.1f}s", flush=True)

        for p_index, (pname, pfunc) in enumerate(organ_perturbation_set.items(), start=1):
            pert_start = time.time()
            perturbed_imgs = [pfunc(img) for img in heldout_patches]
            pert_emb_raw = embed_patches(encoder, preprocess, perturbed_imgs, device, forward_fn=forward_fn)
            pert_emb = fold_scaler.transform(pert_emb_raw)
            pert_pred = head.predict(pert_emb)
            pert_scores = score_programs(pd.DataFrame(pert_pred, columns=common_genes), hallmark, prog_cfg)

            for program in clean_pred_scores.columns:
                for sample_i in range(len(clean_pred_scores)):
                    raw_score_rows.append({
                        "encoder": args.encoder, "organ": ORGAN, "perturbation": pname, "program": program,
                        "sample_index": sample_i, "slide_id": held_out_slide,
                        "clean_score": float(clean_pred_scores[program].values[sample_i]),
                        "pert_score": float(pert_scores[program].values[sample_i]),
                    })
            print(f"  [fold {held_out_slide}] perturbation {pname} ({p_index}/{len(organ_perturbation_set)}) "
                  f"done, elapsed={time.time()-pert_start:.1f}s", flush=True)

        print(f"[fold] held-out slide={held_out_slide} complete, elapsed={time.time()-fold_start:.1f}s", flush=True)

    raw_df = pd.DataFrame(raw_score_rows)
    base_name = f"loso_gpfr_{ORGAN.lower()}" + ("" if args.encoder == "conch" else f"_{args.encoder}")
    raw_out = PROJECT_ROOT / "results" / f"{base_name}_raw_scores.csv.gz"
    raw_df.to_csv(raw_out, index=False, compression="gzip")
    print(f"\nWrote {len(raw_df)} raw per-sample rows to {raw_out}", flush=True)

    # Quick point-estimate summary (pooling all 4 folds); the full-rigor
    # bootstrap+FDR version is produced by analyze_statistics.py from the
    # raw-scores file above, exactly as for the random-split GPFR result.
    summary_rows = []
    for (pert, prog), g in raw_df.groupby(["perturbation", "program"]):
        gpfr = gene_program_flip_rate(g["clean_score"].values, g["pert_score"].values)
        summary_rows.append({"encoder": args.encoder, "organ": ORGAN, "perturbation": pert, "program": prog, "gpfr": gpfr})
    summary_df = pd.DataFrame(summary_rows).sort_values("gpfr", ascending=False)
    summary_out = PROJECT_ROOT / "results" / f"{base_name}.csv"
    summary_df.to_csv(summary_out, index=False)
    print(f"Wrote {len(summary_df)} rows to {summary_out}", flush=True)
    print(summary_df.head(10).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
