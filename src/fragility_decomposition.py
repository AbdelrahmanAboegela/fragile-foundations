"""Decompose GPFR into stages and test a cheap linear-offset mitigation.

Research-recommended addition (2026-09-15/16): rather than just reporting
"GPFR is high for JPEG", localize WHERE the fragility enters the pipeline:

  clean image -> [CONCH encoder] -> embedding -> [Ridge head] -> predicted
  expression -> [Hallmark aggregation] -> program score -> [threshold] -> GPFR

Stage 1 (embedding drift): does the perturbation move the CONCH embedding a
lot or a little, per patch?
Stage 2 (prediction drift): does that embedding shift translate into a big
change in predicted expression?
Stage 3 (program drift / GPFR): does that translate into an actual program
flip?

Then test the linear-offset hypothesis (motivated by prior work showing
scanner-shift is near-linear in pathology FM feature space): if perturbed
embeddings are approximately clean_embedding + a single shared offset vector,
then subtracting an offset estimated on held-out patches should collapse
GPFR on the rest.

IMPORTANT FRAMING CORRECTION (audit-flagged, 2026-09-16): this is a
diagnostic of whether the perturbation's effect on the embedding is
linear/shift-like, NOT a deployable production fix, despite earlier
language in this project describing it that way. It requires (a) a paired
clean image for every perturbed one, and (b) knowing WHICH perturbation
type was applied to select the right offset vector — neither is available
at real deployment time, where you only ever see one (possibly already
perturbed, unknown-how) image. What it legitimately supports: "the fragility
this perturbation causes is [or isn't] a simple additive shift in feature
space" — itself a useful mechanistic finding, just not an actionable fix
on its own.

Uses only CONCH + IDC (already fully available) — no dependency on UNI
access or the other 8 organs' downloads.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
socket.setdefaulttimeout(30)

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from main_experiment import (
    ENCODERS, embed_patches, load_organ_sample, DATA_ROOT, CALIBRATION_RESERVED_TAIL,
    PERTURBATION_CALIBRATION_PATH, build_perturbation_set_for_organ,
)
from perturbations import PERTURBATIONS
from gene_program_scoring import load_hallmark_gene_sets, load_program_config, score_programs
from metrics import gene_program_flip_rate

N_PATCHES = 3000  # match the main IDC run for a directly comparable result
ORGAN = "IDC"
N_FOLDS = 5  # for held-out offset estimation


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}", flush=True)

    sample_ids = sorted(p.stem for p in (DATA_ROOT / ORGAN / "patches").glob("*.h5"))
    base, remainder = divmod(N_PATCHES, len(sample_ids))
    all_patches, expressions = [], []
    # Same calibration/experiment disjointness as main_experiment.py — read
    # the actual sample_ids calibration touched (2026-09-16: IDC has only 4
    # samples, all <= MAX_CALIBRATION_SAMPLES, so calibration now pools
    # patches from ALL 4, not just NCBI783 as an earlier version assumed).
    calibration_sample_ids = set()
    if PERTURBATION_CALIBRATION_PATH.exists():
        with open(PERTURBATION_CALIBRATION_PATH) as f:
            _cal = json.load(f).get(ORGAN, {})
        calibration_sample_ids = set(_cal.get("_calibration_sample_ids", []))
    if not calibration_sample_ids and sample_ids:
        calibration_sample_ids = {sample_ids[0]}  # legacy fallback
    for index, sample_id in enumerate(sample_ids):
        tail = CALIBRATION_RESERVED_TAIL if sample_id in calibration_sample_ids else 0
        patches, expr = load_organ_sample(ORGAN, sample_id, base + (index < remainder), exclude_tail=tail)
        all_patches.extend(patches)
        expressions.append(expr)
        print(f"  {ORGAN}/{sample_id}: {len(patches)} patches, {expr.shape[1]} genes", flush=True)

    common_genes = set(expressions[0].columns)
    for e in expressions[1:]:
        common_genes &= set(e.columns)
    common_genes = sorted(common_genes)
    expr_df = pd.concat([e[common_genes] for e in expressions], axis=0)
    print(f"[setup] {len(all_patches)} patches, {len(common_genes)} common genes", flush=True)

    # LIBRARY-SIZE NORMALIZATION (2026-09-17) — same fix and rationale as
    # main_experiment.py; see that file's comment for the full story
    # (adversarial-review-caught, independently confirmed: proliferation's
    # program score correlated with raw total counts at r=0.82).
    library_size = expr_df.sum(axis=1)
    assert (library_size > 0).all(), "found a patch with zero total panel counts — cannot normalize"
    median_library_size = float(library_size.median())
    print(f"[normalize] per-patch total panel counts range [{library_size.min():.0f}, {library_size.max():.0f}], "
          f"median={median_library_size:.0f} — rescaling every patch's counts to this median total", flush=True)
    expr_df = expr_df.div(library_size, axis=0) * median_library_size

    hallmark, prog_cfg = load_hallmark_gene_sets(), load_program_config()
    print("[setup] loading CONCH encoder...", flush=True)
    encoder, preprocess = ENCODERS["conch"]["loader"](device)
    forward_fn = ENCODERS["conch"]["forward"]

    print("[setup] embedding clean patches...", flush=True)
    clean_emb_all = embed_patches(encoder, preprocess, all_patches, device, forward_fn=forward_fn)

    # DATA-LEAKAGE FIX (2026-09-16, audit-flagged): same issue as
    # main_experiment.py — the head was previously fit AND evaluated on
    # identical data. Split first, cross-validate alpha on TRAIN only,
    # report held-out predictive validity, and run the whole GPFR +
    # fragility-decomposition + linear-offset analysis on held-out patches only.
    n = len(all_patches)
    train_idx, heldout_idx = train_test_split(np.arange(n), test_size=0.3, random_state=42)
    all_patches = [all_patches[i] for i in heldout_idx]  # everything below now refers to held-out only
    clean_emb = clean_emb_all[heldout_idx]
    train_emb = clean_emb_all[train_idx]
    train_expr = expr_df.values[train_idx]
    heldout_expr = expr_df.values[heldout_idx]
    print(f"[split] {len(train_idx)} train, {len(heldout_idx)} held-out patches (seed=42)", flush=True)

    # PER-TARGET ALPHA FIX (2026-09-16, see main_experiment.py's matching
    # comment for the full rationale — applied here too for methodological
    # consistency, even though IDC's 280-gene panel makes the effect much
    # smaller than on whole-transcriptome organs).
    alphas = np.logspace(-2, 4, 13)
    head = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True).fit(train_emb, train_expr)
    alpha_arr = np.atleast_1d(head.alpha_)
    print(f"[validity] per-target alpha selected via efficient LOO-CV on TRAIN: "
          f"median={np.median(alpha_arr):.4g}, min={alpha_arr.min():.4g}, max={alpha_arr.max():.4g}", flush=True)

    clean_pred = pd.DataFrame(head.predict(clean_emb), columns=expr_df.columns)
    gene_corrs = [pearsonr(heldout_expr[:, gi], clean_pred.values[:, gi])[0]
                  for gi in range(heldout_expr.shape[1])
                  if heldout_expr[:, gi].std() > 1e-8 and clean_pred.values[:, gi].std() > 1e-8]
    gene_corrs = [r for r in gene_corrs if not np.isnan(r)]
    mean_r = float(np.mean(gene_corrs)) if gene_corrs else float("nan")
    print(f"[validity] held-out mean Pearson r across {len(gene_corrs)} genes: {mean_r:.4f}", flush=True)

    clean_scores = score_programs(clean_pred, hallmark, prog_cfg)

    # CALIBRATION FIX (2026-09-17, adversarial-review-caught): this script
    # used to iterate PERTURBATIONS directly, importing the module-level
    # defaults in perturbations.py rather than reading the per-organ JSON
    # via build_perturbation_set_for_organ (the function main_experiment.py
    # actually uses). The two only matched by coincidence when the module
    # defaults happened to be hand-kept in sync with IDC's JSON entry — a
    # re-run after the multi-sample calibration fix but before that manual
    # sync silently used STALE, pre-fix magnitudes with no error or warning.
    # Fixed properly now: always read the current JSON, so this can never
    # drift out of sync with main_experiment.py again.
    organ_perturbation_set = build_perturbation_set_for_organ(ORGAN, PERTURBATIONS)

    rows = []
    for pname, pfunc in organ_perturbation_set.items():
        p_start = time.time()
        print(f"\n[perturbation] {pname}", flush=True)
        perturbed_patches = [pfunc(img) for img in all_patches]  # all_patches is held-out-only now
        pert_emb = embed_patches(encoder, preprocess, perturbed_patches, device, forward_fn=forward_fn)

        # Stage 1: embedding drift per patch (L2 and cosine)
        emb_l2 = np.linalg.norm(pert_emb - clean_emb, axis=1)
        emb_cos = np.sum(clean_emb * pert_emb, axis=1) / (
            np.linalg.norm(clean_emb, axis=1) * np.linalg.norm(pert_emb, axis=1) + 1e-8)

        # Stage 2: predicted-expression drift per patch
        pert_pred = pd.DataFrame(head.predict(pert_emb), columns=expr_df.columns)
        expr_drift = np.linalg.norm(pert_pred.values - clean_pred.values, axis=1)

        # Stage 3: program-level GPFR (baseline, uncorrected)
        pert_scores = score_programs(pert_pred, hallmark, prog_cfg)
        gpfr_by_program = {}
        for prog in clean_scores.columns:
            gpfr_by_program[prog] = gene_program_flip_rate(
                clean_scores[prog].values, pert_scores[prog].values, method="per_sample_relative")

        # Correlation: does more embedding drift predict more expression drift?
        emb_to_expr_corr = float(np.corrcoef(emb_l2, expr_drift)[0, 1])

        # Linear-offset mitigation test: estimate a single shared offset
        # vector on a train fold, subtract it on the held-out fold, recompute
        # GPFR there. K-fold so every patch gets a held-out correction.
        # NOTE: these are fold indices INTO THE HELD-OUT SET (0..899), a
        # separate inner split for offset estimation — NOT the same as the
        # outer train_idx/heldout_idx (2100/900) used for the Ridge head
        # above. Audit-caught bug (2026-09-16): this loop used to reuse the
        # name `train_idx`, shadowing the outer one, so every row's
        # "n_train" column silently reported 720 (a KFold fold size)
        # instead of the real 2100 Ridge-training-set size.
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
        corrected_pert_emb = np.zeros_like(pert_emb)
        for offset_fit_idx, offset_apply_idx in kf.split(pert_emb):
            offset = (pert_emb[offset_fit_idx] - clean_emb[offset_fit_idx]).mean(axis=0)
            corrected_pert_emb[offset_apply_idx] = pert_emb[offset_apply_idx] - offset
        corrected_pred = pd.DataFrame(head.predict(corrected_pert_emb), columns=expr_df.columns)
        corrected_scores = score_programs(corrected_pred, hallmark, prog_cfg)
        gpfr_corrected_by_program = {}
        for prog in clean_scores.columns:
            gpfr_corrected_by_program[prog] = gene_program_flip_rate(
                clean_scores[prog].values, corrected_scores[prog].values, method="per_sample_relative")

        for prog in clean_scores.columns:
            baseline = gpfr_by_program[prog]
            corrected = gpfr_corrected_by_program[prog]
            reduction = None
            if baseline is not None and baseline > 0 and corrected is not None:
                reduction = 1.0 - (corrected / baseline)
            rows.append({
                "perturbation": pname, "program": prog,
                "mean_embedding_l2_drift": float(emb_l2.mean()),
                "mean_embedding_cosine_sim": float(emb_cos.mean()),
                "mean_expr_drift": float(expr_drift.mean()),
                "embedding_to_expr_drift_corr": emb_to_expr_corr,
                "gpfr_baseline": baseline,
                "gpfr_after_linear_correction": corrected,
                "gpfr_relative_reduction": reduction,
                "n_train": len(train_idx), "n_heldout": len(heldout_idx),
                "ridge_alpha_selected": float(np.median(alpha_arr)),
                "heldout_mean_pearson_r": mean_r,
            })
        print(f"[perturbation] {pname} complete, elapsed={time.time() - p_start:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "fragility_decomposition.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}\n")
    print(df.to_string(index=False))

    print("\nDoes embedding drift predict expression drift? (per perturbation, should be strongly "
          "positive if fragility mostly enters at the encoder stage rather than being introduced later):")
    print(df.groupby("perturbation")["embedding_to_expr_drift_corr"].first().to_string())

    print("\nDoes the cheap linear-offset correction meaningfully reduce GPFR? "
          "(positive gpfr_relative_reduction = correction helped; near 0 or negative = it didn't):")
    print(df[df["gpfr_baseline"] > 0][["perturbation", "program", "gpfr_baseline",
                                         "gpfr_after_linear_correction", "gpfr_relative_reduction"]].to_string(index=False))


if __name__ == "__main__":
    main()
