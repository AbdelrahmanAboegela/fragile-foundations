"""Full multi-organ CONCH + Ridge morphology-preservation experiment."""
from __future__ import annotations

import argparse
import importlib.util
import os
import socket
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
socket.setdefaulttimeout(30)

import numpy as np
import pandas as pd
import torch
from PIL import Image

import functools
import json

from perturbations import (
    PERTURBATIONS, JPEG_SWEEP, STAIN_MAGNITUDE_SWEEP,
    brightness_shift, contrast_shift, gamma_correction, mild_blur, scanner_color_shift,
)

PERTURBATION_CALIBRATION_PATH = Path(__file__).resolve().parent.parent / "configs" / "perturbation_calibration.json"
_ORGAN_CALIBRATABLE_FNS = {
    "brightness": (brightness_shift, "factor"),
    "contrast": (contrast_shift, "factor"),
    "gamma": (gamma_correction, "gamma"),
    "blur": (mild_blur, "radius"),
    "scanner_color": (scanner_color_shift, "strength"),
}


def build_perturbation_set_for_organ(organ: str, base_set: dict) -> dict:
    """PER-ORGAN CALIBRATION (2026-09-16, audit-flagged): the 5 continuous-
    magnitude perturbations (brightness/contrast/gamma/blur/scanner_color)
    were calibrated to SSIM=0.90 on IDC only and the same constants reused
    unchanged for all 8 organs — an audit found this doesn't transfer
    (PRAD's `gamma` measured SSIM=0.800 with the IDC-calibrated constant,
    below this project's own morphology-preservation anchor). If
    `configs/perturbation_calibration.json` exists (written by
    `calibrate_all_perturbations.py --organs ...`), use THIS organ's own
    calibrated constants for those 5 perturbations; otherwise fall back to
    the IDC-calibrated defaults with an explicit warning, rather than
    silently using a mismatched magnitude. `jpeg` and `stain_shift` are
    deliberately excluded — neither is SSIM-target-calibrated by design
    (see perturbations.py's module docstring), so both pass through
    unchanged from `base_set` for every organ.
    """
    if base_set is not PERTURBATIONS:
        return base_set  # --jpeg-sweep / --stain-sweep runs don't use the 5 calibratable perturbations

    calibration = {}
    if PERTURBATION_CALIBRATION_PATH.exists():
        with open(PERTURBATION_CALIBRATION_PATH) as f:
            all_calibration = json.load(f)
        calibration = all_calibration.get(organ, {})

    if not calibration:
        print(f"[warn] {organ}: no per-organ perturbation calibration found in "
              f"{PERTURBATION_CALIBRATION_PATH.name} — falling back to IDC-calibrated defaults, "
              f"which an audit found do NOT reliably hit SSIM=0.90 on other organs. Run "
              f"`python calibrate_all_perturbations.py --organs {organ}` to fix this.", flush=True)
        return dict(base_set)

    param_key = {"brightness_factor": "brightness", "contrast_factor": "contrast",
                 "gamma": "gamma", "blur_radius": "blur", "scanner_color_strength": "scanner_color"}
    organ_set = dict(base_set)
    applied, fell_back = [], []
    for cal_key, pname in param_key.items():
        if cal_key in calibration and pname in _ORGAN_CALIBRATABLE_FNS:
            fn, kwarg = _ORGAN_CALIBRATABLE_FNS[pname]
            organ_set[pname] = functools.partial(fn, **{kwarg: calibration[cal_key]})
            applied.append(f"{pname}={calibration[cal_key]:.5f}")
        else:
            fell_back.append(pname)  # calibration script couldn't bracket SSIM=0.90 for this one — see its log
    print(f"[setup] {organ}: per-organ calibrated: {', '.join(applied) if applied else '(none)'}"
          + (f" | using IDC default (bracket failure during calibration) for: {fell_back}" if fell_back else ""),
          flush=True)
    return organ_set
from metrics import compute_lpips, compute_ssim, gene_program_flip_rate
from gene_program_scoring import load_hallmark_gene_sets, load_program_config, score_programs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "hest-bench"


def load_conch_encoder(device: str):
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "external/CONCH"))
    from conch.open_clip_custom import create_model_from_pretrained
    from huggingface_hub import get_token

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", "hf_hub:MahmoodLab/conch", hf_auth_token=get_token()
    )
    return model.to(device).eval(), preprocess


def load_uni_encoder(device: str):
    """UNI (ViT-L/16, 0.3B params, 1024-dim output) — the second encoder for
    the CONCH-generalization check (2026-09-16): closest prior benchmark
    (STP-Bench) found architecture effects have been conflated with encoder
    effects across this subfield, so a single-encoder fragility result is
    vulnerable to "this is a CONCH quirk." API verified from
    huggingface.co/MahmoodLab/UNI's README, not guessed.
    """
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform

    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True,
                               init_values=1e-5, dynamic_img_size=True)
    preprocess = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model.to(device).eval(), preprocess


ENCODERS = {
    "conch": {"loader": load_conch_encoder, "forward": lambda m, x: m.encode_image(x, proj_contrast=False, normalize=False)},
    "uni": {"loader": load_uni_encoder, "forward": lambda m, x: m(x)},
}


def embed_patches(encoder, preprocess, patches: list[Image.Image], device: str, forward_fn=None) -> np.ndarray:
    forward_fn = forward_fn or ENCODERS["conch"]["forward"]
    embeddings = []
    embed_start = time.time()
    with torch.inference_mode():
        for start in range(0, len(patches), 32):
            x = torch.stack([preprocess(img) for img in patches[start:start + 32]]).to(device)
            emb = forward_fn(encoder, x)
            embeddings.append(emb.cpu().numpy())
            completed = min(start + 32, len(patches))
            percent = 100.0 * completed / len(patches)
            print(f"[embed] {completed}/{len(patches)} ({percent:.1f}%) "
                  f"elapsed={time.time() - embed_start:.1f}s", flush=True)
    return np.concatenate(embeddings, axis=0)


# Must match calibrate_all_perturbations.py's CALIBRATION_RESERVED_TAIL —
# see that constant's docstring for the disjointness rationale.
CALIBRATION_RESERVED_TAIL = 200


def load_organ_sample(organ: str, sample_id: str, n_patches: int, seed: int = 42, exclude_tail: int = 0):
    # Load this file directly: importing hest normally pulls in the optional
    # trident WSI toolkit, which is not needed for pre-extracted patches.
    st_dataset_path = PROJECT_ROOT / "external/HEST/src/hest/bench/st_dataset.py"
    spec = importlib.util.spec_from_file_location("st_dataset", st_dataset_path)
    st_dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st_dataset)
    import anndata as ad

    patch_ds = st_dataset.H5PatchDataset(
        str(DATA_ROOT / organ / "patches" / f"{sample_id}.h5"), img_transform=None
    )
    adata = ad.read_h5ad(DATA_ROOT / organ / "adata" / f"{sample_id}.h5ad")

    # SPATIAL SAMPLING BIAS FIX (2026-09-16, audit-flagged): previously took
    # patches [0:n_patches] in raw file order. Xenium files are coordinate-
    # ordered, so "first N" meant a contiguous strip covering ~26% of the
    # slide's extent for IDC; Visium files are barcode-ordered (effectively
    # spatially scrambled), so the same code gave ~100% slide coverage for
    # CCRCC/PRAD — an easier, spatially-homogeneous task for IDC that biased
    # any cross-organ comparison. Fixed: a seeded random sample of indices,
    # same seed every organ, so coverage is comparable and the run stays
    # reproducible.
    rng = np.random.RandomState(seed)
    n_total = len(patch_ds)
    # CALIBRATION/EXPERIMENT DISJOINTNESS (2026-09-16): if this is the
    # organ's designated calibration sample (see _first_sample_h5-equivalent
    # logic in main()), exclude the tail range calibrate_all_perturbations.py
    # reserved for itself, so calibration and experiment patches never
    # overlap — provably, not just improbably.
    n_available = max(0, n_total - exclude_tail)
    chosen = rng.choice(n_available, size=min(n_patches, n_available), replace=False)
    chosen.sort()  # ascending order keeps H5 reads sequential-ish, not required for correctness

    patches, barcodes = [], []
    for i in chosen:
        item = patch_ds[int(i)]
        patches.append(Image.fromarray(item["imgs"].astype(np.uint8)))
        barcodes.append(item["barcodes"])

    valid = [barcode in adata.obs_names for barcode in barcodes]
    patches = [p for p, keep in zip(patches, valid) if keep]
    barcodes = [b for b, keep in zip(barcodes, valid) if keep]
    # On non-Xenium organs this deliberately keeps every gene.
    real_genes = [g for g in adata.var_names
                  if "Control" not in g and "BLANK" not in g and "Negative" not in g]
    return patches, adata[barcodes, real_genes].to_df()


def image_tensors(clean_img, pert_img):
    c = torch.from_numpy(np.asarray(clean_img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    p = torch.from_numpy(np.asarray(pert_img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return c, p


def discover_organs(requested: list[str] | None, skip_missing: bool) -> list[str]:
    names = requested or sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    found = []
    for organ in names:
        path = DATA_ROOT / organ
        if not path.is_dir() or not (path / "patches").is_dir() or not (path / "adata").is_dir():
            message = f"[warn] Skipping {organ}: patches/ and adata/ are not both available"
            if skip_missing:
                print(message)
                continue
            raise FileNotFoundError(message)
        found.append(organ)
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organs", nargs="+", help="restrict the run to named organs")
    parser.add_argument("--n-patches", type=int, default=400, help="maximum patches pooled per organ")
    parser.add_argument("--seed", type=int, default=42,
                         help="random seed for BOTH patch sampling (load_organ_sample) and the "
                              "train/held-out split — added 2026-09-16 for a multi-seed split-"
                              "stability check (is the core result an artifact of one arbitrary "
                              "split?), previously hardcoded to 42 in both places")
    parser.add_argument("--no-skip-missing-organs", dest="skip_missing", action="store_false",
                        help="fail when a requested organ is incomplete")
    parser.add_argument("--jpeg-sweep", action="store_true",
                        help="run only the JPEG quality-level sweep (q30/50/70/90) instead of "
                             "the full 7-perturbation suite — for the dose-response analysis")
    parser.add_argument("--stain-sweep", action="store_true",
                        help="run only the stain-magnitude sweep (identity null control / mild / "
                             "moderate) instead of the full 7-perturbation suite — added 2026-09-16 "
                             "so the stain_shift_identity null control (previously wired only into "
                             "generate_example_figures.py, never actually measured against real "
                             "GPFR) can be run through the real pipeline to quantify how much of "
                             "stain_shift's GPFR is reconstruction-round-trip noise vs a real effect")
    parser.add_argument("--encoder", choices=list(ENCODERS.keys()), default="conch",
                        help="frozen encoder to use (conch or uni) — for the encoder-generalization check")
    parser.set_defaults(skip_missing=True)
    args = parser.parse_args()
    if args.n_patches <= 0:
        parser.error("--n-patches must be positive")
    if args.jpeg_sweep and args.stain_sweep:
        parser.error("--jpeg-sweep and --stain-sweep are mutually exclusive")
    perturbation_set = JPEG_SWEEP if args.jpeg_sweep else (STAIN_MAGNITUDE_SWEEP if args.stain_sweep else PERTURBATIONS)
    encoder_spec = ENCODERS[args.encoder]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}, encoder={args.encoder}", flush=True)
    organs = discover_organs(args.organs, args.skip_missing)
    print(f"[setup] organs to run: {organs}", flush=True)
    hallmark, prog_cfg = load_hallmark_gene_sets(), load_program_config()
    print(f"[setup] loading {args.encoder} encoder...", flush=True)
    encoder_start = time.time()
    encoder, preprocess = encoder_spec["loader"](device)
    print(f"[setup] {args.encoder} loaded, elapsed={time.time() - encoder_start:.1f}s", flush=True)
    results = []
    raw_score_rows = []  # per-held-out-sample clean/perturbed scores, for bootstrap CIs — see analyze_statistics.py
    df = pd.DataFrame(columns=["encoder", "organ", "perturbation", "program", "gpfr", "mean_ssim", "mean_lpips",
                                 "n_samples_pooled", "n_train", "n_heldout",
                                 "ridge_alpha_selected", "ridge_alpha_min", "ridge_alpha_max",
                                 "heldout_mean_pearson_r", "heldout_mean_pearson_r_hallmark_genes",
                                 "heldout_mean_r2", "heldout_mean_r2_hallmark_genes"])  # safe default if organs is empty

    for organ_index, organ in enumerate(organs, start=1):
        organ_start = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[organ] === Starting {organ} ({organ_index}/{len(organs)}) "
              f"at {timestamp} ===", flush=True)
        sample_ids = sorted(p.stem for p in (DATA_ROOT / organ / "patches").glob("*.h5"))
        if not sample_ids:
            print(f"[warn] Skipping {organ}: no patch files found")
            continue
        base, remainder = divmod(args.n_patches, len(sample_ids))
        all_patches, expressions, all_slide_ids = [], [], []
        # CALIBRATION/EXPERIMENT DISJOINTNESS (2026-09-16, updated after
        # calibration moved from 1 sample per organ to up to
        # MAX_CALIBRATION_SAMPLES): read the EXACT sample_ids calibration
        # actually touched for this organ from the JSON's
        # "_calibration_sample_ids" (written by calibrate_all_perturbations.py),
        # rather than assuming it was always sample_ids[0] — that assumption
        # became wrong the moment calibration started pooling multiple
        # samples for organs like CCRCC/PRAD. Falls back to the old
        # single-sample assumption only if the JSON predates this field.
        calibration_sample_ids = set()
        if PERTURBATION_CALIBRATION_PATH.exists():
            with open(PERTURBATION_CALIBRATION_PATH) as f:
                _cal = json.load(f).get(organ, {})
            calibration_sample_ids = set(_cal.get("_calibration_sample_ids", []))
        if not calibration_sample_ids and sample_ids:
            calibration_sample_ids = {sample_ids[0]}  # legacy fallback, see comment above
        for index, sample_id in enumerate(sample_ids):
            adata_path = DATA_ROOT / organ / "adata" / f"{sample_id}.h5ad"
            if not adata_path.exists():
                print(f"[warn] Skipping sample {organ}/{sample_id}: matching AnnData missing")
                continue
            tail = CALIBRATION_RESERVED_TAIL if sample_id in calibration_sample_ids else 0
            patches, expr = load_organ_sample(organ, sample_id, base + (index < remainder), seed=args.seed, exclude_tail=tail)
            if len(patches) == 0:
                print(f"[warn] Skipping sample {organ}/{sample_id}: no barcode matches")
                continue
            all_patches.extend(patches)
            all_slide_ids.extend([sample_id] * len(patches))
            expressions.append(expr)
            print(f"  {organ}/{sample_id}: {len(patches)} patches, {expr.shape[1]} genes")
        if not expressions:
            print(f"[warn] Skipping {organ}: no usable samples")
            continue

        common_genes = set(expressions[0].columns)
        for expr in expressions[1:]:
            common_genes &= set(expr.columns)
        common_genes = sorted(common_genes)
        expr_df = pd.concat([expr[common_genes] for expr in expressions], axis=0)
        assert expr_df.isna().sum().sum() == 0, f"unexpected NaN in {organ} expression matrix"
        print(f"{organ}: {len(all_patches)} patches, {len(expressions)} samples, {len(common_genes)} common genes")

        # LIBRARY-SIZE NORMALIZATION (2026-09-17, adversarial-review-caught,
        # independently confirmed real: adata.X is raw uint16 counts with
        # NO normalization applied anywhere downstream. Per-patch total
        # panel counts on IDC ranged 1-27,646 (>10,000x). Directly measured
        # on the real held-out set: `proliferation`'s program score
        # correlated with raw total counts at r=0.82 — i.e. "gene-program
        # activation" was substantially a readout of how much RNA/how many
        # cells were in the patch, not biology, before this fix. Standard
        # fix for targeted-panel spatial data (this is what's actually
        # measurable here — a true whole-transcriptome CPM isn't available
        # for a ~280-gene Xenium panel): scale each patch's counts by its
        # own total panel counts, rescaled to the cohort median total, so
        # the model predicts RELATIVE expression, not absolute abundance.
        # This is a per-row operation (each patch normalized by its own
        # total only) and does not leak information across patches or
        # across the train/held-out split performed below.
        library_size = expr_df.sum(axis=1)
        assert (library_size > 0).all(), f"{organ}: found a patch with zero total panel counts — cannot normalize"
        median_library_size = float(library_size.median())
        print(f"[normalize] {organ}: per-patch total panel counts range "
              f"[{library_size.min():.0f}, {library_size.max():.0f}], median={median_library_size:.0f} — "
              f"rescaling every patch's counts to this median total before fitting", flush=True)
        expr_df = expr_df.div(library_size, axis=0) * median_library_size

        # SILENT GENE-COLLAPSE WARNING (2026-09-16, audit-flagged): PAAD's
        # common-gene intersection collapsed to 98 of ~480 genes (Jaccard
        # 0.12 between two of its samples) with nothing printed about it —
        # the count alone doesn't tell you it collapsed. Warn explicitly
        # when the intersection is much smaller than the largest single
        # sample's panel, so this is visible in the log, not just detectable
        # by comparing platform_table.csv against this print by hand.
        max_single_sample_genes = max(len(expr.columns) for expr in expressions)
        if len(expressions) > 1 and len(common_genes) < 0.5 * max_single_sample_genes:
            print(f"[warn] {organ}: common-gene intersection ({len(common_genes)}) is less than "
                  f"half of the largest single sample's panel ({max_single_sample_genes}) — "
                  f"this organ's samples likely use inconsistent gene panels/probe versions; "
                  f"program scores here are computed from far fewer genes than platform_table.csv's "
                  f"per-sample count would suggest", flush=True)

        # DATA-LEAKAGE FIX (2026-09-16, audit-flagged): the Ridge head was
        # previously fit AND evaluated on the identical data — no held-out
        # split existed anywhere, so (a) no out-of-sample predictive
        # validity was ever measured, meaning "predictions are unstable"
        # was never shown to mean anything, since the predictions' basic
        # accuracy was unverified; and (b) GPFR's denominator (clean scores'
        # std) was an in-sample statistic directly controlled by the
        # untuned alpha=1.0, not a clean measurement of perturbation effect.
        # Fix: split BEFORE fitting anything, cross-validate alpha on TRAIN
        # only, report held-out Pearson r as a genuine validity check, and
        # compute GPFR entirely on the held-out set using the train-fitted
        # head — clean vs. perturbed predictions are both out-of-sample.
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import train_test_split
        from scipy.stats import pearsonr

        n = len(all_patches)
        indices = np.arange(n)
        train_idx, heldout_idx = train_test_split(indices, test_size=0.3, random_state=args.seed)
        heldout_patches = [all_patches[i] for i in heldout_idx]
        # SLIDE ID PER HELD-OUT PATCH (2026-09-17, adversarial-review-caught):
        # analyze_statistics.py's bootstrap previously resampled individual
        # HELD-OUT PATCHES as if they were independent, but patches from the
        # same slide share slide-level technical/biological variation, so
        # patch-level resampling understates the true variance of GPFR and
        # can make CIs artificially narrow / p-values artificially small.
        # Recording which slide each held-out patch came from lets
        # analyze_statistics.py do a proper cluster (slide-level) bootstrap.
        heldout_slide_ids = np.array(all_slide_ids)[heldout_idx]
        print(f"[split] {len(train_idx)} train patches, {len(heldout_idx)} held-out patches "
              f"(random patch-level split, seed={args.seed} — NOTE: nearby patches within a sample may "
              f"be spatially correlated, so this does not rule out all leakage, only the "
              f"identical-data leakage the audit flagged)", flush=True)

        clean_embeddings = embed_patches(encoder, preprocess, all_patches, device, forward_fn=encoder_spec["forward"])
        train_emb, heldout_emb = clean_embeddings[train_idx], clean_embeddings[heldout_idx]
        train_expr, heldout_expr = expr_df.values[train_idx], expr_df.values[heldout_idx]

        # PER-TARGET ALPHA FIX (2026-09-16, audit-flagged): a single shared
        # alpha (the original fix, `cv=5`) is chosen by mean R2 across ALL
        # gene outputs at once. On whole-transcriptome organs (~18-36k
        # genes), thousands of near-zero-count/near-zero-variance genes
        # dominate that mean and drag the shared alpha up (empirically
        # confirmed: alpha=316-1000 on CCRCC/HCC vs ~31.6 on targeted-panel
        # organs), over-shrinking the head even for genes with real signal.
        # `alpha_per_target=True` lets each gene pick its own alpha —
        # verified on a synthetic 1-real-signal + 1999-noise-gene test to
        # correctly assign the real gene alpha~0.03 instead of ~10000.
        # This requires `cv=None` (sklearn's efficient generalized/LOO CV;
        # `cv!=None` and `alpha_per_target=True` are mutually exclusive in
        # sklearn — confirmed via ValueError, not assumed).
        alphas = np.logspace(-2, 4, 13)
        head = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True).fit(train_emb, train_expr)
        alpha_arr = np.atleast_1d(head.alpha_)
        print(f"[validity] per-target alpha selected via efficient LOO-CV on TRAIN only: "
              f"median={np.median(alpha_arr):.4g}, min={alpha_arr.min():.4g}, max={alpha_arr.max():.4g} "
              f"(was a single hardcoded 1.0, then a single shared CV-selected value before this fix)", flush=True)

        heldout_pred_clean = head.predict(heldout_emb)
        gene_corrs = []
        for gi in range(heldout_expr.shape[1]):
            if heldout_expr[:, gi].std() > 1e-8 and heldout_pred_clean[:, gi].std() > 1e-8:
                r, _ = pearsonr(heldout_expr[:, gi], heldout_pred_clean[:, gi])
                if not np.isnan(r):
                    gene_corrs.append(r)
        mean_r = float(np.mean(gene_corrs)) if gene_corrs else float("nan")
        print(f"[validity] held-out (out-of-sample) mean Pearson r across "
              f"{len(gene_corrs)}/{heldout_expr.shape[1]} genes: {mean_r:.4f} "
              f"— THIS is the number that says whether the model predicts anything real; "
              f"GPFR is only meaningful if this is clearly above 0", flush=True)

        # HALLMARK-RESTRICTED VALIDITY (2026-09-16, added after observing the
        # per-target alpha fix barely moved the whole-transcriptome-organ
        # mean_r above): mean_r across ALL genes is dominated by the
        # thousands of genes with weak/no real histology signal regardless
        # of alpha tuning — that's expected biology, not a sign the fix
        # failed. What actually matters for this paper is prediction quality
        # on the genes GPFR is computed from (the 5 Hallmark programs'
        # constituent genes), which is a small, specific subset. Report that
        # separately so the alpha fix's real effect is visible.
        hallmark_gene_union = set()
        for spec in prog_cfg["programs"].values():
            for hset in spec["hallmark_sets"]:
                hallmark_gene_union.update(hallmark.get(hset, []))
        hallmark_cols = [g for g in expr_df.columns if g in hallmark_gene_union]
        hallmark_idx = [expr_df.columns.get_loc(g) for g in hallmark_cols]
        hallmark_corrs = []
        for gi in hallmark_idx:
            if heldout_expr[:, gi].std() > 1e-8 and heldout_pred_clean[:, gi].std() > 1e-8:
                r, _ = pearsonr(heldout_expr[:, gi], heldout_pred_clean[:, gi])
                if not np.isnan(r):
                    hallmark_corrs.append(r)
        mean_r_hallmark = float(np.mean(hallmark_corrs)) if hallmark_corrs else float("nan")
        print(f"[validity] held-out mean Pearson r restricted to the "
              f"{len(hallmark_corrs)}/{len(hallmark_cols)} Hallmark-program genes actually "
              f"present in this organ's panel: {mean_r_hallmark:.4f} — this is the number that "
              f"actually matters for GPFR, as opposed to the whole-transcriptome mean above",
              flush=True)

        # R^2 ALONGSIDE PEARSON r (2026-09-17, adversarial-review-caught,
        # added for the baseline-model comparison in baseline_models.py):
        # Pearson r is undefined (zero variance) for a constant-prediction
        # baseline like "always predict the training-set mean", so it can't
        # be used to compare CONCH-Ridge against that floor. R^2 handles the
        # constant baseline correctly (it evaluates to ~0 by construction),
        # so it's the metric baseline_models.py reports and this needs to
        # report it too for a direct, apples-to-apples comparison in Table 1.
        from sklearn.metrics import r2_score
        mean_r2 = float(r2_score(heldout_expr, heldout_pred_clean, multioutput="uniform_average"))
        mean_r2_hallmark = float(r2_score(heldout_expr[:, hallmark_idx], heldout_pred_clean[:, hallmark_idx],
                                           multioutput="uniform_average")) if hallmark_idx else float("nan")
        print(f"[validity] held-out R^2 (whole panel): {mean_r2:.4f}, "
              f"R^2 (Hallmark genes): {mean_r2_hallmark:.4f} — comparable to baseline_models.py's "
              f"training-mean and total-counts-linear floors, unlike Pearson r which is undefined "
              f"for a constant-prediction baseline", flush=True)

        clean_pred_scores = score_programs(pd.DataFrame(heldout_pred_clean, columns=expr_df.columns), hallmark, prog_cfg)
        organ_perturbation_set = build_perturbation_set_for_organ(organ, perturbation_set)
        for perturbation_index, (pname, pfunc) in enumerate(organ_perturbation_set.items(), start=1):
            perturbation_start = time.time()
            print(f"[perturbation] {pname} "
                  f"({perturbation_index}/{len(organ_perturbation_set)})", flush=True)
            perturbed = [pfunc(img) for img in heldout_patches]  # held-out only, not all_patches
            pert_embeddings = embed_patches(encoder, preprocess, perturbed, device, forward_fn=encoder_spec["forward"])
            pert_scores = score_programs(pd.DataFrame(head.predict(pert_embeddings), columns=expr_df.columns), hallmark, prog_cfg)
            ssim_vals, lpips_vals = [], []
            metrics_start = time.time()
            for i, (clean_img, pert_img) in enumerate(zip(heldout_patches, perturbed), start=1):
                c, p = image_tensors(clean_img, pert_img)
                ssim_vals.append(compute_ssim(c, p))
                lpips_vals.append(compute_lpips(c * 2 - 1, p * 2 - 1))
                if i % 50 == 0 or i == len(heldout_patches):
                    print(f"  [ssim/lpips] {i}/{len(heldout_patches)} "
                          f"({100.0 * i / len(heldout_patches):.1f}%) elapsed={time.time() - metrics_start:.1f}s",
                          flush=True)
            usable_lpips = [v for v in lpips_vals if v is not None]

            # RAW PER-SAMPLE SCORES (2026-09-16, added for bootstrap CIs — a
            # real gap for a Q1 submission: the CSV only ever stored the
            # final GPFR scalar, with no way to get a confidence interval on
            # it post-hoc without rerunning inference. Saving the actual
            # per-held-out-sample clean/perturbed program scores lets
            # analyze_statistics.py resample GPFR directly, cheaply, without
            # touching the GPU again.
            for program in clean_pred_scores.columns:
                for sample_i in range(len(clean_pred_scores)):
                    raw_score_rows.append({
                        "encoder": args.encoder, "organ": organ, "perturbation": pname, "program": program,
                        "sample_index": sample_i, "slide_id": heldout_slide_ids[sample_i],
                        "clean_score": float(clean_pred_scores[program].values[sample_i]),
                        "pert_score": float(pert_scores[program].values[sample_i]),
                    })

            for program in clean_pred_scores.columns:
                results.append({"encoder": args.encoder, "organ": organ, "perturbation": pname, "program": program,
                                "gpfr": gene_program_flip_rate(clean_pred_scores[program].values, pert_scores[program].values,
                                                                method="per_sample_relative"),
                                "mean_ssim": float(np.mean(ssim_vals)),
                                "mean_lpips": float(np.mean(usable_lpips)) if usable_lpips else None,
                                "n_samples_pooled": len(expressions),
                                "n_train": len(train_idx), "n_heldout": len(heldout_idx),
                                # ridge_alpha_selected is now a per-organ MEDIAN across genes
                                # (2026-09-16 fix: alpha_per_target=True gives one alpha per
                                # gene, not one per organ) — min/max recorded separately since
                                # the spread itself is informative (wide spread = the fix is
                                # doing real work, not just changing a single number).
                                "ridge_alpha_selected": float(np.median(alpha_arr)),
                                "ridge_alpha_min": float(alpha_arr.min()),
                                "ridge_alpha_max": float(alpha_arr.max()),
                                "heldout_mean_pearson_r": mean_r,
                                "heldout_mean_pearson_r_hallmark_genes": mean_r_hallmark,
                                "heldout_mean_r2": mean_r2,
                                "heldout_mean_r2_hallmark_genes": mean_r2_hallmark})
            print(f"[perturbation] {pname} complete "
                  f"elapsed={time.time() - perturbation_start:.1f}s", flush=True)
        print(f"[organ] === Completed {organ} "
              f"elapsed={time.time() - organ_start:.1f}s ===", flush=True)

        # INCREMENTAL WRITE (2026-09-16, audit-flagged): previously the CSV
        # was only written once, after ALL organs finished — a crash on the
        # last organ discarded every completed organ's results. Write after
        # every organ instead, so a crash only costs the in-progress organ.
        df = pd.DataFrame(results, columns=["encoder", "organ", "perturbation", "program", "gpfr", "mean_ssim", "mean_lpips",
                                              "n_samples_pooled", "n_train", "n_heldout",
                                              "ridge_alpha_selected", "ridge_alpha_min", "ridge_alpha_max",
                                              "heldout_mean_pearson_r", "heldout_mean_pearson_r_hallmark_genes",
                                              "heldout_mean_r2", "heldout_mean_r2_hallmark_genes"])
        # ORGAN-SCOPE IN FILENAME (2026-09-16, added after a real incident:
        # an IDC-only run silently overwrote a completed 8-organ run's CSV,
        # because the filename previously depended only on --encoder/
        # --jpeg-sweep/--stain-sweep, not on WHICH organs were actually run.
        # 90 minutes of compute was only recoverable because it happened to
        # be committed to git moments earlier — do not rely on that again.
        organ_scope = "-".join(sorted(o.lower() for o in organs)) if len(organs) <= 2 else f"{len(organs)}organs"
        seed_tag = "" if args.seed == 42 else f"_seed{args.seed}"  # default seed keeps existing filenames unchanged
        base_name = "jpeg_sweep_results" if args.jpeg_sweep else ("stain_sweep_results" if args.stain_sweep else "main_experiment_results")
        out_name = f"{base_name}_{organ_scope}{seed_tag}.csv" if args.encoder == "conch" else f"{base_name}_{organ_scope}{seed_tag}_{args.encoder}.csv"
        out = PROJECT_ROOT / "results" / out_name
        out.parent.mkdir(exist_ok=True)
        df.to_csv(out, index=False)
        raw_out = PROJECT_ROOT / "results" / f"{out_name.rsplit('.', 1)[0]}_raw_scores.csv.gz"
        pd.DataFrame(raw_score_rows).to_csv(raw_out, index=False, compression="gzip")
        print(f"[checkpoint] wrote {len(df)} rows ({organ_index}/{len(organs)} organs done) to {out}, "
              f"{len(raw_score_rows)} raw per-sample rows to {raw_out}", flush=True)

    if not df.empty:
        print("\nHighest GPFR by program:")
        print(df.loc[df.groupby("program")["gpfr"].idxmax(), ["program", "organ", "perturbation", "gpfr"]].to_string(index=False))
        print("\nOverall results:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
