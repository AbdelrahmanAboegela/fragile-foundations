"""Bootstrap confidence intervals and a bootstrap-based significance test,
with Benjamini-Hochberg FDR correction, for every (organ, perturbation,
program) GPFR value.

WHY THIS EXISTS (2026-09-16): every results CSV so far has reported GPFR as
a single point estimate with no uncertainty quantification and no
correction for testing 5 programs x 7 perturbations x 8 organs = up to 280
hypotheses at once. A Q1 reviewer will ask "is this GPFR actually different
from noise given your held-out sample size, and did you correct for how
many comparisons you're making?" This script answers both from the raw
per-sample scores main_experiment.py now saves (results/*_raw_scores.csv.gz),
without touching the GPU again.

BOOTSTRAP CI: resample held-out sample INDICES with replacement B times,
recompute GPFR each time (same z_thresh method as the real metric), report
the 2.5th/97.5th percentile — how much would GPFR vary with a different
held-out sample from the same organ.

SIGNIFICANCE: p = the fraction of bootstrap resamples with GPFR <= 0 (a
one-sided test against the boundary null "no sample ever flips"). This
reuses the same bootstrap draws as the CI, so it costs nothing extra.

WHY NOT A PERMUTATION TEST (2026-09-16, dead end kept as documentation, not
deleted, so this isn't silently re-attempted later): the obvious
alternative — shuffle the perturbed-score array relative to clean scores,
breaking the per-sample correspondence, and see if the real (paired) GPFR
exceeds this scrambled-pairing null — does NOT work for this metric. GPFR
compares |perturbed_i - clean_i| against cohort_std(clean), i.e. against
the COHORT'S OWN between-sample spread. A real perturbation effect is
normally a SMALL, bounded per-sample nudge, so properly-paired data stays
close to that cohort spread. But a full shuffle replaces perturbed_i with
an unrelated sample's score, drawn from the SAME wide cohort distribution
as clean_i — so |shuffled_pert_i - clean_i| behaves like the difference of
two independent draws from the cohort, with expected magnitude
sqrt(2)*cohort_std, comfortably ABOVE the z_thresh=1*cohort_std flip
boundary regardless of whether there's a real effect. The scrambled null is
therefore always noisier than real data and always gives a high null GPFR
baseline, making every p-value come out ~1.0 even for an obviously-real,
directly-constructed synthetic effect (verified before trusting this file
— caught during testing, not caught by a reviewer). The bootstrap-based
p-value above avoids this by never needing an artificial null distribution
at all: it just asks how often GPFR's own sampling distribution touches 0.

Multiple testing correction (Benjamini-Hochberg FDR, not Bonferroni — the
standard, appropriately less conservative choice for an exploratory
multi-comparison table like this one) is applied across every (organ,
perturbation, program) triple in the INPUT FILE passed via --raw-scores at
once, not per-perturbation subsets within that file — narrowing the
correction scope to get more "significant" results would itself be a
subtle form of p-hacking. NOTE (2026-09-17, second-adversarial-review-
caught): this script is invoked once per encoder in this project's actual
pipeline (rerun_all.sh), so in practice the correction scope is the 35
(perturbation, program) tests for ONE encoder/organ's raw-scores file, not
jointly across encoders — the paper's "corrected across all 35 combinations
per encoder" phrasing describes the real scope; an earlier version of this
docstring overstated it as "ALL tested ... triples at once" as if that
meant across encoders too, which it does not.

CLUSTER BOOTSTRAP BY SLIDE (2026-09-17, adversarial-review-caught): once
main_experiment.py started recording which slide each held-out patch came
from (`slide_id` column in *_raw_scores.csv.gz), this script resamples
SLIDES with replacement instead of individual patches whenever that column
is present and the organ has more than one slide. Patches from the same
slide are not independent (shared staining batch, scanner, tumor
composition), so a plain patch-level bootstrap overstates the effective
sample size and can make CIs artificially narrow. Older raw-scores files
without `slide_id` fall back to the original patch-level bootstrap, with an
explicit warning at startup so this degradation is never silent.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import gene_program_flip_rate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
N_BOOTSTRAP = 2000
RNG_SEED = 42


def bootstrap_ci_and_pvalue(clean: np.ndarray, pert: np.ndarray, rng: np.random.Generator,
                             n_boot: int = N_BOOTSTRAP, slide_ids: np.ndarray | None = None):
    """Patch-level bootstrap by default. If `slide_ids` is given (and has
    more than one unique slide), does a CLUSTER bootstrap instead: resample
    SLIDES with replacement, then pool every patch belonging to the drawn
    slides, rather than resampling individual patches.

    WHY (2026-09-17, adversarial-review-caught): patches from the same slide
    share slide-level technical variation (staining batch, scanner, section
    thickness) and biological variation (that tumor's specific composition),
    so they are not independent draws. Resampling patches as if they were
    treats N_patches as the effective sample size, which is too optimistic —
    the true effective sample size is closer to N_slides. A patch-level
    bootstrap can therefore make CIs too narrow and p-values too small
    (pseudo-replication). Resampling whole slides respects the real
    clustering: a bootstrap draw can only vary at the slide level, not
    manufacture new between-slide combinations that were never observed.
    """
    n = len(clean)
    use_cluster = slide_ids is not None and len(np.unique(slide_ids)) > 1
    if use_cluster:
        unique_slides = np.unique(slide_ids)
        slide_to_positions = {s: np.where(slide_ids == s)[0] for s in unique_slides}
        n_slides = len(unique_slides)
    boot_gpfrs = np.empty(n_boot)
    for b in range(n_boot):
        if use_cluster:
            drawn_slides = unique_slides[rng.integers(0, n_slides, size=n_slides)]
            idx = np.concatenate([slide_to_positions[s] for s in drawn_slides])
        else:
            idx = rng.integers(0, n, size=n)
        g = gene_program_flip_rate(clean[idx], pert[idx], method="per_sample_relative")
        boot_gpfrs[b] = g if g is not None else np.nan
    valid = boot_gpfrs[~np.isnan(boot_gpfrs)]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan
    ci_lo, ci_hi = float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))
    p_value = float((valid <= 0).sum() + 1) / (len(valid) + 1)  # +1/+1: avoid p=0, standard small-sample correction
    return ci_lo, ci_hi, p_value


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Returns FDR-adjusted q-values, same order as input. NaN p-values pass through as NaN."""
    n = len(pvalues)
    qvalues = np.full(n, np.nan)
    valid_mask = ~np.isnan(pvalues)
    valid_p = pvalues[valid_mask]
    n_valid = len(valid_p)
    if n_valid == 0:
        return qvalues
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    ranks = np.arange(1, n_valid + 1)
    raw_q = ranked_p * n_valid / ranks
    adjusted = np.minimum.accumulate(raw_q[::-1])[::-1]  # standard BH step-up monotonicity fix
    adjusted = np.clip(adjusted, 0, 1)
    valid_q = np.empty(n_valid)
    valid_q[order] = adjusted
    qvalues[valid_mask] = valid_q
    return qvalues


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-scores", required=True, help="path to a *_raw_scores.csv.gz file")
    parser.add_argument("--out", default=None, help="output CSV path (default: alongside --raw-scores)")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_scores)
    rng = np.random.default_rng(RNG_SEED)
    has_slide_id = "slide_id" in raw.columns

    groups = raw.groupby(["encoder", "organ", "perturbation", "program"])
    print(f"[setup] {len(groups)} (encoder, organ, perturbation, program) groups to analyze, "
          f"{N_BOOTSTRAP} bootstrap resamples each "
          f"({'cluster bootstrap by slide' if has_slide_id else 'patch-level bootstrap -- no slide_id column, cannot cluster'})",
          flush=True)

    rows = []
    for i, ((encoder, organ, perturbation, program), g) in enumerate(groups, start=1):
        g = g.sort_values("sample_index")
        clean = g["clean_score"].values
        pert = g["pert_score"].values
        slide_ids = g["slide_id"].values if has_slide_id else None
        observed = gene_program_flip_rate(clean, pert, method="per_sample_relative")
        if observed is None:
            rows.append({"encoder": encoder, "organ": organ, "perturbation": perturbation, "program": program,
                         "gpfr": None, "ci_lo": None, "ci_hi": None, "p_value": None, "n_heldout": len(clean),
                         "n_slides": len(np.unique(slide_ids)) if has_slide_id else None})
            continue
        ci_lo, ci_hi, p = bootstrap_ci_and_pvalue(clean, pert, rng, slide_ids=slide_ids)
        rows.append({"encoder": encoder, "organ": organ, "perturbation": perturbation, "program": program,
                     "gpfr": observed, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_value": p, "n_heldout": len(clean),
                     "n_slides": len(np.unique(slide_ids)) if has_slide_id else None})
        if i % 10 == 0 or i == len(groups):
            print(f"[progress] {i}/{len(groups)} groups done", flush=True)

    out_df = pd.DataFrame(rows)
    out_df["q_value_fdr"] = benjamini_hochberg(out_df["p_value"].values)
    out_df["significant_fdr_0.05"] = out_df["q_value_fdr"] < 0.05

    out_path = Path(args.out) if args.out else Path(args.raw_scores).with_name(
        Path(args.raw_scores).name.replace("_raw_scores.csv.gz", "_with_stats.csv"))
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path}")
    print(f"\n{out_df['significant_fdr_0.05'].sum()}/{out_df['significant_fdr_0.05'].notna().sum()} "
          f"(organ, perturbation, program) triples significant at FDR<0.05 after correcting for "
          f"{out_df['p_value'].notna().sum()} tests")
    print("\nTop 15 by GPFR:")
    print(out_df.sort_values("gpfr", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
