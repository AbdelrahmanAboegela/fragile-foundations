"""Cross-encoder GPFR magnitude-gap check using a shared (pooled) sigma_c.

WHY THIS EXISTS (2026-09-17, third-adversarial-review-caught): GPFR's
denominator (cohort_std of clean scores) is computed per-encoder, so a
cross-encoder magnitude comparison ("UNI's jpeg GPFR exceeds CONCH's") is
not automatically scale-free -- CONCH and UNI have different sigma_c per
program. An earlier draft claimed a specific pooled-sigma recomputation
result ("the jpeg gap shrinks from 0.046 to 0.025") with no producing
script; it does not reproduce under any pooling definition tried by
independent review. This script is the actual, real computation: for a
given (perturbation, program), pool clean scores across BOTH encoders
(same program, same perturbation, same seed) into one array, take its std
as a single shared sigma_c, and recompute each encoder's GPFR using that
shared sigma_c instead of its own per-encoder one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"


def gpfr_with_sigma(clean: np.ndarray, pert: np.ndarray, sigma_c: float, z_thresh: float = 1.0) -> float:
    z = np.abs(pert - clean) / (sigma_c + 1e-8)
    return float((z > z_thresh).mean())


def main():
    conch = pd.read_csv(RESULTS / "main_experiment_results_idc_raw_scores.csv.gz")
    uni = pd.read_csv(RESULTS / "main_experiment_results_idc_uni_raw_scores.csv.gz")

    rows = []
    for pert in ["jpeg", "scanner_color"]:
        for prog in sorted(set(conch.program.unique()) & set(uni.program.unique())):
            c = conch[(conch.perturbation == pert) & (conch.program == prog)].sort_values("sample_index")
            u = uni[(uni.perturbation == pert) & (uni.program == prog)].sort_values("sample_index")

            own_gpfr_c = gpfr_with_sigma(c.clean_score.values, c.pert_score.values, c.clean_score.values.std())
            own_gpfr_u = gpfr_with_sigma(u.clean_score.values, u.pert_score.values, u.clean_score.values.std())

            pooled_sigma = np.concatenate([c.clean_score.values, u.clean_score.values]).std()
            pooled_gpfr_c = gpfr_with_sigma(c.clean_score.values, c.pert_score.values, pooled_sigma)
            pooled_gpfr_u = gpfr_with_sigma(u.clean_score.values, u.pert_score.values, pooled_sigma)

            rows.append({
                "perturbation": pert, "program": prog,
                "conch_gpfr_own_sigma": own_gpfr_c, "uni_gpfr_own_sigma": own_gpfr_u,
                "gap_own_sigma": own_gpfr_u - own_gpfr_c,
                "conch_gpfr_pooled_sigma": pooled_gpfr_c, "uni_gpfr_pooled_sigma": pooled_gpfr_u,
                "gap_pooled_sigma": pooled_gpfr_u - pooled_gpfr_c,
            })

    df = pd.DataFrame(rows)
    out = RESULTS / "pooled_sigma_check.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {out}")

    # max-over-programs summary per perturbation, matching how the paper's
    # seed-stability "max-GPFR" figures are defined
    for pert in ["jpeg", "scanner_color"]:
        sub = df[df.perturbation == pert]
        own_gap = sub.loc[sub["uni_gpfr_own_sigma"].idxmax(), "uni_gpfr_own_sigma"] - sub.loc[sub["conch_gpfr_own_sigma"].idxmax(), "conch_gpfr_own_sigma"]
        pooled_gap = sub.loc[sub["uni_gpfr_pooled_sigma"].idxmax(), "uni_gpfr_pooled_sigma"] - sub.loc[sub["conch_gpfr_pooled_sigma"].idxmax(), "conch_gpfr_pooled_sigma"]
        print(f"\n{pert}: max-GPFR gap (UNI-CONCH) own-sigma={own_gap:.4f}, pooled-sigma={pooled_gap:.4f}")


if __name__ == "__main__":
    main()
