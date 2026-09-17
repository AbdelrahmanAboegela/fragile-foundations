"""Sensitivity sweep for GPFR's z_thresh parameter (default 1.0).

WHY THIS EXISTS (2026-09-16): metrics.py::gene_program_flip_rate's
docstring is explicit that z_thresh=1.0 is a conventional, interpretable
choice ("more than one cohort standard deviation of movement"), not one
derived from a formal power analysis. Presenting a single z_thresh's GPFR
numbers without checking whether the qualitative conclusions (which
perturbation/program/organ combinations look fragile) survive a different
threshold would be exactly the kind of unvalidated-but-load-bearing
convention this project's own standing rule (no unverified claims) argues
against. This script recomputes GPFR at several z_thresh values from the
raw per-sample scores (results/*_raw_scores.csv.gz, written by
main_experiment.py) and reports the RANK CORRELATION between the default
threshold's ranking of (organ, perturbation, program) triples and each
alternative threshold's ranking — the qualitative story ("X is more
fragile than Y") only holds up if that ranking is stable across thresholds,
independent of whether the absolute GPFR numbers themselves shift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import gene_program_flip_rate

PROJECT_ROOT = Path(__file__).resolve().parent.parent
Z_THRESH_SWEEP = [0.5, 0.75, 1.0, 1.5, 2.0]
DEFAULT_Z_THRESH = 1.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-scores", required=True, help="path to a *_raw_scores.csv.gz file")
    parser.add_argument("--out", default=None, help="output CSV path (default: alongside --raw-scores)")
    args = parser.parse_args()

    raw = pd.read_csv(args.raw_scores)
    groups = raw.groupby(["encoder", "organ", "perturbation", "program"])
    print(f"[setup] {len(groups)} (encoder, organ, perturbation, program) groups, "
          f"z_thresh sweep = {Z_THRESH_SWEEP}", flush=True)

    rows = []
    for (encoder, organ, perturbation, program), g in groups:
        g = g.sort_values("sample_index")
        clean = g["clean_score"].values
        pert = g["pert_score"].values
        row = {"encoder": encoder, "organ": organ, "perturbation": perturbation, "program": program}
        for z in Z_THRESH_SWEEP:
            gpfr = gene_program_flip_rate(clean, pert, method="per_sample_relative", z_thresh=z)
            row[f"gpfr_z{z}"] = gpfr
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else Path(args.raw_scores).with_name(
        Path(args.raw_scores).name.replace("_raw_scores.csv.gz", "_z_thresh_sweep.csv"))
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(out_df)} rows to {out_path}")

    default_col = f"gpfr_z{DEFAULT_Z_THRESH}"
    valid = out_df.dropna(subset=[default_col])
    print(f"\nSpearman rank correlation of each z_thresh's GPFR ranking against "
          f"the default (z_thresh={DEFAULT_Z_THRESH}) ranking, across "
          f"{len(valid)} (organ, perturbation, program) triples:")
    for z in Z_THRESH_SWEEP:
        col = f"gpfr_z{z}"
        rho, p = spearmanr(valid[default_col], valid[col])
        flag = "" if z == DEFAULT_Z_THRESH else (
            "  <-- ranking diverges meaningfully from default, treat default-threshold conclusions with caution"
            if rho < 0.8 else "")
        print(f"  z_thresh={z}: rho={rho:.4f} (p={p:.2e}){flag}")


if __name__ == "__main__":
    main()
