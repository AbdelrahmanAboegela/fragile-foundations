"""Generate the two figures the paper actually needs, distinct from
generate_example_figures.py's full 8-row verification grids (too dense for
a paper figure — this script makes a paper-sized crop plus a new bar chart
that directly visualizes the core encoder-divergence finding).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT_DIR = PROJECT_ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def crop_example_rows(src_name: str, out_name: str, n_rows: int = 2):
    """generate_example_figures.py's grids stack 8 patch-rows, each with a
    thin gray separator + label strip. Crop to the first n_rows for a
    paper-sized figure, reusing the exact same images (no regeneration, so
    the paper figure and the verification grid are guaranteed consistent)."""
    src = Image.open(PROJECT_ROOT / "results" / "figures" / src_name)
    w, h = src.size
    row_h = h // 8  # generate_example_figures.py uses N_SAMPLE_PATCHES=8 equal-height rows incl. separators
    crop = src.crop((0, 0, w, row_h * n_rows))
    crop.save(OUT_DIR / out_name)
    print(f"Wrote {OUT_DIR / out_name} ({crop.width}x{crop.height}, cropped from {src_name})")


def gpfr_divergence_bar_chart():
    """ERROR BARS ADDED (2026-09-17, third-adversarial-review-caught): this
    used to plot bare point estimates while the Results text spends a
    paragraph insisting the CI, not the point estimate, is what should be
    read for precision (4-slide cluster bootstrap CIs are wide -- e.g.
    UNI's largest bar has a 95% CI of [0.093, 0.640]). Plotting the bar
    without its CI made the headline figure look far more decisive than
    the underlying data actually is."""
    conch = pd.read_csv(PROJECT_ROOT / "results" / "idc_core_with_stats.csv")
    uni = pd.read_csv(PROJECT_ROOT / "results" / "idc_core_uni_with_stats.csv")

    # Max GPFR per perturbation (across the 5 programs) — the "how bad does
    # this perturbation get" summary, one bar per perturbation per encoder.
    # Use the CI of the specific (perturbation, program) ROW that attains
    # the max, not some other aggregate -- that's the row the bar height
    # actually represents.
    conch_max_row = conch.loc[conch.groupby("perturbation")["gpfr"].idxmax()].set_index("perturbation")
    uni_max_row = uni.loc[uni.groupby("perturbation")["gpfr"].idxmax()].set_index("perturbation")
    perturbations = sorted(set(conch_max_row.index) | set(uni_max_row.index),
                            key=lambda p: -max(conch_max_row["gpfr"].get(p, 0), uni_max_row["gpfr"].get(p, 0)))

    def err(df, perts):
        vals = df["gpfr"].reindex(perts).fillna(0)
        lo = (vals - df["ci_lo"].reindex(perts).fillna(0)).clip(lower=0)
        hi = (df["ci_hi"].reindex(perts).fillna(0) - vals).clip(lower=0)
        return np.array([lo.values, hi.values])

    x = np.arange(len(perturbations))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, conch_max_row["gpfr"].reindex(perturbations).fillna(0), width, label="CONCH", color="#4C72B0",
           yerr=err(conch_max_row, perturbations), capsize=3, ecolor="black", error_kw={"elinewidth": 1})
    ax.bar(x + width/2, uni_max_row["gpfr"].reindex(perturbations).fillna(0), width, label="UNI", color="#DD8452",
           yerr=err(uni_max_row, perturbations), capsize=3, ecolor="black", error_kw={"elinewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in perturbations], fontsize=9)
    ax.set_ylabel("Max GPFR across 5 programs")
    ax.set_title("Peak fragility per perturbation diverges by encoder")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = OUT_DIR / "gpfr_encoder_divergence.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    crop_example_rows("perturbation_suite_comparison.png", "perturbation_suite_example.png", n_rows=2)
    crop_example_rows("stain_shift_comparison.png", "stain_shift_example.png", n_rows=2)
    gpfr_divergence_bar_chart()
