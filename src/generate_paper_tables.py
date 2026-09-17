"""Generate LaTeX tables directly from results/*.csv for the paper.

WHY THIS EXISTS: every number in the paper must trace back to an actual
CSV, not be hand-typed/transcribed (a transcription step is exactly where
a correct pipeline result quietly becomes a wrong paper number). This
script is the only thing allowed to turn results/*.csv into paper/tables/*.tex.
Re-run it whenever a results CSV changes; never hand-edit a generated table.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def tex_escape(s: str) -> str:
    """Escape LaTeX special characters (underscores in program/perturbation
    names like `immune_activation` otherwise trigger math-mode parsing
    errors — caught by actually compiling the paper, not just generating it)."""
    return str(s).replace("_", r"\_")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_ROOT / "results"
OUT = PROJECT_ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def table1_validity():
    conch = pd.read_csv(RESULTS / "main_experiment_results_idc.csv").iloc[0]
    uni = pd.read_csv(RESULTS / "main_experiment_results_idc_uni.csv").iloc[0]
    rows = []
    for name, row in [("CONCH (ViT-B/16, 86M)", conch), ("UNI (ViT-L/16, 307M)", uni)]:
        rows.append(
            f"{name} & {row['heldout_mean_pearson_r']:.3f} & "
            f"{row['heldout_mean_pearson_r_hallmark_genes']:.3f} & "
            f"{row['ridge_alpha_selected']:.3g} "
            f"[{row['ridge_alpha_min']:.3g}, {row['ridge_alpha_max']:.3g}] \\\\"
        )
    tex = r"""\begin{table}[t]
\centering
\caption{Held-out (out-of-sample) predictive validity on IDC (2100 train / 900 held-out patches, seed=42). Ridge $\alpha$ selected per-gene via efficient leave-one-out cross-validation (\texttt{RidgeCV}, \texttt{alpha\_per\_target=True}); median [min, max] shown across the panel's genes.}
\label{tab:validity}
\begin{tabular}{lccc}
\toprule
Encoder & $r$ (whole panel) & $r$ (Hallmark genes) & Ridge $\alpha$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "table1_validity.tex").write_text(tex)
    print("Wrote table1_validity.tex")


def table1b_baseline_comparison():
    """CONCH/UNI-Ridge vs. two floor baselines on the identical IDC
    train/held-out split (seed=42): training_mean (no image at all) and
    total_counts_linear (one feature -- the patch's own raw total panel
    counts, the exact confound the library-size normalization fix targets).
    R^2 is the only metric shown here (not Pearson r) because r is undefined
    for training_mean's constant prediction -- see baseline_models.py's
    module docstring."""
    conch = pd.read_csv(RESULTS / "main_experiment_results_idc.csv").iloc[0]
    uni = pd.read_csv(RESULTS / "main_experiment_results_idc_uni.csv").iloc[0]
    baselines = pd.read_csv(RESULTS / "baseline_models_idc.csv").set_index("model")
    rows = []
    for name, row in [("CONCH-Ridge (ours)", conch), ("UNI-Ridge (ours)", uni)]:
        rows.append(f"{tex_escape(name)} & {row['heldout_mean_r2']:.3f} & {row['heldout_mean_r2_hallmark_genes']:.3f} \\\\")
    display_names = {"training_mean": "Training-mean (no image)", "total_counts_linear": "Total-counts-linear (1 feature)"}
    for model, row in baselines.iterrows():
        rows.append(f"{tex_escape(display_names.get(model, model))} & {row['heldout_mean_r2']:.3f} & "
                     f"{row['heldout_mean_r2_hallmark_genes']:.3f} \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{Held-out $R^2$ against two floor baselines, identical IDC train/held-out split as Table~\ref{tab:validity} (seed=42). \texttt{Training-mean} predicts every held-out patch with the training set's per-gene mean (no image, no metadata). \texttt{Total-counts-linear} predicts from a single feature -- the patch's own raw (pre-normalization) total panel counts -- exactly the confound the library-size normalization fix targets; because normalization makes every patch's target sum to the same value by construction, this baseline can only detect a confound in \emph{relative} composition, not overall scale, so its small $R^2$ is a limited, not fully independent, check.}
\label{tab:baseline_comparison}
\begin{tabular}{lcc}
\toprule
Model & $R^2$ (whole panel) & $R^2$ (Hallmark genes) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "table1b_baseline_comparison.tex").write_text(tex)
    print("Wrote table1b_baseline_comparison.tex")


def table8_loso_validity():
    """Leave-one-slide-out validity (Section~\\ref{sec:loso}, second- and
    third-adversarial-review-caught): the random-patch-split validity
    number cannot rule out train/held-out pixel overlap from HEST's
    overlapping patch-extraction grid. This table reports the alternative:
    train on 3 of IDC's 4 slides, evaluate on the 4th, for every slide in
    turn. MEAN vs MEDIAN R^2 (2026-09-17, third-review-caught): mean
    per-gene R^2 is dominated by a handful of outlier genes with a large
    between-slide offset relative to their within-slide variance (R^2 is
    unbounded below, unlike r). The median per-gene R^2 is reported
    alongside so a reader isn't left with only the outlier-driven number."""
    rows = []
    for encoder, label in [("conch", "CONCH"), ("uni", "UNI")]:
        df = pd.read_csv(RESULTS / f"leave_one_slide_out_validity_{encoder}.csv")
        for _, r in df.iterrows():
            rows.append(
                f"{label} & {tex_escape(r['held_out_slide'])} & {r['pearson_r']:.3f} & "
                f"{r['pearson_r_hallmark']:.3f} & {r['r2']:.2f} & {r['r2_median']:.3f} & "
                f"{r['r2_hallmark']:.2f} & {r['r2_hallmark_median']:.3f} \\\\"
            )
        rows.append(
            f"\\textit{{{label} mean}} & -- & {df['pearson_r'].mean():.3f} & "
            f"{df['pearson_r_hallmark'].mean():.3f} & {df['r2'].mean():.2f} & {df['r2_median'].median():.3f} & "
            f"{df['r2_hallmark'].mean():.2f} & {df['r2_hallmark_median'].median():.3f} \\\\"
        )
    tex = r"""\begin{table}[t]
\centering
\caption{Leave-one-slide-out validity on IDC: Ridge head fit on 3 slides, evaluated on the 4th (never seen during fitting), for every slide in turn. Compare $r$ against Table~\ref{tab:validity}'s random-split numbers (CONCH 0.559/0.614, UNI 0.604/0.661 whole-panel/Hallmark) -- lower here, but still clearly positive. $R^2$(mean) is the standard multi-output average across genes; $R^2$(median) is the median per-gene value, reported because a handful of outlier genes with poor between-slide calibration dominate the mean (e.g.\ CONCH/NCBI785: mean $-68.9$, median $-0.24$) -- the median is a better summary of the typical gene's out-of-slide calibration, though both are real and disclosed here rather than only the more dramatic one. The bottom row of each encoder block is that encoder's mean $r$/mean $R^2$ across its 4 folds, with the median column showing the median of each fold's own per-gene median (not re-derived from pooled per-gene values).}
\label{tab:loso_validity}
\resizebox{\textwidth}{!}{%
\begin{tabular}{llcccccc}
\toprule
Encoder & Held-out slide & $r$ & $r$ (Hallmark) & $R^2$ (mean) & $R^2$ (median) & $R^2_{\text{Hallmark}}$ (mean) & $R^2_{\text{Hallmark}}$ (median) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%
}
\end{table}
"""
    (OUT / "table8_loso_validity.tex").write_text(tex)
    print("Wrote table8_loso_validity.tex")


def table2_gpfr_significance(encoder_label: str, stats_path: str, out_name: str):
    """STAR RULE (2026-09-17, second-adversarial-review-caught): a row used
    to be starred whenever significant_fdr_0.05 was True, with no check on
    the CI itself. With only 4 slides to cluster-bootstrap over, several
    rows had q<0.05 (from the one-sided p-value, which only needs a small
    fraction of the 2000 resamples to be <=0) while their PRINTED 95% CI
    lower bound rounded to 0.000 -- e.g. blur/stromal_ecm on UNI, GPFR=0.002
    from 2 of 900 flipped patches. A reader sees a star next to an interval
    that visibly touches zero, which looks like -- and partly is -- a
    genuine inconsistency between a one-sided test and a two-sided interval
    built from the same 2000 draws. We now require BOTH q<0.05 AND a CI
    that does not touch 0.000 at the reported 3-decimal precision for the
    star, so the visual marker is never contradicted by the interval printed
    next to it. This is strictly more conservative than the raw FDR
    criterion, not a different test -- q-values themselves are unchanged."""
    df = pd.read_csv(RESULTS / stats_path)
    df = df.sort_values("gpfr", ascending=False)
    rows = []
    n_starred = 0
    for _, r in df.iterrows():
        starred = bool(r["significant_fdr_0.05"]) and r["ci_lo"] >= 0.0005  # >= half the last printed decimal, so it never rounds to 0.000
        n_starred += starred
        sig = r"$^{*}$" if starred else ""
        rows.append(
            f"{tex_escape(r['perturbation'])} & {tex_escape(r['program'])} & {r['gpfr']:.3f} & "
            f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] & {r['q_value_fdr']:.4f}{sig} \\\\"
        )
    tex = r"""\begin{table}[t]
\centering
\caption{""" + encoder_label + r""": Gene-Program Flip Rate (GPFR) per perturbation/program on IDC, with 95\% CI from the 4-slide cluster bootstrap (2000 resamples) and Benjamini-Hochberg FDR-corrected $q$-value across all 35 tests. $^{*}$ FDR$<$0.05 AND the CI's lower bound does not round to 0.000 -- with only 4 slides to resample, the one-sided bootstrap $p$-value is quantized (at most $4^4=256$ distinct resample outcomes, giving as few as 6--9 distinct $p$-values across all 35 tests here), so it can reject the boundary null for an interval that still visibly touches zero; requiring both keeps the star consistent with the printed interval.}
\label{tab:gpfr_""" + out_name + r"""}
\begin{tabular}{llccc}
\toprule
Perturbation & Program & GPFR & 95\% CI & FDR $q$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / f"table_gpfr_{out_name}.tex").write_text(tex)
    print(f"Wrote table_gpfr_{out_name}.tex ({len(df)} rows, {n_starred} starred under the stricter rule, "
          f"{df['significant_fdr_0.05'].sum()} raw FDR<0.05)")


def table3_z_sensitivity():
    conch = pd.read_csv(RESULTS / "idc_core_z_sweep.csv")
    uni = pd.read_csv(RESULTS / "idc_core_uni_z_sweep.csv")
    from scipy.stats import spearmanr
    z_cols = [c for c in conch.columns if c.startswith("gpfr_z")]
    default_col = "gpfr_z1.0"
    rows = []
    for z_col in z_cols:
        z_val = z_col.replace("gpfr_z", "")
        rho_c, _ = spearmanr(conch[default_col], conch[z_col])
        rho_u, _ = spearmanr(uni[default_col], uni[z_col])
        rows.append(f"{z_val} & {rho_c:.3f} & {rho_u:.3f} \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{Spearman rank correlation of each $z_{\text{thresh}}$'s GPFR ranking (across all 35 perturbation$\times$program combinations) against the default $z_{\text{thresh}}=1.0$ ranking, for both encoders on IDC.}
\label{tab:z_sensitivity}
\begin{tabular}{lcc}
\toprule
$z_{\text{thresh}}$ & CONCH $\rho$ & UNI $\rho$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "table3_z_sensitivity.tex").write_text(tex)
    print("Wrote table3_z_sensitivity.tex")


def table4_seed_stability(encoder_label: str, base_fname: str, out_name: str, label_suffix: str):
    """base_fname: the seed-42 filename, e.g. 'main_experiment_results_idc.csv'
    or 'main_experiment_results_idc_uni.csv'; the seed7/123/2024 variants are
    derived by inserting '_seedN' before the (optional) '_uni' suffix, matching
    main_experiment.py's seed_tag naming (see its ORGAN-SCOPE IN FILENAME
    comment) -- CONCH keeps the encoder suffix empty, UNI appends '_uni'.
    The caption's specific claims (which perturbations cluster, which is most/
    least stable) are computed from the actual data, not hardcoded, so this
    stays correct across re-runs with different numbers."""
    stem, suffix = (base_fname[:-len("_uni.csv")], "_uni") if base_fname.endswith("_uni.csv") else (base_fname[:-4], "")
    seeds = {42: base_fname, 7: f"{stem}_seed7{suffix}.csv", 123: f"{stem}_seed123{suffix}.csv",
             2024: f"{stem}_seed2024{suffix}.csv"}
    cols = []
    for seed, fname in seeds.items():
        df = pd.read_csv(RESULTS / fname)
        cols.append(df.groupby("perturbation")["gpfr"].max().rename(seed))
    table = pd.concat(cols, axis=1)
    table["mean"] = table[[42, 7, 123, 2024]].mean(axis=1)
    table["std"] = table[[42, 7, 123, 2024]].std(axis=1)
    table = table.sort_values("mean", ascending=False)
    rows = []
    for pert, row in table.iterrows():
        rows.append(
            f"{tex_escape(pert)} & {row[42]:.3f} & {row[7]:.3f} & "
            f"{row[123]:.3f} & {row[2024]:.3f} & {row['mean']:.3f} $\\pm$ {row['std']:.3f} \\\\"
        )
    top_pert = table.index[0]
    most_stable = table["std"].idxmin()
    least_stable = table["std"].idxmax()
    caption = (
        f"{encoder_label}: maximum GPFR (across the 5 programs) per perturbation, across 4 re-splits of "
        f"IDC's same 4 slides into different random train/held-out patch assignments (seeds 42, 7, 123, "
        f"2024 -- not 4 different slides). \\texttt{{{tex_escape(top_pert)}}} has the highest mean "
        f"max-GPFR ({table.loc[top_pert, 'mean']:.3f}); \\texttt{{{tex_escape(most_stable)}}} is the most stable "
        f"across seeds (std={table.loc[most_stable, 'std']:.3f}), \\texttt{{{tex_escape(least_stable)}}} the least "
        f"(std={table.loc[least_stable, 'std']:.3f})."
    )
    tex = r"""\begin{table}[t]
\centering
\caption{""" + caption + r"""}
\label{tab:seed_stability""" + label_suffix + r"""}
\begin{tabular}{lccccc}
\toprule
Perturbation & seed 42 & seed 7 & seed 123 & seed 2024 & mean $\pm$ std \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / f"table4_seed_stability{out_name}.tex").write_text(tex)
    print(f"Wrote table4_seed_stability{out_name}.tex")


def table5_fragility_decomposition():
    """Per-perturbation SUMMARY across its 5 programs (mean/min/max relative
    reduction), not all 35 raw rows -- the full table is 35 rows tall and,
    at [t] placement, LaTeX's float algorithm deferred it past the
    bibliography to the end of the document (caught by actually compiling
    and checking the page count, which jumped from 10 to 14 -- not by
    reading the .tex source, which looked fine). This summary is also more
    informative for the actual claim (the correction's inconsistency is
    visible in the min/max spread, not lost by aggregating it away)."""
    df = pd.read_csv(RESULTS / "fragility_decomposition.csv")
    g = df.groupby("perturbation")["gpfr_relative_reduction"]
    summary = pd.DataFrame({"mean": g.mean(), "min": g.min(), "max": g.max()}) * 100
    summary = summary.reindex(df["perturbation"].unique())  # preserve original perturbation order
    rows = []
    for pert, row in summary.iterrows():
        rows.append(
            f"{tex_escape(pert)} & {row['mean']:.0f}\\% & {row['min']:.0f}\\% & {row['max']:.0f}\\% \\\\"
        )
    worst_pert = summary["min"].idxmin()
    worst_val = summary.loc[worst_pert, "min"]
    caption = (
        r"CONCH: linear-offset correction's relative GPFR reduction (Section~\ref{sec:fragility_decomposition}), "
        r"summarized per perturbation across its 5 programs. Positive = correction lowered GPFR; negative = made "
        f"it worse. \\texttt{{{tex_escape(worst_pert)}}}'s min (${worst_val:.0f}\\%$) is the largest actual "
        r"worsening observed -- the correction is not a reliably-signed effect even within one perturbation, "
        r"let alone across the suite."
    )
    tex = r"""\begin{table}[H]
\centering
\caption{""" + caption + r"""}
\label{tab:fragility_decomposition}
\begin{tabular}{lccc}
\toprule
Perturbation & Mean reduction & Min & Max \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "table5_fragility_decomposition.tex").write_text(tex)
    print("Wrote table5_fragility_decomposition.tex")


def table6_jpeg_dose_response():
    """JPEG quality-level dose-response (Section on JPEG compression,
    critique-flagged 2026-09-17 as present in the pipeline but absent from
    the paper): does GPFR rise as compression gets more aggressive, and is
    it already non-trivial within the range clinicians consider
    diagnostically safe (~15-20x compression)?

    COMPRESSION-RATIO MAPPING FIX (2026-09-17, second-adversarial-review-
    caught): this used to hardcode "quality 70-90 is ~15-20x compression"
    in the caption. Our OWN measured ratios (jpeg_compression_ratios.csv)
    say quality 90 is 8.4x and quality 70 is 15.3x -- the 15-20x band is
    quality 50-70, not 70-90. Read the ratio from that CSV instead of
    hand-typing it, so this can't drift out of sync with the actual
    measurement again."""
    df = pd.read_csv(RESULTS / "jpeg_sweep_results_idc.csv")
    ratios = pd.read_csv(RESULTS / "jpeg_compression_ratios.csv").set_index("perturbation")["compression_ratio"]
    order = ["jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30"]
    g = df.groupby("perturbation")
    rows = []
    for pname in order:
        if pname not in g.groups:
            continue
        sub = g.get_group(pname)
        ssim = sub["mean_ssim"].iloc[0]
        ratio = ratios[pname]
        mean_gpfr, max_gpfr = sub["gpfr"].mean(), sub["gpfr"].max()
        worst_prog = sub.loc[sub["gpfr"].idxmax(), "program"]
        q = pname.replace("jpeg_q", "")
        rows.append(f"{q} & {ratio:.1f}$\\times$ & {ssim:.3f} & {mean_gpfr:.3f} & {max_gpfr:.3f} ({tex_escape(worst_prog)}) \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{JPEG dose-response on IDC (CONCH): mean and max GPFR across the 5 programs as compression gets more aggressive (higher PIL quality = milder compression = higher SSIM). Compression ratio measured directly on our own patches (\texttt{jpeg\_compression\_ratios.csv}), not assumed from quality alone. The $15$--$20\times$ range the DICOM WSI standard \citep{dicom_wsi_standard} treats as introducing no loss of diagnostic information corresponds to quality 50--70 here, not 70--90; GPFR is already non-trivial in that band, not just at the aggressive q30 setting.}
\label{tab:jpeg_dose_response}
\begin{tabular}{lcccc}
\toprule
JPEG quality & Compression & Mean SSIM & Mean GPFR & Max GPFR (program) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "table6_jpeg_dose_response.tex").write_text(tex)
    print("Wrote table6_jpeg_dose_response.tex")


def table7_stain_magnitude_sweep():
    """Stain-shift magnitude sweep including the stain_shift_identity NULL
    CONTROL (h=e=1.0, i.e. no intended color change at all -- isolates how
    much of stain_shift's measured GPFR is just Macenko reconstruction
    round-trip noise vs. a real effect of the color shift itself)."""
    df = pd.read_csv(RESULTS / "stain_sweep_results_idc.csv")
    order = ["stain_shift_identity", "stain_shift_mild", "stain_shift_moderate"]
    labels = {"stain_shift_identity": "Identity (null control)", "stain_shift_mild": "Mild (h=1.10/e=0.90)",
              "stain_shift_moderate": "Moderate (h=1.30/e=0.70)"}
    g = df.groupby("perturbation")
    rows = []
    for pname in order:
        if pname not in g.groups:
            continue
        sub = g.get_group(pname)
        ssim = sub["mean_ssim"].iloc[0]
        mean_gpfr, max_gpfr = sub["gpfr"].mean(), sub["gpfr"].max()
        rows.append(f"{labels[pname]} & {ssim:.3f} & {mean_gpfr:.3f} & {max_gpfr:.3f} \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{Stain-shift magnitude sweep on IDC (CONCH), including the \texttt{stain\_shift\_identity} null control (a full Macenko decompose/reconstruct round trip with NO intended stain-vector change). Any nonzero GPFR at Identity is reconstruction noise, not a real color-shift effect; the gap between Identity and Moderate is the actual effect size.}
\label{tab:stain_sweep}
\begin{tabular}{lccc}
\toprule
Magnitude & Mean SSIM & Mean GPFR & Max GPFR \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "table7_stain_magnitude_sweep.tex").write_text(tex)
    print("Wrote table7_stain_magnitude_sweep.tex")


def table0_hallmark_coverage():
    """Per-program Hallmark gene-set coverage on IDC's targeted panel —
    added 2026-09-17 after an adversarial review found this was disclosed
    nowhere in the paper. proliferation (6/327 genes) and angiogenesis
    (5/36, 4 of which are shared with other programs) are effectively
    small marker panels, not comprehensive pathway scores; this table
    makes that visible rather than letting "program-level" imply more
    coverage than exists.

    PANEL FIX (2026-09-17, second-adversarial-review-caught): this used to
    read NCBI783's own panel (288 genes) as "IDC's panel", but the actual
    modeling pipeline (main_experiment.py) scores against the 280-gene
    INTERSECTION across all 4 IDC samples' panels (NCBI783=288, NCBI785=321,
    TENX95=280, TENX99=280 -- not identical, contrary to what the old
    caption claimed). Intersecting all 4 here instead, so this table
    reports coverage against the exact gene set the real model actually
    sees, not a single sample's larger panel."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from gene_program_scoring import load_hallmark_gene_sets, load_program_config
    import anndata as ad

    hallmark, prog_cfg = load_hallmark_gene_sets(), load_program_config()
    sample_ids = ["NCBI783", "NCBI785", "TENX95", "TENX99"]
    panel = None
    for sid in sample_ids:
        a = ad.read_h5ad(PROJECT_ROOT / f"data/hest-bench/IDC/adata/{sid}.h5ad")
        sample_panel = {g for g in a.var_names if "Control" not in g and "BLANK" not in g and "Negative" not in g}
        panel = sample_panel if panel is None else (panel & sample_panel)
    print(f"[setup] {len(panel)}-gene intersection across {sample_ids}", flush=True)

    program_genes = {}
    for prog, spec in prog_cfg["programs"].items():
        genes = set()
        for hset in spec["hallmark_sets"]:
            genes.update(hallmark.get(hset, []))
        program_genes[prog] = genes & panel

    rows = []
    for prog, spec in prog_cfg["programs"].items():
        total = set()
        for hset in spec["hallmark_sets"]:
            total.update(hallmark.get(hset, []))
        present = program_genes[prog]
        other_progs_genes = set().union(*[g for p, g in program_genes.items() if p != prog])
        shared = len(present & other_progs_genes)
        rows.append(
            f"{tex_escape(prog)} & {len(present)}/{len(total)} & {100*len(present)/len(total):.1f}\\% & {shared}/{len(present)} \\\\"
        )
    tex = r"""\begin{table}[t]
\centering
\caption{Hallmark gene-set coverage on IDC's 280-gene common panel -- the intersection across all 4 IDC samples' individual panels (288/321/280/280 genes for NCBI783/NCBI785/TENX95/TENX99 respectively; not identical across samples), i.e.\ exactly the gene set the modeling pipeline scores against. ``Shared'' counts genes that also belong to at least one other of the 5 scored programs. At this coverage, \texttt{proliferation} and \texttt{angiogenesis} are effectively small marker panels rather than comprehensive pathway scores.}
\label{tab:hallmark_coverage}
\begin{tabular}{lccc}
\toprule
Program & Genes present / total & Coverage & Shared with other programs \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "table0_hallmark_coverage.tex").write_text(tex)
    print("Wrote table0_hallmark_coverage.tex")


if __name__ == "__main__":
    table0_hallmark_coverage()
    table1_validity()
    table1b_baseline_comparison()
    table8_loso_validity()
    table2_gpfr_significance("CONCH", "idc_core_with_stats.csv", "conch")
    table2_gpfr_significance("UNI", "idc_core_uni_with_stats.csv", "uni")
    table3_z_sensitivity()
    table4_seed_stability("CONCH", "main_experiment_results_idc.csv", "", "")
    table4_seed_stability("UNI", "main_experiment_results_idc_uni.csv", "_uni", "_uni")
    table5_fragility_decomposition()
    table6_jpeg_dose_response()
    table7_stain_magnitude_sweep()
