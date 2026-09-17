# Predicted Gene-Program Activations from Histology Are Fragile to Routine Image Perturbations, and the Failure Mode Depends on the Encoder

Code, configuration, and results underlying the paper of the same name. The
study measures whether Hallmark gene-program activations predicted from
H&E histology (via frozen CONCH/UNI foundation models and a RidgeCV linear
probe) are stable under routine, non-adversarial image perturbations —
JPEG compression, brightness/contrast/gamma shift, blur, scanner-color
shift, and H&E stain variation — using a Gene-Program Flip Rate (GPFR)
instability metric on the IDC (breast cancer, Xenium) cohort of
[HEST-Bench](https://github.com/mahmoodlab/HEST).

The manuscript is included at [`paper/main.pdf`](paper/main.pdf) (source:
[`paper/main.tex`](paper/main.tex)), with supplementary tables at
[`paper/supplementary.pdf`](paper/supplementary.pdf); every number in either
traces to a script in `src/` and a file in `results/`.

## Repository layout

```
src/        Analysis pipeline (perturbations, calibration, model fitting,
            statistical testing, table/figure generation)
results/    All results CSVs and figures referenced by the paper's tables
            and figures
configs/    Per-organ perturbation calibration constants
paper/      Manuscript source (main.tex, references.bib), generated
            tables/figures, and the compiled PDF
rerun_all.sh   Reruns the full pipeline (both encoders, all seeds, all
               sweeps) and regenerates every table and figure
run_loso.sh    Runs the leave-one-slide-out validity check
```

## Reproducing the results

1. Install dependencies: `pip install -r requirements.txt`, plus
   PyTorch/CUDA and HEST-Bench per the notes at the bottom of
   `requirements.txt`.
2. Download the IDC cohort of HEST-Bench (patches + Xenium expression) into
   `data/hest-bench/IDC/` following HEST-Bench's own instructions.
3. Run `./rerun_all.sh` to reproduce the main experiment (both encoders,
   4 seeds each, the JPEG and stain-shift magnitude sweeps, the floor
   baselines, the statistical tests, and every table/figure).
4. Run `./run_loso.sh` for the leave-one-slide-out validity check.

Individual analysis steps can also be run directly, e.g.:

```
python3 src/main_experiment.py --organs IDC --n-patches 3000 --encoder conch --seed 42
python3 src/analyze_statistics.py --raw-scores results/main_experiment_results_idc_raw_scores.csv.gz --out results/idc_core_with_stats.csv
python3 src/generate_paper_tables.py
python3 src/generate_paper_figures.py
```

## Data

Patch and expression data are from [HEST-Bench](https://github.com/mahmoodlab/HEST)
(CC BY-NC-SA 4.0); the IDC samples used here are publicly released 10x
Genomics Xenium breast cancer datasets. CONCH and UNI checkpoints are
obtained from their respective gated HuggingFace repositories under each
model's own license terms.

## Citation

If you use this code or these results, please cite the paper (full
citation to be added on publication).

## License

Code in this repository is released under the MIT License (see
[`LICENSE`](LICENSE)). This does not extend to third-party data or model
checkpoints, which remain under their own respective licenses.
