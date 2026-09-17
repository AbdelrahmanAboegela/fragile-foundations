#!/bin/bash
set -x
cd "$(dirname "$0")/src"
source ../.venv/bin/activate
for seed in 42 7 123 2024; do
  python3 main_experiment.py --organs IDC --n-patches 3000 --encoder conch --seed $seed
done
for seed in 42 7 123 2024; do
  python3 main_experiment.py --organs IDC --n-patches 3000 --encoder uni --seed $seed
done
python3 fragility_decomposition.py
python3 main_experiment.py --organs IDC --n-patches 3000 --encoder conch --stain-sweep
python3 main_experiment.py --organs IDC --n-patches 3000 --encoder conch --jpeg-sweep
python3 baseline_models.py
python3 analyze_statistics.py --raw-scores ../results/main_experiment_results_idc_raw_scores.csv.gz --out ../results/idc_core_with_stats.csv
python3 analyze_statistics.py --raw-scores ../results/main_experiment_results_idc_uni_raw_scores.csv.gz --out ../results/idc_core_uni_with_stats.csv
python3 sensitivity_z_thresh.py --raw-scores ../results/main_experiment_results_idc_raw_scores.csv.gz --out ../results/idc_core_z_sweep.csv
python3 sensitivity_z_thresh.py --raw-scores ../results/main_experiment_results_idc_uni_raw_scores.csv.gz --out ../results/idc_core_uni_z_sweep.csv
python3 generate_paper_tables.py
python3 generate_paper_figures.py
echo ALL_RERUNS_COMPLETE
