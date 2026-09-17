#!/bin/bash
# Runs the core GPFR pipeline on COAD (Xenium, 4 samples, same platform as
# IDC) as a second cohort, testing whether the CONCH/UNI fragility
# divergence found on IDC replicates on an independent organ. Scoped to
# the analyses that bear directly on that question (main experiment x 4
# seeds x 2 encoders, floor baselines, leave-one-slide-out, cluster-
# bootstrap stats) -- not the IDC-specific dose-response sweeps, which are
# a separate, secondary line of evidence. Requires COAD's HEST-Bench data
# at data/hest-bench/COAD/ (see README's "Reproducing the results").
set -x
cd "$(dirname "$0")/src"
source ../.venv/bin/activate

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== Step 1/5: main_experiment.py, CONCH, 4 seeds ==="
for seed in 42 7 123 2024; do
  log "-- CONCH seed=$seed --"
  python3 main_experiment.py --organs COAD --n-patches 3000 --encoder conch --seed $seed
done

log "=== Step 2/5: main_experiment.py, UNI, 4 seeds ==="
for seed in 42 7 123 2024; do
  log "-- UNI seed=$seed --"
  python3 main_experiment.py --organs COAD --n-patches 3000 --encoder uni --seed $seed
done

log "=== Step 3/5: floor baselines ==="
python3 baseline_models.py --organ COAD

log "=== Step 4/5: leave-one-slide-out validity (both encoders) ==="
python3 leave_one_slide_out_validity.py --organ COAD --encoder conch
python3 leave_one_slide_out_validity.py --organ COAD --encoder uni

log "=== Step 5/5: cluster-bootstrap + FDR statistics (both encoders) ==="
python3 analyze_statistics.py --raw-scores ../results/main_experiment_results_coad_raw_scores.csv.gz --out ../results/coad_core_with_stats.csv
python3 analyze_statistics.py --raw-scores ../results/main_experiment_results_coad_uni_raw_scores.csv.gz --out ../results/coad_core_uni_with_stats.csv

log COAD_PIPELINE_COMPLETE
