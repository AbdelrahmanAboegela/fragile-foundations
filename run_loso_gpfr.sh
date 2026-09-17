#!/bin/bash
# Runs the leave-one-slide-out GPFR check (loso_gpfr.py) for both organs
# and both encoders -- the primary GPFR result re-evaluated under the
# same slide-disjoint design used for LOSO predictive validity, closing
# the gap between showing the random-split design admits pixel overlap
# and continuing to use a random-split-trained head for the headline
# robustness result. Requires rerun_all.sh (IDC) and run_coad.sh (COAD)
# data/embeddings to already be reachable via data/hest-bench/.
set -x
cd "$(dirname "$0")/src"
source ../.venv/bin/activate

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== loso_gpfr.py: IDC, both encoders ==="
python3 loso_gpfr.py --organ IDC --encoder conch
python3 loso_gpfr.py --organ IDC --encoder uni

log "=== loso_gpfr.py: COAD, both encoders ==="
python3 loso_gpfr.py --organ COAD --encoder conch
python3 loso_gpfr.py --organ COAD --encoder uni

log "=== cluster bootstrap + FDR statistics ==="
python3 analyze_statistics.py --raw-scores ../results/loso_gpfr_idc_raw_scores.csv.gz --out ../results/loso_gpfr_idc_with_stats.csv
python3 analyze_statistics.py --raw-scores ../results/loso_gpfr_idc_uni_raw_scores.csv.gz --out ../results/loso_gpfr_idc_uni_with_stats.csv
python3 analyze_statistics.py --raw-scores ../results/loso_gpfr_coad_raw_scores.csv.gz --out ../results/loso_gpfr_coad_with_stats.csv
python3 analyze_statistics.py --raw-scores ../results/loso_gpfr_coad_uni_raw_scores.csv.gz --out ../results/loso_gpfr_coad_uni_with_stats.csv

log LOSO_GPFR_COMPLETE
