#!/bin/bash
set -x
cd "$(dirname "$0")/src"
source ../.venv/bin/activate
python3 leave_one_slide_out_validity.py --encoder conch
python3 leave_one_slide_out_validity.py --encoder uni
echo LOSO_COMPLETE
