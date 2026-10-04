#!/usr/bin/env bash
# Every ISBI geometry rerun on torchtomo 0.4.0, stamped. Run from the repository root:
#   THIES=~/lab/ext/geometry_gradients_CT bash isbi/run_reruns.sh
set -euo pipefail
PYTHON=${PYTHON:-.venv/bin/python}
THIES=${THIES:-$HOME/lab/ext/geometry_gradients_CT}
OUT=isbi/results
mkdir -p "$OUT/raw" "$OUT/logs"

for session in 1 2 3; do
  cmd="$PYTHON libraries/compare_geometry_gradients.py --thies $THIES --repeats 20 --backends auto torch --output $OUT/raw/geometry_gradients_s${session}.json"
  $cmd 2>&1 | tee "$OUT/logs/geometry_gradients_s${session}.log"
  $PYTHON isbi/provenance.py "$OUT/raw/geometry_gradients_s${session}.json" "$OUT/geometry_gradients_s${session}.json" "$cmd"
done

$PYTHON isbi/gradient_accuracy.py 2>&1 | tee "$OUT/logs/gradient_accuracy.log"

$PYTHON geometry/calibrate_real.py 2>&1 | tee "$OUT/logs/calibrate.log"
$PYTHON isbi/provenance.py geometry/results/cor.json "$OUT/cor.json" "$PYTHON geometry/calibrate_real.py"

$PYTHON geometry/correct_motion.py 2>&1 | tee "$OUT/logs/motion.log"
$PYTHON isbi/provenance.py geometry/results/motion.json "$OUT/motion.json" "$PYTHON geometry/correct_motion.py"

$PYTHON isbi/cor_stats.py | tee "$OUT/logs/cor_stats.log"
$PYTHON isbi/motion_stats.py | tee "$OUT/logs/motion_stats.log"
$PYTHON isbi/make_fig2_calibration.py
$PYTHON isbi/make_fig3_motion.py
$PYTHON isbi/export_tables.py | tee "$OUT/logs/export_tables.log"

# Extras: gradient accuracy at smaller sizes and the cross-library dot test. The dot test
# needs ASTRA and torch-radon, which live in torchtomo's own venv (DOT_PYTHON).
for size in 64 128; do
  $PYTHON isbi/gradient_accuracy.py --size $size --views $size --name gradient_accuracy_size$size \
    2>&1 | tee "$OUT/logs/gradient_accuracy_size$size.log"
done
DOT_PYTHON=${DOT_PYTHON:-../torchtomo/.venv/bin/python}
$DOT_PYTHON isbi/dot_test.py 2>&1 | tee "$OUT/logs/dot_test.log"
