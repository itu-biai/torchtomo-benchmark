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
