#!/bin/sh
# Rerun the two DP-FedCP passes that produced 0 tasks before the v3_eval.py --groups fix
# (step 4 and the cal-frac 0.2 DP-FedCP call of eval_v4.sh), then rebuild the report.
cd "$(dirname "$0")/../.."
C=results/v4/cache; E=results/v4/eval; W=${WORKERS:-8}
log() { echo "[$(date +%H:%M)] $*" | tee -a results/v4/logs/eval_v4.log; }
until grep -q "\] done" results/v4/logs/eval_v4.log; do sleep 60; done
main_tags=$(ls $C | grep -E "_K10_dir0.5_(la|b)_seed" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')
iid_tags=$(ls $C | grep -E "_dir100_la_seed" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')
scale_tags=$(ls $C | grep -E "_K(50|100)_dir0.5_la_seed" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')
if100_tags=$(ls $C | grep -E "_if100_seed" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')
log "followup 4/6 DP-FedCP main group on every config"
python v3_eval.py --cache $C --out $E --workers $W --tags "$main_tags,$iid_tags,$scale_tags,$if100_tags" --groups dpfedcp
log "followup 5/6 DP-FedCP at cal-frac 0.2"
python v3_eval.py --cache $C --out $E --workers $W --tags "$main_tags" --groups dpfedcp --cal-frac 0.2
log "followup report"
python analyze_results_v3.py --eval $E --acc results/v4/accuracy.csv --cache $C --out results/v4/RESULTS_V4.md
log "followup done"
