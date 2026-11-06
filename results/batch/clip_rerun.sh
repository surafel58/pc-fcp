#!/bin/sh
# Test-score clipping fix (2026-11-04, proof review): exact re-runs, with the fixed v3_eval.py, of the
# result files whose thresholds near 1 cannot be corrected from the stored values alone:
#   - the ablations group on the 30 main models (the A1 arms average noisy client thresholds, and
#     the p3_B100 grid has an edge at 0.999999, so a printed 1.00000 is ambiguous there);
#   - the qq and alpha files in which the non-private FedCP-QQ per-class threshold prints as 1.00000.
# The dpfedcp_pc_nonoise rows are recomputed one row at a time by results/batch/clip_dpfedcp_rows.py,
# and every other affected row is corrected exactly by results/batch/clip_correct.py.
cd "$(dirname "$0")/../.."
C=results/v5/cache; E=results/v5/eval
W=${WORKERS:-8}
LOG=results/v5/logs/clip_rerun.log
mkdir -p results/v5/logs
log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
main_tags=$(ls $C | grep -E "_K10_dir0.5_(la|b)_seed" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')
log "ablations on the 30 main models"
python v3_eval.py --cache $C --out $E --workers $W --tags "$main_tags" --groups ablations >> $LOG 2>&1
log "qq / alpha files with an ambiguous FedCP-QQ threshold"
python v3_eval.py --cache $C --out $E --workers $W --tags cifar10_K10_dir0.5_la_seed42 --groups qq --cal-frac 0.2 >> $LOG 2>&1
python v3_eval.py --cache $C --out $E --workers $W --tags text_K10_dir0.5_b_seed48 --groups qq --cal-frac 0.2 >> $LOG 2>&1
python v3_eval.py --cache $C --out $E --workers $W --tags text_K10_dir0.5_b_seed47,text_K10_dir0.5_b_seed51 --groups qq --cal-frac 0.4 >> $LOG 2>&1
ALPHA_TAGS=$(cat results/v5/logs/clip_alpha_tags.txt)
python v3_eval.py --cache $C --out $E --workers $W --tags "$ALPHA_TAGS" --groups alpha >> $LOG 2>&1
log "done"
