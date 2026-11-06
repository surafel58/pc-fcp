#!/bin/sh
# Resume of eval_seeds.sh from step 5 (2026-11-03). Step 5 there passed an explicit --groups
# list, which since commit 4d0212b runs as given and so added DP-FedCP at eps 2/3 (dpfedcp23)
# to the 50- and 100-client configs; v4 never ran that group there (its run predates the
# commit and dropped it), so those 7-seed-only files were moved to results/v4_seeds/eval_extra.
# Here step 5 runs without --groups, so jobs_for picks the groups per config exactly as v4
# has them: near-IID refs,pcfcp,qq,variants,dpfedcp23; client scale refs,pcfcp,qq,variants,a2.
# Steps 6-9 are unchanged from eval_seeds.sh.
cd "$(dirname "$0")/../.."
S=results/v4_seeds
C=$S/cache; E=$S/eval
W=${WORKERS:-8}
OUT=$S/logs/eval.out
log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $S/logs/eval_seeds.log; }
tags() { ls $C | grep -E "$1" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//'; }

main_tags=$(tags "_K10_dir0.5_(la|b)_seed")
iid_tags=$(tags "_dir100_la_seed"); scale_tags=$(tags "_K(50|100)_dir0.5_la_seed"); if100_tags=$(tags "_if100_seed")
log "5/9 (resumed) near-IID and client-scale configs, groups per config as in v4"
python v3_eval.py --cache $C --out $E --workers $W --tags "$iid_tags,$scale_tags" >> $OUT 2>&1
log "6/9 CIFAR-100-LT at IF = 100"
python v3_eval.py --cache $C --out $E --workers $W --tags "$if100_tags" --groups refs,pcfcp,qq,variants >> $OUT 2>&1
log "7/9 DP-FedCP main group (eps 1, 4, 8, sigma_g 100, no noise) on every config (slow)"
python v3_eval.py --cache $C --out $E --workers 4 --tags "$main_tags,$iid_tags,$scale_tags,$if100_tags" --groups dpfedcp >> $OUT 2>&1
log "8/9 DP-FedCP at calibration fractions (slow)"
for f in 0.2 0.4; do
  python v3_eval.py --cache $C --out $E --workers 4 --tags "$main_tags" --groups dpfedcp --cal-frac $f >> $OUT 2>&1
done
log "9/9 merge v4 (seeds 42-44) + v4_seeds (45-51) into results/v5"
mkdir -p results/v5/cache results/v5/eval results/v5/eval_r100 results/v5/logs
cp results/v4/cache/*.npz $C/*.npz results/v5/cache/
cp results/v4/eval/*.csv $E/*.csv results/v5/eval/
cp results/v4/eval_r100/*.csv $S/eval_r100/*.csv results/v5/eval_r100/
{ cat results/v4/accuracy.csv; tail -n +2 $S/accuracy.csv; } > results/v5/accuracy.csv
python analyze_results_v3.py --eval results/v5/eval --acc results/v5/accuracy.csv --cache results/v5/cache --out results/v5/RESULTS_V5.md >> $OUT 2>&1
log "done"
