#!/bin/sh
# Local CPU evaluation of the 2026-10-20 batch (after the caches are copied to results/v4/cache).
# Order: fast groups for every cache, then DP-FedCP (slow), then the calibration-fraction runs,
# then the 100-draw repeats, then the report. Each v3_eval call skips nothing (no cache of
# eval results), so run this once; rerun single steps by hand if one fails.
cd "$(dirname "$0")/../.."
C=results/v4/cache
E=results/v4/eval
W=${WORKERS:-8}
mkdir -p $E results/v4/logs
log() { echo "[$(date +%H:%M)] $*" | tee -a results/v4/logs/eval_v4.log; }

main_tags=$(ls $C | grep -E "_K10_dir0.5_(la|b)_seed" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')
iid_tags=$(ls $C | grep -E "_dir100_la_seed" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')
scale_tags=$(ls $C | grep -E "_K(50|100)_dir0.5_la_seed" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')
if100_tags=$(ls $C | grep -E "_if100_seed" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')

log "1/6 main configs: refs, pcfcp, qq, variants, ablations, alpha, dpfedcp23"
python v3_eval.py --cache $C --out $E --workers $W --tags "$main_tags" --groups refs,pcfcp,qq,variants,ablations,alpha,dpfedcp23
log "2/6 near-IID and client-scale configs"
python v3_eval.py --cache $C --out $E --workers $W --tags "$iid_tags,$scale_tags" --groups refs,pcfcp,qq,variants,a2,dpfedcp23
log "3/6 CIFAR-100-LT at IF = 100"
python v3_eval.py --cache $C --out $E --workers $W --tags "$if100_tags" --groups refs,pcfcp,qq,variants
log "4/6 DP-FedCP main group (eps 1, 4, 8, sigma_g 100, no noise) on every config"
python v3_eval.py --cache $C --out $E --workers $W --tags "$main_tags,$iid_tags,$scale_tags,$if100_tags" --groups dpfedcp
log "5/6 calibration-set fractions (10 and 20 percent of the pool)"
for f in 0.2 0.4; do
  python v3_eval.py --cache $C --out $E --workers $W --tags "$main_tags" --groups refs,pcfcp,qq,variants --cal-frac $f
  python v3_eval.py --cache $C --out $E --workers $W --tags "$main_tags" --groups dpfedcp --cal-frac $f
done
log "6/6 100 noise draws per seed for the box plots (separate dir)"
python v3_eval.py --cache $C --out results/v4/eval_r100 --workers $W --tags "$main_tags" --groups pcfcp,qq,variants --r-fast 100
log "report"
python analyze_results_v3.py --eval $E --acc results/v4/accuracy.csv --cache $C --out results/v4/RESULTS_V4.md
log "done"
