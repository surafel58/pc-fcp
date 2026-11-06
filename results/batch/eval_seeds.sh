#!/bin/sh
# Local CPU evaluation of the 7 new seeds (examiner comment B5, 2026-11-03); the same protocol as
# results/batch/eval_v4.sh, applied to results/v4_seeds/cache, then merged with v4 into results/v5
# (10 seeds, 42..51) for the thesis tables. Runs alongside copy_seeds.sh.
cd "$(dirname "$0")/../.."
S=results/v4_seeds
C=$S/cache; E=$S/eval
W=${WORKERS:-8}
mkdir -p $E $S/logs
OUT=$S/logs/eval.out
log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $S/logs/eval_seeds.log; }
tags() { ls $C | grep -E "$1" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//'; }

until [ -f $S/batch1.copied ]; do sleep 300; done
main_tags=$(tags "_K10_dir0.5_(la|b)_seed")
log "1/9 100 noise draws, main configs (new seeds) -> 10-seed significance"
python v3_eval.py --cache $C --out $S/eval_r100 --workers $W --tags "$main_tags" --groups pcfcp,qq,variants --r-fast 100 >> $OUT 2>&1
python significance_v4.py --eval results/v4/eval_r100 --eval $S/eval_r100 --out $S/SIGNIFICANCE_10seed.md >> $OUT 2>&1
log "2/9 main configs: refs, pcfcp, qq, variants, ablations, alpha, dpfedcp23"
python v3_eval.py --cache $C --out $E --workers $W --tags "$main_tags" --groups refs,pcfcp,qq,variants,ablations,alpha,dpfedcp23 >> $OUT 2>&1
log "3/9 calibration/test re-splits (20 per image model, 20 models) -> significance"
python resplit_caches.py --cache results/v4/cache --cache $C --out results/v4_resplit/cache --repeats 20 >> $OUT 2>&1
python v3_eval.py --cache results/v4_resplit/cache --out results/v4_resplit/eval --workers $W --groups refs,pcfcp,qq >> $OUT 2>&1
python significance_v4.py --eval results/v4_resplit/eval --model-div 100 --out results/v4_resplit/SIGNIFICANCE_resplit.md >> $OUT 2>&1
log "4/9 calibration-set fractions, fast groups"
for f in 0.2 0.4; do
  python v3_eval.py --cache $C --out $E --workers $W --tags "$main_tags" --groups refs,pcfcp,qq,variants --cal-frac $f >> $OUT 2>&1
done

until [ -f $S/batch2.copied ]; do sleep 300; done
iid_tags=$(tags "_dir100_la_seed"); scale_tags=$(tags "_K(50|100)_dir0.5_la_seed"); if100_tags=$(tags "_if100_seed")
log "5/9 near-IID and client-scale configs"
python v3_eval.py --cache $C --out $E --workers $W --tags "$iid_tags,$scale_tags" --groups refs,pcfcp,qq,variants,a2,dpfedcp23 >> $OUT 2>&1
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
