#!/bin/sh
# 20 Newsgroups calibration/test re-splits (2026-11-05, decision: option c). The standard 20NG test
# set is split by date, so calibration (older posts) and test (newer posts) are not exchangeable and
# the coverage guarantees of Chapter 3 do not apply to the main text runs. Pooling each class's
# calibration and test rows and redrawing them at random (as resplit_caches.py already does for the
# images) makes them exchangeable by construction; the date-split runs stay in the thesis as the
# distribution-shift case. 10 models x 20 re-splits, evaluated with the fixed (clipped) v3_eval.py.
cd "$(dirname "$0")/../.."
LOG=results/v5/logs/resplit_text.log
mkdir -p results/v5/logs
log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a $LOG; }
log "re-split the 10 text models, 20 times each"
python resplit_caches.py --cache results/v5/cache --out results/v4_resplit/cache --repeats 20 \
    --pattern '^text_K10_dir0\.5_b_seed\d+\.npz$' >> $LOG 2>&1
tags=$(ls results/v4_resplit/cache | grep -E '^text_K10_dir0\.5_b_rs_seed' | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//')
log "evaluate $(echo $tags | tr ',' '\n' | wc -l) text re-splits: refs, pcfcp, qq"
python v3_eval.py --cache results/v4_resplit/cache --out results/v4_resplit/eval --workers ${WORKERS:-8} \
    --tags "$tags" --groups refs,pcfcp,qq >> $LOG 2>&1
log "significance over the re-splits (images and text)"
python significance_v4.py --eval results/v4_resplit/eval --model-div 100 --out results/v4_resplit/SIGNIFICANCE_resplit.md >> $LOG 2>&1
log "done"
