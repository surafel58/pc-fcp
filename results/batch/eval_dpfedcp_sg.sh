#!/bin/sh
# DP-FedCP at the rest of its paper's own noise grid, sigma_g = 10, 30, 60 (examiner comment
# B6, 2026-11-04): the paper tested {10, 30, 60, 100}; sigma_g = 100 and zero noise are already
# in the 'dpfedcp' group. Main configs (K = 10, Dir(0.5)) of the three datasets, 10 seeds x 3
# noise draws, written into results/v5/eval. CIFAR-10 first (Table tab:noise_elim), then text,
# then CIFAR-100 (slowest). DATASETS="text cifar100" sh results/batch/eval_dpfedcp_sg.sh runs a
# subset (CIFAR-10 was run on its own on 2026-11-04; text and CIFAR-100 deferred, about 2 h).
cd "$(dirname "$0")/../.."
C=results/v5/cache; E=results/v5/eval
W=${WORKERS:-8}
mkdir -p results/v5/logs
OUT=results/v5/logs/dpfedcp_sg.out
log() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a results/v5/logs/dpfedcp_sg.log; }
tags() { ls $C | grep -E "$1" | sed 's/.npz//' | tr '\n' ',' | sed 's/,$//'; }
for ds in ${DATASETS:-cifar10 text cifar100}; do
  t=$(tags "^${ds}_K10_dir0.5_(la|b)_seed")
  log "$ds: dpfedcpsg (sigma_g 10, 30, 60) on $(echo $t | tr ',' '\n' | wc -l) models"
  python v3_eval.py --cache $C --out $E --workers $W --tags "$t" --groups dpfedcpsg >> $OUT 2>&1
done
log "done"
