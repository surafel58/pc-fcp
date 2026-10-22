#!/bin/sh
# BATCH_PLAN_2026-10-19 on the GCP L4 VM. Unattended:
#   1. gate: CIFAR-10-LT seed 42 with logit adjustment (la) and plain cross-entropy (ce)
#   2. decide: adopt la if tail top-1 improves by >= 2 points and head top-1 drops < 1 point
#   3. train everything with the chosen loss, three streams (v3_train.py skips cached runs)
#   4. write results/v3/logs/batch.done
# Rerunning resumes after an interruption.
cd ~/dp-fcp-gate
L=results/v3/logs
mkdir -p $L results/v3/cache
rm -f $L/batch.done

python3 -u v3_train.py --dataset cifar10 --seeds 42 --loss la >> $L/gate_la.log 2>&1
python3 -u v3_train.py --dataset cifar10 --seeds 42 --loss ce --tag-suffix _ce >> $L/gate_ce.log 2>&1

LOSS=$(python3 - <<'PY'
import pandas as pd
a = pd.read_csv('results/v3/accuracy.csv').drop_duplicates('tag', keep='last').set_index('tag')
la, ce = a.loc['cifar10_K10_dir0.5_la_seed42'], a.loc['cifar10_K10_dir0.5_ce_seed42']
d_tail = 100 * (la.top1_tail - ce.top1_tail); d_head = 100 * (la.top1_head - ce.top1_head)
print('la' if (d_tail >= 2.0 and d_head > -1.0) else 'ce')
import sys
sys.stderr.write(f'GATE la tail {100*la.top1_tail:.1f} head {100*la.top1_head:.1f} top1 {100*la.top1:.1f} | '
                 f'ce tail {100*ce.top1_tail:.1f} head {100*ce.top1_head:.1f} top1 {100*ce.top1:.1f} | '
                 f'd_tail {d_tail:+.1f} d_head {d_head:+.1f}\n')
PY
)
echo "GATE_DECISION $LOSS" | tee $L/gate.decision
if [ "$LOSS" = "ce" ]; then SUF="--tag-suffix _ce"; else SUF=""; fi

rm -f $L/s1.done $L/s2.done $L/s3.done
nohup sh -c "python3 -u v3_train.py --dataset cifar10 --seeds 42,43,44 --loss $LOSS $SUF >> $L/b_c10_main.log 2>&1; \
  python3 -u v3_train.py --dataset cifar10 --seeds 42,43,44 --dirichlet 100 --loss $LOSS $SUF >> $L/b_c10_iid.log 2>&1; \
  python3 -u v3_train.py --dataset cifar10 --seeds 42,43,44 --n-clients 50 --loss $LOSS $SUF >> $L/b_c10_k50.log 2>&1; \
  python3 -u v3_train.py --dataset text --seeds 42,43,44 --tag-suffix _b >> $L/b_text.log 2>&1; \
  echo S1_DONE > $L/s1.done" < /dev/null > /dev/null 2>&1 &
nohup sh -c "python3 -u v3_train.py --dataset cifar100 --seeds 42,43,44 --loss $LOSS $SUF >> $L/b_c100_main.log 2>&1; \
  python3 -u v3_train.py --dataset cifar100 --seeds 42,43,44 --if-factor 100 --loss $LOSS $SUF >> $L/b_c100_if100.log 2>&1; \
  python3 -u v3_train.py --dataset cifar100 --seeds 42,43,44 --n-clients 50 --loss $LOSS $SUF >> $L/b_c100_k50.log 2>&1; \
  echo S2_DONE > $L/s2.done" < /dev/null > /dev/null 2>&1 &
nohup sh -c "python3 -u v3_train.py --dataset cifar100 --seeds 42,43,44 --n-clients 100 --loss $LOSS $SUF >> $L/b_c100_k100.log 2>&1; \
  python3 -u v3_train.py --dataset cifar10 --seeds 42,43,44 --n-clients 100 --loss $LOSS $SUF >> $L/b_c10_k100.log 2>&1; \
  echo S3_DONE > $L/s3.done" < /dev/null > /dev/null 2>&1 &

# wait for the three streams, then mark done
while [ ! -f $L/s1.done ] || [ ! -f $L/s2.done ] || [ ! -f $L/s3.done ]; do sleep 120; done
echo BATCH_DONE > $L/batch.done
