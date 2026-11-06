#!/bin/sh
# Extra seeds for the post-defense revision (examiner comment B5, 2026-11-03).
# Same configuration as the v4 main runs (BATCH_PLAN_2026-10-19.md, logit adjustment adopted):
#   cifar10  K=10 Dir(0.5) loss=la   -> cifar10_K10_dir0.5_la_seed<s>
#   cifar100 K=10 Dir(0.5) loss=la   -> cifar100_K10_dir0.5_la_seed<s>
#   text     K=10 Dir(0.5)           -> text_K10_dir0.5_b_seed<s>
# New seeds 45..51 (10 seeds in total with 42..44). Seed 42 of CIFAR-10-LT and of the text model is
# retrained first as a reproducibility check against results/v4 (accuracy.csv). Three streams on one L4.
# When all streams finish: results/v3/logs/seeds.done, and the VM powers off 3 hours later unless
# the shutdown is cancelled (sudo shutdown -c).
cd ~/dp-fcp-gate
L=results/v3/logs
mkdir -p $L results/v3/cache
rm -f $L/seeds_s1.done $L/seeds_s2.done $L/seeds_s3.done $L/seeds.done

nohup sh -c "python3 -u v3_train.py --dataset cifar10 --seeds 42 --loss la >> $L/seeds_c10_repro.log 2>&1; \
  python3 -u v3_train.py --dataset cifar10 --seeds 45,46,47,48,49,50,51 --loss la >> $L/seeds_c10.log 2>&1; \
  echo S1_DONE > $L/seeds_s1.done" < /dev/null > /dev/null 2>&1 &
nohup sh -c "python3 -u v3_train.py --dataset cifar100 --seeds 45,46,47,48 --loss la >> $L/seeds_c100_a.log 2>&1; \
  echo S2_DONE > $L/seeds_s2.done" < /dev/null > /dev/null 2>&1 &
nohup sh -c "python3 -u v3_train.py --dataset text --seeds 42 --tag-suffix _b >> $L/seeds_text_repro.log 2>&1; \
  python3 -u v3_train.py --dataset text --seeds 45,46,47,48,49,50,51 --tag-suffix _b >> $L/seeds_text.log 2>&1; \
  python3 -u v3_train.py --dataset cifar100 --seeds 49,50,51 --loss la >> $L/seeds_c100_b.log 2>&1; \
  echo S3_DONE > $L/seeds_s3.done" < /dev/null > /dev/null 2>&1 &

while [ ! -f $L/seeds_s1.done ] || [ ! -f $L/seeds_s2.done ] || [ ! -f $L/seeds_s3.done ]; do sleep 120; done
echo SEEDS_DONE > $L/seeds.done
sudo shutdown -h +180 "seeds batch done; failsafe power-off"
