#!/bin/sh
# Second extra-seed batch (2026-11-03): seeds 45..51 for every other Chapter 4 setting, so that every
# comparison in the thesis uses 10 seeds (42..51). Same configuration as BATCH_PLAN_2026-10-19.md:
#   cifar10  K=10 Dir(100)            -> cifar10_K10_dir100_la_seed<s>      (near-IID)
#   cifar10  K=50 / K=100 Dir(0.5)    -> cifar10_K50_dir0.5_la_seed<s>, cifar10_K100_...
#   cifar100 K=50 / K=100 Dir(0.5)    -> cifar100_K50_dir0.5_la_seed<s>, cifar100_K100_...
#   cifar100 K=10 Dir(0.5) IF=100     -> cifar100_K10_dir0.5_la_if100_seed<s>
# Each stream starts when one stream of launch_seeds.sh has finished (its seeds_s<i>.done flag).
# When all three finish: results/v3/logs/seeds2.done, then a failsafe power-off 3 hours later.
cd ~/dp-fcp-gate
L=results/v3/logs
rm -f $L/seeds2_a.done $L/seeds2_b.done $L/seeds2_c.done $L/seeds2.done
T="python3 -u v3_train.py --loss la"

nohup sh -c "while [ ! -f $L/seeds_s3.done ]; do sleep 60; done; \
  $T --dataset cifar100 --n-clients 100 --seeds 45,46,47,48,49,50,51 >> $L/seeds2_c100_k100.log 2>&1; \
  $T --dataset cifar100 --n-clients 50 --seeds 45,46,47 >> $L/seeds2_c100_k50_a.log 2>&1; \
  echo A > $L/seeds2_a.done" < /dev/null > /dev/null 2>&1 &
nohup sh -c "while [ ! -f $L/seeds_s2.done ]; do sleep 60; done; \
  $T --dataset cifar10 --dirichlet 100 --seeds 45,46,47,48,49,50,51 >> $L/seeds2_c10_iid.log 2>&1; \
  $T --dataset cifar100 --if-factor 100 --seeds 45,46,47,48,49,50,51 >> $L/seeds2_c100_if100.log 2>&1; \
  $T --dataset cifar100 --n-clients 50 --seeds 48,49 >> $L/seeds2_c100_k50_b.log 2>&1; \
  echo B > $L/seeds2_b.done" < /dev/null > /dev/null 2>&1 &
nohup sh -c "while [ ! -f $L/seeds_s1.done ]; do sleep 60; done; \
  $T --dataset cifar10 --n-clients 50 --seeds 45,46,47,48,49,50,51 >> $L/seeds2_c10_k50.log 2>&1; \
  $T --dataset cifar10 --n-clients 100 --seeds 45,46,47,48,49,50,51 >> $L/seeds2_c10_k100.log 2>&1; \
  $T --dataset cifar100 --n-clients 50 --seeds 50,51 >> $L/seeds2_c100_k50_c.log 2>&1; \
  echo C > $L/seeds2_c.done" < /dev/null > /dev/null 2>&1 &

while [ ! -f $L/seeds2_a.done ] || [ ! -f $L/seeds2_b.done ] || [ ! -f $L/seeds2_c.done ]; do sleep 120; done
echo SEEDS2_DONE > $L/seeds2.done
sudo shutdown -h +180 "seeds batch 2 done; failsafe power-off"
