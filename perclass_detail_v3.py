"""Per-class detail for CIFAR-10-LT (Natnael, Gap 2): each class's threshold, the mean set
size of its test examples, its coverage, and its false-inclusion rate (the share of test
examples of the other classes whose set contains the class), for PC-FCP at eps = 1 and for
the non-private Mondrian reference. PC-FCP is recomputed with the same noise streams as
v3_eval.py (20 draws per seed), so its coverage matches the evaluation files.

Also times one calibration run of each method on one seed (CPU), for the run-time report.

Writes results/v3/perclass_detail_c10.csv and results/v3/runtime_c10.csv."""
import glob
import os
import time

import numpy as np
import pandas as pd

import dp_fcp_gate as g

RES = os.environ.get('PCFCP_RES', 'results/v3')
MAIN_RX = r'cifar10_K10_dir0.5(_la)?_seed'
import v3_eval as V
from step2_bins_study import edges_of

rows = []
for path in sorted([f for f in glob.glob(f'{RES}/cache/cifar10_K10_dir0.5*_seed*.npz') if __import__('re').search(MAIN_RX, f)]):
    D = V.Data(path)
    D.alpha = 0.1
    g.N_CLASSES, g.TAIL_CLASSES = D.C, D.tail
    cs, ta = D.score('aps')
    e = edges_of(*V.MAIN_GRID)
    tl = D.test_labels
    runs = {'mondrian': [V.mondrian(D, cs, 0.1)],
            'pcfcp': [V.pcfcp(D, cs, e, 1.0, 0.1, V.rs(D, 'pcfcp', 1.0, 0.1, r)) for r in range(V.R_FAST)]}
    for m, thrs in runs.items():
        for r, thr in enumerate(thrs):
            inc = ta <= np.asarray(thr, float)[None, :]
            for c in range(D.C):
                rows.append({'seed': D.seed, 'method': m, 'draw': r, 'cls': c, 'n_cal': int(D.n_c[c]),
                             'thr': float(thr[c]), 'cov': inc[tl == c, c].mean(),
                             'size': inc[tl == c].sum(1).mean(), 'false_incl': inc[tl != c, c].mean()})
x = pd.DataFrame(rows)
x.to_csv(f'{RES}/perclass_detail_c10_draws.csv', index=False)
s = x.groupby(['method', 'cls', 'seed']).mean(numeric_only=True).groupby(['method', 'cls']).agg(['mean', 'std'])
s.to_csv(f'{RES}/perclass_detail_c10.csv')
pd.set_option('display.width', 250)
print(s[['n_cal', 'thr', 'cov', 'size', 'false_incl']].round(3).to_string())

# ---------------------------------------------------------------- run time (one seed, CPU)
D = V.Data(sorted([f for f in glob.glob(f'{RES}/cache/cifar10_K10_dir0.5*_seed42.npz') if __import__('re').search(MAIN_RX, f)])[0])
D.alpha = 0.1
g.N_CLASSES, g.TAIL_CLASSES = D.C, D.tail
cs, ta = D.score('aps')
pc = D.per_client(cs)
e, eq = edges_of(*V.MAIN_GRID), edges_of(*V.QQ_GRID)
rng = np.random.RandomState(0)
T = []
for name, fn in [('PC-FCP', lambda: V.pcfcp(D, cs, e, 1.0, 0.1, rng)),
                 ('Mondrian', lambda: V.mondrian(D, cs, 0.1)),
                 ('FedCP2-QQ', lambda: V.fedcpqq(D, pc, 0.1, False, 1.0, eq, rng)),
                 ('FedCP2-QQ, per class', lambda: V.fedcpqq(D, pc, 0.1, True, 1.0, eq, rng)),
                 ('DP-FedCP', lambda: V.dpfedcp_marginal(D, pc, 0.1, rng, eps=1.0)),
                 ('DP-FedCP, per class', lambda: V.dpfedcp_perclass(D, pc, 0.1, rng, g.sigma_g_from_eps(1.0)))]:
    t0 = time.perf_counter()
    fn()
    T.append({'method': name, 'seconds': time.perf_counter() - t0})
    print(name, round(T[-1]['seconds'], 3), flush=True)
pd.DataFrame(T).to_csv(f'{RES}/runtime_c10.csv', index=False)
