"""Set size on a long-tailed test distribution: the per-class mean set sizes weighted by the
class frequencies of the long-tailed calibration pool (the same weighting as the long-tailed
marginal coverage). Mean over noise draws, then mean (std) over seeds."""
import glob
import os
import re

RES = os.environ.get('PCFCP_RES', 'results/v3')

import numpy as np
import pandas as pd

MAIN = {'cifar10': '1.0', 'cifar100': '1.0', 'text': '4.0'}
METHODS = ['mondrian', 'clustered_ding', 'pcfcp', 'fedcp2qq_marg', 'fedcp2qq_pc', 'fedcp2qq_pc_fallback',
           'dpfedcp_marg', 'dpfedcp_pc']
rows = []
for ds, eps in MAIN.items():
    for f in sorted(f for f in glob.glob(f'{RES}/eval/{ds}_K10_dir0.5*_seed*__*.csv') if re.search(r'_dir0\.5(_la|_b)?_seed', f)):
        d = pd.read_csv(f, dtype={'eps': str})
        d = d[(d.alpha - 0.1).abs() < 1e-9 & d.method.isin(METHODS)] if False else d[((d.alpha - 0.1).abs() < 1e-9) & d.method.isin(METHODS)]
        d = d[(d.eps == eps) | (d.eps == 'inf')]
        d = d[~((d.eps == 'inf') & d.method.str.startswith('dpfedcp'))]
        if d.empty:
            continue
        tag = d.tag.iloc[0]
        cal = np.load(f'{RES}/cache/{tag}.npz')['cal_labels']
        prev = np.bincount(cal) / len(cal)
        for _, r in d.iterrows():
            sz = np.array(r.per_class_size.split(), float)
            rows.append({'ds': ds, 'seed': r.seed, 'method': r.method, 'bal': r.set_size, 'lt': float(sz @ prev)})
x = pd.DataFrame(rows).groupby(['ds', 'method', 'seed'])[['bal', 'lt']].mean().groupby(['ds', 'method']).agg(['mean', 'std'])
pd.set_option('display.width', 200)
print(x.round(2).to_string())
