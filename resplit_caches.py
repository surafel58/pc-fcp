"""Random calibration/test re-splits of the cached model outputs (examiner comment B5, option b).

Standard conformal-prediction evaluation repeats the calibration/test split many times for a fixed
model. Here, for every trained model (one cache) and every repeat j = 1..J:
  * per class c, the calibration rows and test rows of c are pooled, and n_c of them (n_c = the
    original calibration count of c, so the long-tailed calibration profile is kept) are drawn
    at random as the new calibration set; the rest are the new test set (so the test set stays
    balanced, with the original per-class test count);
  * the new calibration rows are assigned to clients by a fresh Dirichlet(0.5) draw, with the
    same routine as training (dp_fcp_gate.dirichlet_partition).
Both pools are held out from training and scored with the same clean transform, so within a class
they are exchangeable. Images only: the 20 Newsgroups test set is split by date (a distribution
shift), so pooling it with the calibration rows would change that setting.

Output tag: <dataset>_K10_dir0.5_la_rs_seed<100*s + j>; v3_eval reads the seed from the tag, so each
repeat gets its own APS randomisation and noise stream. The analysis recovers s = seed // 100,
j = seed % 100.

Usage: python resplit_caches.py --cache results/v4/cache [--cache results/v4_seeds/cache]
                                --out results/v4_resplit/cache --repeats 20
"""
import argparse
import glob
import os
import re

import numpy as np

import dp_fcp_gate as g


def resplit(path, j, out_dir):
    d = np.load(path)
    tag = os.path.basename(path)[:-4]
    s = int(tag.split('seed')[-1])
    K = int(re.search(r'_K(\d+)_', tag).group(1))
    alpha = float(re.search(r'_dir([0-9.]+)_', tag).group(1))
    cp, cl, tp, tl = d['cal_probs'], d['cal_labels'], d['test_probs'], d['test_labels']
    C = cp.shape[1]
    rng = np.random.RandomState(100000 + 100 * s + j)
    P = np.concatenate([cp, tp]); Y = np.concatenate([cl, tl])
    cal_rows, test_rows = [], []
    for c in range(C):
        idx = np.nonzero(Y == c)[0]
        rng.shuffle(idx)
        n_c = int((cl == c).sum())
        cal_rows.append(idx[:n_c]); test_rows.append(idx[n_c:])
    cal_rows = np.concatenate(cal_rows); test_rows = np.sort(np.concatenate(test_rows))
    g.N_CLASSES = C
    lists = g.dirichlet_partition(cal_rows.tolist(), Y, K, alpha, 200000 + 100 * s + j)
    client_of = {}
    for k, lst in enumerate(lists):
        for r in lst:
            client_of[r] = k
    order = np.sort(cal_rows)
    cal_client = np.array([client_of[r] for r in order], dtype=np.int64)
    new_tag = tag.replace(f'_seed{s}', f'_rs_seed{100 * s + j}')
    np.savez_compressed(os.path.join(out_dir, new_tag + '.npz'), cal_probs=P[order], cal_labels=Y[order],
                        cal_client=cal_client, test_probs=P[test_rows], test_labels=Y[test_rows])
    return new_tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', action='append', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--pattern', default=r'^cifar(10|100)_K10_dir0\.5_la_seed\d+\.npz$')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(p for c in a.cache for p in glob.glob(os.path.join(c, '*.npz'))
                   if re.match(a.pattern, os.path.basename(p)))
    for p in paths:
        for j in range(1, a.repeats + 1):
            t = resplit(p, j, a.out)
        print(f'{os.path.basename(p)}: {a.repeats} re-splits (last {t})', flush=True)


if __name__ == '__main__':
    main()
