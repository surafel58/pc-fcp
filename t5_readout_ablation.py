"""T5 read-out ablation (examiner comment T5, mock defense 2026-10-12).

Question: does clipping negative noisy bins to zero in the PC-FCP read-out inflate
the per-class count, and therefore the threshold and the coverage of rare classes?

Phase 1 (--phase scores) retrains the CIFAR-10-LT models with the settings of
results/v2/c10_main.csv (IF 100, 10 clients, Dirichlet 0.5, 80 rounds, seeds 42-44)
and caches the calibration scores and test softmax per seed in results/t5/.
The splits and score RNG streams are those of dp_fcp_gate.run_seed.

Phase 2 (--phase readout) compares read-out variants on the cached scores. Every
variant of one repetition sees the SAME Laplace draw, so differences between
variants are caused by the read-out alone:
  raw            : exact per-class quantile of the pooled raw scores (no bins, no noise)
  binned_clean   : histogram read-out, B bins, no noise (binning alone)
  clip_est_n     : CURRENT PC-FCP: negatives -> 0, n estimated from the clipped sum
  noclip_est_n   : Pillutla et al. option 1: no clipping, n = unclipped noisy sum
  clip_true_n    : diagnostic only: clipped bins, true n (n_c is private, not deployable)
  noclip_true_n  : diagnostic only: Pillutla option 2, unclipped bins, true n
  consistent     : Option B: isotonic (non-negative, non-decreasing) projection of the
                   noisy running total, n = its last value
"""
import argparse
import csv
import math
import os
import time

import numpy as np
import torch
from sklearn.isotonic import isotonic_regression
from torch.utils.data import Subset

import dp_fcp_gate as g

OUT_DIR = 'results/t5'


def cache_scores(seed, args, cifar_train, cifar_test, device):
    """Mirror of dp_fcp_gate.run_seed up to build_client_scores; saves an npz."""
    tr_targets = cifar_train.targets
    lt_idx, _ = g.lt_indices(tr_targets, args.if_factor, n_max=5000, seed=seed)
    rng = np.random.RandomState(seed + 1)
    lt_arr = np.asarray(lt_idx); lt_y = tr_targets[lt_arr]
    train_idx = []; cal_idx = []
    for c in range(g.N_CLASSES):
        ci = lt_arr[lt_y == c].tolist(); rng.shuffle(ci)
        n_tr = max(1, int(0.5 * len(ci)))
        train_idx += ci[:n_tr]; cal_idx += ci[n_tr:]
    train_client_lists = g.dirichlet_partition(train_idx, tr_targets, args.n_clients, 0.5, seed)
    client_ds = [Subset(cifar_train, l) for l in train_client_lists if len(l) >= 2]
    t0 = time.time()
    model = g.train_fedavg(client_ds, device, args.n_rounds, 2, 0.02, seed)
    print(f'[seed={seed}] trained ({time.time()-t0:.0f}s)', flush=True)
    cal_src = g._CachedCIFAR(cifar_train.images, cifar_train.targets, train=False)
    cal_probs, cal_labels = g.softmax_on(model, Subset(cal_src, cal_idx), device)
    test_probs, test_labels = g.softmax_on(model, cifar_test, device)
    cal_index_of = {gidx: row for row, gidx in enumerate(cal_idx)}
    cal_client_lists = g.dirichlet_partition(cal_idx, tr_targets, args.n_clients, 0.5, seed + 7)
    score_rng = np.random.RandomState(2024 + seed)
    cal_scores = g.aps_score_true(cal_probs, cal_labels, score_rng)
    cal_client = np.full(len(cal_idx), -1, dtype=np.int64)
    for k, lst in enumerate(cal_client_lists):
        for gidx in lst:
            cal_client[cal_index_of[gidx]] = k
    acc = float((test_probs.argmax(1) == test_labels).mean())
    print(f'[seed={seed}] test top-1 accuracy {acc:.4f}', flush=True)
    np.savez_compressed(os.path.join(OUT_DIR, f'scores_c10_seed{seed}.npz'),
                        cal_probs=cal_probs, cal_labels=cal_labels, cal_scores=cal_scores,
                        cal_client=cal_client, test_probs=test_probs, test_labels=test_labels)


# ---------------- read-out variants ----------------
def readout(h, alpha, edges, clip, n_true=None):
    """Finite-sample quantile from a (noisy) histogram; first crossing, upper bin edge.
    clip=True reproduces dp_fcp_gate.fs_quantile_from_hist exactly."""
    hh = np.maximum(h, 0.0) if clip else h
    n = float(n_true) if n_true is not None else float(hh.sum())
    if n < 1:
        return float('inf'), n
    r = math.ceil((n + 1) * (1 - alpha))
    if r > n:
        return float('inf'), n
    cum = np.cumsum(hh)
    hit = np.nonzero(cum >= r)[0]
    b = int(hit[0]) if len(hit) else len(h) - 1
    return float(edges[b + 1]), n


def readout_consistent(h, alpha, edges):
    """Consistency post-processing (Option B): project the noisy running total onto
    non-negative, non-decreasing sequences by least squares (isotonic regression; the
    bounded solution is the unbounded one clipped at 0), then read the quantile from
    the projected running total. n is its last value. Post-processing, so no extra
    privacy cost."""
    cum = np.maximum(isotonic_regression(np.cumsum(h), increasing=True), 0.0)
    n = float(cum[-1])
    if n < 1:
        return float('inf'), n
    r = math.ceil((n + 1) * (1 - alpha))
    if r > n:
        return float('inf'), n
    b = int(np.nonzero(cum >= r)[0][0])
    return float(edges[b + 1]), n


VARIANTS = ['raw', 'binned_clean', 'clip_est_n', 'noclip_est_n', 'clip_true_n', 'noclip_true_n',
            'consistent']


def coverage(thr, test_all, test_labels, marg_thr):
    C = len(thr)
    t = np.array([marg_thr if math.isinf(x) else x for x in thr])
    inc = test_all <= t[None, :]
    size = inc.sum(1)
    cov = np.array([inc[test_labels == c, c].mean() for c in range(C)])
    marg_bal = inc[np.arange(len(test_labels)), test_labels].mean()
    return cov, float(marg_bal), float(size.mean())


def run_readout(args):
    alpha = 0.1
    rows = []
    perclass = []
    for seed in args.seeds:
        d = np.load(os.path.join(OUT_DIR, f'scores_c10_seed{seed}.npz'))
        cal_labels, cal_scores = d['cal_labels'], d['cal_scores']
        test_probs, test_labels = d['test_probs'], d['test_labels']
        C = g.N_CLASSES
        n_c = np.array([(cal_labels == c).sum() for c in range(C)])
        prev = n_c / n_c.sum()
        # same test APS randomisation as dp_fcp_gate.eval_coverage
        test_all = g.aps_score_all(test_probs, np.random.RandomState(9000 + seed))
        marg_thr = g.fs_quantile(cal_scores, alpha)
        for B in args.bins:
            edges = np.linspace(0.0, 1.0, B + 1)
            h_clean = [np.histogram(np.clip(cal_scores[cal_labels == c], 0, 1), bins=edges)[0]
                       .astype(float) for c in range(C)]
            thr_raw = [g.fs_quantile(cal_scores[cal_labels == c], alpha) for c in range(C)]
            thr_bin = [readout(h_clean[c], alpha, edges, clip=True)[0] for c in range(C)]
            for eps in args.eps:
                acc = {v: {'cov': [], 'thr': [], 'n': [], 'marg': [], 'size': [], 'inf': []}
                       for v in VARIANTS}
                for rep in range(args.reps):
                    # rep 0 uses the exact noise stream of the thesis run (hist_dp_secagg)
                    nrng = np.random.RandomState(888 + seed + int(eps * 10) if rep == 0
                                                 else 10_000_000 + 1000 * seed + rep)
                    z = [g.laplace(1.0 / eps, B, nrng) for _ in range(C)]
                    thr = {'raw': thr_raw, 'binned_clean': thr_bin}
                    ns = {'raw': n_c.astype(float), 'binned_clean': n_c.astype(float)}
                    for v, clip, use_true in (('clip_est_n', True, False),
                                              ('noclip_est_n', False, False),
                                              ('clip_true_n', True, True),
                                              ('noclip_true_n', False, True)):
                        out = [readout(h_clean[c] + z[c], alpha, edges, clip,
                                       n_c[c] if use_true else None) for c in range(C)]
                        thr[v] = [o[0] for o in out]
                        ns[v] = np.array([o[1] for o in out])
                    out = [readout_consistent(h_clean[c] + z[c], alpha, edges) for c in range(C)]
                    thr['consistent'] = [o[0] for o in out]
                    ns['consistent'] = np.array([o[1] for o in out])
                    for v in VARIANTS:
                        cov, marg, size = coverage(thr[v], test_all, test_labels, marg_thr)
                        a = acc[v]
                        a['cov'].append(cov); a['thr'].append(thr[v]); a['n'].append(ns[v])
                        a['marg'].append(marg); a['size'].append(size)
                        a['inf'].append([math.isinf(x) for x in thr[v]])
                for v in VARIANTS:
                    a = acc[v]
                    cov = np.array(a['cov'])                       # reps x C
                    thr_arr = np.array(a['thr'], dtype=float)
                    thr_arr = np.where(np.isinf(thr_arr), np.nan, thr_arr)
                    nn_ = np.array(a['n'])
                    rows.append({
                        'seed': seed, 'B': B, 'eps': eps, 'variant': v, 'reps': args.reps,
                        'marginal_balanced': round(float(np.mean(a['marg'])), 4),
                        'marginal_lt': round(float(np.mean(cov @ prev)), 4),
                        'tail': round(float(cov[:, g.TAIL_CLASSES].mean()), 4),
                        'worst': round(float(cov.min(1).mean()), 4),
                        'covgap': round(float(100 * np.abs(cov - 0.9).mean()), 3),
                        'undercov': round(float(100 * np.maximum(0.9 - cov, 0).mean()), 3),
                        'set_size': round(float(np.mean(a['size'])), 3),
                        'rep0_marginal': round(float(a['marg'][0]), 4),
                        'rep0_tail': round(float(cov[0, g.TAIL_CLASSES].mean()), 4),
                        'rep0_worst': round(float(cov[0].min()), 4),
                        'frac_inf_thr': round(float(np.mean(a['inf'])), 4),
                    })
                    for c in range(C):
                        perclass.append({
                            'seed': seed, 'B': B, 'eps': eps, 'variant': v, 'class': c,
                            'n_true': int(n_c[c]),
                            'n_est_mean': round(float(nn_[:, c].mean()), 2),
                            'thr_mean': round(float(np.nanmean(thr_arr[:, c])), 4)
                            if not np.all(np.isnan(thr_arr[:, c])) else 'inf',
                            'thr_raw': round(thr_raw[c], 4),
                            'cov_mean': round(float(cov[:, c].mean()), 4),
                            'frac_inf': round(float(np.mean([r[c] for r in a['inf']])), 4),
                        })
        print(f'[seed={seed}] read-out done', flush=True)
    for name, data in (('t5_summary.csv', rows), ('t5_perclass.csv', perclass)):
        with open(os.path.join(OUT_DIR, name), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader(); w.writerows(data)
        print(f'[wrote] {OUT_DIR}/{name} ({len(data)} rows)', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['scores', 'readout'], required=True)
    p.add_argument('--cache-path', default='data/cifar10/cache.pt')
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--if-factor', type=int, default=100)
    p.add_argument('--n-clients', type=int, default=10)
    p.add_argument('--n-rounds', type=int, default=80)
    p.add_argument('--bins', default='50')
    p.add_argument('--eps', default='1,4,8')
    p.add_argument('--reps', type=int, default=200)
    args = p.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(',')]
    args.bins = [int(b) for b in args.bins.split(',')]
    args.eps = [float(e) for e in args.eps.split(',')]
    g.N_CLASSES = 10; g.TAIL_CLASSES = [7, 8, 9]; g.ALPHA = 0.1
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.phase == 'scores':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'[t5] device={device}', flush=True)
        cifar_train, cifar_test = g.load_cifar_cached(args.cache_path)
        for seed in args.seeds:
            cache_scores(seed, args, cifar_train, cifar_test, device)
    else:
        run_readout(args)


if __name__ == '__main__':
    main()
