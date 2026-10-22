"""Validate OUR clustered-CP reimplementation against the authors' code.

Both are run on the SAME data (the authors' precomputed CIFAR-100 softmax
scores) with the SAME split protocol, so any difference is an implementation
difference rather than a data or model difference.

Our implementation is imported from the dp-fcp-gate harness so that what is
tested is exactly the code used in the experiments.
"""
import sys, glob, pickle, collections
import numpy as np

HARNESS = r'd:\projects\Seminars in AI\Paper\Proposal\Second Semester\Implementation\dp-fcp-gate'
sys.path.insert(0, HARNESS)

import dp_fcp_gate as G

ALPHA = 0.10
DATA = r'C:\Users\suraf\AppData\Local\Temp\ccc-repo\data\cifar-100.npz'


def softmax_score_all(probs):
    """The authors' 'softmax' score: s(x,y) = 1 - p_y(x). Deterministic."""
    return 1.0 - probs


def covgap_and_size(thr, val_probs, val_labels, n_classes, alpha):
    scores = softmax_score_all(val_probs)          # [N, C]
    thr_v = np.array([thr[c] for c in range(n_classes)], dtype=np.float64)
    included = scores <= thr_v[None, :]
    sizes = included.sum(axis=1)
    cov = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = val_labels == c
        if m.any():
            cov[c] = included[m, c].mean()
    covgap = 100.0 * np.nanmean(np.abs(cov - (1 - alpha)))
    return covgap, float(sizes.mean()), cov


def run_ours(seed, n_avg, probs, labels, n_classes):
    """Our arm_clustered_cp on the authors' data, authors' split protocol."""
    rng = np.random.RandomState(seed)
    n_cal = n_avg * n_classes
    idx = rng.permutation(len(labels))
    cal_idx, val_idx = idx[:n_cal], idx[n_cal:]

    cal_probs, cal_labels = probs[cal_idx], labels[cal_idx]
    val_probs, val_labels = probs[val_idx], labels[val_idx]

    # our harness works on "true-label scores" for calibration
    cal_scores = 1.0 - cal_probs[np.arange(len(cal_labels)), cal_labels]

    G.N_CLASSES = n_classes
    thr, assign, n_cl = G.arm_clustered_cp(
        cal_scores, cal_labels, ALPHA, np.random.RandomState(seed + 1))
    cg, sz, _ = covgap_and_size(thr, val_probs, val_labels, n_classes, ALPHA)
    return cg, sz, n_cl


def run_mondrian(seed, n_avg, probs, labels, n_classes):
    """Our arm_mondrian_centralized (= classwise CP) on the same split."""
    rng = np.random.RandomState(seed)
    n_cal = n_avg * n_classes
    idx = rng.permutation(len(labels))
    cal_idx, val_idx = idx[:n_cal], idx[n_cal:]
    cal_probs, cal_labels = probs[cal_idx], labels[cal_idx]
    val_probs, val_labels = probs[val_idx], labels[val_idx]
    cal_scores = 1.0 - cal_probs[np.arange(len(cal_labels)), cal_labels]

    G.N_CLASSES = n_classes
    thr, _ = G.arm_mondrian_centralized(cal_scores, cal_labels, ALPHA)
    cg, sz, _ = covgap_and_size(thr, val_probs, val_labels, n_classes, ALPHA)
    return cg, sz


def authors_numbers(n_avg):
    files = sorted(glob.glob(
        rf'C:\Users\suraf\AppData\Local\Temp\ccc-repo\.cache\paper\varying_n'
        rf'\cifar-100\random_calset\n_totalcal={n_avg}\score=softmax\*_allresults.pkl'))
    agg = collections.defaultdict(lambda: {'cg': [], 'sz': []})
    for f in files:
        d = pickle.load(open(f, 'rb'))
        for m, t in d.items():
            agg[m]['cg'].append(t[2]['mean_class_cov_gap'] * 100)
            agg[m]['sz'].append(t[3]['mean'])
    return {m: (float(np.mean(v['cg'])), float(np.mean(v['sz']))) for m, v in agg.items()}


def main():
    d = np.load(DATA)
    probs, labels = d['softmax'], d['labels'].astype(int)
    n_classes = probs.shape[1]
    print(f'data: {probs.shape}, classes={n_classes}\n')

    # published Table 2, CIFAR-100 / softmax
    PUB = {20: {'standard': (4.0, 7.9), 'classwise': (5.6, 19.3), 'clustered': (3.6, 8.9)},
           50: {'standard': (4.1, 7.9), 'classwise': (3.8, 11.9), 'clustered': (2.9, 9.3)}}

    for n_avg in (20, 50):
        auth = authors_numbers(n_avg)
        ours_cl = [run_ours(s, n_avg, probs, labels, n_classes) for s in range(10)]
        ours_mo = [run_mondrian(s, n_avg, probs, labels, n_classes) for s in range(10)]
        cg_cl = np.mean([r[0] for r in ours_cl]); sz_cl = np.mean([r[1] for r in ours_cl])
        cg_mo = np.mean([r[0] for r in ours_mo]); sz_mo = np.mean([r[1] for r in ours_mo])
        ncl = [r[2] for r in ours_cl]

        print(f'===== CIFAR-100 softmax, n_avg={n_avg}, 10 seeds =====')
        print(f'{"method":<26}{"CovGap":<10}{"AvgSize":<10}{"source"}')
        p = PUB[n_avg]
        print(f'{"classwise (published)":<26}{p["classwise"][0]:<10.2f}{p["classwise"][1]:<10.2f}paper Table 2')
        print(f'{"classwise (authors code)":<26}{auth["classwise"][0]:<10.2f}{auth["classwise"][1]:<10.2f}their repo')
        print(f'{"classwise (OURS)":<26}{cg_mo:<10.2f}{sz_mo:<10.2f}our harness')
        print()
        print(f'{"clustered (published)":<26}{p["clustered"][0]:<10.2f}{p["clustered"][1]:<10.2f}paper Table 2')
        print(f'{"clustered (authors code)":<26}{auth["cluster_random"][0]:<10.2f}{auth["cluster_random"][1]:<10.2f}their repo')
        print(f'{"clustered (OURS)":<26}{cg_cl:<10.2f}{sz_cl:<10.2f}our harness')
        print(f'   our cluster counts per seed: {ncl}')
        print()
        dcg = abs(cg_cl - auth['cluster_random'][0])
        dmo = abs(cg_mo - auth['classwise'][0])
        print(f'   |ours - authors| CovGap: clustered={dcg:.2f}, classwise={dmo:.2f}')
        print(f'   verdict (tol 0.3): clustered={"PASS" if dcg <= 0.3 else "FAIL"}, '
              f'classwise={"PASS" if dmo <= 0.3 else "FAIL"}')
        print()


if __name__ == '__main__':
    main()
