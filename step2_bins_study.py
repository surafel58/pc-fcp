"""Step 2 (after the mock defense, 2026-10-13): histogram grid for PC-FCP with model C.

With the ImageNet-pretrained ResNet-18 (model C, results/t1_pilot/probs_C_seed42.npz),
PC-FCP's mean set size at B = 50 uniform bins is 3.15 labels against 1.44 for
centralized Mondrian. The gap is binning, not noise: the true thresholds of the two
rarest classes (0.992, 0.994) fall in the last uniform bin [0.98, 1] and round up to
1.0, which admits those labels into almost every set. APS scores of unlikely labels
all lie just below 1, so resolution near 1 matters.

Candidate grids are fixed before seeing any data (public, so no privacy cost):
  uniform B         edges j/B
  power p, B        edges 1 - (1 - j/B)^p   (finer near 1 for p > 1)

SELECTION USES THE CALIBRATION DATA ONLY. Each repetition splits every class's
calibration scores 50/50 into a fit half and a validation half; thresholds are read
from the noisy histogram of the fit half (eps = 1, Laplace(1/eps) per bin, the
current clipped read-out) and judged on the validation half.
Pre-declared rule: among grids whose mean validation UnderCov is no worse than the
current grid (uniform B = 50), choose the one with the smallest mean validation set
size. Test-set results are computed afterwards, for the chosen and the current grid.
"""
import argparse

import numpy as np

import dp_fcp_gate as g
from t5_readout_ablation import readout

ALPHA, REPS = 0.1, 100
EPS = 1.0   # set from --eps in main()
GRIDS = [('uniform', 1, B) for B in (10, 20, 50, 100, 200)] + \
        [('power', p, B) for p in (2, 3) for B in (20, 50, 100)]


def edges_of(kind, p, B):
    u = np.linspace(0.0, 1.0, B + 1)
    return u if kind == 'uniform' else 1.0 - (1.0 - u) ** p


def thresholds(scores, labels, edges, rng, C):
    B = len(edges) - 1
    return np.array([readout(np.histogram(np.clip(scores[labels == c], 0, 1), bins=edges)[0]
                             .astype(float) + g.laplace(1.0 / EPS, B, rng), ALPHA, edges, True)[0]
                     for c in range(C)])


def evaluate(thr, all_scores, labels, marg_thr):
    t = np.where(np.isinf(thr), marg_thr, thr)
    inc = all_scores <= t[None, :]
    cov = np.array([inc[labels == c, c].mean() for c in range(len(t))])
    return cov, float(inc.sum(1).mean())


def main():
    global EPS
    ap = argparse.ArgumentParser()
    ap.add_argument('--probs', default='results/t1_pilot/probs_C_seed42.npz')
    ap.add_argument('--classes', type=int, default=10)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--eps', type=float, default=1.0)
    a = ap.parse_args()
    EPS, SEED, C = a.eps, a.seed, a.classes
    g.N_CLASSES = C
    TAIL = list(range(C - C // 3, C))
    d = np.load(a.probs)
    cp, cl, tp, tl = d['cal_probs'], d['cal_labels'], d['test_probs'], d['test_labels']
    print(f'probs={a.probs} classes={C} eps={EPS} seed={SEED}')
    cs = g.aps_score_true(cp, cl, np.random.RandomState(2024 + SEED))
    cal_all = g.aps_score_all(cp, np.random.RandomState(31 + SEED))
    test_all = g.aps_score_all(tp, np.random.RandomState(9000 + SEED))
    marg_thr = g.fs_quantile(cs, ALPHA)

    # ---------------- selection on calibration splits ----------------
    val = {}
    for rep in range(REPS):
        srng = np.random.RandomState(500 + rep)
        fit = np.zeros(len(cl), bool)
        for c in range(C):
            idx = np.nonzero(cl == c)[0]
            fit[srng.choice(idx, len(idx) // 2, replace=False)] = True
        vmask = ~fit
        for grid in GRIDS:
            nrng = np.random.RandomState(700000 + 1000 * rep + GRIDS.index(grid))
            thr = thresholds(cs[fit], cl[fit], edges_of(*grid), nrng, C)
            cov, size = evaluate(thr, cal_all[vmask], cl[vmask], marg_thr)
            r = val.setdefault(grid, {'cov': [], 'size': []})
            r['cov'].append(cov); r['size'].append(size)
    print('VALIDATION (calibration split 50/50, %d repetitions, eps=%g)' % (REPS, EPS))
    print(f"{'grid':16s} {'size':>6s} {'UnderCov':>9s} {'worst':>6s} {'tail':>6s}")
    summ = {}
    for grid in GRIDS:
        cov = np.array(val[grid]['cov'])
        s = dict(size=np.mean(val[grid]['size']),
                 under=100 * np.maximum(0.9 - cov, 0).mean(),
                 worst=100 * cov.min(1).mean(), tail=100 * cov[:, TAIL].mean())
        summ[grid] = s
        print(f"{grid[0]+' p='+str(grid[1])+' B='+str(grid[2]):16s} {s['size']:6.2f} {s['under']:9.2f} "
              f"{s['worst']:6.1f} {s['tail']:6.1f}")
    ref = summ[('uniform', 1, 50)]
    ok = [gr for gr in GRIDS if summ[gr]['under'] <= ref['under'] + 1e-9]
    chosen = min(ok, key=lambda gr: summ[gr]['size'])
    print(f'\nRULE: UnderCov <= {ref["under"]:.2f} (uniform B=50); eligible: {len(ok)}; CHOSEN: {chosen}')

    # ---------------- test set, full calibration, after selection ----------------
    print('\nTEST (full calibration, 200 noise draws), chosen vs current vs no-privacy reference')
    raw = np.array([g.fs_quantile(cs[cl == c], ALPHA) for c in range(C)])
    cov, size = evaluate(raw, test_all, tl, marg_thr)
    print(f"  {'centralized Mondrian':24s} size={size:5.2f} worst={100*cov.min():5.1f} "
          f"tail={100*cov[TAIL].mean():5.1f} under={100*np.maximum(0.9-cov,0).mean():5.2f}")
    for grid in (('uniform', 1, 50), chosen):
        covs, sizes = [], []
        for rep in range(200):
            thr = thresholds(cs, cl, edges_of(*grid), np.random.RandomState(900000 + rep), C)
            c_, s_ = evaluate(thr, test_all, tl, marg_thr)
            covs.append(c_); sizes.append(s_)
        covs = np.array(covs)
        print(f"  {str(grid):24s} size={np.mean(sizes):5.2f} worst={100*covs.min(1).mean():5.1f} "
              f"tail={100*covs[:, TAIL].mean():5.1f} under={100*np.maximum(0.9-covs,0).mean():5.2f} "
              f"per-class={' '.join(f'{100*x:.0f}' for x in covs.mean(0))}")


if __name__ == '__main__':
    main()
