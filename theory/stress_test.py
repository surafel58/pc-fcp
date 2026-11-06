"""Stress test of the coverage results of Section V of the paper.

For each configuration (bins, alpha, eps, pooled class count n, score distribution) it simulates the
top-down read-out of v3_eval.topdown on one class, R times, and compares the measured per-class
coverage with
  Theorem 1:     without noise, coverage >= r / (n + 1)
  Theorem 2:     coverage >= (r - E[D]) / (n + 1)        (E[D] by Monte Carlo on the noise alone)
  Lemma 1:       E[D] < 1 + f + (2 + alpha) sqrt(2B) / eps
  Proposition 4: margin variant >= 1 - alpha - gamma     (t from the exact Laplace-sum tail, gamma = 0.05)
A violation is flagged when the measured coverage is below the bound by more than 4 standard errors.
The score distributions include continuous ones, heavy ties, point masses on bin edges, and two-point
laws built so that the clean suffix sum sits exactly at the rank.

Usage: python theory/stress_test.py [R]      (the paper's run used R = 10000)
Writes theory/out/stress_results.csv.
"""
import os, sys, math, time, itertools
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'theory', 'out')
sys.path.insert(0, ROOT)
from v3_eval import laplace_sum_margin          # exact t(eps, B, gamma)

rng = np.random.default_rng(20261104)

def edges_cubic(B): return 1 - (1 - np.arange(B + 1) / B) ** 3
def edges_unif(B): return np.arange(B + 1) / B

def make_dists(e):
    """name -> sampler(size) of scores in [0, 1]."""
    mid = (e[:-1] + e[1:]) / 2
    d = {
        'beta(0.5,3)': lambda s: rng.beta(0.5, 3, s),
        'beta(2,2)':   lambda s: rng.beta(2, 2, s),
        'beta(5,1)':   lambda s: rng.beta(5, 1, s),          # mass near the top
        'uniform':     lambda s: rng.random(s),
        'ties5':       lambda s: rng.choice([0.1, 0.5, 0.8, 0.95, 1.0], s, p=[.3, .3, .2, .15, .05]),
        'on_edges':    lambda s: rng.choice(e[1:], s),        # every score exactly on a bin edge
        'top_heavy':   lambda s: np.where(rng.random(s) < 0.5, 1.0, rng.random(s) * e[-2]),
    }
    for q in (0.05, 0.1, 0.2):                                # two-point laws: top bin w.p. q, else lowest
        d[f'2pt_top{q}'] = (lambda q: lambda s: np.where(rng.random(s) < q, mid[-1], mid[0]))(q)
        d[f'2pt_adj{q}'] = (lambda q: lambda s: np.where(rng.random(s) < q, mid[-1], mid[-2]))(q)
    return d

def hist_rows(scores, e):
    """Row-wise numpy.histogram (bins [e_{b-1}, e_b), last bin closed)."""
    B = len(e) - 1
    idx = np.clip(np.searchsorted(e, scores, side='right') - 1, 0, B - 1)
    rows_ = scores.shape[0]
    flat = (np.arange(rows_)[:, None] * B + idx).ravel()
    return np.bincount(flat, minlength=rows_ * B).reshape(rows_, B).astype(float)

def readout_rows(H, e, alpha, margin=0):
    """Vectorized v3_eval.topdown on each row of H (noisy histograms)."""
    k = np.floor(alpha * (H.sum(1) + 1) + 1e-9) - margin
    S = np.cumsum(H[:, ::-1], 1)[:, ::-1]
    ok = S >= k[:, None]
    has = ok.any(1)
    last = np.where(has, ok.shape[1] - 1 - np.argmax(ok[:, ::-1], 1), 0)
    tau = e[last + 1]
    return np.where(k < 1, np.inf, tau)

ED_cache, T_cache = {}, {}
def ED(B, eps, alpha, f, Rn=400000):
    key = (B, eps, alpha, round(f, 9))
    if key not in ED_cache:
        Z = rng.laplace(0, 1 / eps, (Rn, B))
        N = np.cumsum(Z[:, ::-1], 1)[:, ::-1]
        X = np.floor(f + alpha * N[:, :1]) - N[:, 1:]
        ED_cache[key] = np.maximum(0, np.ceil(X.max(1))).mean()
    return ED_cache[key]

def run(R):
    rows = []
    t0 = time.time()
    for B, (ename, efun) in itertools.product((4, 8, 16), (('cubic', edges_cubic), ('unif', edges_unif))):
        e = efun(B)
        dists = make_dists(e)
        for alpha, eps, n in itertools.product((0.05, 0.1, 0.2, 0.35), (0.5, 1, 2, 4, 8), (9, 12, 20, 25, 40, 80, 250)):
            k0 = math.floor(alpha * (n + 1) + 1e-12); f = alpha * (n + 1) - k0; r = n + 1 - k0
            ed = ED(B, eps, alpha, f)
            expl = 1 + f + (2 + alpha) * math.sqrt(2 * B) / eps
            if (eps, B) not in T_cache:
                T_cache[(eps, B)] = laplace_sum_margin(eps, B, 0.05)
            t = T_cache[(eps, B)]; m = t + int(math.ceil(alpha * t - 1e-9))
            for dname, samp in dists.items():
                s = samp((R, n + 1))
                cal, test = s[:, :n], s[:, n]
                h = hist_rows(cal, e)
                Z = rng.laplace(0, 1 / eps, (R, B))
                tau = readout_rows(h + Z, e, alpha)
                cov = (test <= tau).mean()
                tau_m = readout_rows(h + Z, e, alpha, margin=m)
                cov_m = (test <= tau_m).mean()
                tau_0 = readout_rows(h, e, alpha)
                cov_0 = (test <= tau_0).mean()
                se = math.sqrt(max(cov * (1 - cov), 1e-4) / R)
                rows.append(dict(B=B, edges=ename, alpha=alpha, eps=eps, n=n, dist=dname,
                                 cov_main=cov, bound_thm2=(r - ed) / (n + 1), ED=ed, lemma1=expl,
                                 cov_margin=cov_m, bound_prop4=1 - alpha - 0.05,
                                 cov_nonoise=cov_0, bound_thm1=r / (n + 1), se=se))
        print(f'  B={B} {ename}: {len(rows)} configs so far, {time.time() - t0:.0f}s', flush=True)
    return rows

if __name__ == '__main__':
    import pandas as pd
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    os.makedirs(OUT, exist_ok=True)
    d = pd.DataFrame(run(R))
    d.to_csv(os.path.join(OUT, 'stress_results.csv'), index=False)
    tol = 4 * d['se']
    v2 = d[d['cov_main'] < d['bound_thm2'] - tol]; v4 = d[d['cov_margin'] < d['bound_prop4'] - tol]
    v1 = d[d['cov_nonoise'] < d['bound_thm1'] - tol]; vl = d[d['ED'] >= d['lemma1']]
    print(f'\nconfigs: {len(d)}, R = {R} per config')
    print(f'Theorem 1 violations (no-noise cov < r/(n+1)):            {len(v1)}')
    print(f'Theorem 2 violations (cov < (r - E[D])/(n+1) - 4 se):     {len(v2)}')
    print(f'Lemma 1 violations (E[D] >= 1 + f + (2+a)sqrt(2B)/eps):   {len(vl)}')
    print(f'Proposition 4 violations (margin cov < 1 - a - gamma):    {len(v4)}')
    d['slack_thm2'] = d['cov_main'] - d['bound_thm2']
    print('\nTightest Theorem 2 cases (smallest slack):')
    print(d.nsmallest(12, 'slack_thm2')[['B', 'edges', 'alpha', 'eps', 'n', 'dist', 'cov_main', 'bound_thm2', 'ED', 'slack_thm2']].to_string(index=False))
    d['slack_prop4'] = d['cov_margin'] - d['bound_prop4']
    print('\nTightest Proposition 4 cases:')
    print(d.nsmallest(8, 'slack_prop4')[['B', 'edges', 'alpha', 'eps', 'n', 'dist', 'cov_margin', 'bound_prop4', 'slack_prop4']].to_string(index=False))
    for name, v in (('Theorem 1', v1), ('Theorem 2', v2), ('Lemma 1', vl), ('Proposition 4', v4)):
        if len(v):
            print(f'\n{name} VIOLATIONS:'); print(v.head(20).to_string(index=False))
