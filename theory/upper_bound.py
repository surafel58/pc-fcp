"""Coverage upper bound (Proposition 3 of the paper) and the bin trade-off (Table S-II).

With b* the bin chosen by the read-out and p(b) = P(S in [e_{b-1}, e_b] | Y = c) (closed bin),
  P(cover) <= 1 - alpha + (1 + E[U_c])/(n_c + 1) + E[p(b*) 1{k~ >= 1}] + P(k~ < 1)       (Proposition 3)
and, conditioning on the noise before averaging, the tighter
  P(cover) <= P(k~ < 1) + E[1{k~ >= 1} ((r + U_c)/(n_c + 1) + p(b*))].
U_c, k~ and b* come from the same noise draw; the expectation is over calibration data and noise.
Both forms are checked over the stress grid of theory/stress_test.py plus six empirical APS score
laws of real classes: the true-label test scores of the most and the least frequent class of the
seed-42 model of each dataset, resampled i.i.d. The summary at the end gives the values reported in
the paper at B = 8 and alpha = 0.1, and the rows of Table S-II (eps = 1, n_c = 25, B = 4, 8, 16).

Needs the classifier outputs of the seed-42 models (v3_train.py), by default in results/v5/cache;
set PCFCP_CACHE to use another directory.
Usage: python theory/upper_bound.py [R]      (the paper's run used R = 10000)
Writes theory/out/upper_results.csv.
"""
import os, sys, math, time, itertools
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'theory', 'out')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'theory'))
import stress_test as st
import dp_fcp_gate as g

rng = st.rng
CACHE = os.environ.get('PCFCP_CACHE', os.path.join(ROOT, 'results', 'v5', 'cache'))

def real_laws():
    out = {}
    for tag, classes in (('cifar10_K10_dir0.5_la_seed42', (0, 9)), ('cifar100_K10_dir0.5_la_seed42', (0, 99)),
                         ('text_K10_dir0.5_b_seed42', (0, 19))):
        d = np.load(os.path.join(CACHE, tag + '.npz'))
        P, y = d['test_probs'], d['test_labels']
        s = np.clip(g.aps_score_true(P, y, np.random.RandomState(1)), 0, 1)
        for c in classes:
            pool = s[y == c]
            out[f'real_{tag.split("_")[0]}_c{c}'] = (lambda pool: lambda size: rng.choice(pool, size))(pool)
    return out

def bin_mass(samp, e, M=400000):
    s = samp(M)
    return np.array([((s >= e[b]) & (s <= e[b + 1])).mean() for b in range(len(e) - 1)])

def readout_b(H, e, alpha):
    """Vectorized topdown returning (k~, b* as 0-based bin index or -1 when k~ < 1)."""
    k = np.floor(alpha * (H.sum(1) + 1) + 1e-9)
    S = np.cumsum(H[:, ::-1], 1)[:, ::-1]
    ok = S >= k[:, None]
    last = np.where(ok.any(1), ok.shape[1] - 1 - np.argmax(ok[:, ::-1], 1), 0)
    return k, np.where(k < 1, -1, last)

if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    os.makedirs(OUT, exist_ok=True)
    REAL = real_laws()
    rows = []; t0 = time.time()
    grid = list(itertools.product((4, 8, 16), (('cubic', st.edges_cubic), ('unif', st.edges_unif))))
    for B, (ename, efun) in grid:
        e = efun(B)
        laws = dict(st.make_dists(e)); laws.update(REAL)
        masses = {name: bin_mass(s, e) for name, s in laws.items()}
        for alpha, eps, n in itertools.product((0.05, 0.1, 0.2, 0.35), (0.5, 1, 2, 4, 8), (9, 12, 20, 25, 40, 80, 250)):
            k0 = math.floor(alpha * (n + 1) + 1e-12); f = alpha * (n + 1) - k0; r = n + 1 - k0
            for name, samp in laws.items():
                pm = masses[name]
                s = samp((R, n + 1)); cal, test = s[:, :n], s[:, n]
                h = st.hist_rows(cal, e); Z = rng.laplace(0, 1 / eps, (R, B))
                N = np.cumsum(Z[:, ::-1], 1)[:, ::-1]
                fl = np.floor(f + alpha * N[:, 0])
                U = np.maximum(0, np.floor((N - fl[:, None]).max(1)))
                k, bstar = readout_b(h + Z, e, alpha)
                tau = np.where(bstar < 0, np.inf, e[np.maximum(bstar, 0) + 1])
                cov = (test <= tau).mean()
                pb = np.where(bstar < 0, 0.0, pm[np.maximum(bstar, 0)])
                fin = bstar >= 0
                tight = np.where(fin, (r + U) / (n + 1) + pb, 1.0).mean()
                simple = 1 - alpha + (1 + U.mean()) / (n + 1) + (pb * fin).mean() + (~fin).mean()
                se = math.sqrt(max(cov * (1 - cov), 1e-4) / R)
                rows.append(dict(B=B, edges=ename, alpha=alpha, eps=eps, n=n, dist=name, cov_main=cov,
                                 Epb=(pb * fin).mean(), p_max=pm.max(), P_inf=(~fin).mean(), up_tight=tight,
                                 up_simple=simple, up_pmax=1 - alpha + (1 + U.mean()) / (n + 1) + pm.max() + (~fin).mean(), se=se))
        print(f'  B={B} {ename}: {len(rows)} configs, {time.time() - t0:.0f}s', flush=True)

    d = pd.DataFrame(rows); d.to_csv(os.path.join(OUT, 'upper_results.csv'), index=False)
    print(f'\nconfigs: {len(d)}, R = {R}')
    for name, col in (('Proposition 3', 'up_simple'), ('tighter form', 'up_tight')):
        v = d[d['cov_main'] > d[col] + 4 * d['se']]
        print(f'{name:14s} violations: {len(v)}')
        if len(v): print(v.head(10).to_string(index=False))

    real = d.dist.str.startswith('real') & (d.edges == 'cubic') & (d.alpha == 0.1)
    print('\nUpper bound of Proposition 3 at B = 8, alpha = 0.1, eps = 1, over the six real score laws:')
    for n in (25, 250):
        m = d[real & (d.B == 8) & (d.n == n) & (d.eps == 1)]
        print(f'  n_c = {n:3d}: {m.up_simple.min():.3f} to {m.up_simple.max():.3f}')
    print('Mean over the six laws at each eps (B = 8):')
    for n in (25, 250):
        g = d[real & (d.B == 8) & (d.n == n)].groupby('eps').up_simple.mean()
        print(f'  n_c = {n:3d}: ' + '  '.join(f'eps={e:g}: {v:.3f}' for e, v in g.items()))
    print('\nTable S-II (eps = 1, n_c = 25, cubic bins; means over the six real score laws;')
    print('the lower-bound row is given by theory/lower_bound_table.py):')
    m = d[real & (d.eps == 1) & (d.n == 25)].groupby('B')[['Epb', 'cov_main']].mean()
    for B in (4, 8, 16):
        print(f'  B = {B:2d}: bin term {m.loc[B, "Epb"]:.3f}, simulated coverage {m.loc[B, "cov_main"]:.3f}')
    print('\nShare of configs with an informative bound (< 1):', round((d.up_simple < 1).mean(), 3),
          '| with the largest bin in place of p(b*):', round((d.up_pmax < 1).mean(), 3))
