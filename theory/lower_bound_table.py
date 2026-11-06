"""The lower bound (r_c - E[D_c]) / (n_c + 1) of Theorem 2 at alpha = 0.1:
  Table S-I:  B = 8, a class with n_c = 25 and n_c = 250 pooled calibration examples, eps = 1 to 8
              (E[D_c] from 2 x 10^6 simulated noise draws per configuration);
  Table S-II: the lower-bound row, eps = 1 and n_c = 25 for B = 4, 8 and 16 (E[D_c] from 10 x 2 x 10^6
              draws, printed to four decimals; at B = 4 the bound lies at 0.8625, which the table
              rounds half up to 0.863).

D_c = max(0, max_{2<=j<=B} ceil(floor(f + alpha N_1) - N_j)), where N_j is the sum of the Laplace(1/eps)
noise of bins j..B and alpha (n_c + 1) = k_0 + f. E[D_c] depends only on B, alpha, eps and f.

Usage: python theory/lower_bound_table.py
"""
import math
import numpy as np

rng = np.random.default_rng(5)
B, ALPHA = 8, 0.1

def ED(eps, f, Rn=2_000_000, B=B):
    Z = rng.laplace(0, 1 / eps, (Rn, B))
    N = np.cumsum(Z[:, ::-1], 1)[:, ::-1]              # N[:, j] = noise of the suffix sum from bin j
    X = np.floor(f + ALPHA * N[:, :1]) - N[:, 1:]
    return np.maximum(0, np.ceil(X.max(1))).mean()

if __name__ == '__main__':
    print(f'B = {B}, alpha = {ALPHA}')
    for n in (25, 250):
        k0 = math.floor(ALPHA * (n + 1) + 1e-12); f = ALPHA * (n + 1) - k0; r = n + 1 - k0
        eds = {eps: ED(eps, f) for eps in (1, 2, 3, 4, 8)}
        print(f'n_c = {n:3d} (k_0 = {k0}, f = {f:.1f}, r_c = {r})')
        print('  E[D_c]: ' + '  '.join(f'eps={e}: {v:.2f}' for e, v in eds.items()))
        print('  bound:  ' + '  '.join(f'eps={e}: {(r - v) / (n + 1):.3f}' for e, v in eds.items()))
    print('\nTable S-II, lower-bound row (eps = 1, n_c = 25, cubic bins):')
    for b in (4, 8, 16):
        v = np.array([ED(1.0, 0.6, B=b) for _ in range(10)])
        m, se = v.mean(), v.std(ddof=1) / math.sqrt(len(v))
        print(f'  B = {b:2d}: E[D_c] = {m:.4f} (se {se:.4f}), bound = {(24 - m) / 26:.4f}')
