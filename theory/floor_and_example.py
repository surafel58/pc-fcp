"""Two further checks of Section V of the paper.

(i) A ten-class long-tailed federation with the pooled floor: the coverage of every class against the
    lower bound of Theorem 2 (which holds with or without the floor) and against the upper bound of
    Proposition 3 before the floor and with the added term P(tau_pool > tau_c).
(ii) The example of Section S-A of the supplement: with n_c = 10, alpha = 0.1 and B = 8, and the
    11 calibration and test scores a random order of nine scores in the first bin and one each in the
    second and third, coverage is 10/11 without noise and lower at eps = 1000, so the bound of
    Theorem 2 has no exact 1 - alpha form under the noise.

Re-implements v3_eval.topdown and the floor of PC-FCP, vectorized over R runs.
Usage: python theory/floor_and_example.py
"""
import math
import numpy as np

rng = np.random.default_rng(11)
B, alpha, R = 8, 0.1, 20000
e = 1 - (1 - np.arange(B + 1) / B) ** 3

def hist(s):
    """numpy.histogram convention (bins [e_{b-1}, e_b), last bin closed), over the last axis."""
    idx = np.clip(np.searchsorted(e, s, side='right') - 1, 0, B - 1)
    out = np.zeros(s.shape[:-1] + (B,))
    for b in range(B):
        out[..., b] = (idx == b).sum(-1)
    return out

def topdown(H, margin=0):
    k = np.floor(alpha * (H.sum(-1) + 1) + 1e-9) - margin
    S = np.cumsum(H[..., ::-1], -1)[..., ::-1]
    ok = S >= k[..., None]
    last = np.where(ok.any(-1), B - 1 - np.argmax(ok[..., ::-1], -1), 0)
    return np.where(k < 1, np.inf, e[last + 1]), np.where(k < 1, -1, last)

def ED(n, eps, Rn=400000):
    k0 = math.floor(alpha * (n + 1) + 1e-12); f = alpha * (n + 1) - k0
    Z = rng.laplace(0, 1 / eps, (Rn, B)); N = np.cumsum(Z[:, ::-1], 1)[:, ::-1]
    D = np.maximum(0, np.ceil((np.floor(f + alpha * N[:, :1]) - N[:, 1:]).max(1))).mean()
    fl = np.floor(f + alpha * N[:, 0]); U = np.maximum(0, np.floor((N[:, 1:] - fl[:, None]).max(1))).mean()
    return D, U

if __name__ == '__main__':
    # (i) long-tailed federation; class score laws from easy (low scores) to hard
    ns = [250, 160, 100, 64, 40, 25, 25, 16, 12, 10]
    laws = [(0.5, 6), (0.5, 4), (0.7, 3), (1, 3), (1, 2), (0.5, 6), (2, 2), (2, 1.5), (3, 1.5), (1, 1)]
    for eps in (1.0, 4.0):
        cal = [rng.beta(a, b, (R, n)) for (a, b), n in zip(laws, ns)]
        test = [rng.beta(a, b, R) for (a, b) in laws]
        H = np.stack([hist(c) + rng.laplace(0, 1 / eps, (R, B)) for c in cal], 1)   # R x C x B
        tau, bst = topdown(H)
        tpool, _ = topdown(H.sum(1))
        tfl = np.maximum(tau, tpool[:, None])
        print(f'eps = {eps}')
        for c, ((a, b), n) in enumerate(zip(laws, ns)):
            k0 = math.floor(alpha * (n + 1) + 1e-12); r = n + 1 - k0
            D, U = ED(n, eps)
            big = rng.beta(a, b, 400000)
            pb = np.array([((big >= e[j]) & (big <= e[j + 1])).mean() for j in range(B)])
            Epb = np.where(bst[:, c] >= 0, pb[np.maximum(bst[:, c], 0)], 0).mean()
            Pinf = (bst[:, c] < 0).mean()
            up = 1 - alpha + (1 + U) / (n + 1) + Epb + Pinf
            pfl = (tpool > tau[:, c]).mean()
            cov_nf = (test[c] <= tau[:, c]).mean(); cov_f = (test[c] <= tfl[:, c]).mean()
            se = math.sqrt(cov_f * (1 - cov_f) / R)
            print(f'  class {c} n={n:3d} beta{(a, b)}: cov with floor {cov_f:.4f}, Theorem 2 bound {(r - D)/(n+1):.4f} '
                  f'{"OK" if cov_f >= (r - D)/(n+1) - 4*se else "VIOLATED"} | cov without floor {cov_nf:.4f}, '
                  f'Proposition 3 bound {up:.3f} {"OK" if cov_nf <= up + 4*se else "VIOLATED"} | '
                  f'P(pool > own) {pfl:.3f}, bound with floor {up + pfl:.3f} '
                  f'{"OK" if cov_f <= up + pfl + 4*se else "VIOLATED"}')

    # (ii) the n_c = 10 example: exchangeable multiset 9 x bin 1, 1 x bin 2, 1 x bin 3 (scores at bin midpoints)
    mid = (e[:-1] + e[1:]) / 2
    base = np.array([mid[0]] * 9 + [mid[1], mid[2]])
    for eps in (np.inf, 1000.0):
        P = np.array([rng.permutation(base) for _ in range(200000)])
        Hc = hist(P[:, :10]) + (0 if np.isinf(eps) else rng.laplace(0, 1 / eps, (len(P), B)))
        t, _ = topdown(Hc)
        print(f'n_c = 10 example, eps = {eps}: simulated coverage {(P[:, 10] <= t).mean():.4f} (200000 orders)')
    # Small-noise limit in closed form: k = 1 always; the test score is covered surely when it is one of
    # the nine bin-1 scores, never when it is the bin-3 score, and, when it is the bin-2 score, when the
    # noise of the bin-3 suffix sum is non-negative or, failing that, z_2 >= |N_3|. So the coverage is
    # 9/11 + (1 + q)/22 with q = P(z_2 >= |z_3 + ... + z_8|), which does not depend on the noise scale.
    z2 = rng.laplace(0, 1, 10_000_000)
    W = rng.laplace(0, 1, (10_000_000, 6)).sum(1)
    q = (z2 >= np.abs(W)).mean()
    print(f'n_c = 10 example, small-noise limit: 9/11 + (1 + q)/22 = {9 / 11 + (1 + q) / 22:.3f} (q = {q:.4f})')
