"""v3 full rerun, part 2 (2026-10-14): evaluate every calibration method on the cached
classifier outputs of v3_train.py. CPU only; parallel over (cache, method group).

Conventions (decided after the mock defense):
  * scores: APS (main; calibration scores aps_score_true with RandomState(2024+seed), test
    scores aps_score_all with RandomState(9000+seed), as in dp_fcp_gate) and LAC (ablation)
  * PC-FCP: one histogram per class, power grid 1-(1-j/B)^2 with B = 100, the clients'
    histograms summed and ONE Laplace(1/eps) draw per bin added to the sum, TOP-DOWN
    read-out (k = floor(alpha (n_hat + 1)), highest bin whose suffix count >= k, its upper
    edge; k < 1 -> infinite threshold)
  * an infinite threshold means the label is ALWAYS included (main convention for every
    method); FedCP-QQ is additionally reported with the old marginal fallback (T3)
  * privacy-noise randomness is averaged over R independent draws per seed
Writes results/v3/eval/<cache tag>__<group>.csv, one row per (method, eps, alpha, draw).
"""
import argparse
import glob
import math
import multiprocessing as mp
import os
import re
import sys
import time
import zlib

import numpy as np

import dp_fcp_gate as g
from step2_bins_study import edges_of
from t5_readout_ablation import readout, readout_consistent

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines', 'plassier'))
import dp_fedcp as P  # noqa: E402

CACHE = 'results/v3/cache'
OUT = 'results/v3/eval'
R_FAST = 20          # noise draws for histogram-based methods and FedCP-QQ
R_SLOW = 3           # noise draws for DP-FedCP (federated optimisation is slow)
EPS = (1.0, 2.0, 3.0, 4.0, 8.0)
ALPHAS_SWEEP = (0.05, 0.2, 0.3, 0.35)
MAIN_GRID = ('power', 3, 8)     # PC-FCP: cubic bins, B = 8
QQ_GRID = ('power', 2, 100)    # FedCP2-QQ's exponential mechanism: a fine grid (its error grows only with log B)


# ============================================================ data
class Data:
    def __init__(self, path, cal_frac=None):
        d = np.load(path)
        self.tag = os.path.basename(path)[:-4]
        self.seed = int(self.tag.split('seed')[-1])
        self.cal_probs, self.cal_labels = d['cal_probs'], d['cal_labels']
        self.cal_client, self.test_probs, self.test_labels = d['cal_client'], d['test_probs'], d['test_labels']
        if cal_frac is not None:
            # item 1 of BATCH_PLAN_2026-10-19: keep a seeded per-class fraction of the
            # calibration rows (the model and the test set are unchanged)
            rng = np.random.RandomState(7000 + self.seed)
            keep = []
            for c in range(self.cal_probs.shape[1]):
                idx = np.nonzero(self.cal_labels == c)[0]
                rng.shuffle(idx)
                keep += idx[:max(1, int(round(cal_frac * len(idx))))].tolist()
            keep = np.sort(np.array(keep))
            self.cal_probs, self.cal_labels, self.cal_client = self.cal_probs[keep], self.cal_labels[keep], self.cal_client[keep]
            # suffix before the seed, like the other tag suffixes (_la, _if100), so that
            # analyze_results_v3.TAG_RE parses it
            self.tag = self.tag.replace('_seed', f'_cal{int(round(100 * cal_frac))}_seed')
        self.C = self.cal_probs.shape[1]
        self.K = int(re.search(r'_K(\d+)_', self.tag).group(1))   # from the tag: a client may hold no calibration rows
        self.n_c = np.bincount(self.cal_labels, minlength=self.C)
        self.prev = self.n_c / self.n_c.sum()
        self.tail = list(range(self.C - self.C // 3, self.C))
        self.scores = {}

    def score(self, kind):
        """(calibration true-label scores, test all-label scores) for 'aps' or 'lac'."""
        if kind not in self.scores:
            if kind == 'aps':
                cs = g.aps_score_true(self.cal_probs, self.cal_labels, np.random.RandomState(2024 + self.seed))
                ta = g.aps_score_all(self.test_probs, np.random.RandomState(9000 + self.seed))
            else:
                cs = 1.0 - self.cal_probs[np.arange(len(self.cal_labels)), self.cal_labels]
                ta = 1.0 - self.test_probs
            self.scores[kind] = (cs, ta)
        return self.scores[kind]

    def per_client(self, cs):
        """list over clients of {class: scores} (dp_fcp_gate layout)."""
        out = []
        for k in range(self.K):
            m = self.cal_client == k
            out.append({c: cs[m & (self.cal_labels == c)] for c in range(self.C) if (m & (self.cal_labels == c)).any()})
        return out


# ============================================================ metrics
def evaluate(D, thr, ta):
    """thr: array of C thresholds (inf = always include). Returns metric dict."""
    t = np.asarray(thr, dtype=float)
    inc = ta <= t[None, :]
    tl = D.test_labels
    cov = np.array([inc[tl == c, c].mean() for c in range(D.C)])
    size_pc = np.array([inc[tl == c].sum(1).mean() for c in range(D.C)])
    head = [c for c in range(D.C) if c not in D.tail]
    return {
        'marginal_bal': float(inc[np.arange(len(tl)), tl].mean()),
        'marginal_lt': float(cov @ D.prev),
        'tail_cov': float(cov[D.tail].mean()), 'head_cov': float(cov[head].mean()),
        'worst_cov': float(cov.min()), 'worst_class': int(cov.argmin()),
        'covgap': float(100 * np.abs(cov - (1 - D.alpha)).mean()),
        'undercov': float(100 * np.maximum((1 - D.alpha) - cov, 0).mean()),
        'overcov': float(100 * np.maximum(cov - (1 - D.alpha), 0).mean()),
        'n_under': int((cov < 1 - D.alpha).sum()),
        'set_size': float(inc.sum(1).mean()),
        'tail_size': float(size_pc[D.tail].mean()),
        'n_always': int(np.sum(t >= 1.0)),
        'per_class_cov': ' '.join(f'{x:.4f}' for x in cov),
        'per_class_size': ' '.join(f'{x:.3f}' for x in size_pc),
        'per_class_thr': ' '.join('inf' if math.isinf(x) else f'{x:.5f}' for x in t),
    }


# ============================================================ histogram channel (PC-FCP)
def topdown(h, e, alpha, margin=0.0, suffix=None):
    """Top-down read-out. margin: added to the rank k (the noise margin of the coverage
    bound, item 6 of BATCH_PLAN_2026-10-19). suffix: precomputed suffix sums S[b] =
    sum_{j>=b} h[j] (used by the tree-structured release); default from h."""
    # the noisy suffix sum can understate the true count by up to t, which would make
    # the scan continue past the right bin to a lower threshold; stopping at k - t
    # guards against that (and k - t < 1 gives an infinite threshold)
    k = int(np.floor(alpha * (h.sum() + 1) + 1e-9)) - margin
    if k < 1:
        return math.inf
    S = np.cumsum(h[::-1])[::-1] if suffix is None else suffix
    hit = np.nonzero(S >= k)[0]
    return float(e[(int(hit[-1]) if len(hit) else 0) + 1])


def laplace_sum_tail(b, eps, t):
    """P(|sum of b iid Laplace(0, 1/eps)| > t), exact: the sum is the difference of two
    independent Gamma(b, scale 1/eps) variables, so the tail is
    2 * int_0^inf f_G(y) S_G(y + t) dy, evaluated by quadrature."""
    from scipy import integrate, stats
    g = stats.gamma(b, scale=1.0 / eps)
    val, _ = integrate.quad(lambda y: g.pdf(y) * g.sf(y + t), 0, np.inf, limit=200)
    return 2.0 * val


def laplace_sum_margin(eps, B, gamma):
    """t = smallest integer such that |sum of b Laplace(1/eps) draws| <= t for every b <= B
    (the B suffix sums and the class count) with probability at least 1 - gamma, by a
    union bound over the B + 1 sums and the exact tail of each (laplace_sum_tail).
    Values for B = 8, gamma = 0.05: t = 12, 6, 4, 3, 2 at eps = 1, 2, 3, 4, 8 (the same
    as the 200,000-draw Monte Carlo quantile used before 2026-10-22)."""
    q = gamma / (B + 1)
    t = 1
    while max(laplace_sum_tail(b, eps, t) for b in range(1, B + 1)) > q:
        t += 1
    return t


def bound_margin(eps, B, alpha, gamma):
    """Rank margin of the coverage-bound variant: t + ceil(alpha t). On the event that
    every noisy sum is within t of its true value, the noisy class count overstates n_c
    by at most t, which raises floor(alpha (n_tilde + 1)) by at most ceil(alpha t); the
    extra term absorbs that, so at most floor(alpha (n_c + 1)) true scores lie above
    the threshold and coverage is at least 1 - alpha - gamma (Proposition prop:margin)."""
    t = laplace_sum_margin(eps, B, gamma)
    return t + int(math.ceil(alpha * t - 1e-9))


def tree_release(h, eps, rng):
    """Hay et al. (2010) hierarchical release of an 8-bin histogram: levels of 8, 4, 2
    and 1 nodes (dyadic merges), each level noised at eps / L with L = number of levels
    (the levels are not disjoint, so the budget is split). Returns suffix sums
    S[b] = sum_{j>=b} h[j], each formed from the fewest nodes, and the noisy total."""
    B = len(h)
    L = int(round(math.log2(B))) + 1
    levels = []
    cur = np.asarray(h, float)
    for _ in range(L):
        levels.append(cur + g.laplace(L / eps, len(cur), rng))
        cur = cur.reshape(-1, 2).sum(1) if len(cur) > 1 else cur
    S = np.zeros(B)
    for b in range(B):
        # suffix [b, B): greedy decomposition into aligned dyadic blocks, largest first
        tot, pos = 0.0, b
        while pos < B:
            lvl = 0
            while lvl + 1 < L and pos % (2 ** (lvl + 1)) == 0 and pos + 2 ** (lvl + 1) <= B:
                lvl += 1
            tot += levels[lvl][pos // (2 ** lvl)]
            pos += 2 ** lvl
        S[b] = tot
    return S, float(levels[-1][0])


def class_hists(D, cs, e):
    return [np.histogram(np.clip(cs[D.cal_labels == c], 0, 1), bins=e)[0].astype(float) for c in range(D.C)]


def pcfcp(D, cs, e, eps, alpha, rng, read='topdown', local_noise=False, floor=True, margin_gamma=None,
          borrow=False):
    """floor: every class threshold is at least the top-down threshold of the pooled
    histogram, the sum of the released class histograms (post-processing, no extra budget).
    margin_gamma: coverage-bound variant, rank lowered by bound_margin(eps, B, alpha, gamma).
    read='tree': Hay et al. hierarchical release (ablation A8), budget split over levels.
    borrow: a class whose rank k_c < 1 (too few examples for a finite threshold) receives
    the pooled threshold instead of infinity (no extra budget, post-processing)."""
    B = len(e) - 1
    margin = bound_margin(eps, B, alpha, margin_gamma) if margin_gamma else 0
    thr, hs = [], []
    pool_suffix = None
    if read == 'tree':
        tree_S = []
    for c in range(D.C):
        clean = np.histogram(np.clip(cs[D.cal_labels == c], 0, 1), bins=e)[0].astype(float)
        if local_noise:          # A2: every client noises its own histogram before sending
            h = np.zeros(B)
            for k in range(D.K):
                m = (D.cal_client == k) & (D.cal_labels == c)
                h += np.histogram(np.clip(cs[m], 0, 1), bins=e)[0] + g.laplace(1.0 / eps, B, rng)
        elif read == 'tree':
            S, tot = tree_release(clean, eps, rng)
            h = np.concatenate([-np.diff(S), [S[-1]]])   # bins implied by the suffix sums (for the pooled floor)
            tree_S.append(S)
        else:
            h = clean + g.laplace(1.0 / eps, B, rng)
        hs.append(h)
        if read == 'topdown':
            thr.append(topdown(h, e, alpha, margin))
        elif read == 'tree':
            thr.append(topdown(h, e, alpha, margin, suffix=tree_S[-1]))
        elif read == 'bottomup':
            thr.append(readout(h, alpha, e, False)[0])
        elif read == 'consistency':
            thr.append(readout_consistent(h, alpha, e)[0])
        else:
            raise ValueError(read)
    thr = np.array(thr)
    pooled = topdown(np.sum(hs, axis=0), e, alpha, margin)
    if borrow:
        thr = np.where(np.isinf(thr), pooled, thr)
    return np.maximum(thr, pooled) if floor else thr


def pcfcp_noise_free(D, cs, e, alpha, floor=True):
    hs = class_hists(D, cs, e)
    thr = np.array([topdown(h, e, alpha) for h in hs])
    return np.maximum(thr, topdown(np.sum(hs, axis=0), e, alpha)) if floor else thr


def pcfcp_marginal_readout(D, cs, e, eps, alpha, rng):
    """A3: one pooled histogram of all classes, one threshold for every label."""
    B = len(e) - 1
    h = np.histogram(np.clip(cs, 0, 1), bins=e)[0] + g.laplace(1.0 / eps, B, rng)
    return np.full(D.C, topdown(h, e, alpha))


def pcfcp_adaptive(D, cs, eps, alpha, rng, s=0.2, c0=1.0, p=3):
    """Locked adaptive-bucket candidate (step5): noisy class count with share s of eps,
    B_c = clip(round(c0 (eps_h n_hat)^(2/3)), 5, 200) power-p buckets, top-down read-out
    with k from the released count."""
    thr = []
    eps_h = (1.0 - s) * eps
    for c in range(D.C):
        sc = cs[D.cal_labels == c]
        n_hat = len(sc) + g.laplace(1.0 / (s * eps), 1, rng)[0]
        B = int(np.clip(round(c0 * (eps_h * max(n_hat, 1.0)) ** (2.0 / 3.0)), 5, 200))
        e = edges_of('power', p, B)
        h = np.histogram(np.clip(sc, 0, 1), bins=e)[0] + g.laplace(1.0 / eps_h, B, rng)
        k = int(np.floor(alpha * (n_hat + 1) + 1e-9))
        if k < 1:
            thr.append(math.inf); continue
        S = np.cumsum(h[::-1])[::-1]
        hit = np.nonzero(S >= k)[0]
        thr.append(float(e[(int(hit[-1]) if len(hit) else 0) + 1]))
    return np.array(thr)


# ============================================================ references
def mondrian(D, cs, alpha):
    return np.array([g.fs_quantile(cs[D.cal_labels == c], alpha) for c in range(D.C)])


def clustered_ding(D, cs, alpha, rng):
    """Clustered CP (Ding et al., NeurIPS 2023) with the authors' tuning rule:
    n_tilde = max(n_min, 1/alpha - 1), K = #classes with >= n_tilde examples,
    gamma = K/(75+K), M = floor(gamma n_tilde / 2) (at least 1); finite-sample quantile
    embeddings at {0.5,...,0.9, 1-alpha}; classes with fewer than 1/min(alpha,0.1) - 1
    clustering examples go to the null cluster, which uses the marginal quantile of the
    proper calibration set; weighted k-means (weights |I1^y|^(1/2), sklearn KMeans)."""
    from sklearn.cluster import KMeans
    n_alpha = int(round(1.0 / alpha - 1))
    n_tilde = max(int(D.n_c.min()), n_alpha)
    Kc = int(np.sum(D.n_c >= n_tilde))
    gamma = Kc / (75.0 + Kc)
    M = max(1, int(np.floor(gamma * n_tilde / 2.0)))
    idx = np.arange(len(cs)); rng.shuffle(idx)
    n1 = int(np.floor(gamma * len(idx)))
    i1, i2 = idx[:n1], idx[n1:]
    s1, y1, s2, y2 = cs[i1], D.cal_labels[i1], cs[i2], D.cal_labels[i2]
    n_null = int(round(1.0 / min(alpha, 0.1) - 1))
    taus = [0.5, 0.6, 0.7, 0.8, 0.9, 1 - alpha]
    emb, keys, w = [], [], []
    for c in range(D.C):
        sc = np.sort(s1[y1 == c]); n = len(sc)
        if n < n_null:
            continue
        emb.append([sc[min(int(math.ceil((n + 1) * t)), n) - 1] for t in taus])
        keys.append(c); w.append(math.sqrt(n))
    assign = {c: -1 for c in range(D.C)}
    if keys:
        Mc = min(M, len(keys))
        km = KMeans(n_clusters=Mc, random_state=int(rng.randint(0, 2**31 - 1)), n_init=10)
        lab = km.fit_predict(np.array(emb), sample_weight=np.array(w))
        for c, l in zip(keys, lab):
            assign[c] = int(l)
    marg = g.fs_quantile(s2, alpha)
    thr = []
    for c in range(D.C):
        if assign[c] < 0:
            thr.append(marg); continue
        members = [cc for cc in range(D.C) if assign[cc] == assign[c]]
        thr.append(g.fs_quantile(s2[np.isin(y2, members)], alpha))
    return np.array(thr), M, gamma, (Mc if keys else 0)


# ============================================================ DP-FedCP
def dpfedcp_marginal(D, per_client, alpha, rng, eps=None, sigma=None):
    thr, _, _ = g.arm_plassier_dpfedcp(per_client, alpha, eps if eps else 1.0, rng, sigma_g_override=sigma)
    return np.array([thr[c] for c in range(D.C)])


def dpfedcp_perclass(D, per_client, alpha, rng, sigma):
    """DP-FedCP's private federated quantile estimation (Plassier et al., Alg. 1,
    reimplementation in baselines/plassier) run separately for each class, the
    'apply the federated protocol per group' construction stated by Wen et al.
    (GC-FCP). Within one class there is no label shift, so all weights are equal;
    the multinomial subsampling to floor(N/2) and the shuffled agent order are kept."""
    gen = np.random.default_rng(int(rng.randint(0, 2**31 - 1)))
    thr = []
    for c in range(D.C):
        S = []
        for cd in per_client:
            s = np.asarray(cd.get(c, np.array([])), dtype=float)
            S.append(s[gen.permutation(len(s))])
        N_i = np.array([len(s) for s in S], dtype=float)
        N = N_i.sum()
        if N < 1:
            thr.append(math.inf); continue
        keep = np.minimum(N_i, gen.multinomial(int(N // 2), N_i / N)).astype(int)
        kept = [S[i][:keep[i]] for i in range(len(S))]
        denom = keep.sum() + 1.0
        agent_w = [np.full(len(k_), 1.0 / denom) for k_ in kept]
        delta1 = (N_i / N) * (1.0 / denom)
        lam = delta1 + keep / denom
        lam = lam / lam.sum()
        thr.append(P.dp_fedavg_qe(kept, agent_w, delta1, lam, alpha, sigma_g=sigma, rng=gen))
    return np.array(thr)


# ============================================================ FedCP-QQ
_KCACHE = {}


def qq_k(ns, alpha, level=None, ell_shift=0.0, draws=20000):
    """Humbert et al. App. A.1 unequal-size FedCP-QQ: l_j = ceil(level (n_j+1)) (+ shift),
    agent j's l_j-th smallest score is Beta(l_j, n_j - l_j + 1) for continuous i.i.d. scores
    (+inf if l_j > n_j); the coverage of the k-th smallest is E[Q_(k)]. Returns
    (ells, k, coverage) with the smallest k whose coverage >= level, or None.
    Exact up to Monte Carlo error (the authors evaluate the same quantity in closed form)."""
    level = (1 - alpha) if level is None else level
    key = (tuple(ns), round(level, 6), round(ell_shift, 6))   # order-sensitive: ells follow ns
    if key in _KCACHE:
        return _KCACHE[key]
    rng = np.random.default_rng(12345)
    ells = [int(math.ceil(level * (n + 1) + ell_shift)) for n in ns]
    Q = np.ones((draws, len(ns)))
    for j, (n, l) in enumerate(zip(ns, ells)):
        if 1 <= l <= n:
            Q[:, j] = rng.beta(l, n - l + 1, draws)
    covs = np.sort(Q, axis=1).mean(0)
    ok = np.nonzero(covs >= level)[0]
    res = None if len(ok) == 0 else (ells, int(ok[0]) + 1, float(covs[ok[0]]))
    _KCACHE[key] = res
    return res


def qq_cov(ns, ells, k, draws=20000):
    """Coverage E[Q_(k)] of the k-th smallest of the agents' ell_j-th smallest scores (k fixed)."""
    key = ('cov', tuple(ns), tuple(ells), k)
    if key in _KCACHE:
        return _KCACHE[key]
    rng = np.random.default_rng(12345)
    Q = np.ones((draws, len(ns)))
    for j, (n, l) in enumerate(zip(ns, ells)):
        if 1 <= l <= n:
            Q[:, j] = rng.beta(l, n - l + 1, draws)
    cov = float(np.sort(Q, axis=1).mean(0)[k - 1])
    _KCACHE[key] = cov
    return cov


def em_quantile(scores, q, eps, e, rng):
    """Humbert et al. Alg. 2 (after Angelopoulos et al.): exponential mechanism over bins."""
    ub = e[1:]
    sbar = ub[np.clip(np.searchsorted(e, np.clip(scores, 0, 1), side='left') - 1, 0, len(ub) - 1)]
    below = np.array([(sbar < x).sum() for x in ub]); above = np.array([(sbar > x).sum() for x in ub])
    w = np.maximum(below / q, above / (1 - q))
    dq = max(1.0 / q, 1.0 / (1.0 - q))
    logits = -eps * w / (2 * dq)
    pr = np.exp(logits - logits.max()); pr /= pr.sum()
    return float(ub[rng.choice(len(ub), p=pr)])


def fedcpqq_group(scores_by_agent, alpha, eps=None, e=None, rng=None):
    """One FedCP-QQ threshold from a list of per-agent score arrays (agents with >= 1 score).
    eps None -> non-private FedCP-QQ (unequal sizes); else FedCP2-QQ (Humbert Alg. 3, LDP)."""
    S = [np.asarray(s) for s in scores_by_agent if len(s) >= 1]
    if not S:
        return math.inf
    ns = [len(s) for s in S]; m = len(S)
    if eps is None:
        res = qq_k(ns, alpha)
        if res is None:
            return math.inf
        ells, k, _ = res
        loc = sorted(np.sort(s)[l - 1] if l <= len(s) else math.inf for s, l in zip(S, ells))
        return float(loc[k - 1])
    B = len(e) - 1
    best = None
    for gam in np.arange(0.05, 0.951, 0.05):
        level = (1 - alpha) / (1 - gam * alpha)
        base = qq_k(ns, alpha, level=level)
        if base is None:
            continue
        lcor = (2.0 / eps) * math.log(B / (1 - (1 - gam * alpha) ** (1.0 / m)))
        ells, k, _ = base
        # Humbert et al. Sec. 4 'Choosing gamma': coverage of the non-private estimator with ranks
        # l_gamma + l_cor at the SAME k_gamma (ranks rounded up to integers; the paper does not say)
        cov_shift = qq_cov(ns, [int(math.ceil(l + lcor)) for l in ells], k)
        if best is None or cov_shift < best[0]:
            best = (cov_shift, gam, ells, k, lcor)
    if best is None:
        return math.inf
    _, gam, ells, k, lcor = best
    loc = []
    for s, l in zip(S, ells):
        q = max((l + lcor) / len(s), 0.5)
        loc.append(math.inf if q >= 1.0 else em_quantile(s, q, eps, e, rng))
    return float(sorted(loc)[k - 1])


def fedcpqq(D, per_client, alpha, per_class, eps=None, e=None, rng=None):
    if per_class:
        return np.array([fedcpqq_group([cd.get(c, np.array([])) for cd in per_client], alpha, eps, e, rng)
                         for c in range(D.C)])
    allsc = [np.concatenate(list(cd.values())) if cd else np.array([]) for cd in per_client]
    return np.full(D.C, fedcpqq_group(allsc, alpha, eps, e, rng))


# ============================================================ jobs
def jobs_for(tag):
    """Method groups to run for one cache tag."""
    main = ('K10_dir0.5' in tag)
    iid = 'dir100' in tag
    scale = ('K50' in tag) or ('K100' in tag)
    groups = ['refs', 'pcfcp', 'qq', 'variants']
    if main:
        groups += ['ablations', 'alpha', 'dpfedcp23']
    if iid:
        groups += ['dpfedcp23']
    if scale:
        groups += ['a2']
    return groups


def rs(D, method, eps, alpha, r):
    """Independent noise stream per (seed, method, eps, alpha, draw). Methods compared on
    purpose under the same noise (read-out and grid ablations) pass method='pcfcp'."""
    e_ = 0 if eps in ('inf', None) else int(round(float(eps) * 10))
    ss = np.random.SeedSequence([D.seed, zlib.crc32(method.encode()), e_, int(round(alpha * 1000)), r])
    return np.random.RandomState(ss.generate_state(1)[0])


def run_group(args):
    path, group, cfg = args
    global R_FAST, R_SLOW, OUT
    R_FAST, R_SLOW, OUT = cfg['r_fast'], cfg['r_slow'], cfg['out']
    D = Data(path, cfg.get('cal_frac'))
    D.alpha = 0.1
    g.N_CLASSES = D.C
    g.TAIL_CLASSES = D.tail
    rows = []
    t0 = time.time()

    def rec(method, eps, alpha, draw, thr, ta, **extra):
        D.alpha = alpha
        r = {'tag': D.tag, 'seed': D.seed, 'C': D.C, 'K': D.K, 'method': method, 'eps': eps,
             'alpha': alpha, 'draw': draw}
        r.update(extra)
        r.update(evaluate(D, thr, ta))
        rows.append(r)
        D.alpha = 0.1

    cs, ta = D.score('aps')
    e = edges_of(*MAIN_GRID)
    eq = edges_of(*QQ_GRID)
    marg_thr = g.fs_quantile(cs, 0.1)
    pc = D.per_client(cs)

    if group == 'refs':
        rec('mondrian', 'inf', 0.1, 0, mondrian(D, cs, 0.1), ta)
        thr, M, gam, Mc = clustered_ding(D, cs, 0.1, np.random.RandomState(4321 + D.seed))
        rec('clustered_ding', 'inf', 0.1, 0, thr, ta, clusters=M, clusters_used=Mc, gamma=round(gam, 4))
        rec('pcfcp_noise_free', 'inf', 0.1, 0, pcfcp_noise_free(D, cs, e, 0.1), ta)
        rec('pcfcp_noise_free_nofloor', 'inf', 0.1, 0, pcfcp_noise_free(D, cs, e, 0.1, floor=False), ta)

    elif group == 'pcfcp':
        for eps in EPS:
            for r in range(R_FAST):
                rec('pcfcp', eps, 0.1, r, pcfcp(D, cs, e, eps, 0.1, rs(D, 'pcfcp', eps, 0.1, r)), ta)

    elif group == 'ablations':
        cl_, tl_ = D.score('lac')
        rec('mondrian_lac', 'inf', 0.1, 0, mondrian(D, cl_, 0.1), tl_)
        rec('pcfcp_lac_noise_free', 'inf', 0.1, 0, pcfcp_noise_free(D, cl_, e, 0.1), tl_)
        grids = {'p3_B5': ('power', 3, 5), 'p3_B12': ('power', 3, 12), 'p3_B20': ('power', 3, 20),
                 'p3_B50': ('power', 3, 50), 'p3_B100': ('power', 3, 100), 'p2_B8': ('power', 2, 8),
                 'p2_B100': ('power', 2, 100), 'uniform50': ('uniform', 1, 50)}
        for name, gr in grids.items():
            rec(f'pcfcp_{name}_noise_free', 'inf', 0.1, 0, pcfcp_noise_free(D, cs, edges_of(*gr), 0.1), ta)
        for eps in EPS:
            for r in range(R_FAST):
                def same():   # paired with the main PC-FCP draw
                    return rs(D, 'pcfcp', eps, 0.1, r)
                rec('abl_nofloor', eps, 0.1, r, pcfcp(D, cs, e, eps, 0.1, same(), floor=False), ta)
                rec('abl_bottomup', eps, 0.1, r, pcfcp(D, cs, e, eps, 0.1, same(), read='bottomup'), ta)
                rec('abl_consistency', eps, 0.1, r, pcfcp(D, cs, e, eps, 0.1, same(), read='consistency'), ta)
                rec('abl_lac', eps, 0.1, r, pcfcp(D, cl_, e, eps, 0.1, same()), tl_)
                rec('abl_A2_local_noise', eps, 0.1, r, pcfcp(D, cs, e, eps, 0.1, rs(D, 'abl_A2', eps, 0.1, r), local_noise=True), ta)
                rec('abl_A3_marginal', eps, 0.1, r, pcfcp_marginal_readout(D, cs, e, eps, 0.1, rs(D, 'abl_A3', eps, 0.1, r)), ta)
                ta1 = g.arm_naive_dp(pc, 0.1, eps, rs(D, 'abl_A1_avg', eps, 0.1, r))
                rec('abl_A1_avg', eps, 0.1, r, np.array([ta1[c] for c in range(D.C)]), ta)
                # threshold and count are two releases: eps/2 each, so the row is eps-DP overall
                ta2 = g.arm_naive_dp_weighted(pc, 0.1, eps / 2, rs(D, 'abl_A1_weighted', eps, 0.1, r))
                rec('abl_A1_weighted', eps, 0.1, r, np.array([ta2[c] for c in range(D.C)]), ta)
                for name, gr in grids.items():
                    rec(f'pcfcp_{name}', eps, 0.1, r, pcfcp(D, cs, edges_of(*gr), eps, 0.1, same()), ta)

    elif group == 'a2':
        for eps in EPS:
            for r in range(R_FAST):
                rec('abl_A2_local_noise', eps, 0.1, r, pcfcp(D, cs, e, eps, 0.1, rs(D, 'abl_A2', eps, 0.1, r), local_noise=True), ta)

    elif group == 'alpha':
        for alpha in ALPHAS_SWEEP:
            rec('mondrian', 'inf', alpha, 0, mondrian(D, cs, alpha), ta)
            rec('pcfcp_noise_free', 'inf', alpha, 0, pcfcp_noise_free(D, cs, e, alpha), ta)
            rec('fedcpqq_pc', 'inf', alpha, 0, fedcpqq(D, pc, alpha, True), ta)
            for eps in (1.0, 4.0):
                for r in range(R_FAST):
                    rec('pcfcp', eps, alpha, r, pcfcp(D, cs, e, eps, alpha, rs(D, 'pcfcp', eps, alpha, r)), ta)
                    rec('fedcp2qq_marg', eps, alpha, r, fedcpqq(D, pc, alpha, False, eps, eq, rs(D, 'fedcp2qq_marg', eps, alpha, r)), ta)

    elif group == 'qq':
        t_pc = fedcpqq(D, pc, 0.1, True)
        rec('fedcpqq_pc', 'inf', 0.1, 0, t_pc, ta)
        rec('fedcpqq_pc_fallback', 'inf', 0.1, 0, np.where(np.isinf(t_pc), marg_thr, t_pc), ta)
        rec('fedcpqq_marg', 'inf', 0.1, 0, fedcpqq(D, pc, 0.1, False), ta)
        for eps in EPS:
            for r in range(R_FAST):
                t2 = fedcpqq(D, pc, 0.1, True, eps, eq, rs(D, 'fedcp2qq_pc', eps, 0.1, r))
                rec('fedcp2qq_pc', eps, 0.1, r, t2, ta)
                rec('fedcp2qq_pc_fallback', eps, 0.1, r, np.where(np.isinf(t2), marg_thr, t2), ta)
                rec('fedcp2qq_marg', eps, 0.1, r, fedcpqq(D, pc, 0.1, False, eps, eq, rs(D, 'fedcp2qq_marg', eps, 0.1, r)), ta)
                # item 4: private fallback. The per-class releases and the marginal release
                # each get eps/2 (the marginal release touches every example, so the two do
                # not compose in parallel); classes with an infinite per-class threshold take
                # the private marginal one.
                r_ = rs(D, 'fedcp2qq_pc_pfallback', eps, 0.1, r)
                t2h = fedcpqq(D, pc, 0.1, True, eps / 2, eq, r_)
                tmh = fedcpqq(D, pc, 0.1, False, eps / 2, eq, r_)
                rec('fedcp2qq_pc_pfallback', eps, 0.1, r, np.where(np.isinf(t2h), tmh, t2h), ta)

    elif group == 'variants':
        # BATCH_PLAN_2026-10-19 items 6 to 8, paired with the main PC-FCP noise draws
        thr_nf = pcfcp_noise_free(D, cs, e, 0.1)
        pool_nf = topdown(np.sum(class_hists(D, cs, e), axis=0), e, 0.1)
        rec('pcfcp_borrow_noise_free', 'inf', 0.1, 0, np.where(np.isinf(thr_nf), pool_nf, thr_nf), ta)
        for eps in EPS:
            for r in range(R_FAST):
                def same():
                    return rs(D, 'pcfcp', eps, 0.1, r)
                rec('pcfcp_margin', eps, 0.1, r, pcfcp(D, cs, e, eps, 0.1, same(), margin_gamma=0.05), ta,
                    margin=int(bound_margin(eps, len(e) - 1, 0.1, 0.05)),
                    t=int(laplace_sum_margin(eps, len(e) - 1, 0.05)))
                rec('abl_A8_tree', eps, 0.1, r, pcfcp(D, cs, e, eps, 0.1, same(), read='tree'), ta)
                rec('pcfcp_borrow', eps, 0.1, r, pcfcp(D, cs, e, eps, 0.1, same(), borrow=True), ta)

    elif group.startswith('dpfedcp'):
        # 'dpfedcp' runs every draw; 'dpfedcp_r<r>' runs draw r only (same seeds, so the
        # split is only for parallelism: 100-class DP-FedCP takes hours per draw)
        draws = range(R_SLOW) if group in ('dpfedcp', 'dpfedcp23') else [int(group.split('_r')[1])]
        only23 = group.startswith('dpfedcp23')
        for r in draws:
            if only23:
                for eps in (2.0, 3.0):
                    sg = g.sigma_g_from_eps(eps)
                    rec('dpfedcp_marg', eps, 0.1, r, dpfedcp_marginal(D, pc, 0.1, rs(D, 'dpfedcp_marg', eps, 0.1, r), eps=eps), ta, sigma_g=round(sg, 2))
                    rec('dpfedcp_pc', eps, 0.1, r, dpfedcp_perclass(D, pc, 0.1, rs(D, 'dpfedcp_pc', eps, 0.1, r), sg), ta, sigma_g=round(sg, 2))
                continue
            rec('dpfedcp_marg_nonoise', 'inf', 0.1, r, dpfedcp_marginal(D, pc, 0.1, rs(D, 'dpfedcp_marg_nonoise', 'inf', 0.1, r), sigma=0.0), ta)
            rec('dpfedcp_marg_sg100', 'sg100', 0.1, r, dpfedcp_marginal(D, pc, 0.1, rs(D, 'dpfedcp_marg_sg100', 'inf', 0.1, r), sigma=100.0), ta)
            rec('dpfedcp_pc_nonoise', 'inf', 0.1, r, dpfedcp_perclass(D, pc, 0.1, rs(D, 'dpfedcp_pc_nonoise', 'inf', 0.1, r), 0.0), ta)
            rec('dpfedcp_pc_sg100', 'sg100', 0.1, r, dpfedcp_perclass(D, pc, 0.1, rs(D, 'dpfedcp_pc_sg100', 'inf', 0.1, r), 100.0), ta)
            for eps in (1.0, 4.0, 8.0):
                sg = g.sigma_g_from_eps(eps)
                rec('dpfedcp_marg', eps, 0.1, r, dpfedcp_marginal(D, pc, 0.1, rs(D, 'dpfedcp_marg', eps, 0.1, r), eps=eps), ta, sigma_g=round(sg, 2))
                rec('dpfedcp_pc', eps, 0.1, r, dpfedcp_perclass(D, pc, 0.1, rs(D, 'dpfedcp_pc', eps, 0.1, r), sg), ta, sigma_g=round(sg, 2))
    else:
        raise ValueError(group)

    os.makedirs(OUT, exist_ok=True)
    out = os.path.join(OUT, f'{D.tag}__{group}.csv')
    import csv
    fields = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval='')
        w.writeheader(); w.writerows(rows)
    return f'{D.tag}__{group}: {len(rows)} rows in {time.time() - t0:.0f}s'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='', help='comma-separated cache tags (default: all)')
    ap.add_argument('--groups', default='', help='only these groups')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--cache', default=CACHE)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--r-fast', type=int, default=R_FAST)
    ap.add_argument('--r-slow', type=int, default=R_SLOW)
    ap.add_argument('--cal-frac', type=float, default=None,
                    help='keep this fraction of each class\'s calibration rows (item 1 of the batch plan)')
    a = ap.parse_args()
    cfg = {'r_fast': a.r_fast, 'r_slow': a.r_slow, 'out': a.out, 'cal_frac': a.cal_frac}
    paths = sorted(glob.glob(os.path.join(a.cache, '*.npz')))
    if a.tags:
        want = set(a.tags.split(','))
        paths = [p for p in paths if os.path.basename(p)[:-4] in want]
    tasks = []
    for p in paths:
        # an explicit --groups list is run as given (it may name groups that jobs_for does
        # not propose, such as 'dpfedcp'); without it, jobs_for decides per cache tag
        for grp in (a.groups.split(',') if a.groups else jobs_for(os.path.basename(p))):
            if True:
                if grp in ('dpfedcp', 'dpfedcp23'):
                    tasks += [(p, f'{grp}_r{r}', cfg) for r in range(a.r_slow)]
                else:
                    tasks.append((p, grp, cfg))
    tasks.sort(key=lambda t: (not t[1].startswith('dpfedcp'), 'cifar100' not in t[0]))   # slowest first
    print(f'[v3_eval] {len(tasks)} tasks on {a.workers} workers', flush=True)
    with mp.Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run_group, tasks):
            print('  ' + msg, flush=True)


if __name__ == '__main__':
    main()
