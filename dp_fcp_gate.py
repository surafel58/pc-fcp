"""Constrained-channel Federated Conformal Prediction — headroom gate.

Question: when the per-class calibration-score channel is DP-constrained, does a
smarter server-side aggregation (DP histogram of scores -> pooled quantile) beat
the naive baseline (DP-noised local thresholds -> average) on TAIL-class coverage?

This is the live opening the original fed-cp-gate flagged: naive score-list
CONCATENATION equals centralized only when the channel is UNCONSTRAINED. Under a
DP budget on what each client may reveal, naive aggregation degrades — and the
question is whether a better estimator recovers the tail coverage it loses.

Arms (all evaluated on the standard balanced CIFAR-10 test set):
  - CENTRALIZED   : pool all clients' calibration scores, exact per-class APS
                    quantile. Unconstrained oracle ceiling (privacy-violating).
  - NAIVE-DP(eps) : each client computes its local per-class APS threshold and
                    sends a Laplace-noised version (sensitivity=1 on the [0,1]
                    score range); server averages over clients with >=1 sample.
                    The deployable naive constrained baseline.
  - HIST-DP(eps)  : each client sends a Laplace-noised per-class histogram of its
                    scores (B bins over [0,1]); server sums histograms across
                    clients -> pooled noisy histogram -> finite-sample APS
                    quantile from the pooled counts. The "smart" method.
  - LOCAL         : each client calibrates from its own data only (marginal
                    fallback if <2 samples for a class); per-client tail coverage
                    averaged. Screen 2 (federation-needed) check.

Setup: CIFAR-10-LT (IF=100), FedAvg ResNet-8-GN, Dir(0.5) across 10 clients.
Calibration pool split across the SAME 10 clients via Dir(0.5) so tail classes
are sparse per client (the regime where naive quantile estimation fails).

DP note: eps is per-class-per-client query budget (not a tight global account).
This is a HEADROOM gate — NAIVE and HIST are compared at the SAME eps, so the
relative comparison is fair; global DP accounting is a build-stage concern.
"""
import argparse
import copy
import csv
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


NORM_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
NORM_STD  = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
# N_CLASSES and TAIL_CLASSES are set at runtime in main() from --dataset.
N_CLASSES = 10
TAIL_CLASSES = [7, 8, 9]   # bottom third by LT count; recomputed in main()
ALPHA = 0.10               # target coverage 1 - alpha = 0.90


# ---------------- model: ResNet-8 with GroupNorm ----------------
class GNBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, gn_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, out_c)
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                nn.GroupNorm(gn_groups, out_c))
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet8GN(nn.Module):
    def __init__(self, n_classes=10, gn_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, 32)
        self.b1 = GNBlock(32, 32, stride=1, gn_groups=gn_groups)
        self.b2 = GNBlock(32, 64, stride=2, gn_groups=gn_groups)
        self.b3 = GNBlock(64, 128, stride=2, gn_groups=gn_groups)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        return self.fc(self.pool(x).flatten(1))


# ---------------- data ----------------
class _CachedCIFAR(torch.utils.data.Dataset):
    def __init__(self, images_uint8, targets, train=True):
        self.images = images_uint8
        self.targets = np.asarray(targets, dtype=np.int64)
        self.train = train

    def __len__(self): return len(self.targets)

    def __getitem__(self, i):
        img = self.images[i].to(torch.float32) / 255.0
        if self.train:
            pad = F.pad(img.unsqueeze(0), (4, 4, 4, 4), mode='reflect').squeeze(0)
            top = torch.randint(0, 9, (1,)).item()
            left = torch.randint(0, 9, (1,)).item()
            img = pad[:, top:top + 32, left:left + 32]
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[-1])
        img = (img - NORM_MEAN) / NORM_STD
        return img, int(self.targets[i])


def load_cifar_cached(cache_path):
    d = torch.load(cache_path, weights_only=False)
    return (_CachedCIFAR(d['train_imgs'], d['train_targets'], train=True),
            _CachedCIFAR(d['test_imgs'], d['test_targets'], train=False))


def lt_indices(targets, if_factor, n_max=5000, seed=0):
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    mu = if_factor ** (-1.0 / (N_CLASSES - 1))
    per_class = [max(5, int(round(n_max * (mu ** i)))) for i in range(N_CLASSES)]
    kept = []
    for c in range(N_CLASSES):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        kept.extend(idx[:per_class[c]].tolist())
    return kept, per_class


def dirichlet_partition(indices, targets_arr, n_clients, alpha, seed):
    rng = np.random.RandomState(seed)
    idx_arr = np.asarray(indices)
    y = targets_arr[idx_arr]
    lists = [[] for _ in range(n_clients)]
    for c in range(N_CLASSES):
        pool = idx_arr[y == c].tolist()
        rng.shuffle(pool)
        if not pool:
            continue
        props = rng.dirichlet([alpha] * n_clients)
        counts = (props * len(pool)).astype(int)
        leftover = len(pool) - counts.sum()
        for _ in range(leftover):
            counts[rng.randint(n_clients)] += 1
        ptr = 0
        for i, k in enumerate(counts):
            lists[i].extend(pool[ptr:ptr + k]); ptr += k
    for lst in lists:
        rng.shuffle(lst)
    return lists


# ---------------- FedAvg training ----------------
def local_train(model, ds, opt, device, epochs=1, batch=64):
    model.train()
    dl = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    for _ in range(epochs):
        for x, y in dl:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()


def aggregate(global_model, states, weights):
    total = sum(weights)
    with torch.no_grad():
        avg = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in states[0].items()}
        for cs, w in zip(states, weights):
            for k, v in cs.items():
                avg[k] += (w / total) * v.to(torch.float32)
        gs = global_model.state_dict()
        for k in avg:
            avg[k] = avg[k].to(gs[k].dtype)
        global_model.load_state_dict(avg)


def train_fedavg(client_ds, device, n_rounds, local_epochs, lr, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ResNet8GN(N_CLASSES).to(device)
    for rnd in range(1, n_rounds + 1):
        cur_lr = lr * (0.99 ** (rnd - 1))
        states = []; weights = []
        for ds in client_ds:
            if len(ds) == 0:
                continue
            m = copy.deepcopy(model)
            opt = torch.optim.SGD(m.parameters(), lr=cur_lr, momentum=0.9, weight_decay=5e-4)
            local_train(m, ds, opt, device, epochs=local_epochs)
            states.append({k: v.detach().cpu() for k, v in m.state_dict().items()})
            weights.append(len(ds))
            del m, opt
        aggregate(model, states, weights)
    return model


@torch.no_grad()
def softmax_on(model, ds, device):
    model.eval()
    outs = []; labels = []
    dl = DataLoader(ds, batch_size=512, num_workers=0, pin_memory=True)
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        outs.append(F.softmax(model(x), dim=-1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(outs), np.concatenate(labels)


# ---------------- APS conformal ----------------
def aps_score_true(probs, labels, rng):
    N = probs.shape[0]
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    rank_true = np.argmax(order == labels[:, None], axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    U = rng.random(N)
    prob_at_true = np.take_along_axis(sorted_probs, rank_true[:, None], axis=1).squeeze(1)
    cum_before = np.where(rank_true > 0,
                          np.take_along_axis(cumsum, np.maximum(0, rank_true - 1)[:, None], axis=1).squeeze(1),
                          0.0)
    return cum_before + U * prob_at_true


def aps_score_all(probs, rng):
    N, C = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    inv_order = np.argsort(order, axis=1)
    U = rng.random((N, C))
    r_shift = np.maximum(0, inv_order - 1)
    cum_at = np.take_along_axis(cumsum, r_shift, axis=1)
    cum_before = np.where(inv_order > 0, cum_at, 0.0)
    return cum_before + U * probs


def fs_quantile(scores, alpha):
    n = len(scores)
    if n < 1:
        return float('inf')
    idx = math.ceil((n + 1) * (1 - alpha))
    if idx > n:
        return float('inf')
    return float(np.sort(scores)[idx - 1])


def fs_quantile_from_hist(counts, bin_edges, alpha):
    """Finite-sample APS quantile from a (possibly noisy) histogram of scores.
    counts: [B] nonneg. Returns the score value whose cumulative count crosses
    ceil((n+1)(1-alpha))."""
    counts = np.maximum(counts, 0.0)
    n = counts.sum()
    if n < 1:
        return float('inf')
    target = math.ceil((n + 1) * (1 - alpha))
    if target > n:
        return float('inf')
    cum = np.cumsum(counts)
    b = int(np.searchsorted(cum, target))
    b = min(b, len(bin_edges) - 2)
    return float(bin_edges[b + 1])   # upper edge of the bin that crosses target


# ---------------- arms ----------------
def eval_coverage(thr_per_class, test_probs, test_labels, rng, marginal_thr=None):
    C = N_CLASSES
    thr = np.array([
        (thr_per_class[c] if not (isinstance(thr_per_class[c], float) and math.isinf(thr_per_class[c]))
         else (marginal_thr if marginal_thr is not None else float('inf')))
        for c in range(C)], dtype=np.float64)
    all_scores = aps_score_all(test_probs, rng)
    included = all_scores <= thr[None, :]
    set_size = included.sum(axis=1)
    cov = np.zeros(C); size = np.zeros(C)
    for c in range(C):
        m = test_labels == c
        cov[c] = included[m, c].mean() if m.any() else float('nan')
        size[c] = set_size[m].mean() if m.any() else float('nan')
    marginal = included[np.arange(len(test_labels)), test_labels].mean()
    return cov, size, float(marginal)


def laplace(scale, size, rng):
    return rng.laplace(0.0, scale, size=size)


def build_client_scores(cal_probs, cal_labels, cal_client_lists, cal_index_of, rng):
    """Return per-client dict: client -> {class -> np.array of scores}."""
    scores = aps_score_true(cal_probs, cal_labels, rng)   # aligned to cal_probs rows
    per_client = []
    for lst in cal_client_lists:
        by_class = defaultdict(list)
        for global_idx in lst:
            row = cal_index_of[global_idx]
            by_class[int(cal_labels[row])].append(scores[row])
        per_client.append({c: np.asarray(v) for c, v in by_class.items()})
    return per_client, scores


def arm_centralized(all_scores, cal_labels, alpha):
    thr = {}
    for c in range(N_CLASSES):
        thr[c] = fs_quantile(all_scores[cal_labels == c], alpha)
    return thr


def arm_naive_dp(per_client, alpha, eps, rng):
    """Each client sends Laplace-noised local per-class threshold (sensitivity=1 on
    [0,1]); server averages over clients with >=1 sample for that class."""
    thr = {}
    for c in range(N_CLASSES):
        vals = []
        for cd in per_client:
            s = cd.get(c, np.array([]))
            if len(s) >= 1:
                t = fs_quantile(s, alpha)
                if math.isinf(t):
                    t = 1.0   # infinite local threshold -> clip to max score range
                t_noisy = t + laplace(1.0 / eps, 1, rng)[0]
                vals.append(min(max(t_noisy, 0.0), 1.0))
        thr[c] = float(np.mean(vals)) if vals else float('inf')
    return thr


def arm_naive_dp_weighted(per_client, alpha, eps, rng):
    """Fairer naive baseline: clients send a Laplace-noised local threshold AND
    their (noised) sample count; server does a COUNT-WEIGHTED average, which
    down-weights the degenerate single-sample clients. Stronger than plain naive."""
    thr = {}
    for c in range(N_CLASSES):
        vals = []; wts = []
        for cd in per_client:
            s = cd.get(c, np.array([]))
            if len(s) >= 1:
                t = fs_quantile(s, alpha)
                if math.isinf(t):
                    t = 1.0
                t_noisy = min(max(t + laplace(1.0 / eps, 1, rng)[0], 0.0), 1.0)
                w_noisy = max(len(s) + laplace(1.0 / eps, 1, rng)[0], 0.1)
                vals.append(t_noisy); wts.append(w_noisy)
        if vals:
            vals = np.asarray(vals); wts = np.asarray(wts)
            thr[c] = float((vals * wts).sum() / wts.sum())
        else:
            thr[c] = float('inf')
    return thr


# --- Authors' exact FedCP-QQ order-statistic index (Humbert et al. ICML'23) ---
# Ported verbatim from https://github.com/pierreHmbt/FedCP-QQ (class_CP_QQ.py).
# calc_matrix_M(m, n, t, ...) returns matrix A[aa,bb] = coverage prob when the server
# takes the (aa+1)-th order statistic across m agents, each reporting its (bb+1)-th
# local order statistic of n calibration scores. FedCP-QQ picks the (r*,k*) with the
# smallest sets s.t. A >= 1-alpha. NOTE: assumes uniform n across the m agents.
from scipy.stats import binom as _binom
import scipy.special as _sc
_M_CACHE = {}

def _sum_hypergeo(aa, bb, n, t, epsi=0):
    m = len(aa)
    rv1 = _binom(n, t)
    prob = [rv1.cdf(bb[i]) - rv1.cdf(aa[i] - 1) for i in range(m)]
    r1 = 1.0
    for p in prob:
        r1 *= p
    rv2 = _binom(m * n, t)
    gen = []
    for i in range(m):
        tt = rv1.cdf(bb[i]) - rv1.cdf(aa[i] - 1)
        pp = []
        for j in np.arange(n, -1, -1):
            if j <= bb[i] and j >= aa[i] and tt > 0:
                pp.append(rv1.pmf(j) / tt)
            else:
                pp.append(0)
        gen.append(np.poly1d(pp))
    poly = gen[0]
    for i in range(1, len(gen)):
        poly = np.polymul(poly, gen[i])
    s = 0
    for l in range(int(np.sum(aa)), int(np.sum(bb)) + 1):
        r2 = rv2.pmf(l)
        if r2 > epsi:
            s += poly[l] * r1 / r2
    return s

def _calc_matrix_M(m, n, t=0.5, epsi=0.0, mid=False):
    A = np.zeros((m, n))
    for aa in np.arange((m // 2) * mid, m):
        for bb in np.arange((n // 2) * mid, n):
            r = aa + 1; k = bb + 1
            s = 0
            for j in range(r, m + 1):
                a_range = np.ones(m, dtype=int) * k
                a_range[j:] = 0
                b_range = np.ones(m, dtype=int) * n
                b_range[j:] = k - 1
                s += _sum_hypergeo(a_range, b_range, n, 0.5, epsi) * _sc.binom(m, j)
            A[aa, bb] = 1 - s / (m * n + 1)
    return A

def fedcpqq_indices(m, n, alpha, n_cap=12, m_cap=25):
    """Return (r*, k*) 1-based: server takes r*-th order stat across m agents, each
    reporting its k*-th local order stat of n scores, chosen as the authors do --
    smallest expected set (largest k, then smallest r) with coverage >= 1-alpha.

    calc_matrix_M is expensive for large (m,n) (authors precompute it once). The
    finite-sample correction only differs from the standard index for SMALL n (the
    tail regime that matters here); for large n/m it converges to the standard
    quantile index. So we use the AUTHORS' EXACT index for n<=n_cap & m<=m_cap
    (the tail), and the standard finite-sample index otherwise (negligible diff)."""
    if m < 1 or n < 1:
        return None
    key = (m, n, round(alpha, 4))
    if key in _M_CACHE:
        return _M_CACHE[key]
    if n > n_cap or m > m_cap:
        # standard finite-sample order-statistic indices (asymptotically = authors')
        k = math.ceil((n + 1) * (1 - alpha)); r = math.ceil((m + 1) * (1 - alpha))
        res = (min(r, m), min(k, n)) if (k <= n and r <= m) else None
        _M_CACHE[key] = res
        return res
    A = _calc_matrix_M(m, n)
    valid = np.argwhere(A >= (1 - alpha))
    if len(valid) == 0:
        res = None
    else:
        bb = valid[:, 1].max()
        aa = valid[valid[:, 1] == bb][:, 0].min()
        res = (int(aa + 1), int(bb + 1))
    _M_CACHE[key] = res
    return res


def arm_fedcpqq(per_client, alpha, eps, rng, ldp=True):
    """FAITHFUL FedCP-QQ (Humbert et al. ICML'23) using the AUTHORS' exact order-
    statistic index from their calc_matrix_M. Per class: the m clients holding >= n_min
    samples each report their k*-th local order statistic (LDP-noised if ldp); server
    takes the r*-th order statistic across them. Uses the min common count as n so the
    uniform-n assumption of calc_matrix_M holds; classes where too few clients have
    enough samples fall back to +inf (predict-all) -- the honest tail degeneration."""
    thr = {}
    for c in range(N_CLASSES):
        counts = [len(cd.get(c, [])) for cd in per_client]
        active = [(cd.get(c, np.array([])), len(cd.get(c, []))) for cd in per_client if len(cd.get(c, [])) >= 1]
        m = len(active)
        if m < 1:
            thr[c] = float('inf'); continue
        n_min = min(k for _, k in active)
        idx = fedcpqq_indices(m, n_min, alpha)
        if idx is None:
            thr[c] = float('inf'); continue   # authors' estimator cannot certify coverage here
        r_star, k_star = idx
        local_qs = []
        for s, _ in active:
            s_sorted = np.sort(np.clip(s, 0, 1))
            q = float(s_sorted[min(k_star - 1, len(s_sorted) - 1)])   # k*-th local order stat
            if ldp:
                q = min(max(q + laplace(1.0 / eps, 1, rng)[0], 0.0), 1.0)
            local_qs.append(q)
        local_qs = np.sort(local_qs)
        thr[c] = float(local_qs[min(r_star - 1, len(local_qs) - 1)])   # r*-th across agents
    return thr


def arm_qq_ldp(per_client, alpha, eps, rng, ldp=True):
    """Humbert et al. (ICML'23) FedCP-QQ 'quantile-of-quantiles', per class, with
    the LDP variant (Laplace-noised local quantiles). Each client sends its local
    (1-alpha) empirical quantile of its class-c scores (LDP-noised if ldp=True);
    server takes the finite-sample (1-alpha) order statistic across the per-client
    quantiles. This is the principled per-client-summary DP baseline — the real
    must-beat, stronger than plain threshold-averaging."""
    thr = {}
    for c in range(N_CLASSES):
        local_qs = []
        for cd in per_client:
            s = cd.get(c, np.array([]))
            if len(s) >= 1:
                q = float(np.quantile(np.clip(s, 0, 1), 1 - alpha, method='higher'))
                if ldp:
                    q = min(max(q + laplace(1.0 / eps, 1, rng)[0], 0.0), 1.0)
                local_qs.append(q)
        if not local_qs:
            thr[c] = float('inf'); continue
        local_qs = np.sort(local_qs)
        Nq = len(local_qs)
        idx = math.ceil((Nq + 1) * (1 - alpha))
        thr[c] = float(local_qs[idx - 1]) if idx <= Nq else float('inf')
    return thr


def arm_hist_dp(per_client, alpha, eps, rng, n_bins=50):
    """LOCAL-DP histogram: EACH client Laplace-noises its own per-class histogram
    (L1 sens=1 -> Laplace(1/eps) per bin), server sums the K noisy histograms.
    Total noise variance grows with client count K (each client injects its own
    noise) -> degrades in the many-client regime. This is the local-DP trust model."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    thr = {}
    for c in range(N_CLASSES):
        pooled = np.zeros(n_bins)
        any_client = False
        for cd in per_client:
            s = cd.get(c, np.array([]))
            if len(s) >= 1:
                any_client = True
            h, _ = np.histogram(np.clip(s, 0, 1), bins=bin_edges)
            h_noisy = h.astype(float) + laplace(1.0 / eps, n_bins, rng)
            pooled += h_noisy
        thr[c] = fs_quantile_from_hist(pooled, bin_edges, alpha) if any_client else float('inf')
    return thr


def arm_hist_dp_secagg(per_client, alpha, eps, rng, n_bins=50):
    """SECURE-AGGREGATION / distributed-DP histogram (the standard fix): clients
    securely SUM their CLEAN per-class histograms; the server adds ONE Laplace(1/eps)
    noise vector to the pooled total (add/remove-one L1 sensitivity = 1). Total noise
    is a SINGLE draw, independent of client count K -- unlike arm_hist_dp whose noise
    grows with K. Same (eps)-DP guarantee on the released quantile, far better utility.
    Trust model: clients trust the secure-sum protocol (not the server) -- the standard
    cross-device FL deployment model (cf. distributed discrete Gaussian, Pillutla'22)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    thr = {}
    for c in range(N_CLASSES):
        pooled = np.zeros(n_bins)
        any_client = False
        for cd in per_client:
            s = cd.get(c, np.array([]))
            if len(s) >= 1:
                any_client = True
            h, _ = np.histogram(np.clip(s, 0, 1), bins=bin_edges)
            pooled += h.astype(float)                       # CLEAN secure sum
        pooled = pooled + laplace(1.0 / eps, n_bins, rng)   # ONE noise on the total
        thr[c] = fs_quantile_from_hist(pooled, bin_edges, alpha) if any_client else float('inf')
    return thr


# ---------------- baselines added for the advisor's baseline demand ----------------
# Selection criterion (advisor, 2026-07-24): baselines must MEASURE class-conditional
# coverage directly, or be the setting-matched DP+federated incumbent.

def arm_mondrian_centralized(all_scores, cal_labels, alpha):
    """MONDRIAN / CLASSWISE CP (Vovk 2012, Prop. 3): the canonical per-class method.
    One (1-alpha) finite-sample quantile per class, computed from that class's
    calibration scores only. Non-private, centralized -> an ADVANTAGED SKYLINE, not a
    matched competitor.

    NOTE this is mathematically identical to arm_centralized: pooling all calibration
    data and taking a per-class quantile IS classwise CP. It is reported under its
    literature name so the comparison table names the canonical baseline explicitly,
    and so the Vovk floor (|I^y| < 1/alpha - 1 -> q = inf) is visible in the same run."""
    thr = {}
    n_vacuous = 0
    for c in range(N_CLASSES):
        s = all_scores[cal_labels == c]
        q = fs_quantile(s, alpha)
        if math.isinf(q):
            n_vacuous += 1
        thr[c] = q
    return thr, n_vacuous


def arm_mondrian_on_secagg_hist(per_client, alpha, eps, rng, n_bins=50):
    """THE KEY ABLATION. Mondrian/classwise CP whose per-class quantile is read off
    OUR secure-aggregated DP histogram instead of raw pooled scores.

    This isolates the contribution of the histogram CHANNEL alone: it is plain
    per-class calibration, with no method-specific machinery beyond the DP release.
    If this already recovers tail coverage, then the contribution of this work is the
    distribution-sharing channel itself, not anything downstream of it -- which is
    exactly what the write-up should then claim.

    Implementation note: this is the same estimator as arm_hist_dp_secagg. It is run
    and recorded separately under the Mondrian name so the ablation is explicit in the
    results table rather than implied."""
    return arm_hist_dp_secagg(per_client, alpha, eps, rng, n_bins=n_bins)


def _cluster_classes_by_quantile_embedding(per_class_scores, alpha, n_clusters, rng,
                                           embed_qs=(0.5, 0.6, 0.7, 0.8, 0.9)):
    """Ding et al. (NeurIPS 2023) clustered-CP embedding + weighted k-means.
    Each class is embedded by its score quantiles at levels embed_qs U {1-alpha};
    classes are clustered so that rare classes borrow calibration strength from
    distributionally-similar classes. Classes with too few samples are left
    unclustered (assigned -1) as the authors do."""
    levels = list(embed_qs) + [1 - alpha]
    embeds = {}
    for c, s in per_class_scores.items():
        if len(s) >= 2:
            embeds[c] = np.array([np.quantile(s, q, method='higher') for q in levels])
    if len(embeds) < n_clusters or n_clusters < 1:
        return {c: -1 for c in per_class_scores}, 0
    keys = sorted(embeds)
    X = np.stack([embeds[c] for c in keys])
    w = np.array([math.sqrt(len(per_class_scores[c])) for c in keys])  # authors' weight
    # weighted k-means++ (deterministic given rng)
    n_clusters = min(n_clusters, len(keys))
    centers = [X[rng.randint(len(X))]]
    for _ in range(1, n_clusters):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        p = (d2 * w) / max((d2 * w).sum(), 1e-12)
        centers.append(X[rng.choice(len(X), p=p)])
    centers = np.array(centers)
    assign = np.zeros(len(X), dtype=int)
    for _ in range(50):
        d2 = ((X[:, None, :] - centers[None]) ** 2).sum(-1)
        new = d2.argmin(axis=1)
        if (new == assign).all():
            break
        assign = new
        for k in range(n_clusters):
            m = assign == k
            if m.any():
                centers[k] = (X[m] * w[m, None]).sum(0) / w[m].sum()
    out = {c: -1 for c in per_class_scores}
    for c, k in zip(keys, assign):
        out[c] = int(k)
    return out, n_clusters


def arm_clustered_cp(all_scores, cal_labels, alpha, rng, n_clusters=None, gamma=0.5):
    """CLUSTERED CP (Ding et al., NeurIPS 2023, arXiv:2306.09335) -- reimplemented
    from the paper's description. Classes with similar score distributions are
    clustered; the (1-alpha) quantile is computed PER CLUSTER, so a rare class
    inherits a threshold estimated from its cluster's pooled samples instead of its
    own handful of points.

    Splits calibration data: fraction gamma for building the clustering embedding,
    the remainder for the per-cluster quantiles (the authors' two-stage split, which
    keeps the quantile stage independent of the clustering stage).

    Non-private and centralized -> ADVANTAGED SKYLINE. Reported to show what the
    strongest few-samples-per-class method achieves with no privacy cost at all.

    ANTI-STRAWMAN NOTE: this is our implementation from the paper, not the authors'
    code. It must not be presented as a head-to-head result until it has been
    validated against a published number from the paper (see REPORT / protocol)."""
    idx = np.arange(len(all_scores))
    rng.shuffle(idx)
    n_embed = int(gamma * len(idx))
    embed_idx, q_idx = idx[:n_embed], idx[n_embed:]

    embed_by_class = {c: all_scores[embed_idx][cal_labels[embed_idx] == c]
                      for c in range(N_CLASSES)}
    if n_clusters is None:
        # authors' heuristic: scale clusters with available per-class data
        n_avg = np.mean([len(v) for v in embed_by_class.values()])
        n_clusters = max(2, min(N_CLASSES // 2, int(math.sqrt(max(n_avg, 1)))))
    assign, _ = _cluster_classes_by_quantile_embedding(embed_by_class, alpha, n_clusters, rng)

    q_scores = all_scores[q_idx]; q_labels = cal_labels[q_idx]
    cluster_scores = defaultdict(list)
    for c in range(N_CLASSES):
        k = assign.get(c, -1)
        if k >= 0:
            cluster_scores[k].append(q_scores[q_labels == c])
    cluster_thr = {}
    for k, lst in cluster_scores.items():
        pooled = np.concatenate(lst) if lst else np.array([])
        cluster_thr[k] = fs_quantile(pooled, alpha)
    # fallback for unclustered classes: their own classwise quantile, else marginal
    marg = fs_quantile(q_scores, alpha)
    thr = {}
    for c in range(N_CLASSES):
        k = assign.get(c, -1)
        if k >= 0 and k in cluster_thr and not math.isinf(cluster_thr[k]):
            thr[c] = cluster_thr[k]
        else:
            own = fs_quantile(q_scores[q_labels == c], alpha)
            thr[c] = own if not math.isinf(own) else marg
    return thr, assign, n_clusters


def arm_plassier_dpfedcp(per_client, alpha, eps, rng, target=None, n_bar=None,
                         sigma_g_override=None):
    """PLASSIER DP-FedCP (ICML 2023, arXiv:2306.05131), the only prior work matching
    our full setting (federated AND differentially private conformal prediction).

    Reimplemented from the paper (no public code exists) and validated against the
    paper's own Figure 1(b) synthetic experiment: 3 of its 4 qualitative claims
    reproduce. See baselines/plassier/README.md for the validation record and the ten
    specification decisions the paper leaves open.

    IMPORTANT: their guarantee is MARGINAL, for a designated target agent, under label
    shift. It is not a per-class method, and the paper never reports per-class
    coverage. Running it here measures a quantity its authors did not claim; that is
    the point of the comparison, and it must be stated that way rather than as a
    failure of their method.

    Privacy: sigma_g is the Gaussian noise on the quantile-estimation gradients. The
    paper reports sigma_g directly and never converts to epsilon, so a matched-epsilon
    comparison is not directly available from the paper. We map our epsilon budget to
    sigma_g via their Theorem 4.4 bound (see sigma_g_from_eps below) and report both.
    """
    import sys, os
    _bp = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines', 'plassier')
    if _bp not in sys.path:
        sys.path.insert(0, _bp)
    import dp_fedcp as P

    n = len(per_client)
    # Rebuild per-agent score/label arrays in the layout the reimplementation expects.
    # Each agent's points are shuffled: the multinomial subsampling of eq. (5) keeps
    # an agent's FIRST N^i ^ Nbar^i points, which assumes exchangeable order. Built
    # class by class without the shuffle, the kept half systematically dropped the
    # rare (high-score) classes and the target-agent coverage fell to about 84.5
    # percent at sigma_g = 0 (t4_dpfedcp_check.py, 2026-10-13).
    perm_rng = np.random.RandomState(int(rng.randint(0, 2**31 - 1)))
    agent_scores, agent_labels = [], []
    for cd in per_client:
        s_list, l_list = [], []
        for c in range(N_CLASSES):
            v = cd.get(c, np.array([]))
            if len(v):
                s_list.append(np.asarray(v)); l_list.append(np.full(len(v), c, dtype=int))
        s = np.concatenate(s_list) if s_list else np.array([])
        l = np.concatenate(l_list) if l_list else np.array([], dtype=int)
        perm = perm_rng.permutation(len(s))
        agent_scores.append(s[perm]); agent_labels.append(l[perm])

    # Their label-shift weights need per-agent training counts M^i_y. We have none
    # separate from calibration here, so we use each agent's calibration label counts
    # (spec decision D5 in the reimplementation, same choice as in validation).
    counts = np.stack([np.bincount(l, minlength=N_CLASSES).astype(float)
                       if len(l) else np.zeros(N_CLASSES) for l in agent_labels])

    # Target agent: the paper designates one. Absent a natural choice we use the agent
    # holding the most tail-class calibration data, i.e. the one whose distribution is
    # most tail-heavy, which is the most favourable choice for their method.
    if target is None:
        tail_mass = [counts[i][TAIL_CLASSES].sum() for i in range(n)]
        target = int(np.argmax(tail_mass)) if max(tail_mass) > 0 else 0

    sigma_g = sigma_g_from_eps(eps) if sigma_g_override is None else float(sigma_g_override)
    gen = np.random.default_rng(int(rng.randint(0, 2**31 - 1)))
    thr_arr = P.dp_fedcp_thresholds(
        agent_scores, agent_labels, counts, target=target, n_classes=N_CLASSES,
        alpha=alpha, sigma_g=sigma_g, sigma_bar=0.0, n_bar=n_bar, rng=gen)
    return {c: float(thr_arr[c]) for c in range(N_CLASSES)}, target, sigma_g


def sigma_g_from_eps(eps, K=20, T=200, S=None, n=None, delta=1e-5, lam_max=1.0):
    """Map an epsilon budget to the paper's sigma_g via Theorem 4.4:

        sigma_g >= 2 * sqrt( (K * max_i lambda^i_y / eps)
                             * (1 + 24 * S * sqrt(T * log(1/deltabar)) / (eps * n)) )

    With full participation (S = n, as in all their reported experiments) the ratio
    S/n = 1. We use lam_max = 1 as the worst case (lambda sums to 1 over agents, so a
    single agent can hold all the mass). This is the paper's own accountant; note the
    main text and Appendix D typeset the radical differently (recorded in the
    reimplementation README), and we use the main-text form.
    """
    deltabar = max(delta, 1e-12)
    inner = 1.0 + 24.0 * math.sqrt(T * math.log(1.0 / deltabar)) / max(eps, 1e-9)
    return 2.0 * math.sqrt((K * lam_max / max(eps, 1e-9)) * inner)


def arm_hist_dp_secagg_marginal(per_client, alpha, eps, rng, n_bins=50):
    """ABLATION A3: the secure-aggregated DP histogram channel, but calibrated
    MARGINALLY instead of per class.

    Clients securely sum their histograms exactly as in the full method, and the
    server adds one Laplace draw exactly as in the full method. The only change is
    the read-out: instead of one quantile per class, all classes are POOLED into a
    single histogram and a single threshold is used for every class.

    This separates two things the full method does at once:
      (i)  sharing distributions instead of per-client summaries (the channel), and
      (ii) computing a separate quantile per class (per-class calibration).
    A1 (naive_dp) and A2 (hist_dp) ablate the channel; this ablates (ii) while
    keeping (i) intact. If the channel alone were sufficient, this arm would match
    the full method; if per-class calibration is doing real work, it will not.

    Privacy is unchanged: pooling across classes after the noisy release is
    post-processing, and the release itself is identical."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    pooled_all = np.zeros(n_bins)
    for c in range(N_CLASSES):
        for cd in per_client:
            s = cd.get(c, np.array([]))
            h, _ = np.histogram(np.clip(s, 0, 1), bins=bin_edges)
            pooled_all += h.astype(float)          # clean secure sum, all classes
    pooled_all = pooled_all + laplace(1.0 / eps, n_bins, rng)   # ONE noise draw
    thr_marginal = fs_quantile_from_hist(pooled_all, bin_edges, alpha)
    return {c: thr_marginal for c in range(N_CLASSES)}


def arm_local(per_client, alpha, test_probs, test_labels, rng_eval_seed):
    """Each client calibrates per-class threshold from its OWN data (marginal
    fallback if <2 samples); evaluate per-client on the full test set; average
    per-class coverage across clients that have >=1 sample of that class."""
    per_client_cov = []; per_client_size = []
    for cd in per_client:
        # marginal fallback = quantile over all this client's scores
        all_local = np.concatenate([v for v in cd.values()]) if cd else np.array([])
        marg = fs_quantile(all_local, alpha) if len(all_local) else float('inf')
        thr = {}
        for c in range(N_CLASSES):
            s = cd.get(c, np.array([]))
            thr[c] = fs_quantile(s, alpha) if len(s) >= 2 else marg
        rng = np.random.RandomState(rng_eval_seed)
        cov, size, _ = eval_coverage(thr, test_probs, test_labels, rng, marginal_thr=marg)
        per_client_cov.append(cov); per_client_size.append(size)
    return np.nanmean(per_client_cov, axis=0), np.nanmean(per_client_size, axis=0)


# ---------------- runner ----------------
def run_seed(seed, args, cifar_train, cifar_test, device):
    tr_targets = cifar_train.targets
    lt_idx, per_class_n = lt_indices(tr_targets, args.if_factor, n_max=args.n_max, seed=seed)
    print(f'[seed={seed}] LT per-class counts: {per_class_n}', flush=True)

    # split LT pool 50/50 into train / calibration, per class
    rng = np.random.RandomState(seed + 1)
    lt_arr = np.asarray(lt_idx); lt_y = tr_targets[lt_arr]
    train_idx = []; cal_idx = []
    for c in range(N_CLASSES):
        ci = lt_arr[lt_y == c].tolist(); rng.shuffle(ci)
        n_tr = max(1, int(0.5 * len(ci)))
        train_idx += ci[:n_tr]; cal_idx += ci[n_tr:]

    # federated train partition
    # --dirichlet controls client heterogeneity: 0.5 = non-IID (default, used in all
    # reported results); a large value such as 100 approximates an IID split.
    train_client_lists = dirichlet_partition(train_idx, tr_targets, args.n_clients,
                                             args.dirichlet, seed)
    client_ds = [Subset(cifar_train, l) for l in train_client_lists if len(l) >= 2]
    print(f'[seed={seed}] train={len(train_idx)} cal={len(cal_idx)} '
          f'clients={len(client_ds)} sizes={[len(d) for d in client_ds]}', flush=True)

    t0 = time.time()
    model = train_fedavg(client_ds, device, args.n_rounds, args.local_epochs, args.lr, seed)
    print(f'[seed={seed}] trained ({time.time()-t0:.1f}s)', flush=True)

    # softmax on calibration pool + test
    # Calibration images are scored WITHOUT the training augmentation, like the
    # test images, so calibration and test scores are exchangeable. Results before
    # 2026-10-07 scored them through the augmented training dataset; pass
    # --augment-cal to reproduce that older behaviour.
    if getattr(args, 'augment_cal', False):
        cal_src = cifar_train
    else:
        cal_src = _CachedCIFAR(cifar_train.images, cifar_train.targets, train=False)
    cal_ds = Subset(cal_src, cal_idx)
    cal_probs, cal_labels = softmax_on(model, cal_ds, device)
    test_probs, test_labels = softmax_on(model, cifar_test, device)
    cal_index_of = {gidx: row for row, gidx in enumerate(cal_idx)}

    # partition calibration across the SAME clients (Dir 0.5) so tail is sparse
    cal_client_lists = dirichlet_partition(cal_idx, tr_targets, args.n_clients,
                                           args.dirichlet, seed + 7)
    score_rng = np.random.RandomState(2024 + seed)
    per_client, all_scores = build_client_scores(cal_probs, cal_labels, cal_client_lists,
                                                 cal_index_of, score_rng)
    # cal-count per class per client (for reporting tail sparsity)
    tail_counts = []
    for cd in per_client:
        tail_counts.append([len(cd.get(c, [])) for c in TAIL_CLASSES])
    print(f'[seed={seed}] per-client tail(7,8,9) cal counts: {tail_counts}', flush=True)

    # ALPHA SWEEP: each miscoverage level is evaluated on the SAME trained model,
    # the SAME splits and the SAME scores, so only calibration is repeated. Training
    # is the expensive part and is done once.
    global ALPHA
    _alpha_values = args.alpha_list if args.alpha_list else [ALPHA]
    all_rows_seed = []

    def _eval_at_alpha():
        rows = []
        marginal_thr = fs_quantile(all_scores, ALPHA)  # global marginal fallback for eval

        def record(arm, eps, cov, size, marginal, extra=None, thr=None, t_sec=None):
            tail_cov = float(np.nanmean(cov[TAIL_CLASSES]))
            tail_size = float(np.nanmean(size[TAIL_CLASSES]))
            head = [c for c in range(N_CLASSES) if c not in TAIL_CLASSES]
            head_cov = float(np.nanmean(cov[head]))
            head_size = float(np.nanmean(size[head]))
            # CovGap (Ding et al., NeurIPS 2023): 100 * mean_y |cov_y - (1-alpha)|.
            # The field-standard class-conditional metric. Lower is better.
            covgap = float(100.0 * np.nanmean(np.abs(cov - (1 - ALPHA))))
            covgap_tail = float(100.0 * np.nanmean(np.abs(cov[TAIL_CLASSES] - (1 - ALPHA))))
            # count of classes that UNDERCOVER the target (validity failures)
            n_under = int(np.nansum(cov < (1 - ALPHA)))
            # mean set size over ALL classes (efficiency); vacuous == N_CLASSES
            mean_size = float(np.nanmean(size))
            # worst single-class coverage: the metric a rare-class claim lives or dies on
            worst_cov = float(np.nanmin(cov))
            # --- additional diagnostics so later analysis needs no re-runs ------------
            # Undercoverage-only gap: CovGap counts over- and under-coverage equally, but
            # only undercoverage breaks validity. Reported separately so both readings
            # are available.
            under_gap = float(100.0 * np.nanmean(np.maximum((1 - ALPHA) - cov, 0.0)))
            over_gap = float(100.0 * np.nanmean(np.maximum(cov - (1 - ALPHA), 0.0)))
            # How badly the worst class misses, and which class it is.
            worst_class_idx = int(np.nanargmin(cov))
            # Size-stratified coverage (Angelopoulos et al.): the standard conditional
            # proxy the federated CP papers report, included for comparability.
            n_vacuous_sets = int(np.nansum(size >= N_CLASSES - 1e-9))
            row = {'seed': seed, 'arm': arm, 'eps': eps,
                   # --- run configuration (so a CSV is self-describing) ---
                   'dataset': args.dataset, 'if_factor': args.if_factor,
                   'n_clients': args.n_clients, 'n_rounds': args.n_rounds,
                   'dirichlet': args.dirichlet,
                   'alpha': ALPHA, 'n_classes': N_CLASSES, 'n_bins': args.n_bins,
                   'n_cal_total': int(len(cal_labels)), 'n_test_total': int(len(test_labels)),
                   # --- coverage ---
                   'marginal_cov': round(marginal, 4),
                   'tail_cov': round(tail_cov, 4), 'head_cov': round(head_cov, 4),
                   'worst_class_cov': round(worst_cov, 4), 'worst_class_idx': worst_class_idx,
                   # --- gap metrics ---
                   'covgap': round(covgap, 4), 'covgap_tail': round(covgap_tail, 4),
                   'undercov_gap': round(under_gap, 4), 'overcov_gap': round(over_gap, 4),
                   'n_classes_under_target': n_under,
                   # --- efficiency ---
                   'tail_set_size': round(tail_size, 4), 'head_set_size': round(head_size, 4),
                   'mean_set_size': round(mean_size, 4),
                   'n_vacuous_class_sets': n_vacuous_sets,
                   # --- full per-class vectors (the raw material for any later analysis) ---
                   'per_class_cov': ' '.join(f'{cov[c]:.3f}' for c in range(N_CLASSES)),
                   'per_class_size': ' '.join(f'{size[c]:.3f}' for c in range(N_CLASSES)),
                   'per_class_cal_count': ' '.join(str(int((cal_labels == c).sum()))
                                                   for c in range(N_CLASSES)),
                   'per_class_thr': (' '.join(f'{thr[c]:.4f}' if thr and c in thr and
                                              not math.isinf(thr[c]) else 'inf'
                                              for c in range(N_CLASSES)) if thr else ''),
                   'runtime_sec': round(t_sec, 2) if t_sec is not None else ''}
            if extra:
                row.update(extra)
            rows.append(row)
            print(f'  [{arm} eps={eps}] marg={marginal*100:.1f} tail={tail_cov*100:.1f} '
                  f'head={head_cov*100:.1f} worst={worst_cov*100:.1f} '
                  f'CovGap={covgap:.2f} under={n_under}/{N_CLASSES} '
                  f'tail_size={tail_size:.2f} mean_size={mean_size:.2f}', flush=True)

        # CENTRALIZED (unconstrained ceiling)
        thr_c = arm_centralized(all_scores, cal_labels, ALPHA)
        cov, size, marg = eval_coverage(thr_c, test_probs, test_labels,
                                        np.random.RandomState(9000 + seed), marginal_thr)
        record('centralized', 'inf', cov, size, marg, thr=thr_c)

        # MONDRIAN / CLASSWISE CP (Vovk 2012) — canonical per-class baseline, non-private
        # skyline. Also reports how many classes fall below the Vovk/Ding floor
        # (|I^y| < 1/alpha - 1 => vacuous quantile), which is the finite-sample claim
        # the justification document makes.
        thr_m, n_vac = arm_mondrian_centralized(all_scores, cal_labels, ALPHA)
        cov, size, marg = eval_coverage(thr_m, test_probs, test_labels,
                                        np.random.RandomState(9000 + seed), marginal_thr)
        cal_counts = [int((cal_labels == c).sum()) for c in range(N_CLASSES)]
        floor = math.ceil(1.0 / ALPHA) - 1
        n_below_floor = sum(1 for k in cal_counts if k < floor)
        record('mondrian_centralized', 'inf', cov, size, marg,
               extra={'n_vacuous_classes': n_vac, 'n_classes_below_floor': n_below_floor},
               thr=thr_m)
        print(f'  [floor] alpha={ALPHA} needs >={floor} cal pts/class; '
              f'{n_below_floor}/{N_CLASSES} classes below it centrally; '
              f'min per-class cal count={min(cal_counts)}', flush=True)

        # CLUSTERED CP (Ding et al. NeurIPS'23) — non-private skyline, the strongest
        # few-samples-per-class method. OUR reimplementation (anti-strawman caveat).
        thr_cl, assign, n_cl = arm_clustered_cp(all_scores, cal_labels, ALPHA,
                                                np.random.RandomState(4321 + seed))
        cov, size, marg = eval_coverage(thr_cl, test_probs, test_labels,
                                        np.random.RandomState(9000 + seed), marginal_thr)
        record('clustered_cp', 'inf', cov, size, marg, extra={'n_clusters': n_cl}, thr=thr_cl)

        # LOCAL (Screen 2)
        cov, size = arm_local(per_client, ALPHA, test_probs, test_labels, 9000 + seed)
        marg = float(np.nanmean(cov))  # approx marginal for local (per-class averaged)
        record('local', 'inf', cov, size, marg)

        # Plassier at sigma_g = 0: their method with NO privacy noise. Isolates whether
        # any per-class shortfall is structural (their estimator targets a marginal
        # quantile) or merely a DP-noise artifact. Same logic as the qq_nodp reference.
        if not args.skip_plassier:
            try:
                thr_p0, tgt0, _ = arm_plassier_dpfedcp(per_client, ALPHA, 1.0,
                                                       np.random.RandomState(1234 + seed),
                                                       sigma_g_override=0.0)
                cov, size, marg = eval_coverage(thr_p0, test_probs, test_labels,
                                                np.random.RandomState(9000 + seed), marginal_thr)
                record('plassier_nodp', 'inf', cov, size, marg,
                       extra={'plassier_target_agent': tgt0, 'plassier_sigma_g': 0.0},
                       thr=thr_p0)
            except Exception as e:
                print(f'  [plassier_nodp] FAILED: {type(e).__name__}: {e}', flush=True)

        # QQ no-DP reference (Humbert FedCP-QQ without privacy noise) — isolates how much
        # of QQ's tail failure is DP noise vs structural per-client sparsity
        thr_qq0 = arm_qq_ldp(per_client, ALPHA, 1.0, np.random.RandomState(0), ldp=False)
        cov, size, marg = eval_coverage(thr_qq0, test_probs, test_labels,
                                        np.random.RandomState(9000 + seed), marginal_thr)
        record('qq_nodp', 'inf', cov, size, marg, thr=thr_qq0)

        # NAIVE-DP and HIST-DP across eps
        for eps in args.eps_list:
            mrng = np.random.RandomState(555 + seed + int(eps * 10))
            thr_n = arm_naive_dp(per_client, ALPHA, eps, mrng)
            cov, size, marg = eval_coverage(thr_n, test_probs, test_labels,
                                            np.random.RandomState(9000 + seed), marginal_thr)
            record('naive_dp', eps, cov, size, marg, thr=thr_n)

            mrng = np.random.RandomState(333 + seed + int(eps * 10))
            thr_nw = arm_naive_dp_weighted(per_client, ALPHA, eps, mrng)
            cov, size, marg = eval_coverage(thr_nw, test_probs, test_labels,
                                            np.random.RandomState(9000 + seed), marginal_thr)
            record('naive_dp_weighted', eps, cov, size, marg, thr=thr_nw)

            # Humbert FedCP-QQ LDP variant — the real principled per-client-summary baseline
            mrng = np.random.RandomState(444 + seed + int(eps * 10))
            thr_qq = arm_qq_ldp(per_client, ALPHA, eps, mrng, ldp=True)
            cov, size, marg = eval_coverage(thr_qq, test_probs, test_labels,
                                            np.random.RandomState(9000 + seed), marginal_thr)
            record('qq_ldp', eps, cov, size, marg, thr=thr_qq)

            # FAITHFUL FedCP-QQ using authors' exact calc_matrix_M order-statistic index
            mrng = np.random.RandomState(666 + seed + int(eps * 10))
            thr_qqf = arm_fedcpqq(per_client, ALPHA, eps, mrng, ldp=True)
            cov, size, marg = eval_coverage(thr_qqf, test_probs, test_labels,
                                            np.random.RandomState(9000 + seed), marginal_thr)
            record('fedcpqq_exact', eps, cov, size, marg, thr=thr_qqf)

            mrng = np.random.RandomState(777 + seed + int(eps * 10))
            thr_h = arm_hist_dp(per_client, ALPHA, eps, mrng, n_bins=args.n_bins)
            cov, size, marg = eval_coverage(thr_h, test_probs, test_labels,
                                            np.random.RandomState(9000 + seed), marginal_thr)
            record('hist_dp', eps, cov, size, marg, thr=thr_h)

            mrng = np.random.RandomState(888 + seed + int(eps * 10))
            thr_hs = arm_hist_dp_secagg(per_client, ALPHA, eps, mrng, n_bins=args.n_bins)
            cov, size, marg = eval_coverage(thr_hs, test_probs, test_labels,
                                            np.random.RandomState(9000 + seed), marginal_thr)
            record('hist_dp_secagg', eps, cov, size, marg, thr=thr_hs)

            # KEY ABLATION: Mondrian/classwise CP calibrated on OUR secure-agg DP
            # histogram. Same estimator as hist_dp_secagg, recorded under the Mondrian
            # name so the table states explicitly that the tail recovery comes from the
            # distribution-sharing channel and not from downstream machinery.
            mrng = np.random.RandomState(888 + seed + int(eps * 10))   # SAME stream as above
            thr_ms = arm_mondrian_on_secagg_hist(per_client, ALPHA, eps, mrng, n_bins=args.n_bins)
            cov, size, marg = eval_coverage(thr_ms, test_probs, test_labels,
                                            np.random.RandomState(9000 + seed), marginal_thr)
            record('mondrian_on_secagg_hist', eps, cov, size, marg, thr=thr_ms)

            # ABLATION A3: same channel, MARGINAL read-out instead of per-class.
            # Isolates per-class calibration from distribution sharing.
            mrng = np.random.RandomState(888 + seed + int(eps * 10))   # SAME noise stream
            thr_sm = arm_hist_dp_secagg_marginal(per_client, ALPHA, eps, mrng, n_bins=args.n_bins)
            cov, size, marg = eval_coverage(thr_sm, test_probs, test_labels,
                                            np.random.RandomState(9000 + seed), marginal_thr)
            record('secagg_marginal_readout', eps, cov, size, marg, thr=thr_sm)

            # SENSITIVITY S1: number of histogram bins, the method's only free
            # parameter. Swept inside the same seed so every value sees the identical
            # trained model, splits and scores; the only thing that changes is B.
            for B in args.bin_sweep:
                mrng = np.random.RandomState(888 + seed + int(eps * 10))
                thr_b = arm_hist_dp_secagg(per_client, ALPHA, eps, mrng, n_bins=B)
                cov, size, marg = eval_coverage(thr_b, test_probs, test_labels,
                                                np.random.RandomState(9000 + seed), marginal_thr)
                record(f'secagg_bins{B}', eps, cov, size, marg, thr=thr_b,
                       extra={'sweep_n_bins': B})

            # PLASSIER DP-FedCP: the setting-matched incumbent (federated + DP).
            # Marginal guarantee by design; run here to measure per-class coverage,
            # which the original paper never reports.
            #
            # TWO NOISE SETTINGS ARE REPORTED, because a single one would be unfair:
            #  (a) plassier_dpfedcp        : sigma_g from THEIR Theorem 4.4 at our eps.
            #      This is the like-for-like privacy comparison, but the resulting
            #      sigma_g (304 / 76 / 38 at eps = 1 / 4 / 8) is far above the range the
            #      paper itself ever tested (their sweep is {10, 30, 60, 100}), and at
            #      those levels their quantile estimate leaves the [0,1] score range
            #      entirely. Reporting only this would misrepresent their method.
            #  (b) plassier_dpfedcp_sg100  : sigma_g = 100, the NOISIEST setting the paper
            #      itself reports. Their method is not held to a budget it was never
            #      demonstrated at, so this is the charitable comparison.
            # Neither is presented as "their method fails"; see the report.
            if not args.skip_plassier:
                for tag, sg_override in (('plassier_dpfedcp', None),
                                         ('plassier_dpfedcp_sg100', 100.0)):
                    mrng = np.random.RandomState(999 + seed + int(eps * 10))
                    try:
                        thr_p, tgt, sg = arm_plassier_dpfedcp(per_client, ALPHA, eps, mrng,
                                                              sigma_g_override=sg_override)
                        cov, size, marg = eval_coverage(thr_p, test_probs, test_labels,
                                                        np.random.RandomState(9000 + seed), marginal_thr)
                        record(tag, eps, cov, size, marg,
                               extra={'plassier_target_agent': tgt, 'plassier_sigma_g': round(sg, 3)},
                               thr=thr_p)
                    except Exception as e:
                        print(f'  [{tag} eps={eps}] FAILED: {type(e).__name__}: {e}', flush=True)

        return rows

    for _a in _alpha_values:
        ALPHA = float(_a)
        if len(_alpha_values) > 1:
            print(f"[seed={seed}] === alpha={ALPHA} ===", flush=True)
        all_rows_seed += _eval_at_alpha()
    rows = all_rows_seed
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100'])
    p.add_argument('--cache-path', default=None)
    p.add_argument('--n-classes', type=int, default=None)
    p.add_argument('--n-max', type=int, default=None, help='max per-class count (class 0)')
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--n-clients', type=int, default=10)
    p.add_argument('--if-factor', type=int, default=100)
    p.add_argument('--n-rounds', type=int, default=80)
    p.add_argument('--local-epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--n-bins', type=int, default=50)
    p.add_argument('--eps-list', default='1,4,8')
    p.add_argument('--dirichlet', type=float, default=0.5,
                   help='Dirichlet concentration for the client split. 0.5 is the '
                        'non-IID default used in all reported results; a large value '
                        '(e.g. 100) approximates an IID split.')
    p.add_argument('--alpha-list', default='',
                   help='comma-separated miscoverage levels, e.g. 0.05,0.1,0.2. Empty '
                        'uses the single default alpha=0.10. Each value is evaluated on '
                        'the SAME trained model, so only calibration is repeated.')
    p.add_argument('--bin-sweep', default='',
                   help='comma-separated bin counts for sensitivity study S1, e.g. '
                        '10,20,50,100,200. Empty (default) skips the sweep. All values '
                        'run inside the same seed on the same trained model.')
    p.add_argument('--skip-plassier', action='store_true',
                   help='skip the Plassier DP-FedCP arm (it runs one federated '
                        'quantile optimisation per class, so it is the slow arm)')
    p.add_argument('--augment-cal', action='store_true',
                   help='score calibration images through the training augmentation '
                        '(the behaviour of all results before 2026-10-07). Default: '
                        'clean calibration images, exchangeable with the test images.')
    p.add_argument('--out', default='results/dp_fcp.csv')
    args = p.parse_args()
    args.eps_list = [float(x) for x in args.eps_list.split(',')]
    args.bin_sweep = [int(x) for x in args.bin_sweep.split(',') if x.strip()]
    args.alpha_list = [float(x) for x in args.alpha_list.split(',') if x.strip()]

    # dataset-dependent defaults
    global N_CLASSES, TAIL_CLASSES
    if args.dataset == 'cifar100':
        N_CLASSES = args.n_classes or 100
        args.cache_path = args.cache_path or '/home/suraf/fed-cp-gate/data/cifar100/cache.pt'
        args.n_max = args.n_max or 500
    else:
        N_CLASSES = args.n_classes or 10
        args.cache_path = args.cache_path or '/home/suraf/fair-robust-gate/data/cifar10/cache.pt'
        args.n_max = args.n_max or 5000
    # tail = bottom third of classes by LT count; lt_indices orders count descending in class idx
    n_tail = max(1, N_CLASSES // 3)
    TAIL_CLASSES = list(range(N_CLASSES - n_tail, N_CLASSES))
    args.n_max_lt = args.n_max

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[dp-fcp] device={device} dataset={args.dataset} n_classes={N_CLASSES} '
          f'n_max={args.n_max} tail={len(TAIL_CLASSES)} classes seeds={args.seeds} '
          f'eps={args.eps_list} rounds={args.n_rounds} bins={args.n_bins}', flush=True)
    cifar_train, cifar_test = load_cifar_cached(args.cache_path)

    all_rows = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        all_rows += run_seed(seed, args, cifar_train, cifar_test, device)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    # rows carry arm-specific extra columns, so take the union of all keys
    # (ordered: base schema first, then any extras in first-seen order)
    fields = []
    for r in all_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval='')
        w.writeheader(); w.writerows(all_rows)
    print(f'[wrote] {args.out}  ({len(all_rows)} rows, {len(fields)} cols)', flush=True)


if __name__ == '__main__':
    main()
