"""Non-vision scale-up of the DP-FCP gate: 20 Newsgroups (text), TF-IDF + MLP.

Reuses the EXACT conformal + DP-channel arm implementations from dp_fcp_gate.py
(centralized / local / naive_dp / naive_dp_weighted / qq_ldp / hist_dp). Only the
data (20NG text → TF-IDF), model (MLP), and FedAvg training differ. This confirms
the DP-histogram tail-coverage advantage is not vision-specific.

20 classes; long-tailed subsample of the training set (IF); calibration split
across 10 clients via Dirichlet so tail classes are sparse per client.
"""
import argparse
import copy
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import dp_fcp_gate as G   # reuse APS + all arm implementations + dirichlet_partition


# ---------------- MLP model (non-conv, non-vision) ----------------
class MLP(nn.Module):
    def __init__(self, n_feat, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_20ng_tfidf(max_features=5000):
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    tr = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    te = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    vec = TfidfVectorizer(max_features=max_features, stop_words='english', sublinear_tf=True)
    Xtr = vec.fit_transform(tr.data).toarray().astype(np.float32)
    Xte = vec.transform(te.data).toarray().astype(np.float32)
    ytr = np.asarray(tr.target); yte = np.asarray(te.target)
    return Xtr, ytr, Xte, yte, max_features


def lt_subsample(y, n_max, if_factor, n_classes, seed):
    rng = np.random.RandomState(seed)
    mu = if_factor ** (-1.0 / (n_classes - 1))
    per_class = [max(20, int(round(n_max * (mu ** i)))) for i in range(n_classes)]
    kept = []
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        kept.extend(idx[:min(per_class[c], len(idx))].tolist())
    return kept, per_class


def local_train(model, x, y, opt, device, epochs, batch):
    model.train()
    n = x.size(0)
    for _ in range(epochs):
        perm = torch.randperm(n, device=x.device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            F.cross_entropy(model(x[idx]), y[idx]).backward()
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


def softmax_on(model, x, device, bs=1024):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, x.size(0), bs):
            outs.append(F.softmax(model(x[s:s+bs]), dim=-1).cpu().numpy())
    return np.concatenate(outs)


def run_seed(seed, args, data, device):
    Xtr, ytr, Xte, yte, n_feat = data
    C = G.N_CLASSES

    lt_idx, per_class_n = lt_subsample(ytr, args.n_max, args.if_factor, C, seed)
    lt_idx = np.asarray(lt_idx); lt_y = ytr[lt_idx]
    rng = np.random.RandomState(seed + 1)
    train_idx = []; cal_idx = []
    for c in range(C):
        ci = lt_idx[lt_y == c].tolist(); rng.shuffle(ci)
        n_tr = max(1, int(0.5 * len(ci)))
        train_idx += ci[:n_tr]; cal_idx += ci[n_tr:]
    print(f'[seed={seed}] LT counts: {per_class_n}', flush=True)
    print(f'[seed={seed}] train={len(train_idx)} cal={len(cal_idx)}', flush=True)

    # federated train partition (Dirichlet 0.5)
    train_client_lists = G.dirichlet_partition(train_idx, ytr, args.n_clients, 0.5, seed)
    Xtr_t = torch.from_numpy(Xtr).to(device); ytr_t = torch.from_numpy(ytr).long().to(device)

    torch.manual_seed(seed); np.random.seed(seed)
    model = MLP(n_feat, C).to(device)
    t0 = time.time()
    for rnd in range(1, args.n_rounds + 1):
        cur_lr = args.lr * (0.99 ** (rnd - 1))
        states = []; weights = []
        for lst in train_client_lists:
            if len(lst) < 2:
                continue
            m = copy.deepcopy(model)
            opt = torch.optim.SGD(m.parameters(), lr=cur_lr, momentum=0.9, weight_decay=5e-4)
            idx = torch.tensor(lst, device=device)
            local_train(m, Xtr_t[idx], ytr_t[idx], opt, device, args.local_epochs, 64)
            states.append({k: v.detach().cpu() for k, v in m.state_dict().items()})
            weights.append(len(lst))
            del m, opt
        aggregate(model, states, weights)
    print(f'[seed={seed}] trained ({time.time()-t0:.1f}s)', flush=True)

    # softmax on calibration + test
    cal_probs = softmax_on(model, Xtr_t[torch.tensor(cal_idx, device=device)], device)
    cal_labels = ytr[np.asarray(cal_idx)]
    Xte_t = torch.from_numpy(Xte).to(device)
    test_probs = softmax_on(model, Xte_t, device)
    test_labels = yte

    # partition calibration across clients (Dir 0.5) so tail sparse per client
    cal_client_lists = G.dirichlet_partition(cal_idx, ytr, args.n_clients, 0.5, seed + 7)
    cal_index_of = {gidx: row for row, gidx in enumerate(cal_idx)}
    score_rng = np.random.RandomState(2024 + seed)
    per_client, all_scores = G.build_client_scores(cal_probs, cal_labels, cal_client_lists,
                                                   cal_index_of, score_rng)
    marginal_thr = G.fs_quantile(all_scores, G.ALPHA)

    rows = []

    def record(arm, eps, cov, size, marg, extra=None, thr=None):
        """Same 36-column schema as dp_fcp_gate.py, so text and vision results can
        be concatenated and analysed together."""
        tail = float(np.nanmean(cov[G.TAIL_CLASSES]))
        ts = float(np.nanmean(size[G.TAIL_CLASSES]))
        head = [c for c in range(C) if c not in G.TAIL_CLASSES]
        head_cov = float(np.nanmean(cov[head]))
        head_size = float(np.nanmean(size[head]))
        a = G.ALPHA
        covgap = float(100.0 * np.nanmean(np.abs(cov - (1 - a))))
        covgap_tail = float(100.0 * np.nanmean(np.abs(cov[G.TAIL_CLASSES] - (1 - a))))
        under_gap = float(100.0 * np.nanmean(np.maximum((1 - a) - cov, 0.0)))
        over_gap = float(100.0 * np.nanmean(np.maximum(cov - (1 - a), 0.0)))
        row = {'seed': seed, 'arm': arm, 'eps': eps,
               'dataset': '20newsgroups', 'if_factor': args.if_factor,
               'n_clients': args.n_clients, 'n_rounds': args.n_rounds,
               'alpha': a, 'n_classes': C, 'n_bins': args.n_bins,
               'n_cal_total': int(len(cal_labels)), 'n_test_total': int(len(test_labels)),
               'marginal_cov': round(marg, 4),
               'tail_cov': round(tail, 4), 'head_cov': round(head_cov, 4),
               'worst_class_cov': round(float(np.nanmin(cov)), 4),
               'worst_class_idx': int(np.nanargmin(cov)),
               'covgap': round(covgap, 4), 'covgap_tail': round(covgap_tail, 4),
               'undercov_gap': round(under_gap, 4), 'overcov_gap': round(over_gap, 4),
               'n_classes_under_target': int(np.nansum(cov < (1 - a))),
               'tail_set_size': round(ts, 4), 'head_set_size': round(head_size, 4),
               'mean_set_size': round(float(np.nanmean(size)), 4),
               'n_vacuous_class_sets': int(np.nansum(size >= C - 1e-9)),
               'per_class_cov': ' '.join(f'{cov[c]:.3f}' for c in range(C)),
               'per_class_size': ' '.join(f'{size[c]:.3f}' for c in range(C)),
               'per_class_cal_count': ' '.join(str(int((cal_labels == c).sum()))
                                               for c in range(C)),
               'per_class_thr': (' '.join(f'{thr[c]:.4f}' if thr and c in thr and
                                          not math.isinf(thr[c]) else 'inf'
                                          for c in range(C)) if thr else '')}
        if extra:
            row.update(extra)
        rows.append(row)
        print(f'  [{arm} eps={eps}] marg={marg*100:.1f} tail={tail*100:.1f} '
              f'worst={float(np.nanmin(cov))*100:.1f} CovGap={covgap:.2f} '
              f'under={under_gap:.2f} tail_size={ts:.2f}', flush=True)

    ev = lambda t: G.eval_coverage(t, test_probs, test_labels,
                                   np.random.RandomState(9000 + seed), marginal_thr)

    thr = G.arm_centralized(all_scores, cal_labels, G.ALPHA)
    cov, size, marg = ev(thr)
    record('centralized', 'inf', cov, size, marg, thr=thr)

    # Mondrian / classwise CP (Vovk 2012) — canonical per-class baseline.
    thr_m, n_vac = G.arm_mondrian_centralized(all_scores, cal_labels, G.ALPHA)
    cov, size, marg = ev(thr_m)
    floor = math.ceil(1.0 / G.ALPHA) - 1
    n_below = sum(1 for c in range(C) if int((cal_labels == c).sum()) < floor)
    record('mondrian_centralized', 'inf', cov, size, marg, thr=thr_m,
           extra={'n_vacuous_classes': n_vac, 'n_classes_below_floor': n_below})

    # Clustered CP (Ding et al. NeurIPS'23) — validated reimplementation.
    thr_cl, _assign, n_cl = G.arm_clustered_cp(all_scores, cal_labels, G.ALPHA,
                                               np.random.RandomState(4321 + seed))
    cov, size, marg = ev(thr_cl)
    record('clustered_cp', 'inf', cov, size, marg, thr=thr_cl, extra={'n_clusters': n_cl})

    cov, size = G.arm_local(per_client, G.ALPHA, test_probs, test_labels, 9000 + seed)
    record('local', 'inf', cov, size, float(np.nanmean(cov)))

    thr = G.arm_qq_ldp(per_client, G.ALPHA, 1.0, np.random.RandomState(0), ldp=False)
    cov, size, marg = ev(thr)
    record('qq_nodp', 'inf', cov, size, marg, thr=thr)

    # Plassier DP-FedCP at sigma_g = 0: structural check, no privacy noise.
    if not args.skip_plassier:
        try:
            thr_p0, tgt0, _ = G.arm_plassier_dpfedcp(per_client, G.ALPHA, 1.0,
                                                     np.random.RandomState(1234 + seed),
                                                     sigma_g_override=0.0)
            cov, size, marg = ev(thr_p0)
            record('plassier_nodp', 'inf', cov, size, marg, thr=thr_p0,
                   extra={'plassier_target_agent': tgt0, 'plassier_sigma_g': 0.0})
        except Exception as e:
            print(f'  [plassier_nodp] FAILED: {type(e).__name__}: {e}', flush=True)

    for eps in args.eps_list:
        thr = G.arm_naive_dp(per_client, G.ALPHA, eps, np.random.RandomState(555+seed+int(eps*10)))
        cov, size, marg = ev(thr); record('naive_dp', eps, cov, size, marg, thr=thr)

        thr = G.arm_qq_ldp(per_client, G.ALPHA, eps, np.random.RandomState(444+seed+int(eps*10)), ldp=True)
        cov, size, marg = ev(thr); record('qq_ldp', eps, cov, size, marg, thr=thr)

        # FedCP-QQ with the authors' exact order-statistic index.
        thr = G.arm_fedcpqq(per_client, G.ALPHA, eps, np.random.RandomState(666+seed+int(eps*10)), ldp=True)
        cov, size, marg = ev(thr); record('fedcpqq_exact', eps, cov, size, marg, thr=thr)

        thr = G.arm_hist_dp(per_client, G.ALPHA, eps, np.random.RandomState(777+seed+int(eps*10)), n_bins=args.n_bins)
        cov, size, marg = ev(thr); record('hist_dp', eps, cov, size, marg, thr=thr)

        thr = G.arm_hist_dp_secagg(per_client, G.ALPHA, eps, np.random.RandomState(888+seed+int(eps*10)), n_bins=args.n_bins)
        cov, size, marg = ev(thr); record('hist_dp_secagg', eps, cov, size, marg, thr=thr)

        # Plassier at two noise settings (see HEAD_TO_HEAD_REPORT.md §4 for why both).
        if not args.skip_plassier:
            for tag, sg in (('plassier_dpfedcp', None), ('plassier_dpfedcp_sg100', 100.0)):
                try:
                    thr_p, tgt, sgv = G.arm_plassier_dpfedcp(
                        per_client, G.ALPHA, eps, np.random.RandomState(999+seed+int(eps*10)),
                        sigma_g_override=sg)
                    cov, size, marg = ev(thr_p)
                    record(tag, eps, cov, size, marg, thr=thr_p,
                           extra={'plassier_target_agent': tgt, 'plassier_sigma_g': round(sgv, 3)})
                except Exception as e:
                    print(f'  [{tag} eps={eps}] FAILED: {type(e).__name__}: {e}', flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--n-clients', type=int, default=10)
    p.add_argument('--if-factor', type=int, default=10)
    p.add_argument('--n-max', type=int, default=450)
    p.add_argument('--n-rounds', type=int, default=60)
    p.add_argument('--local-epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.5)
    p.add_argument('--n-bins', type=int, default=50)
    p.add_argument('--max-features', type=int, default=5000)
    p.add_argument('--eps-list', default='1,4,8')
    p.add_argument('--skip-plassier', action='store_true',
                   help='skip the Plassier DP-FedCP arms (the slow ones: one '
                        'federated quantile optimisation per class)')
    p.add_argument('--out', default='results/dp_fcp_text.csv')
    args = p.parse_args()
    args.eps_list = [float(x) for x in args.eps_list.split(',')]

    G.N_CLASSES = 20
    n_tail = max(1, 20 // 3)
    G.TAIL_CLASSES = list(range(20 - n_tail, 20))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[dp-fcp-text] device={device} 20NG n_classes=20 tail={len(G.TAIL_CLASSES)} '
          f'if={args.if_factor} n_max={args.n_max} rounds={args.n_rounds} eps={args.eps_list}', flush=True)

    data = load_20ng_tfidf(args.max_features)
    print(f'[data] Xtr={data[0].shape} Xte={data[2].shape} feats={data[4]}', flush=True)

    all_rows = []
    for seed in [int(s) for s in args.seeds.split(',')]:
        all_rows += run_seed(seed, args, data, device)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    # rows carry arm-specific extra columns, so take the union of all keys
    fields = []
    for r in all_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval='')
        w.writeheader(); w.writerows(all_rows)
    print(f'[wrote] {args.out}', flush=True)


if __name__ == '__main__':
    main()
