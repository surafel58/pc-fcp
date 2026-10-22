"""v3 full rerun, part 3 (2026-10-14): turn results/v3/eval/*.csv and results/v3/accuracy.csv
into every table of Chapter 4, as one Markdown report (results/v3/RESULTS_V3.md).

Aggregation: a metric is averaged over the noise draws of a seed, then reported as the
mean and standard deviation over the seeds (Section 4.1.4). Per-class coverage vectors
are averaged over draws and then over seeds.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

MAIN_EPS = {'cifar10': '1.0', 'cifar100': '1.0', 'text': '4.0'}
NAMES = {'cifar10': 'CIFAR-10-LT (IF 100)', 'cifar100': 'CIFAR-100-LT (IF 10)', 'text': '20 Newsgroups (IF 10)'}
COLS = ['marginal_bal', 'marginal_lt', 'tail_cov', 'worst_cov', 'worst_exp', 'covgap', 'undercov', 'set_size', 'n_always']
PCT = {'marginal_bal', 'marginal_lt', 'tail_cov', 'head_cov', 'worst_cov', 'worst_exp'}
HEAD = {'marginal_bal': 'Marg. (bal.)', 'marginal_lt': 'Marg. (LT)', 'tail_cov': 'Tail', 'worst_cov': 'Worst', 'worst_exp': 'Worst (noise-avg.)',
        'covgap': 'CovGap', 'undercov': 'UnderCov', 'set_size': 'Set size', 'n_always': 'Always-incl.'}


TAG_RE = r'^(?P<dataset>[a-z0-9]+)_K(?P<K>\d+)_dir(?P<dir>[\d.]+)(?P<suf>(?:_[a-z0-9]+)*?)_seed(?P<seed_>\d+)$'
# suffixes that mark the main configuration of the 2026-10-20 batch (logit-adjusted images,
# rerun text); they are dropped from cfg so the report keys stay 'cifar10_K10_dir0.5' etc.
MAIN_SUFFIXES = ('_la', '_b')


def cfg_of(dataset, K, dr, suf):
    s = suf or ''
    for m in MAIN_SUFFIXES:
        s = s.replace(m, '')
    return f'{dataset}_K{K}_dir{dr}{s}'


def load(eval_dir):
    fs = sorted(glob.glob(os.path.join(eval_dir, '*.csv')))
    d = pd.concat([pd.read_csv(f, dtype={'eps': str}) for f in fs], ignore_index=True)
    m = d['tag'].str.extract(TAG_RE)
    d = pd.concat([d.drop(columns=['K'], errors='ignore'), m], axis=1)
    bad = d['dataset'].isna()
    if bad.any():
        print(f'[analyze] {bad.sum()} rows with unparsed tags dropped: {sorted(d.loc[bad, "tag"].unique())[:5]}')
        d = d[~bad]
    d['K'] = d['K'].astype(int)
    d['cfg'] = [cfg_of(a, b, c, s) for a, b, c, s in zip(d['dataset'], d['K'], d['dir'], d['suf'].fillna(''))]
    for c in PCT & set(d.columns):
        d[c] = 100 * d[c]
    return d


def per_seed(d):
    keys = ['cfg', 'dataset', 'K', 'dir', 'seed', 'method', 'eps', 'alpha']
    num = [c for c in COLS + ['head_cov', 'tail_size', 'n_under', 'overcov'] if c in d.columns and c != 'worst_exp']
    out = d.groupby(keys, dropna=False)[num].mean()
    # worst class of the class coverage averaged over the noise draws of the seed
    we = d.groupby(keys, dropna=False)['per_class_cov'].apply(
        lambda x: 100 * np.mean([np.array(t.split(), float) for t in x], axis=0).min())
    return out.join(we.rename('worst_exp')).reset_index()


def summary(ps):
    keys = ['cfg', 'dataset', 'K', 'dir', 'method', 'eps', 'alpha']
    num = [c for c in ps.columns if c not in keys + ['seed']]
    g = ps.groupby(keys, dropna=False)[num]
    mean, std, n = g.mean(), g.std(ddof=1), g.size()
    out = mean.add_suffix('_m').join(std.add_suffix('_s')).join(n.rename('n_seeds'))
    return out.reset_index()


def fmt(row, c, nd=None):
    if nd is None:
        nd = 1 if c in PCT else (2 if c in ('covgap', 'undercov', 'set_size') else 1)
    m, s = row[c + '_m'], row[c + '_s']
    if pd.isna(m):
        return '—'
    return f'{m:.{nd}f} ± {s:.{nd}f}' if not pd.isna(s) else f'{m:.{nd}f}'


def table(S, cfg, rows, cols=COLS, alpha=0.1):
    """rows: list of (label, method, eps)."""
    out = ['| Method | ' + ' | '.join(HEAD.get(c, c) for c in cols) + ' |',
           '|---|' + '---|' * len(cols)]
    for label, meth, eps in rows:
        r = S[(S.cfg == cfg) & (S.method == meth) & (S.eps == eps) & (np.isclose(S.alpha, alpha))]
        if r.empty:
            out.append(f'| {label} | ' + ' | '.join('n/a' for _ in cols) + ' |'); continue
        r = r.iloc[0]
        out.append(f'| {label} | ' + ' | '.join(fmt(r, c) for c in cols) + ' |')
    return '\n'.join(out)


def perclass(d, cfg, methods, alpha=0.1):
    """mean per-class coverage over draws then seeds, plus n_y from the cache."""
    out = {}
    for label, meth, eps in methods:
        x = d[(d.cfg == cfg) & (d.method == meth) & (d.eps == eps) & np.isclose(d.alpha, alpha)]
        if x.empty:
            continue
        v = x.groupby('seed')['per_class_cov'].apply(
            lambda s: np.mean([np.array(t.split(), float) for t in s], axis=0))
        out[label] = 100 * np.mean(np.stack(v.values), axis=0)
    return pd.DataFrame(out)


def main_rows(eps):
    return [('Centralized Mondrian †', 'mondrian', 'inf'),
            ('Clustered CP (Ding rule) †', 'clustered_ding', 'inf'),
            ('PC-FCP noise-free (reference)', 'pcfcp_noise_free', 'inf'),
            (f'DP-FedCP, marginal (mapped eps={eps})', 'dpfedcp_marg', eps),
            (f'DP-FedCP, per class (mapped eps={eps})', 'dpfedcp_pc', eps),
            (f'FedCP2-QQ, marginal (eps={eps})', 'fedcp2qq_marg', eps),
            (f'FedCP2-QQ, per class (eps={eps})', 'fedcp2qq_pc', eps),
            (f'FedCP2-QQ, per class + fallback (eps={eps})', 'fedcp2qq_pc_fallback', eps),
            (f'FedCP2-QQ, per class + private fallback (eps/2 + eps/2) (eps={eps})', 'fedcp2qq_pc_pfallback', eps),
            (f'**PC-FCP (eps={eps})**', 'pcfcp', eps),
            (f'PC-FCP, margin gamma=0.05 (eps={eps})', 'pcfcp_margin', eps),
            (f'PC-FCP, borrow (eps={eps})', 'pcfcp_borrow', eps),
            (f'A8 tree-structured release (eps={eps})', 'abl_A8_tree', eps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval', default='results/v3/eval')
    ap.add_argument('--acc', default='results/v3/accuracy.csv')
    ap.add_argument('--cache', default='results/v3/cache')
    ap.add_argument('--out', default='results/v3/RESULTS_V3.md')
    a = ap.parse_args()
    d = load(a.eval)
    ps = per_seed(d)
    S = summary(ps)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    S.to_csv(os.path.splitext(a.out)[0] + '_summary.csv', index=False)
    L = ['# v3 results (mean ± std over seeds; noise averaged over draws within a seed)', '']

    # ---------------- accuracy (T1)
    if os.path.exists(a.acc):
        acc = pd.read_csv(a.acc).drop_duplicates('tag', keep='last')
        acc['cfg'] = acc['tag'].str.replace(r'_seed\d+$', '', regex=True)
        for s in MAIN_SUFFIXES:
            acc['cfg'] = acc['cfg'].str.replace(s, '', regex=False)
        L += ['## Classifier accuracy (test set, percent)', '',
              '| Config | seeds | Top-1 | Head | Tail | Worst class | Top-5 |', '|---|---|---|---|---|---|---|']
        for cfg, x in acc.groupby('cfg'):
            f = lambda c: f'{100*x[c].mean():.1f} ± {100*x[c].std(ddof=1):.1f}' if len(x) > 1 else f'{100*x[c].mean():.1f}'
            L.append(f'| {cfg} | {len(x)} | {f("top1")} | {f("top1_head")} | {f("top1_tail")} | {f("top1_worst")} | {f("top5")} |')
        L.append('')

    # ---------------- main comparisons
    for ds in ('cifar10', 'cifar100', 'text'):
        cfg = f'{ds}_K10_dir0.5'
        if cfg not in set(S.cfg):
            continue
        eps = MAIN_EPS[ds]
        L += [f'## Baseline comparison: {NAMES[ds]}, K=10, Dir(0.5), target 90%', '',
              table(S, cfg, main_rows(eps)), '']
        L += ['Non-private / noise settings of the baselines:', '',
              table(S, cfg, [('DP-FedCP marginal, sigma_g=100', 'dpfedcp_marg_sg100', 'sg100'),
                             ('DP-FedCP marginal, no noise', 'dpfedcp_marg_nonoise', 'inf'),
                             ('DP-FedCP per class, sigma_g=100', 'dpfedcp_pc_sg100', 'sg100'),
                             ('DP-FedCP per class, no noise', 'dpfedcp_pc_nonoise', 'inf'),
                             ('FedCP-QQ marginal, non-private', 'fedcpqq_marg', 'inf'),
                             ('FedCP-QQ per class, non-private', 'fedcpqq_pc', 'inf'),
                             ('FedCP-QQ per class, non-private + fallback', 'fedcpqq_pc_fallback', 'inf')]
                    + [(f'{m} eps={e}', m, e) for m in ('pcfcp', 'fedcp2qq_pc', 'dpfedcp_pc', 'dpfedcp_marg', 'fedcp2qq_marg')
                       for e in ('1.0', '4.0', '8.0') if e != eps]), '']
        pc = perclass(d, cfg, [('Mondrian', 'mondrian', 'inf'), ('Clustered', 'clustered_ding', 'inf'),
                               ('DP-FedCP marg', 'dpfedcp_marg', eps), ('DP-FedCP pc', 'dpfedcp_pc', eps),
                               ('DP-FedCP pc noiseless', 'dpfedcp_pc_nonoise', 'inf'),
                               ('FedCP-QQ pc nonpriv', 'fedcpqq_pc', 'inf'),
                               ('FedCP2-QQ marg', 'fedcp2qq_marg', eps), ('FedCP2-QQ pc', 'fedcp2qq_pc', eps),
                               ('PC-FCP', 'pcfcp', eps)])
        if not pc.empty:
            npz = sorted(glob.glob(os.path.join(a.cache, cfg + '_seed*.npz')))
            if npz:
                pc.insert(0, 'n_y', np.bincount(np.load(npz[0])['cal_labels'], minlength=len(pc)))
            if ds == 'cifar100':
                pc = pc.groupby(np.arange(len(pc)) // 10).mean()
                pc.index.name = 'decile'
            L += [f'Per-class coverage ({"deciles of 10 classes" if ds == "cifar100" else "classes"}, most frequent first):', '',
                  pc.round(1).to_markdown(), '']
        cl = d[(d.cfg == cfg) & (d.method == 'clustered_ding')]
        if 'clusters_used' in cl:
            L += [f'Clustered CP clusters M per seed: {cl.clusters.tolist()}, used {cl.clusters_used.tolist()}, gamma {cl.gamma.tolist()}', '']

    # ---------------- ablations
    abl = [('A1 threshold averaging', 'abl_A1_avg'), ('A1 weighted (eps/2 + eps/2)', 'abl_A1_weighted'),
           ('A2 per-client noise', 'abl_A2_local_noise'), ('A3 marginal read-out', 'abl_A3_marginal'),
           ('No floor', 'abl_nofloor'),
           ('Read-out: bottom-up', 'abl_bottomup'), ('Read-out: consistency', 'abl_consistency'),
           ('Score: LAC', 'abl_lac'),
           ('Bins p3 B5', 'pcfcp_p3_B5'), ('Bins p3 B12', 'pcfcp_p3_B12'), ('Bins p3 B20', 'pcfcp_p3_B20'),
           ('Bins p3 B50', 'pcfcp_p3_B50'), ('Bins p3 B100', 'pcfcp_p3_B100'), ('Bins p2 B8', 'pcfcp_p2_B8'),
           ('Bins p2 B100', 'pcfcp_p2_B100'), ('Bins uniform 50', 'pcfcp_uniform50'),
           ('A8 tree-structured release', 'abl_A8_tree'), ('Margin gamma=0.05', 'pcfcp_margin'),
           ('Borrow (k_c < 1 -> pooled)', 'pcfcp_borrow'),
           ('**PC-FCP (p3 B8, top-down, floor, APS)**', 'pcfcp')]
    for ds in ('cifar10', 'cifar100', 'text'):
        cfg = f'{ds}_K10_dir0.5'
        if cfg not in set(S.cfg):
            continue
        for eps in ('1.0', '2.0', '3.0', '4.0', '8.0'):
            L += [f'## Ablations: {NAMES[ds]}, eps={eps}', '', table(S, cfg, [(l, m, eps) for l, m in abl]), '']
        L += [f'## Noise-free references: {NAMES[ds]}', '',
              table(S, cfg, [('Mondrian APS', 'mondrian', 'inf'), ('Mondrian LAC', 'mondrian_lac', 'inf'),
                             ('PC-FCP noise-free power100', 'pcfcp_noise_free', 'inf'),
                             ('PC-FCP noise-free LAC', 'pcfcp_lac_noise_free', 'inf'),
                             ('PC-FCP noise-free, no floor', 'pcfcp_noise_free_nofloor', 'inf')]
                    + [(f'noise-free {g}', f'pcfcp_{g}_noise_free', 'inf') for g in
                       ('p3_B5', 'p3_B12', 'p3_B20', 'p3_B50', 'p3_B100', 'p2_B8', 'p2_B100', 'uniform50')]), '']

    # ---------------- privacy-budget sweep
    for ds in ('cifar10', 'cifar100', 'text'):
        cfg = f'{ds}_K10_dir0.5'
        if cfg not in set(S.cfg):
            continue
        L += [f'## Privacy budget sweep: {NAMES[ds]} (worst / worst noise-avg. / set size / always-incl.)', '',
              '| Method | ' + ' | '.join(f'eps={e}' for e in ('1', '2', '3', '4', '8')) + ' |', '|---|' + '---|' * 5]
        for lab, m in [('PC-FCP', 'pcfcp'), ('FedCP2-QQ marginal', 'fedcp2qq_marg'), ('FedCP2-QQ per class', 'fedcp2qq_pc'),
                       ('FedCP2-QQ per class + fallback', 'fedcp2qq_pc_fallback'), ('DP-FedCP marginal', 'dpfedcp_marg'),
                       ('FedCP2-QQ per class + private fallback', 'fedcp2qq_pc_pfallback'),
                       ('DP-FedCP per class', 'dpfedcp_pc'), ('A2 per-client noise', 'abl_A2_local_noise'), ('No floor', 'abl_nofloor'),
                       ('PC-FCP margin gamma=0.05', 'pcfcp_margin'), ('A8 tree', 'abl_A8_tree'), ('PC-FCP borrow', 'pcfcp_borrow')]:
            cells = []
            for e in ('1.0', '2.0', '3.0', '4.0', '8.0'):
                r = S[(S.cfg == cfg) & (S.method == m) & (S.eps == e) & np.isclose(S.alpha, 0.1)]
                cells.append('n/a' if r.empty else f"{r.iloc[0].worst_cov_m:.1f} / {r.iloc[0].worst_exp_m:.1f} / {r.iloc[0].set_size_m:.2f} / {r.iloc[0].n_always_m:.1f}")
            L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
        L.append('')

    # ---------------- client scale (CIFAR-100)
    L += ['## Client scale: CIFAR-100-LT, Dir(0.5)', '',
          '| K | eps | PC-FCP tail | PC-FCP worst | PC-FCP UnderCov | A2 tail | A2 worst | A2 UnderCov |', '|---|---|---|---|---|---|---|---|']
    for K in (10, 50, 100):
        cfg = f'cifar100_K{K}_dir0.5'
        for eps in ('1.0', '2.0', '3.0', '4.0', '8.0'):
            r1 = S[(S.cfg == cfg) & (S.method == 'pcfcp') & (S.eps == eps) & np.isclose(S.alpha, 0.1)]
            r2 = S[(S.cfg == cfg) & (S.method == 'abl_A2_local_noise') & (S.eps == eps) & np.isclose(S.alpha, 0.1)]
            if r1.empty:
                continue
            r1 = r1.iloc[0]; r2 = r2.iloc[0] if not r2.empty else None
            L.append(f'| {K} | {eps} | {fmt(r1, "tail_cov")} | {fmt(r1, "worst_cov")} | {fmt(r1, "undercov")} | '
                     + (f'{fmt(r2, "tail_cov")} | {fmt(r2, "worst_cov")} | {fmt(r2, "undercov")} |' if r2 is not None else 'n/a | n/a | n/a |'))
    L.append('')
    for K in (50, 100):
        cfg = f'cifar100_K{K}_dir0.5'
        if cfg in set(S.cfg):
            L += [f'### CIFAR-100-LT K={K} comparison (eps=1)', '', table(S, cfg, main_rows('1.0')), '']

    # ---------------- target level
    for ds in ('cifar10', 'cifar100', 'text'):
        cfg = f'{ds}_K10_dir0.5'
        if cfg not in set(S.cfg):
            continue
        L += [f'## Target level: {NAMES[ds]}', '',
              '| Target | Method | Marg. (bal.) | Marg. (LT) | Tail | Worst | UnderCov | Set size | Always-incl. |',
              '|---|---|---|---|---|---|---|---|---|']
        for al in (0.05, 0.1, 0.2, 0.3, 0.35):
            for lab, m, e in [('Mondrian †', 'mondrian', 'inf'), ('PC-FCP noise-free', 'pcfcp_noise_free', 'inf'),
                              ('FedCP-QQ pc non-private', 'fedcpqq_pc', 'inf'),
                              ('FedCP2-QQ marg eps=1', 'fedcp2qq_marg', '1.0'), ('FedCP2-QQ marg eps=4', 'fedcp2qq_marg', '4.0'),
                              ('PC-FCP eps=1', 'pcfcp', '1.0'), ('PC-FCP eps=4', 'pcfcp', '4.0')]:
                r = S[(S.cfg == cfg) & (S.method == m) & (S.eps == e) & np.isclose(S.alpha, al)]
                if r.empty:
                    continue
                r = r.iloc[0]
                L.append(f'| {100*(1-al):.0f}% | {lab} | ' + ' | '.join(fmt(r, c) for c in
                         ['marginal_bal', 'marginal_lt', 'tail_cov', 'worst_cov', 'undercov', 'set_size', 'n_always']) + ' |')
        L.append('')

    # ---------------- heterogeneity
    for cfg in ('cifar10_K10_dir100',):
        if cfg in set(S.cfg):
            L += ['## Near-IID partition: CIFAR-10-LT, Dir(100)', '', table(S, cfg, main_rows('1.0')), '',
                  table(S, cfg, [('DP-FedCP marginal, no noise', 'dpfedcp_marg_nonoise', 'inf'),
                                 ('DP-FedCP per class, no noise', 'dpfedcp_pc_nonoise', 'inf'),
                                 ('FedCP-QQ per class, non-private', 'fedcpqq_pc', 'inf')]), '']

    # ---------------- 2026-10-20 batch: client scale on CIFAR-10-LT (Natnael Gap 3)
    if any(c in set(S.cfg) for c in ('cifar10_K50_dir0.5', 'cifar10_K100_dir0.5')):
        L += ['## Client scale: CIFAR-10-LT, Dir(0.5)', '',
              '| K | eps | PC-FCP tail | PC-FCP worst | PC-FCP set size | A2 tail | A2 worst | A2 set size | FedCP2-QQ worst | FedCP2-QQ set size |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for K in (10, 50, 100):
            cfg = f'cifar10_K{K}_dir0.5'
            for eps in ('1.0', '2.0', '3.0', '4.0', '8.0'):
                r1 = S[(S.cfg == cfg) & (S.method == 'pcfcp') & (S.eps == eps) & np.isclose(S.alpha, 0.1)]
                r2 = S[(S.cfg == cfg) & (S.method == 'abl_A2_local_noise') & (S.eps == eps) & np.isclose(S.alpha, 0.1)]
                r3 = S[(S.cfg == cfg) & (S.method == 'fedcp2qq_marg') & (S.eps == eps) & np.isclose(S.alpha, 0.1)]
                if r1.empty:
                    continue
                r1 = r1.iloc[0]
                c2 = (f'{fmt(r2.iloc[0], "tail_cov")} | {fmt(r2.iloc[0], "worst_cov")} | {fmt(r2.iloc[0], "set_size")}'
                      if not r2.empty else 'n/a | n/a | n/a')
                c3 = f'{fmt(r3.iloc[0], "worst_cov")} | {fmt(r3.iloc[0], "set_size")}' if not r3.empty else 'n/a | n/a'
                L.append(f'| {K} | {eps} | {fmt(r1, "tail_cov")} | {fmt(r1, "worst_cov")} | {fmt(r1, "set_size")} | {c2} | {c3} |')
        L.append('')

    # ---------------- 2026-10-20 batch: CIFAR-100-LT at IF = 100 (Natnael Gap 3)
    cfg = 'cifar100_K10_dir0.5_if100'
    if cfg in set(S.cfg):
        L += ['## CIFAR-100-LT at IF = 100, K=10, Dir(0.5), target 90% (eps=1)', '', table(S, cfg, main_rows('1.0')), '']
        npz = sorted(glob.glob(os.path.join(a.cache, 'cifar100_K10_dir0.5*if100_seed*.npz')))
        if npz:
            n = np.bincount(np.load(npz[0])['cal_labels'], minlength=100)
            L += [f'Calibration counts: min {n.min()}, classes below 9 examples: {(n < 9).sum()} of 100', '']

    # ---------------- 2026-10-20 batch: calibration-set size (Natnael Gaps 3, 14)
    for ds in ('cifar10', 'cifar100', 'text'):
        eps = MAIN_EPS[ds]
        rows_ = [(f'{ds}_K10_dir0.5_cal20', '10% of the pool (20% of the calibration half)'),
                 (f'{ds}_K10_dir0.5_cal40', '20% of the pool (40% of the calibration half)'),
                 (f'{ds}_K10_dir0.5', '50% of the pool (all of the calibration half)')]
        if not any(c in set(S.cfg) for c, _ in rows_[:2]):
            continue
        L += [f'## Calibration-set size: {NAMES[ds]}, eps={eps} (worst / set size / always-incl.)', '',
              '| Calibration | Mondrian † | PC-FCP noise-free | PC-FCP | PC-FCP borrow | FedCP2-QQ marg | FedCP2-QQ pc | DP-FedCP |',
              '|---|---|---|---|---|---|---|---|']
        for cfg, lab in rows_:
            cells = []
            for m, e in [('mondrian', 'inf'), ('pcfcp_noise_free', 'inf'), ('pcfcp', eps), ('pcfcp_borrow', eps),
                         ('fedcp2qq_marg', eps), ('fedcp2qq_pc', eps), ('dpfedcp_marg', eps)]:
                r = S[(S.cfg == cfg) & (S.method == m) & (S.eps == e) & np.isclose(S.alpha, 0.1)]
                cells.append('n/a' if r.empty else f'{r.iloc[0].worst_cov_m:.1f} / {r.iloc[0].set_size_m:.2f} / {r.iloc[0].n_always_m:.1f}')
            L.append(f'| {lab} | ' + ' | '.join(cells) + ' |')
            npz = sorted(glob.glob(os.path.join(a.cache, cfg.replace('_cal20', '').replace('_cal40', '') + '*_seed*.npz')))
        L.append('')

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, 'w', encoding='utf-8').write('\n'.join(L))
    print(f'wrote {a.out} ({len(L)} lines)')


if __name__ == '__main__':
    main()
