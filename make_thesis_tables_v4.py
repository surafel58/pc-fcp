"""Chapter 4 tables of the thesis from the 2026-10-20 batch (results/v4), in the exact LaTeX
layout of main_c3_c4_f.tex, one file per table in tables/thesis_v4/. Each file holds only the
rows between \\midrule and \\bottomrule (plus the header line), so splice_thesis_tables.py can
replace the body of the existing table by its \\label. Rounding: Python format (as the v3
tables), half-up only where the thesis used it."""
import glob
import os
import re

import numpy as np
import pandas as pd

OUT = 'tables/thesis_v4'
SUM = 'results/v4/RESULTS_V4_summary.csv'
ACC = 'results/v4/accuracy.csv'
CACHE = 'results/v4/cache'
MAIN = {'cifar10': '1.0', 'cifar100': '1.0', 'text': '4.0'}
EPS = ('1.0', '2.0', '3.0', '4.0', '8.0')

S = pd.read_csv(SUM, dtype={'eps': str})


def row(cfg, m, eps, alpha=0.1):
    r = S[(S.cfg == cfg) & (S.method == m) & (S.eps == eps) & np.isclose(S.alpha, alpha)]
    return None if r.empty else r.iloc[0]


def pm(r, c, nd=1):
    return f'{r[c + "_m"]:.{nd}f} $\\pm$ {r[c + "_s"]:.{nd}f}' if not pd.isna(r[c + '_s']) else f'{r[c + "_m"]:.{nd}f}'


def lt_size(cfg, m, eps, alpha=0.1):
    """set size weighted by the long-tailed class frequencies (mean over draws, then seeds)"""
    base = cfg.replace('_cal20', '').replace('_cal40', '')
    vals = []
    for f in sorted(glob.glob(f'results/v4/eval/{base}*_seed*__*.csv')):
        tag = os.path.basename(f).split('__')[0]
        # exact configuration match: strip the seed and the classifier tag (_la/_b) only,
        # so cifar100_K10_dir0.5 does not pick up the _if100 files (and vice versa)
        if re.sub(r'_(la|b)(?=_|$)', '', re.sub(r'_seed\d+$', '', tag)) != cfg:
            continue
        d = pd.read_csv(f, dtype={'eps': str})
        d = d[(d.method == m) & (d.eps == eps) & np.isclose(d.alpha, alpha)]
        if d.empty:
            continue
        cal = np.load(os.path.join(CACHE, tag.split('_cal')[0] + '.npz'))['cal_labels'] if '_cal' in tag else \
            np.load(os.path.join(CACHE, tag + '.npz'))['cal_labels']
        prev = np.bincount(cal) / len(cal)
        vals.append(np.mean([np.array(s.split(), float) @ prev for s in d.per_class_size]))
    return float(np.mean(vals)) if vals else float('nan')


def macro_cov(cfg, m, eps, alpha=0.1):
    """macro-coverage: unweighted mean of the class coverages (mean over draws, then seeds).
    Equal to the balanced-test marginal on CIFAR; differs on the near-balanced 20NG test set."""
    vals = []
    for f in sorted(glob.glob(f'results/v4/eval/{cfg}*_seed*__*.csv')):
        tag = os.path.basename(f).split('__')[0]
        if re.sub(r'_(la|b)(?=_|$)', '', re.sub(r'_seed\d+$', '', tag)) != cfg:
            continue
        d = pd.read_csv(f, dtype={'eps': str})
        d = d[(d.method == m) & (d.eps == eps) & np.isclose(d.alpha, alpha)]
        if d.empty:
            continue
        vals.append(np.mean([np.array(s.split(), float).mean() for s in d.per_class_cov]))
    return 100 * float(np.mean(vals)) if vals else float('nan')


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('wrote', name)


# ------------------------------------------------ Tables 4.2 / 4.5 / 4.6 (tab:h2h_*)
H2H = [('Mondrian (central.)$^\\dagger$', 'mondrian', 'inf'), ('Clustered CP$^\\dagger$', 'clustered_ding', 'inf'),
       ('DP-FedCP', 'dpfedcp_marg', None), ('DP-FedCP, per class', 'dpfedcp_pc', None),
       ('FedCP2-QQ', 'fedcp2qq_marg', None), ('FedCP2-QQ, per class', 'fedcp2qq_pc', None),
       ('FedCP2-QQ, per class, private fallback', 'fedcp2qq_pc_pfallback', None),
       ('\\textbf{PC-FCP (ours)}', 'pcfcp', None)]
for ds, lab in [('cifar10', 'h2h_c10'), ('cifar100', 'h2h_c100'), ('text', 'h2h_text')]:
    cfg, eps = f'{ds}_K10_dir0.5', MAIN[ds]
    L = ['Method & Macro-cov. & Marg.\\ (LT) & Tail & Worst class & UnderCov & Set size & Size (LT) & Always \\\\', '\\midrule']
    for name, m, e in H2H:
        r = row(cfg, m, e or eps)
        if r is None:
            L.append(f'{name} & ' + ' & '.join(['n/a'] * 8) + ' \\\\'); continue
        L.append(f'{name} & {macro_cov(cfg, m, e or eps):.1f} & {r.marginal_lt_m:.1f} & {r.tail_cov_m:.1f} & {pm(r, "worst_cov")} & '
                 f'{pm(r, "undercov", 2)} & {pm(r, "set_size", 2)} & {lt_size(cfg, m, e or eps):.2f} & {r.n_always_m:.1f} \\\\')
    write(f'{lab}.tex', L)

# ------------------------------------------------ Table: noise removed (tab:noise_elim), CIFAR-10-LT
NOISE = [('DP-FedCP', 'mapped $\\varepsilon = 1$', 'dpfedcp_marg', '1.0'), ('DP-FedCP', '$\\sigma_g = 100$', 'dpfedcp_marg_sg100', 'sg100'),
         ('DP-FedCP', 'none', 'dpfedcp_marg_nonoise', 'inf'), ('DP-FedCP, per class', 'mapped $\\varepsilon = 1$', 'dpfedcp_pc', '1.0'),
         ('DP-FedCP, per class', 'none', 'dpfedcp_pc_nonoise', 'inf'), ('FedCP2-QQ', '$\\varepsilon = 1$', 'fedcp2qq_marg', '1.0'),
         ('FedCP-QQ', 'none', 'fedcpqq_marg', 'inf'), ('FedCP2-QQ, per class', '$\\varepsilon = 1$', 'fedcp2qq_pc', '1.0'),
         ('FedCP-QQ, per class', 'none', 'fedcpqq_pc', 'inf'), ('PC-FCP', '$\\varepsilon = 1$', 'pcfcp', '1.0'),
         ('PC-FCP', 'none', 'pcfcp_noise_free', 'inf')]
L = ['Method & Noise & Marg.\\ (LT) & Tail & Worst class & Set size \\\\', '\\midrule']
groups = {'DP-FedCP, per class': 1, 'FedCP2-QQ': 1, 'FedCP2-QQ, per class': 1, 'PC-FCP': 1}
last = None
for name, nz, m, e in NOISE:
    if name != last and last is not None and name in groups:
        L.append('\\midrule')
    last = name
    r = row('cifar10_K10_dir0.5', m, e)
    L.append(f'{name} & {nz} & ' + (' & '.join(['n/a'] * 4) if r is None else
             f'{r.marginal_lt_m:.1f} & {r.tail_cov_m:.1f} & {r.worst_cov_m:.1f} & {r.set_size_m:.2f}') + ' \\\\')
write('noise_elim.tex', L)

# ------------------------------------------------ Table: privacy budgets (tab:eps)
L = ['Dataset & Method & ' + ' & '.join(f'$\\varepsilon = {e[0]}$' for e in EPS) + ' \\\\', '\\midrule']
NAMES = {'cifar10': 'CIFAR-10-LT', 'cifar100': 'CIFAR-100-LT', 'text': '20 Newsgroups'}
for i, ds in enumerate(('cifar10', 'cifar100', 'text')):
    cfg = f'{ds}_K10_dir0.5'
    for j, (name, m) in enumerate([('DP-FedCP', 'dpfedcp_marg'), ('FedCP2-QQ', 'fedcp2qq_marg'), ('PC-FCP', 'pcfcp')]):
        cells = []
        for e in EPS:
            r = row(cfg, m, e)
            if r is None:
                cells.append('n/a'); continue
            w = f'{r.worst_cov_m:.1f}'
            cells.append(f'{w} / {r.set_size_m:.2f}')
        L.append((NAMES[ds] if j == 0 else '') + f' & {name} & ' + ' & '.join(cells) + ' \\\\')
    if i < 2:
        L.append('\\midrule')
write('eps.tex', L)

# ------------------------------------------------ Table: ablations (tab:ablation), 3 datasets
ABL = [('A1: threshold averaging', 'abl_A1_avg'), ('A1: weighted averaging', 'abl_A1_weighted'),
       ('A2: per-client noise', 'abl_A2_local_noise'), ('A3: marginal read-out', 'abl_A3_marginal'),
       ('A4: without the floor', 'abl_nofloor'), ('A5: bottom-up read-out', 'abl_bottomup'),
       ('A5: bottom-up with consistency', 'abl_consistency'), ('A6: LAC score', 'abl_lac'),
       ('A8: tree-structured release', 'abl_A8_tree'), ('PC-FCP (full method)', 'pcfcp')]
L = ['Variant & CIFAR-10-LT & CIFAR-100-LT & 20 Newsgroups \\\\', '\\midrule']
for name, m in ABL:
    cells = []
    for ds in ('cifar10', 'cifar100', 'text'):
        r = row(f'{ds}_K10_dir0.5', m, MAIN[ds])
        cells.append('n/a' if r is None else f'{r.worst_cov_m:.1f} / {r.set_size_m:.2f}')
    if m == 'pcfcp':
        L.append('\\midrule')
    L.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
write('ablation.tex', L)

# ------------------------------------------------ Table: bins (tab:bins)
BINS = [('Cubic, $B = 5$', 'pcfcp_p3_B5'), ('Cubic, $B = 8$ (PC-FCP)', 'pcfcp'), ('Cubic, $B = 12$', 'pcfcp_p3_B12'),
        ('Cubic, $B = 20$', 'pcfcp_p3_B20'), ('Cubic, $B = 50$', 'pcfcp_p3_B50'), ('Cubic, $B = 100$', 'pcfcp_p3_B100'),
        ('Quadratic, $B = 8$', 'pcfcp_p2_B8'), ('Quadratic, $B = 100$', 'pcfcp_p2_B100'), ('Equal width, $B = 50$', 'pcfcp_uniform50')]
L = ['Bins & CIFAR-10-LT & CIFAR-100-LT \\\\', '\\midrule']
for name, m in BINS:
    cells = []
    for ds in ('cifar10', 'cifar100'):
        r = row(f'{ds}_K10_dir0.5', m, '1.0')
        cells.append('n/a' if r is None else f'{r.worst_cov_m:.1f} / {r.set_size_m:.2f}')
    L.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
write('bins.tex', L)

# ------------------------------------------------ Table: target levels (tab:alpha), CIFAR-10-LT
L = ['Target (percent) & Centralized Mondrian$^\\dagger$ & FedCP2-QQ & PC-FCP \\\\', '\\midrule']
for a in (0.05, 0.1, 0.2, 0.3, 0.35):
    cells = []
    for m, e in [('mondrian', 'inf'), ('fedcp2qq_marg', '1.0'), ('pcfcp', '1.0')]:
        r = row('cifar10_K10_dir0.5', m, e, a)
        cells.append('n/a' if r is None else f'{r.worst_cov_m:.1f} / {r.undercov_m:.2f} / {r.set_size_m:.2f}')
    L.append(f'{round(100 * (1 - a))} & ' + ' & '.join(cells) + ' \\\\')
write('alpha.tex', L)

# ------------------------------------------------ Table: near-IID (tab:iid)
L = ['Method & Non-IID & Near-IID \\\\', '\\midrule']
for name, m, e in [('Centralized Mondrian$^\\dagger$', 'mondrian', 'inf'), ('DP-FedCP', 'dpfedcp_marg', '1.0'),
                   ('FedCP2-QQ', 'fedcp2qq_marg', '1.0'), ('FedCP-QQ, per class, no noise', 'fedcpqq_pc', 'inf'),
                   ('PC-FCP', 'pcfcp', '1.0')]:
    a, b = row('cifar10_K10_dir0.5', m, e), row('cifar10_K10_dir100', m, e)
    f = lambda r: 'n/a' if r is None else f'{r.worst_cov_m:.1f} / {r.set_size_m:.2f}'
    L.append(f'{name} & {f(a)} & {f(b)} \\\\')
write('iid.tex', L)

# ------------------------------------------------ Table: client scale (tab:clients), CIFAR-100 and CIFAR-10
for ds, lab in [('cifar100', 'clients'), ('cifar10', 'clients_c10')]:
    L = ['Clients & Method & $\\varepsilon = 1$ & $\\varepsilon = 4$ & $\\varepsilon = 8$ \\\\', '\\midrule']
    for K in (10, 50, 100):
        for name, m in [('PC-FCP', 'pcfcp'), ('A2', 'abl_A2_local_noise')]:
            cells = []
            for e in ('1.0', '4.0', '8.0'):
                r = row(f'{ds}_K{K}_dir0.5', m, e)
                cells.append('n/a' if r is None else pm(r, 'worst_cov'))
            L.append(f'{K} & {name} & ' + ' & '.join(cells) + ' \\\\')
    write(f'{lab}.tex', L)

# ------------------------------------------------ Table: accuracy (tab:accuracy)
acc = pd.read_csv(ACC).drop_duplicates('tag', keep='last')
acc['cfg'] = acc.tag.str.replace(r'_seed\d+$', '', regex=True).str.replace('_la', '', regex=False).str.replace('_b', '', regex=False)
ROWS = [('cifar10_K10_dir0.5', 'CIFAR-10-LT, $K = 10$'), ('cifar10_K10_dir100', 'CIFAR-10-LT, $K = 10$, Dirichlet(100)'),
        ('cifar10_K50_dir0.5', 'CIFAR-10-LT, $K = 50$'), ('cifar10_K100_dir0.5', 'CIFAR-10-LT, $K = 100$'),
        ('cifar100_K10_dir0.5', 'CIFAR-100-LT, $K = 10$'), ('cifar100_K10_dir0.5_if100', 'CIFAR-100-LT (IF = 100), $K = 10$'),
        ('cifar100_K50_dir0.5', 'CIFAR-100-LT, $K = 50$'), ('cifar100_K100_dir0.5', 'CIFAR-100-LT, $K = 100$'),
        ('text_K10_dir0.5', '20 Newsgroups, $K = 10$')]
L = ['Dataset and clients & Top-1 & Head & Tail & Worst class & Top-5 \\\\', '\\midrule']
for cfg, name in ROWS:
    x = acc[acc.cfg == cfg]
    if x.empty:
        continue
    f = lambda c: f'{100 * x[c].mean():.1f} $\\pm$ {100 * x[c].std(ddof=1):.1f}'
    L.append(f'{name} & {f("top1")} & {f("top1_head")} & {f("top1_tail")} & {f("top1_worst")} & {f("top5")} \\\\')
write('accuracy.tex', L)

# ------------------------------------------------ Table: calibration-set size (new), 3 datasets
L = ['Dataset & Calibration (percent) & Mondrian$^\\dagger$ & PC-FCP & PC-FCP, borrow & FedCP2-QQ & FedCP2-QQ, per class \\\\', '\\midrule']
for i, ds in enumerate(('cifar10', 'cifar100', 'text')):
    eps = MAIN[ds]
    for j, (suf, name) in enumerate([('_cal20', '10'), ('_cal40', '20'), ('', '50')]):
        cfg = f'{ds}_K10_dir0.5{suf}'
        cells = []
        for m, e in [('mondrian', 'inf'), ('pcfcp', eps), ('pcfcp_borrow', eps), ('fedcp2qq_marg', eps), ('fedcp2qq_pc', eps)]:
            r = row(cfg, m, e)
            cells.append('n/a' if r is None else f'{r.worst_cov_m:.1f} / {r.set_size_m:.2f} / {r.n_always_m:.1f}')
        L.append((NAMES[ds] if j == 0 else '') + f' & {name} & ' + ' & '.join(cells) + ' \\\\')
    if i < 2:
        L.append('\\midrule')
write('calfrac.tex', L)

# ------------------------------------------------ Table: IF = 100 on CIFAR-100 (new)
cfg = 'cifar100_K10_dir0.5_if100'
L = ['Method & Marg.\\ (LT) & Tail & Worst class & UnderCov & Set size & Always \\\\', '\\midrule']
for name, m, e in H2H + [('PC-FCP, borrow', 'pcfcp_borrow', None)]:
    r = row(cfg, m, e or '1.0')
    L.append(f'{name} & ' + ('n/a & ' * 5 + 'n/a' if r is None else
             f'{r.marginal_lt_m:.1f} & {r.tail_cov_m:.1f} & {pm(r, "worst_cov")} & {r.undercov_m:.2f} & {r.set_size_m:.2f} & {r.n_always_m:.1f}') + ' \\\\')
write('if100.tex', L)
