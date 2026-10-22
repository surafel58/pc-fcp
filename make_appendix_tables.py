"""Appendix tables of the thesis (Natnael, Gaps 2 and 4): the three comparison tables with
a standard deviation over seeds for every column, and the per-seed results of the main
methods. Every metric is averaged over the noise draws of a seed first. Writes
tables/appendix_sd.tex and tables/appendix_seeds.tex."""
import glob
import os

import numpy as np
import pandas as pd

RES = os.environ.get('PCFCP_RES', 'results/v3')   # 2026-10-20 batch: PCFCP_RES=results/v4
TAB = os.environ.get('PCFCP_TABLES', 'tables')     # output directory (tables_v4 for the batch)
MAIN = [('cifar10', '1.0', 'CIFAR-10-LT (IF = 100), $\\varepsilon = 1$', 'tab:app_sd_c10'),
        ('cifar100', '1.0', 'CIFAR-100-LT (IF = 10), $\\varepsilon = 1$', 'tab:app_sd_c100'),
        ('text', '4.0', '20 Newsgroups (IF = 10), $\\varepsilon = 4$', 'tab:app_sd_text')]
ROWS = [('Mondrian (central.)$^\\dagger$', 'mondrian'), ('Clustered CP$^\\dagger$', 'clustered_ding'),
        ('DP-FedCP', 'dpfedcp_marg'), ('DP-FedCP, per class', 'dpfedcp_pc'), ('FedCP2-QQ', 'fedcp2qq_marg'),
        ('FedCP2-QQ, per class', 'fedcp2qq_pc'), ('FedCP2-QQ, per class, fallback', 'fedcp2qq_pc_fallback'),
        ('\\textbf{PC-FCP (ours)}', 'pcfcp')]
COLS = [('macro', 1), ('marg_lt', 1), ('tail', 1), ('worst', 1), ('undercov', 2), ('size', 2), ('size_lt', 2),
        ('always', 1)]


def f(x, nd):
    # same rounding as the comparison tables of Chapter 4 (Python formatting)
    return f'{float(x):.{nd}f}'


def per_seed(ds, eps):
    rows = []
    for fn in sorted(glob.glob(f'{RES}/eval/{ds}_K10_dir0.5*_seed*__*.csv')):
        if '_cal' in os.path.basename(fn) or '_if100' in os.path.basename(fn):
            continue
        d = pd.read_csv(fn, dtype={'eps': str})
        d = d[((d.alpha - 0.1).abs() < 1e-9) & d.method.isin([m for _, m in ROWS])]
        d = d[(d.eps == eps) | (d.eps == 'inf')]
        d = d[~((d.eps == 'inf') & d.method.str.startswith('dpfedcp'))]
        if d.empty:
            continue
        prev = np.bincount(np.load(f'{RES}/cache/{d.tag.iloc[0]}.npz')['cal_labels'])
        prev = prev / prev.sum()
        for _, r in d.iterrows():
            cov = np.array(r.per_class_cov.split(), float)
            sz = np.array(r.per_class_size.split(), float)
            rows.append({'method': r.method, 'seed': r.seed, 'macro': 100 * cov.mean(),
                         'marg_lt': 100 * r.marginal_lt, 'tail': 100 * r.tail_cov, 'worst': 100 * r.worst_cov,
                         'undercov': r.undercov, 'size': r.set_size, 'size_lt': float(sz @ prev),
                         'always': r.n_always})
    return pd.DataFrame(rows).groupby(['method', 'seed']).mean()


def sd_tables():
    out = []
    for ds, eps, title, lab in MAIN:
        x = per_seed(ds, eps)
        g = x.groupby('method')
        mu, sd = g.mean(), g.std(ddof=1)
        L = ['\\begin{table}[H]', '\\centering',
             '\\caption[Comparison with standard deviations, ' + title.split(' (')[0] + ']{Baseline comparison on '
             + title + ', mean $\\pm$ standard deviation over 3 seeds for every column. Columns as in '
             'Table~\\ref{tab:h2h_c10}.}', '\\label{' + lab + '}', '\\footnotesize', '\\setlength{\\tabcolsep}{2.5pt}',
             '\\begin{adjustbox}{max width=\\textwidth}', '\\begin{tabular}{l' + '|r' * len(COLS) + '}', '\\toprule',
             'Method & Macro-cov. & Marg.\\ (LT) & Tail & Worst class & UnderCov & Set size & Size (LT) & Always \\\\',
             '\\midrule']
        for name, m in ROWS:
            cells = [f(mu.loc[m, c], nd) + ' $\\pm$ ' + f(sd.loc[m, c], nd) for c, nd in COLS]
            L.append(name + ' & ' + ' & '.join(cells) + ' \\\\')
        L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\end{table}', '']
        out.append('\n'.join(L))
    return '\n'.join(out)


def seed_table():
    meth = [('Mondrian$^\\dagger$', 'mondrian'), ('DP-FedCP', 'dpfedcp_marg'), ('FedCP2-QQ', 'fedcp2qq_marg'),
            ('PC-FCP', 'pcfcp')]
    L = ['\\begin{table}[H]', '\\centering',
         '\\caption[Per-seed results]{Per-seed results of the main methods: worst-class coverage in percent / mean '
         'set size in labels, at $\\varepsilon = 1$ on the image datasets and $\\varepsilon = 4$ on 20 '
         'Newsgroups. Each entry is averaged over the noise draws of its seed.}',
         '\\label{tab:app_seeds}', '\\small', '\\begin{tabular}{ll|r|r|r|r}', '\\toprule',
         'Dataset & Seed & ' + ' & '.join(n for n, _ in meth) + ' \\\\', '\\midrule']
    names = {'cifar10': 'CIFAR-10-LT', 'cifar100': 'CIFAR-100-LT', 'text': '20 Newsgroups'}
    for i, (ds, eps, _, _) in enumerate(MAIN):
        x = per_seed(ds, eps)
        for j, seed in enumerate(sorted(x.index.get_level_values('seed').unique())):
            cells = [f(x.loc[(m, seed), 'worst'], 1) + ' / ' + f(x.loc[(m, seed), 'size'], 2) for _, m in meth]
            L.append((names[ds] if j == 0 else '') + f' & {seed} & ' + ' & '.join(cells) + ' \\\\')
        if i < len(MAIN) - 1:
            L.append('\\midrule')
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table}', '']
    return '\n'.join(L)


if __name__ == '__main__':
    os.makedirs(TAB, exist_ok=True)
    open(f'{TAB}/appendix_sd.tex', 'w', encoding='utf-8').write(sd_tables())
    open(f'{TAB}/appendix_seeds.tex', 'w', encoding='utf-8').write(seed_table())
    print(open(f'{TAB}/appendix_seeds.tex', encoding='utf-8').read())
