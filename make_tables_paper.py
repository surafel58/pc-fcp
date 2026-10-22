"""LaTeX tables for the IEEE TAI manuscript (paper/tables/), generated from the evaluation
summary (results/v3/RESULTS_V3_summary.csv, written by analyze_results_v3.py) and
results/v3/accuracy.csv, so no number is typed by hand. Coverage in percent; UnderCov in
percentage points; set size in labels. Means over 3 seeds of the per-seed means over 20
noise draws (3 for DP-FedCP); +- is the standard deviation over seeds."""
import os

import numpy as np
import pandas as pd

OUT = 'paper/tables'
RES = os.environ.get('PCFCP_RES', 'results/v3')   # 2026-10-20 batch: PCFCP_RES=results/v4
Sall = pd.read_csv(f'{RES}/RESULTS_V{RES[-1]}_summary.csv', dtype={'eps': str})
S = Sall[np.isclose(Sall.alpha, 0.1)]
ACC = pd.read_csv(f'{RES}/accuracy.csv').drop_duplicates('tag', keep='last')
ACC['cfg'] = ACC.tag.str.replace(r'_seed\d+$', '', regex=True).str.replace('_la', '', regex=False).str.replace('_b', '', regex=False)

MAIN = [('cifar10_K10_dir0.5', '1.0', 'CIFAR-10-LT ($\\varepsilon = 1$)'),
        ('cifar100_K10_dir0.5', '1.0', 'CIFAR-100-LT ($\\varepsilon = 1$)'),
        ('text_K10_dir0.5', '4.0', '20 Newsgroups ($\\varepsilon = 4$)')]
ROWS = [('Mondrian$^\\dagger$', 'mondrian', 'ref'),
        ('Clustered CP$^\\dagger$', 'clustered_ding', 'ref'),
        ('DP-FedCP', 'dpfedcp_marg', 'eps'),
        ('\\quad per class', 'dpfedcp_pc', 'eps'),
        ('FedCP2-QQ', 'fedcp2qq_marg', 'eps'),
        ('\\quad per class', 'fedcp2qq_pc', 'eps'),
        ('\\quad per class, private fallback', 'fedcp2qq_pc_pfallback', 'eps'),
        ('\\textbf{PC-FCP (ours)}', 'pcfcp', 'eps')]


def get(cfg, method, eps):
    r = S[(S.cfg == cfg) & (S.method == method) & (S.eps == eps)]
    assert len(r) == 1, (cfg, method, eps, len(r))
    return r.iloc[0]


def f(x, nd=1):
    # round half up, as in the thesis tables (0.215 -> 0.22)
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(round(float(x), 10))).quantize(q, rounding=ROUND_HALF_UP))


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), 'w', encoding='utf-8').write(text)
    print('wrote', name)


# ------------------------------------------------------------ Table III: main comparison
def table_main():
    L = [r'\begin{table*}[!t]',
         r'\caption{Comparison on the three datasets at the main privacy budget of each (target coverage 90\%). '
         r'Marg.\ is the marginal coverage weighted by the long-tailed class frequencies; Worst is the '
         r'worst-class coverage ($\pm$ standard deviation over three seeds); UnderCov is the mean shortfall '
         r'below the target over classes, in percentage points; Size is the mean prediction set size in labels on the '
         r'class-balanced test set. '
         r'$^\dagger$Non-private references that pool the raw calibration scores.}',
         r'\label{tab:main}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{l' + 'rrrr' * 3 + '}', r'\toprule',
         ' & ' + ' & '.join(r'\multicolumn{4}{c}{' + t + '}' for _, _, t in MAIN) + r' \\',
         ''.join(r'\cmidrule(lr){%d-%d}' % (2 + 4 * i, 5 + 4 * i) for i in range(3)),
         'Method' + ' & Marg. & Worst & UnderCov & Size' * 3 + r' \\', r'\midrule']
    for lab, m, kind in ROWS:
        cells = []
        for cfg, eps, _ in MAIN:
            r = get(cfg, m, 'inf' if kind == 'ref' else eps)
            cells += [f(r.marginal_lt_m), f'{f(r.worst_cov_m)}$\\pm${f(r.worst_cov_s)}',
                      f(r.undercov_m, 2), f(r.set_size_m, 2)]
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    write('table_main.tex', '\n'.join(L))


# ------------------------------------------------------------ Table IV: with and without noise
def table_noise():
    cfg = 'cifar10_K10_dir0.5'
    rows = [('DP-FedCP', 'dpfedcp_marg', '1.0', '$\\varepsilon = 1$'),
            ('', 'dpfedcp_marg_sg100', 'sg100', '$\\sigma_g = 100$'),
            ('', 'dpfedcp_marg_nonoise', 'inf', 'none'),
            ('DP-FedCP, per class', 'dpfedcp_pc', '1.0', '$\\varepsilon = 1$'),
            ('', 'dpfedcp_pc_nonoise', 'inf', 'none'),
            ('FedCP-QQ', 'fedcp2qq_marg', '1.0', '$\\varepsilon = 1$'),
            ('', 'fedcpqq_marg', 'inf', 'none'),
            ('FedCP-QQ, per class', 'fedcp2qq_pc', '1.0', '$\\varepsilon = 1$'),
            ('', 'fedcpqq_pc', 'inf', 'none'),
            ('PC-FCP', 'pcfcp', '1.0', '$\\varepsilon = 1$'),
            ('', 'pcfcp_noise_free', 'inf', 'none')]
    L = [r'\begin{table}[!t]',
         r'\caption{Baselines with and without their privacy noise on CIFAR-10-LT (target coverage 90\%). '
         r"Columns as in Table~\ref{tab:main}. With noise, FedCP-QQ is the authors' private variant FedCP2-QQ; "
         r"$\sigma_g = 100$ is the noisiest "
         r'setting tested in the DP-FedCP paper.}',
         r'\label{tab:noise}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{llrrr}', r'\toprule', r'Method & Noise & Marg. & Worst & Size \\', r'\midrule']
    for i, (lab, m, eps, noise) in enumerate(rows):
        r = get(cfg, m, eps)
        if lab and i:
            L.append(r'\addlinespace[2pt]')
        L.append(f'{lab} & {noise} & {f(r.marginal_lt_m)} & {f(r.worst_cov_m)} & {f(r.set_size_m, 2)} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    write('table_noise.tex', '\n'.join(L))


# ------------------------------------------------------------ Table V: ablations
def table_ablation():
    rows = [('A1: threshold averaging', 'abl_A1_avg'), ('A1: weighted averaging', 'abl_A1_weighted'),
            ('A2: per-client noise', 'abl_A2_local_noise'), ('A3: marginal read-out', 'abl_A3_marginal'),
            ('A4: without the floor', 'abl_nofloor'), ('A5: bottom-up read-out', 'abl_bottomup'),
            ('A5: bottom-up, consistency', 'abl_consistency'), ('A6: LAC score', 'abl_lac'),
            ('A7: cubic, $B = 5$', 'pcfcp_p3_B5'), ('A7: cubic, $B = 20$', 'pcfcp_p3_B20'),
            ('A7: cubic, $B = 100$', 'pcfcp_p3_B100'), ('A7: quadratic, $B = 8$', 'pcfcp_p2_B8'),
            ('A7: equal width, $B = 50$', 'pcfcp_uniform50'),
            ('PC-FCP (cubic, $B = 8$)', 'pcfcp')]
    L = [r'\begin{table}[!t]',
         r'\caption{Ablations: worst-class coverage (\%) / mean set size (labels), at $\varepsilon = 1$ on the image '
         r'datasets and $\varepsilon = 4$ on 20 Newsgroups. A1 removes the histogram channel, A2 secure aggregation, '
         r'A3 the per-class structure, A4 the floor; A5 and A6 replace the read-out and the score; A7 varies the '
         r'bins of~\eqref{eq:edges}.}',
         r'\label{tab:ablation}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{2.5pt}',
         r'\begin{tabular}{lrrr}', r'\toprule', r'Variant & CIFAR-10-LT & CIFAR-100-LT & 20 Newsgr. \\', r'\midrule']
    for lab, m in rows:
        cells = []
        for cfg, eps, _ in MAIN:
            r = get(cfg, m, eps)
            cells.append(f'{f(r.worst_cov_m)} / {f(r.set_size_m, 2)}')
        if m == 'pcfcp':
            L.append(r'\midrule')
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    write('table_ablation.tex', '\n'.join(L))


# ------------------------------------------------------------ Table VI: number of clients
def table_clients():
    L = [r'\begin{table}[!t]',
         r'\caption{Worst-class coverage (\%, $\pm$ standard deviation over three seeds) on CIFAR-100-LT against the '
         r'number of clients $K$: PC-FCP with noise inside the secure sum against per-client noise (A2).}',
         r'\label{tab:clients}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{llrrr}', r'\toprule',
         r'$K$ & Method & $\varepsilon = 1$ & $\varepsilon = 4$ & $\varepsilon = 8$ \\', r'\midrule']
    for K in (10, 50, 100):
        cfg = f'cifar100_K{K}_dir0.5'
        for lab, m in (('PC-FCP', 'pcfcp'), ('A2', 'abl_A2_local_noise')):
            cells = []
            for eps in ('1.0', '4.0', '8.0'):
                r = get(cfg, m, eps)
                cells.append(f'{f(r.worst_cov_m)}$\\pm${f(r.worst_cov_s)}')
            L.append(f'{K if lab == "PC-FCP" else ""} & {lab} & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    write('table_clients.tex', '\n'.join(L))


# ------------------------------------------------------------ Table II: datasets
def table_datasets():
    splits = {'cifar10_K10_dir0.5': ('CIFAR-10-LT', 10, 100, '6{,}201', '6{,}207', '10{,}000'),
              'cifar100_K10_dir0.5': ('CIFAR-100-LT', 100, 10, '9{,}790', '9{,}839', '10{,}000'),
              'text_K10_dir0.5': ('20 Newsgroups', 20, 10, '1{,}789', '1{,}805', '7{,}532')}
    L = [r'\begin{table}[!t]',
         r'\caption{Datasets: classes $C$, imbalance factor IF, sizes of the training, calibration, and test splits, '
         r'and top-1 test accuracy of the federated classifier with $K = 10$ clients (\%, mean $\pm$ standard '
         r'deviation over three seeds), overall and on the tail classes.}',
         r'\label{tab:datasets}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{2.5pt}',
         r'\begin{tabular}{lrrrrrrr}', r'\toprule',
         r'Dataset & $C$ & IF & Train & Calib. & Test & Top-1 & Tail \\', r'\midrule']
    for cfg, (name, C, IF, tr, ca, te) in splits.items():
        a = ACC[ACC.cfg == cfg]
        t1 = f'{f(100 * a.top1.mean())}$\\pm${f(100 * a.top1.std(ddof=1))}'
        tt = f'{f(100 * a.top1_tail.mean())}$\\pm${f(100 * a.top1_tail.std(ddof=1))}'
        L.append(f'{name} & {C} & {IF} & {tr} & {ca} & {te} & {t1} & {tt} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    write('table_datasets.tex', '\n'.join(L))


# ============================================================ supplement (tables S-I to S-V)
NAMES = {'cifar10_K10_dir0.5': 'CIFAR-10-LT', 'cifar100_K10_dir0.5': 'CIFAR-100-LT', 'text_K10_dir0.5': '20 Newsgroups'}
EPS = ('1.0', '2.0', '3.0', '4.0', '8.0')


def ws(r):
    return f'{f(r.worst_cov_m)} / {f(r.set_size_m, 2)}'


def supp_eps():
    rows = [('PC-FCP', 'pcfcp'), ('FedCP2-QQ', 'fedcp2qq_marg'), ('FedCP2-QQ, per class', 'fedcp2qq_pc'),
            ('FedCP2-QQ, per class, private fallback', 'fedcp2qq_pc_pfallback'), ('DP-FedCP', 'dpfedcp_marg'),
            ('DP-FedCP, per class', 'dpfedcp_pc'), ('A2: per-client noise', 'abl_A2_local_noise'),
            ('A4: without the floor', 'abl_nofloor')]
    L = [r'\begin{table*}[!t]',
         r'\caption{Privacy budget: worst-class coverage (\%) / mean set size (labels) at each $\varepsilon$ '
         r'(target coverage 90\%, $K = 10$).}',
         r'\label{tab:s_eps}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{l' + 'r' * len(EPS) + '}', r'\toprule',
         'Method & ' + ' & '.join(f'$\\varepsilon = {e[:-2]}$' for e in EPS) + r' \\']
    for cfg, name in NAMES.items():
        L += [r'\midrule', r'\multicolumn{6}{l}{\textit{' + name + r'}} \\']
        for lab, m in rows:
            L.append(lab + ' & ' + ' & '.join(ws(get(cfg, m, e)) for e in EPS) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    write('supp_eps.tex', '\n'.join(L))


def supp_target():
    MS = [('Mondrian$^\\dagger$', 'mondrian', 'inf'), ('PC-FCP, no noise', 'pcfcp_noise_free', 'inf'),
          ('PC-FCP', 'pcfcp', None), ('FedCP2-QQ', 'fedcp2qq_marg', None)]
    L = [r'\begin{table}[!t]',
         r'\caption{Target level: worst-class coverage (\%) / mean set size (labels), at $\varepsilon = 1$ on the '
         r'image datasets and $\varepsilon = 4$ on 20 Newsgroups. $^\dagger$Non-private reference.}',
         r'\label{tab:s_target}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{lrrrr}', r'\toprule', 'Target & ' + ' & '.join(m[0] for m in MS) + r' \\']
    for cfg, eps, _ in MAIN:
        L += [r'\midrule', r'\multicolumn{5}{l}{\textit{' + NAMES[cfg] + r'}} \\']
        for a in (0.05, 0.1, 0.2, 0.3, 0.35):
            cells = []
            for _, m, e in MS:
                r = Sall[(Sall.cfg == cfg) & (Sall.method == m) & (Sall.eps == (e or eps)) & np.isclose(Sall.alpha, a)]
                assert len(r) == 1, (cfg, m, a)
                cells.append(ws(r.iloc[0]))
            L.append(f'{round(100 * (1 - a))}\\% & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    write('supp_target.tex', '\n'.join(L))


def supp_neariid():
    cfg = 'cifar10_K10_dir100'
    rows = ROWS[:-1] + [('PC-FCP, no noise', 'pcfcp_noise_free', 'ref'), ROWS[-1]]
    L = [r'\begin{table}[!t]',
         r'\caption{Near-IID partition: CIFAR-10-LT with Dirichlet concentration 100, $K = 10$, '
         r'$\varepsilon = 1$, target coverage 90\%. Columns as in Table~\ref{tab:main}; the DP-FedCP rows are '
         r'from one seed, the others from three. $^\dagger$Non-private references.}',
         r'\label{tab:s_neariid}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{lrrrr}', r'\toprule', r'Method & Marg. & Worst & UnderCov & Size \\', r'\midrule']
    for lab, m, kind in rows:
        r = get(cfg, m, 'inf' if kind == 'ref' else '1.0')
        sd = f'$\\pm${f(r.worst_cov_s)}' if r.n_seeds > 1 and not np.isnan(r.worst_cov_s) else ''
        L.append(f'{lab} & {f(r.marginal_lt_m)} & {f(r.worst_cov_m)}{sd} & {f(r.undercov_m, 2)} & '
                 f'{f(r.set_size_m, 2)} \\\\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    write('supp_neariid.tex', '\n'.join(L))


def supp_bins():
    grid = [('cubic, $B = 5$', 'pcfcp_p3_B5'), ('cubic, $B = 8$ (PC-FCP)', 'pcfcp'),
            ('cubic, $B = 12$', 'pcfcp_p3_B12'), ('cubic, $B = 20$', 'pcfcp_p3_B20'),
            ('cubic, $B = 50$', 'pcfcp_p3_B50'), ('cubic, $B = 100$', 'pcfcp_p3_B100'),
            ('quadratic, $B = 8$', 'pcfcp_p2_B8'), ('quadratic, $B = 100$', 'pcfcp_p2_B100'),
            ('equal width, $B = 50$', 'pcfcp_uniform50')]
    L = [r'\begin{table*}[!t]',
         r'\caption{Bin layout: worst-class coverage (\%) / mean set size (labels) with the privacy noise '
         r'($\varepsilon = 1$ on the image datasets, $\varepsilon = 4$ on 20 Newsgroups) and without it.}',
         r'\label{tab:s_bins}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
         r'\begin{tabular}{lrrrrrr}', r'\toprule',
         ' & ' + ' & '.join(r'\multicolumn{2}{c}{' + NAMES[c] + '}' for c, _, _ in MAIN) + r' \\',
         ''.join(r'\cmidrule(lr){%d-%d}' % (2 + 2 * i, 3 + 2 * i) for i in range(3)),
         'Bins' + ' & private & no noise' * 3 + r' \\', r'\midrule']
    for lab, m in grid:
        cells = []
        for cfg, eps, _ in MAIN:
            cells += [ws(get(cfg, m, eps)), ws(get(cfg, m + '_noise_free', 'inf'))]
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table*}', '']
    write('supp_bins.tex', '\n'.join(L))


def supp_accuracy():
    cfgs = [('cifar10_K10_dir0.5', 'CIFAR-10-LT', 10, '0.5'), ('cifar10_K10_dir100', 'CIFAR-10-LT', 10, '100'),
            ('cifar100_K10_dir0.5', 'CIFAR-100-LT', 10, '0.5'), ('cifar100_K50_dir0.5', 'CIFAR-100-LT', 50, '0.5'),
            ('cifar100_K100_dir0.5', 'CIFAR-100-LT', 100, '0.5'), ('text_K10_dir0.5', '20 Newsgroups', 10, '0.5')]
    L = [r'\begin{table}[!t]',
         r'\caption{Top-1 test accuracy of the federated classifier (\%, mean $\pm$ standard deviation over three '
         r'seeds) for every training configuration: number of clients $K$ and Dirichlet concentration.}',
         r'\label{tab:s_accuracy}', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{3pt}',
         r'\begin{tabular}{lrrrrr}', r'\toprule', r'Dataset & $K$ & Dir. & Top-1 & Head & Tail \\', r'\midrule']
    for cfg, name, K, dr in cfgs:
        a = ACC[ACC.cfg == cfg]
        assert len(a) == 3, cfg
        c = [f'{f(100 * a[k].mean())}$\\pm${f(100 * a[k].std(ddof=1))}' for k in ('top1', 'top1_head', 'top1_tail')]
        L.append(f'{name} & {K} & {dr} & ' + ' & '.join(c) + r' \\')
    L += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    write('supp_accuracy.tex', '\n'.join(L))


if __name__ == '__main__':
    table_main(); table_noise(); table_ablation(); table_clients(); table_datasets()
    supp_eps(); supp_target(); supp_neariid(); supp_bins(); supp_accuracy()
