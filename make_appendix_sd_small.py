"""Appendix: standard deviations over seeds for the smaller tables of Chapter 4 (privacy
budgets, ablations, bins, target levels, near-IID, noise removed), which show only means to
stay readable (Beakal T8, Natnael Gap 2). Each cell is worst-class coverage +- sd / set size
+- sd. Writes tables/appendix_sd_small.tex."""
import numpy as np
import pandas as pd

import os
RES = os.environ.get('PCFCP_RES', 'results/v3')   # 2026-10-20 batch: PCFCP_RES=results/v4
S = pd.read_csv(f'{RES}/RESULTS_V{RES[-1]}_summary.csv', dtype={'eps': str})
C10, C100, TX, IID = 'cifar10_K10_dir0.5', 'cifar100_K10_dir0.5', 'text_K10_dir0.5', 'cifar10_K10_dir100'


def cell(cfg, m, eps, alpha=0.1):
    r = S[(S.cfg == cfg) & (S.method == m) & (S.eps == eps) & np.isclose(S.alpha, alpha)]
    assert len(r) == 1, (cfg, m, eps, alpha)
    r = r.iloc[0]
    ws = '' if np.isnan(r.worst_cov_s) else f' $\\pm$ {r.worst_cov_s:.1f}'
    ss = '' if np.isnan(r.set_size_s) else f' $\\pm$ {r.set_size_s:.2f}'
    return f'{r.worst_cov_m:.1f}{ws} / {r.set_size_m:.2f}{ss}'


def table(label, caption, head, rows, spec):
    L = ['\\begin{table}[H]', '\\centering', '\\caption{' + caption + '}', '\\label{' + label + '}',
         '\\footnotesize', '\\setlength{\\tabcolsep}{3pt}', '\\begin{adjustbox}{max width=\\textwidth}',
         '\\begin{tabular}{' + spec + '}', '\\toprule', head + ' \\\\', '\\midrule']
    L += [r if r.startswith('\\midrule') else r + ' \\\\' for r in rows]
    L += ['\\bottomrule', '\\end{tabular}', '\\end{adjustbox}', '\\end{table}', '']
    return '\n'.join(L)


out = []
EPS = ['1.0', '2.0', '3.0', '4.0', '8.0']
rows = []
for cfg, name in [(C10, 'CIFAR-10-LT'), (C100, 'CIFAR-100-LT'), (TX, '20 Newsgroups')]:
    for i, (lab, m) in enumerate([('DP-FedCP', 'dpfedcp_marg'), ('FedCP2-QQ', 'fedcp2qq_marg'), ('PC-FCP', 'pcfcp')]):
        rows.append((name if i == 0 else '') + ' & ' + lab + ' & ' + ' & '.join(cell(cfg, m, e) for e in EPS))
    rows.append('\\midrule')
out.append(table('tab:app_sd_eps', 'Privacy budgets (Table~\\ref{tab:eps}) with standard deviations over 3 seeds: '
                 'worst-class coverage in percent / set size in labels.',
                 'Dataset & Method & ' + ' & '.join(f'$\\varepsilon = {e[0]}$' for e in EPS), rows[:-1], 'll|r|r|r|r|r'))

ABL = [('A1: threshold averaging', 'abl_A1_avg'), ('A1: weighted averaging', 'abl_A1_weighted'),
       ('A2: per-client noise', 'abl_A2_local_noise'), ('A3: marginal read-out', 'abl_A3_marginal'),
       ('A4: without the floor', 'abl_nofloor'), ('A5: bottom-up read-out', 'abl_bottomup'),
       ('A5: bottom-up with consistency', 'abl_consistency'), ('A6: LAC score', 'abl_lac'),
       ('PC-FCP (full method)', 'pcfcp')]
MAIN = [(C10, '1.0'), (C100, '1.0'), (TX, '4.0')]
out.append(table('tab:app_sd_ablation', 'Component ablations (Table~\\ref{tab:ablation}) with standard deviations '
                 'over 3 seeds.', 'Variant & CIFAR-10-LT & CIFAR-100-LT & 20 Newsgroups',
                 [lab + ' & ' + ' & '.join(cell(c, m, e) for c, e in MAIN) for lab, m in ABL], 'l|r|r|r'))

BINS = [('Cubic, $B = 5$', 'pcfcp_p3_B5'), ('Cubic, $B = 8$ (PC-FCP)', 'pcfcp'), ('Cubic, $B = 12$', 'pcfcp_p3_B12'),
        ('Cubic, $B = 20$', 'pcfcp_p3_B20'), ('Cubic, $B = 50$', 'pcfcp_p3_B50'), ('Cubic, $B = 100$', 'pcfcp_p3_B100'),
        ('Quadratic, $B = 8$', 'pcfcp_p2_B8'), ('Quadratic, $B = 100$', 'pcfcp_p2_B100'),
        ('Equal width, $B = 50$', 'pcfcp_uniform50')]
out.append(table('tab:app_sd_bins', 'Histogram resolution (Table~\\ref{tab:bins}) with standard deviations over '
                 '3 seeds.', 'Bins & CIFAR-10-LT & CIFAR-100-LT',
                 [lab + ' & ' + ' & '.join(cell(c, m, e) for c, e in MAIN[:2]) for lab, m in BINS], 'l|r|r'))

rows = []
for a in (0.05, 0.1, 0.2, 0.3, 0.35):
    rows.append(f'{round(100 * (1 - a))} & ' + ' & '.join(
        [cell(C10, 'mondrian', 'inf', a), cell(C10, 'fedcp2qq_marg', '1.0', a), cell(C10, 'pcfcp', '1.0', a)]))
out.append(table('tab:app_sd_alpha', 'Target levels on CIFAR-10-LT (Table~\\ref{tab:alpha}) with standard '
                 'deviations over 3 seeds: worst-class coverage / set size.',
                 'Target (percent) & Centralized Mondrian$^\\dagger$ & FedCP2-QQ & PC-FCP', rows, 'l|r|r|r'))

IIDR = [('Centralized Mondrian$^\\dagger$', 'mondrian', 'inf'), ('DP-FedCP', 'dpfedcp_marg', '1.0'),
        ('FedCP2-QQ', 'fedcp2qq_marg', '1.0'), ('FedCP-QQ, per class, no noise', 'fedcpqq_pc', 'inf'),
        ('PC-FCP', 'pcfcp', '1.0')]
out.append(table('tab:app_sd_iid', 'Non-IID against near-IID partitions (Table~\\ref{tab:iid}) with standard '
                 'deviations over 3 seeds.',
                 'Method & Non-IID & Near-IID', [lab + ' & ' + cell(C10, m, e) + ' & ' + cell(IID, m, e)
                                                 for lab, m, e in IIDR], 'l|r|r'))

NOISE = [('DP-FedCP', 'mapped $\\varepsilon = 1$', 'dpfedcp_marg', '1.0'),
         ('DP-FedCP', '$\\sigma_g = 100$', 'dpfedcp_marg_sg100', 'sg100'),
         ('DP-FedCP', 'none', 'dpfedcp_marg_nonoise', 'inf'),
         ('DP-FedCP, per class', 'mapped $\\varepsilon = 1$', 'dpfedcp_pc', '1.0'),
         ('DP-FedCP, per class', 'none', 'dpfedcp_pc_nonoise', 'inf'),
         ('FedCP2-QQ', '$\\varepsilon = 1$', 'fedcp2qq_marg', '1.0'), ('FedCP-QQ', 'none', 'fedcpqq_marg', 'inf'),
         ('FedCP2-QQ, per class', '$\\varepsilon = 1$', 'fedcp2qq_pc', '1.0'),
         ('FedCP-QQ, per class', 'none', 'fedcpqq_pc', 'inf'),
         ('PC-FCP', '$\\varepsilon = 1$', 'pcfcp', '1.0'), ('PC-FCP', 'none', 'pcfcp_noise_free', 'inf')]
out.append(table('tab:app_sd_noise', 'Baselines with and without their privacy noise on CIFAR-10-LT '
                 '(Table~\\ref{tab:noise_elim}) with standard deviations over 3 seeds: worst-class coverage / set '
                 'size.', 'Method & Noise & Worst class / Set size',
                 [lab + ' & ' + nz + ' & ' + cell(C10, m, e) for lab, nz, m, e in NOISE], 'll|r'))

TAB = os.environ.get('PCFCP_TABLES', 'tables')
os.makedirs(TAB, exist_ok=True)
open(f'{TAB}/appendix_sd_small.tex', 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out)[:1500])
