"""Figures for the IEEE TAI manuscript (paper/figures/), from the same evaluation results as
the thesis figures (results/v3/eval). Vector PDF at IEEE widths (3.5 in one column,
7.16 in full width), Times-like serif text at 7-8 pt, and methods told apart by marker
shape and fill as well as colour, so the figures stay readable in grayscale."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import make_figures_v3 as M  # noqa: E402  (data loading helpers only)

OUT = 'paper/figures'
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.markersize': 4.5,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})

# label, colour, marker, filled
STY = {'mondrian': ('Centralized Mondrian (non-private)', '#000000', 'o', True),
       'clustered_ding': ('Clustered CP (non-private)', '#7f7f7f', 'P', True),
       'dpfedcp_marg': ('DP-FedCP', '#c0392b', 'v', False),
       'fedcp2qq_marg': ('FedCP2-QQ', '#e67e22', 's', False),
       'pcfcp': ('PC-FCP (ours)', '#1f5fa8', '^', True)}


def mk(ax, x, y, e, m, dx=0.0):
    lab, col, mkr, filled = STY[m]
    ax.errorbar(np.asarray(x) + dx, y, yerr=e, fmt=mkr, color=col, mfc=col if filled else 'white',
                mec=col, mew=0.9, ms=4.5, capsize=1.5, elinewidth=0.7, label=lab)


def fig_eps():
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=True)
    E = ['1.0', '2.0', '3.0', '4.0', '8.0']
    for ax, (cfg, name) in zip(axes, [('cifar10_K10_dir0.5', 'CIFAR-10-LT'), ('cifar100_K10_dir0.5', 'CIFAR-100-LT'),
                                      ('text_K10_dir0.5', '20 Newsgroups')]):
        d = M.load(cfg + '_seed*')
        for j, m in enumerate(['pcfcp', 'fedcp2qq_marg', 'dpfedcp_marg']):
            mu, sd = [], []
            for e in E:
                v = 100 * M.seed_metric(d, m, e, 'worst_cov')
                mu.append(v.mean()); sd.append(v.std(ddof=1))
            mk(ax, np.arange(5), mu, sd, m, dx=(j - 1) * 0.13)
        ref = 100 * M.seed_metric(d, 'mondrian', 'inf', 'worst_cov').mean()
        ax.axhline(ref, color='black', ls=':', lw=0.8, label='Centralized Mondrian (non-private)')
        ax.set_xticks(range(5)); ax.set_xticklabels(['1', '2', '3', '4', '8'])
        ax.set_xlabel(r'Privacy budget $\varepsilon$'); ax.set_title(name)
        ax.set_ylim(-4, 104); ax.grid(axis='y', lw=0.3, color='#cccccc')
    axes[0].set_ylabel('Worst-class coverage (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, 'fig_eps.pdf')); plt.close(fig)


def fig_perclass():
    cfg = 'cifar10_K10_dir0.5'
    d = M.load(cfg + '_seed*')
    n = M.counts(cfg)
    methods = ['mondrian', 'clustered_ding', 'dpfedcp_marg', 'fedcp2qq_marg', 'pcfcp']
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for i, m in enumerate(methods):
        P = M.per_class(d, m, 'inf' if m in ('mondrian', 'clustered_ding') else '1.0')
        mk(ax, np.arange(P.shape[1]), P.mean(0), P.std(0, ddof=1), m, dx=(i - 2) * 0.15)
    ax.axhline(90, color='black', ls='--', lw=0.7)
    ax.set_xticks(range(len(n))); ax.set_xticklabels([f'{c}\n{v}' for c, v in enumerate(n)], fontsize=6.5)
    ax.set_xlabel('Class (calibration examples), most frequent first')
    ax.set_ylabel('Coverage (%)'); ax.set_ylim(-4, 104); ax.grid(axis='y', lw=0.3, color='#cccccc')
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02), fontsize=6.5)
    fig.tight_layout(rect=(0, 0.2, 1, 1))
    fig.savefig(os.path.join(OUT, 'fig_perclass_c10.pdf')); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    fig_eps()
    fig_perclass()
    print('paper figures written to', OUT)
