"""Chapter 4 figures from the v3 evaluation (results/v3/eval). Classes are categorical, so
every figure uses markers with error bars (standard deviation over the three seeds), not
connecting lines."""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13, 'axes.labelsize': 13, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11, 'legend.fontsize': 10.5, 'lines.markersize': 7})

EV = os.environ.get('PCFCP_EVAL', 'results/v3/eval')      # 2026-10-20 batch: PCFCP_EVAL=results/v4/eval
CACHE = os.environ.get('PCFCP_CACHE', 'results/v3/cache')
FIG = os.environ.get('PCFCP_FIG', 'figures')
STY = {'mondrian': ('Centralized Mondrian (non-private)', 'k', 'o'),
       'dpfedcp_marg': ('DP-FedCP', 'tab:red', 'v'),
       'fedcp2qq_marg': ('FedCP2-QQ, marginal', 'tab:orange', 's'),
       'fedcp2qq_pc_fallback': ('FedCP2-QQ, per class + fallback', 'tab:purple', 'D'),
       'pcfcp': ('PC-FCP (ours)', 'tab:blue', '^'),
       'clustered_ding': ('Clustered CP (non-private)', 'tab:gray', 'P'),
       'abl_A2_local_noise': ('A2: per-client noise', 'tab:gray', 's')}


def tag_files(cfg, ev=None, pattern='__*.csv'):
    """Evaluation files of one configuration: tags are <cfg>[_la|_b]_seed<N>, so the main
    runs of the 2026-10-20 batch match and the _cal20/_if100 variants do not."""
    import re
    rx = re.compile(r'^' + re.escape(cfg) + r'(_la|_b)?_seed\d+$')
    out = []
    for f in glob.glob(os.path.join(ev or EV, cfg + '*' + pattern)):
        tag = os.path.basename(f).split('__')[0]
        if rx.match(tag):
            out.append(f)
    return sorted(out)


def load(cfg_glob, ev=None):
    cfg = cfg_glob.replace('_seed*', '')
    fs = tag_files(cfg, ev)
    d = pd.concat([pd.read_csv(f, dtype={'eps': str}) for f in fs], ignore_index=True)
    return d[(d.alpha - 0.1).abs() < 1e-9]


def per_class(d, method, eps):
    x = d[(d.method == method) & (d.eps == eps)]
    seeds = []
    for _, y in x.groupby('seed'):
        seeds.append(100 * np.mean([np.array(s.split(), float) for s in y.per_class_cov], 0))
    return np.array(seeds)


def counts(cfg):
    import re
    rx = re.compile(r'^' + re.escape(cfg) + r'(_la|_b)?_seed\d+\.npz$')
    p = sorted(f for f in glob.glob(os.path.join(CACHE, f'{cfg}*_seed*.npz')) if rx.match(os.path.basename(f)))[0]
    return np.bincount(np.load(p)['cal_labels'])


def fig_perclass(cfg, eps, out, deciles=False):
    d = load(cfg + '_seed*')
    n = counts(cfg)
    methods = ['mondrian', 'clustered_ding', 'dpfedcp_marg', 'fedcp2qq_marg', 'pcfcp']
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    k = len(methods)
    for i, m in enumerate(methods):
        P = per_class(d, m, 'inf' if m in ('mondrian', 'clustered_ding') else eps)
        if P.size == 0:
            continue
        if deciles:
            P = P.reshape(P.shape[0], 10, -1).mean(2)
        x = np.arange(P.shape[1]) + (i - (k - 1) / 2) * 0.13
        lab, col, mk = STY[m]
        ax.errorbar(x, P.mean(0), yerr=P.std(0, ddof=1) if P.shape[0] > 1 else None, fmt=mk, color=col,
                    label=lab, ms=6.5, capsize=2.5, lw=1.1)
    ax.axhline(90, color='k', ls='--', lw=0.8)
    if deciles:
        nd = n.reshape(10, -1).mean(1)
        ax.set_xticks(range(10)); ax.set_xticklabels([f'{i + 1}\n({v:.0f})' for i, v in enumerate(nd)])
        ax.set_xlabel('Frequency decile, most frequent first (mean calibration count per class)')
    else:
        ax.set_xticks(range(len(n)))
        ax.set_xticklabels([f'{c}\n{v}' if len(n) > 10 else f'{c}\n({v})' for c, v in enumerate(n)],
                           fontsize=8.5 if len(n) > 10 else None)
        ax.set_xlabel('Class, most frequent first (calibration count)')
    ax.set_ylabel('Coverage (%)'); ax.set_ylim(-3, 103)
    ax.legend(ncol=3, loc='lower left', framealpha=0.95)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, out), dpi=200); plt.close(fig)


def seed_metric(d, method, eps, col):
    x = d[(d.method == method) & (d.eps == eps)]
    return x.groupby('seed')[col].mean()


def fig_eps(out):
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.0), sharey=True)
    for ax, (cfg, name) in zip(axes, [('cifar10_K10_dir0.5', 'CIFAR-10-LT'), ('cifar100_K10_dir0.5', 'CIFAR-100-LT'),
                                      ('text_K10_dir0.5', '20 Newsgroups')]):
        d = load(cfg + '_seed*')
        E = ['1.0', '2.0', '3.0', '4.0', '8.0']
        for j, m in enumerate(['pcfcp', 'fedcp2qq_marg', 'dpfedcp_marg']):
            mu, sd = [], []
            for e in E:
                v = 100 * seed_metric(d, m, e, 'worst_cov')
                mu.append(v.mean() if len(v) else np.nan); sd.append(v.std(ddof=1) if len(v) > 1 else 0)
            lab, col, mk = STY[m]
            ax.errorbar(np.arange(5) + (j - 1) * 0.12, mu, yerr=sd, fmt=mk, color=col, label=lab, ms=7, capsize=2.5)
        ref = 100 * seed_metric(d, 'mondrian', 'inf', 'worst_cov').mean()
        ax.axhline(ref, color='k', ls=':', lw=1, label='Centralized Mondrian (non-private)')
        ax.set_xticks(range(5)); ax.set_xticklabels(['1', '2', '3', '4', '8'])
        ax.set_xlabel('Privacy budget $\\varepsilon$'); ax.set_title(name, fontsize=10)
    axes[0].set_ylabel('Worst-class coverage (%)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.3)
    fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig(os.path.join(FIG, out), dpi=200); plt.close(fig)


def fig_bins(out):
    d = load('cifar10_K10_dir0.5_seed*')
    d2 = load('cifar100_K10_dir0.5_seed*')
    lay = [('p3_B5', 'cubic 5'), ('pcfcp', 'cubic 8'), ('p3_B12', 'cubic 12'), ('p3_B20', 'cubic 20'),
           ('p3_B50', 'cubic 50'), ('p3_B100', 'cubic 100'), ('p2_B8', 'quadr. 8'), ('p2_B100', 'quadr. 100'),
           ('uniform50', 'equal 50')]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6))
    for j, (dd, name, col) in enumerate([(d, 'CIFAR-10-LT', 'tab:blue'), (d2, 'CIFAR-100-LT', 'tab:green')]):
        for k, e in enumerate(['1.0', '4.0']):
            w, s, ws, ss = [], [], [], []
            for m, _ in lay:
                mm = m if m == 'pcfcp' else f'pcfcp_{m}'
                v = 100 * seed_metric(dd, mm, e, 'worst_cov'); z = seed_metric(dd, mm, e, 'set_size')
                w.append(v.mean()); ws.append(v.std(ddof=1)); s.append(z.mean()); ss.append(z.std(ddof=1))
            x = np.arange(len(lay)) + (2 * j + k - 1.5) * 0.12
            mk = 'o' if k == 0 else 's'
            axes[0].errorbar(x, w, yerr=ws, fmt=mk, color=col, alpha=1 if k == 0 else 0.55, ms=5, capsize=2,
                             label=f'{name}, $\\varepsilon$={e[0]}')
            axes[1].errorbar(x, np.array(s) / s[1], yerr=np.array(ss) / s[1], fmt=mk, color=col,
                             alpha=1 if k == 0 else 0.55, ms=5, capsize=2, label=f'{name}, $\\varepsilon$={e[0]}')
    for ax in axes:
        ax.set_xticks(range(len(lay))); ax.set_xticklabels([l for _, l in lay], rotation=35, ha='right')
    axes[0].set_ylabel('Worst-class coverage (%)'); axes[1].set_ylabel('Set size relative to cubic 8')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(FIG, out), dpi=200); plt.close(fig)


def fig_clients(out):
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for j, m in enumerate(['pcfcp', 'abl_A2_local_noise']):
        mu, sd = [], []
        for K in (10, 50, 100):
            d = load(f'cifar100_K{K}_dir0.5_seed*')
            v = 100 * seed_metric(d, m, '1.0', 'worst_cov')
            mu.append(v.mean()); sd.append(v.std(ddof=1))
        lab, col, mk = STY[m]
        ax.errorbar(np.arange(3) + (j - 0.5) * 0.1, mu, yerr=sd, fmt=mk, color=col, label=lab, ms=10, capsize=4)
    ax.axhline(90, color='k', ls='--', lw=0.8)
    ax.set_xticks(range(3)); ax.set_xticklabels(['10', '50', '100']); ax.set_xlabel('Number of clients $K$')
    ax.set_ylabel('Worst-class coverage (%)'); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, out), dpi=200); plt.close(fig)


def fig_box(out, ev_r100=None):
    """Natnael Gap 15: distribution over 100 noise draws x 3 seeds of the worst-class
    coverage and of the per-class coverage of the two rarest classes, CIFAR-10-LT eps = 1."""
    ev = ev_r100 or EV.replace('eval', 'eval_r100')
    d = load('cifar10_K10_dir0.5_seed*', ev)
    meth = [('pcfcp', '1.0'), ('fedcp2qq_marg', '1.0'), ('abl_A2_local_noise', '1.0'), ('pcfcp_margin', '1.0')]
    names = {'pcfcp': 'PC-FCP', 'fedcp2qq_marg': 'FedCP2-QQ', 'abl_A2_local_noise': 'A2: per-client noise',
             'pcfcp_margin': 'PC-FCP, margin'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [('worst_cov', 'Worst-class coverage (%)', None), ('per_class_cov', 'Class 8 coverage (%)', 8),
              ('per_class_cov', 'Class 9 coverage (%)', 9)]
    for ax, (col, lab, cls) in zip(axes, panels):
        data, labels = [], []
        for m, e in meth:
            x = d[(d.method == m) & (d.eps == e)]
            if x.empty:
                continue
            v = 100 * x[col].values if cls is None else 100 * np.array([float(s.split()[cls]) for s in x[col]])
            data.append(v); labels.append(names[m])
        ax.boxplot(data, tick_labels=labels, widths=0.55, showfliers=True, flierprops=dict(markersize=3))
        ax.axhline(90, color='k', ls='--', lw=0.8)
        ax.set_ylabel(lab); ax.tick_params(axis='x', labelrotation=20)
        ax.grid(axis='y', lw=0.3, color='#cccccc')
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, out), dpi=200); plt.close(fig)


if __name__ == '__main__':
    fig_perclass('cifar10_K10_dir0.5', '1.0', 'fig_perclass_c10_v3.png')
    fig_perclass('cifar100_K10_dir0.5', '1.0', 'fig_decile_c100_v3.png', deciles=True)
    fig_perclass('text_K10_dir0.5', '4.0', 'fig_perclass_text_v3.png')
    fig_eps('fig_eps_v3.png')
    fig_bins('fig_bins_v3.png')
    fig_clients('fig_clients_c100_v3.png')
    if os.path.isdir(EV.replace('eval', 'eval_r100')):
        fig_box('fig_box_c10_v3.png')
    print('figures written')
