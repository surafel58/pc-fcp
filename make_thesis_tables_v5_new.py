"""New Chapter 4 tables for the post-defense thesis (2026-11-05), written as table bodies to
tables/thesis_v5/ like make_thesis_tables_v5.py:
  significance.tex  B5: worst-class difference PC-FCP minus FedCP2-QQ per budget, Holm-adjusted p,
                    over the 10 training seeds and over the calibration/test re-splits, with the trend
  noise_grid.tex    B6: DP-FedCP worst-class and marginal coverage across noise levels sigma_g
  bounds.tex        objective 5: per-class coverage averaged over seeds and noise draws against the
                    lower bound of Theorem thm:coverage computed from each class's own pooled count
  text_resplit.tex  rows for Table tab:h2h_text: the 20 Newsgroups re-splits
It also prints a summary for the prose."""
import glob, math, os, re
import numpy as np, pandas as pd

OUT = 'tables/thesis_v5'
EPS = ('1.0', '2.0', '3.0', '4.0', '8.0')
NAMES = {'cifar10': 'CIFAR-10-LT', 'cifar100': 'CIFAR-100-LT', 'text': '20 Newsgroups'}
ALPHA, B = 0.1, 8
rng = np.random.default_rng(20261105)


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print('wrote', name)


def pfmt(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def sgn(s):
    """typeset a signed number string with a math minus"""
    return s.replace('-', '$-$', 1) if s.startswith('-') else s


def trend(md, ds_name):
    """slope and p of the worst-class trend vs FedCP2-QQ (marginal) for one dataset of a SIGNIFICANCE md."""
    sec = md[md.index('## ' + ds_name):]
    sec = sec[sec.index('### Worst-class coverage (pp): PC-FCP vs FedCP2-QQ\n'):]
    m = re.search(r'slope of the difference per doubling of eps = ([+-][0-9.]+).*?t-test p = ([<0-9.]+)', sec, re.S)
    return m.group(1), m.group(2).rstrip('.')


# ------------------------------------------------------------- significance (B5)
L = ['Dataset & Unit & ' + ' & '.join(f'$\\varepsilon = {e[0]}$' for e in EPS) + ' & Trend \\\\', '\\midrule']
src = [('results/v5/SIGNIFICANCE_10seed', 'seeds'), ('results/v4_resplit/SIGNIFICANCE_resplit', 're-splits')]
for k, ds in enumerate(('cifar10', 'cifar100', 'text')):
    for j, (base, unit) in enumerate(src):
        d = pd.read_csv(base + '.csv')
        d = d[(d.dataset == ds) & (d.comparator == 'fedcp2qq_marg') & (d.metric == 'worst_cov')]
        md = open(base + '.md', encoding='utf-8').read()
        cells = []
        for e in EPS:
            r = d[np.isclose(d.eps, float(e))].iloc[0]
            cells.append(f'{sgn(format(r["diff"], "+.1f"))} [{pfmt(r.holm_p)}]')
        sl, sp = trend(md, NAMES[ds])
        name = NAMES[ds] + (' (date split)' if ds == 'text' and unit == 'seeds' else '')
        L.append((name if j == 0 else ('' if ds != 'text' else '')) + f' & {unit} & ' + ' & '.join(cells)
                 + f' & {sgn(sl)} [{sp}] \\\\')
    if k < 2:
        L.append('\\midrule')
write('significance.tex', L)


# ------------------------------------------------------------- DP-FedCP noise grid (B6)
def grid(cfg):
    fs = glob.glob(f'results/v5/eval/{cfg}_seed*__dpfedcp*_r*.csv')
    d = pd.concat([pd.read_csv(f, dtype={'eps': str}).assign(seed=int(re.search(r'_seed(\d+)__', f).group(1))) for f in fs])
    d = d[d.method.str.startswith('dpfedcp')]
    d['form'] = np.where(d.method.str.contains('_pc'), 'pc', 'marg')

    def sg(r):
        if r.method.endswith('nonoise'):
            return 0.0
        s = re.search(r'_sg(\d+)$', r.method)
        return float(s.group(1)) if s else float(r.sigma_g)
    d['sg'] = d.apply(sg, axis=1)
    d['epsmap'] = np.where(d.method.isin(['dpfedcp_marg', 'dpfedcp_pc']), d.eps, '')
    per_seed = d.groupby(['form', 'sg', 'seed'])[['worst_cov', 'marginal_lt', 'set_size']].mean()
    return per_seed.groupby(['form', 'sg']).mean(), d[d.epsmap != ''].groupby('sg').epsmap.first()


G = {ds: grid(f'{ds}_K10_dir0.5_{"b" if ds == "text" else "la"}') for ds in ('cifar10', 'cifar100', 'text')}
sgs = sorted(G['cifar10'][0].loc['marg'].index)
emap = G['cifar10'][1]
L = ['$\\sigma_g$ & $\\varepsilon$ & ' + ' & '.join(f'\\multicolumn{{3}}{{c{"|" if i < 2 else ""}}}{{{NAMES[ds]}}}'
                                                     for i, ds in enumerate(('cifar10', 'cifar100', 'text'))) + ' \\\\',
     ' & & ' + ' & '.join(['Worst & Worst (pc) & Avg.'] * 3) + ' \\\\', '\\midrule']
for s in sgs:
    e = emap.get(s, '')
    if s == 0:
        e = 'none'
    elif e == '':
        e = {10.0: '$\\approx 31$', 30.0: '$\\approx 10$', 60.0: '$\\approx 5$', 100.0: '$\\approx 3$'}.get(s, '')
    else:
        e = f'{float(e):g}'
    cells = []
    for ds in ('cifar10', 'cifar100', 'text'):
        mg, pc = G[ds][0].loc['marg'].loc[s], G[ds][0].loc['pc'].loc[s]
        cells.append(f'{100 * mg.worst_cov:.1f} & {100 * pc.worst_cov:.1f} & {100 * mg.marginal_lt:.1f}')
    L.append(f'{("0" if s == 0 else f"{s:.1f}" if s % 1 else f"{s:.0f}")} & {e} & ' + ' & '.join(cells) + ' \\\\')
write('noise_grid.tex', L)


# ------------------------------------------------------------- measured coverage vs the bound (objective 5)
ED_cache = {}


def ED(eps, f, R=400000):
    key = (eps, round(f, 9))
    if key not in ED_cache:
        Z = rng.laplace(0, 1 / eps, (R, B))
        N = np.cumsum(Z[:, ::-1], 1)[:, ::-1]
        X = np.ceil(np.floor(f + ALPHA * N[:, :1]) - N[:, 1:])
        ED_cache[key] = np.maximum(0, X.max(1)).mean()
    return ED_cache[key]


def bound(n, eps):
    k0 = math.floor(ALPHA * (n + 1) + 1e-12)
    return (n + 1 - k0 - ED(eps, ALPHA * (n + 1) - k0)) / (n + 1)


def check(evdir, cachedir, prefix):
    """per (eps, class): mean coverage over seeds/re-splits and draws vs mean bound."""
    rows = []
    for f in sorted(glob.glob(f'{evdir}/{prefix}*__pcfcp.csv')):
        tag = os.path.basename(f).split('__')[0]
        if '_cal' in tag or '_if100' in tag:
            continue
        n_c = np.bincount(np.load(f'{cachedir}/{tag}.npz')['cal_labels'])
        d = pd.read_csv(f, dtype={'eps': str})
        d = d[np.isclose(d.alpha, ALPHA)]
        for e in EPS:
            x = d[d.eps == e]
            cov = np.mean([np.array(s.split(), float) for s in x.per_class_cov], 0)
            for c in range(len(n_c)):
                rows.append(dict(eps=e, c=c, n=n_c[c], cov=cov[c], bound=bound(n_c[c], float(e))))
    r = pd.DataFrame(rows).groupby(['eps', 'c']).mean()
    r['slack'] = r['cov'] - r['bound']
    return r


SETS = [('CIFAR-10-LT', 'seeds', 'results/v5/eval', 'results/v5/cache', 'cifar10_K10_dir0.5_la_seed'),
        ('CIFAR-10-LT', 're-splits', 'results/v4_resplit/eval', 'results/v4_resplit/cache', 'cifar10_K10_dir0.5_la_rs_seed'),
        ('CIFAR-100-LT', 'seeds', 'results/v5/eval', 'results/v5/cache', 'cifar100_K10_dir0.5_la_seed'),
        ('CIFAR-100-LT', 're-splits', 'results/v4_resplit/eval', 'results/v4_resplit/cache', 'cifar100_K10_dir0.5_la_rs_seed'),
        ('20 Newsgroups (date split)', 'seeds', 'results/v5/eval', 'results/v5/cache', 'text_K10_dir0.5_b_seed'),
        ('20 Newsgroups', 're-splits', 'results/v4_resplit/eval', 'results/v4_resplit/cache', 'text_K10_dir0.5_b_rs_seed')]
L = ['Data & Unit & ' + ' & '.join(f'$\\varepsilon = {e[0]}$' for e in EPS) + ' \\\\', '\\midrule']
summary = []
for name, unit, ev, ca, pre in SETS:
    r = check(ev, ca, pre)
    cells = []
    for e in EPS:
        x = r.loc[e]
        cells.append(f'{int((x.slack < 0).sum())}/{len(x)} ({100 * x.slack.min():+.1f})')
        summary.append((name, unit, e, int((x.slack < 0).sum()), len(x), 100 * x.slack.min(),
                        int(x.n.idxmin()), float(x.loc[x.n.idxmin(), 'cov']), float(x.loc[x.n.idxmin(), 'bound'])))
    L.append(f'{name} & {unit} & ' + ' & '.join(cells) + ' \\\\')
write('bounds.tex', L)
print('\nper-class check: classes below the bound / total (min slack pp); rarest class coverage vs bound')
for s in summary:
    print(f'  {s[0]:28s} {s[1]:9s} eps={s[2]}: {s[3]}/{s[4]} below, min slack {s[5]:+.1f} pp; rarest class {s[6]}: '
          f'coverage {100 * s[7]:.1f} vs bound {100 * s[8]:.1f}')


# ------------------------------------------------------------- 20 Newsgroups re-split rows for tab:h2h_text
def resplit_rows():
    out = []
    for name, m, e in [('Mondrian (central.)$^\\dagger$, re-splits', 'mondrian', 'inf'),
                       ('FedCP2-QQ, re-splits', 'fedcp2qq_marg', '4.0'),
                       ('\\textbf{PC-FCP (ours), re-splits}', 'pcfcp', '4.0')]:
        per_seed = {}
        for f in sorted(glob.glob('results/v4_resplit/eval/text_K10_dir0.5_b_rs_seed*__*.csv')):
            tag = os.path.basename(f).split('__')[0]
            d = pd.read_csv(f, dtype={'eps': str})
            d = d[(d.method == m) & (d.eps == e) & np.isclose(d.alpha, ALPHA)]
            if d.empty:
                continue
            prev = np.bincount(np.load(f'results/v4_resplit/cache/{tag}.npz')['cal_labels'])
            prev = prev / prev.sum()
            seed = int(tag.split('seed')[-1]) // 100
            v = dict(macro=100 * np.mean([np.array(s.split(), float).mean() for s in d.per_class_cov]),
                     lt=100 * d.marginal_lt.mean(), tail=100 * d.tail_cov.mean(), worst=100 * d.worst_cov.mean(),
                     under=d.undercov.mean(), size=d.set_size.mean(),
                     sizelt=np.mean([np.array(s.split(), float) @ prev for s in d.per_class_size]),
                     always=d.n_always.mean())
            per_seed.setdefault(seed, []).append(v)
        S = pd.DataFrame([pd.DataFrame(v).mean() for v in per_seed.values()])
        m_, s_ = S.mean(), S.std(ddof=1)
        out.append(f'{name} & {m_["macro"]:.1f} & {m_["lt"]:.1f} & {m_["tail"]:.1f} & {m_["worst"]:.1f} $\\pm$ '
                   f'{s_["worst"]:.1f} & {m_["under"]:.2f} $\\pm$ {s_["under"]:.2f} & {m_["size"]:.2f} $\\pm$ '
                   f'{s_["size"]:.2f} & {m_["sizelt"]:.2f} & {m_["always"]:.1f} \\\\')
    return out


write('text_resplit.tex', resplit_rows())
