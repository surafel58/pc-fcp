"""Test-score clipping fix (2026-11-04): exact correction of the stored evaluation rows of the
grid-based methods. With test scores clipped to [0, 1], a class whose threshold is exactly the top
edge 1 admits its label for every test example; before the fix the labels whose unclipped APS score
exceeded 1 (float rounding) were excluded. Thresholds are unchanged, so every metric of evaluate()
changes by an exactly computable amount. For each corrected class the stored per-class coverage is
checked against 1 - P(score > 1), which confirms that the threshold was exactly 1.

Skipped, because they are re-run or recomputed exactly elsewhere (clip_rerun.sh,
clip_dpfedcp_rows.py): ablations files, DP-FedCP files, and the FedCP-QQ qq/alpha files listed in
SKIP. A row with a printed threshold 1.00000 whose method is not grid-based raises an error.
Usage: python results/batch/clip_correct.py"""
import glob, os, re, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import v3_eval as V
import dp_fcp_gate as g

DIRS = [('results/v5/eval', 'results/v5/cache'), ('results/v5/eval_r100', 'results/v5/cache'),
        ('results/v4_resplit/eval', 'results/v4_resplit/cache')]
SKIP = {'cifar10_K10_dir0.5_la_cal20_seed42__qq.csv', 'text_K10_dir0.5_b_cal20_seed48__qq.csv',
        'text_K10_dir0.5_b_cal40_seed47__qq.csv', 'text_K10_dir0.5_b_cal40_seed51__qq.csv',
        'text_K10_dir0.5_b_seed42__alpha.csv', 'text_K10_dir0.5_b_seed47__alpha.csv',
        'text_K10_dir0.5_b_seed51__alpha.csv'}
GRID_GRIDS = ['p3_B5', 'p3_B12', 'p3_B20', 'p3_B50', 'p2_B8', 'p2_B100', 'uniform50']   # not p3_B100
GRID = {'pcfcp', 'pcfcp_noise_free', 'pcfcp_noise_free_nofloor', 'pcfcp_borrow', 'pcfcp_borrow_noise_free',
        'pcfcp_margin', 'abl_A8_tree', 'abl_A2_local_noise', 'abl_A3_marginal', 'abl_nofloor',
        'abl_bottomup', 'abl_consistency', 'fedcp2qq_pc', 'fedcp2qq_pc_fallback', 'fedcp2qq_marg',
        'fedcp2qq_pc_pfallback'} | {f'pcfcp_{n}' for n in GRID_GRIDS} | {f'pcfcp_{n}_noise_free' for n in GRID_GRIDS}

_data = {}
def data_for(tag, cache):
    if tag not in _data:
        m = re.search(r'_cal(\d+)_', tag)
        cal_frac = int(m.group(1)) / 100 if m else None
        base = re.sub(r'_cal\d+', '', tag)
        D = V.Data(f'{cache}/{base}.npz', cal_frac)
        assert D.tag == tag, (D.tag, tag)
        g.N_CLASSES = D.C
        raw = g.aps_score_all(D.test_probs, np.random.RandomState(9000 + D.seed))   # unclipped, as stored
        cs = g.aps_score_true(D.cal_probs, D.cal_labels, np.random.RandomState(2024 + D.seed))
        marg = g.fs_quantile(cs, 0.1)
        _data.clear()
        _data[tag] = (D, raw > 1.0, marg)
    return _data[tag]

def correct_row(r, D, over, marg):
    thr = r['per_class_thr'].split()
    T1 = [c for c, x in enumerate(thr) if x == '1.00000']
    if not T1:
        return False
    if r['method'] not in GRID:
        raise RuntimeError(f"non-grid method {r['method']} with a threshold printed as 1.00000 in {D.tag}")
    if r['method'] == 'fedcp2qq_pc_fallback':
        assert abs(marg - 1.0) > 1e-5, (D.tag, marg)
    y = D.test_labels; C = D.C; N = len(y)
    t = 1.0 - float(r['alpha'])
    cov = np.array([float(x) for x in r['per_class_cov'].split()])
    size_pc = np.array([float(x) for x in r['per_class_size'].split()])
    tail = D.tail; head = [c for c in range(C) if c not in tail]
    changed = False
    d_bal = d_lt = d_tail = d_head = d_gap = d_under = d_over = d_size = 0.0
    d_nunder = 0
    d_size_pc = np.zeros(C)
    for c in T1:
        dc = float(over[y == c, c].mean())
        cov_old = 1.0 - dc
        assert abs(cov[c] - cov_old) < 6e-5, (D.tag, r['method'], c, cov[c], cov_old)
        if over[:, c].any():
            changed = True
        d_bal += over[y == c, c].sum() / N
        d_lt += D.prev[c] * dc
        if c in tail: d_tail += dc / len(tail)
        else: d_head += dc / len(head)
        d_gap += abs(1.0 - t) - abs(cov_old - t)
        d_under += 0.0 - max(t - cov_old, 0.0)
        d_over += (1.0 - t) - max(cov_old - t, 0.0)
        d_nunder -= int(cov_old < t)
        d_size += float(over[:, c].mean())
        d_size_pc += np.array([over[y == k, c].mean() for k in range(C)])
        cov[c] = 1.0
    if not changed:
        return False
    r['marginal_bal'] = float(r['marginal_bal']) + d_bal
    r['marginal_lt'] = float(r['marginal_lt']) + d_lt
    r['tail_cov'] = float(r['tail_cov']) + d_tail
    r['head_cov'] = float(r['head_cov']) + d_head
    r['covgap'] = float(r['covgap']) + 100 * d_gap / C
    r['undercov'] = float(r['undercov']) + 100 * d_under / C
    r['overcov'] = float(r['overcov']) + 100 * d_over / C
    r['n_under'] = int(r['n_under']) + d_nunder
    r['set_size'] = float(r['set_size']) + d_size
    r['tail_size'] = float(r['tail_size']) + float(d_size_pc[tail].mean())
    if int(r['worst_class']) in T1:
        r['worst_cov'] = float(cov.min()); r['worst_class'] = int(cov.argmin())
    r['per_class_cov'] = ' '.join(f'{x:.4f}' for x in cov)
    r['per_class_size'] = ' '.join(f'{x:.3f}' for x in size_pc + d_size_pc)
    return True

if __name__ == '__main__':
    log = []
    for E, cache in DIRS:
        files = sorted(glob.glob(f'{E}/*.csv'))
        n_files = n_rows = 0
        for f in files:
            name = os.path.basename(f)
            grp = name[:-4].split('__')[1]
            if name in SKIP or grp == 'ablations' or grp.startswith('dpfedcp'):
                continue
            df = pd.read_csv(f, dtype={'per_class_thr': str, 'per_class_cov': str, 'per_class_size': str})
            if not df.per_class_thr.str.contains('1.00000', regex=False).any():
                continue
            D, over, marg = data_for(name.split('__')[0], cache)
            rows = df.to_dict('records')
            k = sum(correct_row(r, D, over, marg) for r in rows)
            if k:
                pd.DataFrame(rows, columns=df.columns).to_csv(f, index=False)
                n_files += 1; n_rows += k
        msg = f'{E}: corrected {n_rows} rows in {n_files} files'
        print(msg, flush=True); log.append(msg)
    open('results/v5/logs/clip_correct.log', 'a').write('\n'.join(log) + '\n')
