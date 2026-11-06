"""Test-score clipping fix (2026-11-04): recompute, with the fixed v3_eval.py, only the
dpfedcp_pc_nonoise rows whose per-class threshold prints as 1.00000. DP-FedCP thresholds are
continuous (a weighted quantile of calibration scores), so a printed 1.00000 may lie just below or
just above 1 and the row cannot be corrected from the stored values; the threshold is recomputed with
the same function and noise stream as in run_group, checked against the stored threshold string, and
the row's metrics are re-evaluated on the clipped test scores. Other rows of the file are unchanged.
Usage: python results/batch/clip_dpfedcp_rows.py [workers]"""
import glob, os, re, sys
import multiprocessing as mp
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import v3_eval as V
import dp_fcp_gate as g

E = 'results/v5/eval'
METHOD = 'dpfedcp_pc_nonoise'


def todo():
    out = []
    for f in sorted(glob.glob(f'{E}/*__dpfedcp_r*.csv')):
        d = pd.read_csv(f, usecols=['method', 'per_class_thr'])
        if any(m == METHOD and '1.00000' in s.split() for m, s in zip(d.method, d.per_class_thr)):
            out.append(f)
    return out


def fix(f):
    name = os.path.basename(f)
    tag, grp = name[:-4].split('__')
    r = int(grp.split('_r')[1])
    m = re.search(r'_cal(\d+)_', tag)
    cal_frac = int(m.group(1)) / 100 if m else None
    base = re.sub(r'_cal\d+', '', tag)
    D = V.Data(f'results/v5/cache/{base}.npz', cal_frac)
    assert D.tag == tag, (D.tag, tag)
    D.alpha = 0.1
    g.N_CLASSES = D.C
    cs, ta = D.score('aps')                      # ta clipped to [0, 1] by the fixed code
    pc = D.per_client(cs)
    thr = V.dpfedcp_perclass(D, pc, 0.1, V.rs(D, METHOD, 'inf', 0.1, r), 0.0)
    new = V.evaluate(D, thr, ta)
    df = pd.read_csv(f)
    idx = df.index[df.method == METHOD]
    assert len(idx) == 1, (f, len(idx))
    i = idx[0]
    same_thr = df.at[i, 'per_class_thr'] == new['per_class_thr']
    changed = {k: (df.at[i, k], v) for k, v in new.items() if k in df.columns and str(df.at[i, k]) != str(v)}
    for k, v in new.items():
        if k in df.columns:
            df.at[i, k] = v
    df.to_csv(f, index=False)
    return name, same_thr, {k: changed[k] for k in ('set_size', 'worst_cov', 'marginal_bal') if k in changed}


if __name__ == '__main__':
    files = todo()
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f'{len(files)} files to patch on {w} workers', flush=True)
    with mp.Pool(w) as pool:
        for name, same_thr, ch in pool.imap_unordered(fix, files):
            print(f'  {name}: threshold reproduced = {same_thr}; changed {ch}', flush=True)
