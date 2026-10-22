"""T4 sanity check of the DP-FedCP reimplementation (examiner comment T4, 2026-10-12).

DP-FedCP (Plassier et al., ICML 2023) promises marginal coverage for one designated
target agent under label shift. In the thesis run its noiseless marginal coverage is
61.3 percent on the balanced test set, so this script measures its coverage on the
distribution it actually targets (the target agent's label mix) and separates the
possible causes of any shortfall, on the CIFAR-10-LT scores cached by T5:

  dpfedcp_T200      : the reimplementation as used in the thesis (sigma_g = 0)
  exact_subsampled  : the same importance weights and the same multinomial
                      subsampling, but the exact weighted quantile instead of the
                      200-round federated gradient estimate (isolates optimisation)
  exact_full        : exact weighted quantile on all calibration points (no subsampling)

Coverage is reported on the balanced test set, on the pooled long-tailed mix and on
the target agent's label mix, averaged over --reps draws of the subsampling.
"""
import argparse
import os
import sys

import numpy as np

import dp_fcp_gate as g

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baselines', 'plassier'))
import dp_fedcp as P  # noqa: E402


def weights_from_counts(counts, target):
    """Importance weights of eq. (10), exactly as dp_fedcp_thresholds computes them
    (sigma_bar = 0): w_y = (M * M*_y) / (M* * M_y)."""
    M_hat = np.maximum(1.0, counts)
    M_y = M_hat.sum(0); M = M_y.sum(); M_star_y = M_hat[target]; M_star = M_star_y.sum()
    return (M * M_star_y) / (M_star * M_y)


def exact_weighted_thresholds(scores, labels, w, alpha):
    """Exact weighted (1-alpha) quantile of eq. (5), one per candidate label y
    (the delta_1 atom carries weight w_y)."""
    out = np.empty(len(w))
    for y in range(len(w)):
        vals = np.concatenate([scores, [1.0]])
        wts = np.concatenate([w[labels], [w[y]]])
        o = np.argsort(vals); cum = np.cumsum(wts[o]) / wts.sum()
        out[y] = vals[o][min(int(np.searchsorted(cum, 1 - alpha)), len(vals) - 1)]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', default='42,43,44')
    p.add_argument('--reps', type=int, default=10)
    args = p.parse_args()
    g.N_CLASSES = 10; g.TAIL_CLASSES = [7, 8, 9]; alpha = 0.1
    res = {}
    for seed in [int(s) for s in args.seeds.split(',')]:
        d = np.load(f'results/t5/scores_c10_seed{seed}.npz')
        cl, cs, cc = d['cal_labels'], d['cal_scores'], d['cal_client']
        tp, tl = d['test_probs'], d['test_labels']
        K = int(cc.max()) + 1
        per_client = [{c: cs[(cc == k) & (cl == c)] for c in range(10) if ((cc == k) & (cl == c)).any()}
                      for k in range(K)]
        counts = np.stack([np.bincount(cl[cc == k], minlength=10) for k in range(K)]).astype(float)
        target = int(np.argmax(counts[:, g.TAIL_CLASSES].sum(1)))
        mixes = {'balanced': np.full(10, 0.1),
                 'pooled_LT': np.bincount(cl, minlength=10) / len(cl),
                 'target_agent': counts[target] / counts[target].sum()}
        test_all = g.aps_score_all(tp, np.random.RandomState(9000 + seed))
        w = weights_from_counts(counts, target)
        agent_s = [np.concatenate([per_client[k][c] for c in sorted(per_client[k])]) for k in range(K)]
        agent_l = [np.concatenate([np.full(len(per_client[k][c]), c) for c in sorted(per_client[k])])
                   for k in range(K)]

        def cov_of(thr):
            inc = test_all <= np.asarray(thr)[None, :]
            cov = np.array([inc[tl == c, c].mean() for c in range(10)])
            return {m: float(cov @ v) for m, v in mixes.items()}, cov

        for rep in range(args.reps):
            rng = np.random.RandomState(1000 * seed + rep)
            thr_dp, tgt, _ = g.arm_plassier_dpfedcp(per_client, alpha, 1.0, rng, sigma_g_override=0.0)
            assert tgt == target
            # same multinomial subsampling as the reimplementation, exact quantile
            gen = np.random.default_rng(rep)
            N_i = np.array([len(s) for s in agent_s], float)
            keep = np.minimum(N_i, gen.multinomial(int(N_i.sum() // 2), N_i / N_i.sum())).astype(int)
            sub_s = np.concatenate([agent_s[k][:keep[k]] for k in range(K)])
            sub_l = np.concatenate([agent_l[k][:keep[k]] for k in range(K)])
            variants = {'dpfedcp_T200': [thr_dp[c] for c in range(10)],
                        'exact_subsampled': exact_weighted_thresholds(sub_s, sub_l, w, alpha),
                        'exact_full': exact_weighted_thresholds(np.concatenate(agent_s),
                                                                np.concatenate(agent_l), w, alpha)}
            for v, thr in variants.items():
                m, cov = cov_of(thr)
                r = res.setdefault((seed, v), {'thr': [], **{k: [] for k in mixes}, 'cov': []})
                r['thr'].append(np.mean(thr)); r['cov'].append(cov)
                for k in mixes:
                    r[k].append(m[k])
        print(f'[seed={seed}] target agent {target}, target class counts {counts[target].astype(int).tolist()}',
              flush=True)

    print('\nmean over reps; coverage in percent on three test mixes')
    print(f"{'seed':4s} {'variant':17s} {'mean thr':>8s} {'balanced':>9s} {'pooledLT':>9s} {'TARGET':>7s}")
    for (seed, v), r in sorted(res.items()):
        print(f"{seed:4d} {v:17s} {np.mean(r['thr']):8.4f} {100*np.mean(r['balanced']):9.1f} "
              f"{100*np.mean(r['pooled_LT']):9.1f} {100*np.mean(r['target_agent']):7.1f}")
    for v in ['dpfedcp_T200', 'exact_subsampled', 'exact_full']:
        t = [np.mean(r['target_agent']) for (s, vv), r in res.items() if vv == v]
        print(f'  {v:17s} target-agent coverage, mean of seeds: {100*np.mean(t):.1f}')


if __name__ == '__main__':
    main()
