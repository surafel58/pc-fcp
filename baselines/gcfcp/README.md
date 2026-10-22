# GC-FCP baseline: setup, patches, and results

**Paper:** H. Wen, O. Simeone, H. Xing, "Efficient Federated Conformal Prediction with Group-Conditional Guarantee," arXiv:2603.14198.
**Code:** https://github.com/HaifengWen/GC-FCP (cloned 2026-08-14).
**Why this baseline:** it is the closest published work to ours (federated, conditional coverage, coreset aggregation) and the only one that measures group-conditional coverage directly. It is **non-private**, which makes it an advantaged skyline rather than a matched competitor.

---

## 1. What we changed in their code, and why

All changes are in `our_patches.diff` (58 insertions, 3 modified lines). The patched file is kept as `realdata_rebuttal_v4_patched.py`. Every change is marked in-line in the source.

**(a) `--group_mode per_class` (new).** Their finest built-in grouping is two classes wide: `disjoint_pred` uses intervals `[(0,2), (2,4), (4,6), (6,8), (8,10)]`. To compare against a per-class method we need one group per class, so `per_class` uses width-1 intervals `[(0,1), (1,2), ..., (9,10)]`. This reuses their own `phi_from_intervals` and changes nothing else about the method.

> **Important caveat, stated in the code and required in any results table.** Their groups are defined on the **predicted** label `ŷ(x) = argmax f(x)`, which is a function of X. Their Theorem 5.1 conditions on `X ∈ G`. Ours conditions on the **true** label `Y = y`. Under class confusion these differ, so GC-FCP's "worst group" is not the same quantity as our "worst class" and must never be presented as if it were.

**(b) `--lt_if` (new).** Their pipeline draws a uniform random calibration set, so every class is well represented and no rare-class regime exists. `--lt_if F` subsamples the **calibration pool** by class: class `c` keeps a fraction `mu^c` of its points with `mu = F^(-1/(n_classes-1))`, the same exponential profile used in the long-tail literature and in our own harness.

The **test set is deliberately left balanced.** Coverage has to be measured on every class equally; imbalancing the test set would hide rare-class failures instead of exposing them.

Verified at `--lt_if 100`, per-class calibration counts are `[495, 295, 186, 109, 66, 38, 22, 14, 8, 5]`, so the rarest classes sit at or below the `1/alpha - 1 = 9` floor.

**(c) `trust_repo=True`.** Their `torch.hub.load` call blocks on an interactive y/N prompt, which cannot be answered in a non-interactive run. Same repository, same weights.

**Nothing in the GC-FCP estimator itself was modified.** The T-Digest coresets, the LP, and the aggregation are theirs, unchanged.

---

## 2. Environment notes

- Python 3.13, CPU only. No GPU needed: the CIFAR-10 model is a pretrained ResNet-56 from `torch.hub` (`chenyaofo/pytorch-cifar-models`) and the work is inference plus an LP.
- Extra installs beyond their `requirements.txt` that were missing locally: `torchvision`, `tdigest`, `tabulate`, `seaborn`.
- CIFAR-10 was reused from the FedCCFA repo (`data/cifar-10-batches-py`) rather than re-downloaded.
- Their smoke test (`python realdata_rebuttal_v4.py smoke`) passes, confirming the install is sound before any real run.

---

## 3. Results: balanced CIFAR-10, per-predicted-class groups

10 seeds, 10 clients, `n_cal = n_test = 5000`, `delta = 250`, alpha = 0.1. Raw rows in `../results/gcfcp/gcfcp_balanced_perclass.csv`.

| score | method | marginal | worst group | max gap | set size |
|---|---|---|---|---|---|
| APS | Centralized CP | 89.88 | 76.77 | 13.23 | 5.080 |
| APS | Centralized CondCP | 89.81 | 86.71 | 3.60 | 4.941 |
| APS | FedCP | 90.08 | 77.14 | 12.86 | 5.107 |
| APS | Naive GC-FCP | 91.51 | 88.61 | 3.90 | 5.203 |
| APS | **GC-FCP (delta=250)** | 91.49 | **88.65** | **3.91** | 5.200 |
| THR | Centralized CP | 90.25 | 78.73 | 11.27 | 0.930 |
| THR | Centralized CondCP | 90.46 | 87.04 | 3.82 | 0.948 |
| THR | FedCP | 90.38 | 79.01 | 10.99 | 0.932 |
| THR | Naive GC-FCP | 92.05 | 88.78 | 4.19 | 0.978 |
| THR | **GC-FCP (delta=250)** | 92.08 | **89.08** | **4.22** | 0.978 |

**Two readings.**

1. **GC-FCP does what their paper claims.** Marginal-only methods (CP, FedCP) reach ~90% marginal coverage while their worst group falls to 77 to 79%. GC-FCP lifts the worst group to about 89% for almost no extra set size. Reproduced independently here.

2. **Their own experiment reproduces our motivating claim.** FedCP reports 90.38% marginal coverage while its worst group gets 79.01%: a marginally valid federated method leaving a group 11 points short. That is our thesis, measured by a competing paper's code, on balanced data with no rare classes at all.

**What this run does not settle.** This is balanced CIFAR-10 with about 500 calibration points per class, far above the floor. GC-FCP has no rare class to struggle with here, so this is their best case, not a test of our regime. The long-tailed run (`--lt_if 100`) is the comparison that matters.

---

## 3b. Results: long-tailed CIFAR-10 (the comparison that matters)

10 seeds, same configuration, `--lt_if 100`. Raw rows in `../results/gcfcp/gcfcp_lt_if100_perclass.csv`.

| score | method | marginal | worst group | set size |
|---|---|---|---|---|
| APS | Centralized CP | 88.79 | 74.94 | 4.914 |
| APS | Centralized CondCP | 89.11 | 77.47 | 4.997 |
| APS | FedCP | 92.44 | 81.49 | 5.499 |
| APS | Naive GC-FCP | 100.00 | 100.00 | **10.000** |
| APS | **GC-FCP (delta=250)** | 100.00 | 100.00 | **10.000** |
| THR | Centralized CP | 90.15 | 78.42 | 0.929 |
| THR | Centralized CondCP | 93.44 | 84.28 | 1.450 |
| THR | FedCP | 94.20 | 87.65 | 1.004 |
| THR | Naive GC-FCP | 100.00 | 100.00 | **10.000** |
| THR | **GC-FCP (delta=250)** | 100.00 | 100.00 | **10.000** |

**GC-FCP becomes vacuous under severe imbalance.** A set size of 10.000 on a 10-class problem means every label is predicted for every input: 100% coverage carrying zero information. This is the degenerate all-label set, technically valid and useless.

### Ruling out a tuning artifact

A competitor going fully vacuous can indicate a misconfiguration rather than a real failure, so three checks were run before recording this.

**(i) Compression parameter `delta` is not the cause** (`gcfcp_lt_delta_sensitivity.csv`, 3 seeds, THR):

| method | marginal | worst group | set size |
|---|---|---|---|
| GC-FCP (delta=25) | 100.00 | 100.00 | 10.000 |
| GC-FCP (delta=50) | 100.00 | 100.00 | 10.000 |
| GC-FCP (delta=250) | 100.00 | 100.00 | 10.000 |
| Naive GC-FCP | 100.00 | 100.00 | 10.000 |

All compression levels are identical, and **Naive GC-FCP, which applies no coreset compression at all, is equally vacuous**. So the T-Digest sketch is not responsible.

**(ii) It is a graded degradation with a threshold, not a cliff at every imbalance** (`gcfcp_lt_if_sweep.csv`, THR, GC-FCP delta=250):

| imbalance factor | min per-class cal points | marginal | worst group | set size |
|---|---|---|---|---|
| 1 (balanced) | ~500 | 92.43 | 89.14 | 0.982 |
| 5 | 100 | 95.15 | 92.22 | 1.058 |
| 10 | ~50 | 97.30 | 93.36 | 1.193 |
| 50 | 10 | 100.00 | 100.00 | 10.000 |
| 100 | 5 | 100.00 | 100.00 | 10.000 |

Set size grows smoothly (0.98, 1.06, 1.19) while the rarest class stays above the finite-sample floor, then jumps to fully vacuous once it falls to about 10 points or fewer. The breaking point is bracketed between IF=10 (min 50 points, works) and IF=50 (min 10 points, vacuous).

**(iii) The mechanism is the documented finite-sample floor.** At alpha = 0.1 a class needs at least `1/alpha - 1 = 9` calibration points for a finite quantile; below that the threshold is infinite and every label is admitted (Ding et al., NeurIPS 2023). At IF=100 the per-class calibration counts are `[495, 295, 186, 109, 66, 38, 22, 14, 8, 5]`, so the two rarest classes are at or below the floor. This is the same mechanism our justification cites, observed in the closest competing method.

### What this establishes, and what it does not

**Establishes.** GC-FCP is strong where it was designed to operate and degrades gracefully with moderate imbalance, but under severe imbalance it returns all-label sets. Our method, at full DP cost, reaches 97.88% tail coverage at set size 5.72 on CIFAR-10-LT IF=100, while GC-FCP, which is non-private and therefore advantaged, returns set size 10.0 in the same regime.

**Does not establish.** This is not a claim that GC-FCP is a weak method, nor that its authors erred. Their paper makes no long-tail claim and evaluates on balanced benchmarks. The honest statement is that **their published configuration does not extend to the severe-imbalance regime, because per-group calibration data falls below the finite-sample floor**, which is a property of group-conditional calibration generally rather than a flaw specific to their estimator. A long-tail-aware variant of GC-FCP might well avoid this; none is published.

**Also note.** FedCP outperforms GC-FCP under long tails (THR worst group 87.65 versus vacuous) simply because a marginal method has no per-group sample requirement to violate. That is not evidence that FedCP is better at rare classes; its worst group is still 87.65 against a 90 target, and on balanced data it leaves an 11-point gap.

---

## 4. Reproduce

```bash
git clone https://github.com/HaifengWen/GC-FCP.git
cd GC-FCP
# apply our_patches.diff, or use realdata_rebuttal_v4_patched.py
pip install -r requirements.txt tdigest tabulate seaborn

# balanced baseline
python realdata_rebuttal_v4.py run --dataset cifar10 --clients 10 \
  --scores thr,aps --x_feature pred_label --group_mode per_class \
  --client_split label --n_cal 5000 --n_test 5000 --times 10 \
  --deltas 250 --methods cp,fedcp,condcp,naive_gcfcp,gcfcp \
  --lp_max_iter 50 --n_jobs 4 --out_dir results/perclass_main

# long-tailed calibration (imbalance factor 100, balanced test set)
python realdata_rebuttal_v4.py run --dataset cifar10 --clients 10 \
  --scores thr,aps --x_feature pred_label --group_mode per_class \
  --client_split label --n_cal 5000 --n_test 5000 --times 10 \
  --deltas 250 --methods cp,fedcp,condcp,naive_gcfcp,gcfcp \
  --lt_if 100 --lp_max_iter 50 --n_jobs 4 --out_dir results/lt_main
```
