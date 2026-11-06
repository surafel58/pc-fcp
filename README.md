# PC-FCP: Per-Class Federated Conformal Prediction under Differential Privacy

Code and experiment results for the MSc thesis *Per-Class Coverage for Rare Classes in
Federated Conformal Prediction under Differential Privacy* (Addis Ababa University, 2026) and the
journal manuscript of the same title.

PC-FCP calibrates one conformal threshold per class in a federation without moving any
calibration score. Each client counts its calibration scores into one histogram per class,
secure aggregation sums the histograms with Laplace noise added inside the sum, and the
server reads one threshold per class from the top of each released histogram, floored at
the pooled threshold. The release is example-level differentially private under a single
budget and equals the release of a central curator. Every class is covered at least at the
target minus a noise term that shrinks with its pooled count, and a margin variant reaches the
target when run at a stricter level.

## Layout

| Path | What it holds |
|---|---|
| `dp_fcp_gate.py` | APS and LAC scores, long-tail construction, Dirichlet partition, the histogram channel, read-outs, Laplace noise, and the FedCP-QQ, Mondrian and Clustered CP baselines |
| `dp_fcp_text.py` | the 20 Newsgroups pipeline (sentence embeddings and the MLP) |
| `step2_bins_study.py` | bin edges (`edges_of`) and the bin study |
| `t5_readout_ablation.py` | bottom-up read-out and the consistency projection (ablation A5) |
| `baselines/plassier/` | reimplementation of DP-FedCP and its validation on the paper's synthetic experiment |
| `baselines/gcfcp/` | patches used to check the GC-FCP code |
| `build_cifar_cache.py` | downloads CIFAR-10 and CIFAR-100 |
| `v3_train.py` | federated training (FedAvg, logit-adjusted loss) and caching of the classifier outputs |
| `v3_eval.py` | every calibration method, ablation and sweep, run on the cached outputs |
| `resplit_caches.py` | random calibration and test re-splits of the cached outputs |
| `significance_v4.py` | paired t-test, exact Wilcoxon signed-rank test, Holm adjustment over the budgets, and a trend test, with the training seed or the re-split as the unit |
| `analyze_results_v3.py` | the results report |
| `theory/` | simulations behind the coverage bounds (see below), with their outputs in `theory/out/` |
| `make_thesis_tables_v5.py`, `make_thesis_tables_v5_new.py` | thesis tables from the ten-seed run, including the significance tests, the DP-FedCP noise grid and the bounds against measured coverage |
| `make_paper_v2.py`, `make_tables_paper.py`, `make_figures_paper.py` | tables and figures of the journal version |
| `make_figures_v3.py`, `make_thesis_tables_v4.py`, `make_appendix_tables.py`, `make_appendix_sd_small.py` | figures and tables of the first version |
| `perclass_detail_v3.py`, `lt_setsize.py` | per-class thresholds and false-inclusion rates, set sizes under the long-tailed class mix |
| `verify_privacy.py`, `validate_clustered.py`, `t4_dpfedcp_check.py` | implementation checks |
| `launch_batch.sh`, `launch_seeds.sh`, `launch_seeds2.sh`, `results/batch/` | the training and evaluation batches |
| `results/v5/` | the results report, summary tables, accuracies and significance tests of the final run (ten seeds, 42 to 51) |
| `results/v4_resplit/` | significance tests over the random calibration and test re-splits |
| `results/v4/` | results of the first version (three seeds) |
| `figures/` | the result figures (ten seeds) |

## Setup

```
pip install -r requirements.txt
```

The image models are an ImageNet-pretrained ResNet-18 fine-tuned with FedAvg; training
was run on one NVIDIA L4. Evaluation runs on the CPU.

## Reproduce

Download the image data and train the classifiers, one cache file per dataset, client count,
partition and seed:

```
python build_cifar_cache.py cifar10
python build_cifar_cache.py cifar100
python v3_train.py --dataset cifar10  --seeds 42,43,44,45,46,47,48,49,50,51
python v3_train.py --dataset cifar100 --seeds 42,43,44,45,46,47,48,49,50,51
python v3_train.py --dataset text     --seeds 42,43,44,45,46,47,48,49,50,51 --tag-suffix _b
```

Put the caches in `results/v5/cache`, evaluate every method, and build the report:

```
python v3_eval.py --cache results/v5/cache --out results/v5/eval
python analyze_results_v3.py --eval results/v5/eval --acc results/v5/accuracy.csv \
    --cache results/v5/cache --out results/v5/RESULTS_V5.md
```

Significance over the ten seeds (100 noise draws per seed) and over 20 random calibration and
test re-splits of each model:

```
python v3_eval.py --cache results/v5/cache --out results/v5/eval_r100 --groups pcfcp,qq,variants --r-fast 100
python significance_v4.py --eval results/v5/eval_r100 --out results/v5/SIGNIFICANCE_10seed.md
python resplit_caches.py --cache results/v5/cache --out results/v4_resplit/cache --repeats 20
python resplit_caches.py --cache results/v5/cache --out results/v4_resplit/cache --repeats 20 \
    --pattern '^text_K10_dir0\.5_b_seed\d+\.npz$'
python v3_eval.py --cache results/v4_resplit/cache --out results/v4_resplit/eval --groups refs,pcfcp,qq
python significance_v4.py --eval results/v4_resplit/eval --model-div 100 \
    --out results/v4_resplit/SIGNIFICANCE_resplit.md
```

Tables and figures:

```
PCFCP_EVAL=results/v5/eval PCFCP_CACHE=results/v5/cache python make_figures_v3.py
python make_thesis_tables_v5.py
python make_thesis_tables_v5_new.py
python make_paper_v2.py
```

The final run was assembled in stages: seeds 42 to 44 from the first version and seeds 45 to 51
from `launch_seeds.sh`, `launch_seeds2.sh` and `results/batch/eval_seeds.sh`. The test scores were
then clipped to [0, 1] (`results/batch/clip_*`), because float32 probability rows can sum to
slightly more than 1; `v3_eval.py` now clips them, so a fresh run gives the corrected results
directly. `results/batch/resplit_text.sh` adds the 20 Newsgroups re-splits and
`results/batch/eval_dpfedcp_sg.sh` runs DP-FedCP at the noise levels of its paper.

## Coverage bounds

The scripts in `theory/` check the coverage results of the paper by simulation and give the
values it reports:

| Script | What it gives |
|---|---|
| `theory/lower_bound_table.py` | the lower bound of Theorem 2 for a class with 25 and with 250 pooled calibration examples (Table S-I), and its row of Table S-II |
| `theory/stress_test.py` | Theorems 1 and 2, Lemma 1 and Proposition 4 over 10,920 configurations of bins, levels, budgets, class counts and score distributions; no violation |
| `theory/upper_bound.py` | the upper bound of Proposition 3 over the same grid and on the score distributions of real classes (no violation), its value at eight bins, and the bin term and simulated coverage of Table S-II; needs the seed-42 caches |
| `theory/floor_and_example.py` | both bounds with the pooled floor in a ten-class federation, and the n = 10 example showing that the lower bound has no exact 1 - alpha form under the noise |

```
python theory/lower_bound_table.py
python theory/stress_test.py 10000
python theory/upper_bound.py 10000
python theory/floor_and_example.py
```

## Main result

Worst-class coverage in percent / mean set size in labels, target 90 percent, mean over
ten seeds, example-level privacy budget epsilon (from `results/v5/RESULTS_V5_summary.csv`):

| Dataset | Method | eps = 1 | eps = 2 | eps = 4 | eps = 8 |
|---|---|---|---|---|---|
| CIFAR-10-LT (IF 100) | DP-FedCP | 0.0 / 5.55 | 4.7 / 5.62 | 28.5 / 6.02 | 56.2 / 5.38 |
| | FedCP2-QQ | 95.7 / 2.29 | 90.9 / 1.63 | 87.1 / 1.42 | 85.0 / 1.35 |
| | PC-FCP | 90.5 / 1.74 | 90.5 / 1.66 | 90.7 / 1.64 | 90.8 / 1.64 |
| CIFAR-100-LT (IF 10) | DP-FedCP | 0.0 / 55.97 | 0.0 / 58.87 | 1.4 / 57.19 | 42.1 / 46.86 |
| | FedCP2-QQ | 83.5 / 8.58 | 78.1 / 6.07 | 76.2 / 5.40 | 75.2 / 5.16 |
| | PC-FCP | 84.2 / 10.38 | 84.0 / 9.06 | 84.0 / 8.80 | 84.1 / 8.74 |
| 20 Newsgroups (IF 10) | DP-FedCP | 0.0 / 11.50 | 0.0 / 11.61 | 15.9 / 12.75 | 47.8 / 12.04 |
| | FedCP2-QQ | 100.0 / 20.00 | 97.1 / 12.83 | 91.0 / 7.70 | 86.3 / 5.96 |
| | PC-FCP | 86.4 / 6.38 | 86.4 / 6.09 | 86.5 / 6.02 | 86.7 / 6.02 |

The non-private per-class reference (centralized Mondrian) is 85.0 / 1.44, 72.5 / 5.90 and
79.2 / 4.54 on the three datasets. From eps = 3 on CIFAR-10-LT and from eps = 2 on CIFAR-100-LT,
the worst class of PC-FCP is higher than that of FedCP2-QQ in every seed (Holm-adjusted
p <= 0.008, `results/v5/SIGNIFICANCE_10seed.md`). The standard 20 Newsgroups test set is split
by date, so calibration and test are not exchangeable there; its random re-splits are in
`results/v4_resplit/`.

## Baselines

DP-FedCP (Plassier et al., ICML 2023) is reimplemented from the paper, which has no public
code; the validation against the paper's synthetic experiment is in `baselines/plassier/`.
FedCP-QQ and its private variant FedCP2-QQ (Humbert et al., ICML 2023) follow the authors'
construction for clients of unequal size. Clustered CP (Ding et al., NeurIPS 2023) was
checked against the authors' released code.
