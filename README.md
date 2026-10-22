# PC-FCP: Per-Class Federated Conformal Prediction under Differential Privacy

Code and experiment results for the MSc thesis *Per-Class Coverage for Rare Classes in
Federated Conformal Prediction under Differential Privacy* (Addis Ababa University, 2026).

PC-FCP calibrates one conformal threshold per class in a federation without moving any
calibration score. Each client counts its calibration scores into one histogram per class,
secure aggregation sums the histograms with Laplace noise added inside the sum, and the
server reads one threshold per class from the top of each released histogram, floored at
the pooled threshold. The release is example-level differentially private under a single
budget.

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
| `analyze_results_v3.py` | the results report |
| `make_figures_v3.py`, `make_thesis_tables_v4.py`, `make_appendix_tables.py`, `make_appendix_sd_small.py` | figures and tables |
| `make_tables_paper.py`, `make_figures_paper.py` | tables and figures of the journal version |
| `perclass_detail_v3.py`, `lt_setsize.py` | per-class thresholds and false-inclusion rates, set sizes under the long-tailed class mix |
| `verify_privacy.py`, `validate_clustered.py`, `t4_dpfedcp_check.py` | implementation checks |
| `launch_batch.sh`, `results/batch/` | the training and evaluation batch |
| `results/v4/` | the results report, summary tables and accuracies of the final run |
| `figures/` | the result figures |

## Setup

```
pip install -r requirements.txt
```

The image models are an ImageNet-pretrained ResNet-18 fine-tuned with FedAvg; training
was run on one NVIDIA L4. Evaluation runs on the CPU.

## Reproduce

Download the image data and train the classifiers (one cache file per dataset, client
count, partition and seed):

```
python build_cifar_cache.py cifar10
python build_cifar_cache.py cifar100
python v3_train.py --dataset cifar10  --seeds 42,43,44
python v3_train.py --dataset cifar100 --seeds 42,43,44
python v3_train.py --dataset text     --seeds 42,43,44
```

Evaluate every method on the caches, then build the report, tables and figures:

```
python v3_eval.py --cache results/v4/cache --out results/v4/eval
python analyze_results_v3.py --eval results/v4/eval --acc results/v4/accuracy.csv \
    --cache results/v4/cache --out results/v4/RESULTS_V4.md
PCFCP_EVAL=results/v4/eval PCFCP_CACHE=results/v4/cache python make_figures_v3.py
python make_thesis_tables_v4.py
```

`launch_batch.sh` and `results/batch/eval_v4.sh` run the whole batch in the order used for
the thesis, including the client sweep, the calibration-set-size study and the imbalance
factor 100 study.

## Main result

Worst-class coverage in percent / mean set size in labels, target 90 percent, mean over
3 seeds, example-level privacy budget epsilon (from `results/v4/RESULTS_V4.md`):

| Dataset | Method | eps = 1 | eps = 2 | eps = 4 | eps = 8 |
|---|---|---|---|---|---|
| CIFAR-10-LT (IF 100) | DP-FedCP | 0.0 / 6.29 | 14.8 / 5.67 | 41.6 / 6.37 | 53.3 / 5.50 |
| | FedCP2-QQ | 94.8 / 2.01 | 90.6 / 1.57 | 87.0 / 1.37 | 84.9 / 1.31 |
| | PC-FCP | 90.0 / 1.69 | 90.1 / 1.60 | 90.1 / 1.56 | 90.1 / 1.56 |
| CIFAR-100-LT (IF 10) | DP-FedCP | 0.0 / 56.43 | 0.0 / 59.06 | 2.9 / 57.69 | 43.2 / 47.60 |
| | FedCP2-QQ | 85.3 / 8.40 | 81.8 / 6.18 | 79.6 / 5.43 | 78.9 / 5.17 |
| | PC-FCP | 85.3 / 10.74 | 85.8 / 9.39 | 85.9 / 9.26 | 85.8 / 9.10 |
| 20 Newsgroups (IF 10) | DP-FedCP | 0.0 / 12.03 | 0.0 / 11.93 | 15.6 / 13.00 | 35.3 / 12.13 |
| | FedCP2-QQ | 100.0 / 20.00 | 96.8 / 13.50 | 90.3 / 7.50 | 84.8 / 5.82 |
| | PC-FCP | 86.0 / 6.59 | 85.5 / 6.14 | 85.6 / 6.11 | 85.8 / 6.10 |

The non-private per-class reference (centralized Mondrian) is 85.9 / 1.35, 72.7 / 6.09 and
79.0 / 4.56 on the three datasets.

## Baselines

DP-FedCP (Plassier et al., ICML 2023) is reimplemented from the paper, which has no public
code; the validation against the paper's synthetic experiment is in `baselines/plassier/`.
FedCP-QQ and its private variant FedCP2-QQ (Humbert et al., ICML 2023) follow the authors'
construction for clients of unequal size. Clustered CP (Ding et al., NeurIPS 2023) was
checked against the authors' released code.
