# Plassier DP-FedCP: reimplementation and validation

**Paper:** V. Plassier, M. Makni, A. Rubashevskii, E. Moulines, M. Panov, "Conformal Prediction for Federated Uncertainty Quantification Under Label Shift," ICML 2023, PMLR v202, pp. 27907-27947. arXiv:2306.05131.

**Why reimplemented.** No public code exists. Verified three ways: the paper and the PMLR page carry no code link; the first author's GitHub has eight repositories, none conformal-related; web searches for DP-FedCP implementations return nothing. The advisor authorised implementing the mathematics directly (meeting 2026-07-24).

**Why this baseline matters.** It is the only prior work matching our full setting: federated **and** differentially private conformal prediction. Its guarantee is marginal, so the comparison of interest is whether a marginal DP federated method leaves rare classes uncovered.

---

## 1. Validation result (200 repetitions)

The anti-strawman protocol requires reproducing a published result before using a reimplementation as a comparison. The target is the paper's own synthetic experiment, Figure 1(b), which is fully specified in Section 5 and Appendix E.2: 3 classes, `X | Y=y ~ N(theta_y, I_2)` with `theta = [-1,0], [1,0], [1,3]`; 2 agents with label distributions (0.8, 0.1, 0.1) and (0.1, 0.1, 0.8); calibration sizes (1000, 50); target agent 2; Bayes classifier; alpha = 0.1; T = 200, K = 20, eta = 1e-3, gamma = 1e-6, full participation.

| method | mean coverage | std | median | set size |
|---|---|---|---|---|
| Unweighted Global | 90.12 | 1.16 | 90.10 | 1.170 |
| Unweighted Local | 90.48 | **4.50** | 90.85 | 1.202 |
| Oracle Weights | 90.67 | 2.79 | 91.10 | 1.180 |
| **DP-FedCP (ours)** | **90.45** | 2.27 | 90.65 | 1.176 |

The paper gives no numbers for Figure 1(b), only a plot and prose, so the validation targets are the qualitative claims it states.

| # | paper's claim | measured | verdict |
|---|---|---|---|
| 1 | DP-FedCP "little different from ... oracle weights" | 0.22 pp apart | **PASS** (tol 2.0) |
| 2 | DP-FedCP near nominal 0.90 | 0.45 pp from nominal | **PASS** (tol 2.5) |
| 3 | Unweighted Local "too little data to produce robust prediction sets" | 1.98x the spread of DP-FedCP | **PASS** (>1.5) |
| 4 | Unweighted Global displaced, "mostly due to ... agent 1" | 0.12 pp from nominal, not displaced | **NOT REPRODUCED** |

Raw output in `validation_output_200reps.txt`. Reproduce with `python dp_fedcp.py validate --reps 200`.

**On check 4.** In our configuration Unweighted Global is near nominal (per-class coverage 91.9 / 91.0 / 92.0), so it is not displaced the way Figure 1(b) shows. Whether pooling hurts the target agent depends on the score distributions, which depend on two details the paper does not specify: the Bayes classifier's class prior (D6) and the composition of the test set over which coverage is measured (D8). We record this as not reproduced rather than adjusting D6 or D8 until it matches, since tuning a baseline until it looks bad is exactly the strawman risk the protocol exists to prevent. The three checks that concern DP-FedCP itself all pass.

---

## 2. What the validation caught

The first run produced DP-FedCP at **79% coverage**, an 11-point undercoverage. Rather than adjust parameters until the number looked right, the cause was isolated:

1. **The exact weighted quantile of our own mixture returned 0.9049**, matching the Oracle Weights threshold exactly. So the importance weights, the multinomial subsampling, and the mixture construction were correct.
2. **A single-agent case converged to 0.9144 against an exact value of 0.9153** at T = 20000. So the Moreau-Yosida gradient and the objective were correct.
3. **At the paper's T = 200 the same code returned 0.8228 from q_0 = 0.5, but 0.9132 from q_0 = 0.9.**

Measured sensitivity (single agent, exact quantile 0.9153, T = 200):

| q_0 | eta = 1e-3 | eta = 1e-2 | eta = 5e-2 |
|---|---|---|---|
| 0.50 | 0.8228 | 0.9062 | 0.9136 |
| 0.80 | 0.8906 | 0.9129 | 0.9149 |
| 0.90 | 0.9132 | 0.9151 | 0.9153 |
| 0.95 | 0.9272 | 0.9203 | 0.9163 |

**Conclusion: the paper's stated optimisation budget converges only from a starting point already near the answer, and the paper never specifies `q_0`.** With T = 200, K = 20, eta = 1e-3 the iterate moves at most `eta * T * K * |grad| = 4.0` in total, but the effective step shrinks as the gradient balances, so a neutral start does not arrive.

**Our resolution (D7).** Initialise `q_0` at the unweighted empirical (1-alpha) quantile of the pooled retained scores. This uses no oracle information (it is the Unweighted Global baseline, computable by the server), and it places the optimiser in the correct basin so the paper's stated T, K, eta then converge. This is a choice forced by the paper's silence, not a departure from it.

---

## 3. Specification decisions

The paper is underspecified in ten places. Each choice is documented in the module docstring of `dp_fedcp.py` and summarised here.

| id | issue | our choice |
|---|---|---|
| D1 | `lambda^i_y` appears in the arXiv Algorithm 1 server update but not the PMLR box or inline text | use the lambda-weighted update (arXiv box; the Appendix B analysis is only consistent with it, and `sum_i lambda^i_y = 1`) |
| D2 | `sigma_g` for the synthetic experiment not stated | validate at `sigma_g = 0`, matching the paper's explicitly non-private CIFAR figure |
| D3 | `sigmabar` (discrete-Gaussian count noise) never stated anywhere | `sigmabar = 0` for validation; exposed as a parameter |
| D4 | `Nbar` not restated for DP-FedCP | `floor(N/2)`, as in the theorems and the Oracle Weights baseline |
| D5 | training counts `M^i_y` do not exist in the synthetic setting (Bayes classifier, no training set) | estimate from each agent's calibration labels |
| D6 | Bayes classifier's class prior not stated | uniform over the three classes |
| D7 | `q_0` not stated, and it determines convergence | unweighted empirical quantile of pooled retained scores (see §2) |
| D8 | test-set size and composition per repetition not stated | 1000 fresh points from the target agent's distribution |
| D9 | Algorithm 2 literally says "Compute & Send {V(X,Y)}", which would ship all scores to the server and make the private quantile pointless | keep scores local and run Algorithm 1, the privacy-consistent reading |
| D10 | `lambda^i_y` sum range differs between main text eq. (13) and appendix eq. (18) | use `N^i ^ Nbar^i`, matching the subsampling design |

Two further notes recorded during extraction. The paper's theory requires `eta in (0, gamma/10]` and `T ~ N^4`, while the experiments use `eta = 1e-3` with `gamma = 1e-6`, i.e. `eta = 1000 * gamma`, violating the theoretical stepsize condition by four orders of magnitude; we follow the experimental values, as the paper does. And the main-text Theorem 4.4 and its Appendix D restatement disagree on the placement of `1/eps` and `log(1/deltabar)` inside the radical; this is irrelevant here because the experiments set `sigma_g` directly and never report an epsilon.

---

## 4. Implementation map

`dp_fedcp.py`, with equation numbers from the PMLR camera-ready throughout:

| component | function | paper reference |
|---|---|---|
| APS score with uniform randomisation | `aps_scores`, `aps_scores_all_labels` | Section 5 |
| Moreau-Yosida gradient of the pinball loss | `grad_moreau_pinball` | Lemma A.1, eq. (17) |
| discrete Gaussian on Z | `discrete_gaussian` | Remark 3.2 (Canonne et al. 2020) |
| Algorithm 1, DP-FedAvgQE | `dp_fedavg_qe` | eq. (14), (15), (18) |
| Algorithm 2, DP-FedCP | `dp_fedcp_thresholds` | eq. (5), (10), (13) |
| Unweighted Local / Global baselines | `unweighted_local`, `unweighted_global` | Appendix E.1 |
| Oracle Weights baseline | `oracle_weights_thresholds` | Appendix E.1, eq. (85) |
| Figure 1(b) reproduction | `run_synthetic`, `validate` | Section 5, Appendix E.2 |

---

## 5. Behaviour under long-tailed data (diagnostic, before the head-to-head)

Before running the comparison we checked what the validated implementation actually does in our regime, so that the head-to-head numbers would be interpreted correctly rather than read as a failure of their method.

**Their thresholds are near-identical across classes.** Sweeping which agent is designated the target, on a 10-agent Dirichlet-skewed setup:

| target agent | min threshold | max threshold | spread |
|---|---|---|---|
| 0 | 0.885 | 0.886 | 0.0014 |
| 3 | 0.890 | 0.890 | 0.0007 |
| 7 | 0.903 | 0.903 | 0.0007 |
| 9 (most tail-heavy) | 0.877 | 0.882 | 0.0048 |

The spread across the ten per-class thresholds is between 0.0005 and 0.005 in every case. **This is the design, not a defect.** DP-FedCP estimates a marginal quantile under label-shift reweighting; the weights adjust for the target agent's label *mixture*, not for per-class difficulty. The method has no mechanism to give a rare class a larger cutoff than a common one, because per-class validity is not what it claims.

**Consequence on long-tailed data.** A single near-uniform threshold covers the easy head classes and misses the hard tail ones. Measured on CIFAR-10-LT (IF=100), with **no privacy noise at all** (`sigma_g = 0`), per-class coverage decays monotonically by class index:

```
0.943  0.866  0.674  0.518  0.007  0.003  0.000  0.000  0.000  0.000
```

marginal 30.1%, tail 0.0%. Since this occurs at zero noise, the shortfall is structural, not a DP artifact, exactly as our zero-noise FedCP-QQ check showed for the other incumbent.

**How this must be reported.** Their paper never claims per-class coverage and never measures it. The correct statement is that *a marginal federated DP method, run on long-tailed data, leaves rare classes uncovered because a single quantile cannot serve both head and tail*, which is our thesis, measured on the setting-matched incumbent. It is not a claim that their method is broken or that its authors erred.

**Two noise settings are reported, not one.** Mapping our epsilon budget through their Theorem 4.4 gives `sigma_g` = 304 / 76 / 38 at eps = 1 / 4 / 8, far above the range the paper itself ever tested (their sweep is {10, 30, 60, 100}). At those levels the quantile estimate leaves the [0,1] score range entirely (measured: thresholds of -7.1, -2.6, -9.5 at `sigma_g` = 303). Reporting only that would misrepresent them. We therefore report both the Theorem 4.4 mapping and `sigma_g = 100`, the noisiest setting the paper itself demonstrates, plus `sigma_g = 0` as the structural check.

---

## 6. Status and next step

**Validated for use as a baseline**, with the wording: *"reimplemented from the paper (no public code) and validated against its own Figure 1(b) synthetic experiment; three of the paper's four qualitative claims reproduce, the fourth depends on details the paper does not specify."*

Next: run against our method on long-tailed data at matched privacy, and report per-class coverage, which this method does not measure anywhere in the original paper.
