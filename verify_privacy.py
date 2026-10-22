"""Verify the privacy accounting claims in PRIVACY_ACCOUNTING.md.

Three checks:
  1. Global L1 sensitivity of the FULL release (all C per-class histograms
     concatenated) under remove-one and replace-one neighbouring datasets.
     Claim: 1 and 2 respectively, independent of C.
  2. Empirical privacy loss of the Laplace mechanism at scale 1/eps, which must
     not exceed eps.
  3. Client-level sensitivity, which is m (the client's calibration count), not
     1. This is the caveat that fixes the privacy UNIT of our claim.

Run: python verify_privacy.py
"""
import numpy as np


def build_release(labels, bins, C, B):
    """The published object: C per-class histograms over B score bins."""
    H = np.zeros((C, B))
    for y, b in zip(labels, bins):
        H[y, b] += 1
    return H


def check_sensitivity(C=5, B=8, trials=20000, seed=0):
    rng = np.random.default_rng(seed)
    worst_remove = 0.0
    worst_replace = 0.0
    for _ in range(trials):
        n = 40
        lab = rng.integers(0, C, n)
        bn = rng.integers(0, B, n)
        H1 = build_release(lab, bn, C, B)

        i = rng.integers(0, n)
        H2 = build_release(np.delete(lab, i), np.delete(bn, i), C, B)
        worst_remove = max(worst_remove, np.abs(H1 - H2).sum())

        lab2, bn2 = lab.copy(), bn.copy()
        j = rng.integers(0, n)
        lab2[j] = rng.integers(0, C)
        bn2[j] = rng.integers(0, B)
        H3 = build_release(lab2, bn2, C, B)
        worst_replace = max(worst_replace, np.abs(H1 - H3).sum())
    return worst_remove, worst_replace


def check_privacy_loss(eps=1.0, B=50, trials=200000, seed=0):
    """Max log-likelihood ratio between neighbouring datasets must be <= eps."""
    rng = np.random.default_rng(seed)
    h1 = np.zeros(B); h1[7] = 5.0
    h2 = h1.copy(); h2[7] = 4.0          # remove one person from that bin
    scale = 1.0 / eps

    def logpdf(x, mu):
        return (-np.abs(x - mu) / scale - np.log(2 * scale)).sum()

    worst = -np.inf
    for _ in range(trials):
        out = h1 + rng.laplace(0.0, scale, B)
        worst = max(worst, logpdf(out, h1) - logpdf(out, h2))
    return worst


def check_client_level(C=5, B=8, m=7, trials=5000, seed=0):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(trials):
        n_other = 60
        lab = rng.integers(0, C, n_other); bn = rng.integers(0, B, n_other)
        clab = rng.integers(0, C, m); cbn = rng.integers(0, B, m)
        H_with = build_release(np.concatenate([lab, clab]),
                               np.concatenate([bn, cbn]), C, B)
        H_without = build_release(lab, bn, C, B)
        worst = max(worst, np.abs(H_with - H_without).sum())
    return worst


if __name__ == "__main__":
    print("=" * 70)
    print("PRIVACY ACCOUNTING VERIFICATION")
    print("=" * 70)

    for C in (5, 10, 100):
        rm, rp = check_sensitivity(C=C, trials=5000)
        ok_rm = "PASS" if rm == 1.0 else "FAIL"
        ok_rp = "PASS" if rp == 2.0 else "FAIL"
        print(f"C={C:<4} remove-one L1 = {rm:.1f} (expect 1.0) {ok_rm}   "
              f"replace-one L1 = {rp:.1f} (expect 2.0) {ok_rp}")
    print("  -> sensitivity is independent of C: parallel composition holds,")
    print("     so the reported epsilon is GLOBAL, not per class.")
    print()

    for eps in (1.0, 4.0):
        worst = check_privacy_loss(eps=eps, trials=100000)
        ok = "PASS" if worst <= eps + 1e-9 else "FAIL"
        print(f"eps={eps}: max observed log-ratio = {worst:.4f}  bound = {eps:.4f}  {ok}")
    print("  -> Laplace(1/eps) per bin is correctly calibrated for sensitivity 1.")
    print()

    for m in (1, 7, 50):
        w = check_client_level(m=m, trials=2000)
        ok = "PASS" if w == float(m) else "FAIL"
        print(f"client holding m={m:<3} -> L1 change = {w:.1f} (expect {m}) {ok}")
    print("  -> CLIENT-level DP would need scale m_max/eps. Our claim is")
    print("     EXAMPLE-level DP, which must be stated wherever epsilon appears.")
