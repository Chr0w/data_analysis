#!/usr/bin/env python3
"""
Read ESI and RMSE values from spearman_rank_test_data and compute Spearman rank correlation.
"""

import os
import sys
import numpy as np

DATA_DIR = '/home/mircrda/data_analysis/spearman_rank_test_data_esi_vs_rmse'
ESI_PATH = os.path.join(DATA_DIR, 'esi.txt')
RMSE_PATH = os.path.join(DATA_DIR, 'rmse.txt')

try:
    from scipy.stats import spearmanr as scipy_spearmanr
    from scipy.stats import norm
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    norm = None


def spearmanr_numpy(x, y):
    """Spearman rank correlation (Pearson correlation of ranks). No tie correction."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = np.argsort(np.argsort(x)).astype(float) + 1
    ry = np.argsort(np.argsort(y)).astype(float) + 1
    return np.corrcoef(rx, ry)[0, 1]


def main():
    if not os.path.exists(ESI_PATH):
        print(f"Error: File not found: {ESI_PATH}")
        sys.exit(1)
    if not os.path.exists(RMSE_PATH):
        print(f"Error: File not found: {RMSE_PATH}")
        sys.exit(1)

    with open(ESI_PATH) as f:
        esi = np.array([float(line.strip()) for line in f if line.strip()])
    with open(RMSE_PATH) as f:
        rmse = np.array([float(line.strip()) for line in f if line.strip()])

    if len(esi) != len(rmse):
        print(f"Error: Length mismatch: esi={len(esi)}, rmse={len(rmse)}")
        sys.exit(1)
    if len(esi) < 2:
        print("Error: Need at least 2 pairs for correlation.")
        sys.exit(1)

    if HAS_SCIPY:
        res = scipy_spearmanr(esi, rmse)
        rho = getattr(res, 'statistic', res[0])
        p_value = getattr(res, 'pvalue', res[1])
    else:
        rho = spearmanr_numpy(esi, rmse)
        p_value = None

    print("Spearman rank correlation (ESI vs RMSE)")
    print("=" * 50)
    print(f"  Correlation (rho): {rho:.6f}")
    if p_value is not None:
        if p_value == 0.0 or p_value < 1e-300:
            # Underflow: compute approximate p in log-space (asymptotic normal for Spearman)
            # z = rho * sqrt(n-1) ~ N(0,1) under H0; two-tailed p = 2 * norm.sf(|z|)
            if HAS_SCIPY and norm is not None and len(esi) > 2:
                z = abs(rho) * np.sqrt(len(esi) - 1)
                log_p = np.log(2) + norm.logsf(z)
                log10_p = log_p / np.log(10)
                # Report in scientific notation for article: p ≈ 10^{exponent}
                print(f"  p-value:           ≈ 10^{log10_p:.2f}  (two-tailed, asymptotic)")
            else:
                print(f"  p-value:           < 1e-300 (effectively zero)")
        else:
            print(f"  p-value:           {p_value:.4e}")
    print(f"  Number of pairs:   {len(esi)}")
    print("=" * 50)


if __name__ == '__main__':
    main()
