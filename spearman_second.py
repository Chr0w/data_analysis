#!/usr/bin/env python3
"""
Read ESI and RMSE values from spearman_rank_test_data and compute Spearman rank correlation.
"""

import os
import sys
import numpy as np
from scipy.stats import spearmanr as scipy_spearmanr

DATA_DIR = '/home/mircrda/data_analysis/spearman_rank_test_data_esi_vs_mir'
X_PATH = os.path.join(DATA_DIR, 'esi.txt')
Y_PATH = os.path.join(DATA_DIR, 'mir.txt')

def main():
    if not os.path.exists(X_PATH):
        print(f"Error: File not found: {X_PATH}")
        sys.exit(1)
    if not os.path.exists(Y_PATH):
        print(f"Error: File not found: {Y_PATH}")
        sys.exit(1)

    with open(X_PATH) as f:
        x = np.array([float(line.strip()) for line in f if line.strip()])
    with open(Y_PATH) as f:
        y = np.array([float(line.strip()) for line in f if line.strip()])

    if len(x) != len(y):
        print(f"Error: Length mismatch: x={len(x)}, y={len(y)}")
        sys.exit(1)
    if len(x) < 2:
        print("Error: Need at least 2 pairs for correlation.")
        sys.exit(1)

    res = scipy_spearmanr(x, y)
    rho = getattr(res, 'statistic', res[0])
    p_value = getattr(res, 'pvalue', res[1])

    print(f"rho: {round(rho, 5)}")
    print(f"p_value: {p_value}")
    print(f"Number of pairs: {len(x)}")

if __name__ == '__main__':
    main()
