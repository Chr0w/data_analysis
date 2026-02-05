"""Kruskal-Wallis H-test for comparing three independent groups."""

import numpy as np
from scipy import stats

# Groups: no, light, and moderate changes (5 datapoints each)
no_change = np.array([20, 45, 55, 65, 88])
light_change = np.array([20, 57, 60, 54, 72])
moderate_change = np.array([30, 41, 48, 76, 23])

alpha = 0.01

# Kruskal-Wallis H-test
statistic, p_value = stats.kruskal(no_change, light_change, moderate_change)

print("Kruskal-Wallis H-test")
print("=" * 40)
print(f"Groups: no change, light change, moderate change")
print(f"Sample sizes: n1={len(no_change)}, n2={len(light_change)}, n3={len(moderate_change)}")
print()
print(f"H-statistic: {statistic:.4f}")
print(f"p-value:     {p_value:.4f}")
print(f"Alpha:       {alpha}")
print()
if p_value < alpha:
    print(f"Result: Reject H0 at α={alpha}. There is a significant difference between at least two groups.")
else:
    print(f"Result: Fail to reject H0 at α={alpha}. No significant difference detected between groups.")
