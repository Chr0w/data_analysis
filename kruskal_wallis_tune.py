#!/usr/bin/env python3
"""
Kruskal-Wallis H-test on box_plot_individual_results.csv:
pairwise comparisons of dynamic (tuning) vs default (default_02) and
dynamic vs low alpha (default_001), for both periods (before lost, after regained).
"""

import pandas as pd
import os
from scipy import stats

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'box_plot_individual_results.csv')

DYNAMIC = "tuning"
DEFAULT = "default_02"
LOW_ALPHA = "default_001"

# (column, display name) per period
PERIODS = [
    (
        "Before lost",
        [
            ("time_until_first_exceed_s", "Time until first exceed (s)"),
            ("position_rmse_pre_lost_m", "Position RMSE (m)"),
            ("yaw_rmse_pre_lost_deg", "Yaw RMSE (deg)"),
        ],
    ),
    (
        "After regained",
        [
            ("time_after_last_exceed_s", "Time after last exceed (s)"),
            ("position_rmse_post_lost_m", "Position RMSE (m)"),
            ("yaw_rmse_post_lost_deg", "Yaw RMSE (deg)"),
        ],
    ),
]


def kruskal_two_groups(df, col, mode_a, mode_b):
    """Kruskal-Wallis with two groups; returns (H, p) or (None, None) if invalid."""
    a = df.loc[df["mode"] == mode_a, col].dropna().values
    b = df.loc[df["mode"] == mode_b, col].dropna().values
    if len(a) == 0 or len(b) == 0:
        return None, None
    try:
        return stats.kruskal(a, b)
    except Exception:
        return None, None


def test_proportions(df, mode_a, mode_b):
    """
    Test if the proportion of runs ending localized differs between two modes.
    Uses Fisher's exact test for small samples, chi-square for larger samples.
    Returns (statistic, p_value) or (None, None) if invalid.
    """
    # A run ends localized if it has post-lost RMSE data (recovered and ended within thresholds)
    # OR if time_after_last_exceed > 0 (recovered after last exceed)
    # We'll use: has post-lost position RMSE data as the indicator
    def is_localized(row):
        # Check if post-lost RMSE exists (not NaN) - indicates recovery and ending within thresholds
        return pd.notna(row.get('position_rmse_post_lost_m', None)) or (row.get('time_after_last_exceed_s', 0) > 0)
    
    a_localized = df.loc[df["mode"] == mode_a].apply(is_localized, axis=1).sum()
    a_total = len(df.loc[df["mode"] == mode_a])
    b_localized = df.loc[df["mode"] == mode_b].apply(is_localized, axis=1).sum()
    b_total = len(df.loc[df["mode"] == mode_b])
    
    if a_total == 0 or b_total == 0:
        return None, None
    
    a_not_localized = a_total - a_localized
    b_not_localized = b_total - b_localized
    
    # Create contingency table
    contingency = [[a_localized, a_not_localized],
                   [b_localized, b_not_localized]]
    
    try:
        # Use Fisher's exact test (more appropriate for small samples)
        if a_total < 30 or b_total < 30:
            oddsratio, p_value = stats.fisher_exact(contingency)
            return ('fisher', oddsratio), p_value
        else:
            # Chi-square test for larger samples
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            return ('chi2', chi2), p_value
    except Exception:
        return None, None


def test_proportions_hardcoded(df, mode_a, mode_b, localized_counts):
    """
    Test if the proportion of runs ending localized differs between two modes.
    Uses hardcoded values for localized counts.
    Uses Fisher's exact test for small samples, chi-square for larger samples.
    Returns (statistic, p_value) or (None, None) if invalid.
    """
    a_localized = localized_counts.get(mode_a, 0)
    a_total = len(df.loc[df["mode"] == mode_a])
    b_localized = localized_counts.get(mode_b, 0)
    b_total = len(df.loc[df["mode"] == mode_b])
    
    if a_total == 0 or b_total == 0:
        return None, None
    
    a_not_localized = a_total - a_localized
    b_not_localized = b_total - b_localized
    
    # Create contingency table
    contingency = [[a_localized, a_not_localized],
                   [b_localized, b_not_localized]]
    
    try:
        # Use Fisher's exact test (more appropriate for small samples)
        if a_total < 30 or b_total < 30:
            oddsratio, p_value = stats.fisher_exact(contingency)
            return ('fisher', oddsratio), p_value
        else:
            # Chi-square test for larger samples
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            return ('chi2', chi2), p_value
    except Exception:
        return None, None


def run_tests(df, metrics):
    """For each metric: (H, p) vs default, sig, (H, p) vs low alpha, sig."""
    results = []
    for col, label in metrics:
        if col not in df.columns:
            results.append((label, None, None, "—", None, None, "—"))
            continue
        h_def, p_vs_default = kruskal_two_groups(df, col, DYNAMIC, DEFAULT)
        h_low, p_vs_low = kruskal_two_groups(df, col, DYNAMIC, LOW_ALPHA)
        sig_default = "Yes" if p_vs_default is not None and p_vs_default < 0.05 else ("No" if p_vs_default is not None else "—")
        sig_low = "Yes" if p_vs_low is not None and p_vs_low < 0.05 else ("No" if p_vs_low is not None else "—")
        results.append((label, h_def, p_vs_default, sig_default, h_low, p_vs_low, sig_low))
    return results


def print_table(results, period_name):
    pad = 2
    w_metric = 32
    w_h, w_p, w_sig = 10, 12, 6
    block_w = w_h + pad + w_p + pad + w_sig
    total_w = w_metric + pad + block_w + pad + block_w
    sep = "-" * total_w

    print(period_name)
    print(sep)
    # Header row 1: comparison labels
    h1 = "Metric".ljust(w_metric) + " " * pad + "vs default".center(block_w) + " " * pad + "vs low α".center(block_w)
    print(h1)
    # Header row 2: sub-headers H, p-value, Sig. (no repeated "Metric")
    h2 = (
        " ".ljust(w_metric)
        + " " * pad + "H".rjust(w_h) + " " * pad + "p-value".rjust(w_p) + " " * pad + "Sig.".rjust(w_sig)
        + " " * pad + "H".rjust(w_h) + " " * pad + "p-value".rjust(w_p) + " " * pad + "Sig.".rjust(w_sig)
    )
    print(h2)
    print(sep)

    for label, h_def, p_def, sig_def, h_low, p_low, sig_low in results:
        h_def_s = f"{h_def:.4f}" if h_def is not None else "—"
        p_def_s = f"{p_def:.4f}" if p_def is not None else "—"
        h_low_s = f"{h_low:.4f}" if h_low is not None else "—"
        p_low_s = f"{p_low:.4f}" if p_low is not None else "—"
        row = (
            label.ljust(w_metric)
            + " " * pad + h_def_s.rjust(w_h) + " " * pad + p_def_s.rjust(w_p) + " " * pad + sig_def.rjust(w_sig)
            + " " * pad + h_low_s.rjust(w_h) + " " * pad + p_low_s.rjust(w_p) + " " * pad + sig_low.rjust(w_sig)
        )
        print(row)
    print(sep)
    print()


def main():
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    print(f"Loaded {len(df)} rows.\n")
    print("Kruskal-Wallis (pairwise: dynamic vs default, dynamic vs low alpha)\n")

    for period_name, metrics in PERIODS:
        results = run_tests(df, metrics)
        print_table(results, f"Period: {period_name}")
    
    # Test proportion of runs ending localized
    print("=" * 80)
    print("Proportion of Runs Ending Localized")
    print("=" * 80)
    
    # Hardcoded values for runs ending localized
    localized_counts = {
        DEFAULT: 10,
        LOW_ALPHA: 11,
        DYNAMIC: 20,
    }
    
    # Get total counts from data
    for mode in [DYNAMIC, DEFAULT, LOW_ALPHA]:
        mode_df = df.loc[df["mode"] == mode]
        if len(mode_df) > 0:
            localized = localized_counts.get(mode, 0)
            total = len(mode_df)
            prop = localized / total if total > 0 else 0
            print(f"{mode}: {localized}/{total} ({prop:.1%})")
    
    print()
    
    # Test differences using hardcoded values
    result_def = test_proportions_hardcoded(df, DYNAMIC, DEFAULT, localized_counts)
    result_low = test_proportions_hardcoded(df, DYNAMIC, LOW_ALPHA, localized_counts)
    
    print("Comparison: dynamic vs default")
    if result_def[0] is not None:
        test_type, stat_def = result_def[0]
        p_def = result_def[1]
        test_name = "Fisher's exact (odds ratio)" if test_type == 'fisher' else "Chi-square"
        print(f"  {test_name}: {stat_def:.4f}")
        print(f"  p-value: {p_def:.4f}")
        print(f"  Significant: {'Yes' if p_def < 0.05 else 'No'}")
    else:
        print("  Could not compute test")
    print()
    
    print("Comparison: dynamic vs low alpha")
    if result_low[0] is not None:
        test_type, stat_low = result_low[0]
        p_low = result_low[1]
        test_name = "Fisher's exact (odds ratio)" if test_type == 'fisher' else "Chi-square"
        print(f"  {test_name}: {stat_low:.4f}")
        print(f"  p-value: {p_low:.4f}")
        print(f"  Significant: {'Yes' if p_low < 0.05 else 'No'}")
    else:
        print("  Could not compute test")
    print()


if __name__ == "__main__":
    main()
