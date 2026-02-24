#!/usr/bin/env python3
"""
Kruskal-Wallis H-test on box_plot_individual_results.csv:
pairwise comparisons of dynamic (tuning) vs default (default_02) and
dynamic vs low alpha (default_001), for both periods (before lost, after regained).
"""

import pandas as pd
from scipy import stats

CSV_PATH = "/home/mircrda/data_analysis/box_plot_individual_results.csv"

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


if __name__ == "__main__":
    main()
