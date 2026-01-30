#!/usr/bin/env python3
"""
Normality check for Pearson correlation: plot histograms with normal overlay
and Q-Q plots for MIR and mean ESI (same data as pearson_mir_esi.py).
Pearson assumes both variables are approximately normally distributed.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from data_loader import load_data
from scipy import stats


def correct_map_integrity_ratio(mir_series: pd.Series) -> pd.Series:
    """Correct map integrity ratio values."""
    if mir_series is None or len(mir_series) == 0:
        return None
    old_mir = mir_series.values
    unmoved = old_mir * 20
    moved = 20 - unmoved
    new_mir = unmoved / (unmoved + 2 * moved)
    return pd.Series(new_mir, index=mir_series.index)


def main():
    read_file_percentage = 0.5
    N = 30
    user_home = os.path.expanduser('~')
    data_folder = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl')

    # Load combined data (same as pearson_mir_esi.py)
    print("Loading data (first 50% of each file)...")
    combined_df = load_data(
        data_folder,
        N,
        mode='percentage',
        error_threshold=0.50,
        read_file_percentage=read_file_percentage
    )

    required_cols = ['esi', 'timestamp']
    missing = [c for c in required_cols if c not in combined_df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)

    combined_df['timestamp'] = pd.to_numeric(combined_df['timestamp'], errors='coerce')

    file_data = {}
    all_timestamps = []
    for file_id in sorted(combined_df['file_id'].unique()):
        df_file = combined_df[combined_df['file_id'] == file_id].copy()
        if len(df_file) > 0:
            first_ts = df_file['timestamp'].iloc[0]
            df_file['time_normalized'] = df_file['timestamp'] - first_ts
            file_data[file_id] = df_file
            all_timestamps.append(df_file['time_normalized'].values)

    max_time = max([ts.max() if len(ts) > 0 else 0 for ts in all_timestamps])
    min_dt = min([np.diff(ts).min() if len(ts) > 1 and np.diff(ts).min() > 0 else 1.0
                 for ts in all_timestamps if len(ts) > 1])
    common_time = np.arange(0, max_time + min_dt, min_dt)

    interpolated_esi = []
    for file_id in sorted(file_data.keys()):
        df_file = file_data[file_id]
        if len(df_file) > 1:
            esi_interp = np.interp(common_time, df_file['time_normalized'].values, df_file['esi'].values)
            interpolated_esi.append(esi_interp)

    if len(interpolated_esi) == 0:
        print("Error: No interpolated ESI data.")
        sys.exit(1)

    mean_esi = np.mean(interpolated_esi, axis=0)

    first_file_path = os.path.join(data_folder, '1.csv')
    if not os.path.exists(first_file_path):
        print(f"Error: First file not found at {first_file_path}")
        sys.exit(1)

    df_first = pd.read_csv(first_file_path, skiprows=range(1, 100))
    df_first.columns = df_first.columns.str.strip()
    total_rows = len(df_first)
    num_rows = int(total_rows * read_file_percentage)
    df_first = df_first.iloc[:num_rows].copy()
    df_first['timestamp'] = pd.to_numeric(df_first['timestamp'], errors='coerce')
    first_ts = df_first['timestamp'].iloc[0]
    df_first['time_normalized'] = df_first['timestamp'] - first_ts

    if 'map_integrity_ratio' not in df_first.columns:
        print("Error: map_integrity_ratio not in first file.")
        sys.exit(1)

    corrected_mir = correct_map_integrity_ratio(df_first['map_integrity_ratio'])
    if corrected_mir is None:
        print("Error: Could not compute corrected MIR.")
        sys.exit(1)

    mir_interp = np.interp(common_time, df_first['time_normalized'].values, corrected_mir.values)

    valid = np.isfinite(mir_interp) & np.isfinite(mean_esi)
    mir_plot = mir_interp[valid]
    mean_esi_plot = mean_esi[valid]
    if len(mir_plot) < 2:
        print("Error: Not enough valid (MIR, mean ESI) pairs.")
        sys.exit(1)

    # Shapiro–Wilk normality tests
    sw_mir, p_mir = stats.shapiro(mir_plot)
    sw_esi, p_esi = stats.shapiro(mean_esi_plot)

    print("Normality (Shapiro–Wilk)")
    print("  MIR:     W = {:.4f}, p = {:.4e}  {}".format(sw_mir, p_mir, "normal (p>0.05)" if p_mir > 0.05 else "not normal (p≤0.05)"))
    print("  Mean ESI: W = {:.4f}, p = {:.4e}  {}".format(sw_esi, p_esi, "normal (p>0.05)" if p_esi > 0.05 else "not normal (p≤0.05)"))

    # Figure: 2 rows (MIR, Mean ESI), 2 cols (histogram + normal, Q–Q)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    def plot_normality(ax_hist, ax_qq, data, label, color, p_val):
        # Histogram with normal PDF overlay
        n, bins, _ = ax_hist.hist(data, bins=min(50, max(15, len(data) // 30)), density=True,
                                   color=color, alpha=0.6, edgecolor='k', linewidth=0.5)
        mu, sigma = data.mean(), data.std()
        x = np.linspace(data.min(), data.max(), 200)
        ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), 'k-', lw=2, label='Normal fit')
        ax_hist.set_ylabel('Density')
        ax_hist.set_title('{} — Histogram vs normal\nShapiro–Wilk p = {:.4e}'.format(label, p_val))
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)

        # Q–Q plot
        stats.probplot(data, dist='norm', plot=ax_qq)
        ax_qq.set_title('{} — Q–Q plot'.format(label))
        ax_qq.grid(True, alpha=0.3)

    plot_normality(axes[0, 0], axes[0, 1], mir_plot, 'MIR', 'steelblue', p_mir)
    plot_normality(axes[1, 0], axes[1, 1], mean_esi_plot, 'Mean ESI', 'darkorange', p_esi)

    fig.suptitle('Normality check for Pearson (MIR vs Mean ESI)\nIf data are normal: histogram matches curve, Q–Q points follow the line.',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig('pearson_normality_check.png', dpi=150, bbox_inches='tight')
    print("Saved pearson_normality_check.png")
    plt.show()


if __name__ == '__main__':
    main()
