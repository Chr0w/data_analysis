#!/usr/bin/env python3
"""
Load first 50% of data, compute Pearson correlation between map_integrity_ratio (MIR)
and interpolated mean ESI, and plot MIR (x) vs mean ESI (y).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from data_loader import load_data
from scipy.stats import pearsonr


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
    read_file_percentage = 0.5  # First 50% of data
    N = 30
    user_home = os.path.expanduser('~')
    data_folder = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl')

    # Load combined data (first 50% of each file) for mean ESI
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

    # Build file_data and time-normalized ESI per file
    file_data = {}
    all_timestamps = []
    for file_id in sorted(combined_df['file_id'].unique()):
        df_file = combined_df[combined_df['file_id'] == file_id].copy()
        if len(df_file) > 0:
            first_ts = df_file['timestamp'].iloc[0]
            df_file['time_normalized'] = df_file['timestamp'] - first_ts
            file_data[file_id] = df_file
            all_timestamps.append(df_file['time_normalized'].values)

    # Common time grid and interpolated mean ESI
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

    # Load first file for MIR (first 50%)
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

    # Restrict to valid pairs
    valid = np.isfinite(mir_interp) & np.isfinite(mean_esi)
    mir_plot = mir_interp[valid]
    mean_esi_plot = mean_esi[valid]
    if len(mir_plot) < 2:
        print("Error: Not enough valid (MIR, mean ESI) pairs.")
        sys.exit(1)

    # Pearson correlation (scipy)
    r, p_value = pearsonr(mir_plot, mean_esi_plot)

    print("Pearson correlation (MIR vs mean ESI)")
    print("=" * 50)
    print(f"  Correlation (r):  {r:.6f}")
    print(f"  p-value:          {p_value:.6e}")
    print(f"  Number of pairs:   {len(mir_plot)}")
    print("=" * 50)

    # Plot: MIR (x) vs mean ESI (y)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(mir_plot, mean_esi_plot, 'b.', markersize=2, alpha=0.7)
    ax.set_xlabel('Map Integrity Ratio (MIR)', fontsize=12)
    ax.set_ylabel('Interpolated Mean ESI', fontsize=12)
    ax.set_title(f'MIR vs Mean ESI (first 50% of data, N={N} runs)\nPearson r = {r:.4f}, p = {p_value:.4e}', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
