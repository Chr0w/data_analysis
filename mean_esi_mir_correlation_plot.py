#!/usr/bin/env python3
"""
Script to plot Map Integrity Ratio (x) vs interpolated mean ESI (y) and compute correlation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from data_loader import load_data


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
    # Configuration parameters
    N = 30
    read_file_percentage = 0.6

    user_home = os.path.expanduser('~')
    data_folder = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl')

    # Load combined data for mean ESI
    print("Loading data...")
    combined_df = load_data(
        data_folder,
        N,
        mode='percentage',
        error_threshold=0.30,
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

    # Load first file for MIR and interpolate to common_time
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

    # Interpolate MIR to common_time
    mir_interp = np.interp(common_time, df_first['time_normalized'].values, corrected_mir.values)

    # Restrict to valid range (drop NaNs if any)
    valid = np.isfinite(mir_interp) & np.isfinite(mean_esi)
    mir_plot = mir_interp[valid]
    mean_esi_plot = mean_esi[valid]
    if len(mir_plot) < 2:
        print("Error: Not enough valid (MIR, mean ESI) pairs.")
        sys.exit(1)

    # Correlation
    correlation = np.corrcoef(mir_plot, mean_esi_plot)[0, 1]

    # Plot: MIR (x) vs interpolated mean ESI (y)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(mir_plot, mean_esi_plot, 'b.', markersize=2, alpha=0.7)
    ax.set_xlabel('Map Integrity Ratio', fontsize=12)
    ax.set_ylabel('Interpolated Mean ESI', fontsize=12)
    ax.set_title(f'MIR vs Mean ESI (N={N} runs, {read_file_percentage*100:.0f}% of each file)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(data_folder, 'mean_esi_mir_correlation_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.show()

    print("\n" + "="*60)
    print("MIR vs Mean ESI correlation")
    print("="*60)
    print(f"Pearson correlation: {correlation:.6f}")
    print(f"Number of points: {len(mir_plot)}")
    print("="*60)


if __name__ == '__main__':
    main()
