#!/usr/bin/env python3
"""
Save time-synced mean ESI and Yaw RMSE to files
(one value per line, like esi.txt and yaw_rmse.txt). Uses first 50% of all 30 runs.
"""

import pandas as pd
import numpy as np
import os
import sys
from data_loader import load_data


def calculate_yaw_error(gt_yaw, amcl_yaw):
    """Absolute yaw error in degrees (handles wrapping)."""
    diff = gt_yaw - amcl_yaw
    diff = ((diff + 180) % 360) - 180
    return abs(diff)


def main():
    read_file_percentage = 0.5  # First 50% of data
    N = 30
    user_home = os.path.expanduser('~')
    data_folder = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/old_data/default_amcl')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spearman_rank_test_data_esi_vs_yaw_rmse')

    # Load combined data (first 50% of each file) for mean ESI and yaw errors
    print("Loading data (first 50% of each file)...")
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

    # Check for yaw columns
    has_yaw = 'ground_truth_yaw' in combined_df.columns and 'amcl_yaw' in combined_df.columns
    if not has_yaw:
        print("Error: ground_truth_yaw and/or amcl_yaw columns not found.")
        sys.exit(1)

    combined_df['timestamp'] = pd.to_numeric(combined_df['timestamp'], errors='coerce')

    # Calculate yaw errors
    print("Calculating yaw errors...")
    yaw_errors = []
    for idx, row in combined_df.iterrows():
        yaw_err = calculate_yaw_error(
            row['ground_truth_yaw'],
            row['amcl_yaw']
        )
        yaw_errors.append(yaw_err)
    combined_df['yaw_error'] = yaw_errors

    # Build file_data and time-normalized ESI and yaw errors per file
    file_data = {}
    all_timestamps = []
    for file_id in sorted(combined_df['file_id'].unique()):
        df_file = combined_df[combined_df['file_id'] == file_id].copy()
        if len(df_file) > 0:
            first_ts = df_file['timestamp'].iloc[0]
            df_file['time_normalized'] = df_file['timestamp'] - first_ts
            file_data[file_id] = df_file
            all_timestamps.append(df_file['time_normalized'].values)

    # Common time grid
    max_time = max([ts.max() if len(ts) > 0 else 0 for ts in all_timestamps])
    min_dt = min([np.diff(ts).min() if len(ts) > 1 and np.diff(ts).min() > 0 else 1.0
                 for ts in all_timestamps if len(ts) > 1])
    common_time = np.arange(0, max_time + min_dt, min_dt)

    # Interpolate ESI for each file
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

    # Interpolate yaw errors for each file
    interpolated_yaw_errors = []
    for file_id in sorted(file_data.keys()):
        df_file = file_data[file_id]
        if len(df_file) > 1:
            yaw_err_interp = np.interp(common_time, df_file['time_normalized'].values, df_file['yaw_error'].values)
            interpolated_yaw_errors.append(yaw_err_interp)

    if len(interpolated_yaw_errors) == 0:
        print("Error: No interpolated yaw error data.")
        sys.exit(1)

    # Calculate Yaw RMSE across runs at each time point
    # RMSE = sqrt(mean(yaw_errors^2)) across runs
    yaw_errors_array = np.array(interpolated_yaw_errors)  # Shape: (n_runs, n_time_points)
    yaw_rmse = np.sqrt(np.mean(yaw_errors_array**2, axis=0))  # RMSE across runs at each time point

    # Restrict to valid pairs
    valid = np.isfinite(yaw_rmse) & np.isfinite(mean_esi)
    yaw_rmse_plot = yaw_rmse[valid]
    mean_esi_plot = mean_esi[valid]
    if len(yaw_rmse_plot) < 2:
        print("Error: Not enough valid (Yaw RMSE, mean ESI) pairs.")
        sys.exit(1)

    # Save to files: one value per line
    os.makedirs(out_dir, exist_ok=True)
    yaw_rmse_path = os.path.join(out_dir, 'yaw_rmse.txt')
    esi_path = os.path.join(out_dir, 'esi.txt')
    with open(yaw_rmse_path, 'w') as f:
        f.write('\n'.join(str(v) for v in yaw_rmse_plot))
    with open(esi_path, 'w') as f:
        f.write('\n'.join(str(v) for v in mean_esi_plot))
    print(f"Time-synced Yaw RMSE and mean ESI written to {out_dir}/ (yaw_rmse.txt, esi.txt)")
    print(f"  Number of pairs: {len(yaw_rmse_plot)}")


if __name__ == '__main__':
    main()
