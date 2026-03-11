#!/usr/bin/env python3
"""
Save time-synced mean ESI and Position RMSE to files
(one value per line, like esi.txt and rmse.txt). Uses first 50% of all 30 runs.
"""

import pandas as pd
import numpy as np
import os
import sys
from data_loader import load_data, calculate_position_error


def main():
    read_file_percentage = 0.5  # First 50% of data
    N = 30
    user_home = os.path.expanduser('~')
    data_folder = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/old_data/default_amcl')
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spearman_rank_test_data_esi_vs_rmse')

    # Load combined data (first 50% of each file) for mean ESI and position errors
    print("Loading data (first 50% of each file)...")
    combined_df = load_data(
        data_folder,
        N,
        mode='percentage',
        error_threshold=0.30,
        read_file_percentage=read_file_percentage
    )

    required_cols = ['esi', 'timestamp', 'ground_truth_x', 'ground_truth_y', 'amcl_x', 'amcl_y']
    missing = [c for c in required_cols if c not in combined_df.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        sys.exit(1)

    combined_df['timestamp'] = pd.to_numeric(combined_df['timestamp'], errors='coerce')

    # Calculate position errors
    print("Calculating position errors...")
    position_errors = []
    for idx, row in combined_df.iterrows():
        pos_err = calculate_position_error(
            row['ground_truth_x'], row['ground_truth_y'],
            row['amcl_x'], row['amcl_y']
        )
        position_errors.append(pos_err)
    combined_df['position_error'] = position_errors

    # Build file_data and time-normalized ESI and position errors per file
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

    # Interpolate position errors for each file
    interpolated_position_errors = []
    for file_id in sorted(file_data.keys()):
        df_file = file_data[file_id]
        if len(df_file) > 1:
            pos_err_interp = np.interp(common_time, df_file['time_normalized'].values, df_file['position_error'].values)
            interpolated_position_errors.append(pos_err_interp)

    if len(interpolated_position_errors) == 0:
        print("Error: No interpolated position error data.")
        sys.exit(1)

    # Calculate Position RMSE across runs at each time point
    # RMSE = sqrt(mean(position_errors^2)) across runs
    position_errors_array = np.array(interpolated_position_errors)  # Shape: (n_runs, n_time_points)
    position_rmse = np.sqrt(np.mean(position_errors_array**2, axis=0))  # RMSE across runs at each time point

    # Restrict to valid pairs
    valid = np.isfinite(position_rmse) & np.isfinite(mean_esi)
    position_rmse_plot = position_rmse[valid]
    mean_esi_plot = mean_esi[valid]
    if len(position_rmse_plot) < 2:
        print("Error: Not enough valid (Position RMSE, mean ESI) pairs.")
        sys.exit(1)

    # Save to files: one value per line
    os.makedirs(out_dir, exist_ok=True)
    rmse_path = os.path.join(out_dir, 'rmse.txt')
    esi_path = os.path.join(out_dir, 'esi.txt')
    with open(rmse_path, 'w') as f:
        f.write('\n'.join(str(v) for v in position_rmse_plot))
    with open(esi_path, 'w') as f:
        f.write('\n'.join(str(v) for v in mean_esi_plot))
    print(f"Time-synced Position RMSE and mean ESI written to {out_dir}/ (rmse.txt, esi.txt)")
    print(f"  Number of pairs: {len(position_rmse_plot)}")


if __name__ == '__main__':
    main()
