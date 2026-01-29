#!/usr/bin/env python3
"""
Script to plot mean RMSE (x) vs interpolated mean ESI (y) with interactive
exponential decay overlay: y = a * exp(-b * x + c) - d (same formula as prediction_experiment_1.py).
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import numpy as np
import os
import sys
from data_loader import load_data, calculate_position_error

# Optimization range limits from prediction_experiment_1.py (lines 482-485, 524-527)
A_MIN, A_MAX = 0.3, 10.0
B_MIN, B_MAX = 4.0, 50.0
C_MIN, C_MAX = -2.0, 2.0
D_MIN, D_MAX = -0.2, 0.05


def decay_curve(x, a, b, c, d):
    """Exponential decay: a * exp(-b * x + c) - d (per prediction_experiment_1.py line 46)."""
    return a * np.exp(-b * np.array(x) + c) - d


def main():
    # Configuration parameters
    N = 30
    read_file_percentage = 0.6

    user_home = os.path.expanduser('~')
    data_folder = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl')

    # Load combined data
    print("Loading data...")
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

    # Build file_data with time-normalized, ESI, and position error per file
    file_data = {}
    all_timestamps = []
    for file_id in sorted(combined_df['file_id'].unique()):
        df_file = combined_df[combined_df['file_id'] == file_id].copy()
        if len(df_file) > 0:
            first_ts = df_file['timestamp'].iloc[0]
            df_file['time_normalized'] = df_file['timestamp'] - first_ts
            # Position error per row
            position_errors = []
            for _, row in df_file.iterrows():
                err = calculate_position_error(
                    row['ground_truth_x'], row['ground_truth_y'],
                    row['amcl_x'], row['amcl_y']
                )
                position_errors.append(err)
            df_file['position_error'] = position_errors
            file_data[file_id] = df_file
            all_timestamps.append(df_file['time_normalized'].values)

    # Common time grid
    max_time = max([ts.max() if len(ts) > 0 else 0 for ts in all_timestamps])
    min_dt = min([np.diff(ts).min() if len(ts) > 1 and np.diff(ts).min() > 0 else 1.0
                 for ts in all_timestamps if len(ts) > 1])
    common_time = np.arange(0, max_time + min_dt, min_dt)

    # Interpolate ESI per run to common_time → mean ESI
    interpolated_esi = []
    interpolated_rmse = []
    for file_id in sorted(file_data.keys()):
        df_file = file_data[file_id]
        if len(df_file) > 1:
            esi_interp = np.interp(common_time, df_file['time_normalized'].values, df_file['esi'].values)
            pos_err_interp = np.interp(common_time, df_file['time_normalized'].values, df_file['position_error'].values)
            interpolated_esi.append(esi_interp)
            interpolated_rmse.append(pos_err_interp)

    if len(interpolated_esi) == 0:
        print("Error: No interpolated ESI data.")
        sys.exit(1)

    mean_esi = np.mean(interpolated_esi, axis=0)
    mean_rmse = np.mean(interpolated_rmse, axis=0)

    # Restrict to valid range
    valid = np.isfinite(mean_rmse) & np.isfinite(mean_esi)
    mean_rmse_plot = mean_rmse[valid]
    mean_esi_plot = mean_esi[valid]
    if len(mean_rmse_plot) < 2:
        print("Error: Not enough valid (mean RMSE, mean ESI) pairs.")
        sys.exit(1)

    # Write Spearman rank test data: one file per variable, one value per line
    spearman_dir = '/home/mircrda/data_analysis/spearman_rank_test_data'
    os.makedirs(spearman_dir, exist_ok=True)
    esi_path = os.path.join(spearman_dir, 'esi.txt')
    rmse_path = os.path.join(spearman_dir, 'rmse.txt')
    with open(esi_path, 'w') as f:
        f.write('\n'.join(str(v) for v in mean_esi_plot))
    with open(rmse_path, 'w') as f:
        f.write('\n'.join(str(v) for v in mean_rmse_plot))
    print(f"Spearman rank test data written to {spearman_dir}/ (esi.txt, rmse.txt)")

    # Initial decay parameters (mid-range of optimization bounds)
    a0 = (A_MIN + A_MAX) / 2
    b0 = (B_MIN + B_MAX) / 2
    c0 = (C_MIN + C_MAX) / 2
    d0 = (D_MIN + D_MAX) / 2

    # Plot: mean RMSE (x) vs interpolated mean ESI (y)
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.subplots_adjust(bottom=0.35)

    # Data: line and points
    ax.plot(mean_rmse_plot, mean_esi_plot, 'b-', alpha=0.5, linewidth=0.5)
    ax.plot(mean_rmse_plot, mean_esi_plot, 'b.', markersize=2, alpha=0.7)

    # Start and end points
    ax.scatter(mean_rmse_plot[0], mean_esi_plot[0], s=80, c='green', marker='o', edgecolors='darkgreen', linewidths=2, zorder=5, label='Start')
    ax.scatter(mean_rmse_plot[-1], mean_esi_plot[-1], s=80, c='red', marker='s', edgecolors='darkred', linewidths=2, zorder=5, label='End')

    # x range for decay curve (same as data x range)
    x_curve = np.linspace(mean_rmse_plot.min(), mean_rmse_plot.max(), 300)
    y_curve = decay_curve(x_curve, a0, b0, c0, d0)
    line_decay, = ax.plot(x_curve, y_curve, 'r-', linewidth=2, label='a*exp(-b*x+c)-d')

    ax.set_xlabel('Mean RMSE [m]', fontsize=12)
    ax.set_ylabel('Interpolated Mean ESI', fontsize=12)
    ax.set_title(f'Mean RMSE vs Mean ESI (N={N} runs, {read_file_percentage*100:.0f}% of each file)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)

    # Checkbox to toggle decay curve
    ax_check = plt.axes([0.02, 0.07, 0.12, 0.08])
    check = CheckButtons(ax_check, ['Decay curve'], [True])

    def toggle_decay(label):
        line_decay.set_visible(not line_decay.get_visible())
        fig.canvas.draw_idle()

    check.on_clicked(toggle_decay)

    # Sliders for a, b, c, d
    ax_a = plt.axes([0.15, 0.22, 0.7, 0.02])
    ax_b = plt.axes([0.15, 0.17, 0.7, 0.02])
    ax_c = plt.axes([0.15, 0.12, 0.7, 0.02])
    ax_d = plt.axes([0.15, 0.07, 0.7, 0.02])
    slider_a = Slider(ax_a, 'a', A_MIN, A_MAX, valinit=a0, valfmt='%.3f')
    slider_b = Slider(ax_b, 'b', B_MIN, B_MAX, valinit=b0, valfmt='%.3f')
    slider_c = Slider(ax_c, 'c', C_MIN, C_MAX, valinit=c0, valfmt='%.3f')
    slider_d = Slider(ax_d, 'd', D_MIN, D_MAX, valinit=d0, valfmt='%.3f')

    def update(_):
        a, b, c, d = slider_a.val, slider_b.val, slider_c.val, slider_d.val
        y_curve = decay_curve(x_curve, a, b, c, d)
        line_decay.set_ydata(y_curve)
        fig.canvas.draw_idle()

    slider_a.on_changed(update)
    slider_b.on_changed(update)
    slider_c.on_changed(update)
    slider_d.on_changed(update)

    output_path = os.path.join(data_folder, 'mean_esi_mean_rmse_correlation_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.show()

    print("\n" + "="*60)
    print("Mean RMSE vs Mean ESI")
    print("="*60)
    print(f"Number of points: {len(mean_rmse_plot)}")
    print("="*60)


if __name__ == '__main__':
    main()
