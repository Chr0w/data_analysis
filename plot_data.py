#!/usr/bin/env python3
"""
Script to plot data from CSV files as line plots.
Creates one subplot for each numeric column, using row index (line number) as x-axis.
Plots data from default, default_02, and tuning CSV files for comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
from matplotlib.widgets import Slider

def main():
    # Paths to the CSV files
    user_home = os.path.expanduser('~')
    default_csv_path = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl/default_combined_results_new.csv')
    default_02_csv_path = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_02/default_02_combined_results_new.csv')
    tuning_csv_path = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/alpha_tuning/tuning_combined_results_new.csv')
    
    # Check if files exist
    if not os.path.exists(default_csv_path):
        print(f"Error: File not found at {default_csv_path}")
        sys.exit(1)
    if not os.path.exists(default_02_csv_path):
        print(f"Error: File not found at {default_02_csv_path}")
        sys.exit(1)
    if not os.path.exists(tuning_csv_path):
        print(f"Error: File not found at {tuning_csv_path}")
        sys.exit(1)
    
    # Read the CSV files
    try:
        df_default = pd.read_csv(default_csv_path)
        df_default.columns = df_default.columns.str.strip()
        print(f"Loaded {len(df_default)} rows from {default_csv_path}")
        print(f"Default columns: {list(df_default.columns)}")

        df_default_02 = pd.read_csv(default_02_csv_path)
        df_default_02.columns = df_default_02.columns.str.strip()
        print(f"Loaded {len(df_default_02)} rows from {default_02_csv_path}")
        print(f"Default_02 columns: {list(df_default_02.columns)}")
        
        df_tuning = pd.read_csv(tuning_csv_path)
        df_tuning.columns = df_tuning.columns.str.strip()
        print(f"Loaded {len(df_tuning)} rows from {tuning_csv_path}")
        print(f"Tuning columns: {list(df_tuning.columns)}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Get numeric columns from both dataframes (exclude timestamp and specified columns)
    numeric_cols_default = df_default.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_default_02 = df_default_02.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_tuning = df_tuning.select_dtypes(include=[np.number]).columns.tolist()
    
    # Columns to exclude (including std deviation - only plot RMSE)
    exclude_cols = ['timestamp', 'total_messages', 'duration_s', 'msg_rate_hz', 'ESI_range', 'ESI_std_dev', 'mean_ESI', 'position_max_error', 'yaw_max_error', 'position_mean_error', 'yaw_mean_error', 'sum_cov_trace', 'position_std_dev', 'yaw_std_dev', 'run_id', 'yaw_RMSE']
    numeric_cols_default = [col for col in numeric_cols_default if col not in exclude_cols]
    numeric_cols_default_02 = [col for col in numeric_cols_default_02 if col not in exclude_cols]
    numeric_cols_tuning = [col for col in numeric_cols_tuning if col not in exclude_cols]
    
    # Get common columns to plot
    numeric_cols = list(set(numeric_cols_default) & set(numeric_cols_default_02) & set(numeric_cols_tuning))
    
    if not numeric_cols:
        print("Error: No common numeric columns found to plot")
        sys.exit(1)
    
    # Sort columns for consistent ordering (position_RMSE first, then yaw_RMSE)
    numeric_cols = sorted(numeric_cols)
    
    print(f"Plotting {len(numeric_cols)} common numeric columns: {numeric_cols}")
    
    # Calculate and print mean values for position and yaw RMSE
    print("\n" + "="*60)
    print("Mean RMSE Values:")
    print("="*60)
    
    if 'position_RMSE' in df_default.columns and 'position_RMSE' in df_default_02.columns and 'position_RMSE' in df_tuning.columns:
        mean_pos_rmse_default = df_default['position_RMSE'].mean()
        mean_pos_rmse_default_02 = df_default_02['position_RMSE'].mean()
        mean_pos_rmse_tuning = df_tuning['position_RMSE'].mean()
        print(f"Position RMSE")
        print(f"Default: {mean_pos_rmse_default:.6f}")
        print(f"Default_02: {mean_pos_rmse_default_02:.6f}")
        print(f"Tuning:  {mean_pos_rmse_tuning:.6f}")
        reduction_default = ((mean_pos_rmse_default - mean_pos_rmse_tuning) / mean_pos_rmse_default) * 100
        reduction_default_02 = ((mean_pos_rmse_default_02 - mean_pos_rmse_tuning) / mean_pos_rmse_default_02) * 100
        print(f"Reduction vs Default: {reduction_default:.2f}%")
        print(f"Reduction vs Default_02: {reduction_default_02:.2f}%")

    else:
        print("Position RMSE column not found in one or both datasets")
    
    print()
    
    if 'yaw_RMSE' in df_default.columns and 'yaw_RMSE' in df_default_02.columns and 'yaw_RMSE' in df_tuning.columns:
        print(f"Yaw RMSE")
        mean_yaw_rmse_default = df_default['yaw_RMSE'].mean()
        mean_yaw_rmse_default_02 = df_default_02['yaw_RMSE'].mean()
        mean_yaw_rmse_tuning = df_tuning['yaw_RMSE'].mean()
        print(f"Default: {mean_yaw_rmse_default:.6f}")
        print(f"Default_02: {mean_yaw_rmse_default_02:.6f}")
        print(f"Tuning:  {mean_yaw_rmse_tuning:.6f}")
        reduction_yaw_default = ((mean_yaw_rmse_default - mean_yaw_rmse_tuning) / mean_yaw_rmse_default) * 100
        reduction_yaw_default_02 = ((mean_yaw_rmse_default_02 - mean_yaw_rmse_tuning) / mean_yaw_rmse_default_02) * 100
        print(f"Reduction vs Default: {reduction_yaw_default:.2f}%")
        print(f"Reduction vs Default_02: {reduction_yaw_default_02:.2f}%")
    else:
        print("Yaw RMSE column not found in one or both datasets")
    
    print("="*60 + "\n")
    
    # Calculate grid size for subplots (RMSE side by side: 1 row, 2 cols)
    n_cols = len(numeric_cols)
    n_rows = 1
    n_cols_grid = n_cols
    if n_cols_grid < 1:
        n_cols_grid = 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(5*n_cols_grid, 4*n_rows))
    # fig.suptitle('Line Plots: Default vs Tuning Comparison', fontsize=16, y=0.995)
    
    # Flatten axes array if needed
    axes = np.atleast_1d(axes).flatten()
    
    # Create row indices for both dataframes
    row_indices_default = df_default.index.values
    row_indices_default_02 = df_default_02.index.values
    row_indices_tuning = df_tuning.index.values
    
    # Collect title text objects for the title-size slider
    title_texts = []

    # Plot each numeric column
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        # Use correct_map_integrity_ratio if col is map_integrity_ratio
        col_default = 'correct_map_integrity_ratio' if col == 'map_integrity_ratio' else col
        col_default_02 = 'correct_map_integrity_ratio' if col == 'map_integrity_ratio' else col
        col_tuning = 'correct_map_integrity_ratio' if col == 'map_integrity_ratio' else col
        
        # Check if the correct column exists, otherwise use original
        if col == 'map_integrity_ratio':
            if 'correct_map_integrity_ratio' not in df_default.columns:
                col_default = col
            if 'correct_map_integrity_ratio' not in df_default_02.columns:
                col_default_02 = col
            if 'correct_map_integrity_ratio' not in df_tuning.columns:
                col_tuning = col
        
        # Plot default data (add 1 to indices to show 1-30 instead of 0-29)
        ax.plot(row_indices_default + 1, df_default[col_default], linewidth=2,
                markersize=6, alpha=0.7, color='red', label='Default')
        ax.plot(row_indices_default_02 + 1, df_default_02[col_default_02], linewidth=2,
                markersize=6, alpha=0.7, color='green', label='Default_02')
        # Plot tuning data (add 1 to indices to show 1-30 instead of 0-29)
        ax.plot(row_indices_tuning + 1, df_tuning[col_tuning], linewidth=3,
                markersize=6, alpha=0.7, color='blue', label='Tuning')
        # Add horizontal lines for mean values
        mean_default = df_default[col_default].mean()
        mean_default_02 = df_default_02[col_default_02].mean()
        mean_tuning = df_tuning[col_tuning].mean()
        ax.axhline(mean_default, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axhline(mean_default_02, color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.axhline(mean_tuning, color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Run number', fontsize=15)
        ax.set_ylabel(col, fontsize=15)
        
        # Add units to title based on column name
        title = col
        if 'position' in col.lower() and 'rmse' in col.lower():
            title = f"Position RMSE [m]"
            t = ax.text(0.30, 0.96, title, transform=ax.transAxes, fontsize=17, 
            fontweight='bold', verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            title_texts.append(t)
        elif 'yaw' in col.lower() and 'rmse' in col.lower():
            title = f"Yaw RMSE [deg]"
            t = ax.text(0.30, 0.96, title, transform=ax.transAxes, fontsize=17, 
            fontweight='bold', verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            title_texts.append(t)
        elif 'position_std_dev' in col.lower():
            title = f"Position std deviation [m]"
            t = ax.text(0.25, 0.96, title, transform=ax.transAxes, fontsize=17, 
            fontweight='bold', verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            title_texts.append(t)
        elif 'yaw_std_dev' in col.lower():
            title = f"Yaw std deviation [deg]"
            t = ax.text(0.20, 0.96, title, transform=ax.transAxes, fontsize=17, 
            fontweight='bold', verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            title_texts.append(t)

        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, loc='upper right')
        # Set x-ticks to show specific values up to the largest dataset length.
        max_runs = max(len(df_default), len(df_default_02), len(df_tuning))
        x_ticks = [1] + [x for x in range(5, max_runs + 1, 5)]
        if x_ticks[-1] != max_runs:
            x_ticks.append(max_runs)
        ax.set_xticks(x_ticks)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    # Axis and title text size sliders
    axis_fontsize_init = 15
    title_fontsize_init = 17
    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.22, left=0.05, right=0.95)
    ax_slider_axis = plt.axes([0.25, 0.10, 0.5, 0.03])
    ax_slider_title = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_axis_fontsize = Slider(ax_slider_axis, "Axis text size", 6, 28, valinit=axis_fontsize_init, valstep=1)
    slider_title_fontsize = Slider(ax_slider_title, "Title text size", 6, 32, valinit=title_fontsize_init, valstep=1)

    def update_axis_fontsize(val):
        fs = int(slider_axis_fontsize.val)
        for i in range(len(numeric_cols)):
            ax = axes[i]
            ax.tick_params(axis="x", labelsize=fs)
            ax.tick_params(axis="y", labelsize=fs)
            ax.xaxis.get_label().set_fontsize(fs)
            ax.yaxis.get_label().set_fontsize(fs)
        fig.canvas.draw_idle()

    def update_title_fontsize(val):
        fs = int(slider_title_fontsize.val)
        for t in title_texts:
            t.set_fontsize(fs)
        fig.canvas.draw_idle()

    slider_axis_fontsize.on_changed(update_axis_fontsize)
    slider_title_fontsize.on_changed(update_title_fontsize)
    update_axis_fontsize(axis_fontsize_init)
    update_title_fontsize(title_fontsize_init)

    # Save the plot
    output_path = f'/home/{os.getenv("USER")}/data_analysis/line_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the line plot
    plt.show()

if __name__ == '__main__':
    main()
