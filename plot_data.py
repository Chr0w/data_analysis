#!/usr/bin/env python3
"""
Script to plot data from CSV files as line plots.
Creates one subplot for each numeric column, using row index (line number) as x-axis.
Plots data from both default and tuning CSV files for comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math

def main():
    # Paths to the CSV files
    user_home = os.path.expanduser('~')
    default_csv_path = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl/default_combined_results.csv')
    tuning_csv_path = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/alpha_tuning/tuning_combined_results.csv')
    
    # Check if files exist
    if not os.path.exists(default_csv_path):
        print(f"Error: File not found at {default_csv_path}")
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
        
        df_tuning = pd.read_csv(tuning_csv_path)
        df_tuning.columns = df_tuning.columns.str.strip()
        print(f"Loaded {len(df_tuning)} rows from {tuning_csv_path}")
        print(f"Tuning columns: {list(df_tuning.columns)}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Get numeric columns from both dataframes (exclude timestamp and specified columns)
    numeric_cols_default = df_default.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_tuning = df_tuning.select_dtypes(include=[np.number]).columns.tolist()
    
    # Columns to exclude (including std deviation - only plot RMSE)
    exclude_cols = ['timestamp', 'total_messages', 'duration_s', 'msg_rate_hz', 'ESI_range', 'ESI_std_dev', 'mean_ESI', 'position_max_error', 'yaw_max_error', 'position_mean_error', 'yaw_mean_error', 'sum_cov_trace', 'position_std_dev', 'yaw_std_dev']
    numeric_cols_default = [col for col in numeric_cols_default if col not in exclude_cols]
    numeric_cols_tuning = [col for col in numeric_cols_tuning if col not in exclude_cols]
    
    # Get common columns to plot
    numeric_cols = list(set(numeric_cols_default) & set(numeric_cols_tuning))
    
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
    
    if 'position_RMSE' in df_default.columns and 'position_RMSE' in df_tuning.columns:
        mean_pos_rmse_default = df_default['position_RMSE'].mean()
        mean_pos_rmse_tuning = df_tuning['position_RMSE'].mean()
        print(f"Position RMSE")
        print(f"Default: {mean_pos_rmse_default:.6f}")
        print(f"Tuning:  {mean_pos_rmse_tuning:.6f}")
        reduction_pes = ((mean_pos_rmse_default - mean_pos_rmse_tuning) / mean_pos_rmse_default) * 100
        print(f"Reduction: {reduction_pes:.2f}%")

    else:
        print("Position RMSE column not found in one or both datasets")
    
    print()
    
    if 'yaw_RMSE' in df_default.columns and 'yaw_RMSE' in df_tuning.columns:
        print(f"Yaw RMSE")
        mean_yaw_rmse_default = df_default['yaw_RMSE'].mean()
        mean_yaw_rmse_tuning = df_tuning['yaw_RMSE'].mean()
        print(f"Default: {mean_yaw_rmse_default:.6f}")
        print(f"Tuning:  {mean_yaw_rmse_tuning:.6f}")
        reduction_yaw = ((mean_yaw_rmse_default - mean_yaw_rmse_tuning) / mean_yaw_rmse_default) * 100
        print(f"Reduction: {reduction_yaw:.2f}%")
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
    row_indices_tuning = df_tuning.index.values
    
    # Plot each numeric column
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        # Use correct_map_integrity_ratio if col is map_integrity_ratio
        col_default = 'correct_map_integrity_ratio' if col == 'map_integrity_ratio' else col
        col_tuning = 'correct_map_integrity_ratio' if col == 'map_integrity_ratio' else col
        
        # Check if the correct column exists, otherwise use original
        if col == 'map_integrity_ratio':
            if 'correct_map_integrity_ratio' not in df_default.columns:
                col_default = col
            if 'correct_map_integrity_ratio' not in df_tuning.columns:
                col_tuning = col
        
        # Plot default data (add 1 to indices to show 1-30 instead of 0-29)
        ax.plot(row_indices_default + 1, df_default[col_default], linewidth=2, 
                markersize=6, alpha=0.7, label='Default', color='red')
        # Plot tuning data (add 1 to indices to show 1-30 instead of 0-29)
        ax.plot(row_indices_tuning + 1, df_tuning[col_tuning], linewidth=2, 
                markersize=6, alpha=0.7, label='Tuning', color='blue')
        # Add horizontal lines for mean values
        mean_default = df_default[col_default].mean()
        mean_tuning = df_tuning[col_tuning].mean()
        ax.axhline(mean_default, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Default mean ({mean_default:.4g})')
        ax.axhline(mean_tuning, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label=f'Tuning mean ({mean_tuning:.4g})')
        ax.set_xlabel('Run number', fontsize=15)
        ax.set_ylabel(col, fontsize=15)
        
        # Add units to title based on column name
        title = col
        if 'position' in col.lower() and 'rmse' in col.lower():
            title = f"Position RMSE [m]"
            ax.text(0.30, 0.96, title, transform=ax.transAxes, fontsize=17, 
            fontweight='bold', verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif 'yaw' in col.lower() and 'rmse' in col.lower():
            title = f"Yaw RMSE [deg]"
            ax.text(0.30, 0.96, title, transform=ax.transAxes, fontsize=17, 
            fontweight='bold', verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif 'position_std_dev' in col.lower():
            title = f"Position std deviation [m]"
            ax.text(0.25, 0.96, title, transform=ax.transAxes, fontsize=17, 
            fontweight='bold', verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif 'yaw_std_dev' in col.lower():
            title = f"Yaw std deviation [deg]"
            ax.text(0.20, 0.96, title, transform=ax.transAxes, fontsize=17, 
            fontweight='bold', verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, loc='upper right')
        # Set x-ticks to show specific values: 1, 5, 10, 15, 20, 25, 30
        x_ticks = [1, 5, 10, 15, 20, 25, 30]
        ax.set_xticks(x_ticks)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    # Save the plot
    output_path = f'/home/{os.getenv("USER")}/data_analysis/line_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the line plot
    plt.show()
    
    # Create histogram plots for RMSE values
    if 'position_RMSE' in df_default.columns and 'position_RMSE' in df_tuning.columns and \
       'yaw_RMSE' in df_default.columns and 'yaw_RMSE' in df_tuning.columns:
        
        fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig_hist.suptitle('RMSE Histograms: Default vs Tuning', fontsize=24, y=1.0)
        
        # Calculate shared bin edges for Position RMSE
        pos_rmse_min = min(df_default['position_RMSE'].min(), df_tuning['position_RMSE'].min())
        pos_rmse_max = max(df_default['position_RMSE'].max(), df_tuning['position_RMSE'].max())
        pos_rmse_bins = np.linspace(pos_rmse_min, pos_rmse_max, 21)  # 21 edges = 20 bins
        bin_width_pos = pos_rmse_bins[1] - pos_rmse_bins[0]
        
        # Calculate histogram counts for Position RMSE
        counts_default_pos, _ = np.histogram(df_default['position_RMSE'], bins=pos_rmse_bins)
        counts_tuning_pos, _ = np.histogram(df_tuning['position_RMSE'], bins=pos_rmse_bins)
        bin_centers_pos = (pos_rmse_bins[:-1] + pos_rmse_bins[1:]) / 2
        
        # Position RMSE histogram - side by side bars with default on left
        ax1.bar(bin_centers_pos - bin_width_pos/4, counts_default_pos, width=bin_width_pos/2, 
                alpha=0.7, label='Default', color='blue', edgecolor='black', linewidth=1.2)
        ax1.bar(bin_centers_pos + bin_width_pos/4, counts_tuning_pos, width=bin_width_pos/2, 
                alpha=0.7, label='Tuning', color='red', edgecolor='black', linewidth=1.2)
        ax1.set_xlabel('Position RMSE [m]', fontsize=18)
        ax1.set_ylabel('Frequency', fontsize=18)
        ax1.set_title('Position RMSE Distribution [m]', fontsize=20, fontweight='bold')
        ax1.legend(fontsize=15, loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Calculate shared bin edges for Yaw RMSE
        yaw_rmse_min = min(df_default['yaw_RMSE'].min(), df_tuning['yaw_RMSE'].min())
        yaw_rmse_max = max(df_default['yaw_RMSE'].max(), df_tuning['yaw_RMSE'].max())
        yaw_rmse_bins = np.linspace(yaw_rmse_min, yaw_rmse_max, 21)  # 21 edges = 20 bins
        bin_width_yaw = yaw_rmse_bins[1] - yaw_rmse_bins[0]
        
        # Calculate histogram counts for Yaw RMSE
        counts_default_yaw, _ = np.histogram(df_default['yaw_RMSE'], bins=yaw_rmse_bins)
        counts_tuning_yaw, _ = np.histogram(df_tuning['yaw_RMSE'], bins=yaw_rmse_bins)
        bin_centers_yaw = (yaw_rmse_bins[:-1] + yaw_rmse_bins[1:]) / 2
        
        # Yaw RMSE histogram - side by side bars with default on left
        ax2.bar(bin_centers_yaw - bin_width_yaw/4, counts_default_yaw, width=bin_width_yaw/2, 
                alpha=0.7, label='Default', color='blue', edgecolor='black', linewidth=1.2)
        ax2.bar(bin_centers_yaw + bin_width_yaw/4, counts_tuning_yaw, width=bin_width_yaw/2, 
                alpha=0.7, label='Tuning', color='red', edgecolor='black', linewidth=1.2)
        ax2.set_xlabel('Yaw RMSE [deg]', fontsize=18)
        ax2.set_ylabel('Frequency', fontsize=18)
        ax2.set_title('Yaw RMSE Distribution [deg]', fontsize=20, fontweight='bold')
        ax2.legend(fontsize=15, loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save the histogram plot
        hist_output_path = f'/home/{os.getenv("USER")}/data_analysis/rmse_histograms.png'
        fig_hist.savefig(hist_output_path, dpi=300, bbox_inches='tight')
        print(f"Histogram plot saved to {hist_output_path}")
        
        # Show the histogram plot
        plt.show()

if __name__ == '__main__':
    main()
