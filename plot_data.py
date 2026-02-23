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
from data_loader import calculate_position_error

# Thresholds
POSITION_THRESHOLD = 0.5  # meters
YAW_THRESHOLD = 15.0  # degrees


def calculate_yaw_error(gt_yaw, amcl_yaw):
    """Absolute yaw error in degrees (handles wrapping)."""
    diff = gt_yaw - amcl_yaw
    diff = ((diff + 180) % 360) - 180
    return abs(diff)

def main():
    # Paths to the CSV files
    user_home = os.path.expanduser('~')
    csv_files = [
        ('default', os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl/default_combined_results_new.csv')),
        ('default_02', os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_02/default_02_combined_results_new.csv')),
        ('default_001', os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_001/default_001_combined_results_new.csv')),
        ('tuning', os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/alpha_tuning/tuning_combined_results_new.csv')),
    ]
    
    # Read the CSV files (skip missing ones)
    dataframes = {}
    for name, csv_path in csv_files:
        if not os.path.exists(csv_path):
            print(f"Warning: File not found at {csv_path}, skipping...", file=sys.stderr)
            continue
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()
            if len(df) == 0:
                print(f"Warning: File {csv_path} is empty, skipping...", file=sys.stderr)
                continue
            dataframes[name] = df
            print(f"Loaded {len(df)} rows from {csv_path}")
            print(f"{name} columns: {list(df.columns)}")
        except Exception as e:
            print(f"Warning: Error reading {csv_path}: {e}, skipping...", file=sys.stderr)
            continue
    
    if not dataframes:
        print("Error: No valid CSV files found to plot", file=sys.stderr)
        sys.exit(1)
    
    # Assign to variables for backward compatibility (if they exist)
    df_default = dataframes.get('default')
    df_default_02 = dataframes.get('default_02')
    df_default_001 = dataframes.get('default_001')
    df_tuning = dataframes.get('tuning')
    
    # Get numeric columns from all available dataframes (exclude timestamp and specified columns)
    exclude_cols = ['timestamp', 'total_messages', 'duration_s', 'msg_rate_hz', 'ESI_range', 'ESI_std_dev', 'mean_ESI', 'position_max_error', 'yaw_max_error', 'position_mean_error', 'yaw_mean_error', 'sum_cov_trace', 'position_std_dev', 'yaw_std_dev', 'run_id', 'yaw_RMSE']
    
    all_numeric_cols = []
    for name, df in dataframes.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        all_numeric_cols.append(set(numeric_cols))

    # Get common columns to plot (intersection of all available datasets)
    if all_numeric_cols:
        numeric_cols = list(set.intersection(*all_numeric_cols))
    else:
        numeric_cols = []
    
    if not numeric_cols:
        print("Error: No common numeric columns found to plot", file=sys.stderr)
        sys.exit(1)
    
    # Sort columns for consistent ordering (position_RMSE first, then yaw_RMSE)
    numeric_cols = sorted(numeric_cols)
    
    print(f"Plotting {len(numeric_cols)} common numeric columns: {numeric_cols}")
    
    # Calculate and print mean values for position and yaw RMSE
    print("\n" + "="*60)
    print("Mean RMSE Values:")
    print("="*60)
    
    # Position RMSE
    pos_rmse_data = {}
    for name, df in dataframes.items():
        if 'position_RMSE' in df.columns:
            pos_rmse_data[name] = df['position_RMSE'].mean()
    
    if pos_rmse_data:
        print(f"Position RMSE")
        for name, mean_val in pos_rmse_data.items():
            print(f"{name}: {mean_val:.6f}")
        # Calculate reductions if tuning exists
        if 'tuning' in pos_rmse_data:
            tuning_val = pos_rmse_data['tuning']
            for name, mean_val in pos_rmse_data.items():
                if name != 'tuning':
                    reduction = ((mean_val - tuning_val) / mean_val) * 100
                    print(f"Reduction vs {name}: {reduction:.2f}%")
    else:
        print("Position RMSE column not found in any dataset")
    
    print()
    
    # Yaw RMSE
    yaw_rmse_data = {}
    for name, df in dataframes.items():
        if 'yaw_RMSE' in df.columns:
            yaw_rmse_data[name] = df['yaw_RMSE'].mean()
    
    if yaw_rmse_data:
        print(f"Yaw RMSE")
        for name, mean_val in yaw_rmse_data.items():
            print(f"{name}: {mean_val:.6f}")
        # Calculate reductions if tuning exists
        if 'tuning' in yaw_rmse_data:
            tuning_val = yaw_rmse_data['tuning']
            for name, mean_val in yaw_rmse_data.items():
                if name != 'tuning':
                    reduction = ((mean_val - tuning_val) / mean_val) * 100
                    print(f"Reduction vs {name}: {reduction:.2f}%")
    else:
        print("Yaw RMSE column not found in any dataset")
    
    print("="*60 + "\n")
    
    # Calculate additional metrics from individual run files
    print("\n" + "="*60)
    print("Threshold-Based Metrics:")
    print(f"Position threshold: {POSITION_THRESHOLD} m")
    print(f"Yaw threshold: {YAW_THRESHOLD} deg")
    print("="*60)
    
    # Map mode names to folder paths
    mode_folders = {
        'default': os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl'),
        'default_02': os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_02'),
        'default_001': os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_001'),
        'tuning': os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/alpha_tuning'),
    }
    
    # Calculate metrics for each available mode
    for mode_name, folder_path in mode_folders.items():
        if mode_name not in dataframes:
            continue  # Skip if combined results don't exist
        
        if not os.path.isdir(folder_path):
            print(f"\n{mode_name}: Folder not found, skipping threshold metrics...")
            continue
        
        metrics = calculate_threshold_metrics(folder_path, POSITION_THRESHOLD, YAW_THRESHOLD)
        if metrics is None:
            print(f"\n{mode_name}: No valid run files found")
            continue
        
        print(f"\n{mode_name}:")
        print(f"  Average time below threshold (both pos and yaw): {metrics['avg_time_below']:.2f} s")
        print(f"  Average time below threshold until first exceed (per run): {metrics['avg_time_until_exceed']:.2f} s")
        pos_rmse = metrics['avg_position_RMSE_until_exceed']
        yaw_rmse = metrics['avg_yaw_RMSE_until_exceed']
        if not np.isnan(pos_rmse):
            print(f"  Average position RMSE until first exceed: {pos_rmse:.6f} m")
        if not np.isnan(yaw_rmse):
            print(f"  Average yaw RMSE until first exceed: {yaw_rmse:.6f} deg")
        print(f"  Total runs analyzed: {metrics['total_runs']}")
    
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
    
    # Create row indices for available dataframes
    row_indices = {}
    for name, df in dataframes.items():
        row_indices[name] = df.index.values
    
    # Collect title text objects for the title-size slider
    title_texts = []

    # Define colors and labels for each dataset
    dataset_config = {
        'default': {'color': 'red', 'label': 'Default', 'linewidth': 2},
        'default_02': {'color': 'green', 'label': 'Default_02', 'linewidth': 2},
        'default_001': {'color': 'yellow', 'label': 'Default_001', 'linewidth': 2},
        'tuning': {'color': 'blue', 'label': 'Tuning', 'linewidth': 3},
    }
    
    # Plot each numeric column
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        
        # Plot data for each available dataframe
        for name, df in dataframes.items():
            # Use correct_map_integrity_ratio if col is map_integrity_ratio
            plot_col = 'correct_map_integrity_ratio' if col == 'map_integrity_ratio' else col
            
            # Check if the correct column exists, otherwise use original
            if col == 'map_integrity_ratio' and 'correct_map_integrity_ratio' not in df.columns:
                plot_col = col
            
            if plot_col not in df.columns:
                continue
                
            config = dataset_config.get(name, {'color': 'gray', 'label': name, 'linewidth': 2})
            ax.plot(row_indices[name] + 1, df[plot_col], linewidth=config['linewidth'],
                    markersize=6, alpha=0.7, color=config['color'], label=config['label'])
            
            # Add horizontal line for mean value
            mean_val = df[plot_col].mean()
            ax.axhline(mean_val, color=config['color'], linestyle='--', linewidth=1.5, alpha=0.8)
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
        max_runs = max(len(df) for df in dataframes.values()) if dataframes else 1
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

def calculate_threshold_metrics(data_folder: str, pos_threshold: float, yaw_threshold: float, n_files: int = 30, skip_rows: int = 100):
    """
    Calculate threshold-based metrics from individual run CSV files.
    
    Returns:
        dict with:
        - runs_outside_threshold: number of runs where final entry is outside threshold
        - avg_time_below: average time (per run) where both pos and yaw are below threshold
        - avg_time_until_exceed: average time (per run) below threshold until first exceed
        - avg_position_RMSE_until_exceed: average position RMSE (per run) up until first threshold exceed
        - avg_yaw_RMSE_until_exceed: average yaw RMSE (per run) up until first threshold exceed
        - total_runs: total number of runs analyzed
    """
    runs_outside = 0
    times_below = []
    times_until_exceed = []
    position_rmse_until_exceed = []
    yaw_rmse_until_exceed = []
    total_runs = 0
    
    for run_id in range(1, n_files + 1):
        # Try both naming conventions
        csv_path = os.path.join(data_folder, f"{run_id}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(data_folder, f"{run_id:02d}.csv")
        if not os.path.exists(csv_path):
            continue
        
        try:
            df = pd.read_csv(csv_path, skiprows=range(1, skip_rows))
            df.columns = df.columns.str.strip()
            
            required = ["timestamp", "ground_truth_x", "ground_truth_y", "amcl_x", "amcl_y"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                continue
            
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            if len(df) == 0:
                continue
            
            # Calculate position and yaw errors for each row
            pos_errors = []
            yaw_errors = []
            has_yaw = "ground_truth_yaw" in df.columns and "amcl_yaw" in df.columns
            
            for _, row in df.iterrows():
                pos_err = calculate_position_error(
                    row["ground_truth_x"], row["ground_truth_y"],
                    row["amcl_x"], row["amcl_y"]
                )
                pos_errors.append(pos_err)
                
                if has_yaw:
                    yaw_err = calculate_yaw_error(row["ground_truth_yaw"], row["amcl_yaw"])
                    yaw_errors.append(yaw_err)
                else:
                    yaw_errors.append(0.0)  # Assume perfect yaw if not available
            
            pos_errors = np.array(pos_errors)
            yaw_errors = np.array(yaw_errors)
            
            # Check if final entry is outside threshold
            final_pos = pos_errors[-1]
            final_yaw = yaw_errors[-1]
            if final_pos > pos_threshold or final_yaw > yaw_threshold:
                runs_outside += 1
            
            # Calculate time intervals
            timestamps = df["timestamp"].values
            if len(timestamps) < 2:
                continue
            
            # Total time below threshold (both pos and yaw) - across entire run
            time_below = 0.0
            # Time below threshold until first exceed - per run
            time_until_exceed = 0.0
            exceeded = False
            exceed_index = len(pos_errors)  # Index where threshold is first exceeded (default to end if never exceeded)
            
            # Check if first point already exceeds threshold
            if pos_errors[0] >= pos_threshold or yaw_errors[0] >= yaw_threshold:
                exceeded = True
                exceed_index = 0
            
            for i in range(len(timestamps) - 1):
                dt = timestamps[i + 1] - timestamps[i]
                if dt <= 0:
                    continue
                
                # Check if both pos and yaw are below threshold at the start of this interval
                pos_below = pos_errors[i] < pos_threshold
                yaw_below = yaw_errors[i] < yaw_threshold
                both_below = pos_below and yaw_below
                
                # Add to total time below if both are below (for entire run)
                if both_below:
                    time_below += dt
                
                # Track time until first exceed: add interval if not exceeded yet and both are below
                if not exceeded and both_below:
                    time_until_exceed += dt
                
                # Check if threshold is exceeded at the end of this interval
                if not exceeded and i + 1 < len(pos_errors):
                    pos_exceeded = pos_errors[i + 1] >= pos_threshold
                    yaw_exceeded = yaw_errors[i + 1] >= yaw_threshold
                    if pos_exceeded or yaw_exceeded:
                        exceeded = True
                        exceed_index = i + 1
            
            # Calculate RMSE up until first threshold exceed
            if exceed_index > 0:
                pos_errors_until_exceed = pos_errors[:exceed_index]
                if len(pos_errors_until_exceed) > 0:
                    position_rmse = float(np.sqrt(np.mean(pos_errors_until_exceed**2)))
                    position_rmse_until_exceed.append(position_rmse)
                
                if has_yaw:
                    yaw_errors_until_exceed = yaw_errors[:exceed_index]
                    yaw_finite = yaw_errors_until_exceed[np.isfinite(yaw_errors_until_exceed)]
                    if len(yaw_finite) > 0:
                        yaw_rmse = float(np.sqrt(np.mean(yaw_finite**2)))
                        yaw_rmse_until_exceed.append(yaw_rmse)
            
            times_below.append(time_below)
            times_until_exceed.append(time_until_exceed)
            total_runs += 1
            
        except Exception as e:
            print(f"Warning: Error processing {csv_path}: {e}", file=sys.stderr)
            continue
    
    if total_runs == 0:
        return None
    
    avg_time_below = np.mean(times_below) if times_below else 0.0
    avg_time_until_exceed = np.mean(times_until_exceed) if times_until_exceed else 0.0
    avg_position_rmse = np.mean(position_rmse_until_exceed) if position_rmse_until_exceed else np.nan
    avg_yaw_rmse = np.mean(yaw_rmse_until_exceed) if yaw_rmse_until_exceed else np.nan
    
    return {
        'runs_outside_threshold': runs_outside,
        'avg_time_below': avg_time_below,
        'avg_time_until_exceed': avg_time_until_exceed,
        'avg_position_RMSE_until_exceed': avg_position_rmse,
        'avg_yaw_RMSE_until_exceed': avg_yaw_rmse,
        'total_runs': total_runs,
    }


if __name__ == '__main__':
    main()
