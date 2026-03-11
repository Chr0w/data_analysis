#!/usr/bin/env python3
"""
Script to plot individual run data for pose and yaw errors across multiple modes.
Has a slider to select which run (1-30) to display.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from matplotlib.widgets import Slider
from data_loader import calculate_position_error

def calculate_yaw_error(gt_yaw, amcl_yaw):
    """Absolute yaw error in degrees (handles wrapping)."""
    diff = gt_yaw - amcl_yaw
    diff = ((diff + 180) % 360) - 180
    return abs(diff)

def load_run_data(folder_path, run_id, skip_rows=100):
    """Load data for a specific run from a folder."""
    # Try both naming conventions
    csv_path = os.path.join(folder_path, f"{run_id}.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(folder_path, f"{run_id:02d}.csv")
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path, skiprows=range(1, skip_rows))
        df.columns = df.columns.str.strip()
        
        required = ["timestamp", "ground_truth_x", "ground_truth_y", "amcl_x", "amcl_y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return None
        
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        
        if len(df) == 0:
            return None
        
        # Calculate position and yaw errors
        pos_errors = []
        yaw_errors = []
        timestamps = []
        has_yaw = "ground_truth_yaw" in df.columns and "amcl_yaw" in df.columns
        
        for _, row in df.iterrows():
            pos_err = calculate_position_error(
                row["ground_truth_x"], row["ground_truth_y"],
                row["amcl_x"], row["amcl_y"]
            )
            pos_errors.append(pos_err)
            timestamps.append(row["timestamp"])
            
            if has_yaw:
                yaw_err = calculate_yaw_error(row["ground_truth_yaw"], row["amcl_yaw"])
                yaw_errors.append(yaw_err)
            else:
                yaw_errors.append(0.0)
        
        return {
            'timestamps': np.array(timestamps),
            'pos_errors': np.array(pos_errors),
            'yaw_errors': np.array(yaw_errors),
            'has_yaw': has_yaw
        }
    except Exception as e:
        print(f"Error loading {csv_path}: {e}", file=sys.stderr)
        return None

def main():
    user_home = os.path.expanduser('~')
    
    # Define the modes to plot
    mode_folders = {
        'default': os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl'),
        'default_02': os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_02'),
        'default_001': os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_001'),
        'tuning': os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/alpha_tuning'),
    }
    
    mode_labels = {
        'default': 'Default',
        'default_02': 'Default_02',
        'default_001': 'Default_001',
        'tuning': 'Tuning',
    }
    
    mode_colors = {
        'default': 'blue',
        'default_02': 'green',
        'default_001': 'red',
        'tuning': 'orange',
    }
    
    # Thresholds for reference lines
    POSITION_THRESHOLD = 1.0  # meters
    YAW_THRESHOLD = 45.0  # degrees
    
    # Create figure with 2 subplots (position error and yaw error)
    fig, (ax_pos, ax_yaw) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Pose and Yaw Errors for Individual Run', fontsize=14, fontweight='bold')
    
    # Store plot lines for each mode
    pos_lines = {}
    yaw_lines = {}
    
    # Initial run ID
    initial_run = 1
    
    def update_plot(run_id):
        """Update the plot with data for the selected run."""
        # Clear previous data
        ax_pos.clear()
        ax_yaw.clear()
        
        # Load and plot data for each mode
        for mode_name, folder_path in mode_folders.items():
            if not os.path.isdir(folder_path):
                continue
            
            data = load_run_data(folder_path, run_id)
            if data is None:
                continue
            
            timestamps = data['timestamps']
            pos_errors = data['pos_errors']
            yaw_errors = data['yaw_errors']
            
            # Normalize timestamps to start from 0
            if len(timestamps) > 0:
                timestamps_norm = timestamps - timestamps[0]
            else:
                timestamps_norm = timestamps
            
            # Plot position error
            line_pos, = ax_pos.plot(timestamps_norm, pos_errors, 
                                   color=mode_colors[mode_name],
                                   label=mode_labels[mode_name],
                                   linewidth=1.5)
            pos_lines[mode_name] = line_pos
            
            # Plot yaw error
            if data['has_yaw']:
                line_yaw, = ax_yaw.plot(timestamps_norm, yaw_errors,
                                      color=mode_colors[mode_name],
                                      label=mode_labels[mode_name],
                                      linewidth=1.5)
                yaw_lines[mode_name] = line_yaw
        
        # Add threshold lines
        if len(ax_pos.lines) > 0:
            ax_pos.axhline(y=POSITION_THRESHOLD, color='red', linestyle='--', 
                         linewidth=1, alpha=0.7, label=f'Threshold ({POSITION_THRESHOLD} m)')
        
        if len(ax_yaw.lines) > 0:
            ax_yaw.axhline(y=YAW_THRESHOLD, color='red', linestyle='--', 
                          linewidth=1, alpha=0.7, label=f'Threshold ({YAW_THRESHOLD} deg)')
        
        # Customize position error subplot
        ax_pos.set_xlabel('Time [s]', fontsize=11)
        ax_pos.set_ylabel('Position Error [m]', fontsize=11)
        ax_pos.set_title(f'Position Error - Run {int(run_id)}', fontsize=12, fontweight='bold')
        ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(loc='best', fontsize=10)
        
        # Customize yaw error subplot
        ax_yaw.set_xlabel('Time [s]', fontsize=11)
        ax_yaw.set_ylabel('Yaw Error [deg]', fontsize=11)
        ax_yaw.set_title(f'Yaw Error - Run {int(run_id)}', fontsize=12, fontweight='bold')
        ax_yaw.grid(True, alpha=0.3)
        ax_yaw.legend(loc='best', fontsize=10)
        
        plt.tight_layout()
        fig.canvas.draw_idle()
    
    # Create slider
    plt.subplots_adjust(bottom=0.15)
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, 'Run ID', 1, 30, valinit=initial_run, valstep=1)
    
    # Update plot when slider changes
    slider.on_changed(update_plot)
    
    # Initial plot
    update_plot(initial_run)
    
    plt.show()

if __name__ == '__main__':
    main()
