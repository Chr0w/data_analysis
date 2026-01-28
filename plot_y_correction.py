#!/usr/bin/env python3
"""
Script to plot Y correction (perpendicular change component).
Reads a percentage of N files every M'th sample, uses AMCL yaw for the last 10 entries
to identify the direction of travel, and plots the signed magnitude of the AMCL position
change in meters on the axis that is perpendicular to the yaw.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import os
import sys
import argparse
from data_loader import load_data, calculate_position_error

def calculate_travel_direction(amcl_yaws):
    """
    Calculate the direction of travel from the last N yaw values.
    Uses the mean of the yaw angles to determine travel direction.
    
    Parameters:
    -----------
    amcl_yaws : list or array
        List of AMCL yaw angles (in degrees)
    
    Returns:
    --------
    mean_yaw : float
        Mean yaw angle in degrees (direction of travel)
    """
    # Convert to numpy array and handle angle wrapping
    yaws = np.array(amcl_yaws)
    
    # Convert to radians for circular mean calculation
    yaws_rad = np.deg2rad(yaws)
    
    # Calculate circular mean (handles angle wrapping)
    mean_sin = np.mean(np.sin(yaws_rad))
    mean_cos = np.mean(np.cos(yaws_rad))
    mean_yaw_rad = np.arctan2(mean_sin, mean_cos)
    
    # Convert back to degrees
    mean_yaw = np.rad2deg(mean_yaw_rad)
    
    # Normalize to [0, 360) range
    mean_yaw = mean_yaw % 360
    
    return mean_yaw

def calculate_perpendicular_change(amcl_x_prev, amcl_y_prev, amcl_x_curr, amcl_y_curr, travel_yaw_deg):
    """
    Calculate the signed magnitude of the AMCL position change on the axis
    perpendicular to the travel direction.
    
    Parameters:
    -----------
    amcl_x_prev, amcl_y_prev : float
        Previous AMCL position
    amcl_x_curr, amcl_y_curr : float
        Current AMCL position
    travel_yaw_deg : float
        Direction of travel (yaw angle in degrees)
    
    Returns:
    --------
    perpendicular_change : float
        Signed magnitude of change in meters on the perpendicular axis
        Positive = movement to the left of travel direction
        Negative = movement to the right of travel direction
    """
    # Calculate position change vector (AMCL reported change)
    change_x = amcl_x_curr - amcl_x_prev
    change_y = amcl_y_curr - amcl_y_prev
    
    # Convert travel direction to radians
    travel_yaw_rad = np.deg2rad(travel_yaw_deg)
    
    # Calculate the perpendicular direction (90 degrees to the left of travel direction)
    # In standard coordinates: if traveling at angle θ, perpendicular is at θ + 90°
    # Unit vector in perpendicular direction
    perp_angle_rad = travel_yaw_rad + np.pi / 2
    perp_unit_x = np.cos(perp_angle_rad)
    perp_unit_y = np.sin(perp_angle_rad)
    
    # Project change vector onto perpendicular axis
    perpendicular_change = change_x * perp_unit_x + change_y * perp_unit_y
    
    return perpendicular_change

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Plot Y correction (perpendicular error component)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_y_correction.py threshold
  python plot_y_correction.py percentage
        '''
    )
    parser.add_argument('mode', choices=['threshold', 'percentage'],
                       help='Stopping mode: "threshold" stops when error >= threshold, "percentage" reads specified percentage of file')
    args = parser.parse_args()
    
    # Configuration parameters
    N = 30  # Number of files to read (starting with 1)
    M = 5  # Sample every M'th entry
    error_threshold = 0.5  # Position error threshold in meters (for threshold mode)
    read_file_percentage = 0.75  # Percentage of files to read
    num_yaw_samples = 10  # Number of previous yaw values to use for travel direction
    
    # Path to the folder containing CSV files
    data_folder = '/home/mircrda/pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl'
    
    # Load data using shared function
    print("Loading data...")
    combined_df = load_data(
        data_folder, 
        N, 
        mode=args.mode,
        error_threshold=error_threshold,
        read_file_percentage=read_file_percentage
    )
    
    # Check for required columns
    required_cols = ['amcl_x', 'amcl_y', 'amcl_yaw']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(combined_df.columns)}")
        sys.exit(1)
    
    # Check for ground truth columns (needed for error calculation)
    has_ground_truth = 'ground_truth_x' in combined_df.columns and 'ground_truth_y' in combined_df.columns
    if not has_ground_truth:
        print("Warning: Ground truth columns not found. Cannot identify lost runs.")
        print(f"Available columns: {list(combined_df.columns)}")
    
    # Check for timestamp column
    if 'timestamp' not in combined_df.columns:
        print("Warning: 'timestamp' column not found. Using sample indices instead.")
        use_timestamp = False
    else:
        use_timestamp = True
        # Normalize timestamps per file (start from 0 for each file)
        combined_df = combined_df.copy()
        combined_df['time_normalized'] = 0.0
        for file_id in combined_df['file_id'].unique():
            file_mask = combined_df['file_id'] == file_id
            file_timestamps = pd.to_numeric(combined_df.loc[file_mask, 'timestamp'], errors='coerce')
            if file_timestamps.notna().any():
                first_timestamp = file_timestamps[file_timestamps.notna()].iloc[0]
                combined_df.loc[file_mask, 'time_normalized'] = file_timestamps - first_timestamp
    
    # Sample every M'th entry and calculate perpendicular change
    perpendicular_changes = []
    times = []
    travel_directions = []
    file_ids = []
    position_errors = []  # Track position errors to identify lost runs
    
    print(f"Sampling every {M}'th entry and calculating perpendicular change...")
    print(f"Using last {num_yaw_samples} AMCL yaw values to determine travel direction...")
    
    prev_amcl_x = None
    prev_amcl_y = None
    prev_idx = None
    prev_file_id = None
    
    for idx in range(0, len(combined_df), M):
        # Need at least num_yaw_samples previous entries to calculate travel direction
        if idx < num_yaw_samples:
            # Store first position for next iteration
            row = combined_df.iloc[idx]
            prev_amcl_x = row['amcl_x']
            prev_amcl_y = row['amcl_y']
            prev_file_id = row['file_id']
            prev_idx = idx
            continue
        
        # Get the last num_yaw_samples AMCL yaw values (including current)
        yaw_start_idx = max(0, idx - num_yaw_samples + 1)
        amcl_yaws = combined_df.iloc[yaw_start_idx:idx+1]['amcl_yaw'].values
        
        # Calculate travel direction from these yaw values
        travel_yaw = calculate_travel_direction(amcl_yaws)
        
        # Get current AMCL position
        row = combined_df.iloc[idx]
        amcl_x_curr = row['amcl_x']
        amcl_y_curr = row['amcl_y']
        current_file_id = row['file_id']
        
        # Get time for this sample
        if use_timestamp:
            current_time = row['time_normalized']
        else:
            current_time = idx
        
        # Calculate position error if ground truth is available
        current_position_error = None
        if has_ground_truth:
            gt_x = row['ground_truth_x']
            gt_y = row['ground_truth_y']
            current_position_error = calculate_position_error(gt_x, gt_y, amcl_x_curr, amcl_y_curr)
        
        # Reset previous position when switching files
        if prev_file_id is not None and current_file_id != prev_file_id:
            prev_amcl_x = None
            prev_amcl_y = None
        
        # Calculate perpendicular change from previous sample
        if prev_amcl_x is not None and prev_file_id == current_file_id:
            perp_change = calculate_perpendicular_change(
                prev_amcl_x, prev_amcl_y, amcl_x_curr, amcl_y_curr, travel_yaw
            )
            
            perpendicular_changes.append(perp_change)
            times.append(current_time)
            travel_directions.append(travel_yaw)
            file_ids.append(current_file_id)
            position_errors.append(current_position_error if current_position_error is not None else 0.0)
        
        # Update previous position for next iteration
        prev_amcl_x = amcl_x_curr
        prev_amcl_y = amcl_y_curr
        prev_file_id = current_file_id
        prev_idx = idx
        
        if (len(perpendicular_changes) % 50 == 0):
            print(f"  Processed {len(perpendicular_changes)} samples...")
    
    print(f"Total samples: {len(perpendicular_changes)}")
    
    # Identify lost runs and track first loss time for each run
    lost_runs = set()
    first_loss_time = {}  # file_id -> first time when error > threshold
    if has_ground_truth:
        print(f"\nIdentifying lost runs (error threshold: {error_threshold}m)...")
        for i, (file_id, pos_error, time_val) in enumerate(zip(file_ids, position_errors, times)):
            if pos_error is not None and pos_error > error_threshold:
                lost_runs.add(file_id)
                # Track the first time this file becomes lost
                if file_id not in first_loss_time:
                    first_loss_time[file_id] = time_val
                    print(f"File {file_id} first lost at time {time_val:.2f}s (error={pos_error:.4f}m)")
        if lost_runs:
            print(f"Lost runs (files that exceed error threshold): {sorted(lost_runs)}")
        else:
            print("No lost runs found (all runs stayed below error threshold)")
        print(f"Total lost runs: {len(lost_runs)} out of {len(set(file_ids))} runs")
    else:
        print("Skipping lost run identification (no ground truth data)")
    
    # Create title suffix based on mode
    if args.mode == 'threshold':
        title_suffix = f'(N={N} files, sampling every {M}\'th, error threshold={error_threshold}m)'
    else:
        title_suffix = f'(N={N} files, sampling every {M}\'th, {read_file_percentage*100:.0f}% of each file)'
    
    # Create figure with space for slider
    print("\nCreating plot...")
    fig = plt.figure(figsize=(12, 7))
    ax = plt.subplot(111)
    plt.subplots_adjust(bottom=0.15)  # Make room for slider
    
    # Get unique file IDs and assign colors
    unique_file_ids = sorted(set(file_ids))
    num_files = len(unique_file_ids)
    
    # Use a colormap to assign different colors to each file
    cmap = plt.cm.get_cmap('tab20')  # Use tab20 colormap for up to 20 distinct colors
    if num_files > 20:
        cmap = plt.cm.get_cmap('tab20c')  # Use tab20c for more files
    
    # Create color mapping
    file_id_to_color = {fid: cmap(i / max(1, num_files - 1)) for i, fid in enumerate(unique_file_ids)}
    
    # Store plot objects for each file so we can show/hide them
    plot_objects = {}  # file_id -> {'scatter': scatter_obj, 'line': line_obj}
    
    # Prepare data for each file
    for file_id in unique_file_ids:
        # Get data for this file
        file_mask = [fid == file_id for fid in file_ids]
        file_times = [times[i] for i in range(len(times)) if file_mask[i]]
        file_changes = [perpendicular_changes[i] for i in range(len(perpendicular_changes)) if file_mask[i]]
        
        if len(file_times) > 0:
            # Check if this run becomes lost and when
            is_lost = file_id in lost_runs
            loss_time = first_loss_time.get(file_id, None)
            
            # Split data into before and after loss
            if is_lost and loss_time is not None:
                # Split points into before and after first loss
                before_mask = [t < loss_time for t in file_times]
                after_mask = [t >= loss_time for t in file_times]
                
                before_times = [file_times[i] for i in range(len(file_times)) if before_mask[i]]
                before_changes = [file_changes[i] for i in range(len(file_changes)) if before_mask[i]]
                after_times = [file_times[i] for i in range(len(file_times)) if after_mask[i]]
                after_changes = [file_changes[i] for i in range(len(file_changes)) if after_mask[i]]
                
                # Normal color for before loss
                normal_color = file_id_to_color[file_id]
                scatter_before = ax.scatter(before_times, before_changes, 
                          s=20, alpha=0.6, color=normal_color, 
                          edgecolors='black', linewidth=0.3,
                          visible=False)
                line_before, = ax.plot(before_times, before_changes, 
                       linewidth=0.5, alpha=0.3, 
                       color=normal_color, visible=False)
                
                # Bright green for after loss
                scatter_after = ax.scatter(after_times, after_changes, 
                          s=20, alpha=0.8, color='lime', 
                          edgecolors='darkgreen', linewidth=0.5,
                          zorder=10, visible=False)
                line_after, = ax.plot(after_times, after_changes, 
                       linewidth=1.0, alpha=0.5, 
                       color='green', zorder=9, visible=False)
                
                label = f'File {file_id} (LOST at {loss_time:.1f}s)'
                plot_objects[file_id] = {
                    'scatter_before': scatter_before, 
                    'line_before': line_before,
                    'scatter_after': scatter_after,
                    'line_after': line_after,
                    'label': label
                }
            else:
                # Normal plotting for runs that never get lost
                color = file_id_to_color[file_id]
                label = f'File {file_id}'
                
                scatter = ax.scatter(file_times, file_changes, 
                          s=20, alpha=0.6, color=color, 
                          edgecolors='black', linewidth=0.3,
                          visible=False)
                line, = ax.plot(file_times, file_changes, 
                       linewidth=0.5, alpha=0.3, 
                       color=color, visible=False)
                
                plot_objects[file_id] = {
                    'scatter': scatter, 
                    'line': line, 
                    'label': label
                }
    
    # Show only the first file initially
    if unique_file_ids:
        first_file_id = unique_file_ids[0]
        obj = plot_objects[first_file_id]
        # Handle both cases: split (lost) or single (not lost)
        if 'scatter_before' in obj:
            obj['scatter_before'].set_visible(True)
            obj['line_before'].set_visible(True)
            obj['scatter_after'].set_visible(True)
            obj['line_after'].set_visible(True)
        else:
            obj['scatter'].set_visible(True)
            obj['line'].set_visible(True)
        current_label = obj['label']
    else:
        current_label = "No data"
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    if use_timestamp:
        ax.set_xlabel('Time [s]', fontsize=12)
    else:
        ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Perpendicular Change [m]', fontsize=12)
    ax.set_title(f'Y Correction: Perpendicular Change Component (AMCL)\n{title_suffix}\nRun: {current_label}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add statistics (will be updated by slider)
    stats_text_obj = None
    if len(perpendicular_changes) > 0:
        stats_text = f'Mean: {np.mean(perpendicular_changes):.4f} m\nStd: {np.std(perpendicular_changes):.4f} m\nRange: [{np.min(perpendicular_changes):.4f}, {np.max(perpendicular_changes):.4f}] m'
        stats_text_obj = ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Create slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Run', 1, num_files, valinit=1, valstep=1, valfmt='%d')
    
    # Update function for slider
    def update(val):
        run_index = int(slider.val) - 1  # Convert to 0-based index
        if 0 <= run_index < len(unique_file_ids):
            selected_file_id = unique_file_ids[run_index]
            
            # Hide all plots
            for file_id, objs in plot_objects.items():
                if 'scatter_before' in objs:
                    # Split plot (lost run)
                    objs['scatter_before'].set_visible(False)
                    objs['line_before'].set_visible(False)
                    objs['scatter_after'].set_visible(False)
                    objs['line_after'].set_visible(False)
                else:
                    # Single plot (not lost)
                    objs['scatter'].set_visible(False)
                    objs['line'].set_visible(False)
            
            # Show selected run
            obj = plot_objects[selected_file_id]
            if 'scatter_before' in obj:
                # Split plot (lost run)
                obj['scatter_before'].set_visible(True)
                obj['line_before'].set_visible(True)
                obj['scatter_after'].set_visible(True)
                obj['line_after'].set_visible(True)
            else:
                # Single plot (not lost)
                obj['scatter'].set_visible(True)
                obj['line'].set_visible(True)
            
            # Update title
            current_label = obj['label']
            ax.set_title(f'Y Correction: Perpendicular Change Component (AMCL)\n{title_suffix}\nRun: {current_label}', fontsize=14)
            
            # Update statistics for selected run
            if stats_text_obj is not None:
                file_mask = [fid == selected_file_id for fid in file_ids]
                selected_changes = [perpendicular_changes[i] for i in range(len(perpendicular_changes)) if file_mask[i]]
                if len(selected_changes) > 0:
                    mean_change = np.mean(selected_changes)
                    std_change = np.std(selected_changes)
                    max_change = np.max(selected_changes)
                    min_change = np.min(selected_changes)
                    stats_text = f'Mean: {mean_change:.4f} m\nStd: {std_change:.4f} m\nRange: [{min_change:.4f}, {max_change:.4f}] m'
                    stats_text_obj.set_text(stats_text)
            
            fig.canvas.draw_idle()
    
    # Connect slider to update function
    slider.on_changed(update)
    
    # Save the plot (will save with first run visible)
    output_path = os.path.join(data_folder, 'y_correction_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    print("Use the slider at the bottom to select which run to display.")
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(f"Total samples: {len(perpendicular_changes)}")
    if len(perpendicular_changes) > 0:
        print(f"Mean perpendicular change: {mean_change:.6f} m")
        print(f"Std deviation: {std_change:.6f} m")
        print(f"Min change: {min_change:.6f} m")
        print(f"Max change: {max_change:.6f} m")
        print(f"Range: [{min_change:.6f}, {max_change:.6f}] m")
    print("="*60)

if __name__ == '__main__':
    main()
