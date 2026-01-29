#!/usr/bin/env python3
"""
Script to plot ESI and Map Integrity Ratio over time for all runs.
Plots ESI for all 30 runs, map_integrity_ratio from first file, and mean ESI.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys
from data_loader import load_data, calculate_position_error


def correct_map_integrity_ratio(mir_series: pd.Series) -> pd.Series:
    """Correct map integrity ratio values."""
    if mir_series is None or len(mir_series) == 0:
        return None
    old_mir = mir_series.values
    unmoved = old_mir * 20
    moved = 20 - unmoved
    new_mir = unmoved / (unmoved + 2 * moved)
    return pd.Series(new_mir, index=mir_series.index)


def calculate_yaw_error(gt_yaw, amcl_yaw):
    """
    Calculate absolute yaw error (angular difference) in degrees.
    Handles angle wrapping (e.g., difference between 359° and 1° should be 2°, not 358°).
    
    Parameters:
    -----------
    gt_yaw : float
        Ground truth yaw angle (in degrees)
    amcl_yaw : float
        AMCL yaw angle (in degrees)
    
    Returns:
    --------
    yaw_error : float
        Absolute yaw error in degrees
    """
    # Calculate the difference
    diff = gt_yaw - amcl_yaw
    
    # Normalize to [-180, 180] range to handle angle wrapping
    diff = ((diff + 180) % 360) - 180
    
    # Return absolute value
    return abs(diff)

def main():
    # Configuration parameters
    N = 30  # Number of files to read (all 30 runs)
    read_file_percentage = 1.0  # Read 50% of each file
    
    # Path to the folder containing CSV files
    user_home = os.path.expanduser('~')
    data_folder = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl')
    
    # Load data using shared function
    print("Loading data...")
    combined_df = load_data(
        data_folder, 
        N, 
        mode='percentage',
        error_threshold=0.30,  # Not used in percentage mode
        read_file_percentage=read_file_percentage
    )
    
    # Check for required columns
    required_cols = ['esi', 'timestamp', 'ground_truth_x', 'ground_truth_y', 'amcl_x', 'amcl_y']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(combined_df.columns)}")
        sys.exit(1)
    
    # Check for yaw columns
    has_yaw = 'ground_truth_yaw' in combined_df.columns and 'amcl_yaw' in combined_df.columns
    if not has_yaw:
        print("Warning: Yaw columns not found, yaw error plot will be skipped")


    # Check if map_integrity_ratio exists
    if 'map_integrity_ratio' not in combined_df.columns:
        print(f"Warning: map_integrity_ratio column not found")
        print(f"Available columns: {list(combined_df.columns)}")
        # Continue anyway, just won't plot MIR
    
    # Load first file separately for map_integrity_ratio (they are identical)
    print("Loading first file for map_integrity_ratio...")
    first_file_path = os.path.join(data_folder, '1.csv')
    if os.path.exists(first_file_path):
        df_first = pd.read_csv(first_file_path, skiprows=range(1, 100))
        df_first.columns = df_first.columns.str.strip()
        
        # Read 50% of first file
        total_rows = len(df_first)
        num_rows_to_read = int(total_rows * read_file_percentage)
        df_first = df_first.iloc[:num_rows_to_read].copy()
        
        # Convert timestamp to numeric if needed
        if 'timestamp' in df_first.columns:
            df_first['timestamp'] = pd.to_numeric(df_first['timestamp'], errors='coerce')
        
        print(f"Loaded {len(df_first)} rows from first file")
    else:
        print(f"Warning: First file not found at {first_file_path}")
        df_first = None
    
    # Convert timestamp to numeric for combined_df
    combined_df['timestamp'] = pd.to_numeric(combined_df['timestamp'], errors='coerce')
    
    # Normalize timestamps to start from 0 for each file
    print("Processing data for plotting...")
    file_data = {}
    all_esi_values = []
    all_timestamps = []
    
    for file_id in sorted(combined_df['file_id'].unique()):
        df_file = combined_df[combined_df['file_id'] == file_id].copy()
        
        # Normalize timestamps to start from 0
        if len(df_file) > 0:
            first_timestamp = df_file['timestamp'].iloc[0]
            df_file['time_normalized'] = df_file['timestamp'] - first_timestamp
            
            # Calculate position errors
            position_errors = []
            for idx, row in df_file.iterrows():
                error = calculate_position_error(
                    row['ground_truth_x'], row['ground_truth_y'],
                    row['amcl_x'], row['amcl_y']
                )
                position_errors.append(error)
            df_file['position_error'] = position_errors
            
            # Calculate yaw errors if available
            if has_yaw:
                yaw_errors = []
                for idx, row in df_file.iterrows():
                    error = calculate_yaw_error(
                        row['ground_truth_yaw'],
                        row['amcl_yaw']
                    )
                    yaw_errors.append(error)
                df_file['yaw_error'] = yaw_errors
            
            file_data[file_id] = df_file
            all_esi_values.append(df_file['esi'].values)
            all_timestamps.append(df_file['time_normalized'].values)
    
    # Last absolute position error per run (sorted by error ascending)
    last_position_errors = []
    for file_id in sorted(file_data.keys()):
        df_file = file_data[file_id]
        if len(df_file) > 0:
            last_err = df_file['position_error'].iloc[-1]
            last_position_errors.append((file_id, last_err))
    last_position_errors.sort(key=lambda x: x[1])
    print("\n" + "="*60)
    print("Last absolute position error per run (sorted by error, ascending):")
    print("="*60)
    for file_id, err in last_position_errors:
        print(f"  Run {file_id:2d}: {err:.6f} m")
    print("="*60)

    # Calculate mean ESI across all files
    # Need to interpolate to common time points for averaging
    print("Calculating mean ESI...")
    
    # Find the maximum time across all files
    max_time = max([ts.max() if len(ts) > 0 else 0 for ts in all_timestamps])
    
    # Create common time grid (use finest resolution)
    min_dt = min([np.diff(ts).min() if len(ts) > 1 and np.diff(ts).min() > 0 else 1.0 
                  for ts in all_timestamps if len(ts) > 1])
    common_time = np.arange(0, max_time + min_dt, min_dt)
    
    # Interpolate ESI values for each file to common time grid
    interpolated_esi = []
    for file_id in sorted(file_data.keys()):
        df_file = file_data[file_id]
        if len(df_file) > 1:
            # Interpolate ESI to common time grid
            esi_interp = np.interp(common_time, df_file['time_normalized'].values, 
                                   df_file['esi'].values)
            interpolated_esi.append(esi_interp)
    
    # Calculate mean ESI
    if len(interpolated_esi) > 0:
        mean_esi = np.mean(interpolated_esi, axis=0)
    else:
        mean_esi = np.array([])
        common_time = np.array([])
    
    # Create the plot with subplots
    print("Creating plot...")
    fig = plt.figure(figsize=(14, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # Sync figure subplotpars with GridSpec so toolbar "Configure subplots" can adjust wspace/hspace
    fig.subplots_adjust(left=gs.left, right=gs.right, bottom=gs.bottom, top=gs.top,
                        wspace=gs.wspace, hspace=gs.hspace)
    
    def _sync_gridspec_to_subplotpars(event):
        if event.canvas.figure is fig:
            gs.update(left=fig.subplotpars.left, right=fig.subplotpars.right,
                      bottom=fig.subplotpars.bottom, top=fig.subplotpars.top,
                      wspace=fig.subplotpars.wspace, hspace=fig.subplotpars.hspace)
    
    fig.canvas.mpl_connect('draw_event', _sync_gridspec_to_subplotpars)
    
    # Top plot: ESI and MIR (spans both columns)
    ax_top = fig.add_subplot(gs[0, :])
    
    # Plot ESI for each run (thin lines, semi-transparent)
    colors = plt.cm.tab20(np.linspace(0, 1, len(file_data)))
    for idx, (file_id, df_file) in enumerate(sorted(file_data.items())):
        ax_top.plot(df_file['time_normalized'], df_file['esi'], 
               alpha=0.3, linewidth=0.8, color=colors[idx % len(colors)],
               label=f'Run 1-30' if idx== 1 else None)  # Only label first 10 to avoid clutter
    
    # Plot mean ESI (thick line)
    if len(mean_esi) > 0:
        ax_top.plot(common_time, mean_esi, 
               color='black', linewidth=3, label='Mean ESI', zorder=10)
    
    # Plot map_integrity_ratio from first file
    if df_first is not None and 'map_integrity_ratio' in df_first.columns:
        # Normalize timestamp for first file
        if len(df_first) > 0:
            first_timestamp = df_first['timestamp'].iloc[0]
            df_first['time_normalized'] = df_first['timestamp'] - first_timestamp
            
            # Update max_time to include first file if needed
            max_time_first = df_first['time_normalized'].max()
            max_time = max(max_time, max_time_first)
            
            # Apply correction to map_integrity_ratio
            corrected_mir = correct_map_integrity_ratio(df_first['map_integrity_ratio'])
            
            if corrected_mir is not None:
                ax_top.plot(df_first['time_normalized'], corrected_mir,
                       color='red', linewidth=2, linestyle='--', label='Map Integrity Ratio', zorder=9)
    
    # Set labels and title for top plot
    ax_top.set_xlabel('Time [s]', fontsize=12)
    ax_top.set_ylabel('ESI / Map Integrity Ratio', fontsize=12)
    ax_top.set_title(f'ESI and Map Integrity Ratio over Time\n({N} runs, {read_file_percentage*100:.0f}% of each file)', fontsize=14)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc='best', fontsize=10)
    ax_top.set_xlim(0, max_time)
    
    # Bottom left: Position error
    ax_pos = fig.add_subplot(gs[1, 0])
    
    for idx, (file_id, df_file) in enumerate(sorted(file_data.items())):
        ax_pos.plot(df_file['time_normalized'], df_file['position_error'], 
                   alpha=0.3, linewidth=0.8, color=colors[idx % len(colors)])
    
    # Calculate and plot mean position error
    interpolated_pos_errors = []
    for file_id in sorted(file_data.keys()):
        df_file = file_data[file_id]
        if len(df_file) > 1:
            pos_error_interp = np.interp(common_time, df_file['time_normalized'].values, 
                                         df_file['position_error'].values)
            interpolated_pos_errors.append(pos_error_interp)
    
    if len(interpolated_pos_errors) > 0:
        mean_pos_error = np.mean(interpolated_pos_errors, axis=0)
        ax_pos.plot(common_time, mean_pos_error, 
                   color='black', linewidth=3, zorder=10)
    
    ax_pos.set_xlabel('Time [s]', fontsize=11)
    ax_pos.set_ylabel('Position Error [m]', fontsize=11)
    ax_pos.set_title('Absolute Position Error', fontsize=12)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_xlim(0, max_time)
    if len(interpolated_pos_errors) > 0:
        ax_pos.legend(loc='best', fontsize=9)
    
    # Bottom right: Yaw error
    ax_yaw = fig.add_subplot(gs[1, 1])
    
    if has_yaw:
        for idx, (file_id, df_file) in enumerate(sorted(file_data.items())):
            ax_yaw.plot(df_file['time_normalized'], df_file['yaw_error'], 
                       alpha=0.3, linewidth=0.8, color=colors[idx % len(colors)])
        
        # Calculate and plot mean yaw error
        interpolated_yaw_errors = []
        for file_id in sorted(file_data.keys()):
            df_file = file_data[file_id]
            if len(df_file) > 1:
                yaw_error_interp = np.interp(common_time, df_file['time_normalized'].values, 
                                            df_file['yaw_error'].values)
                interpolated_yaw_errors.append(yaw_error_interp)
        
        if len(interpolated_yaw_errors) > 0:
            mean_yaw_error = np.mean(interpolated_yaw_errors, axis=0)
            ax_yaw.plot(common_time, mean_yaw_error, 
                       color='black', linewidth=3, zorder=10)
        
        ax_yaw.set_xlabel('Time [s]', fontsize=11)
        ax_yaw.set_ylabel('Yaw Error [deg]', fontsize=11)
        ax_yaw.set_title('Absolute Yaw Error', fontsize=12)
        ax_yaw.grid(True, alpha=0.3)
        ax_yaw.set_xlim(0, max_time)
        if len(interpolated_yaw_errors) > 0:
            ax_yaw.legend(loc='best', fontsize=9)
    else:
        ax_yaw.text(0.5, 0.5, 'Yaw data not available', 
                   ha='center', va='center', transform=ax_yaw.transAxes, fontsize=12)
        ax_yaw.set_title('Absolute Yaw Error', fontsize=12)
    
    # Save the plot
    output_path = os.path.join(data_folder, 'esi_and_mir_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    # Calculate average position and yaw errors across all data
    all_position_errors = []
    all_yaw_errors = []
    
    for file_id, df_file in file_data.items():
        all_position_errors.extend(df_file['position_error'].values)
        if has_yaw:
            all_yaw_errors.extend(df_file['yaw_error'].values)
    
    avg_position_error = np.mean(all_position_errors) if len(all_position_errors) > 0 else None
    avg_yaw_error = np.mean(all_yaw_errors) if len(all_yaw_errors) > 0 else None
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(f"Number of runs: {len(file_data)}")
    print(f"Total data points: {len(combined_df)}")
    
    # Print average errors
    if avg_position_error is not None:
        print(f"\nAverage Position Error: {avg_position_error:.6f} m")
    if avg_yaw_error is not None:
        print(f"Average Yaw Error: {avg_yaw_error:.6f} deg")
    
    if len(mean_esi) > 0:
        print(f"\nMean ESI statistics:")
        print(f"  Mean: {np.mean(mean_esi):.6f}")
        print(f"  Std: {np.std(mean_esi):.6f}")
        print(f"  Min: {np.min(mean_esi):.6f}")
        print(f"  Max: {np.max(mean_esi):.6f}")
    
    if df_first is not None and 'map_integrity_ratio' in df_first.columns:
        # Apply correction to map_integrity_ratio for statistics
        corrected_mir = correct_map_integrity_ratio(df_first['map_integrity_ratio'])
        if corrected_mir is not None:
            mir_values = corrected_mir.dropna()
            if len(mir_values) > 0:
                print(f"\nMap Integrity Ratio statistics:")
                print(f"  Mean: {mir_values.mean():.6f}")
                print(f"  Std: {mir_values.std():.6f}")
                print(f"  Min: {mir_values.min():.6f}")
                print(f"  Max: {mir_values.max():.6f}")
    
    print("="*60)


if __name__ == '__main__':
    main()

