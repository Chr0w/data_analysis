#!/usr/bin/env python3
"""
Script to plot threshold-based metrics as box plots.
Creates 2 subplots: average time after last exceed and average time until first exceed.
Boxes are outline-only (no fill), black outlines.

Box plot elements (matplotlib default):
- Box: 25th to 75th percentile (IQR). The thick line inside is the median (50th percentile).
- Whiskers: extend from the box to the furthest data point within 1.5×IQR of the
  lower/upper quartile (i.e. no further than Q1 - 1.5*IQR and Q3 + 1.5*IQR).
  Points outside that range are drawn as fliers (outliers).
- Diamond marker: arithmetic mean (matches printed averages).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from matplotlib.widgets import Slider
from data_loader import calculate_position_error

# Thresholds
POSITION_THRESHOLD = 1.0  # meters
YAW_THRESHOLD = 45.0  # degrees
do_threshold_variation_plot = False
vary_metric_position = True
print_medians = True

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
        except Exception as e:
            print(f"Warning: Error reading {csv_path}: {e}, skipping...", file=sys.stderr)
            continue
    
    if not dataframes:
        print("Error: No valid CSV files found", file=sys.stderr)
        sys.exit(1)
    
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
    
    # Collect metrics for all modes
    all_metrics = {}
    
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
        
        all_metrics[mode_name] = metrics
    
    # Get default_02 metrics for comparison
    default_02_metrics = all_metrics.get('default_02')
    
    # Print metrics with improvement percentages
    for mode_name, metrics in all_metrics.items():
        print(f"\n{mode_name}:")
        
        # Average time below threshold
        time_below = metrics['avg_time_below']
        median_time_below = metrics.get('median_time_below', np.nan)
        if default_02_metrics and mode_name != 'default_02':
            default_02_time = default_02_metrics['avg_time_below']
            if default_02_time > 0:
                improvement = ((time_below - default_02_time) / default_02_time) * 100
                print(f"  Average time below threshold (both pos and yaw): {time_below:.2f} s ({improvement:+.1f}%)")
            else:
                print(f"  Average time below threshold (both pos and yaw): {time_below:.2f} s")
        else:
            print(f"  Average time below threshold (both pos and yaw): {time_below:.2f} s")
        if print_medians and not np.isnan(median_time_below):
            if default_02_metrics and mode_name != 'default_02':
                default_02_median_time = default_02_metrics.get('median_time_below', np.nan)
                if not np.isnan(default_02_median_time) and default_02_median_time > 0:
                    improvement = ((median_time_below - default_02_median_time) / default_02_median_time) * 100
                    print(f"    Median: {median_time_below:.2f} s ({improvement:+.1f}%)")
                else:
                    print(f"    Median: {median_time_below:.2f} s")
            else:
                print(f"    Median: {median_time_below:.2f} s")
        
        # Average time until first exceed
        time_until = metrics['avg_time_until_exceed']
        median_time_until = metrics.get('median_time_until_exceed', np.nan)
        if default_02_metrics and mode_name != 'default_02':
            default_02_time_until = default_02_metrics['avg_time_until_exceed']
            if default_02_time_until > 0:
                improvement = ((time_until - default_02_time_until) / default_02_time_until) * 100
                print(f"  Average time below threshold until first exceed (per run): {time_until:.2f} s ({improvement:+.1f}%)")
            else:
                print(f"  Average time below threshold until first exceed (per run): {time_until:.2f} s")
        else:
            print(f"  Average time below threshold until first exceed (per run): {time_until:.2f} s")
        if print_medians and not np.isnan(median_time_until):
            if default_02_metrics and mode_name != 'default_02':
                default_02_median_time_until = default_02_metrics.get('median_time_until_exceed', np.nan)
                if not np.isnan(default_02_median_time_until) and default_02_median_time_until > 0:
                    improvement = ((median_time_until - default_02_median_time_until) / default_02_median_time_until) * 100
                    print(f"    Median: {median_time_until:.2f} s ({improvement:+.1f}%)")
                else:
                    print(f"    Median: {median_time_until:.2f} s")
            else:
                print(f"    Median: {median_time_until:.2f} s")
        
        # Average time after last instance of being lost (per run)
        time_after_last = metrics['avg_time_after_last_exceed']
        median_time_after_last = metrics.get('median_time_after_last_exceed', np.nan)
        if default_02_metrics and mode_name != 'default_02':
            default_02_after = default_02_metrics['avg_time_after_last_exceed']
            if default_02_after > 0:
                improvement = ((time_after_last - default_02_after) / default_02_after) * 100
                print(f"  Average time after last instance of being lost (per run): {time_after_last:.2f} s ({improvement:+.1f}%)")
            else:
                print(f"  Average time after last instance of being lost (per run): {time_after_last:.2f} s")
        else:
            print(f"  Average time after last instance of being lost (per run): {time_after_last:.2f} s")
        if not np.isnan(median_time_after_last):
            if default_02_metrics and mode_name != 'default_02':
                default_02_median_after = default_02_metrics.get('median_time_after_last_exceed', np.nan)
                if not np.isnan(default_02_median_after) and default_02_median_after > 0:
                    improvement = ((median_time_after_last - default_02_median_after) / default_02_median_after) * 100
                    print(f"    Median: {median_time_after_last:.2f} s ({improvement:+.1f}%)")
                else:
                    print(f"    Median: {median_time_after_last:.2f} s")
            else:
                print(f"    Median: {median_time_after_last:.2f} s")
        
        # Position RMSE (pre-lost: until first exceed)
        pos_rmse_pre = metrics['avg_position_RMSE_until_exceed']
        median_pos_rmse_pre = metrics.get('median_position_RMSE_until_exceed', np.nan)
        if not np.isnan(pos_rmse_pre):
            if default_02_metrics and mode_name != 'default_02':
                default_02_pos = default_02_metrics['avg_position_RMSE_until_exceed']
                if not np.isnan(default_02_pos) and default_02_pos > 0:
                    improvement = ((default_02_pos - pos_rmse_pre) / default_02_pos) * 100
                    print(f"  Position RMSE (pre-lost, until first exceed): {pos_rmse_pre:.6f} m ({improvement:+.1f}%)")
                else:
                    print(f"  Position RMSE (pre-lost, until first exceed): {pos_rmse_pre:.6f} m")
            else:
                print(f"  Position RMSE (pre-lost, until first exceed): {pos_rmse_pre:.6f} m")
            if print_medians and not np.isnan(median_pos_rmse_pre):
                if default_02_metrics and mode_name != 'default_02':
                    default_02_median_pos = default_02_metrics.get('median_position_RMSE_until_exceed', np.nan)
                    if not np.isnan(default_02_median_pos) and default_02_median_pos > 0:
                        improvement = ((default_02_median_pos - median_pos_rmse_pre) / default_02_median_pos) * 100
                        print(f"    Median: {median_pos_rmse_pre:.6f} m ({improvement:+.1f}%)")
                    else:
                        print(f"    Median: {median_pos_rmse_pre:.6f} m")
                else:
                    print(f"    Median: {median_pos_rmse_pre:.6f} m")
        
        # Position RMSE (post-lost: after last exceed)
        pos_rmse_post = metrics.get('avg_position_RMSE_after_last_exceed', np.nan)
        median_pos_rmse_post = metrics.get('median_position_RMSE_after_last_exceed', np.nan)
        if not np.isnan(pos_rmse_post):
            if default_02_metrics and mode_name != 'default_02':
                default_02_pos_post = default_02_metrics.get('avg_position_RMSE_after_last_exceed', np.nan)
                if not np.isnan(default_02_pos_post) and default_02_pos_post > 0:
                    improvement = ((default_02_pos_post - pos_rmse_post) / default_02_pos_post) * 100
                    print(f"  Position RMSE (post-lost, after last exceed): {pos_rmse_post:.6f} m ({improvement:+.1f}%)")
                else:
                    print(f"  Position RMSE (post-lost, after last exceed): {pos_rmse_post:.6f} m")
            else:
                print(f"  Position RMSE (post-lost, after last exceed): {pos_rmse_post:.6f} m")
            if print_medians and not np.isnan(median_pos_rmse_post):
                if default_02_metrics and mode_name != 'default_02':
                    default_02_median_pos_post = default_02_metrics.get('median_position_RMSE_after_last_exceed', np.nan)
                    if not np.isnan(default_02_median_pos_post) and default_02_median_pos_post > 0:
                        improvement = ((default_02_median_pos_post - median_pos_rmse_post) / default_02_median_pos_post) * 100
                        print(f"    Median: {median_pos_rmse_post:.6f} m ({improvement:+.1f}%)")
                    else:
                        print(f"    Median: {median_pos_rmse_post:.6f} m")
                else:
                    print(f"    Median: {median_pos_rmse_post:.6f} m")
        
        # Yaw RMSE (pre-lost: until first exceed)
        yaw_rmse_pre = metrics['avg_yaw_RMSE_until_exceed']
        median_yaw_rmse_pre = metrics.get('median_yaw_RMSE_until_exceed', np.nan)
        if not np.isnan(yaw_rmse_pre):
            if default_02_metrics and mode_name != 'default_02':
                default_02_yaw = default_02_metrics['avg_yaw_RMSE_until_exceed']
                if not np.isnan(default_02_yaw) and default_02_yaw > 0:
                    improvement = ((default_02_yaw - yaw_rmse_pre) / default_02_yaw) * 100
                    print(f"  Yaw RMSE (pre-lost, until first exceed): {yaw_rmse_pre:.6f} deg ({improvement:+.1f}%)")
                else:
                    print(f"  Yaw RMSE (pre-lost, until first exceed): {yaw_rmse_pre:.6f} deg")
            else:
                print(f"  Yaw RMSE (pre-lost, until first exceed): {yaw_rmse_pre:.6f} deg")
            if print_medians and not np.isnan(median_yaw_rmse_pre):
                if default_02_metrics and mode_name != 'default_02':
                    default_02_median_yaw = default_02_metrics.get('median_yaw_RMSE_until_exceed', np.nan)
                    if not np.isnan(default_02_median_yaw) and default_02_median_yaw > 0:
                        improvement = ((default_02_median_yaw - median_yaw_rmse_pre) / default_02_median_yaw) * 100
                        print(f"    Median: {median_yaw_rmse_pre:.6f} deg ({improvement:+.1f}%)")
                    else:
                        print(f"    Median: {median_yaw_rmse_pre:.6f} deg")
                else:
                    print(f"    Median: {median_yaw_rmse_pre:.6f} deg")
        
        # Yaw RMSE (post-lost: after last exceed)
        yaw_rmse_post = metrics.get('avg_yaw_RMSE_after_last_exceed', np.nan)
        median_yaw_rmse_post = metrics.get('median_yaw_RMSE_after_last_exceed', np.nan)
        if not np.isnan(yaw_rmse_post):
            if default_02_metrics and mode_name != 'default_02':
                default_02_yaw_post = default_02_metrics.get('avg_yaw_RMSE_after_last_exceed', np.nan)
                if not np.isnan(default_02_yaw_post) and default_02_yaw_post > 0:
                    improvement = ((default_02_yaw_post - yaw_rmse_post) / default_02_yaw_post) * 100
                    print(f"  Yaw RMSE (post-lost, after last exceed): {yaw_rmse_post:.6f} deg ({improvement:+.1f}%)")
                else:
                    print(f"  Yaw RMSE (post-lost, after last exceed): {yaw_rmse_post:.6f} deg")
            else:
                print(f"  Yaw RMSE (post-lost, after last exceed): {yaw_rmse_post:.6f} deg")
            if print_medians and not np.isnan(median_yaw_rmse_post):
                if default_02_metrics and mode_name != 'default_02':
                    default_02_median_yaw_post = default_02_metrics.get('median_yaw_RMSE_after_last_exceed', np.nan)
                    if not np.isnan(default_02_median_yaw_post) and default_02_median_yaw_post > 0:
                        improvement = ((default_02_median_yaw_post - median_yaw_rmse_post) / default_02_median_yaw_post) * 100
                        print(f"    Median: {median_yaw_rmse_post:.6f} deg ({improvement:+.1f}%)")
                    else:
                        print(f"    Median: {median_yaw_rmse_post:.6f} deg")
                else:
                    print(f"    Median: {median_yaw_rmse_post:.6f} deg")
        
        # Number of runs that exceeded threshold
        runs_exceeded = metrics['runs_exceeded_threshold']
        total_runs = metrics['total_runs']
        runs_ended_within = total_runs - metrics['runs_outside_threshold']
        print(f"  Runs that exceeded threshold (either pos or yaw): {runs_exceeded} / {total_runs}")
        print(f"  Runs that ended within thresholds: {runs_ended_within} / {total_runs}")
        
        print(f"  Total runs analyzed: {total_runs}")
    
    print("="*60 + "\n")
    
    if not all_metrics:
        print("Error: No metrics collected for plotting", file=sys.stderr)
        sys.exit(1)
    
    # Save per-run (individual) results for each metric to CSV for loading in other scripts
    rows = []
    for mode, metrics in all_metrics.items():
        n_runs = metrics['total_runs']
        time_below = metrics['avg_time_below_per_run']
        time_until_exceed = metrics['avg_time_until_exceed_per_run']
        time_after_last = metrics['avg_time_after_last_exceed_per_run']
        pos_rmse_pre = metrics['avg_position_RMSE_until_exceed_per_run']
        yaw_rmse_pre = metrics['avg_yaw_RMSE_until_exceed_per_run']
        pos_rmse_post = metrics.get('avg_position_RMSE_after_last_exceed_per_run', [])
        yaw_rmse_post = metrics.get('avg_yaw_RMSE_after_last_exceed_per_run', [])
        for run_idx in range(n_runs):
            rows.append({
                'mode': mode,
                'run_idx': run_idx,
                'time_below_threshold_s': time_below[run_idx] if run_idx < len(time_below) else np.nan,
                'time_until_first_exceed_s': time_until_exceed[run_idx] if run_idx < len(time_until_exceed) else np.nan,
                'time_after_last_exceed_s': time_after_last[run_idx] if run_idx < len(time_after_last) else np.nan,
                'position_rmse_pre_lost_m': pos_rmse_pre[run_idx] if run_idx < len(pos_rmse_pre) else np.nan,
                'yaw_rmse_pre_lost_deg': yaw_rmse_pre[run_idx] if run_idx < len(yaw_rmse_pre) else np.nan,
                'position_rmse_post_lost_m': pos_rmse_post[run_idx] if run_idx < len(pos_rmse_post) else np.nan,
                'yaw_rmse_post_lost_deg': yaw_rmse_post[run_idx] if run_idx < len(yaw_rmse_post) else np.nan,
            })
    results_df = pd.DataFrame(rows)
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'box_plot_individual_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Per-run results saved to {results_path}")
    
    # Define mode labels
    mode_labels = {
        'default': 'Default',
        'default_02': 'Default_02',
        'default_001': 'Default_001',
        'tuning': 'Tuning',
    }
    
    # Prepare data for plotting
    modes = list(all_metrics.keys())
    n_modes = len(modes)
    
    # Extract per-run data for each metric
    metric_data = {
        'avg_time_after_last_exceed': {
            'values': {},
            'means': {},
            'stds': {},
            'title': 'Average Time After Last Exceed',
            'ylabel': 'Time [s]'
        },
        'avg_time_until_exceed': {
            'values': {},
            'means': {},
            'stds': {},
            'title': 'Average Time Until First Exceed',
            'ylabel': 'Time [s]'
        },
    }
    
    # Collect per-run values and calculate statistics
    for mode in modes:
        metrics = all_metrics[mode]
        
        # Map metric keys to per-run keys
        metric_to_per_run = {
            'avg_time_after_last_exceed': 'avg_time_after_last_exceed_per_run',
            'avg_time_until_exceed': 'avg_time_until_exceed_per_run',
        }
        
        # Get per-run data from the metrics dict
        for metric_key in metric_data.keys():
            per_run_key = metric_to_per_run.get(metric_key)
            if per_run_key and per_run_key in metrics:
                values = metrics[per_run_key]
                if len(values) > 0:
                    # Filter out NaN values
                    values = [v for v in values if not np.isnan(v)]
                    if len(values) > 0:
                        metric_data[metric_key]['values'][mode] = values
                        metric_data[metric_key]['means'][mode] = np.mean(values)
                        metric_data[metric_key]['stds'][mode] = np.std(values)
                    else:
                        metric_data[metric_key]['means'][mode] = np.nan
                        metric_data[metric_key]['stds'][mode] = 0.0
                else:
                    metric_data[metric_key]['means'][mode] = np.nan
                    metric_data[metric_key]['stds'][mode] = 0.0
            elif metric_key in metrics:
                # Fallback: use the average value (no std dev available)
                mean_val = metrics[metric_key]
                metric_data[metric_key]['means'][mode] = mean_val if not np.isnan(mean_val) else np.nan
                metric_data[metric_key]['stds'][mode] = 0.0
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes = axes.flatten()
    
    # Store plot data for each subplot to enable updates
    plot_data_storage = []
    
    # Define initial font sizes for sliders (used in plotting)
    text_fontsize_init = 9
    spacing_init = 1.0
    
    # Plot each metric
    for idx, (metric_key, metric_info) in enumerate(metric_data.items()):
        ax = axes[idx]
        
        # Collect data for each mode
        mode_data = []
        labels = []
        
        for mode in modes:
            values = metric_info['values'].get(mode, [])
            mean_val = metric_info['means'].get(mode, np.nan)
            std_val = metric_info['stds'].get(mode, 0.0)
            
            if len(values) > 0 and not np.isnan(mean_val):
                mode_data.append({
                    'mode': mode,
                    'values': values,
                    'mean': mean_val,
                    'std': std_val,
                    'label': mode_labels.get(mode, mode)
                })
                labels.append(mode_labels.get(mode, mode))
        
        if not mode_data:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(metric_info['title'], fontsize=14, fontweight='bold')
            plot_data_storage.append(None)
            continue
        
        # Store plot elements for this subplot (for spacing slider redraw)
        plot_elements = {
            'ax': ax,
            'mode_data': mode_data,
            'x_labels': [],
            'metric_info': metric_info,
            'idx': idx
        }
        
        # Create initial x positions for each mode (spacing = 1.0)
        spacing = 1.0
        x_positions = np.arange(len(mode_data)) * spacing
        box_width = 0.5 * spacing
        
        # Data for boxplot: list of arrays, one per mode
        data_for_boxplot = [data['values'] for data in mode_data]
        
        # Regular box plot: outline only, no fill, black outlines
        bp = ax.boxplot(
            data_for_boxplot,
            positions=x_positions,
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            zorder=2,
        )
        for box in bp['boxes']:
            box.set_facecolor('none')
            box.set_edgecolor('black')
            box.set_linewidth(1.5)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linestyle('--')
        for cap in bp['caps']:
            cap.set_color('black')
        for flier in bp['fliers']:
            flier.set_marker('o')
            flier.set_markerfacecolor('none')
            flier.set_markeredgecolor('black')
        
        # Draw mean (average) as a diamond so it matches the printed metrics
        means = [data['mean'] for data in mode_data]
        ax.scatter(x_positions, means, marker='D', s=40, color='black', edgecolors='black',
                   linewidths=1, zorder=5)
        
        # Set y-limits to accommodate all data
        all_values = []
        for data in mode_data:
            all_values.extend(data['values'])
        if all_values:
            y_min = min(all_values)
            y_max = max(all_values)
            y_range = y_max - y_min if y_max > y_min else 1
            ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
        
        # Create x-axis labels (mode names)
        x_labels = [data['label'] for data in mode_data]
        plot_elements['x_labels'] = x_labels
        
        # Customize plot
        ax.set_xlabel('')
        ax.set_ylabel(metric_info['ylabel'], fontsize=text_fontsize_init)
        ax.set_title(metric_info['title'], fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=text_fontsize_init)
        ax.grid(True, alpha=0.3, axis='y', zorder=0)
        
        plot_data_storage.append(plot_elements)
    
    # Add sliders for text sizes and spacing
    plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.22, left=0.05, right=0.95)
    ax_slider_text = plt.axes([0.25, 0.12, 0.5, 0.03])
    ax_slider_spacing = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider_text_fontsize = Slider(ax_slider_text, "Axis text size", 6, 20, valinit=text_fontsize_init, valstep=1)
    slider_spacing = Slider(ax_slider_spacing, "Mode spacing", 0.5, 3.0, valinit=spacing_init, valstep=0.1)
    
    def update_text_fontsize(val):
        fs = int(slider_text_fontsize.val)
        for ax in axes:
            # Update x-axis tick labels
            for label in ax.get_xticklabels():
                label.set_fontsize(fs)
            # Update y-axis tick labels (values)
            for label in ax.get_yticklabels():
                label.set_fontsize(fs)
            # Update y-axis label text
            ylabel = ax.get_ylabel()
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=fs)
        fig.canvas.draw_idle()
    
    def update_spacing(val):
        spacing = slider_spacing.val
        for plot_elements in plot_data_storage:
            if plot_elements is None:
                continue
            
            ax = plot_elements['ax']
            mode_data = plot_elements['mode_data']
            n_modes = len(mode_data)
            metric_info = plot_elements['metric_info']
            idx = plot_elements['idx']
            
            # Recalculate x positions with new spacing
            x_positions = np.arange(n_modes) * spacing
            box_width = 0.5 * spacing
            
            # Redraw box plot with new positions
            ax.clear()
            data_for_boxplot = [data['values'] for data in mode_data]
            bp = ax.boxplot(
                data_for_boxplot,
                positions=x_positions,
                widths=box_width,
                patch_artist=True,
                showfliers=True,
                zorder=2,
            )
            for box in bp['boxes']:
                box.set_facecolor('none')
                box.set_edgecolor('black')
                box.set_linewidth(1.5)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linestyle('--')
            for cap in bp['caps']:
                cap.set_color('black')
            for flier in bp['fliers']:
                flier.set_marker('o')
                flier.set_markerfacecolor('none')
                flier.set_markeredgecolor('black')
            
            # Draw mean (average) as a diamond
            means = [data['mean'] for data in mode_data]
            ax.scatter(x_positions, means, marker='D', s=40, color='black', edgecolors='black',
                       linewidths=1, zorder=5)
            
            all_values = []
            for data in mode_data:
                all_values.extend(data['values'])
            if all_values:
                y_min = min(all_values)
                y_max = max(all_values)
                y_range = y_max - y_min if y_max > y_min else 1
                ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)
            
            fs = int(slider_text_fontsize.val)
            ax.set_xlabel('')
            ax.set_ylabel(metric_info['ylabel'], fontsize=fs)
            ax.set_title(metric_info['title'], fontsize=14, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(plot_elements['x_labels'], rotation=0, ha='center', fontsize=fs)
            ax.grid(True, alpha=0.3, axis='y', zorder=0)
            
            if len(x_positions) > 0:
                x_min = x_positions[0] - spacing * 0.5
                x_max = x_positions[-1] + spacing * 0.5
                ax.set_xlim(x_min, x_max)
        
        fig.canvas.draw_idle()
    
    slider_text_fontsize.on_changed(update_text_fontsize)
    slider_spacing.on_changed(update_spacing)
    update_text_fontsize(text_fontsize_init)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = f'/home/{os.getenv("USER")}/data_analysis/box_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Box plot saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    # Create threshold variation plot if enabled
    if do_threshold_variation_plot:
        create_threshold_variation_plot(all_metrics, mode_folders, dataframes, user_home)

def create_threshold_variation_plot(all_metrics, mode_folders, dataframes, user_home):
    """
    Create a plot showing how metrics vary with threshold.
    4 subplots, one for each metric, with lines for each mode.
    Varies position threshold if vary_metric_position is True, otherwise varies yaw threshold.
    """
    print("\n" + "="*60)
    print("Creating threshold variation plot...")
    print("="*60)
    
    # Determine which threshold to vary
    if vary_metric_position:
        # Vary position threshold
        thresholds = np.arange(0.1, 1.6, 0.1)  # 0.1, 0.2, 0.3, ..., 1.5
        pos_threshold_base = None  # Will vary
        yaw_threshold_base = YAW_THRESHOLD  # Keep constant
        xlabel = 'Position Threshold [m]'
        print(f"Varying position threshold from 0.1 to 1.5 m (yaw threshold fixed at {YAW_THRESHOLD} deg)")
    else:
        # Vary yaw threshold
        thresholds = np.arange(20, 100, 10)  # 20, 30, 40, 50, 60, 70, 80, 90
        pos_threshold_base = POSITION_THRESHOLD  # Keep constant
        yaw_threshold_base = None  # Will vary
        xlabel = 'Yaw Threshold [deg]'
        print(f"Varying yaw threshold from 20 to 90 deg (position threshold fixed at {POSITION_THRESHOLD} m)")
    
    # Define colors for each mode (consistent with main plot)
    mode_colors = {
        'default': 'yellow',
        'default_02': 'green',
        'default_001': 'red',
        'tuning': 'blue',
    }
    
    # Define mode labels
    mode_labels = {
        'default': 'Default',
        'default_02': 'Default_02',
        'default_001': 'Default_001',
        'tuning': 'Tuning',
    }
    
    # Collect data for each mode and threshold
    threshold_data = {}
    
    for mode_name, folder_path in mode_folders.items():
        if mode_name not in dataframes:
            continue  # Skip if combined results don't exist
        
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Processing {mode_name}...")
        threshold_data[mode_name] = {
            'avg_time_below': [],
            'avg_time_until_exceed': [],
            'avg_position_RMSE_until_exceed': [],
            'avg_yaw_RMSE_until_exceed': [],
        }
        
        for threshold_val in thresholds:
            # Set thresholds based on which one we're varying
            if vary_metric_position:
                pos_thresh = threshold_val
                yaw_thresh = yaw_threshold_base
            else:
                pos_thresh = pos_threshold_base
                yaw_thresh = threshold_val
            
            metrics = calculate_threshold_metrics(folder_path, pos_thresh, yaw_thresh)
            if metrics is None:
                # If no data, append NaN
                threshold_data[mode_name]['avg_time_below'].append(np.nan)
                threshold_data[mode_name]['avg_time_until_exceed'].append(np.nan)
                threshold_data[mode_name]['avg_position_RMSE_until_exceed'].append(np.nan)
                threshold_data[mode_name]['avg_yaw_RMSE_until_exceed'].append(np.nan)
            else:
                threshold_data[mode_name]['avg_time_below'].append(metrics['avg_time_below'])
                threshold_data[mode_name]['avg_time_until_exceed'].append(metrics['avg_time_until_exceed'])
                threshold_data[mode_name]['avg_position_RMSE_until_exceed'].append(metrics['avg_position_RMSE_until_exceed'])
                threshold_data[mode_name]['avg_yaw_RMSE_until_exceed'].append(metrics['avg_yaw_RMSE_until_exceed'])
    
    if not threshold_data:
        print("Error: No data collected for threshold variation plot", file=sys.stderr)
        return
    
    # Create figure with 4 subplots (2x2 grid)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # Define metrics for plotting
    metric_configs = [
        {
            'key': 'avg_time_below',
            'title': 'Time Below Threshold',
            'ylabel': 'Time [s]',
            'idx': 0
        },
        {
            'key': 'avg_time_until_exceed',
            'title': 'Time Until First Exceed',
            'ylabel': 'Time [s]',
            'idx': 1
        },
        {
            'key': 'avg_position_RMSE_until_exceed',
            'title': 'Pos. RMSE Until First Exceed',
            'ylabel': 'RMSE [m]',
            'idx': 2
        },
        {
            'key': 'avg_yaw_RMSE_until_exceed',
            'title': 'Yaw RMSE Until First Exceed',
            'ylabel': 'RMSE [deg]',
            'idx': 3
        }
    ]
    
    # Plot each metric
    for config in metric_configs:
        ax = axes[config['idx']]
        metric_key = config['key']
        
        # Plot lines for each mode
        for mode_name in threshold_data.keys():
            values = threshold_data[mode_name][metric_key]
            # Filter out NaN values for plotting
            valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
            if valid_indices:
                valid_thresholds = [thresholds[i] for i in valid_indices]
                valid_values = [values[i] for i in valid_indices]
                
                ax.plot(valid_thresholds, valid_values, 
                       color=mode_colors.get(mode_name, 'gray'),
                       marker='o', linewidth=2, markersize=6,
                       label=mode_labels.get(mode_name, mode_name),
                       alpha=0.8)
        
        # Customize plot
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(config['ylabel'], fontsize=12)
        ax.set_title(config['title'], fontsize=14, fontweight='bold')
        ax.set_xticks(thresholds)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = f'/home/{os.getenv("USER")}/data_analysis/threshold_variation_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Threshold variation plot saved to {output_path}")
    
    # Show the plot
    plt.show()

def calculate_threshold_metrics(data_folder: str, pos_threshold: float, yaw_threshold: float, n_files: int = 30, skip_rows: int = 100):
    """
    Calculate threshold-based metrics from individual run CSV files.
    Returns per-run data for calculating statistics.
    """
    runs_outside = 0
    runs_exceeded = 0  # Runs that exceeded threshold at any point
    times_below = []
    times_until_exceed = []
    times_after_last_exceed = []  # Time from last "lost" (exceed) to end of run, per run
    position_rmse_until_exceed = []
    yaw_rmse_until_exceed = []
    position_rmse_after_last_exceed = []
    yaw_rmse_after_last_exceed = []
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
            last_exceed_index = -1  # Index of last time robot was "lost" (exceeded threshold)
            
            # Check if first point already exceeds threshold
            if pos_errors[0] >= pos_threshold or yaw_errors[0] >= yaw_threshold:
                exceeded = True
                exceed_index = 0
                last_exceed_index = 0
            
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
                if i + 1 < len(pos_errors):
                    pos_exceeded = pos_errors[i + 1] >= pos_threshold
                    yaw_exceeded = yaw_errors[i + 1] >= yaw_threshold
                    if pos_exceeded or yaw_exceeded:
                        if not exceeded:
                            exceeded = True
                            exceed_index = i + 1
                        last_exceed_index = i + 1
            
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
            # One entry per run for alignment (NaN when no exceed)
            if len(position_rmse_until_exceed) <= total_runs:
                position_rmse_until_exceed.append(np.nan)
            if len(yaw_rmse_until_exceed) <= total_runs:
                yaw_rmse_until_exceed.append(np.nan)
            
            # RMSE after last exceed (post-lost: strictly after the last sample that exceeded;
            # by definition that period has no exceed, so errors are within threshold or period is empty)
            if last_exceed_index >= 0:
                start_after = last_exceed_index + 1
            else:
                start_after = len(pos_errors)  # never lost → no post-lost period
            if start_after < len(pos_errors):
                pos_errors_after = pos_errors[start_after:]
                if len(pos_errors_after) > 0:
                    position_rmse_after_last_exceed.append(float(np.sqrt(np.mean(pos_errors_after**2))))
                else:
                    position_rmse_after_last_exceed.append(np.nan)
                if has_yaw:
                    yaw_errors_after = yaw_errors[start_after:]
                    yaw_finite_after = yaw_errors_after[np.isfinite(yaw_errors_after)]
                    if len(yaw_finite_after) > 0:
                        yaw_rmse_after_last_exceed.append(float(np.sqrt(np.mean(yaw_finite_after**2))))
                    else:
                        yaw_rmse_after_last_exceed.append(np.nan)
                else:
                    yaw_rmse_after_last_exceed.append(np.nan)
            else:
                position_rmse_after_last_exceed.append(np.nan)
                yaw_rmse_after_last_exceed.append(np.nan)
            
            # Check if this run exceeded the threshold at any point
            # If exceed_index < len(pos_errors), threshold was exceeded before the end
            if exceed_index < len(pos_errors):
                runs_exceeded += 1
            
            # Time after last instance of being lost (duration strictly after the last exceeded sample;
            # same period as post-lost RMSE — the recovered period only, or 0 if never recovered).
            if last_exceed_index >= 0 and (last_exceed_index + 1) < len(timestamps):
                raw = float(timestamps[-1] - timestamps[last_exceed_index + 1])
                time_after_last_exceed = max(0.0, raw)
            else:
                time_after_last_exceed = 0.0  # Never lost, or no samples after last exceed
            times_after_last_exceed.append(time_after_last_exceed)
            
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
    avg_time_after_last_exceed = np.mean(times_after_last_exceed) if times_after_last_exceed else 0.0
    avg_position_rmse = float(np.nanmean(position_rmse_until_exceed)) if position_rmse_until_exceed else np.nan
    avg_yaw_rmse = float(np.nanmean(yaw_rmse_until_exceed)) if yaw_rmse_until_exceed else np.nan
    avg_position_rmse_after = float(np.nanmean(position_rmse_after_last_exceed)) if position_rmse_after_last_exceed else np.nan
    avg_yaw_rmse_after = float(np.nanmean(yaw_rmse_after_last_exceed)) if yaw_rmse_after_last_exceed else np.nan
    
    # Calculate medians
    median_time_below = np.median(times_below) if times_below else 0.0
    median_time_until_exceed = np.median(times_until_exceed) if times_until_exceed else 0.0
    median_time_after_last_exceed = np.median(times_after_last_exceed) if times_after_last_exceed else 0.0
    if position_rmse_until_exceed and len(position_rmse_until_exceed) > 0:
        pos_rmse_valid = [v for v in position_rmse_until_exceed if not np.isnan(v)]
        median_position_rmse = np.median(pos_rmse_valid) if pos_rmse_valid else np.nan
    else:
        median_position_rmse = np.nan
    if yaw_rmse_until_exceed and len(yaw_rmse_until_exceed) > 0:
        yaw_rmse_valid = [v for v in yaw_rmse_until_exceed if not np.isnan(v)]
        median_yaw_rmse = np.median(yaw_rmse_valid) if yaw_rmse_valid else np.nan
    else:
        median_yaw_rmse = np.nan
    if position_rmse_after_last_exceed and len(position_rmse_after_last_exceed) > 0:
        pos_after_valid = [v for v in position_rmse_after_last_exceed if not np.isnan(v)]
        median_position_rmse_after = np.median(pos_after_valid) if pos_after_valid else np.nan
    else:
        median_position_rmse_after = np.nan
    if yaw_rmse_after_last_exceed and len(yaw_rmse_after_last_exceed) > 0:
        yaw_after_valid = [v for v in yaw_rmse_after_last_exceed if not np.isnan(v)]
        median_yaw_rmse_after = np.median(yaw_after_valid) if yaw_after_valid else np.nan
    else:
        median_yaw_rmse_after = np.nan
    
    return {
        'runs_outside_threshold': runs_outside,
        'runs_exceeded_threshold': runs_exceeded,
        'avg_time_below': avg_time_below,
        'avg_time_until_exceed': avg_time_until_exceed,
        'avg_time_after_last_exceed': avg_time_after_last_exceed,
        'avg_position_RMSE_until_exceed': avg_position_rmse,
        'avg_yaw_RMSE_until_exceed': avg_yaw_rmse,
        'avg_position_RMSE_after_last_exceed': avg_position_rmse_after,
        'avg_yaw_RMSE_after_last_exceed': avg_yaw_rmse_after,
        'median_position_RMSE_until_exceed': median_position_rmse,
        'median_yaw_RMSE_until_exceed': median_yaw_rmse,
        'median_position_RMSE_after_last_exceed': median_position_rmse_after,
        'median_yaw_RMSE_after_last_exceed': median_yaw_rmse_after,
        'median_time_below': median_time_below,
        'median_time_until_exceed': median_time_until_exceed,
        'median_time_after_last_exceed': median_time_after_last_exceed,
        # Return per-run data for statistics
        'avg_time_below_per_run': times_below,
        'avg_time_until_exceed_per_run': times_until_exceed,
        'avg_time_after_last_exceed_per_run': times_after_last_exceed,
        'avg_position_RMSE_until_exceed_per_run': position_rmse_until_exceed,
        'avg_yaw_RMSE_until_exceed_per_run': yaw_rmse_until_exceed,
        'avg_position_RMSE_after_last_exceed_per_run': position_rmse_after_last_exceed,
        'avg_yaw_RMSE_after_last_exceed_per_run': yaw_rmse_after_last_exceed,
        'total_runs': total_runs,
    }


if __name__ == '__main__':
    main()
