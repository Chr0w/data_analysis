#!/usr/bin/env python3
"""
Script to plot Position Error vs ESI correlation.
Reads the first N CSV files, samples every M'th entry, calculates absolute position error,
and plots ESI on x-axis vs Position Error on y-axis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from data_loader import load_data, calculate_all_position_errors

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Plot Position Error vs ESI correlation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_error_vs_esi_correlation.py threshold
  python plot_error_vs_esi_correlation.py percentage
        '''
    )
    parser.add_argument('mode', choices=['threshold', 'percentage'],
                       help='Stopping mode: "threshold" stops when error >= threshold, "percentage" reads specified percentage of file')
    args = parser.parse_args()
    
    # Configuration parameters
    N = 30  # Number of files to read (starting with 1)
    M = 20  # Sample every M'th entry (starting with 50)
    error_threshold = 0.30  # Position error threshold in meters
    read_file_percentage = 0.5  # Percentage of files to read
    ESI_threshold = 0.35  # ESI threshold in meters
    time_window = 15  # Time window in seconds

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
    
    # Calculate position errors for all entries
    print("Calculating position errors...")
    position_errors_array = calculate_all_position_errors(combined_df)
    position_errors = position_errors_array.tolist()  # Keep as list for compatibility
    
    # Calculate percentiles from 80 to 99
    percentiles = np.arange(80, 100)  # 80, 81, 82, ..., 99
    percentile_thresholds = np.percentile(position_errors_array, percentiles)
    
    print(f"\nPercentile thresholds (80-99):")
    for p, thresh in zip(percentiles, percentile_thresholds):
        percentage_below = (position_errors_array < thresh).sum() / len(position_errors_array) * 100
        print(f"  {p}th percentile: {thresh:.6f} m ({percentage_below:.2f}% below)")
    
    # Create percentile vs threshold plot (flipped axes)
    plt.figure(figsize=(10, 6))
    plt.plot(percentile_thresholds, percentiles, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Position Error Threshold [m]', fontsize=12)
    plt.ylabel('Percentile', fontsize=12)
    plt.title(f'Percentile vs Position Error Threshold\n(N={N} files, {len(position_errors)} total measurements)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yticks(percentiles)
    
    # Set x-axis to show 0.01 meter increments
    # min_threshold = np.min(percentile_thresholds)
    # max_threshold = np.max(percentile_thresholds)
    # x_ticks = np.arange(np.floor(min_threshold * 100) / 100, np.ceil(max_threshold * 100) / 100 + 0.01, 0.01)
    # plt.xticks(x_ticks)
    
    # Add value labels on points
    for p, thresh in zip(percentiles, percentile_thresholds):
        plt.text(thresh, p, f'{p}', fontsize=8, 
                ha='left', va='center')
    
    plt.tight_layout()
    
    # Save the percentile plot
    percentile_output_path = os.path.join(data_folder, 'percentile_threshold_plot.png')
    plt.savefig(percentile_output_path, dpi=300, bbox_inches='tight')
    print(f"\nPercentile plot saved to {percentile_output_path}")
    
    # Show the percentile plot
    plt.show()
    
    # Sample every M'th entry and get absolute position error at that measurement
    esi_values = []
    position_error_values = []
    line_numbers = []  # Track line numbers for color coding
    
    print(f"Sampling every {M}'th entry and getting position error...")
    for idx in range(0, len(combined_df), M):
        # Get the absolute position error at this specific measurement
        position_error = position_errors[idx]
        
        # Get ESI value at this index
        esi = combined_df.iloc[idx]['esi']
        
        esi_values.append(esi)
        position_error_values.append(position_error)
        line_numbers.append(idx)  # Store the line number/index
        
        if (idx // M + 1) % 10 == 0:
            print(f"  Processed {idx // M + 1} samples...")
    
    print(f"Total samples: {len(esi_values)}")
    
    # Create the plot with color coding based on line number
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(esi_values, position_error_values, 
                         c=line_numbers, cmap='coolwarm', 
                         alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    plt.xlabel('ESI', fontsize=12)
    plt.ylabel('Position Error [m]', fontsize=12)
    # Create title based on mode
    if args.mode == 'threshold':
        title_suffix = f'error threshold={error_threshold}m'
    else:
        title_suffix = f'{read_file_percentage*100:.0f}% of each file'
    plt.title(f'Position Error vs ESI Correlation\n(N={N} files, sampling every {M}\'th entry, {title_suffix})', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add colorbar to show line number mapping
    cbar = plt.colorbar(scatter)
    cbar.set_label('Line Number (Blue=Early, Red=Late)', fontsize=10)
    
    # Add some statistics
    if len(esi_values) > 0:
        correlation = np.corrcoef(esi_values, position_error_values)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(data_folder, 'position_error_esi_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(f"Number of samples: {len(esi_values)}")
    print(f"Total measurements: {len(position_errors)}")
    print(f"ESI range: [{min(esi_values):.4f}, {max(esi_values):.4f}]")
    print(f"Position Error range: [{min(position_error_values):.6f}, {max(position_error_values):.6f}]")
    
    # Print 95th percentile specifically
    percentile_95_idx = np.where(percentiles == 95)[0][0]
    percentile_95_threshold = percentile_thresholds[percentile_95_idx]
    percentage_below_95 = (position_errors_array < percentile_95_threshold).sum() / len(position_errors_array) * 100
    print(f"95th percentile threshold: {percentile_95_threshold:.6f} m ({percentage_below_95:.2f}% below)")
    
    if len(esi_values) > 1:
        print(f"Correlation coefficient: {correlation:.4f}")
    print("="*60)

if __name__ == '__main__':
    main()

