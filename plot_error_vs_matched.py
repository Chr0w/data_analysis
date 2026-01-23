#!/usr/bin/env python3
"""
Script to plot Position Error vs ESI and normalized Number of Matched.
Reads CSV files, calculates absolute position error, and creates two stacked plots.
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
        description='Plot Position Error vs Number of Matched',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_error_vs_matched.py threshold
  python plot_error_vs_matched.py percentage
        '''
    )
    parser.add_argument('mode', choices=['threshold', 'percentage'],
                       help='Stopping mode: "threshold" stops when error >= threshold, "percentage" reads specified percentage of file')
    args = parser.parse_args()
    
    # Configuration parameters
    N = 30  # Number of files to read (starting with 1)
    error_threshold = 0.30  # Position error threshold in meters
    read_file_percentage = 0.5  # Percentage of files to read
    
    # Path to the folder containing CSV files
    data_folder = '/home/mircrda/pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl'
    
    # Load data using shared function (already skips first 100 rows)
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
    
    # Check for required columns
    required_cols = ['no_matched', 'esi']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(combined_df.columns)}")
        sys.exit(1)
    
    # Get data
    matched_values = combined_df['no_matched'].values
    esi_values = combined_df['esi'].values
    
    # Normalize matched values to 0-1 range
    matched_min = matched_values.min()
    matched_max = matched_values.max()
    if matched_max > matched_min:
        matched_normalized = (matched_values - matched_min) / (matched_max - matched_min)
    else:
        matched_normalized = np.zeros_like(matched_values)
    
    # Create title suffix based on mode
    if args.mode == 'threshold':
        title_suffix = f'(N={N} files, error threshold={error_threshold}m)'
    else:
        title_suffix = f'(N={N} files, {read_file_percentage*100:.0f}% of each file)'
    
    # Create figure with 2 subplots stacked vertically
    print("\nCreating plots...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Initialize correlation variables
    corr_esi = None
    corr_matched = None
    
    # Top plot: Position Error vs ESI (green)
    axes[0].scatter(esi_values, position_errors_array, 
                    color='green', alpha=0.6, s=50, edgecolors='darkgreen', linewidth=0.5)
    axes[0].set_xlabel('ESI', fontsize=12)
    axes[0].set_ylabel('Position Error [m]', fontsize=12)
    axes[0].set_title(f'Position Error vs ESI\n{title_suffix}', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # Add correlation for ESI
    if len(esi_values) > 1:
        corr_esi = np.corrcoef(esi_values, position_errors_array)[0, 1]
        axes[0].text(0.05, 0.95, f'Correlation: {corr_esi:.3f}', 
                    transform=axes[0].transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Bottom plot: Position Error vs Normalized Number of Matched (blue)
    axes[1].scatter(matched_normalized, position_errors_array, 
                    color='blue', alpha=0.6, s=50, edgecolors='darkblue', linewidth=0.5)
    axes[1].set_xlabel('Normalized Number of Matched', fontsize=12)
    axes[1].set_ylabel('Position Error [m]', fontsize=12)
    axes[1].set_title(f'Position Error vs Normalized Number of Matched\n{title_suffix}', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add correlation for normalized matched
    if len(matched_normalized) > 1:
        corr_matched = np.corrcoef(matched_normalized, position_errors_array)[0, 1]
        axes[1].text(0.05, 0.95, f'Correlation: {corr_matched:.3f}', 
                     transform=axes[1].transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(data_folder, 'position_error_vs_esi_and_matched.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(f"Total measurements: {len(position_errors_array)}")
    print(f"\nESI range: [{min(esi_values):.4f}, {max(esi_values):.4f}]")
    if corr_esi is not None:
        print(f"Correlation (ESI): {corr_esi:.4f}")
    print(f"\nNumber of Matched range: [{min(matched_values)}, {max(matched_values)}]")
    print(f"Normalized Number of Matched range: [{min(matched_normalized):.4f}, {max(matched_normalized):.4f}]")
    if corr_matched is not None:
        print(f"Correlation (normalized no_matched): {corr_matched:.4f}")
    print(f"\nPosition Error range: [{min(position_errors_array):.6f}, {max(position_errors_array):.6f}]")
    print("="*60)

if __name__ == '__main__':
    main()
