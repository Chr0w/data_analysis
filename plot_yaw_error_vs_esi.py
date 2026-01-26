#!/usr/bin/env python3
"""
Script to plot Yaw Error vs ESI.
Reads CSV files, calculates absolute yaw error, and creates a scatter plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from data_loader import load_data

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

def calculate_all_yaw_errors(df):
    """
    Calculate yaw errors for all rows in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with ground truth and AMCL yaw angles
    
    Returns:
    --------
    yaw_errors : numpy.ndarray
        Array of yaw errors in degrees
    """
    yaw_errors = []
    for idx, row in df.iterrows():
        error = calculate_yaw_error(
            row['ground_truth_yaw'], 
            row['amcl_yaw']
        )
        yaw_errors.append(error)
    return np.array(yaw_errors)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Plot Yaw Error vs ESI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python plot_yaw_error_vs_esi.py threshold
  python plot_yaw_error_vs_esi.py percentage
        '''
    )
    parser.add_argument('mode', choices=['threshold', 'percentage'],
                       help='Stopping mode: "threshold" stops when error >= threshold, "percentage" reads specified percentage of file')
    args = parser.parse_args()
    
    # Configuration parameters
    N = 30  # Number of files to read (starting with 1)
    error_threshold = 0.30  # Position error threshold in meters (for threshold mode)
    read_file_percentage = 0.33  # Percentage of files to read
    
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
    
    # Check for required columns
    required_cols = ['ground_truth_yaw', 'amcl_yaw', 'esi']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(combined_df.columns)}")
        sys.exit(1)
    
    # Calculate yaw errors for all entries
    print("Calculating yaw errors...")
    yaw_errors_array = calculate_all_yaw_errors(combined_df)
    
    # Get ESI values
    esi_values = combined_df['esi'].values
    
    # Create title suffix based on mode
    if args.mode == 'threshold':
        title_suffix = f'(N={N} files, error threshold={error_threshold}m)'
    else:
        title_suffix = f'(N={N} files, {read_file_percentage*100:.0f}% of each file)'
    
    # Create figure
    print("\nCreating plot...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot: Yaw Error vs ESI
    ax.scatter(esi_values, yaw_errors_array, 
               color='green', alpha=0.6, s=50, edgecolors='darkgreen', linewidth=0.5)
    ax.set_xlabel('ESI', fontsize=12)
    ax.set_ylabel('Yaw Error [deg]', fontsize=12)
    ax.set_title(f'Yaw Error vs ESI\n{title_suffix}', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add correlation
    corr_esi = None
    if len(esi_values) > 1:
        corr_esi = np.corrcoef(esi_values, yaw_errors_array)[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {corr_esi:.3f}', 
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(data_folder, 'yaw_error_vs_esi.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    print(f"Total measurements: {len(yaw_errors_array)}")
    print(f"\nESI range: [{min(esi_values):.4f}, {max(esi_values):.4f}]")
    if corr_esi is not None:
        print(f"Correlation (ESI): {corr_esi:.4f}")
    print(f"\nYaw Error range: [{min(yaw_errors_array):.6f}°, {max(yaw_errors_array):.6f}°]")
    print("="*60)

if __name__ == '__main__':
    main()
