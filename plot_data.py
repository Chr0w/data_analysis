#!/usr/bin/env python3
"""
Script to plot data from sim_test_1.csv as line plots.
Creates one subplot for each numeric column, using row index (line number) as x-axis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math

def main():
    # Path to the CSV file
    csv_path = '/home/chrdam/data_analysis/data/2026/sim_test_1.csv'
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        sys.exit(1)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        print(f"Loaded {len(df)} rows from {csv_path}")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)
    
    # Get numeric columns (exclude timestamp and specified columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Columns to exclude
    exclude_cols = ['timestamp', 'total_messages', 'duration_s', 'msg_rate_hz', 'ESI_range']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if not numeric_cols:
        print("Error: No numeric columns found to plot")
        sys.exit(1)
    
    print(f"Plotting {len(numeric_cols)} numeric columns: {numeric_cols}")
    
    # Calculate grid size for subplots
    n_cols = len(numeric_cols)
    n_rows = math.ceil(math.sqrt(n_cols))
    n_cols_grid = math.ceil(n_cols / n_rows)
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols_grid, figsize=(5*n_cols_grid, 4*n_rows))
    fig.suptitle('Line Plots: Values vs Row Index (Line Number)', fontsize=16, y=0.995)
    
    # Flatten axes array if needed
    if n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    # Create row index (line number, starting from 0)
    row_indices = df.index.values
    
    # Plot each numeric column
    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        ax.plot(row_indices, df[col], marker='o', linewidth=2, markersize=6, alpha=0.7)
        ax.set_xlabel('Row Index (Line Number)', fontsize=10)
        ax.set_ylabel(col, fontsize=10)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(row_indices)
    
    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    # Increase vertical spacing between subplots
    plt.subplots_adjust(hspace=0.5, top=0.93, bottom=0.05, left=0.1, right=0.95)
    
    # Save the plot
    output_path = '/home/chrdam/data_analysis/line_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
