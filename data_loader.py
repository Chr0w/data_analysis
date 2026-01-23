#!/usr/bin/env python3
"""
Shared data loading functions for AMCL analysis scripts.
"""

import pandas as pd
import numpy as np
import os
import sys

def calculate_position_error(gt_x, gt_y, amcl_x, amcl_y):
    """Calculate absolute position error (Euclidean distance)."""
    return np.sqrt((gt_x - amcl_x)**2 + (gt_y - amcl_y)**2)

def load_data(data_folder, N, mode='percentage', error_threshold=2.0, read_file_percentage=0.5):
    """
    Load CSV files from the specified folder.
    
    Parameters:
    -----------
    data_folder : str
        Path to folder containing CSV files (named 1.csv, 2.csv, etc.)
    N : int
        Number of files to read (starting from 1)
    mode : str
        'threshold' or 'percentage' - determines how to limit data from each file
    error_threshold : float
        Position error threshold in meters (used when mode='threshold')
    read_file_percentage : float
        Percentage of each file to read (used when mode='percentage')
    
    Returns:
    --------
    combined_df : pandas.DataFrame
        Combined dataframe with all loaded data
    """
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"Error: Folder not found at {data_folder}")
        sys.exit(1)
    
    # Read the first N CSV files using the selected mode
    all_data = []
    for i in range(1, N + 1):
        csv_path = os.path.join(data_folder, f'{i}.csv')
        if not os.path.exists(csv_path):
            print(f"Warning: File not found at {csv_path}, skipping...")
            continue
        
        try:
            # Read the CSV file, skipping the first 5 data rows (rows 1-5, keeping header in row 0)
            df_full = pd.read_csv(csv_path, skiprows=range(1, 100))
            df_full.columns = df_full.columns.str.strip()
            
            # Verify required columns exist
            required_cols = ['ground_truth_x', 'ground_truth_y', 'amcl_x', 'amcl_y']
            missing_cols = [col for col in required_cols if col not in df_full.columns]
            if missing_cols:
                print(f"Warning: Missing required columns in {csv_path}: {missing_cols}, skipping...")
                continue
            
            if mode == 'threshold':
                # Process rows until error threshold is reached
                valid_rows = []
                for idx, row in df_full.iterrows():
                    # Calculate position error for this row
                    error = calculate_position_error(
                        row['ground_truth_x'], row['ground_truth_y'],
                        row['amcl_x'], row['amcl_y']
                    )
                    
                    # Stop if error >= threshold
                    if error >= error_threshold:
                        print(f"File {i}: Stopped at row {idx} (error={error:.6f} >= threshold={error_threshold})")
                        break
                    
                    valid_rows.append(row)
                
                # Create dataframe with only valid rows
                if valid_rows:
                    df = pd.DataFrame(valid_rows)
                    # Add file_id column to track which file this data comes from
                    df['file_id'] = i
                    print(f"Loaded {len(df)} rows from {csv_path} (stopped before threshold)")
                    all_data.append(df)
                else:
                    print(f"Warning: No valid rows in {csv_path} (first row exceeded threshold)")
            
            elif mode == 'percentage':
                # Read specified percentage of rows
                total_rows = len(df_full)
                num_rows_to_read = int(total_rows * read_file_percentage)
                df = df_full.iloc[:num_rows_to_read].copy()
                # Add file_id column to track which file this data comes from
                df['file_id'] = i
                print(f"Loaded {len(df)} rows from {csv_path} ({read_file_percentage*100:.1f}% of {total_rows} total rows)")
                all_data.append(df)
            else:
                print(f"Error: Unknown mode '{mode}'. Use 'threshold' or 'percentage'")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error reading {csv_path}: {e}") 
            continue
    
    if not all_data:
        print("Error: No data files were successfully loaded")
        sys.exit(1)
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total rows in combined data: {len(combined_df)}")
    
    # Verify required columns exist
    required_cols = ['ground_truth_x', 'ground_truth_y', 'amcl_x', 'amcl_y', 'esi']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(combined_df.columns)}")
        sys.exit(1)
    
    return combined_df

def calculate_all_position_errors(df):
    """
    Calculate position errors for all rows in the dataframe.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with ground truth and AMCL positions
    
    Returns:
    --------
    position_errors : numpy.ndarray
        Array of position errors
    """
    position_errors = []
    for idx, row in df.iterrows():
        error = calculate_position_error(
            row['ground_truth_x'], row['ground_truth_y'],
            row['amcl_x'], row['amcl_y']
        )
        position_errors.append(error)
    return np.array(position_errors)

