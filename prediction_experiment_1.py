#!/usr/bin/env python3
"""
Prediction Experiment 1: Test exponential function for lost track prediction.
Generates 100 subsamples of 20-second windows and predicts lost track using exponential ESI function.
"""

import pandas as pd
import numpy as np
import os
import sys
import random
from data_loader import load_data, calculate_position_error

def exponential_b_value(esi, a=0.8, k=8.2):
    """
    Calculate b value using exponential function: b(esi) = a * exp(-k * esi)
    
    Parameters:
    -----------
    esi : float
        ESI value
    a : float
        Amplitude parameter (default: 0.8)
    k : float
        Decay rate parameter (default: 8.2)
    
    Returns:
    --------
    b : float
        The b value for the given ESI
    """
    return a * np.exp(-k * esi)

def update_lost_track_probability(accumulated_prob, b):
    """
    Update the accumulated probability and calculate probability of lost track.
    
    Parameters:
    -----------
    accumulated_prob : float
        Current accumulated probability of still being on track
    b : float
        Probability value for this sample
    
    Returns:
    --------
    new_accumulated_prob : float
        Updated accumulated probability of still being on track
    probability_of_lost_track : float
        Probability of being lost (1 - accumulated_prob)
    """
    # Normal case: multiply by (1 - b)
    new_accumulated_prob = accumulated_prob * (1 - b)
    
    probability_of_lost_track = 1 - new_accumulated_prob
    return new_accumulated_prob, probability_of_lost_track

def get_20_second_window(df_file, timestamp_col='timestamp', target_duration=20.0, tolerance=0.5):
    """
    Extract a random 20-second window from a file.
    
    Parameters:
    -----------
    df_file : pandas.DataFrame
        Dataframe for a single file
    timestamp_col : str
        Name of the timestamp column
    target_duration : float
        Target duration in seconds (default: 20.0)
    tolerance : float
        Allowed deviation from target duration (default: 0.5)
    
    Returns:
    --------
    window_df : pandas.DataFrame or None
        Dataframe containing the 20-second window, or None if not possible
    """
    if timestamp_col not in df_file.columns:
        return None
    
    # Convert timestamp to numeric (assuming it's in seconds or can be converted)
    timestamps = pd.to_numeric(df_file[timestamp_col], errors='coerce')
    
    # Remove rows with invalid timestamps
    valid_mask = pd.notna(timestamps)
    if valid_mask.sum() < 2:
        return None
    
    df_valid = df_file[valid_mask].copy()
    timestamps_valid = timestamps[valid_mask].values
    
    # Normalize timestamps to start from 0 (relative to first timestamp)
    first_timestamp = timestamps_valid[0]
    timestamps_relative = timestamps_valid - first_timestamp
    
    # Get the total duration
    total_duration = timestamps_relative[-1] - timestamps_relative[0]
    
    # Check if we have enough data for a 20-second window
    min_duration = target_duration - tolerance
    if total_duration < min_duration:
        return None
    
    # Calculate maximum start time to ensure we can get at least min_duration
    max_start_time = total_duration - min_duration
    
    # Randomly select a start time
    start_time = random.uniform(0, max_start_time)
    end_time = start_time + target_duration
    
    # Find indices within the window
    mask = (timestamps_relative >= start_time) & (timestamps_relative <= end_time)
    window_df = df_valid[mask].copy()
    
    # Verify the actual duration is within tolerance
    if len(window_df) < 2:
        return None
    
    # Calculate actual duration using timestamps from the window
    window_timestamps = pd.to_numeric(window_df[timestamp_col], errors='coerce')
    window_timestamps_valid = window_timestamps[pd.notna(window_timestamps)].values
    
    if len(window_timestamps_valid) < 2:
        return None
    
    actual_duration = window_timestamps_valid[-1] - window_timestamps_valid[0]
    if abs(actual_duration - target_duration) > tolerance:
        return None
    
    return window_df

def process_subsample(df_file, file_id, a=0.8, k=8.2, probability_threshold=0.95, error_threshold=0.5):
    """
    Process a single subsample and determine if it's predicted as lost track and actually lost track.
    
    Parameters:
    -----------
    df_file : pandas.DataFrame
        Dataframe for a single file (20-second window)
    file_id : int
        File ID for tracking
    a : float
        Exponential function amplitude
    k : float
        Exponential function decay rate
    probability_threshold : float
        Threshold for positive prediction (default: 0.95)
    error_threshold : float
        Position error threshold in meters (default: 0.5)
    
    Returns:
    --------
    is_predicted_positive : bool
        True if predicted as lost track (probability > threshold)
    is_actually_positive : bool
        True if any position error exceeds error_threshold
    final_probability : float
        Final probability of lost track
    max_position_error : float
        Maximum position error in the window
    """
    if 'esi' not in df_file.columns:
        return False, False, 0.0, 0.0
    
    # Check required columns for position error calculation
    required_cols = ['ground_truth_x', 'ground_truth_y', 'amcl_x', 'amcl_y']
    if not all(col in df_file.columns for col in required_cols):
        return False, False, 0.0, 0.0
    
    # Initialize accumulated probability (start on track)
    accumulated_prob = 1.0
    
    # Track maximum position error
    max_position_error = 0.0
    
    # Process each row in the subsample
    for idx, row in df_file.iterrows():
        esi = row['esi']
        
        # Calculate b value using exponential function
        if esi > 0.40:
            b = 0.0
        else:
            b = exponential_b_value(esi, a=a, k=k)
        
        # Update probability
        accumulated_prob, probability_of_lost_track = update_lost_track_probability(
            accumulated_prob, b
        )
        
        # Calculate position error
        position_error = calculate_position_error(
            row['ground_truth_x'], row['ground_truth_y'],
            row['amcl_x'], row['amcl_y']
        )
        max_position_error = max(max_position_error, position_error)
    
    # Determine if positive (predicted lost track)
    final_probability = 1.0 - accumulated_prob
    is_predicted_positive = final_probability > probability_threshold
    
    # Determine if actually positive (any position error exceeds threshold)
    is_actually_positive = max_position_error > error_threshold
    
    return is_predicted_positive, is_actually_positive, final_probability, max_position_error

def calculate_confusion_matrix(predictions, actuals):
    """
    Calculate confusion matrix metrics for lost track prediction.
    
    Parameters:
    -----------
    predictions : list of bool
        List of predictions (True = predicted lost, False = predicted not lost)
    actuals : list of bool
        List of actual values (True = actually lost, False = not actually lost)
    
    Returns:
    --------
    tp : int
        True Positives (predicted lost AND actually lost)
    fp : int
        False Positives (predicted lost BUT not actually lost)
    tn : int
        True Negatives (predicted not lost AND not actually lost)
    fn : int
        False Negatives (predicted not lost BUT actually lost)
    """
    tp = 0  # Predicted lost (True) AND actually lost (True)
    fp = 0  # Predicted lost (True) BUT not actually lost (False)
    tn = 0  # Predicted not lost (False) AND not actually lost (False)
    fn = 0  # Predicted not lost (False) BUT actually lost (True)
    
    for pred, actual in zip(predictions, actuals):
        if pred and actual:
            tp += 1
        elif pred and not actual:
            fp += 1
        elif not pred and not actual:
            tn += 1
        else:  # not pred and actual
            fn += 1
    
    return tp, fp, tn, fn

def print_confusion_matrix_results(tp, fp, tn, fn):
    """
    Print confusion matrix results to terminal.
    
    Parameters:
    -----------
    tp, fp, tn, fn : int
        True Positives, False Positives, True Negatives, False Negatives
    """
    total = tp + fp + tn + fn
    
    print("\n" + "="*60)
    print("Confusion Matrix Results:")
    print("="*60)
    print(f"True Positives  (TP): {tp:4d} - Predicted lost AND actually lost")
    print(f"False Positives (FP): {fp:4d} - Predicted lost BUT not actually lost")
    print(f"True Negatives  (TN): {tn:4d} - Predicted not lost AND not actually lost")
    print(f"False Negatives (FN): {fn:4d} - Predicted not lost BUT actually lost")
    print("-" * 60)
    print(f"Total samples: {total}")
    print()
    
    # Calculate metrics
    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"Precision: {precision:.4f} (TP / (TP + FP))")
    else:
        print("Precision: N/A (no positive predictions)")
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"Recall (Sensitivity): {recall:.4f} (TP / (TP + FN))")
    else:
        print("Recall: N/A (no actual positives)")
    
    if tn + fp > 0:
        specificity = tn / (tn + fp)
        print(f"Specificity: {specificity:.4f} (TN / (TN + FP))")
    else:
        print("Specificity: N/A (no actual negatives)")
    
    if total > 0:
        accuracy = (tp + tn) / total
        print(f"Accuracy: {accuracy:.4f} ((TP + TN) / Total)")
    
    if tp + fp + tn + fn > 0:
        f1_score = 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        print(f"F1 Score: {f1_score:.4f} (2*TP / (2*TP + FP + FN))")
    
    print("="*60)

def main():
    # Configuration parameters
    N = 30  # Number of files to read (starting with 1)
    M = 20  # Sample every M'th entry
    read_file_percentage = 0.75  # Percentage of files to read
    num_subsamples = 100  # Number of subsamples to generate
    
    # Exponential function parameters
    a = 1.0 # 0.8
    k = 10.5 # 8.2
    
    # Probability threshold for positive prediction
    probability_threshold = 0.99
    
    # Position error threshold for actual lost track (in meters)
    error_threshold = 0.5
    
    # Path to the folder containing CSV files
    data_folder = '/home/mircrda/pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl'
    
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
    required_cols = ['esi', 'timestamp']
    missing_cols = [col for col in required_cols if col not in combined_df.columns]
    
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(combined_df.columns)}")
        sys.exit(1)
    
    # Sample every M'th entry
    print(f"Sampling every {M}'th entry...")
    sampled_df = combined_df.iloc[::M].copy()
    print(f"Sampled {len(sampled_df)} entries from {len(combined_df)} total entries")
    
    # Group by file_id
    file_groups = {}
    for file_id in sampled_df['file_id'].unique():
        file_groups[file_id] = sampled_df[sampled_df['file_id'] == file_id].copy()
        print(f"File {file_id}: {len(file_groups[file_id])} entries")
    
    # Generate 100 subsamples
    print(f"\nGenerating {num_subsamples} subsamples...")
    print("=" * 60)
    
    results = []
    attempts = 0
    max_attempts = num_subsamples * 10  # Prevent infinite loop
    
    while len(results) < num_subsamples and attempts < max_attempts:
        attempts += 1
        
        # Select a random file
        file_id = random.choice(list(file_groups.keys()))
        df_file = file_groups[file_id]
        
        # Get 20-second window
        window_df = get_20_second_window(df_file, timestamp_col='timestamp')
        
        if window_df is None or len(window_df) < 2:
            continue
        
        # Process the subsample
        is_predicted_positive, is_actually_positive, final_probability, max_position_error = process_subsample(
            window_df, file_id, a=a, k=k, probability_threshold=probability_threshold, error_threshold=error_threshold
        )
        
        results.append({
            'file_id': file_id,
            'num_entries': len(window_df),
            'is_predicted_positive': is_predicted_positive,
            'is_actually_positive': is_actually_positive,
            'probability': final_probability,
            'max_position_error': max_position_error
        })
        
        if len(results) % 10 == 0:
            print(f"Generated {len(results)}/{num_subsamples} subsamples...")
    
    if len(results) < num_subsamples:
        print(f"Warning: Only generated {len(results)} subsamples out of {num_subsamples} requested")
    
    # Calculate confusion matrix
    predictions = [r['is_predicted_positive'] for r in results]
    actuals = [r['is_actually_positive'] for r in results]
    tp, fp, tn, fn = calculate_confusion_matrix(predictions, actuals)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    predicted_positive_count = sum(1 for r in results if r['is_predicted_positive'])
    predicted_negative_count = len(results) - predicted_positive_count
    actually_positive_count = sum(1 for r in results if r['is_actually_positive'])
    actually_negative_count = len(results) - actually_positive_count
    
    print(f"Total subsamples: {len(results)}")
    print(f"\nPredictions:")
    print(f"  Positive predictions (P(lost) > {probability_threshold}): {predicted_positive_count}")
    print(f"  Negative predictions (P(lost) <= {probability_threshold}): {predicted_negative_count}")
    print(f"\nActual (position error > {error_threshold}m):")
    print(f"  Actually lost track: {actually_positive_count}")
    print(f"  Not lost track: {actually_negative_count}")
    
    # Print confusion matrix
    print_confusion_matrix_results(tp, fp, tn, fn)
    
    print("\n" + "-" * 60)
    print("Probability Statistics:")
    print("-" * 60)
    probabilities = [r['probability'] for r in results]
    print(f"Mean probability: {np.mean(probabilities):.6f}")
    print(f"Median probability: {np.median(probabilities):.6f}")
    print(f"Min probability: {np.min(probabilities):.6f}")
    print(f"Max probability: {np.max(probabilities):.6f}")
    print(f"Std deviation: {np.std(probabilities):.6f}")
    
    print("\n" + "-" * 60)
    print("Position Error Statistics:")
    print("-" * 60)
    max_errors = [r['max_position_error'] for r in results]
    print(f"Mean max error: {np.mean(max_errors):.6f} m")
    print(f"Median max error: {np.median(max_errors):.6f} m")
    print(f"Min max error: {np.min(max_errors):.6f} m")
    print(f"Max max error: {np.max(max_errors):.6f} m")
    print(f"Std deviation: {np.std(max_errors):.6f} m")
    
    print("\n" + "-" * 60)
    print("Subsample Details (first 20):")
    print("-" * 60)
    print(f"{'File ID':<10} {'Entries':<10} {'Pred':<6} {'Actual':<6} {'Prob':<12} {'Max Err':<10}")
    print("-" * 60)
    for i, r in enumerate(results[:20]):
        pred_str = "Yes" if r['is_predicted_positive'] else "No"
        actual_str = "Yes" if r['is_actually_positive'] else "No"
        print(f"{r['file_id']:<10} {r['num_entries']:<10} {pred_str:<6} {actual_str:<6} {r['probability']:<12.6f} {r['max_position_error']:<10.6f}")
    
    if len(results) > 20:
        print(f"... and {len(results) - 20} more subsamples")
    
    print("=" * 60)

if __name__ == '__main__':
    main()

