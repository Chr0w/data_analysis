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
import argparse
from data_loader import load_data, calculate_position_error

def calculate_lost_track_probability(esi_values, a=0.8, k=8.2, offset=0.05):
    """
    Calculate probability of lost track using cumulative integral formula:
    p_lost(esi) = integral(a * exp(-k * esi)) - offset
    
    For discrete data, the integral is approximated as a cumulative sum.
    
    Parameters:
    -----------
    esi_values : array-like
        Array of ESI values
    a : float
        Amplitude parameter (default: 0.8)
    k : float
        Decay rate parameter (default: 8.2)
    offset : float
        Offset to subtract from the integral (default: 0.05)
    
    Returns:
    --------
    p_lost : float
        Probability of lost track, clamped to [0, 1]
    """
    # Calculate cumulative integral: sum of a * exp(-k * esi) for all ESI values
    integral = np.sum(a * np.exp(-k * np.array(esi_values)))
    
    # Subtract offset
    p_lost = integral - offset
    
    # Clamp to valid probability range [0, 1]
    p_lost = max(0.0, min(1.0, p_lost))
    
    return p_lost

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
    
    # Track maximum position error
    max_position_error = 0.0
    
    # Collect ESI values
    esi_values = []
    
    # Process each row in the subsample
    for idx, row in df_file.iterrows():
        esi = row['esi']
        esi_values.append(esi)
        
        # Calculate position error
        position_error = calculate_position_error(
            row['ground_truth_x'], row['ground_truth_y'],
            row['amcl_x'], row['amcl_y']
        )
        max_position_error = max(max_position_error, position_error)
    
    # Calculate probability of lost track using single formula
    final_probability = calculate_lost_track_probability(esi_values, a=a, k=k, offset=0.05)
    
    # Determine if positive (predicted lost track)
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

def generate_balanced_subsamples(file_groups, num_lost=50, num_not_lost=50, error_threshold=0.5, seed=None):
    """
    Generate a balanced set of subsamples (50 lost, 50 not lost).
    
    Parameters:
    -----------
    file_groups : dict
        Dictionary of file_id -> dataframe
    num_lost : int
        Number of lost track samples to generate (default: 50)
    num_not_lost : int
        Number of not lost track samples to generate (default: 50)
    error_threshold : float
        Position error threshold in meters to determine if actually lost
    seed : int or None
        Random seed for reproducibility
    
    Returns:
    --------
    subsamples : list
        List of (file_id, window_df) tuples, balanced with num_lost lost and num_not_lost not lost
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    lost_subsamples = []
    not_lost_subsamples = []
    attempts = 0
    max_attempts = (num_lost + num_not_lost) * 50  # Prevent infinite loop
    
    # Check required columns
    required_cols = ['ground_truth_x', 'ground_truth_y', 'amcl_x', 'amcl_y']
    
    while (len(lost_subsamples) < num_lost or len(not_lost_subsamples) < num_not_lost) and attempts < max_attempts:
        attempts += 1
        
        # Select a random file
        file_id = random.choice(list(file_groups.keys()))
        df_file = file_groups[file_id]
        
        # Get 20-second window
        window_df = get_20_second_window(df_file, timestamp_col='timestamp')
        
        if window_df is None or len(window_df) < 2:
            continue
        
        # Check if required columns exist
        if not all(col in window_df.columns for col in required_cols):
            continue
        
        # Calculate maximum position error to determine if actually lost
        max_position_error = 0.0
        for idx, row in window_df.iterrows():
            position_error = calculate_position_error(
                row['ground_truth_x'], row['ground_truth_y'],
                row['amcl_x'], row['amcl_y']
            )
            max_position_error = max(max_position_error, position_error)
        
        # Classify as lost or not lost
        is_actually_lost = max_position_error > error_threshold
        
        # Add to appropriate list if we still need samples of that type
        if is_actually_lost and len(lost_subsamples) < num_lost:
            lost_subsamples.append((file_id, window_df))
        elif not is_actually_lost and len(not_lost_subsamples) < num_not_lost:
            not_lost_subsamples.append((file_id, window_df))
    
    # Combine balanced sets
    subsamples = lost_subsamples + not_lost_subsamples
    
    # Shuffle to mix lost and not lost samples
    random.shuffle(subsamples)
    
    # Print balance information
    print(f"  Generated {len(lost_subsamples)} lost and {len(not_lost_subsamples)} not lost samples")
    
    return subsamples

def evaluate_parameters(a, k, subsamples, probability_threshold, error_threshold):
    """
    Evaluate parameters a and k using pre-generated subsamples.
    
    Parameters:
    -----------
    a : float
        Exponential function amplitude
    k : float
        Exponential function decay rate
    subsamples : list
        List of (file_id, window_df) tuples
    probability_threshold : float
        Threshold for positive prediction
    error_threshold : float
        Position error threshold in meters
    
    Returns:
    --------
    f1_score : float
        F1 score for this parameter combination
    tp, fp, tn, fn : int
        Confusion matrix values
    """
    results = []
    
    for file_id, window_df in subsamples:
        # Process the subsample
        is_predicted_positive, is_actually_positive, _, _ = process_subsample(
            window_df, file_id, a=a, k=k, 
            probability_threshold=probability_threshold, 
            error_threshold=error_threshold
        )
        
        results.append({
            'is_predicted_positive': is_predicted_positive,
            'is_actually_positive': is_actually_positive
        })
    
    if len(results) < len(subsamples):
        # Not enough samples, return poor score
        return 0.0, 0, 0, 0, 0
    
    # Calculate confusion matrix
    predictions = [r['is_predicted_positive'] for r in results]
    actuals = [r['is_actually_positive'] for r in results]
    tp, fp, tn, fn = calculate_confusion_matrix(predictions, actuals)
    
    # Calculate F1 score
    if tp + fp + fn == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * tp / (2 * tp + fp + fn)
    
    return f1_score, tp, fp, tn, fn

def optimize_parameters(training_subsamples, probability_threshold, error_threshold, max_runs=500):
    """
    Optimize parameters a and k to maximize F1 score using training subsamples.
    
    Parameters:
    -----------
    training_subsamples : list
        List of (file_id, window_df) tuples for training
    probability_threshold : float
        Threshold for positive prediction
    error_threshold : float
        Position error threshold in meters
    max_runs : int
        Maximum number of function evaluations
    
    Returns:
    --------
    best_params : dict
        Best parameters found
    best_f1 : float
        Best F1 score on training set
    best_confusion : tuple
        Confusion matrix at best parameters
    """
    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Optimizing a and k to maximize F1 score")
    print(f"Max evaluations: {max_runs}")
    print(f"Parameter bounds: a in [0.1, 2.0], k in [1.0, 20.0]")
    print("=" * 60)
    
    # Track evaluations
    evaluation_count = 0
    best_f1 = -1.0
    best_params = None
    best_confusion = None
    
    # Start with a coarse grid search
    print("Phase 1: Coarse grid search...")
    a_values = np.linspace(0.1, 2.0, 10)
    k_values = np.linspace(1.0, 20.0, 10)
    grid_evaluations = len(a_values) * len(k_values)
    
    if grid_evaluations <= max_runs:
        for a in a_values:
            for k in k_values:
                if evaluation_count >= max_runs:
                    break
                evaluation_count += 1
                f1_score, tp, fp, tn, fn = evaluate_parameters(
                    a, k, training_subsamples, probability_threshold, error_threshold
                )
                
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_params = {'a': a, 'k': k}
                    best_confusion = (tp, fp, tn, fn)
                
                if evaluation_count % 10 == 0 or evaluation_count == 1:
                    print(f"Run {evaluation_count:4d}: a={a:.4f}, k={k:.4f}, F1={f1_score:.4f} (Best: {best_f1:.4f})")
            if evaluation_count >= max_runs:
                break
    
    # Then do random search with remaining evaluations
    remaining_runs = max_runs - evaluation_count
    if remaining_runs > 0:
        print(f"\nPerforming random search with {remaining_runs} remaining evaluations...")
        for i in range(remaining_runs):
            a = np.random.uniform(0.1, 2.0)
            k = np.random.uniform(1.0, 20.0)
            f1_score, tp, fp, tn, fn = evaluate_parameters(
                a, k, training_subsamples, probability_threshold, error_threshold
            )
            evaluation_count += 1
            
            if f1_score > best_f1:
                best_f1 = f1_score
                best_params = {'a': a, 'k': k}
                best_confusion = (tp, fp, tn, fn)
            
            if evaluation_count % 10 == 0:
                print(f"Run {evaluation_count:4d}: a={a:.4f}, k={k:.4f}, F1={f1_score:.4f} (Best: {best_f1:.4f})")
    
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total evaluations: {evaluation_count}")
    if best_params:
        print(f"Best parameters: a={best_params['a']:.6f}, k={best_params['k']:.6f}")
        print(f"Best F1 score: {best_f1:.6f}")
        if best_confusion:
            tp, fp, tn, fn = best_confusion
            print(f"\nConfusion matrix at best parameters:")
            print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    else:
        print("Warning: No valid parameters found")
    print("=" * 60)
    
    return best_params, best_f1, best_confusion

def main(optimize=False, lost_percentage=50):
    # Configuration parameters
    N = 30  # Number of files to read (starting with 1)
    M = 20  # Sample every M'th entry
    read_file_percentage = 0.75  # Percentage of files to read
    num_subsamples = 100  # Number of subsamples to generate
    
    # Calculate number of lost and not lost samples based on percentage
    num_lost = int(num_subsamples * lost_percentage / 100)
    num_not_lost = num_subsamples - num_lost
    
    # Exponential function parameters default values
    a = 1.67 # 0.8
    k = 19.33 # 8.2
    
    # Probability threshold for positive prediction
    probability_threshold = 0.99
    
    # Position error threshold for actual lost track (in meters)
    error_threshold = 0.5
    
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
    
    # Generate balanced evaluation subsamples
    print(f"\nGenerating balanced evaluation subsamples ({num_lost} lost, {num_not_lost} not lost, {lost_percentage}% lost)...")
    evaluation_subsamples = generate_balanced_subsamples(
        file_groups, num_lost=num_lost, num_not_lost=num_not_lost, error_threshold=error_threshold, seed=123
    )
    print(f"Generated {len(evaluation_subsamples)} evaluation subsamples (should be {num_subsamples}: {num_lost} lost + {num_not_lost} not lost)")
    
    # Optimize parameters if requested
    best_f1_train = None
    if optimize:
        # Generate balanced training subsamples
        print(f"\nGenerating balanced training subsamples ({num_lost} lost, {num_not_lost} not lost, {lost_percentage}% lost)...")
        training_subsamples = generate_balanced_subsamples(
            file_groups, num_lost=num_lost, num_not_lost=num_not_lost, error_threshold=error_threshold, seed=42
        )
        print(f"Generated {len(training_subsamples)} training subsamples (should be {num_subsamples}: {num_lost} lost + {num_not_lost} not lost)")
        
        # Optimize parameters using training subsamples
        best_params, best_f1_train, best_confusion_train = optimize_parameters(
            training_subsamples, probability_threshold, error_threshold, max_runs=500
        )
        
        # Use best parameters for final evaluation
        a = best_params['a']
        k = best_params['k']
        
        print(f"\nUsing optimized parameters: a={a:.6f}, k={k:.6f}")
        print(f"Training F1 score: {best_f1_train:.6f}")
    else:
        print(f"\nUsing default parameters: a={a:.6f}, k={k:.6f}")
        print("(Optimization skipped)")
    
    print(f"\nEvaluating on evaluation set...")
    print("=" * 60)
    
    # Evaluate on evaluation subsamples
    results = []
    for file_id, window_df in evaluation_subsamples:
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
    
    if len(results) < num_subsamples:
        print(f"Warning: Only generated {len(results)} subsamples out of {num_subsamples} requested")
    
    # Calculate confusion matrix
    predictions = [r['is_predicted_positive'] for r in results]
    actuals = [r['is_actually_positive'] for r in results]
    tp, fp, tn, fn = calculate_confusion_matrix(predictions, actuals)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (on evaluation set)")
    print("=" * 60)
    print(f"Parameters used: a={a:.6f}, k={k:.6f}")
    if best_f1_train is not None:
        print(f"Training F1 score: {best_f1_train:.6f}")
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
    parser = argparse.ArgumentParser(description='Prediction Experiment 1: Test exponential function for lost track prediction.')
    parser.add_argument('--optimize', action='store_true', 
                        help='Perform parameter optimization. If not set, uses default values (a=1.67, k=19.33).')
    parser.add_argument('--lost_percentage', type=int, default=50,
                        help='Percentage of lost track samples in the dataset (default: 50). Remaining will be not lost samples.')
    args = parser.parse_args()
    
    # Validate lost_percentage
    if args.lost_percentage < 0 or args.lost_percentage > 100:
        print(f"Error: lost_percentage must be between 0 and 100, got {args.lost_percentage}")
        sys.exit(1)
    
    main(optimize=args.optimize, lost_percentage=args.lost_percentage)

