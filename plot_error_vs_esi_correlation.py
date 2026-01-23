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

def get_b_value(esi, b1, b2, b3, b4, b5, b6):
    """
    Determine which b value to use based on ESI.
    
    Parameters:
    -----------
    esi : float
        ESI value
    b1, b2, b3, b4, b5 : float
        Probability values for different ESI ranges
    
    Returns:
    --------
    b : float
        The appropriate b value for the given ESI
    """
    if esi < 0.1:
        return b1
    elif esi < 0.2:
        return b2
    elif esi < 0.3:
        return b3
    elif esi < 0.4:
        return b4
    elif esi < 0.5:
        return b5
    else:
        return b6

def update_lost_track_probability(accumulated_prob, b):
    """
    Update the accumulated probability and calculate probability of lost track.
    
    Parameters:
    -----------
    accumulated_prob : float
        Current accumulated probability of still being on track
    b : float
        Probability value for this sample. If negative (b6), reduces P(lost) by 5%
    
    Returns:
    --------
    new_accumulated_prob : float
        Updated accumulated probability of still being on track
    probability_of_lost_track : float
        Probability of being lost (1 - accumulated_prob)
    """
    if b < 0:
        # b6 case: reduce probability of being lost by 5%
        # P(lost)_new = P(lost)_old * 0.95
        # accumulated_prob_new = 1 - 0.95 * (1 - accumulated_prob_old)
        # accumulated_prob_new = 0.05 + 0.95 * accumulated_prob_old
        new_accumulated_prob = 0.05 + 0.95 * accumulated_prob
    else:
        # Normal case: multiply by (1 - b)
        new_accumulated_prob = accumulated_prob * (1 - b)
    
    probability_of_lost_track = 1 - new_accumulated_prob
    return new_accumulated_prob, probability_of_lost_track

def calculate_confusion_matrix(predictions, actual_errors, error_threshold):
    """
    Calculate confusion matrix metrics for lost track prediction.
    
    Parameters:
    -----------
    predictions : list of bool
        List of predictions (True = predicted lost, False = predicted not lost)
    actual_errors : list of float
        List of actual position errors
    error_threshold : float
        Error threshold in meters (above this = actually lost)
    
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
    tp = 0  # Predicted lost (True) AND actually lost (error > threshold)
    fp = 0  # Predicted lost (True) BUT not actually lost (error <= threshold)
    tn = 0  # Predicted not lost (False) AND not actually lost (error <= threshold)
    fn = 0  # Predicted not lost (False) BUT actually lost (error > threshold)
    
    for pred, error in zip(predictions, actual_errors):
        actually_lost = error > error_threshold
        
        if pred and actually_lost:
            tp += 1
        elif pred and not actually_lost:
            fp += 1
        elif not pred and not actually_lost:
            tn += 1
        else:  # not pred and actually_lost
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
    
    if tp + fp + tn + fn > 0:
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        print(f"Accuracy: {accuracy:.4f} ((TP + TN) / Total)")
    
    print("="*60)

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
    

    show_position_error= True
    show_percentile_plot = False
    show_histogram = False
    predict_lost_track = True
    
    # Configuration parameters
    N = 15  # Number of files to read (starting with 1)
    M = 20  # Sample every M'th entry (starting with 50)
    error_threshold = 0.35  # Position error threshold in meters
    read_file_percentage = 1.0  # Percentage of files to read
    ESI_threshold = 0.35  # ESI threshold in meters
    bin_threshold = 0.35  # Distance threshold in meters that determines which histogram to place the point in 
    probability_error_threshold = 0.95

    probability_of_lost_track = 0.0
    acculumated_probability_of_lost_track = 1.0  # Start at 1.0 (probability of still being on track)

    common_factor = 1

    b1 = 0.5 * common_factor
    b2 = 0.287 * common_factor
    b3 = 0.105 * common_factor
    b4 = 0.027 * common_factor
    b5 = 0.001 * common_factor
    b6 = -0.05




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
    
    if show_percentile_plot:
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
    mark_green = []  # Track which entries should be marked bright green
    
    # Reset accumulated probability (start on track, so probability of being lost = 0)
    acculumated_probability_of_lost_track = 1.0
    current_file_id = None
    
    print(f"Sampling every {M}'th entry and getting position error...")
    print("\nProbability tracking:")
    print("-" * 60)
    for idx in range(0, len(combined_df), M):
        # Get the absolute position error at this specific measurement
        position_error = position_errors[idx]
        
        # Get ESI value at this index
        esi = combined_df.iloc[idx]['esi']
        
        # Get file_id for this entry
        file_id = combined_df.iloc[idx]['file_id']
        
        if predict_lost_track:
            # Reset probability when switching to a new file
            if current_file_id is not None and file_id != current_file_id:
                acculumated_probability_of_lost_track = 1.0
                print(f"--- Switched to file {file_id}, resetting probability ---")
            
            current_file_id = file_id
            
            # Get b value based on ESI
            b = get_b_value(esi, b1, b2, b3, b4, b5, b6)
            
            # Update probability
            acculumated_probability_of_lost_track, probability_of_lost_track = update_lost_track_probability(
                acculumated_probability_of_lost_track, b
            )
            
            # Print probability for this entry
            print(f"Sample {idx // M + 1} (File {file_id}): ESI={esi:.4f}, b={b:.4f}, P(lost)={probability_of_lost_track:.6f}")
            
            # Check if we should mark this entry green
            should_mark_green = probability_of_lost_track >= probability_error_threshold
            mark_green.append(should_mark_green)
        else:
            # If not predicting lost track, don't mark any entries green
            mark_green.append(False)
            
        esi_values.append(esi)
        position_error_values.append(position_error)
        line_numbers.append(idx)  # Store the line number/index
        
        # if (idx // M + 1) % 10 == 0:
        #     print(f"  Processed {idx // M + 1} samples...")
    
    print("-" * 60)
    print(f"Total samples: {len(esi_values)}")
    
    # Calculate and print confusion matrix if predictions were made
    if predict_lost_track:
        tp, fp, tn, fn = calculate_confusion_matrix(mark_green, position_error_values, error_threshold)
        print_confusion_matrix_results(tp, fp, tn, fn)
        
    if show_position_error:
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
        
        # Mark entries that meet probability threshold in bright green
        if any(mark_green):
            esi_green = [esi_values[i] for i in range(len(esi_values)) if mark_green[i]]
            pos_error_green = [position_error_values[i] for i in range(len(esi_values)) if mark_green[i]]
            plt.scatter(esi_green, pos_error_green, 
                       color='lime', s=100, marker='.', 
                       edgecolors='darkgreen', linewidths=2, zorder=10,
                       label=f'P(lost) ≥ {probability_error_threshold}')
        
        # Add colorbar to show line number mapping
        cbar = plt.colorbar(scatter)
        cbar.set_label('Line Number (Blue=Early, Red=Late)', fontsize=10)
        
        # Add legend if we have green markers
        if any(mark_green):
            plt.legend(loc='upper right', fontsize=10)
        
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
    
    if show_histogram:
        # Create histograms based on bin_threshold
        print(f"\nCreating histograms with bin_threshold={bin_threshold}m...")
        
        # Determine ESI range and create exactly 10 bins (each 0.1 wide)
        esi_min = min(esi_values)
        esi_max = max(esi_values)
        # Round down to nearest 0.1 for clean bin edges
        esi_min_bin = np.floor(esi_min * 10) / 10
        # Create exactly 10 bins of 0.1 width
        num_bins = 10
        bin_width = 0.1
        bin_edges = np.arange(esi_min_bin, esi_min_bin + num_bins * bin_width + 0.001, bin_width)
        
        # Separate points into above and below threshold
        esi_above = [esi_values[i] for i in range(len(esi_values)) if position_error_values[i] > bin_threshold]
        esi_below = [esi_values[i] for i in range(len(esi_values)) if position_error_values[i] <= bin_threshold]
        
        # Calculate histograms
        counts_above, _ = np.histogram(esi_above, bins=bin_edges)
        counts_below, _ = np.histogram(esi_below, bins=bin_edges)
        
        # Calculate bin centers for plotting
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Print counts for both histograms
        print("\n" + "="*60)
        print("Histogram Counts for All Bins:")
        print("="*60)
        below_label = f'Error ≤ {bin_threshold:.2f}m'
        above_label = f'Error > {bin_threshold:.2f}m'
        print(f"{'Bin Center':<12} {'Bin Range':<20} {below_label:<20} {above_label:<20} {'Total':<10} {'Ratio':<10}")
        print("-"*60)
        for i, center in enumerate(bin_centers):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i+1]
            count_below = counts_below[i]
            count_above = counts_above[i]
            total = count_below + count_above
            ratio = count_above / total if total > 0 else 0.0
            ratio_str = f'{ratio:.3f}' if total > 0 else '0.000'
            print(f"{center:<12.2f} [{bin_start:.2f}, {bin_end:.2f})  {count_below:<20} {count_above:<20} {total:<10} {ratio_str:<10}")
        print("="*60)
        
        # Create histogram plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot histograms overlaid on top of each other
        width = bin_width * 0.8  # Slightly narrower bars for spacing
        bars_below = ax.bar(bin_centers, counts_below, width=width,
                            alpha=0.7, label=f'Error ≤ {bin_threshold}m',
                            color='blue', edgecolor='black', linewidth=1)
        bars_above = ax.bar(bin_centers, counts_above, width=width,
                            alpha=0.7, label=f'Error > {bin_threshold}m', 
                            color='red', edgecolor='black', linewidth=1)
        
        # Calculate and display ratios above each bar
        max_count = max(counts_above.max() if len(counts_above) > 0 else 0,
                    counts_below.max() if len(counts_below) > 0 else 0)
        
        for i, (center, count_above, count_below) in enumerate(zip(bin_centers, counts_above, counts_below)):
            total = count_above + count_below
            if total > 0:
                ratio = count_above / total
                # Display ratio above the bar (at the top of the higher bar)
                max_height = max(count_above, count_below)
                ax.text(center, max_height + max_count * 0.05,
                    f'{ratio:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('ESI', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Histogram of ESI Values\n(Error > {bin_threshold}m vs Error ≤ {bin_threshold}m)', fontsize=14)
        ax.set_xticks(bin_centers)
        ax.set_xticklabels([f'{bc:.2f}' for bc in bin_centers], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save the histogram plot
        hist_output_path = os.path.join(data_folder, 'esi_histogram.png')
        plt.savefig(hist_output_path, dpi=300, bbox_inches='tight')
        print(f"Histogram plot saved to {hist_output_path}")
        
        # Show the histogram plot
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

