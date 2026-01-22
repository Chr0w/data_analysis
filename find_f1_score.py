#!/usr/bin/env python3
"""
Script to find optimal ESI threshold using F1 score from Precision-Recall analysis.
Uses ESI to predict when AMCL's error is above a given threshold.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, classification_report
from data_loader import load_data, calculate_all_position_errors

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Find optimal ESI threshold using F1 score',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python find_f1_score.py threshold
  python find_f1_score.py percentage
        '''
    )
    parser.add_argument('mode', choices=['threshold', 'percentage'],
                       help='Stopping mode: "threshold" stops when error >= threshold, "percentage" reads specified percentage of file')
    args = parser.parse_args()
    
    # Configuration parameters
    N = 30  # Number of files to read (starting with 1)
    error_threshold = 0.50  # Position error threshold in meters (for prediction)
    read_file_percentage = 0.5  # Percentage of files to read
    min_recall = 0.7  # Minimum recall requirement for precision-optimal threshold
    
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
    esi_values_all = combined_df['esi'].values
    
    # Create binary labels: 1 if error >= threshold, 0 otherwise
    error_labels = (position_errors_array >= error_threshold).astype(int)
    
    # Check class balance
    print(f"\nClass Distribution:")
    print(f"Total samples: {len(error_labels)}")
    print(f"High error cases (error >= {error_threshold}m): {error_labels.sum()} ({error_labels.mean()*100:.2f}%)")
    print(f"Low error cases: {(1-error_labels).sum()} ({(1-error_labels).mean()*100:.2f}%)")
    
    # Calculate Precision-Recall curve
    # Note: We use -esi_values_all because lower ESI typically indicates higher error
    # If your data shows higher ESI = higher error, use esi_values_all directly
    print("\nCalculating Precision-Recall curve...")
    precision, recall, esi_thresholds_pr = precision_recall_curve(error_labels, -esi_values_all)
    avg_precision = average_precision_score(error_labels, -esi_values_all)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    
    # Find optimal threshold using different criteria
    optimal_idx_f1 = np.argmax(f1_scores)
    optimal_esi_threshold_f1 = -esi_thresholds_pr[optimal_idx_f1]  # Convert back (negated)
    
    # Also find threshold that maximizes precision while maintaining minimum recall
    valid_indices = recall[:-1] >= min_recall
    if valid_indices.any():
        # Get indices where recall >= min_recall
        valid_idx_array = np.where(valid_indices)[0]
        # Find the one with maximum precision among valid ones
        precision_valid = precision[:-1][valid_indices]
        optimal_idx_in_valid = np.argmax(precision_valid)
        # Map back to original index
        optimal_idx_prec = valid_idx_array[optimal_idx_in_valid]
        optimal_esi_threshold_prec = -esi_thresholds_pr[optimal_idx_prec]
        prec_idx = optimal_idx_prec
    else:
        optimal_esi_threshold_prec = None
        prec_idx = None
    
    # Print results
    print("\n" + "="*60)
    print("Precision-Recall Analysis Results:")
    print("="*60)
    print(f"Average Precision (AP): {avg_precision:.4f}")
    print(f"\nOptimal threshold (F1-maximizing):")
    print(f"  ESI threshold: {optimal_esi_threshold_f1:.4f}")
    print(f"  Precision: {precision[optimal_idx_f1]:.4f}")
    print(f"  Recall: {recall[optimal_idx_f1]:.4f}")
    print(f"  F1-score: {f1_scores[optimal_idx_f1]:.4f}")
    
    if optimal_esi_threshold_prec is not None:
        print(f"\nOptimal threshold (precision-max, recall >= {min_recall}):")
        print(f"  ESI threshold: {optimal_esi_threshold_prec:.4f}")
        print(f"  Precision: {precision[prec_idx]:.4f}")
        print(f"  Recall: {recall[prec_idx]:.4f}")
    print("="*60)
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Precision-Recall Curve
    ax1 = axes[0, 0]
    ax1.plot(recall, precision, linewidth=2, label=f'PR curve (AP = {avg_precision:.4f})')
    ax1.scatter(recall[optimal_idx_f1], precision[optimal_idx_f1], 
               color='red', s=100, zorder=5, 
               label=f'F1-optimal (ESI={optimal_esi_threshold_f1:.4f})')
    if optimal_esi_threshold_prec is not None:
        ax1.scatter(recall[prec_idx], precision[prec_idx], 
                   color='green', s=100, zorder=5, marker='s',
                   label=f'Prec-optimal (ESI={optimal_esi_threshold_prec:.4f})')
    ax1.axhline(y=0.5, color='k', linestyle='--', alpha=0.3, label='Baseline (random)')
    ax1.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax1.set_ylabel('Precision', fontsize=11)
    ax1.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Note: precision_recall_curve returns thresholds with length n, and precision/recall with length n+1
    # So we use thresholds as-is, and slice precision/recall to match
    esi_thresholds_plot = -esi_thresholds_pr  # Convert back (negated)
    precision_plot = precision[:-1]  # Remove last element to match thresholds length
    recall_plot = recall[:-1]  # Remove last element to match thresholds length
    
    # 2. Precision vs ESI Threshold
    ax2 = axes[0, 1]
    ax2.plot(esi_thresholds_plot, precision_plot, linewidth=2, label='Precision')
    ax2.axvline(x=optimal_esi_threshold_f1, color='red', linestyle='--', alpha=0.7, label='F1-optimal')
    ax2.set_xlabel('ESI Threshold', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision vs ESI Threshold', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Recall vs ESI Threshold
    ax3 = axes[1, 0]
    ax3.plot(esi_thresholds_plot, recall_plot, linewidth=2, label='Recall', color='orange')
    ax3.axvline(x=optimal_esi_threshold_f1, color='red', linestyle='--', alpha=0.7, label='F1-optimal')
    ax3.axhline(y=min_recall, color='green', linestyle='--', alpha=0.5, label=f'Min recall ({min_recall})')
    ax3.set_xlabel('ESI Threshold', fontsize=11)
    ax3.set_ylabel('Recall', fontsize=11)
    ax3.set_title('Recall vs ESI Threshold', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. F1 Score vs ESI Threshold
    ax4 = axes[1, 1]
    ax4.plot(esi_thresholds_plot, f1_scores, linewidth=2, label='F1 Score', color='purple')
    ax4.axvline(x=optimal_esi_threshold_f1, color='red', linestyle='--', alpha=0.7, label='F1-optimal')
    ax4.set_xlabel('ESI Threshold', fontsize=11)
    ax4.set_ylabel('F1 Score', fontsize=11)
    ax4.set_title('F1 Score vs ESI Threshold', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(data_folder, 'f1_score_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nF1 score analysis plot saved to {output_path}")
    
    # Show the plot
    plt.show()
    
    # Confusion matrix at optimal threshold
    predictions_optimal = esi_values_all < optimal_esi_threshold_f1
    cm = confusion_matrix(error_labels, predictions_optimal)
    
    print("\n" + "="*60)
    print("Confusion Matrix at Optimal ESI Threshold:")
    print("="*60)
    print(f"ESI threshold: {optimal_esi_threshold_f1:.4f}")
    print("\nConfusion Matrix:")
    print("                Predicted")
    print("              Low Error  High Error")
    print(f"Actual Low    {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"      High    {cm[1,0]:6d}    {cm[1,1]:6d}")
    print("\nClassification Report:")
    print(classification_report(error_labels, predictions_optimal, 
                              target_names=['Low Error', 'High Error']))
    print("="*60)

if __name__ == '__main__':
    main()

