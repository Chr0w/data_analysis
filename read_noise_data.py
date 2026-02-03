#!/usr/bin/env python3
"""
Read noise data from CSV and plot the values.
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "/home/mircrda/devcontainer/ros2_ws/noise_data/noise_latest.csv"


def main():
    if not os.path.exists(CSV_PATH):
        print(f"Error: File not found at {CSV_PATH}")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from {CSV_PATH}")
    print(f"Columns: {list(df.columns)}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns to plot.")
        sys.exit(1)

    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, min(3, n_cols), figsize=(12, 4 * n_rows), squeeze=False)
    axes = axes.flatten()

    x = np.arange(len(df))
    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.plot(x, df[col].values, linewidth=0.8)
        ax.set_title(col)
        ax.set_xlabel("Index")
        ax.grid(True, alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("noise_data_plot.png", dpi=150, bbox_inches="tight")
    print("Saved noise_data_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
