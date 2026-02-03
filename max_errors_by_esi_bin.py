#!/usr/bin/env python3
"""
First 50 seconds of all 30 default_amcl files, split into 10 map_integrity_ratio bins (0.1 increment).
Balance bins by taking the same number of random samples as the smallest bin.
Print mean, min and max for position and yaw error on the balanced data.
Mean of means for first 7 bins as reference; print and plot differences for all 10 bins.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

from data_loader import calculate_position_error

USER_HOME = os.path.expanduser("~")
DATA_FOLDER = os.path.join(
    USER_HOME,
    "pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl",
)
N_FILES = 30
FIRST_N_SECONDS = 80.0

# 10 bins, 0.1 increment: [1.0–0.9], [0.9–0.8], ..., [0.1–0.0]
MIR_BINS = [(1.0 - (i + 1) * 0.1, 1.0 - i * 0.1) for i in range(10)]
BIN_LABELS = [f"{1.0 - i*0.1:.1f}–{1.0 - (i+1)*0.1:.1f}" for i in range(10)]


def assign_mir_bin(mir):
    """Return group index 0..9 for given map_integrity_ratio value."""
    for i, (lo, hi) in enumerate(MIR_BINS):
        if lo <= mir <= hi:
            return i
    return None


def calculate_yaw_error(gt_yaw, amcl_yaw):
    """Absolute yaw error in degrees (handles wrapping)."""
    diff = gt_yaw - amcl_yaw
    diff = ((diff + 180) % 360) - 180
    return abs(diff)


def main():
    required_cols = [
        "timestamp", "map_integrity_ratio", "ground_truth_x", "ground_truth_y",
        "amcl_x", "amcl_y", "ground_truth_yaw", "amcl_yaw",
    ]
    n_bins = len(MIR_BINS)
    groups_pos = [[] for _ in range(n_bins)]
    groups_yaw = [[] for _ in range(n_bins)]
    files_loaded = 0
    for i in range(1, N_FILES + 1):
        csv_path = os.path.join(DATA_FOLDER, f"{i}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, skiprows=range(1, 100))
        df.columns = df.columns.str.strip()
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            continue
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
        df["map_integrity_ratio"] = pd.to_numeric(df["map_integrity_ratio"], errors="coerce")
        df = df.dropna(subset=["timestamp", "map_integrity_ratio"]).sort_values("timestamp").reset_index(drop=True)
        if len(df) == 0:
            continue
        t0 = df["timestamp"].iloc[0]
        df["time_norm"] = df["timestamp"] - t0
        df = df[df["time_norm"] <= FIRST_N_SECONDS].copy()
        for _, row in df.iterrows():
            b = assign_mir_bin(row["map_integrity_ratio"])
            if b is None:
                continue
            groups_pos[b].append(calculate_position_error(
                row["ground_truth_x"], row["ground_truth_y"],
                row["amcl_x"], row["amcl_y"],
            ))
            groups_yaw[b].append(calculate_yaw_error(row["ground_truth_yaw"], row["amcl_yaw"]))
        files_loaded += 1
    for b in range(n_bins):
        groups_pos[b] = np.array(groups_pos[b])
        groups_yaw[b] = np.array(groups_yaw[b])
    if files_loaded == 0:
        print(f"Error: No default_amcl files found in {DATA_FOLDER} (expected 1..{N_FILES}.csv)")
        sys.exit(1)
    print(f"First {int(FIRST_N_SECONDS)} s of {files_loaded} files (default_amcl): {DATA_FOLDER}")
    print(f"Total rows: {sum(len(g) for g in groups_pos)}\n")
    # Bin counts (original)
    bin_counts = [len(groups_pos[b]) for b in range(n_bins)]
    min_count = min(bin_counts)
    print(f"Bin counts (original): {bin_counts}")
    if min_count == 0:
        print("Smallest bin has 0 entries; cannot balance. Exiting.")
        sys.exit(1)
    print(f"Smallest bin has {min_count} entries → balancing to {min_count} random samples per bin.\n")
    rng = np.random.default_rng(42)
    # Take min_count random samples from each bin
    balanced_pos = []
    balanced_yaw = []
    for b in range(n_bins):
        pos_errors = groups_pos[b]
        yaw_errors = groups_yaw[b]
        n = len(pos_errors)
        if n == 0:
            balanced_pos.append(np.array([]))
            balanced_yaw.append(np.array([]))
            continue
        k = min(min_count, n)
        idx = rng.choice(n, size=k, replace=False)
        balanced_pos.append(pos_errors[idx])
        balanced_yaw.append(yaw_errors[idx])
    print("Mean, min, max for position error [m] and yaw error [deg] (balanced):\n")
    for b in range(n_bins):
        pos_errors = balanced_pos[b]
        yaw_errors = balanced_yaw[b]
        n = len(pos_errors)
        print(f"MIR bin {b} ({BIN_LABELS[b]})  N={n} (balanced)")
        if n == 0:
            print("  (no data)\n")
            continue
        mean_pos, min_pos, max_pos = np.mean(pos_errors), np.min(pos_errors), np.max(pos_errors)
        mean_yaw, min_yaw, max_yaw = np.mean(yaw_errors), np.min(yaw_errors), np.max(yaw_errors)
        print(f"  Position error [m]:  mean = {mean_pos:.6f},  min = {min_pos:.6f},  max = {max_pos:.6f}")
        print(f"  Yaw error [deg]:     mean = {mean_yaw:.6f},  min = {min_yaw:.6f},  max = {max_yaw:.6f}")
        print()

    # Mean of means for first 7 bins (reference)
    means_pos = np.array([np.mean(balanced_pos[b]) if len(balanced_pos[b]) else np.nan for b in range(n_bins)])
    means_yaw = np.array([np.mean(balanced_yaw[b]) if len(balanced_yaw[b]) else np.nan for b in range(n_bins)])
    n_ref = 7
    mean_of_means_pos = np.nanmean(means_pos[:n_ref])
    mean_of_means_yaw = np.nanmean(means_yaw[:n_ref])
    print(f"Mean of means (first {n_ref} bins):")
    print(f"  Position error [m]:  {mean_of_means_pos:.6f}")
    print(f"  Yaw error [deg]:     {mean_of_means_yaw:.6f}\n")
    # Differences from this mean for all 10 bins
    diff_pos = means_pos - mean_of_means_pos
    diff_yaw = means_yaw - mean_of_means_yaw
    print("Difference from mean-of-means (all 10 bins):")
    for b in range(n_bins):
        print(f"  Bin {b} ({BIN_LABELS[b]}):  position diff = {diff_pos[b]:+.6f} m,  yaw diff = {diff_yaw[b]:+.6f} deg")
    print()

    # Plot: mean position and yaw by map_integrity_ratio bin (twin axes) with sliders
    fig, ax_means = plt.subplots(1, 1, figsize=(10, 6))
    ax_means2 = ax_means.twinx()

    # Initial values
    bar_width_init = 0.35
    gap_in_bin_init = 0.4   # center-to-center distance between the two bars in a bin
    bin_spacing_init = 0.0  # extra space between bin groups (0 = bins at 0,1,2,...)

    def get_x_positions(bin_spacing):
        return np.arange(n_bins) * (1.0 + bin_spacing)

    x_init = get_x_positions(bin_spacing_init)
    offset_init = gap_in_bin_init / 2
    bars_pos = ax_means.bar(x_init - offset_init, means_pos, width=bar_width_init, label="Position mean [m]", color="C0", alpha=0.9, align="center")
    bars_yaw = ax_means2.bar(x_init + offset_init, means_yaw, width=bar_width_init, label="Yaw mean [deg]", color="C2", alpha=0.9, align="center")
    ax_means.set_ylabel("Position error [m]", color="C0")
    ax_means2.set_ylabel("Yaw error [deg]", color="C2")
    ax_means.tick_params(axis="y", labelcolor="C0")
    ax_means2.tick_params(axis="y", labelcolor="C2")
    ax_means.set_xlabel("Map integrity ratio bin")
    ax_means.set_xticks(x_init)
    ax_means.set_xticklabels(BIN_LABELS, rotation=45, ha="right")
    ax_means.grid(True, alpha=0.3, axis="y")
    ax_means.set_title("Mean position and yaw error by map_integrity_ratio bin (balanced)")
    lines1, labels1 = ax_means.get_legend_handles_labels()
    lines2, labels2 = ax_means2.get_legend_handles_labels()
    legend_fontsize_init = 12
    legend = ax_means.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=legend_fontsize_init)
    plt.suptitle(f"Balanced N={min_count} per bin")

    # Slider axes (below main plot)
    plt.subplots_adjust(bottom=0.32)
    ax_legend_size = plt.axes([0.2, 0.22, 0.6, 0.03])
    ax_bar_width = plt.axes([0.2, 0.17, 0.6, 0.03])
    ax_gap_in_bin = plt.axes([0.2, 0.12, 0.6, 0.03])
    ax_axis_fontsize = plt.axes([0.2, 0.07, 0.6, 0.03])
    slider_legend_size = Slider(ax_legend_size, "Legend font size", 6, 24, valinit=legend_fontsize_init, valstep=1)
    slider_bar_width = Slider(ax_bar_width, "Bar width", 0.05, 0.5, valinit=bar_width_init, valstep=0.01)
    slider_gap_in_bin = Slider(ax_gap_in_bin, "Gap between bars (same bin)", 0.05, 0.8, valinit=gap_in_bin_init, valstep=0.01)
    axis_fontsize_init = 10
    slider_axis_fontsize = Slider(ax_axis_fontsize, "Axis numbers & labels size", 6, 24, valinit=axis_fontsize_init, valstep=1)

    def update(_):
        bar_width = slider_bar_width.val
        gap_in_bin = slider_gap_in_bin.val
        bin_spacing = bin_spacing_init  # fixed (no longer a slider)
        axis_fontsize = slider_axis_fontsize.val
        x = get_x_positions(bin_spacing)
        offset = gap_in_bin / 2
        # matplotlib bar with align='center' has rectangle left edge at (x - width/2)
        for i, (patch, val) in enumerate(zip(bars_pos.patches, means_pos)):
            cx = x[i] - offset
            patch.set_x(cx - bar_width / 2)
            patch.set_width(bar_width)
            patch.set_height(val)
        for i, (patch, val) in enumerate(zip(bars_yaw.patches, means_yaw)):
            cx = x[i] + offset
            patch.set_x(cx - bar_width / 2)
            patch.set_width(bar_width)
            patch.set_height(val)
        ax_means.set_xticks(x)
        # Update x-axis range to fit all bars
        margin = 0.2
        x_min = -offset - bar_width / 2 - margin
        x_max = (n_bins - 1) * (1.0 + bin_spacing) + offset + bar_width / 2 + margin
        ax_means.set_xlim(x_min, x_max)
        for t in legend.get_texts():
            t.set_fontsize(slider_legend_size.val)
        # Axis numbers and labels size
        ax_means.tick_params(axis="x", labelsize=axis_fontsize)
        ax_means.tick_params(axis="y", labelsize=axis_fontsize)
        ax_means2.tick_params(axis="y", labelsize=axis_fontsize)
        ax_means.xaxis.get_label().set_fontsize(axis_fontsize)
        ax_means.yaxis.get_label().set_fontsize(axis_fontsize)
        ax_means2.yaxis.get_label().set_fontsize(axis_fontsize)
        fig.canvas.draw_idle()

    slider_legend_size.on_changed(update)
    slider_bar_width.on_changed(update)
    slider_gap_in_bin.on_changed(update)
    slider_axis_fontsize.on_changed(update)
    update(None)  # set initial x-axis range

    plt.savefig("max_errors_by_mir_bin.png", dpi=150, bbox_inches="tight")
    print("Saved max_errors_by_mir_bin.png")
    plt.show()


if __name__ == "__main__":
    main()
