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
MIRA_BINS = [(1.0 - (i + 1) * 0.1, 1.0 - i * 0.1) for i in range(10)]
# Same bin edges for ESI
ESI_BINS = [(1.0 - (i + 1) * 0.1, 1.0 - i * 0.1) for i in range(10)]
BIN_LABELS = [f"{1.0 - i*0.1:.1f}–{1.0 - (i+1)*0.1:.1f}" for i in range(10)]


def assign_mira_bin(mira):
    """Return group index 0..9 for given map_integrity_ratio value."""
    for i, (lo, hi) in enumerate(MIRA_BINS):
        if lo <= mira <= hi:
            return i
    return None


def assign_esi_bin(esi):
    """Return group index 0..9 for given ESI value."""
    for i, (lo, hi) in enumerate(ESI_BINS):
        if lo <= esi <= hi:
            return i
    return None


def calculate_yaw_error(gt_yaw, amcl_yaw):
    """Absolute yaw error in degrees (handles wrapping)."""
    diff = gt_yaw - amcl_yaw
    diff = ((diff + 180) % 360) - 180
    return abs(diff)


def main():
    required_cols = [
        "timestamp", "map_integrity_ratio", "esi", "ground_truth_x", "ground_truth_y",
        "amcl_x", "amcl_y", "ground_truth_yaw", "amcl_yaw",
    ]
    n_bins = len(MIRA_BINS)
    groups_pos = [[] for _ in range(n_bins)]
    groups_yaw = [[] for _ in range(n_bins)]
    groups_pos_esi = [[] for _ in range(n_bins)]
    groups_yaw_esi = [[] for _ in range(n_bins)]
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
        df["esi"] = pd.to_numeric(df["esi"], errors="coerce")
        df = df.dropna(subset=["timestamp", "map_integrity_ratio", "esi"]).sort_values("timestamp").reset_index(drop=True)
        if len(df) == 0:
            continue
        t0 = df["timestamp"].iloc[0]
        df["time_norm"] = df["timestamp"] - t0
        df = df[df["time_norm"] <= FIRST_N_SECONDS].copy()
        for _, row in df.iterrows():
            pos_err = calculate_position_error(
                row["ground_truth_x"], row["ground_truth_y"],
                row["amcl_x"], row["amcl_y"],
            )
            yaw_err = calculate_yaw_error(row["ground_truth_yaw"], row["amcl_yaw"])
            b_mira = assign_mira_bin(row["map_integrity_ratio"])
            if b_mira is not None:
                groups_pos[b_mira].append(pos_err)
                groups_yaw[b_mira].append(yaw_err)
            b_esi = assign_esi_bin(row["esi"])
            if b_esi is not None:
                groups_pos_esi[b_esi].append(pos_err)
                groups_yaw_esi[b_esi].append(yaw_err)
        files_loaded += 1
    for b in range(n_bins):
        groups_pos[b] = np.array(groups_pos[b])
        groups_yaw[b] = np.array(groups_yaw[b])
        groups_pos_esi[b] = np.array(groups_pos_esi[b])
        groups_yaw_esi[b] = np.array(groups_yaw_esi[b])
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
    # Take min_count random samples from each bin (MIRa)
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
    # Balance ESI bins the same way
    bin_counts_esi = [len(groups_pos_esi[b]) for b in range(n_bins)]
    min_count_esi = min(bin_counts_esi) if bin_counts_esi else 0
    if min_count_esi == 0:
        print("Warning: Smallest ESI bin has 0 entries; ESI bars will show NaN where empty.")
    balanced_pos_esi = []
    balanced_yaw_esi = []
    for b in range(n_bins):
        pos_errors = groups_pos_esi[b]
        yaw_errors = groups_yaw_esi[b]
        n = len(pos_errors)
        if n == 0:
            balanced_pos_esi.append(np.array([]))
            balanced_yaw_esi.append(np.array([]))
            continue
        k = min(min_count_esi, n) if min_count_esi > 0 else 0
        if k == 0:
            balanced_pos_esi.append(np.array([]))
            balanced_yaw_esi.append(np.array([]))
            continue
        idx = rng.choice(n, size=k, replace=False)
        balanced_pos_esi.append(pos_errors[idx])
        balanced_yaw_esi.append(yaw_errors[idx])
    print("Mean, min, max for position error [m] and yaw error [deg] (balanced):\n")
    for b in range(n_bins):
        pos_errors = balanced_pos[b]
        yaw_errors = balanced_yaw[b]
        n = len(pos_errors)
        print(f"MIRa bin {b} ({BIN_LABELS[b]})  N={n} (balanced)")
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
    means_pos_esi = np.array([np.mean(balanced_pos_esi[b]) if len(balanced_pos_esi[b]) else np.nan for b in range(n_bins)])
    means_yaw_esi = np.array([np.mean(balanced_yaw_esi[b]) if len(balanced_yaw_esi[b]) else np.nan for b in range(n_bins)])
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

    # Plot: mean position and yaw by bin as lines (twin axes) with sliders
    fig, ax_means = plt.subplots(1, 1, figsize=(10, 6))
    ax_means2 = ax_means.twinx()

    x_bins = np.arange(n_bins)
    linewidth_mira_init = 2.0
    linewidth_esi_init = 2.0
    # MIRa: light and dark blue (behind); ESI: light and dark red (in front)
    color_mira_pos = "#93c5fd"   # light blue
    color_mira_yaw = "#1e40af"   # dark blue
    color_esi_pos = "#fca5a5"   # light red
    color_esi_yaw = "#991b1b"   # dark red

    (line_mira_pos,) = ax_means.plot(x_bins, means_pos, "o-", color=color_mira_pos, linewidth=linewidth_mira_init, label="MIRa position [m]", zorder=0)
    (line_esi_pos,) = ax_means.plot(x_bins, np.nan_to_num(means_pos_esi, nan=np.nan), "o-", color=color_esi_pos, linewidth=linewidth_esi_init, label="ESI position [m]", zorder=2)
    (line_mira_yaw,) = ax_means2.plot(x_bins, means_yaw, "s:", color=color_mira_yaw, linewidth=linewidth_mira_init, label="MIRa yaw [deg]", zorder=0)
    (line_esi_yaw,) = ax_means2.plot(x_bins, np.nan_to_num(means_yaw_esi, nan=np.nan), "s:", color=color_esi_yaw, linewidth=linewidth_esi_init, label="ESI yaw [deg]", zorder=2)

    ax_means.set_ylabel("Position error [m]", color="black")
    ax_means2.set_ylabel("Yaw error [deg]", color="black")
    ax_means.tick_params(axis="y", labelcolor="black")
    ax_means2.tick_params(axis="y", labelcolor="black")
    ax_means.set_xlabel("Bin (same edges for MIRa and ESI)")
    ax_means.set_xticks(x_bins)
    ax_means.set_xticklabels(BIN_LABELS, rotation=45, ha="right")
    ax_means.grid(True, alpha=0.3, axis="y")
    ax_means.set_title("Mean position and yaw error by bin")
    lines1, labels1 = ax_means.get_legend_handles_labels()
    lines2, labels2 = ax_means2.get_legend_handles_labels()
    legend_fontsize_init = 12
    legend = ax_means.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=legend_fontsize_init)
    plt.suptitle(f"MIRa balanced N={min_count}, ESI balanced N={min_count_esi} per bin")

    # Slider axes (below main plot)
    plt.subplots_adjust(bottom=0.32)
    ax_legend_size = plt.axes([0.2, 0.22, 0.6, 0.03])
    ax_linewidth_mira = plt.axes([0.2, 0.17, 0.6, 0.03])
    ax_linewidth_esi = plt.axes([0.2, 0.12, 0.6, 0.03])
    ax_axis_fontsize = plt.axes([0.2, 0.07, 0.6, 0.03])
    slider_legend_size = Slider(ax_legend_size, "Legend font size", 6, 24, valinit=legend_fontsize_init, valstep=1)
    slider_linewidth_mira = Slider(ax_linewidth_mira, "MIRa line thickness", 0.5, 6.0, valinit=linewidth_mira_init, valstep=0.25)
    slider_linewidth_esi = Slider(ax_linewidth_esi, "ESI line thickness", 0.5, 6.0, valinit=linewidth_esi_init, valstep=0.25)
    axis_fontsize_init = 10
    slider_axis_fontsize = Slider(ax_axis_fontsize, "Axis numbers & labels size", 6, 24, valinit=axis_fontsize_init, valstep=1)

    def update(_):
        lw_mira = slider_linewidth_mira.val
        lw_esi = slider_linewidth_esi.val
        axis_fontsize = slider_axis_fontsize.val
        line_mira_pos.set_linewidth(lw_mira)
        line_mira_yaw.set_linewidth(lw_mira)
        line_esi_pos.set_linewidth(lw_esi)
        line_esi_yaw.set_linewidth(lw_esi)
        for t in legend.get_texts():
            t.set_fontsize(slider_legend_size.val)
        ax_means.tick_params(axis="x", labelsize=axis_fontsize, labelcolor="black")
        ax_means.tick_params(axis="y", labelsize=axis_fontsize, labelcolor="black")
        ax_means2.tick_params(axis="y", labelsize=axis_fontsize, labelcolor="black")
        ax_means.xaxis.get_label().set_fontsize(axis_fontsize)
        ax_means.xaxis.get_label().set_color("black")
        ax_means.yaxis.get_label().set_fontsize(axis_fontsize)
        ax_means.yaxis.get_label().set_color("black")
        ax_means2.yaxis.get_label().set_fontsize(axis_fontsize)
        ax_means2.yaxis.get_label().set_color("black")
        fig.canvas.draw_idle()

    slider_legend_size.on_changed(update)
    slider_linewidth_mira.on_changed(update)
    slider_linewidth_esi.on_changed(update)
    slider_axis_fontsize.on_changed(update)
    update(None)

    plt.savefig("max_errors_by_mira_bin.png", dpi=150, bbox_inches="tight")
    print("Saved max_errors_by_mira_bin.png")
    plt.show()


if __name__ == "__main__":
    main()
