#!/usr/bin/env python3
"""
Read first 50 seconds of the first file (1.csv) and noise data.
Bin rows by ESI into four groups; for each row, match noise by timestamp.
Compute mean absolute pose and yaw error per group; print and plot.
First window: 4 plots for position error (mean, ±1/2/3σ, mean odometry noise).
Second window: 4 plots for yaw error (same structure).
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import calculate_position_error

# Paths
USER_HOME = os.path.expanduser("~")
DATA_FOLDER = os.path.join(
    USER_HOME,
    "pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl",
)
FIRST_FILE = os.path.join(DATA_FOLDER, "1.csv")
NOISE_PATH = "/home/mircrda/devcontainer/ros2_ws/noise_data/noise_latest.csv"

ESI_BINS = [
    (0.95, 1.0),   # group 0
    (0.66, 0.94),  # group 1
    (0.33, 0.65),  # group 2
    (0.0, 0.32),   # group 3
]
BIN_LABELS = ["1.0–0.95", "0.94–0.66", "0.65–0.33", "0.32–0.0"]
FIRST_N_SECONDS = 50.0


def calculate_yaw_error(gt_yaw, amcl_yaw):
    """Absolute yaw error in degrees (handles wrapping)."""
    diff = gt_yaw - amcl_yaw
    diff = ((diff + 180) % 360) - 180
    return abs(diff)


def assign_esi_bin(esi):
    """Return group index 0..3 for given ESI value."""
    for i, (lo, hi) in enumerate(ESI_BINS):
        if lo <= esi <= hi:
            return i
    return None


def detect_timestamp_column(df):
    """Return name of timestamp-like column in df."""
    for name in ["timestamp", "time", "t", "stamp"]:
        if name in df.columns:
            return name
    # Fallback: first numeric column
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


# Noise file columns: difference from perfect odom to noisy odom (measure of noise added)
NOISE_POS_COLS = ["diff_x", "diff_y"]
NOISE_YAW_COL = "diff_yaw"  # or "diff_xaw" if that's the column name


def main():
    if not os.path.exists(FIRST_FILE):
        print(f"Error: File not found at {FIRST_FILE}")
        sys.exit(1)
    if not os.path.exists(NOISE_PATH):
        print(f"Error: Noise file not found at {NOISE_PATH}")
        sys.exit(1)

    required_cols = [
        "esi", "timestamp", "ground_truth_x", "ground_truth_y",
        "amcl_x", "amcl_y", "ground_truth_yaw", "amcl_yaw",
    ]

    # Load first 50 seconds of first file
    df = pd.read_csv(FIRST_FILE, skiprows=range(1, 100))
    df.columns = df.columns.str.strip()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Error: Missing columns in {FIRST_FILE}: {missing}")
        sys.exit(1)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    t0 = df["timestamp"].iloc[0]
    df["time_norm"] = df["timestamp"] - t0
    df = df[df["time_norm"] <= FIRST_N_SECONDS].copy()
    df = df.dropna(subset=["timestamp", "esi"])
    if len(df) == 0:
        print("No rows in first 50 seconds.")
        sys.exit(1)

    # Position and yaw errors
    pos_errors = []
    yaw_errors = []
    for _, row in df.iterrows():
        pos_errors.append(calculate_position_error(
            row["ground_truth_x"], row["ground_truth_y"],
            row["amcl_x"], row["amcl_y"],
        ))
        yaw_errors.append(calculate_yaw_error(row["ground_truth_yaw"], row["amcl_yaw"]))
    df = df.copy()
    df["position_error"] = pos_errors
    df["yaw_error"] = yaw_errors

    # Load noise and match by timestamp
    noise_df = pd.read_csv(NOISE_PATH)
    noise_df.columns = noise_df.columns.str.strip()
    ts_col_noise = detect_timestamp_column(noise_df)
    if ts_col_noise is None:
        print("Error: No timestamp column found in noise CSV.")
        sys.exit(1)
    noise_df[ts_col_noise] = pd.to_numeric(noise_df[ts_col_noise], errors="coerce")
    noise_df = noise_df.dropna(subset=[ts_col_noise]).sort_values(ts_col_noise).reset_index(drop=True)
    if len(noise_df) == 0:
        print("No valid rows in noise file.")
        sys.exit(1)
    # Require diff_x, diff_y and diff_yaw (or diff_xaw) in noise file
    yaw_col = NOISE_YAW_COL if NOISE_YAW_COL in noise_df.columns else "diff_xaw"
    missing_noise = [c for c in NOISE_POS_COLS + [yaw_col] if c not in noise_df.columns]
    if missing_noise:
        print(f"Error: Noise file missing columns {missing_noise}. Need diff_x, diff_y, and diff_yaw (or diff_xaw).")
        sys.exit(1)

    # Merge ESI data with nearest noise row by timestamp
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    merged = pd.merge_asof(
        df_sorted,
        noise_df.add_prefix("noise_"),
        left_on="timestamp",
        right_on=f"noise_{ts_col_noise}",
        direction="nearest",
    )

    # Four groups: position error, yaw error, and noise from diff_x/diff_y/diff_yaw
    groups_pos = [[] for _ in range(4)]
    groups_yaw = [[] for _ in range(4)]
    groups_pos_noise = [[] for _ in range(4)]   # sqrt(diff_x^2 + diff_y^2) [m]
    groups_yaw_noise = [[] for _ in range(4)]   # abs(diff_yaw) [deg or rad]

    for _, row in merged.iterrows():
        b = assign_esi_bin(row["esi"])
        if b is None:
            continue
        groups_pos[b].append(row["position_error"])
        groups_yaw[b].append(row["yaw_error"])
        dx = row.get(f"noise_{NOISE_POS_COLS[0]}", np.nan)
        dy = row.get(f"noise_{NOISE_POS_COLS[1]}", np.nan)
        dyaw = row.get(f"noise_{yaw_col}", np.nan)
        if pd.notna(dx) and pd.notna(dy):
            groups_pos_noise[b].append(np.sqrt(float(dx)**2 + float(dy)**2))
        if pd.notna(dyaw):
            groups_yaw_noise[b].append(abs(float(dyaw)))

    for b in range(4):
        groups_pos[b] = np.array(groups_pos[b])
        groups_yaw[b] = np.array(groups_yaw[b])
        groups_pos_noise[b] = np.array(groups_pos_noise[b]) if groups_pos_noise[b] else np.array([])
        groups_yaw_noise[b] = np.array(groups_yaw_noise[b]) if groups_yaw_noise[b] else np.array([])

    # Mean absolute pose and yaw error, and mean noise (from diff_x/diff_y/diff_yaw) per group
    print("Mean position error [m], yaw error [deg], and mean odom noise (diff_x/diff_y/diff_yaw) by ESI group:")
    for b in range(4):
        n = len(groups_pos[b])
        mean_p = np.mean(groups_pos[b]) if n else np.nan
        mean_y = np.mean(groups_yaw[b]) if n else np.nan
        mean_p_noise = np.mean(groups_pos_noise[b]) if len(groups_pos_noise[b]) else np.nan
        mean_y_noise = np.mean(groups_yaw_noise[b]) if len(groups_yaw_noise[b]) else np.nan
        print(f"  {BIN_LABELS[b]}: N={n}, mean_pos={mean_p:.6f} m, mean_yaw={mean_y:.4f} deg, mean_pos_noise={mean_p_noise:.6f} m, mean_yaw_noise={mean_y_noise:.6f}")

    tick_width = 0.15

    def draw_one_group(ax, values, ylabel, title, mean_odom_val):
        """Draw stick (mean ± 3σ), mean, ±1/2/3σ ticks, and mean odometry noise line."""
        n = len(values)
        if n == 0:
            ax.set_title(title + " — no data")
            return
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            std = np.finfo(float).eps
        x = 0
        lo = max(mean - 3 * std, values.min())
        hi = min(mean + 3 * std, values.max())
        ax.plot([x, x], [lo, hi], color="black", linewidth=1.5, zorder=2)
        ax.plot([x - tick_width, x + tick_width], [mean, mean], color="red", linewidth=2.5, zorder=3)
        ax.scatter([x], [mean], color="red", s=40, zorder=4, edgecolors="darkred")
        for k, sigma in enumerate([1, 2, 3]):
            w = tick_width * (1 - k * 0.25)
            for sign in (-1, 1):
                val = mean + sign * sigma * std
                if lo <= val <= hi:
                    ax.plot([x - w, x + w], [val, val], color="gray", linewidth=1, zorder=2)
        if not np.isnan(mean_odom_val):
            ax.axhline(mean_odom_val, color="green", linestyle=":", linewidth=1.5, label=f"Mean odom noise = {mean_odom_val:.4f}")
            ax.legend(loc="best", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([0])
        ax.set_xticklabels([f"N={n}"])
        ax.grid(True, alpha=0.3, axis="y")

    # First window: 4 plots for position error (mean odom noise = mean of sqrt(diff_x^2+diff_y^2))
    fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8))
    axes1 = axes1.flatten()
    for b in range(4):
        mean_pos_noise = np.mean(groups_pos_noise[b]) if len(groups_pos_noise[b]) else np.nan
        draw_one_group(
            axes1[b],
            groups_pos[b],
            "Position error [m]",
            f"Group {b} ({BIN_LABELS[b]}) — Position",
            mean_pos_noise,
        )
    plt.suptitle("First 50 s of file 1 — Position error by ESI group (mean, ±1/2/3σ, mean odom noise)")
    plt.tight_layout()
    plt.savefig("compare_noise_in_ranges_position.png", dpi=150, bbox_inches="tight")
    print("Saved compare_noise_in_ranges_position.png")
    plt.show()

    # Second window: 4 plots for yaw error (mean odom noise = mean of |diff_yaw|)
    fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))
    axes2 = axes2.flatten()
    for b in range(4):
        mean_yaw_noise = np.mean(groups_yaw_noise[b]) if len(groups_yaw_noise[b]) else np.nan
        draw_one_group(
            axes2[b],
            groups_yaw[b],
            "Yaw error [deg]",
            f"Group {b} ({BIN_LABELS[b]}) — Yaw",
            mean_yaw_noise,
        )
    plt.suptitle("First 50 s of file 1 — Yaw error by ESI group (mean, ±1/2/3σ, mean odom noise)")
    plt.tight_layout()
    plt.savefig("compare_noise_in_ranges_yaw.png", dpi=150, bbox_inches="tight")
    print("Saved compare_noise_in_ranges_yaw.png")
    plt.show()


if __name__ == "__main__":
    main()
