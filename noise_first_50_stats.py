#!/usr/bin/env python3
"""
Read the first 50 seconds of the noise file and of all 30 default_amcl files.
Noise file: Euclidean position noise from diff_x/diff_y, |diff_yaw|.
default_amcl: position and yaw error (gt vs amcl), split into four ESI bins.
Print mean and 1/2/3 standard deviations for each (noise overall; default_amcl per bin).
"""

import os
import sys

import numpy as np
import pandas as pd

from data_loader import calculate_position_error

USER_HOME = os.path.expanduser("~")
DATA_FOLDER = os.path.join(
    USER_HOME,
    "pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl",
)
N_DEFAULT_AMCL_FILES = 30
NOISE_PATH = "/home/mircrda/devcontainer/ros2_ws/noise_data/noise_latest.csv"
FIRST_N_SECONDS = 50.0

NOISE_POS_COLS = ["diff_x", "diff_y"]
NOISE_YAW_COL = "diff_yaw"

# ESI bins (same as mean_esi_plot / compare_noise_in_ranges)
ESI_BINS = [
    (0.95, 1.0),   # group 0
    (0.66, 0.94),  # group 1
    (0.33, 0.65),  # group 2
    (0.0, 0.32),   # group 3
]
BIN_LABELS = ["1.0–0.95", "0.94–0.66", "0.65–0.33", "0.32–0.0"]


def assign_esi_bin(esi):
    """Return group index 0..3 for given ESI value."""
    for i, (lo, hi) in enumerate(ESI_BINS):
        if lo <= esi <= hi:
            return i
    return None


def calculate_yaw_error(gt_yaw, amcl_yaw):
    """Absolute yaw error in degrees (handles wrapping)."""
    diff = gt_yaw - amcl_yaw
    diff = ((diff + 180) % 360) - 180
    return abs(diff)


def detect_timestamp_column(df):
    """Return name of timestamp-like column in df."""
    for name in ["timestamp", "time", "t", "stamp"]:
        if name in df.columns:
            return name
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return None


def main():
    if not os.path.exists(NOISE_PATH):
        print(f"Error: Noise file not found at {NOISE_PATH}")
        sys.exit(1)

    df = pd.read_csv(NOISE_PATH)
    df.columns = df.columns.str.strip()

    ts_col = detect_timestamp_column(df)
    if ts_col is None:
        print("Error: No timestamp column found in noise file.")
        sys.exit(1)
    df[ts_col] = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)

    t0 = df[ts_col].iloc[0]
    df["time_norm"] = df[ts_col] - t0
    df = df[df["time_norm"] <= FIRST_N_SECONDS].copy()

    yaw_col = NOISE_YAW_COL if NOISE_YAW_COL in df.columns else "diff_xaw"
    missing = [c for c in NOISE_POS_COLS + [yaw_col] if c not in df.columns]
    if missing:
        print(f"Error: Noise file missing columns {missing}. Need diff_x, diff_y, and diff_yaw (or diff_xaw).")
        sys.exit(1)

    # Euclidean position noise magnitude: sqrt(diff_x^2 + diff_y^2)
    pos_noise = np.sqrt(df[NOISE_POS_COLS[0]].astype(float)**2 + df[NOISE_POS_COLS[1]].astype(float)**2).values
    # Yaw noise: absolute value
    yaw_noise = np.abs(df[yaw_col].astype(float).values)

    n = len(pos_noise)
    print(f"First 50 s of noise file: {NOISE_PATH}")
    print(f"Rows: {n}\n")

    # Position noise stats
    mean_pos = np.mean(pos_noise)
    std_pos = np.std(pos_noise)
    print("Position noise (Euclidean magnitude from diff_x, diff_y) [m]:")
    print(f"  Mean  = {mean_pos:.6f}")
    print(f"  Std   = {std_pos:.6f}")
    print(f"  ±1 std: [{mean_pos - std_pos:.6f}, {mean_pos + std_pos:.6f}]")
    print(f"  ±2 std: [{mean_pos - 2*std_pos:.6f}, {mean_pos + 2*std_pos:.6f}]")
    print(f"  ±3 std: [{mean_pos - 3*std_pos:.6f}, {mean_pos + 3*std_pos:.6f}]")
    print()

    # Yaw noise stats
    mean_yaw = np.mean(yaw_noise)
    std_yaw = np.std(yaw_noise)
    print("Yaw noise (|diff_yaw|):")
    print(f"  Mean  = {mean_yaw:.6f}")
    print(f"  Std   = {std_yaw:.6f}")
    print(f"  ±1 std: [{mean_yaw - std_yaw:.6f}, {mean_yaw + std_yaw:.6f}]")
    print(f"  ±2 std: [{mean_yaw - 2*std_yaw:.6f}, {mean_yaw + 2*std_yaw:.6f}]")
    print(f"  ±3 std: [{mean_yaw - 3*std_yaw:.6f}, {mean_yaw + 3*std_yaw:.6f}]")

    # ---- default_amcl (all 30 files, first 50 s): position and yaw error by ESI bin ----
    print("\n" + "=" * 60 + "\n")
    required_cols = [
        "timestamp", "esi", "ground_truth_x", "ground_truth_y",
        "amcl_x", "amcl_y", "ground_truth_yaw", "amcl_yaw",
    ]
    groups_pos = [[] for _ in range(4)]
    groups_yaw = [[] for _ in range(4)]
    files_loaded = 0
    for i in range(1, N_DEFAULT_AMCL_FILES + 1):
        csv_path = os.path.join(DATA_FOLDER, f"{i}.csv")
        if not os.path.exists(csv_path):
            continue
        df1 = pd.read_csv(csv_path, skiprows=range(1, 100))
        df1.columns = df1.columns.str.strip()
        missing = [c for c in required_cols if c not in df1.columns]
        if missing:
            continue
        df1["timestamp"] = pd.to_numeric(df1["timestamp"], errors="coerce")
        df1["esi"] = pd.to_numeric(df1["esi"], errors="coerce")
        df1 = df1.dropna(subset=["timestamp", "esi"]).sort_values("timestamp").reset_index(drop=True)
        if len(df1) == 0:
            continue
        t0_1 = df1["timestamp"].iloc[0]
        df1["time_norm"] = df1["timestamp"] - t0_1
        df1 = df1[df1["time_norm"] <= FIRST_N_SECONDS].copy()
        for _, row in df1.iterrows():
            b = assign_esi_bin(row["esi"])
            if b is None:
                continue
            groups_pos[b].append(calculate_position_error(
                row["ground_truth_x"], row["ground_truth_y"],
                row["amcl_x"], row["amcl_y"],
            ))
            groups_yaw[b].append(calculate_yaw_error(row["ground_truth_yaw"], row["amcl_yaw"]))
        files_loaded += 1
    for b in range(4):
        groups_pos[b] = np.array(groups_pos[b])
        groups_yaw[b] = np.array(groups_yaw[b])
    if files_loaded == 0:
        print(f"Error: No default_amcl files found in {DATA_FOLDER} (expected 1..{N_DEFAULT_AMCL_FILES}.csv)")
        return
    print(f"First 50 s of {files_loaded} files (default_amcl): {DATA_FOLDER}")
    print(f"Total rows: {sum(len(g) for g in groups_pos)}\n")
    for b in range(4):
        pos_errors = groups_pos[b]
        yaw_errors = groups_yaw[b]
        n1 = len(pos_errors)
        print(f"--- ESI bin {b} ({BIN_LABELS[b]})  N={n1} ---")
        if n1 == 0:
            print("  (no data)\n")
            continue
        mean_pos = np.mean(pos_errors)
        std_pos = np.std(pos_errors)
        print("  Position error (Euclidean distance gt vs amcl) [m]:")
        print(f"    Mean  = {mean_pos:.6f}")
        print(f"    Std   = {std_pos:.6f}")
        print(f"    ±1 std: [{mean_pos - std_pos:.6f}, {mean_pos + std_pos:.6f}]")
        print(f"    ±2 std: [{mean_pos - 2*std_pos:.6f}, {mean_pos + 2*std_pos:.6f}]")
        print(f"    ±3 std: [{mean_pos - 3*std_pos:.6f}, {mean_pos + 3*std_pos:.6f}]")
        mean_yaw = np.mean(yaw_errors)
        std_yaw = np.std(yaw_errors)
        print("  Yaw error (absolute, degrees):")
        print(f"    Mean  = {mean_yaw:.6f}")
        print(f"    Std   = {std_yaw:.6f}")
        print(f"    ±1 std: [{mean_yaw - std_yaw:.6f}, {mean_yaw + std_yaw:.6f}]")
        print(f"    ±2 std: [{mean_yaw - 2*std_yaw:.6f}, {mean_yaw + 2*std_yaw:.6f}]")
        print(f"    ±3 std: [{mean_yaw - 3*std_yaw:.6f}, {mean_yaw + 3*std_yaw:.6f}]")
        print()


if __name__ == "__main__":
    main()
