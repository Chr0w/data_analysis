#!/usr/bin/env python3
"""
Read first 50 seconds of all 30 files, bin by ESI into four groups.
For each group: mean position error and mean yaw error.
Two subplots: position error and yaw error, each as a bar/stick plot
with mean, 1/2/3 std dev markers, and outliers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import os
from data_loader import calculate_position_error


def calculate_yaw_error(gt_yaw, amcl_yaw):
    """Absolute yaw error in degrees (handles wrapping)."""
    diff = gt_yaw - amcl_yaw
    diff = ((diff + 180) % 360) - 180
    return abs(diff)


# ESI bins: [1.0-0.95], [0.94-0.66], [0.65-0.33], [0.32-0.0]
ESI_BINS = [
    (0.95, 1.0),   # group 0
    (0.66, 0.94),  # group 1
    (0.33, 0.65),  # group 2
    (0.0, 0.32),   # group 3
]
BIN_LABELS = ['1.0–0.95', '0.94–0.66', '0.65–0.33', '0.32–0.0']


def assign_esi_bin(esi):
    """Return group index 0..3 for given ESI value."""
    for i, (lo, hi) in enumerate(ESI_BINS):
        if lo <= esi <= hi:
            return i
    return None


def main():
    N = 30
    first_n_seconds = 50.0
    user_home = os.path.expanduser('~')
    data_folder = os.path.join(user_home, 'pCloudDrive/Offline/PhD/Folders/test_data/article_data/default_amcl')

    required_cols = ['esi', 'timestamp', 'ground_truth_x', 'ground_truth_y', 'amcl_x', 'amcl_y',
                    'ground_truth_yaw', 'amcl_yaw']
    groups_pos = [[] for _ in range(4)]
    groups_yaw = [[] for _ in range(4)]

    print("Loading first 50 seconds from each file...")
    for i in range(1, N + 1):
        csv_path = os.path.join(data_folder, f'{i}.csv')
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, skiprows=range(1, 100))
        df.columns = df.columns.str.strip()
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            continue
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        t0 = df['timestamp'].iloc[0]
        df['time_norm'] = df['timestamp'] - t0
        df = df[df['time_norm'] <= first_n_seconds].copy()
        if len(df) == 0:
            continue
        pos_errors = []
        yaw_errors = []
        for _, row in df.iterrows():
            pos_errors.append(calculate_position_error(
                row['ground_truth_x'], row['ground_truth_y'],
                row['amcl_x'], row['amcl_y']))
            yaw_errors.append(calculate_yaw_error(row['ground_truth_yaw'], row['amcl_yaw']))
        df['position_error'] = pos_errors
        df['yaw_error'] = yaw_errors
        for _, row in df.iterrows():
            b = assign_esi_bin(row['esi'])
            if b is not None:
                groups_pos[b].append(row['position_error'])
                groups_yaw[b].append(row['yaw_error'])

    for b in range(4):
        groups_pos[b] = np.array(groups_pos[b])
        groups_yaw[b] = np.array(groups_yaw[b])

    # Stick plot: for each group draw vertical line (mean ± 3*std), mean, ticks at ±1,2,3 sigma, and outliers
    tick_width = 0.15  # half-width of std dev ticks
    outlier_jitter = 0.08  # x jitter for outlier points

    fig, (ax_pos, ax_yaw) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, groups, ylabel, title in [
        (ax_pos, groups_pos, 'Position error [m]', 'Position error by ESI group'),
        (ax_yaw, groups_yaw, 'Yaw error [deg]', 'Yaw error by ESI group'),
    ]:
        n_groups = len(groups)
        x_centers = np.arange(n_groups)
        total = sum(len(g) for g in groups)

        for i, g in enumerate(groups):
            if len(g) == 0:
                continue
            mean = np.mean(g)
            std = np.std(g)
            if std == 0:
                std = np.finfo(float).eps  # avoid div by zero for constant group
            x = x_centers[i]

            # Vertical stick from mean-3*std to mean+3*std (clip to data range for clarity)
            lo = max(mean - 3 * std, g.min())
            hi = min(mean + 3 * std, g.max())
            ax.plot([x, x], [lo, hi], color='black', linewidth=1.5, zorder=2)

            # Mean: thick dot or short horizontal line
            ax.plot([x - tick_width, x + tick_width], [mean, mean], color='red', linewidth=2.5, zorder=3)
            ax.scatter([x], [mean], color='red', s=40, zorder=4, edgecolors='darkred')

            # Ticks at ±1σ, ±2σ, ±3σ (only if within stick range); inner ticks shorter
            for k, sigma in enumerate([1, 2, 3]):
                w = tick_width * (1 - k * 0.25)  # 1σ longest, 3σ shortest
                for sign in (-1, 1):
                    val = mean + sign * sigma * std
                    if lo <= val <= hi:
                        ax.plot([x - w, x + w], [val, val], color='gray', linewidth=1, zorder=2)

            # Outliers: points beyond 3*std from mean
            mask = np.abs(g - mean) > 3 * std
            if np.any(mask):
                out_vals = g[mask]
                np.random.seed(42)
                jitter = np.random.uniform(-outlier_jitter, outlier_jitter, size=out_vals.size)
                ax.scatter(x + jitter, out_vals, color='orange', s=12, alpha=0.8, zorder=5, label='Outliers' if i == 0 and np.any(mask) else '')

            # Count and percentage of entire dataset (below the figure)
            pct = 100 * len(g) / total if total > 0 else 0
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(x, -0.12, f'N={len(g):,}\n({pct:.1f}%)',
                    ha='center', va='top', fontsize=8, transform=trans, clip_on=False)

        ax.set_xticks(x_centers)
        ax.set_xticklabels(BIN_LABELS)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlabel('ESI group')

    plt.suptitle('First 50 s, 30 files — mean (red), ±1/2/3σ (gray), outliers (orange)', fontsize=10)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
