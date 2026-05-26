#!/usr/bin/env python3
"""
Folie 3: Failure Pattern Effect (v2)
- Focus on line_consecutive (main disruption mode)
- Vary p_fail (failure probability): 0.3, 0.5, 0.7
- Each panel: k=1,2,3 boxplots
- k-labels only in middle panel
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
results_dir = Path(__file__).parent / "Results"
sf_data = pd.read_csv(results_dir / "vss_sf_main_redo" / "vss_map.csv", sep=";")
mumford0_data = pd.read_csv(results_dir / "vss_mumford0_redo" / "vss_map.csv", sep=";")

print(f"SiouxFalls rows: {len(sf_data)}, Mumford0 rows: {len(mumford0_data)}")

# Prepare data: filter for line_consecutive and k=1,2,3
def prepare_data(df):
    """Filter for line_consecutive, valid cases, and k=1,2,3"""
    valid = df[["RP", "VSS_nom"]].notna().all(axis=1)
    df_valid = df[valid].copy()
    df_valid["VSS_over_RP"] = (df_valid["VSS_nom"] / df_valid["RP"]) * 100
    # Filter only line_consecutive and k=1,2,3
    df_valid = df_valid[
        (df_valid["case_selection"] == "line_consecutive") &
        (df_valid["case_k"].isin([1, 2, 3]))
    ]
    return df_valid

sf_data = prepare_data(sf_data)
mumford0_data = prepare_data(mumford0_data)

print(f"Valid cases (line_consecutive, k=1..3): SF={len(sf_data)}, Mumford0={len(mumford0_data)}")

# Get p_fail values
p_fail_values = sorted(sf_data["case_p_fail"].unique())
print(f"p_fail values: {p_fail_values}")

# Create figure: 2 rows (networks) × 3 columns (p_fail values)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

k_values = [1, 2, 3]
colors = {"SiouxFalls": "#1f77b4", "Mumford0": "#ff7f0e"}

for row, (data, network_name) in enumerate([(sf_data, "SiouxFalls"), (mumford0_data, "Mumford0")]):
    for col, p_fail in enumerate(p_fail_values):
        ax = axes[row, col]

        # Filter data for this p_fail
        p_fail_data = data[data["case_p_fail"] == p_fail]

        if len(p_fail_data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"p_fail = {p_fail}")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Prepare data for boxplot: one group per k
        boxplot_data = []
        for k in k_values:
            subset = p_fail_data[p_fail_data["case_k"] == k]["VSS_over_RP"].values
            if len(subset) > 0:
                boxplot_data.append(subset)
            else:
                boxplot_data.append([np.nan])

        # Create boxplot
        bp = ax.boxplot(boxplot_data, patch_artist=True, widths=0.6)

        # Styling
        for patch in bp['boxes']:
            patch.set_facecolor(colors[network_name])
            patch.set_alpha(0.7)

        # Y-axis label only on left panels
        if col == 0:
            ax.set_ylabel("VSS / RP (%)", fontsize=10, weight="bold")

        # Title with case counts
        counts = [len(p_fail_data[p_fail_data["case_k"] == k]) for k in k_values]
        title = f"p_fail = {p_fail}\n(n={counts[0]}, {counts[1]}, {counts[2]})"
        ax.set_title(title, fontsize=10, weight="bold")
        ax.grid(axis="y", alpha=0.3)

        # X-axis labels only in middle panel
        if col == 1:
            ax.set_xticklabels([f"k={k}" for k in k_values])
        else:
            ax.set_xticklabels([])

# Overall title and row labels
fig.text(0.5, 0.98, "How Failure Probability Affects Disruption Impact",
         fontsize=14, weight="bold", ha="center")
fig.text(0.02, 0.75, "SiouxFalls", fontsize=12, weight="bold", va="center", rotation=90)
fig.text(0.02, 0.25, "Mumford0", fontsize=12, weight="bold", va="center", rotation=90)

plt.tight_layout(rect=[0.04, 0, 1, 0.97])

# Save
output_path = Path(__file__).parent / "Analysis" / "folie3_failure_pattern.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_path}")
plt.close()

# Print summary stats
print("\n=== Summary Stats per p_fail (line_consecutive) ===")
for network_name, data in [("SiouxFalls", sf_data), ("Mumford0", mumford0_data)]:
    print(f"\n{network_name}:")
    for p_fail in p_fail_values:
        p_fail_data = data[data["case_p_fail"] == p_fail]
        if len(p_fail_data) > 0:
            print(f"  p_fail={p_fail}: median={p_fail_data['VSS_over_RP'].median():.3f}%, " +
                  f"max={p_fail_data['VSS_over_RP'].max():.3f}%, n={len(p_fail_data)}")
