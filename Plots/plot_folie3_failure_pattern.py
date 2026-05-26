#!/usr/bin/env python3
"""
Folie 3: Failure Pattern Effect
- How spatial distribution of disruption (selection mode) affects VSS/RP
- 2×4 Boxplot grid: 2 networks × 4 selection modes
- Each shows k=1,2,3 side-by-side
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

# Prepare data
def prepare_data(df):
    """Filter valid cases and compute VSS/RP ratio"""
    valid = df[["RP", "VSS_nom"]].notna().all(axis=1)
    df_valid = df[valid].copy()
    df_valid["VSS_over_RP"] = (df_valid["VSS_nom"] / df_valid["RP"]) * 100
    # Filter only k=1,2,3
    df_valid = df_valid[df_valid["case_k"].isin([1, 2, 3])]
    return df_valid

sf_data = prepare_data(sf_data)
mumford0_data = prepare_data(mumford0_data)

print(f"Valid cases (k=1..3): SF={len(sf_data)}, Mumford0={len(mumford0_data)}")

# Get selection modes
sf_modes = sorted(sf_data["case_selection"].unique())
m0_modes = sorted(mumford0_data["case_selection"].unique())

print(f"SF selection modes: {sf_modes}")
print(f"Mumford0 selection modes: {m0_modes}")

# Ensure consistent modes across both networks
all_modes = sorted(set(sf_modes) | set(m0_modes))
print(f"All modes: {all_modes}")

# Use only modes that exist in both networks
common_modes = sorted(set(sf_modes) & set(m0_modes))
print(f"Common modes: {common_modes}")

# Create figure: 2 rows (networks) × n columns (selection modes)
n_modes = len(common_modes)
fig, axes = plt.subplots(2, n_modes, figsize=(4*n_modes, 8))
if n_modes == 1:
    axes = axes.reshape(2, 1)

for row, (data, network_name) in enumerate([(sf_data, "SiouxFalls"), (mumford0_data, "Mumford0")]):
    for col, mode in enumerate(common_modes):
        ax = axes[row, col]

        # Filter data for this mode
        mode_data = data[data["case_selection"] == mode]

        if len(mode_data) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{mode}\n(no data)", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Prepare data for boxplot: one group per k
        boxplot_data = []
        k_values = [1, 2, 3]
        for k in k_values:
            subset = mode_data[mode_data["case_k"] == k]["VSS_over_RP"].values
            if len(subset) > 0:
                boxplot_data.append(subset)
            else:
                boxplot_data.append([np.nan])

        # Create boxplot
        bp = ax.boxplot(boxplot_data, tick_labels=[f"k={k}" for k in k_values],
                       patch_artist=True, widths=0.6)

        # Styling
        for patch in bp['boxes']:
            patch.set_facecolor("#1f77b4" if row == 0 else "#ff7f0e")
            patch.set_alpha(0.7)

        ax.set_ylabel("VSS / RP (%)", fontsize=9)
        ax.set_title(f"{mode}", fontsize=10, weight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add count annotations
        counts = [len(mode_data[mode_data["case_k"] == k]) for k in k_values]
        title_with_counts = f"{mode}\n(n={', '.join(map(str, counts))})"
        ax.set_title(title_with_counts, fontsize=9)

# Overall title and labels
fig.text(0.5, 0.98, "How Spatial Disruption Pattern Matters", fontsize=14, weight="bold", ha="center")
fig.text(0.02, 0.75, "SiouxFalls", fontsize=12, weight="bold", va="center", rotation=90)
fig.text(0.02, 0.25, "Mumford0", fontsize=12, weight="bold", va="center", rotation=90)

plt.suptitle("", fontsize=1)  # Empty to avoid double title
plt.tight_layout(rect=[0.04, 0, 1, 0.97])

# Save
output_path = Path(__file__).parent / "Analysis" / "folie3_failure_pattern.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_path}")
plt.close()

# Print summary stats
print("\n=== Summary Stats per Selection Mode ===")
for network_name, data in [("SiouxFalls", sf_data), ("Mumford0", mumford0_data)]:
    print(f"\n{network_name}:")
    for mode in all_modes:
        mode_data = data[data["case_selection"] == mode]
        if len(mode_data) > 0:
            print(f"  {mode:20s}: median={mode_data['VSS_over_RP'].median():.3f}%, max={mode_data['VSS_over_RP'].max():.3f}%, n={len(mode_data)}")
