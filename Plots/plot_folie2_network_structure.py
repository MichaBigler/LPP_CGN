#!/usr/bin/env python3
"""
Folie 2: How Network Structure Matters
- Linienplot: k (x-axis) vs. median VSS/RP (y-axis) with error bars (P25/P75)
- Top 5 Cases Tabelle pro Netz
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
def prepare_data(df, network_name):
    """Filter valid cases"""
    valid = df[["RP", "VSS_nom"]].notna().all(axis=1)
    df_valid = df[valid].copy()
    return df_valid

sf_data = prepare_data(sf_data, "SiouxFalls")
mumford0_data = prepare_data(mumford0_data, "Mumford0")

print(f"\nValid cases: SF={len(sf_data)}, Mumford0={len(mumford0_data)}")

# Compute k-trend statistics
def compute_k_trend(df):
    """Group by k, compute median + P25/P75 for VSS/RP"""
    groups = []
    for k in sorted(df["case_k"].unique()):
        subset = df[df["case_k"] == k]["VSS_nom"]
        if len(subset) > 0:
            groups.append({
                "k": k,
                "median": subset.median(),
                "p25": subset.quantile(0.25),
                "p75": subset.quantile(0.75),
                "n": len(subset),
            })
    return pd.DataFrame(groups)

sf_trend = compute_k_trend(sf_data)
mumford0_trend = compute_k_trend(mumford0_data)

print(f"\nSiouxFalls k-trend:")
print(sf_trend)
print(f"\nMumford0 k-trend:")
print(mumford0_trend)

# Get top 1 case per k (most extreme case per disruption size)
def get_top_cases_per_k(df, network_name):
    """Return top case per disruption size k"""
    top_cases = []
    for k in sorted(df["case_k"].unique()):
        subset = df[df["case_k"] == k]
        best = subset.nlargest(1, "VSS_nom").iloc[0]
        top_cases.append({
            "k": int(best["case_k"]),
            "Edges": best["case_edges"],
            "RP": best["RP"],
            "VSS": best["VSS_nom"],
            "VSS/RP": best["VSS_nom"],
        })
    return pd.DataFrame(top_cases)

sf_top = get_top_cases_per_k(sf_data, "SiouxFalls")
mumford0_top = get_top_cases_per_k(mumford0_data, "Mumford0")

print(f"\nMost Extreme Case per k - SiouxFalls:")
print(sf_top.to_string(index=False))
print(f"\nMost Extreme Case per k - Mumford0:")
print(mumford0_top.to_string(index=False))

# Create figure
fig = plt.figure(figsize=(14, 10))

# --- Lineplot with error bars ---
ax_line = fig.add_subplot(2, 1, 1)

# Ensure both trend dataframes have the same k values for plotting
k_vals = sorted(set(sf_trend["k"]) | set(mumford0_trend["k"]))

# Plot SF
sf_x = sf_trend["k"].values
sf_med = sf_trend["median"].values
sf_err_lower = (sf_trend["median"] - sf_trend["p25"]).values
sf_err_upper = (sf_trend["p75"] - sf_trend["median"]).values

ax_line.errorbar(sf_x, sf_med, yerr=[sf_err_lower, sf_err_upper],
                 marker="o", markersize=8, linewidth=2.5, label="SiouxFalls",
                 color="#1f77b4", capsize=5, capthick=2)

# Plot Mumford0
m0_x = mumford0_trend["k"].values
m0_med = mumford0_trend["median"].values
m0_err_lower = (mumford0_trend["median"] - mumford0_trend["p25"]).values
m0_err_upper = (mumford0_trend["p75"] - mumford0_trend["median"]).values

ax_line.errorbar(m0_x, m0_med, yerr=[m0_err_lower, m0_err_upper],
                 marker="s", markersize=8, linewidth=2.5, label="Mumford0",
                 color="#ff7f0e", capsize=5, capthick=2)

ax_line.set_xlabel("Disruption Size (k = # disrupted edges)", fontsize=12, weight="bold")
ax_line.set_ylabel("VSS (cost units)", fontsize=12, weight="bold")
ax_line.set_title("How Network Structure Affects Stochastic Planning Value", fontsize=13, weight="bold")
ax_line.legend(fontsize=11, loc="best")
ax_line.grid(alpha=0.3)
ax_line.set_xticks(k_vals)

# --- Most Extreme Cases Tables (matplotlib table) ---
ax_table_bottom = fig.add_subplot(2, 1, 2)
ax_table_bottom.axis("off")

# Create matplotlib tables side by side
def create_table_display(ax, sf_df, m0_df):
    """Create two matplotlib tables side by side"""
    # SiouxFalls table
    sf_table_data = [["k", "Edges", "RP", "VSS", "VSS/RP (%)"]]
    for _, row in sf_df.iterrows():
        sf_table_data.append([
            str(int(row['k'])),
            str(row['Edges'])[:15],
            f"{row['RP']:.0f}",
            f"{row['VSS']:.0f}",
            f"{row['VSS/RP']:.2f}%"
        ])

    # Mumford0 table
    m0_table_data = [["k", "Edges", "RP", "VSS", "VSS/RP (%)"]]
    for _, row in m0_df.iterrows():
        m0_table_data.append([
            str(int(row['k'])),
            str(row['Edges'])[:15],
            f"{row['RP']:.0f}",
            f"{row['VSS']:.0f}",
            f"{row['VSS/RP']:.2f}%"
        ])

    # Create tables
    table_sf = ax.table(cellText=sf_table_data, loc="left", bbox=[0, 0, 0.45, 1],
                        cellLoc="center")
    table_sf.auto_set_font_size(False)
    table_sf.set_fontsize(10)
    table_sf.scale(1, 2)

    table_m0 = ax.table(cellText=m0_table_data, loc="right", bbox=[0.52, 0, 0.45, 1],
                        cellLoc="center")
    table_m0.auto_set_font_size(False)
    table_m0.set_fontsize(10)
    table_m0.scale(1, 2)

    # Header styling
    for i in range(5):
        table_sf[(0, i)].set_facecolor("#1f77b4")
        table_sf[(0, i)].set_text_props(weight="bold", color="white")
        table_m0[(0, i)].set_facecolor("#ff7f0e")
        table_m0[(0, i)].set_text_props(weight="bold", color="white")

    # Alternating row colors
    for i in range(1, len(sf_table_data)):
        for j in range(5):
            if i % 2 == 0:
                table_sf[(i, j)].set_facecolor("#f9f9f9")

    for i in range(1, len(m0_table_data)):
        for j in range(5):
            if i % 2 == 0:
                table_m0[(i, j)].set_facecolor("#f9f9f9")

    # Add titles
    ax.text(0.225, 1.08, "SiouxFalls - Most Extreme Case per k",
           transform=ax.transAxes, ha="center", fontsize=11, weight="bold")
    ax.text(0.745, 1.08, "Mumford0 - Most Extreme Case per k",
           transform=ax.transAxes, ha="center", fontsize=11, weight="bold")

create_table_display(ax_table_bottom, sf_top, mumford0_top)

plt.suptitle("Network Structure & Disruption Size Effects", fontsize=15, weight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save
output_path = Path(__file__).parent / "Analysis" / "folie2_network_structure.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_path}")
plt.close()
