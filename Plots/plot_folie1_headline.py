#!/usr/bin/env python3
"""
Folie 1: Headline – The Case for Stochastic Planning
- Headline-Tabelle (Median + P95 EVPI/RP, VSS/RP)
- 2 Balkenplots nebeneinander (EVPI/RP und VSS/RP)
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

# Calculate metrics per network
def compute_metrics(df, network_name):
    """Compute headline metrics: absolute EVPI and VSS"""

    # Filter valid cases (must have all required values)
    valid = df[["EVPI", "VSS_nom"]].notna().all(axis=1)
    df_valid = df[valid].copy()

    # Use absolute values (already in CSV)
    # Stats
    metrics = {
        "network": network_name,
        "median_EVPI": df_valid["EVPI"].median(),
        "p95_EVPI": df_valid["EVPI"].quantile(0.95),
        "median_VSS": df_valid["VSS_nom"].median(),
        "p95_VSS": df_valid["VSS_nom"].quantile(0.95),
        "n_cases": len(df_valid),
    }

    return metrics, df_valid

sf_metrics, sf_valid = compute_metrics(sf_data, "SiouxFalls")
mumford0_metrics, mumford0_valid = compute_metrics(mumford0_data, "Mumford0")

# Print summary
print(f"\n=== Metrics ===")
print(f"SiouxFalls ({sf_metrics['n_cases']} cases):")
print(f"  Median EVPI: {sf_metrics['median_EVPI']:.0f}")
print(f"  P95 EVPI: {sf_metrics['p95_EVPI']:.0f}")
print(f"  Median VSS: {sf_metrics['median_VSS']:.0f}")
print(f"  P95 VSS: {sf_metrics['p95_VSS']:.0f}")

print(f"\nMumford0 ({mumford0_metrics['n_cases']} cases):")
print(f"  Median EVPI: {mumford0_metrics['median_EVPI']:.0f}")
print(f"  P95 EVPI: {mumford0_metrics['p95_EVPI']:.0f}")
print(f"  Median VSS: {mumford0_metrics['median_VSS']:.0f}")
print(f"  P95 VSS: {mumford0_metrics['p95_VSS']:.0f}")

# Create figure with headline table + 2 subplots
fig = plt.figure(figsize=(16, 10))

# --- Table at top ---
ax_table = fig.add_subplot(2, 2, (1, 2))
ax_table.axis("tight")
ax_table.axis("off")

table_data = [
    ["Metric", "SiouxFalls", "Mumford0"],
    ["Median EVPI", f"{sf_metrics['median_EVPI']:.0f}", f"{mumford0_metrics['median_EVPI']:.0f}"],
    ["P95 EVPI", f"{sf_metrics['p95_EVPI']:.0f}", f"{mumford0_metrics['p95_EVPI']:.0f}"],
    ["Median VSS", f"{sf_metrics['median_VSS']:.0f}", f"{mumford0_metrics['median_VSS']:.0f}"],
    ["P95 VSS", f"{sf_metrics['p95_VSS']:.0f}", f"{mumford0_metrics['p95_VSS']:.0f}"],
    ["# Cases", f"{sf_metrics['n_cases']}", f"{mumford0_metrics['n_cases']}"],
]

table = ax_table.table(cellText=table_data, cellLoc="center", loc="center",
                       colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2.5)

# Header row styling
for i in range(3):
    table[(0, i)].set_facecolor("#1f77b4")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor("#f0f0f0")

# --- Barplot EVPI (left) ---
ax_evpi = fig.add_subplot(2, 2, 3)
networks = ["SiouxFalls", "Mumford0"]
medians_evpi = [sf_metrics["median_EVPI"], mumford0_metrics["median_EVPI"]]
p95_evpi = [sf_metrics["p95_EVPI"], mumford0_metrics["p95_EVPI"]]

x = np.arange(len(networks))
width = 0.35

bars1 = ax_evpi.bar(x - width/2, medians_evpi, width, label="Median", color="#1f77b4")
bars2 = ax_evpi.bar(x + width/2, p95_evpi, width, label="P95", color="#ff7f0e")

ax_evpi.set_ylabel("EVPI (cost units)", fontsize=12, weight="bold")
ax_evpi.set_title("Value of Perfect Information", fontsize=13, weight="bold")
ax_evpi.set_xticks(x)
ax_evpi.set_xticklabels(networks, fontsize=11)
ax_evpi.legend(fontsize=11)
ax_evpi.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax_evpi.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:.0f}", ha="center", va="bottom", fontsize=10)

# --- Barplot VSS (right) ---
ax_vss = fig.add_subplot(2, 2, 4)
medians_vss = [sf_metrics["median_VSS"], mumford0_metrics["median_VSS"]]
p95_vss = [sf_metrics["p95_VSS"], mumford0_metrics["p95_VSS"]]

bars3 = ax_vss.bar(x - width/2, medians_vss, width, label="Median", color="#2ca02c")
bars4 = ax_vss.bar(x + width/2, p95_vss, width, label="P95", color="#d62728")

ax_vss.set_ylabel("VSS (cost units)", fontsize=12, weight="bold")
ax_vss.set_title("Value of Stochastic Solution", fontsize=13, weight="bold")
ax_vss.set_xticks(x)
ax_vss.set_xticklabels(networks, fontsize=11)
ax_vss.legend(fontsize=11)
ax_vss.grid(axis="y", alpha=0.3)

# Add value labels on bars
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax_vss.text(bar.get_x() + bar.get_width()/2., height,
                   f"{height:.0f}", ha="center", va="bottom", fontsize=10)

plt.suptitle("The Case for Stochastic Planning", fontsize=16, weight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_path = Path(__file__).parent / "Analysis" / "folie1_headline.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_path}")
plt.close()
