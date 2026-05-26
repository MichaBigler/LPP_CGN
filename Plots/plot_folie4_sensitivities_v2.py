#!/usr/bin/env python3
"""
Folie 4: Operational Sensitivities (v2)
- Log x-axis for train_capacity and cost_repl_freq
- Secondary y-axis: SF (left) vs. Mumford0 (right) with different scales
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data from sensitivity sweeps
results_dir = Path(__file__).parent / "Results"

sweeps = {
    "train_capacity": {
        "sf": "vss_sf_traincap_redo/vss_map.csv",
        "m0": "vss_mumford0_traincap_redo/vss_map.csv",
        "param_col": "train_capacity",
        "label": "Train Capacity (seats)",
        "log_x": True,
    },
    "cost_repl_freq": {
        "sf": "vss_sf_replcost_redo/vss_map.csv",
        "m0": "vss_mumford0_replcost_redo/vss_map.csv",
        "param_col": "cost_repl_freq",
        "label": "Replacement Cost Frequency",
        "log_x": True,
    },
    "p_fail": {
        "sf": "vss_sf_pfail_redo/vss_map.csv",
        "m0": "vss_mumford0_pfail_redo/vss_map.csv",
        "param_col": "case_p_fail",
        "label": "Failure Probability",
        "log_x": False,
    },
    "overdemand": {
        "sf": "vss_sf_overdemand_redo/vss_map.csv",
        "m0": "vss_mumford0_overdemand_redo/vss_map.csv",
        "param_col": "overdemand_multiplier",
        "label": "Overdemand Multiplier",
        "log_x": False,
    },
    "bypass": {
        "sf": "vss_sf_bypass_redo/vss_map.csv",
        "m0": "vss_mumford0_bypass_redo/vss_map.csv",
        "param_col": "bypass_multiplier",
        "label": "Bypass Multiplier",
        "log_x": False,
    },
}

# Load and prepare each sweep
def load_sweep(file_path):
    """Load CSV"""
    try:
        df = pd.read_csv(file_path, sep=";")
        valid = df[["RP", "VSS_nom"]].notna().all(axis=1)
        df_valid = df[valid].copy()
        return df_valid
    except FileNotFoundError:
        return None

# Load all sweeps
sweep_data = {}
for name, config in sweeps.items():
    sf_data = load_sweep(results_dir / config["sf"])
    m0_data = load_sweep(results_dir / config["m0"])
    sweep_data[name] = {
        "sf": sf_data,
        "m0": m0_data,
        "param_col": config["param_col"],
        "label": config["label"],
        "log_x": config["log_x"],
    }
    print(f"{name}: SF={len(sf_data) if sf_data is not None else 0}, M0={len(m0_data) if m0_data is not None else 0}")

# Create figure: 2 rows × 3 columns (5 subplots, 1 empty)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for idx, (sweep_name, sweep_config) in enumerate(sweep_data.items()):
    ax1 = axes[idx]  # Primary y-axis (SF)

    sf_data = sweep_config["sf"]
    m0_data = sweep_config["m0"]
    param_col = sweep_config["param_col"]
    label = sweep_config["label"]
    log_x = sweep_config["log_x"]

    if sf_data is None or m0_data is None:
        ax1.text(0.5, 0.5, f"{sweep_name}\nNo data", ha="center", va="center", transform=ax1.transAxes)
        ax1.set_xticks([])
        ax1.set_yticks([])
        continue

    # Group by parameter value and compute median VSS/RP
    def compute_median_per_param(df, param_col):
        """Group by param and get median VSS/RP"""
        groups = []
        for param_val in sorted(df[param_col].unique()):
            subset = df[df[param_col] == param_val]["VSS_nom"]
            if len(subset) > 0:
                groups.append({
                    "param": param_val,
                    "median": subset.median(),
                    "p25": subset.quantile(0.25),
                    "p75": subset.quantile(0.75),
                    "n": len(subset),
                })
        return pd.DataFrame(groups)

    sf_trend = compute_median_per_param(sf_data, param_col)
    m0_trend = compute_median_per_param(m0_data, param_col)

    # Create secondary y-axis for Mumford0
    ax2 = ax1.twinx()

    # Plot SF on primary axis (left)
    if len(sf_trend) > 0:
        sf_err_lower = (sf_trend["median"] - sf_trend["p25"]).values
        sf_err_upper = (sf_trend["p75"] - sf_trend["median"]).values
        ax1.errorbar(sf_trend["param"], sf_trend["median"],
                    yerr=[sf_err_lower, sf_err_upper],
                    marker="o", markersize=7, linewidth=2.5, label="SiouxFalls",
                    color="#1f77b4", capsize=4, capthick=1.5, alpha=0.8)
        ax1.set_ylabel("SF: VSS (cost units)", fontsize=9, weight="bold", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

    # Plot Mumford0 on secondary axis (right)
    if len(m0_trend) > 0:
        m0_err_lower = (m0_trend["median"] - m0_trend["p25"]).values
        m0_err_upper = (m0_trend["p75"] - m0_trend["median"]).values
        ax2.errorbar(m0_trend["param"], m0_trend["median"],
                    yerr=[m0_err_lower, m0_err_upper],
                    marker="s", markersize=7, linewidth=2.5, label="Mumford0",
                    color="#ff7f0e", capsize=4, capthick=1.5, alpha=0.8)
        ax2.set_ylabel("M0: VSS (cost units)", fontsize=9, weight="bold", color="#ff7f0e")
        ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    # X-axis settings
    ax1.set_xlabel(label, fontsize=10, weight="bold")
    if log_x:
        ax1.set_xscale("log")

    ax1.set_title(sweep_name.replace("_", " ").title(), fontsize=11, weight="bold")
    ax1.grid(alpha=0.3)

# Hide the last (empty) subplot
axes[-1].axis("off")

# Overall title
fig.text(0.5, 0.98, "Operational Sensitivities: How Parameters Affect Stochastic Planning Value",
         fontsize=13, weight="bold", ha="center")

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save
output_path = Path(__file__).parent / "Analysis" / "folie4_sensitivities.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {output_path}")
plt.close()

print("\n=== Summary: Direction of Effects ===")
for sweep_name, sweep_config in sweep_data.items():
    sf_data = sweep_config["sf"]
    if sf_data is None:
        continue

    param_col = sweep_config["param_col"]
    trend = compute_median_per_param(sf_data, param_col)

    if len(trend) > 1:
        first_val = trend.iloc[0]["median"]
        last_val = trend.iloc[-1]["median"]
        direction = "INCREASES" if last_val > first_val else "DECREASES" if last_val < first_val else "STABLE"
        print(f"{sweep_name:20s}: {direction}")
