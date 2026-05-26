#!/usr/bin/env python3
# scripts/compare_failure_structures.py
# -*- coding: utf-8 -*-
"""
Failure-structure analysis: how does VSS / EVPI depend on the SPATIAL pattern
and COUNT of disruption scenarios, controlling for the network operational
baseline?

Reads the dedicated `failurestructures` sweeps plus (optionally) the main map
sweeps for the full k-trend overlay. Emits:

  - Analysis/failure_structures/summary.md
  - Analysis/failure_structures/selection_modes.png    Axis 1
  - Analysis/failure_structures/scenario_count.png     Axis 2
  - Analysis/failure_structures/k_trend.png            Axis 3 (combined)

Usage:
    python scripts/compare_failure_structures.py
    python scripts/compare_failure_structures.py --metric EVPI   # EVPI/RP
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    print(f"[ERR] matplotlib not available: {e}", file=sys.stderr)
    sys.exit(1)


def _load_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path, sep=";")
    df["_source_path"] = path
    return df


def _format_pct(v: float) -> str:
    if pd.isna(v):
        return "—"
    return f"{v*100:.3f}%"


def _relative(df: pd.DataFrame, metric: str) -> pd.Series:
    s = pd.to_numeric(df[metric], errors="coerce")
    rp = pd.to_numeric(df["RP"], errors="coerce")
    rel = (s / rp).replace([np.inf, -np.inf], np.nan)
    return rel


# --- Axis 1: selection-mode comparison ---------------------------------------

SELECTION_ORDER = ["line_consecutive", "line_all", "share_stop", "random"]
SELECTION_LABELS = {
    "line_consecutive": "consecutive\n(corridor)",
    "line_all":         "line_all\n(scattered\non one line)",
    "share_stop":       "share_stop\n(hub cluster)",
    "random":           "random\n(diffuse)",
}


def _filter_axis1(df: pd.DataFrame) -> pd.DataFrame:
    """Axis 1 data: per_run=1, k in {1,2,3}, any selection."""
    return df[
        (pd.to_numeric(df["case_per_run"], errors="coerce") == 1)
        & (pd.to_numeric(df["case_k"], errors="coerce").isin([1, 2, 3]))
        & (df["case_selection"].isin(SELECTION_ORDER))
    ].copy()


def _plot_axis1(sf: pd.DataFrame, mu: pd.DataFrame, metric: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, df, net_name in zip(axes, [sf, mu], ["SiouxFalls", "Mumford0"]):
        if df.empty:
            ax.set_title(f"{net_name} (no data)")
            continue
        df = _filter_axis1(df)
        df["_rel"] = _relative(df, metric)
        ks = sorted(set(df["case_k"].dropna().astype(int)))
        n_ks = len(ks)
        n_sels = len(SELECTION_ORDER)
        width = 0.18
        positions = []
        data = []
        colors_per_sel = ["#08519c", "#6baed6", "#f16913", "#74c476"]
        for sel_i, sel in enumerate(SELECTION_ORDER):
            for k_i, k in enumerate(ks):
                sub = df[(df["case_selection"] == sel) & (df["case_k"] == k)]
                vals = sub["_rel"].dropna().values
                if len(vals) == 0:
                    continue
                pos = k_i + (sel_i - (n_sels - 1) / 2) * (width + 0.02)
                positions.append((pos, vals, colors_per_sel[sel_i]))
        for pos, vals, c in positions:
            bp = ax.boxplot([vals], positions=[pos], widths=width, patch_artist=True,
                            showfliers=True,
                            flierprops=dict(marker=".", markersize=3, alpha=0.4))
            bp["boxes"][0].set_facecolor(c)
            bp["boxes"][0].set_alpha(0.7)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"k={k}" for k in ks])
        ax.set_ylabel(f"{metric} / RP  (relative to nominal cost)")
        ax.set_title(net_name)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.grid(True, axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.2f}%"))
    # Shared legend
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=c, alpha=0.7, label=SELECTION_LABELS[s].replace("\n", " "))
               for c, s in zip(["#08519c", "#6baed6", "#f16913", "#74c476"], SELECTION_ORDER)]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{metric}/RP by failure-set spatial pattern (per_run=1, p_fail=0.5)", y=1.07)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# --- Axis 2: per_run sweep ----------------------------------------------------

def _filter_axis2(df: pd.DataFrame) -> pd.DataFrame:
    """per_run sweep at k=1, line_consecutive."""
    return df[
        (pd.to_numeric(df["case_k"], errors="coerce") == 1)
        & (df["case_selection"] == "line_consecutive")
        & (pd.to_numeric(df["case_per_run"], errors="coerce").isin([1, 2, 3, 5, 10]))
    ].copy()


def _plot_axis2(sf: pd.DataFrame, mu: pd.DataFrame, metric: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for df, net_name, color in [(sf, "SiouxFalls", "#2b8cbe"),
                                 (mu, "Mumford0",   "#e6550d")]:
        if df.empty:
            continue
        df = _filter_axis2(df)
        df["_rel"] = _relative(df, metric)
        grouped = (df.dropna(subset=["_rel"])
                     .groupby("case_per_run")["_rel"]
                     .agg(["median", "mean", "count"])
                     .reset_index()
                     .sort_values("case_per_run"))
        if grouped.empty:
            continue
        ax.plot(grouped["case_per_run"], grouped["median"], "o-",
                color=color, label=f"{net_name} (median)", linewidth=2)
        ax.plot(grouped["case_per_run"], grouped["mean"], "x--",
                color=color, label=f"{net_name} (mean)", alpha=0.5, markersize=6)
    ax.set_xlabel("case_per_run  (failure scenarios per case, S = per_run + 1 nominal)")
    ax.set_ylabel(f"{metric} / RP")
    ax.set_title(f"{metric}/RP vs scenario count per case  (k=1, line_consecutive, p_fail=0.5)")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.3f}%"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# --- Axis 3: k trend (combined with main map) --------------------------------

def _filter_ktrend(df: pd.DataFrame) -> pd.DataFrame:
    """k trend uses per_run=1, line_consecutive only."""
    return df[
        (pd.to_numeric(df["case_per_run"], errors="coerce") == 1)
        & (df["case_selection"] == "line_consecutive")
    ].copy()


def _plot_axis3(combined_sf: pd.DataFrame, combined_mu: pd.DataFrame,
                metric: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for df, net_name, color in [(combined_sf, "SiouxFalls", "#2b8cbe"),
                                 (combined_mu, "Mumford0",   "#e6550d")]:
        if df.empty:
            continue
        df = _filter_ktrend(df)
        df["_rel"] = _relative(df, metric)
        grouped = (df.dropna(subset=["_rel"])
                     .groupby("case_k")["_rel"]
                     .agg(["median", "mean", "count"])
                     .reset_index()
                     .sort_values("case_k"))
        if grouped.empty:
            continue
        ax.plot(grouped["case_k"], grouped["median"], "o-",
                color=color, label=f"{net_name} (median, n per point shown)", linewidth=2)
        ax.plot(grouped["case_k"], grouped["mean"], "x--",
                color=color, label=f"{net_name} (mean)", alpha=0.5, markersize=6)
        for _, row in grouped.iterrows():
            ax.annotate(f"n={int(row['count'])}", (row["case_k"], row["median"]),
                        textcoords="offset points", xytext=(0, 6),
                        fontsize=7, ha="center", color=color, alpha=0.7)
    ax.set_xlabel("case_k  (number of disrupted edges per failure scenario)")
    ax.set_ylabel(f"{metric} / RP")
    ax.set_title(f"{metric}/RP vs disruption size k  (line_consecutive, per_run=1, main + failurestructures combined)")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.3f}%"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


# --- Summary writer ---------------------------------------------------------

def _summary_block(df: pd.DataFrame, metric: str, axis_filter, title: str,
                   group_cols: List[str]) -> List[str]:
    if df.empty:
        return [f"## {title}", "", "_no data_", ""]
    df = axis_filter(df)
    df["_rel"] = _relative(df, metric)
    grouped = (df.dropna(subset=["_rel"])
                 .groupby(group_cols)["_rel"]
                 .agg(["count", "median", "mean", lambda s: s.quantile(0.95)])
                 .reset_index())
    grouped = grouped.rename(columns={"<lambda_0>": "p95"})
    lines = [f"## {title}", ""]
    header = "| " + " | ".join(group_cols) + " | n | median | mean | P95 |"
    align = "|" + "---|" * (len(group_cols) + 4)
    lines.append(header)
    lines.append(align)
    for _, row in grouped.iterrows():
        cells = [str(row[c]) for c in group_cols] + [
            f"{int(row['count'])}",
            _format_pct(row["median"]),
            _format_pct(row["mean"]),
            _format_pct(row["p95"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _write_summary(
    sf: pd.DataFrame, mu: pd.DataFrame,
    combined_sf: pd.DataFrame, combined_mu: pd.DataFrame,
    metric: str, out_path: str,
) -> None:
    lines: List[str] = []
    lines.append(f"# Failure-structure analysis: SiouxFalls vs Mumford0")
    lines.append("")
    lines.append(f"Primary metric: **{metric} / RP** (relative to nominal cost)")
    lines.append("")
    lines.append("Three controlled experiments:")
    lines.append("1. **Selection-mode comparison** — does the spatial pattern of disruption matter?")
    lines.append("2. **Scenario count (per_run)** — does VSS grow when more scenarios per case?")
    lines.append("3. **Disruption size (k)** — does VSS scale with number of failed edges?")
    lines.append("")

    lines.append("---")
    for net_name, df in [("SiouxFalls", sf), ("Mumford0", mu)]:
        lines.append(f"# {net_name}")
        lines.append("")
        lines.extend(_summary_block(df, metric, _filter_axis1,
                                     "Axis 1 — selection mode × k (per_run=1)",
                                     ["case_k", "case_selection"]))
        lines.extend(_summary_block(df, metric, _filter_axis2,
                                     "Axis 2 — scenario count (k=1, line_consecutive)",
                                     ["case_per_run"]))
    lines.append("---")
    lines.append("")
    lines.append("# k-trend (failurestructures + main map combined, line_consecutive only)")
    lines.append("")
    for net_name, df in [("SiouxFalls", combined_sf), ("Mumford0", combined_mu)]:
        lines.extend(_summary_block(df, metric, _filter_ktrend,
                                     f"{net_name} — {metric}/RP by k",
                                     ["case_k"]))

    lines.append("---")
    lines.append("")
    lines.append("Plots: `selection_modes.png`, `scenario_count.png`, `k_trend.png`")
    lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="Results")
    ap.add_argument("--metric", default="VSS_nom", choices=["VSS_nom", "EVPI"],
                    help="Primary metric (default VSS_nom; pass --metric EVPI to use EVPI/RP)")
    ap.add_argument("--out-dir", default="Analysis/failure_structures")
    ns = ap.parse_args()

    # Failure-structure sweeps (the new ones)
    sf = _load_csv(os.path.join(ns.results_root, "vss_sf_failurestructures_redo", "vss_map.csv"))
    mu = _load_csv(os.path.join(ns.results_root, "vss_mumford0_failurestructures_redo", "vss_map.csv"))
    if sf is None and mu is None:
        print("[ERR] no failurestructures vss_map.csv found in either network", file=sys.stderr)
        sys.exit(1)
    sf = sf if sf is not None else pd.DataFrame(columns=["case_k", "case_per_run", "case_selection"])
    mu = mu if mu is not None else pd.DataFrame(columns=["case_k", "case_per_run", "case_selection"])
    print(f"  SF failurestructures: {len(sf)} rows")
    print(f"  Mumford0 failurestructures: {len(mu)} rows")

    # Main map sweeps for the combined k-trend
    sf_main = _load_csv(os.path.join(ns.results_root, "vss_sf_main_redo", "vss_map.csv"))
    mu_main = _load_csv(os.path.join(ns.results_root, "vss_mumford0_redo", "vss_map.csv"))
    combined_sf = pd.concat([d for d in [sf, sf_main] if d is not None], ignore_index=True)
    combined_mu = pd.concat([d for d in [mu, mu_main] if d is not None], ignore_index=True)
    print(f"  combined SF (failurestructures + main):    {len(combined_sf)} rows")
    print(f"  combined Mumford0 (failurestructures + main): {len(combined_mu)} rows")

    os.makedirs(ns.out_dir, exist_ok=True)
    _write_summary(sf, mu, combined_sf, combined_mu, ns.metric,
                   os.path.join(ns.out_dir, "summary.md"))
    _plot_axis1(sf, mu, ns.metric, os.path.join(ns.out_dir, "selection_modes.png"))
    _plot_axis2(sf, mu, ns.metric, os.path.join(ns.out_dir, "scenario_count.png"))
    _plot_axis3(combined_sf, combined_mu, ns.metric,
                os.path.join(ns.out_dir, "k_trend.png"))
    print()
    print(f"All artefacts in {ns.out_dir}/")


if __name__ == "__main__":
    main()
