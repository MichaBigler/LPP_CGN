#!/usr/bin/env python3
# scripts/compare_vss_networks.py
# -*- coding: utf-8 -*-
"""
Side-by-side comparison of VSS results between SiouxFalls and Mumford0.

Reads per-sweep `vss_map.csv` files (produced by aggregate_all_sweeps.sh)
and emits:
  - Analysis/network_comparison/summary.md         human-readable narrative
  - Analysis/network_comparison/main_map_vss.png   k-stratified boxplot
  - Analysis/network_comparison/sensitivity.png    parameter sweeps overlaid

VSS_nom is the primary metric (VSS_mean as cross-check). EVPI is NOT shown
when running against the original sweep (WS unreliable); set USE_EVPI=1 to
include it once the deterministic-WS re-run is aggregated.

Usage:
    python scripts/compare_vss_networks.py
    python scripts/compare_vss_networks.py --results-root Results --suffix _redo

The --suffix flag selects which set of aggregated sweeps to read:
    "" (default) → Results/vss_<slug>/vss_map.csv     (original sweeps)
    "_redo"       → Results/vss_<slug>_redo/vss_map.csv  (deterministic re-run)
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
    matplotlib.use("Agg")  # headless on scicore
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    print(f"[ERR] matplotlib not available: {e}", file=sys.stderr)
    sys.exit(1)


# Sweep slugs, mapped to network and the (display name, parameter column)
# we want to scan in the sensitivity plot. main map is handled separately.
SWEEPS: Dict[str, Tuple[str, str, Optional[str]]] = {
    # slug:                  (network,    label,                 param_col_or_None)
    "sf_main":               ("SF",       "main",                None),
    "sf_bypass":             ("SF",       "bypass",              "bypass_multiplier"),
    "sf_overdemand":         ("SF",       "overdemand",          "overdemand_multiplier"),
    "sf_pfail":              ("SF",       "p_fail",              "case_p_fail"),
    "sf_replcost":           ("SF",       "repl cost",           "cost_repl_freq"),
    "sf_traincap":           ("SF",       "train capacity",      "train_capacity"),
    "mumford0":              ("Mumford0", "main",                None),
    "mumford0_bypass":       ("Mumford0", "bypass",              "bypass_multiplier"),
    "mumford0_overdemand":   ("Mumford0", "overdemand",          "overdemand_multiplier"),
    "mumford0_pfail":        ("Mumford0", "p_fail",              "case_p_fail"),
    "mumford0_replcost":     ("Mumford0", "repl cost",           "cost_repl_freq"),
    "mumford0_traincap":     ("Mumford0", "train capacity",      "train_capacity"),
}


def _load_sweep(results_root: str, slug: str, suffix: str) -> Optional[pd.DataFrame]:
    # SF main sweep historically lived at Results/vss_544_complete/ (legacy)
    # before the slug naming convention. For local OLD-data analysis we
    # accept several aliases per slug.
    aliases = {
        "sf_main":      ["vss_544_complete"],
        "sf_bypass":    [],
        "sf_overdemand": [],
        "sf_pfail":     [],
        "sf_replcost":  [],
        "sf_traincap":  [],
    }
    # Trim "sf_" prefix to fall back to legacy folder names like vss_bypass
    local_short = slug[3:] if slug.startswith("sf_") else slug
    candidates = [
        os.path.join(results_root, f"vss_{slug}{suffix}", "vss_map.csv"),
        os.path.join(results_root, f"vss_{slug}", "vss_map.csv"),
        # legacy SF: drop the "sf_" prefix to get vss_bypass, vss_pfail, ...
        os.path.join(results_root, f"vss_{local_short}", "vss_map.csv"),
    ]
    for alias in aliases.get(slug, []):
        candidates.append(os.path.join(results_root, alias, "vss_map.csv"))
    for path in candidates:
        if os.path.isfile(path):
            df = pd.read_csv(path, sep=";")
            df["_source_path"] = path
            return df
    return None


def _format_eur(v: float) -> str:
    if pd.isna(v):
        return "—"
    if abs(v) >= 1e6:
        return f"{v/1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v/1e3:.1f}k"
    return f"{v:.0f}"


def _summary_stats(df: pd.DataFrame, metric: str = "VSS_nom") -> Dict[str, float]:
    s = pd.to_numeric(df[metric], errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "n":       len(s),
        "mean":    float(s.mean()),
        "median":  float(s.median()),
        "p75":     float(s.quantile(0.75)),
        "p95":     float(s.quantile(0.95)),
        "max":     float(s.max()),
        "neg":     int((s < -1).sum()),  # negatives — sign of solver noise
    }


def _make_main_map_boxplot(
    by_slug: Dict[str, pd.DataFrame], out_path: str, use_evpi: bool,
) -> None:
    """k-stratified boxplot of VSS_nom for both networks on the main map."""
    sf = by_slug.get("sf_main")
    mu = by_slug.get("mumford0")
    if sf is None or mu is None:
        print(f"[WARN] main_map_vss.png skipped — missing sf_main or mumford0", file=sys.stderr)
        return

    metric = "EVPI" if use_evpi else "VSS_nom"
    ks = sorted(set(sf["case_k"].dropna().astype(int)) | set(mu["case_k"].dropna().astype(int)))

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35
    positions = []
    data = []
    labels = []
    colors = []
    for i, k in enumerate(ks):
        for off, (net_df, net_name, color) in enumerate([
            (sf, "SF", "#2b8cbe"),
            (mu, "Mumford0", "#e6550d"),
        ]):
            vals = pd.to_numeric(
                net_df.loc[net_df["case_k"] == k, metric], errors="coerce"
            ).dropna()
            if vals.empty:
                continue
            positions.append(i * 1.0 + (off - 0.5) * (width + 0.05))
            data.append(vals.values)
            labels.append(f"{net_name} k={k}")
            colors.append(color)

    if not data:
        print(f"[WARN] no data for main_map plot", file=sys.stderr)
        return

    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True,
                    showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_xticks([i * 1.0 for i in range(len(ks))])
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel(f"{metric} [cost units]")
    ax.set_title(f"{metric} by network and disruption size (main map sweep)")
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
    ax.grid(True, axis="y", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#2b8cbe", alpha=0.6, label="SiouxFalls"),
        Patch(facecolor="#e6550d", alpha=0.6, label="Mumford0"),
    ]
    ax.legend(handles=handles, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _make_sensitivity_plot(
    by_slug: Dict[str, pd.DataFrame], out_path: str, use_evpi: bool,
) -> None:
    """5-panel sensitivity sweep with both networks overlaid per panel."""
    metric = "EVPI" if use_evpi else "VSS_nom"
    sens_slugs = [(slug, info) for slug, info in SWEEPS.items()
                  if info[2] is not None]
    # Group by sensitivity label (de-duplicated across networks)
    by_label: Dict[str, List[Tuple[str, str, str]]] = {}
    for slug, (net, label, param) in sens_slugs:
        by_label.setdefault(label, []).append((net, slug, param))

    labels = list(by_label.keys())
    n = len(labels)
    if n == 0:
        print(f"[WARN] no sensitivity sweeps available", file=sys.stderr)
        return
    ncol = min(3, n)
    nrow = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 4.5, nrow * 3.5),
                              squeeze=False)
    flat_axes = [ax for row in axes for ax in row]
    for ax, label in zip(flat_axes, labels):
        for net, slug, param in by_label[label]:
            df = by_slug.get(slug)
            if df is None or param not in df.columns:
                continue
            xs = pd.to_numeric(df[param], errors="coerce")
            ys = pd.to_numeric(df[metric], errors="coerce")
            mask = xs.notna() & ys.notna()
            if not mask.any():
                continue
            grouped = (
                pd.DataFrame({"x": xs[mask], "y": ys[mask]})
                .groupby("x")["y"]
                .agg(["median", "mean", "count"])
                .reset_index()
                .sort_values("x")
            )
            color = "#2b8cbe" if net == "SF" else "#e6550d"
            ax.plot(grouped["x"], grouped["median"], "o-", color=color,
                    label=f"{net} (median)", alpha=0.85)
            ax.plot(grouped["x"], grouped["mean"], "x--", color=color,
                    label=f"{net} (mean)", alpha=0.5, markersize=5)
        ax.set_title(label)
        ax.set_ylabel(metric)
        ax.set_xlabel(by_label[label][0][2])
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    # Hide unused axes
    for ax in flat_axes[len(labels):]:
        ax.set_visible(False)
    fig.suptitle(f"{metric} sensitivity per parameter — SiouxFalls vs Mumford0", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _write_summary(
    by_slug: Dict[str, pd.DataFrame], out_path: str, use_evpi: bool,
) -> None:
    metric = "EVPI" if use_evpi else "VSS_nom"
    lines: List[str] = []
    lines.append(f"# Network comparison: SiouxFalls vs Mumford0")
    lines.append("")
    lines.append(f"Primary metric: **{metric}**  "
                 f"({'EVPI = RP − WS' if use_evpi else 'VSS_nom = EEV_nom − RP'})")
    lines.append("")

    # --- 1) Per-sweep counts and missing data ---
    lines.append("## Sweep inventory")
    lines.append("")
    lines.append("| slug | rows | source |")
    lines.append("|---|---:|---|")
    for slug in SWEEPS:
        df = by_slug.get(slug)
        if df is None:
            lines.append(f"| {slug} | — | _missing_ |")
        else:
            lines.append(f"| {slug} | {len(df)} | `{df['_source_path'].iloc[0]}` |")
    lines.append("")

    # --- 2) Main-map summary table ---
    sf_main = by_slug.get("sf_main")
    mu_main = by_slug.get("mumford0")
    if sf_main is not None and mu_main is not None:
        lines.append(f"## Main map — {metric} by case_k")
        lines.append("")
        lines.append(f"| network | k | n | median | mean | P75 | P95 | max | n<sub><0</sub> |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for net_name, df in [("SF", sf_main), ("Mumford0", mu_main)]:
            ks = sorted(df["case_k"].dropna().astype(int).unique())
            for k in ks:
                sub = df[df["case_k"] == k]
                stats = _summary_stats(sub, metric)
                if not stats:
                    continue
                lines.append(
                    f"| {net_name} | {k} | {stats['n']} | "
                    f"{_format_eur(stats['median'])} | {_format_eur(stats['mean'])} | "
                    f"{_format_eur(stats['p75'])} | {_format_eur(stats['p95'])} | "
                    f"{_format_eur(stats['max'])} | {stats['neg']} |"
                )
        lines.append("")
        # narrative
        sf_stats = _summary_stats(sf_main, metric)
        mu_stats = _summary_stats(mu_main, metric)
        if sf_stats and mu_stats:
            lines.append(
                f"**Observation:** SF median {metric} = {_format_eur(sf_stats['median'])} "
                f"vs Mumford0 median = {_format_eur(mu_stats['median'])}. "
                f"SF P95 = {_format_eur(sf_stats['p95'])} vs Mumford0 P95 = {_format_eur(mu_stats['p95'])}."
            )
            lines.append("")
            if sf_stats["neg"] + mu_stats["neg"] > 0:
                lines.append(
                    f"**Solver noise check:** SF has {sf_stats['neg']} cases with negative {metric}, "
                    f"Mumford0 has {mu_stats['neg']}. (Should be 0 if WS sub-solves are correct.)"
                )
                lines.append("")

    # --- 3) Top-N worst failures per network (main map) ---
    if sf_main is not None and mu_main is not None and "case_edges" in sf_main.columns:
        lines.append(f"## Top-10 highest {metric} per network (main map)")
        lines.append("")
        for net_name, df in [("SF", sf_main), ("Mumford0", mu_main)]:
            top = (df.assign(_m=pd.to_numeric(df[metric], errors="coerce"))
                     .dropna(subset=["_m"])
                     .nlargest(10, "_m")
                     [["case_k", "case_edges", "_m"]])
            lines.append(f"### {net_name}")
            lines.append("")
            lines.append("| k | edges | {0} |".format(metric))
            lines.append("|---|---|---:|")
            for _, row in top.iterrows():
                edges = str(row["case_edges"])[:40]
                lines.append(f"| {int(row['case_k'])} | `{edges}` | {_format_eur(row['_m'])} |")
            lines.append("")

    # --- 4) Sensitivity sweeps: per-parameter summary table ---
    sens_slugs = [(slug, info) for slug, info in SWEEPS.items() if info[2] is not None]
    if sens_slugs:
        lines.append(f"## Sensitivity sweeps — median {metric}")
        lines.append("")
        lines.append("| sweep | parameter | network | values | median range |")
        lines.append("|---|---|---|---|---|")
        for slug, (net, label, param) in sens_slugs:
            df = by_slug.get(slug)
            if df is None or param not in df.columns:
                continue
            xs = pd.to_numeric(df[param], errors="coerce")
            ys = pd.to_numeric(df[metric], errors="coerce")
            mask = xs.notna() & ys.notna()
            if not mask.any():
                continue
            grouped = pd.DataFrame({"x": xs[mask], "y": ys[mask]}).groupby("x")["y"].median()
            vmin, vmax = float(grouped.min()), float(grouped.max())
            nvals = grouped.size
            lines.append(
                f"| {slug} | {param} | {net} | {nvals} pts | "
                f"{_format_eur(vmin)} … {_format_eur(vmax)} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("See `main_map_vss.png` and `sensitivity.png` in this directory.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default="Results")
    ap.add_argument("--suffix", default="_redo",
                    help="vss_<slug><suffix>/ subdir (default '_redo', falls back to '' if missing)")
    ap.add_argument("--out-dir", default="Analysis/network_comparison")
    ap.add_argument("--use-evpi", action="store_true",
                    help="Use EVPI instead of VSS_nom (only safe with deterministic-WS data)")
    ns = ap.parse_args()

    by_slug: Dict[str, pd.DataFrame] = {}
    for slug in SWEEPS:
        df = _load_sweep(ns.results_root, slug, ns.suffix)
        if df is not None:
            by_slug[slug] = df
            print(f"  loaded {slug}: {len(df)} rows from {df['_source_path'].iloc[0]}")
        else:
            print(f"  [MISSING] {slug}")

    if not by_slug:
        print("[ERR] no sweep CSVs loaded — check --results-root and --suffix", file=sys.stderr)
        sys.exit(1)

    os.makedirs(ns.out_dir, exist_ok=True)
    _write_summary(by_slug, os.path.join(ns.out_dir, "summary.md"), ns.use_evpi)
    _make_main_map_boxplot(by_slug, os.path.join(ns.out_dir, "main_map_vss.png"), ns.use_evpi)
    _make_sensitivity_plot(by_slug, os.path.join(ns.out_dir, "sensitivity.png"), ns.use_evpi)
    print()
    print(f"All artefacts in {ns.out_dir}/")


if __name__ == "__main__":
    main()
