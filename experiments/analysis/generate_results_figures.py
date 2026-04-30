"""generate_results_figures.py
====================================================================
Regenerate every figure used in Section 5 (Results and Analysis) of
the IC-FS / ESWA manuscript.

Supports OULAD and both UCI datasets (Math, Portuguese), with multi-budget
sweeps over k ∈ {5, 7, 10, 15} for UCI.

INPUT — OULAD (unchanged from prior version):
    results/oulad/k15/stat8_oulad_h{0,1,2}_k15.csv
    results/oulad/k15/dre_multi_oulad_h{0,1,2}_k15.csv
    results/oulad/k15/oulad_icfs_h{0,1,2}_k15.csv
    results/oulad/baselines_oulad_h{0,1,2}.csv

INPUT — UCI (revised path scheme; **bug fix**: removed duplicated dataset
name in the search path):
    results/uci/{dataset}/k{k}/stat8_uci_{dataset}_h{0,1,2}_k{k}.csv
    results/uci/{dataset}/k{k}/uci_{dataset}_icfs_h{0,1,2}_k{k}.csv
    results/uci/{dataset}/baselines_uci_{dataset}_h{0,1,2}.csv
        with dataset ∈ {math, portuguese} and k ∈ {5, 7, 10, 15}.

OUTPUT — manuscript/figures/results/{dataset}/...
    fig_R1_method_vs_budget[_{dataset}].{pdf,png}
                                            ONE figure per dataset: IC-FS
                                            variants as lines across k, baselines
                                            as horizontal reference lines at their
                                            own n_features (value annotated).
    fig_R2_dre_leakage.{pdf,png}            OULAD only
    fig_R3_alpha_sweep_k{k}.{pdf,png}       one per k for UCI
    fig_R4_stability_perf_k{k}.{pdf,png}    one per k for UCI
    fig_R5_budget_sweep.{pdf,png}           IUS_deploy vs k (IC-FS variants only)
    fig_R6_cross_dataset_summary.{pdf,png}  OULAD vs UCI overlay
    fig_R7_alpha_search_value.{pdf,png}     full minus HardFilter+DE-FS

Typography: Times-style serif font, Wong colour-blind safe palette.

Usage:
    # OULAD figures
    python experiments/analysis/generate_results_figures.py

    # All UCI figures across all four budgets
    python experiments/analysis/generate_results_figures.py --dataset uci_math --all-k
    python experiments/analysis/generate_results_figures.py --dataset uci_portuguese --all-k

    # Cross-dataset comparison figure
    python experiments/analysis/generate_results_figures.py --cross-dataset

    # Single budget for UCI
    python experiments/analysis/generate_results_figures.py --dataset uci_math --k 10
====================================================================
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------------------------
# Style
# --------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          8.5,
    "axes.titlesize":     9.0,
    "axes.labelsize":     8.5,
    "legend.fontsize":    7.5,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "axes.linewidth":     0.7,
    "axes.edgecolor":     "#333333",
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.4,
    "lines.markersize":   4.5,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
})

# Wong colour-blind safe palette
C_BLUE    = "#0072B2"
C_ORANGE  = "#D55E00"
C_GREEN   = "#009E73"
C_PINK    = "#CC79A7"
C_YELLOW  = "#E6AC00"
C_SKY     = "#56B4E9"
C_GREY    = "#666666"
C_BLACK   = "#222222"

# --------------------------------------------------------------------
# Paths and Configuration
# --------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()

HORIZONS   = [0, 1, 2]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]

OULAD_K    = 15
UCI_K_GRID = [5, 7, 10, 15]

_ICFS = [
    ("IUS_deploy_full",     "IC-FS (full)",           C_BLUE,   "-",  "o"),
    ("IUS_deploy_noTemp",   "IC-FS (−temporal)",       C_ORANGE, "--", "s"),
    ("IUS_deploy_noAction", "IC-FS (−actionability)",  C_PINK,   ":",  "^"),
    ("IUS_deploy_hardDEFS", "HardFilter+DE-FS",        C_GREEN,  "-.", "D"),
]
 
# Baselines: (method name in CSV, label, colour, marker)
_BASELINES = [
    ("NSGA-II-MOFS",       "NSGA-II",    C_SKY,    "P"),
    ("StabilitySelection", "Stab. Sel.", C_YELLOW,  "X"),
    ("Boruta",             "Boruta",     C_GREY,    "*"),
]
 
_HORIZONS   = [0, 1, 2]
_UCI_K_GRID = [5, 7, 10, 15]

def setup_paths(dataset: str = "oulad"):
    """
    Resolve results directory and output directory for a given dataset.

    Bug fix (vs prior version): UCI path no longer duplicates the dataset
    name. Previously: results/uci/math/math/  →  Now: results/uci/math/
    """
    if dataset == "oulad":
        results_dir = ROOT / "results" / "oulad"
        outdir      = ROOT / "manuscript" / "figures" / "results" / "oulad"
    elif dataset.startswith("uci_"):
        dataset_name = dataset.split("_", 1)[1]   # "math" or "portuguese"
        results_dir = ROOT / "results" / "uci" / dataset_name
        outdir      = ROOT / "manuscript" / "figures" / "results" / dataset
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. "
                         f"Use 'oulad', 'uci_math', or 'uci_portuguese'.")

    outdir.mkdir(parents=True, exist_ok=True)
    return results_dir, outdir


def _dataset_short(dataset: str) -> str:
    """Map dataset key to the short token used inside CSV filenames."""
    if dataset == "oulad":            return "oulad"
    if dataset == "uci_math":         return "math"
    if dataset == "uci_portuguese":   return "portuguese"
    raise ValueError(dataset)

def _short(dataset: str) -> str:
    return {"oulad": "oulad",
            "uci_math": "math",
            "uci_portuguese": "portuguese"}[dataset]
 
 
def _pretty(dataset: str) -> str:
    return {"oulad": "OULAD",
            "uci_math": "UCI Math",
            "uci_portuguese": "UCI Portuguese"}[dataset]
 
 
def _load_stat8(dataset, results_dir, k, h):
    short = _short(dataset)
    p = (results_dir / f"k{k}" / f"stat8_oulad_h{h}_k{k}.csv"
         if dataset == "oulad"
         else results_dir / f"k{k}" / f"stat8_uci_{short}_h{h}_k{k}.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def _load_baselines(dataset, results_dir, h):
    short = _short(dataset)
    p = (results_dir / f"baselines_oulad_h{h}.csv"
         if dataset == "oulad"
         else results_dir / f"baselines_uci_{short}_h{h}.csv")
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

# ── Core figure ──────────────────────────────────────────────────────────────
 
def _fig_method_vs_budget(dataset, results_dir, outdir, k_list):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(10.8, 3.5),
        sharey=False,
        gridspec_kw=dict(wspace=0.42),
    )
 
    x_min = min(k_list) - 1.5
    x_max = max(k_list) + 1.5
 
    for ax, h in zip(axes, _HORIZONS):
 
        # ── 1. IC-FS variant lines across k ───────────────────────────
        for col, label, colour, ls, mk in _ICFS:
            ks, means, stds = [], [], []
            for k in k_list:
                df = _load_stat8(dataset, results_dir, k, h)
                if df.empty or col not in df.columns:
                    continue
                v = df[col].dropna().to_numpy()
                if len(v) == 0:
                    continue
                ks.append(k)
                means.append(v.mean())
                stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
 
            if not ks:
                continue
 
            ks    = np.array(ks,    dtype=float)
            means = np.array(means, dtype=float)
            stds  = np.array(stds,  dtype=float)
 
            ax.plot(ks, means,
                    color=colour, linestyle=ls, marker=mk,
                    linewidth=1.7, markersize=5.5,
                    label=label, zorder=4)
            ax.fill_between(ks, means - stds, means + stds,
                            color=colour, alpha=0.11,
                            linewidth=0, zorder=3)
 
        # ── 2. Baselines: horizontal reference lines ───────────────────
        df_base = _load_baselines(dataset, results_dir, h)
 
        if not df_base.empty and "method" in df_base.columns:
            for method_name, label, colour, mk in _BASELINES:
                row = df_base[df_base["method"] == method_name]
                if row.empty:
                    continue
 
                k_actual   = float(row["n_features"].iloc[0])
                ius_actual = float(row["IUS_deploy"].iloc[0])
 
                # Horizontal dashed line spanning full x-range
                ax.axhline(
                    y=ius_actual,
                    color=colour,
                    linestyle=(0, (4, 3)),   # custom dash: 4pt on, 3pt off
                    linewidth=1.3,
                    alpha=0.85,
                    zorder=2,
                    label=label,
                )
 
                # Filled marker at the baseline's actual k
                ax.scatter(
                    [k_actual], [ius_actual],
                    color=colour, marker=mk,
                    s=90, zorder=6,
                    edgecolors="white", linewidths=0.8,
                )
 
                # IUS value annotated at the right edge of the line
                # Format: "62.5  (k=7)"  — two pieces of info in one label
                ax.text(
                    x_max - 0.05,              # just inside right border
                    ius_actual,
                    f"{ius_actual:.1f}  (k={int(k_actual)})",
                    color=colour,
                    fontsize=6.5,
                    va="center",
                    ha="right",
                    zorder=7,
                    # white halo so text is legible over grid lines
                    bbox=dict(boxstyle="round,pad=0.1",
                              fc="white", ec="none", alpha=0.7),
                )
 
        # ── 3. Cosmetics ───────────────────────────────────────────────
        ax.set_title(f"$h = {h}$", fontsize=9)
        ax.set_xlabel("Feature budget $k$", fontsize=8.5)
        if h == 0:
            ax.set_ylabel(r"$\mathrm{IUS}_{\mathrm{deploy}}$", fontsize=8.5)
 
        ax.set_xticks(k_list)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.grid(True, linewidth=0.4, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)
 
    # ── 4. Shared legend (outside right panel) ─────────────────────────
    icfs_handles = [
        Line2D([0], [0], color=col, linestyle=ls, marker=mk,
               linewidth=1.5, markersize=5.5, label=lbl)
        for _, lbl, col, ls, mk in _ICFS
    ]
 
    # Separator between two groups
    sep = Line2D([0], [0], color="none", label="— baselines (fixed own k) —")
 
    baseline_handles = [
        Line2D([0], [0],
               color=col,
               linestyle=(0, (4, 3)),
               linewidth=1.3,
               marker=mk,
               markersize=6.5,
               markeredgecolor="white",
               markeredgewidth=0.8,
               label=lbl)
        for _, lbl, col, mk in _BASELINES
    ]
 
    # Explain the right-edge annotation
    note = Line2D([0], [0], color="none",
                  label="(value & k shown at right edge →)")
 
    axes[-1].legend(
        handles=icfs_handles + [sep] + baseline_handles + [note],
        loc="upper left",
        bbox_to_anchor=(1.03, 1.0),
        frameon=False,
        fontsize=7.5,
        handlelength=2.8,
        labelspacing=0.55,
    )
 
    fig.suptitle(
        f"{_pretty(dataset)} — "
        r"$\mathrm{IUS}_{\mathrm{deploy}}$ vs. feature budget $k$  "
        r"(IC-FS: lines ± 1σ;   baselines: dashed reference)",
        fontsize=9, y=1.03,
    )
 
    # ── 5. Save ────────────────────────────────────────────────────────
    short = _short(dataset)
    stem  = ("fig_R1_method_vs_budget" if dataset == "oulad"
             else f"fig_R1_method_vs_budget_{short}")
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  [fig_R1] Saved → {outdir / stem}.pdf / .png")
 
 
# --------------------------------------------------------------------
# Data loaders — dataset- and budget-aware
# --------------------------------------------------------------------
def load_stat8(dataset: str, results_dir: Path, k: int, h: int) -> pd.DataFrame:
    """
    Eight-seed paired comparisons of IC-FS(full), --temporal,
    --actionability, hardDEFS for a given (dataset, h, k) cell.
    """
    short = _dataset_short(dataset)
    if dataset == "oulad":
        path = results_dir / f"k{k}" / f"stat8_oulad_h{h}_k{k}.csv"
    else:
        path = results_dir / f"k{k}" / f"stat8_uci_{short}_h{h}_k{k}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_alpha_sweep(dataset: str, results_dir: Path, k: int, h: int) -> pd.DataFrame:
    """Five-row α-sweep (one row per α) for a (dataset, h, k) cell."""
    short = _dataset_short(dataset)
    if dataset == "oulad":
        path = results_dir / f"k{k}" / f"oulad_icfs_h{h}_k{k}.csv"
    else:
        path = results_dir / f"k{k}" / f"uci_{short}_icfs_h{h}_k{k}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_dre(dataset: str, results_dir: Path, k: int, h: int) -> pd.DataFrame:
    """Eight-seed DRE diagnostics — currently OULAD only."""
    if dataset == "oulad":
        path = results_dir / f"k{k}" / f"dre_multi_oulad_h{h}_k{k}.csv"
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def load_baselines(dataset: str, results_dir: Path, h: int) -> pd.DataFrame:
    """One-seed external baselines (NSGA-II, StabSel, Boruta)."""
    short = _dataset_short(dataset)
    if dataset == "oulad":
        path = results_dir / f"baselines_oulad_h{h}.csv"
    else:
        path = results_dir / f"baselines_uci_{short}_h{h}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# --------------------------------------------------------------------
# FIGURE R1 — IUS_deploy across method × horizon at one budget k
# --------------------------------------------------------------------
def fig_method_horizon_comparison(
    dataset: str,
    results_dir: Path,
    outdir: Path,
    k: int,          # kept for API compatibility; ignored internally
) -> None:
    """
    Drop-in replacement for the old per-k grouped-bar-chart function.
 
    The old loop in main() called this once per k; now every call produces
    (or skips if already present) a single cross-k figure per dataset.
    """
    short  = _short(dataset)
    stem   = ("fig_R1_method_vs_budget" if dataset == "oulad"
              else f"fig_R1_method_vs_budget_{short}")
    target = outdir / f"{stem}.pdf"
 
    if target.exists():
        print(f"  [fig_R1] Already generated → {target.name}  (skipping)")
        return
 
    # Auto-detect available k values from the file system
    if dataset == "oulad":
        found = sorted(
            int(p.name[1:])
            for p in results_dir.glob("k*")
            if (p / f"stat8_oulad_h0_k{p.name[1:]}.csv").exists()
        )
        k_list = found if found else [15]
    else:
        k_list = _UCI_K_GRID
 
    _fig_method_vs_budget(dataset, results_dir, outdir, k_list)


def _pretty_dataset_name(dataset: str) -> str:
    return {"oulad": "OULAD",
            "uci_math": "UCI Student Performance — Math",
            "uci_portuguese": "UCI Student Performance — Portuguese"}[dataset]


# --------------------------------------------------------------------
# FIGURE R2 — DRE leakage diagnostic at h = 0 (OULAD only)
# --------------------------------------------------------------------
def fig_dre_leakage_diagnostic(dataset: str, results_dir: Path,
                                outdir: Path, k: int) -> None:
    """Two-panel: (a) IUS_paper vs IUS_deploy; (b) τ leakage coefficient."""
    df_stat = load_stat8(dataset, results_dir, k, 0)
    df_dre  = load_dre  (dataset, results_dir, k, 0)

    if df_stat.empty or df_dre.empty or "tau_full" not in df_dre.columns:
        print(f"  [fig_R2] Skipping DRE figure — DRE data not available "
              f"for {dataset} (this is expected for UCI; preprocessing "
              f"already enforces δ at the data-loading stage, so τ ≡ 0).")
        return

    paper_full  = df_stat["IUS_paper_full"].to_numpy()
    deploy_full = df_stat["IUS_deploy_full"].to_numpy()
    paper_temp  = df_stat["IUS_paper_noTemp"].to_numpy()
    deploy_temp = df_stat["IUS_deploy_noTemp"].to_numpy()
    tau_full    = df_dre["tau_full"].to_numpy()
    tau_temp    = df_dre["tau_notemp"].to_numpy()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                      gridspec_kw=dict(width_ratios=[1.4, 1.0]))

    x = np.arange(2)
    bar_w = 0.34
    means_paper  = [paper_full.mean(),  paper_temp.mean()]
    means_deploy = [deploy_full.mean(), deploy_temp.mean()]
    stds_paper   = [paper_full.std(ddof=1),  paper_temp.std(ddof=1)]
    stds_deploy  = [deploy_full.std(ddof=1), deploy_temp.std(ddof=1)]

    ax_a.bar(x - bar_w/2, means_paper,  width=bar_w, yerr=stds_paper,
              capsize=2.5, color=C_GREY, edgecolor="white",
              label=r"$\mathrm{IUS}_{\mathrm{paper}}$")
    ax_a.bar(x + bar_w/2, means_deploy, width=bar_w, yerr=stds_deploy,
              capsize=2.5, color=C_BLUE, edgecolor="white",
              label=r"$\mathrm{IUS}_{\mathrm{deploy}}$")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["IC-FS (full)", "IC-FS (--temporal)"])
    ax_a.set_ylabel(r"Intervention Utility Score")
    ax_a.set_title(r"(a) Conventional vs deployment-realistic eval ($h=0$)")
    ax_a.legend(loc="upper left", frameon=False)

    gap = means_paper[1] - means_deploy[1]
    ax_a.annotate(f"{gap:.0f}-pt drop\n($\\tau \\approx {tau_temp.mean():.2f}$)",
                   xy=(1 + bar_w/2, means_deploy[1]),
                   xytext=(1.05, means_paper[1] * 0.5),
                   ha="left", fontsize=7.5, color=C_ORANGE,
                   arrowprops=dict(arrowstyle="->", color=C_ORANGE,
                                   lw=0.7, shrinkA=2, shrinkB=2))

    ax_b.bar([0, 1], [tau_full.mean(), tau_temp.mean()],
              yerr=[tau_full.std(ddof=1), tau_temp.std(ddof=1)],
              capsize=2.5, color=[C_BLUE, C_ORANGE],
              edgecolor="white", width=0.55)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["IC-FS (full)", "--temporal"])
    ax_b.set_ylabel(r"Leakage coefficient $\tau$")
    ax_b.set_ylim(0, 1.0)
    ax_b.set_title(r"(b) Leakage coefficient ($h=0$)")
    ax_b.axhline(0, color="#000", lw=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_R2_dre_leakage.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------
# FIGURE R3 — α-sweep with dual axes at one budget k
# --------------------------------------------------------------------
def fig_alpha_sweep(dataset: str, results_dir: Path, outdir: Path, k: int) -> None:
    """Three-panel α-sweep with nested-best α* highlighted by a star."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True)

    for ax, h in zip(axes, HORIZONS):
        df = load_alpha_sweep(dataset, results_dir, k, h)
        if df.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color=C_GREY, fontsize=8)
            continue
        df = df.sort_values("alpha")
        alphas = df["alpha"].to_numpy()
        ius    = df["IUS_deploy"].to_numpy()
        ar     = df["AR_available"].to_numpy()
        nested = df["nested_best"].to_numpy()

        ax.plot(alphas, ius, marker="o", color=C_BLUE,
                label=r"$\mathrm{IUS}_{\mathrm{deploy}}$")
        ax.set_xlabel(r"Trade-off parameter $\alpha$")
        ax.set_xticks(ALPHA_GRID)
        if h == 0:
            ax.set_ylabel(r"$\mathrm{IUS}_{\mathrm{deploy}}$", color=C_BLUE)
        ax.tick_params(axis="y", labelcolor=C_BLUE)

        ax_r = ax.twinx()
        ax_r.plot(alphas, ar, marker="s", color=C_ORANGE, linestyle="--",
                  label=r"$\mathrm{AR}_{\mathrm{available}}$")
        ax_r.set_ylim(0, 1.05)
        if h == 2:
            ax_r.set_ylabel(r"$\mathrm{AR}_{\mathrm{available}}$",
                            color=C_ORANGE)
        ax_r.tick_params(axis="y", labelcolor=C_ORANGE)
        ax_r.grid(False)

        if nested.any():
            best_idx = int(np.where(nested)[0][0])
            ax.scatter([alphas[best_idx]], [ius[best_idx]],
                       s=110, marker="*", color=C_GREEN, zorder=5,
                       edgecolor="black", linewidth=0.5,
                       label=r"$\alpha^\ast$ (nested)")

        ax.set_title(f"$h = {h}$")
        ax.set_xlim(-0.05, 1.05)

        if h == 0:
            ax.legend(loc="lower left", frameon=False, fontsize=7)

    fig.suptitle(f"{_pretty_dataset_name(dataset)} — feature budget $k = {k}$",
                  fontsize=9, y=1.02)
    fig.tight_layout()
    suffix = f"_k{k}" if dataset != "oulad" else ""
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_R3_alpha_sweep{suffix}.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------
# FIGURE R4 — Stability vs IUS_deploy
# --------------------------------------------------------------------
def fig_stability_performance(dataset: str, results_dir: Path,
                               outdir: Path, k: int) -> None:
    """Scatter of (Jaccard stability, IUS_deploy) for every (α, h)."""
    fig, ax = plt.subplots(figsize=(3.6, 3.0))

    horizon_colours = {0: C_ORANGE, 1: C_BLUE, 2: C_GREEN}
    horizon_marker  = {0: "o",       1: "s",     2: "^"}

    for h in HORIZONS:
        df = load_alpha_sweep(dataset, results_dir, k, h)
        if df.empty:
            continue
        for _, row in df.iterrows():
            star = bool(row["nested_best"])
            ax.scatter(row["stability"], row["IUS_deploy"],
                       marker=horizon_marker[h],
                       s=80 if star else 36,
                       color=horizon_colours[h],
                       edgecolor="black" if star else "none",
                       linewidth=0.7 if star else 0,
                       alpha=0.9 if star else 0.55,
                       zorder=4 if star else 3)

    handles = [plt.Line2D([0], [0], marker=horizon_marker[h], linestyle="",
                          color=horizon_colours[h], label=f"$h={h}$",
                          markersize=6) for h in HORIZONS]
    handles.append(plt.Line2D([0], [0], marker="*", linestyle="",
                               color="black", markersize=8,
                               markerfacecolor="white",
                               label=r"$\alpha^\ast$ (nested)"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7)
    ax.set_xlabel(r"Bootstrap Jaccard stability $J$")
    ax.set_ylabel(r"$\mathrm{IUS}_{\mathrm{deploy}}$")
    ax.set_xlim(0.20, 1.05)
    ax.set_ylim(0, 90)
    ax.set_title(f"{_pretty_dataset_name(dataset)}, $k = {k}$",
                  fontsize=8.5)

    fig.tight_layout()
    suffix = f"_k{k}" if dataset != "oulad" else ""
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_R4_stability_perf{suffix}.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------
# NEW FIGURE R5 — Budget sweep: IUS_deploy versus k for each horizon
# --------------------------------------------------------------------
def fig_budget_sweep(dataset: str, results_dir: Path, outdir: Path,
                      k_list: List[int]) -> None:
    """
    For UCI: line plot of IC-FS(full), --action, HardFilter+DE-FS
    versus feature budget k, one panel per horizon.

    This figure motivates the §5 finding that the value of α-search
    over hard filtering grows with k.
    """
    if dataset == "oulad":
        # OULAD is run at k=15 only in the current pipeline; skip.
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), sharey=True)

    for ax, h in zip(axes, HORIZONS):
        ks_full,    ms_full,    ss_full    = [], [], []
        ks_action,  ms_action,  ss_action  = [], [], []
        ks_hard,    ms_hard,    ss_hard    = [], [], []

        for k in k_list:
            df = load_stat8(dataset, results_dir, k, h)
            if df.empty:
                continue
            for col, ks, ms, ss in [
                ("IUS_deploy_full",     ks_full,   ms_full,   ss_full),
                ("IUS_deploy_noAction", ks_action, ms_action, ss_action),
                ("IUS_deploy_hardDEFS", ks_hard,   ms_hard,   ss_hard),
            ]:
                if col in df.columns:
                    v = df[col].to_numpy()
                    ks.append(k); ms.append(v.mean()); ss.append(v.std(ddof=1))

        if ks_full:
            ax.errorbar(ks_full, ms_full, yerr=ss_full, marker="o",
                         color=C_BLUE, label="IC-FS (full)",
                         capsize=2.0, linewidth=1.4)
        if ks_hard:
            ax.errorbar(ks_hard, ms_hard, yerr=ss_hard, marker="s",
                         color=C_GREEN, linestyle="--",
                         label="HardFilter+DE-FS",
                         capsize=2.0, linewidth=1.4)
        if ks_action:
            ax.errorbar(ks_action, ms_action, yerr=ss_action, marker="^",
                         color=C_PINK, linestyle=":",
                         label="--actionability",
                         capsize=2.0, linewidth=1.4)

        ax.set_title(f"$h = {h}$")
        ax.set_xlabel("Feature budget $k$")
        ax.set_xticks(k_list)
        ax.set_ylim(0, 90)
        if h == 0:
            ax.set_ylabel(r"$\mathrm{IUS}_{\mathrm{deploy}}$")
            ax.legend(loc="lower right", frameon=False, fontsize=7)

    fig.suptitle(f"{_pretty_dataset_name(dataset)} — IC-FS, ablation, and "
                  f"hard-filter baseline across feature budget",
                  fontsize=9, y=1.03)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_R5_budget_sweep.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------
# NEW FIGURE R6 — Cross-dataset summary (OULAD vs UCI Math vs UCI Por)
# --------------------------------------------------------------------
def fig_cross_dataset_summary(out_root: Path) -> None:
    """
    Headline cross-dataset comparison: IC-FS(full) at the
    operationally-recommended budget per dataset, alongside three
    external baselines, separately for each horizon.

    Recommended budgets:
        OULAD k=15 (all horizons)
        UCI Math k=5 at h=0, k=10 at h∈{1,2}
        UCI Portuguese k=10 at all horizons
    """
    panels = [
        ("OULAD",            "oulad",          {0: 15, 1: 15, 2: 15}),
        ("UCI Math",         "uci_math",       {0: 5,  1: 10, 2: 10}),
        ("UCI Portuguese",   "uci_portuguese", {0: 10, 1: 10, 2: 10}),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.2), sharey=False)

    bar_w = 0.18
    methods = [
        ("IC-FS\n(full)",     "IUS_deploy_full",     "stat8",    C_BLUE),
        ("HardFilt.\n+DE-FS", "IUS_deploy_hardDEFS", "stat8",    C_GREEN),
        ("NSGA-II",           "IUS_deploy",          "baseline", C_SKY),
        ("Boruta",            "IUS_deploy",          "baseline", C_YELLOW),
        ("Stab.\nSel.",       "IUS_deploy",          "baseline", C_GREY),
    ]
    baseline_method_names = {"NSGA-II":  "NSGA-II-MOFS",
                              "Boruta":   "Boruta",
                              "Stab.\nSel.": "StabilitySelection"}

    for ax, (title, dataset, k_per_h) in zip(axes, panels):
        results_dir, _ = setup_paths(dataset)
        x = np.arange(len(HORIZONS))

        for i, (label, col, src, colour) in enumerate(methods):
            means, stds = [], []
            for h in HORIZONS:
                k = k_per_h[h]
                if src == "stat8":
                    df = load_stat8(dataset, results_dir, k, h)
                    if not df.empty and col in df.columns:
                        v = df[col].to_numpy()
                        means.append(v.mean()); stds.append(v.std(ddof=1))
                    else:
                        means.append(np.nan); stds.append(0.0)
                else:
                    df = load_baselines(dataset, results_dir, h)
                    if not df.empty and "method" in df.columns:
                        row = df[df["method"] == baseline_method_names[label]]
                        means.append(row[col].iloc[0] if len(row) else np.nan)
                        stds.append(0.0)
                    else:
                        means.append(np.nan); stds.append(0.0)
            offset = (i - (len(methods)-1)/2) * bar_w
            ax.bar(x + offset, means, width=bar_w, yerr=stds,
                    capsize=1.8, error_kw=dict(elinewidth=0.5),
                    color=colour, edgecolor="white", linewidth=0.4,
                    label=label if ax is axes[-1] else None)

        ax.set_xticks(x)
        ax.set_xticklabels([f"$h={h}$" for h in HORIZONS])
        ax.set_xlabel("Prediction horizon")
        ax.set_title(title)
        ax.set_ylim(0, 90)
        if ax is axes[0]:
            ax.set_ylabel(r"$\mathrm{IUS}_{\mathrm{deploy}}$")

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     frameon=False, fontsize=7.5)

    fig.suptitle("Cross-dataset comparison at operationally-recommended budgets",
                  fontsize=9.5, y=1.02)
    fig.tight_layout()

    out = out_root / "manuscript" / "figures" / "results" / "cross_dataset"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_R6_cross_dataset_summary.{ext}")
    plt.close(fig)
    print(f"  [fig_R6] Saved to {out}")


# --------------------------------------------------------------------
# NEW FIGURE R7 — Value of α-search over HardFilter+DE-FS as k varies
# --------------------------------------------------------------------
def fig_alpha_search_value(out_root: Path) -> None:
    """
    The headline new finding: Δ(IC-FS full minus HardFilter+DE-FS)
    across budget k, for each horizon and dataset.

    Demonstrates that α-search adds value precisely when budget exceeds
    Tier-1 supply.
    """
    panels = [
        ("UCI Math",         "uci_math",       UCI_K_GRID),
        ("UCI Portuguese",   "uci_portuguese", UCI_K_GRID),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)

    for ax, (title, dataset, k_list) in zip(axes, panels):
        results_dir, _ = setup_paths(dataset)
        for h, colour in zip(HORIZONS, [C_ORANGE, C_BLUE, C_GREEN]):
            ks, deltas, errs = [], [], []
            for k in k_list:
                df = load_stat8(dataset, results_dir, k, h)
                if df.empty: continue
                a = df["IUS_deploy_full"].to_numpy()
                b = df["IUS_deploy_hardDEFS"].to_numpy()
                d = a - b
                ks.append(k)
                deltas.append(d.mean())
                errs.append(d.std(ddof=1))
            ax.errorbar(ks, deltas, yerr=errs, marker="o", color=colour,
                         label=f"$h = {h}$", capsize=2.0, linewidth=1.4)
        ax.axhline(0, color="black", lw=0.5, linestyle="--")
        ax.set_xlabel("Feature budget $k$")
        ax.set_xticks(UCI_K_GRID)
        ax.set_title(title)
        if ax is axes[0]:
            ax.set_ylabel(r"$\Delta\,\mathrm{IUS}_{\mathrm{deploy}}$  "
                           r"(IC-FS full $-$ HardFilter+DE-FS)")
            ax.legend(loc="upper left", frameon=False, fontsize=7.5)
        ax.set_ylim(-5, 30)

    fig.suptitle("Value of $\\alpha$-search over hard filtering across "
                  "feature budget", fontsize=9.5, y=1.02)
    fig.tight_layout()

    out = out_root / "manuscript" / "figures" / "results" / "cross_dataset"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_R7_alpha_search_value.{ext}")
    plt.close(fig)
    print(f"  [fig_R7] Saved to {out}")


# --------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate IC-FS results figures")
    parser.add_argument("--dataset", type=str, default="oulad",
                          help="Dataset: 'oulad', 'uci_math', or "
                               "'uci_portuguese' (default: oulad)")
    parser.add_argument("--k", type=int, default=None,
                          help="Single feature budget (UCI only). "
                               "If unspecified and --all-k is not set, uses 5.")
    parser.add_argument("--all-k", action="store_true",
                          help="UCI only — generate figures for k ∈ {5,7,10,15}.")
    parser.add_argument("--cross-dataset", action="store_true",
                          help="Generate fig_R6 and fig_R7 cross-dataset overlays.")
    args = parser.parse_args()

    if args.cross_dataset:
        print("[results-figures] Generating cross-dataset summary figures...")
        fig_cross_dataset_summary(ROOT)
        fig_alpha_search_value(ROOT)
        print("[results-figures] Done.")
        return

    dataset = args.dataset.lower()
    if dataset not in ("oulad", "uci_math", "uci_portuguese"):
        raise ValueError(f"Unknown dataset: {dataset!r}")

    results_dir, outdir = setup_paths(dataset)
    print(f"[results-figures] Dataset: {dataset}")
    print(f"[results-figures] Reading data from {results_dir}")
    print(f"[results-figures] Writing figures to {outdir}")

    # Determine which budgets to render
    if dataset == "oulad":
        k_list = [OULAD_K]
    elif args.all_k:
        k_list = UCI_K_GRID
    else:
        k_list = [args.k if args.k is not None else 5]

    for k in k_list:
        print(f"\n[results-figures] Rendering at k = {k}")
        fig_method_horizon_comparison(dataset, results_dir, outdir, k)
        fig_alpha_sweep              (dataset, results_dir, outdir, k)
        fig_stability_performance    (dataset, results_dir, outdir, k)
        if dataset == "oulad":
            fig_dre_leakage_diagnostic(dataset, results_dir, outdir, k)

    # UCI-specific budget-sweep figure
    if dataset != "oulad":
        print(f"\n[results-figures] Rendering budget sweep across "
              f"k ∈ {UCI_K_GRID}")
        fig_budget_sweep(dataset, results_dir, outdir, UCI_K_GRID)

    print("\n[results-figures] Done. Generated:")
    for f in sorted(outdir.glob("fig_R*.pdf")):
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()