"""
experiments/analysis/generate_figures_eswa.py
==============================================
Generate publication-ready figures (Figs. 1–4) for the IC-FS ESWA submission.

Reads experimental results from the verified CSV files in:
    results/oulad/k{5,7,10,15}/stat8_oulad_h{0,1,2}_k{*}.csv
    results/oulad/k{5,7,10,15}/dre_multi_oulad_h{0,1,2}_k{*}.csv
    results/oulad/baselines_oulad_h{0,1,2}.csv

Outputs (saved to manuscript/figures/):
    fig1_dre_leakage.pdf   — DRE leakage exposure (grouped bar, t=0, k=10)
    fig2_ablation.pdf      — Ablation IUS_deploy vs k (3-panel, h∈{0,1,2})
    fig3_main_vs_baselines.pdf — IC-FS(full) vs baselines across horizons
    fig4_budget_metrics.pdf    — Precision@20% and Recall@20%

Usage:
    cd <project_root>
    python experiments/analysis/generate_figures_eswa.py

Requirements: matplotlib>=3.7, numpy, pandas, scipy
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RES_OULAD = PROJECT_ROOT / "results" / "oulad"
OUT_DIR = PROJECT_ROOT / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── IEEE/ESWA typography constants ───────────────────────────────────────────
FONT_FAMILY = "Times New Roman"
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9
FONT_SIZE_CAPTION = 8

LINE_WIDTH = 1.5
MARKER_SIZE = 6
GRID_COLOR = "#d8d8d8"
GRID_LW = 0.6

SERIES = [
    {"color": "#0072B2", "linestyle": "-",  "marker": "o", "label": "IC-FS (full)"},
    {"color": "#D55E00", "linestyle": "--", "marker": "s", "label": "IC-FS (−temporal)"},
    {"color": "#009E73", "linestyle": ":",  "marker": "^", "label": "IC-FS (−actionability)"},
    {"color": "#CC79A7", "linestyle": "-.", "marker": "D", "label": "HardFilter+DE-FS"},
]

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE_TICK,
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": GRID_LW,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Data loading helpers ──────────────────────────────────────────────────────

def load_stat(h: int, k: int) -> pd.DataFrame:
    path = RES_OULAD / f"k{k}" / f"stat8_oulad_h{h}_k{k}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def load_dre(h: int, k: int) -> pd.DataFrame:
    path = RES_OULAD / f"k{k}" / f"dre_multi_oulad_h{h}_k{k}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def load_baselines(h: int) -> pd.DataFrame:
    path = RES_OULAD / f"baselines_oulad_h{h}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def ms(arr):
    """Return (mean, std) for an array."""
    a = np.asarray(arr, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


# ── Collect all data ──────────────────────────────────────────────────────────

K_VALUES = [5, 7, 10, 15]
HORIZONS = [0, 1, 2]

# stat_data[h][k] = {full: [...], notemp: [...], noact: [...], hard: [...]}
stat_data: dict = {}
for h in HORIZONS:
    stat_data[h] = {}
    for k in K_VALUES:
        try:
            df = load_stat(h, k)
            stat_data[h][k] = {
                "full":   df["IUS_deploy_full"].tolist(),
                "notemp": df["IUS_deploy_noTemp"].tolist(),
                "noact":  df["IUS_deploy_noAction"].tolist(),
                "hard":   df["IUS_deploy_hardDEFS"].tolist(),
            }
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            stat_data[h][k] = None

# dre_data[h][k] = DataFrame
dre_data: dict = {}
for h in HORIZONS:
    dre_data[h] = {}
    for k in K_VALUES:
        try:
            dre_data[h][k] = load_dre(h, k)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
            dre_data[h][k] = None

# baselines[h] = DataFrame
baselines: dict = {}
for h in HORIZONS:
    try:
        baselines[h] = load_baselines(h)
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        baselines[h] = None


# ── Helper: add significance bars ─────────────────────────────────────────────

def sig_label(p: float, bonf: float = 0.017) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < bonf:
        return "*"
    return "ns"


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — DRE Leakage Exposure (t=0, k=10)
# ═════════════════════════════════════════════════════════════════════════════

def make_fig1():
    """
    Two-panel figure: F1 scores (paper vs deploy) and IUS (paper vs deploy)
    for IC-FS(full) and IC-FS(-temporal) at t=0, k=10.
    """
    h, k = 0, 10
    dre = dre_data[h].get(k)
    if dre is None:
        print("  Fig 1: DRE data missing for h=0, k=10. Skipping.")
        return

    # Compute per-seed values
    f1_full_paper   = dre["f1_full_paper"].values
    f1_full_deploy  = dre["f1_full_deploy"].values
    f1_nt_paper     = dre["f1_notemp_paper"].values
    f1_nt_deploy    = dre["f1_notemp_deploy"].values

    ius_full_paper  = dre["IUS_paper_full"].values
    ius_full_deploy = dre["IUS_deploy_full"].values
    ius_nt_paper    = dre["IUS_paper_notemp"].values
    ius_nt_deploy   = dre["IUS_deploy_notemp"].values

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    fig.subplots_adjust(wspace=0.38, left=0.10, right=0.97, top=0.88, bottom=0.15)

    bar_w = 0.32
    x = np.array([0.0, 1.0])
    colors_paper  = ["#4C72B0", "#DD8452"]
    colors_deploy = ["#55A868", "#C44E52"]
    hatch_paper   = ["", ""]
    hatch_deploy  = ["///", "///"]

    # ── Panel (a): F1 ─────────────────────────────────────────────────────────
    ax = axes[0]
    means_paper  = [f1_full_paper.mean(),  f1_nt_paper.mean()]
    stds_paper   = [f1_full_paper.std(ddof=1), f1_nt_paper.std(ddof=1)]
    means_deploy = [f1_full_deploy.mean(), f1_nt_deploy.mean()]
    stds_deploy  = [f1_full_deploy.std(ddof=1), f1_nt_deploy.std(ddof=1)]

    b1 = ax.bar(x - bar_w/2, means_paper,  bar_w,
                color=colors_paper,  hatch=hatch_paper,
                edgecolor="#000", linewidth=0.8,
                yerr=stds_paper,  capsize=4, error_kw={"linewidth":1},
                label="Paper-style")
    b2 = ax.bar(x + bar_w/2, means_deploy, bar_w,
                color=colors_deploy, hatch=hatch_deploy,
                edgecolor="#000", linewidth=0.8,
                yerr=stds_deploy, capsize=4, error_kw={"linewidth":1},
                label="DRE (deploy)")

    # Annotate the collapse arrow
    ax.annotate("", xy=(x[1] + bar_w/2, means_deploy[1] + 2),
                 xytext=(x[1] - bar_w/2, means_paper[1] - 2),
                 arrowprops=dict(arrowstyle="-|>", color="#cc0000",
                                 lw=1.5, mutation_scale=10))
    ax.text(x[1] + 0.18, (means_paper[1] + means_deploy[1]) / 2,
            f"−{means_paper[1]-means_deploy[1]:.1f} pp",
            fontsize=7.5, color="#cc0000", fontweight="bold", va="center")

    ax.set_xticks(x)
    ax.set_xticklabels(["IC-FS\n(full)", "IC-FS\n(−temporal)"], fontsize=9)
    ax.set_ylabel("F1 score (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 90)
    ax.set_title("(a) F1 score comparison", fontsize=FONT_SIZE_TITLE, loc="left")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # ── Panel (b): IUS ─────────────────────────────────────────────────────────
    ax = axes[1]
    means_paper_ius  = [ius_full_paper.mean(),  ius_nt_paper.mean()]
    stds_paper_ius   = [ius_full_paper.std(ddof=1), ius_nt_paper.std(ddof=1)]
    means_deploy_ius = [ius_full_deploy.mean(), ius_nt_deploy.mean()]
    stds_deploy_ius  = [ius_full_deploy.std(ddof=1), ius_nt_deploy.std(ddof=1)]

    ax.bar(x - bar_w/2, means_paper_ius,  bar_w,
           color=colors_paper, hatch=hatch_paper,
           edgecolor="#000", linewidth=0.8,
           yerr=stds_paper_ius, capsize=4, error_kw={"linewidth":1},
           label="IUS_paper (inflated)")
    ax.bar(x + bar_w/2, means_deploy_ius, bar_w,
           color=colors_deploy, hatch=hatch_deploy,
           edgecolor="#000", linewidth=0.8,
           yerr=stds_deploy_ius, capsize=4, error_kw={"linewidth":1},
           label="IUS_deploy (honest)")

    # Annotate IUS gap for noTemp
    gap = means_paper_ius[1] - means_deploy_ius[1]
    ax.annotate("", xy=(x[1] + bar_w/2, means_deploy_ius[1] + 0.5),
                 xytext=(x[1] - bar_w/2, means_paper_ius[1] - 0.5),
                 arrowprops=dict(arrowstyle="-|>", color="#cc0000",
                                 lw=1.5, mutation_scale=10))
    ax.text(x[1] + 0.18, (means_paper_ius[1] + means_deploy_ius[1]) / 2,
            f"−{gap:.1f} pp\n(Illusion of\nActionability)",
            fontsize=7, color="#cc0000", fontweight="bold", va="center")

    # τ annotation
    ax.text(x[1], means_paper_ius[1] + 3.0, "τ = 0.831–0.848",
            fontsize=7, ha="center", color="#cc0000")

    ax.set_xticks(x)
    ax.set_xticklabels(["IC-FS\n(full)", "IC-FS\n(−temporal)"], fontsize=9)
    ax.set_ylabel("IUS_deploy (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 54)
    ax.set_title("(b) IUS comparison", fontsize=FONT_SIZE_TITLE, loc="left")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    #fig.suptitle(
    #    "Fig. 1. DRE leakage exposure at t = 0, k = 10 (n = 8 seeds, ±1 std).\n",
    #    fontsize=FONT_SIZE_CAPTION, y=0.02, va="bottom"
    #)

    out = OUT_DIR / "fig1_dre_leakage.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Ablation: IUS_deploy vs k (3-panel, t∈{0,1,2})
# ═════════════════════════════════════════════════════════════════════════════

def make_fig2():
    """
    3-panel line chart: IUS_deploy vs k for four IC-FS variants at t=0, t=1, t=2.
    Error bars = ±1 std across 8 seeds.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.8))
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.97, top=0.85, bottom=0.18)

    x = np.array(K_VALUES, dtype=float)
    variants = ["full", "notemp", "noact", "hard"]
    labels_map = {
        "full":   "IC-FS (full)",
        "notemp": "IC-FS (−temporal)",
        "noact":  "IC-FS (−actionability)",
        "hard":   "HardFilter+DE-FS",
    }
    horizon_labels = ["(a) t = 0", "(b) t = 1", "(c) t = 2"]
    ymax_map = {0: 60, 1: 65, 2: 70}

    for col, h in enumerate(HORIZONS):
        ax = axes[col]

        for vi, variant in enumerate(variants):
            s = SERIES[vi]
            means, stds = [], []
            for k in K_VALUES:
                d = stat_data[h].get(k)
                if d is None:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    m, sd = ms(d[variant])
                    means.append(m)
                    stds.append(sd)

            means = np.array(means)
            stds  = np.array(stds)

            # Special: mark HardDEFS at h=0 as degenerate with dashed annotation
            if variant == "hard" and h == 0:
                ax.plot(x, means, color=s["color"], linestyle=s["linestyle"],
                        marker=s["marker"], linewidth=LINE_WIDTH,
                        markersize=MARKER_SIZE, label=labels_map[variant],
                        alpha=0.5)
                ax.fill_between(x, means - stds, means + stds,
                                alpha=0.12, color=s["color"])
                # Annotate degenerate case
                ax.text(K_VALUES[-1] - 0.5, means[-1] + 2.5,
                        "k_eff=1†", fontsize=7, color=s["color"],
                        ha="right", style="italic")
            else:
                ax.plot(x, means, color=s["color"], linestyle=s["linestyle"],
                        marker=s["marker"], linewidth=LINE_WIDTH,
                        markersize=MARKER_SIZE, label=labels_map[variant])
                ax.fill_between(x, means - stds, means + stds,
                                alpha=0.12, color=s["color"])

        ax.set_xticks(K_VALUES)
        ax.set_xticklabels([f"k={k}" for k in K_VALUES], fontsize=FONT_SIZE_TICK)
        ax.set_xlabel("Feature budget k", fontsize=FONT_SIZE_LABEL)
        if col == 0:
            ax.set_ylabel("IUS_deploy (%)", fontsize=FONT_SIZE_LABEL)
        ax.set_ylim(0, ymax_map[h])
        ax.set_title(horizon_labels[col], fontsize=FONT_SIZE_TITLE, loc="left", fontweight="bold")

        # Significance annotation at t=1, k=5 (full vs noact)
        if h == 1:
            ax.text(5.0, 34.5, "** (p=0.004)", fontsize=7.5, color="#333333",
                    ha="center")
            ax.annotate("", xy=(5.0, 32.7), xytext=(5.0, 56.0),
                        arrowprops=dict(arrowstyle="<->", color="#333333", lw=0.8))

    # Shared legend below panels
    handles = [
        Line2D([0],[0], color=SERIES[i]["color"], linestyle=SERIES[i]["linestyle"],
               marker=SERIES[i]["marker"], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
               label=labels_map[v])
        for i, v in enumerate(variants)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=FONT_SIZE_LEGEND,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04), columnspacing=1.5)

    #fig.suptitle(
    #    "Fig. 2. Ablation study: IUS_deploy (%) vs feature budget k for four IC-FS variants\n",
    #    fontsize=FONT_SIZE_CAPTION, y=0.02, va="bottom"
    #)

    out = OUT_DIR / "fig2_ablation.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — IC-FS (full) vs External Baselines: IUS_deploy across horizons
# ═════════════════════════════════════════════════════════════════════════════

def make_fig3():
    """
    Grouped bar chart: IC-FS(full) at k∈{5,10,15} + 3 baselines across t={0,1,2}.
    Each group = one horizon. Bars side-by-side.
    """
    # Build data
    horizon_names = ["t = 0", "t = 1", "t = 2"]

    # IC-FS(full) mean/std at k=5,10,15
    icfs_means = {}
    icfs_stds  = {}
    for k in [5, 10, 15]:
        icfs_means[k] = []
        icfs_stds[k]  = []
        for h in HORIZONS:
            d = stat_data[h].get(k)
            if d:
                m, s = ms(d["full"])
            else:
                m, s = np.nan, np.nan
            icfs_means[k].append(m)
            icfs_stds[k].append(s)

    # Baselines (seed=42)
    base_ius = {"NSGA-II": [], "Stab. Sel.": [], "Boruta": []}
    method_map = {"NSGA-II": "NSGA-II-MOFS", "Stab. Sel.": "StabilitySelection",
                  "Boruta": "Boruta"}
    for h in HORIZONS:
        bdf = baselines.get(h)
        for label, mkey in method_map.items():
            if bdf is not None and mkey in bdf["method"].values:
                row = bdf[bdf["method"] == mkey].iloc[0]
                base_ius[label].append(row["IUS_deploy"])
            else:
                base_ius[label].append(np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.22)

    n_groups = 3  # horizons
    n_bars   = 6  # k=5, k=10, k=15, NSGA-II, StabSel, Boruta
    bar_w    = 0.12
    x        = np.arange(n_groups)
    offsets  = np.linspace(-(n_bars-1)/2*bar_w, (n_bars-1)/2*bar_w, n_bars)

    # Styles for 6 series
    series6 = [
    # IC-FS (blue family)
    {"color": "#4C72B0", "hatch": "",    "edgecolor": "#000", "label": "IC-FS (k=5)"},
    {"color": "#4C72B0", "hatch": "///", "edgecolor": "#000", "label": "IC-FS (k=10)"},
    {"color": "#4C72B0", "hatch": "...", "edgecolor": "#000", "label": "IC-FS (k=15)"},

    # Baselines (distinct colors)
    {"color": "#DD8452", "hatch": "",    "edgecolor": "#000", "label": "NSGA-II-MOFS"},
    {"color": "#55A868", "hatch": "",    "edgecolor": "#000", "label": "Stability Selection"},
    {"color": "#C44E52", "hatch": "",    "edgecolor": "#000", "label": "Boruta"},
]

    all_series = [
        (icfs_means[5],  icfs_stds[5],  None),
        (icfs_means[10], icfs_stds[10], None),
        (icfs_means[15], icfs_stds[15], None),
        (base_ius["NSGA-II"],      [0]*3, None),
        (base_ius["Stab. Sel."],   [0]*3, None),
        (base_ius["Boruta"],       [0]*3, None),
    ]

    for bi, (means, stds, _) in enumerate(all_series):
        s = series6[bi]
        xs = x + offsets[bi]
        valid = [not np.isnan(m) for m in means]
        m_arr = [m if valid[i] else 0 for i, m in enumerate(means)]
        s_arr = [sd if valid[i] else 0 for i, sd in enumerate(stds)]
        eb = [sd for sd in s_arr] if any(s > 0 for s in s_arr) else None

        ax.bar(xs, m_arr, bar_w,
               color=s["color"], hatch=s["hatch"], edgecolor=s["edgecolor"],
               linewidth=0.8, label=s["label"],
               yerr=eb if eb else None,
               capsize=2.5, error_kw={"linewidth": 0.8} if eb else {})

    ax.set_xticks(x)
    ax.set_xticklabels(horizon_names, fontsize=FONT_SIZE_TICK + 1)
    ax.set_xlabel("Prediction horizon", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("IUS_deploy (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 70)
    ax.set_xlim(-0.5, 2.5)

    # Annotate advantage at t=1 and t=2
    ax.annotate("+19.4 pp\nvs NSGA-II", xy=(1.0, 55.98), xytext=(1.22, 62),
                fontsize=7.5, color="#000",
                arrowprops=dict(arrowstyle="->", color="#000", lw=0.8))
    ax.annotate("+27.2 pp\nvs NSGA-II", xy=(2.0, 60.77), xytext=(2.22, 66),
                fontsize=7.5, color="#000",
                arrowprops=dict(arrowstyle="->", color="#000", lw=0.8))

    handles = [mpatches.Patch(
        facecolor=series6[bi]["color"], hatch=series6[bi]["hatch"],
        edgecolor="#000", linewidth=0.8, label=series6[bi]["label"]
    ) for bi in range(6)]

    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=FONT_SIZE_LEGEND,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.06), columnspacing=1.2)

    #fig.suptitle(
    #    "Fig. 3. IUS_deploy (%) for IC-FS (full, k∈{5,10,15}) and three external baselines\n",
    #    fontsize=FONT_SIZE_CAPTION, y=0.02, va="bottom"
    #)

    out = OUT_DIR / "fig3_main_vs_baselines.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Budget-Constrained Metrics (Prec@20%, Rec@20%) across horizons
# ═════════════════════════════════════════════════════════════════════════════

def make_fig4():
    """
    Two-panel line chart: Precision@20% (left) and Recall@20% (right)
    for IC-FS(full) at k=5 across t∈{0,1,2}, with random-selection baseline.
    """
    # Read from DRE files (k=5)
    prec_means, prec_stds = [], []
    rec_means,  rec_stds  = [], []

    for h in HORIZONS:
        dre_k5 = dre_data[h].get(5)
        if dre_k5 is not None:
            pm, ps = ms(dre_k5["precision20_full_deploy"])
            rm, rs = ms(dre_k5["recall20_full_deploy"])
        else:
            pm, ps, rm, rs = np.nan, np.nan, np.nan, np.nan
        prec_means.append(pm)
        prec_stds.append(ps)
        rec_means.append(rm)
        rec_stds.append(rs)

    x = np.arange(len(HORIZONS))
    h_labels = ["t = 0", "t = 1", "t = 2"]
    s0 = SERIES[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    fig.subplots_adjust(wspace=0.35, left=0.10, right=0.97, top=0.88, bottom=0.18)

    # ── Panel (a): Precision@20% ──────────────────────────────────────────────
    ax = axes[0]
    ax.errorbar(x, prec_means, yerr=prec_stds,
                color=s0["color"], linestyle=s0["linestyle"],
                marker=s0["marker"], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                capsize=4, elinewidth=1.0, label="IC-FS (full, k=5)")

    # Random baseline: population at-risk rate ≈ 38.5%
    baseline_prec = 38.5
    ax.axhline(baseline_prec, color="#777777", linestyle=":", linewidth=1.2,
               label=f"Random baseline (≈{baseline_prec}%)")
    ax.text(2.05, baseline_prec + 1.5, f"{baseline_prec}%", fontsize=8,
            color="#777777", va="bottom")

    # Lift annotations
    for i, (h, pm) in enumerate(zip(HORIZONS, prec_means)):
        if not np.isnan(pm):
            lift = pm / baseline_prec
            ax.text(i, pm + 2.0, f"{lift:.2f}×", fontsize=8,
                    ha="center", color="#000", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(h_labels)
    ax.set_xlabel("Prediction horizon", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Precision@20% (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 110)
    ax.set_title("(a) Precision@20%", fontsize=FONT_SIZE_TITLE, loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    # ── Panel (b): Recall@20% ─────────────────────────────────────────────────
    ax = axes[1]
    ax.errorbar(x, rec_means, yerr=rec_stds,
                color=s0["color"], linestyle=s0["linestyle"],
                marker=s0["marker"], linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
                capsize=4, elinewidth=1.0, label="IC-FS (full, k=5)")

    # Theoretical maximum: 20/38.5 × 100 ≈ 52%
    theo_max = (20.0 / 38.5) * 100
    ax.axhline(theo_max, color="#777777", linestyle="--", linewidth=1.2,
               label=f"Theoretical max (≈{theo_max:.1f}%)")
    ax.text(2.05, theo_max + 0.8, f"Max≈{theo_max:.0f}%", fontsize=8,
            color="#777777", va="bottom")

    # Efficiency: how close to theoretical max
    for i, (h, rm) in enumerate(zip(HORIZONS, rec_means)):
        if not np.isnan(rm):
            eff = rm / theo_max * 100
            ax.text(i, rm + 0.8, f"{eff:.0f}%\nof max", fontsize=7.5,
                    ha="center", color="#000")

    ax.set_xticks(x)
    ax.set_xticklabels(h_labels)
    ax.set_xlabel("Prediction horizon", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Recall@20% (%)", fontsize=FONT_SIZE_LABEL)
    ax.set_ylim(0, 55)
    ax.set_title("(b) Recall@20%", fontsize=FONT_SIZE_TITLE, loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    #fig.suptitle(
    #    "Fig. 4. Budget-constrained metrics (k=5, n=8 seeds, ±1 std) for IC-FS (full) on OULAD.\n",
    #    fontsize=FONT_SIZE_CAPTION, y=0.02, va="bottom"
    #)

    out = OUT_DIR / "fig4_budget_metrics.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ═════════════════════════════════════════════════════════════════════════════
# BONUS TABLE — Print LaTeX-ready main results table to stdout
# ═════════════════════════════════════════════════════════════════════════════

def print_latex_table2():
    print("\n" + "="*80)
    print("LaTeX Table 2 — Main Comparison (IUS_deploy)")
    print("="*80)
    rows_icfs = []
    for k in K_VALUES:
        row = [f"IC-FS (full) & {k}"]
        for h in HORIZONS:
            d = stat_data[h].get(k)
            if d:
                m, s = ms(d["full"])
                row.append(f"{m:.2f} $\\pm$ {s:.2f}")
            else:
                row.append("—")
        rows_icfs.append(row)

    print(r"\begin{table}[!t]")
    print(r"\caption{IUS$_\text{deploy}$ (\%) for IC-FS (full) and external baselines on OULAD (8 seeds, mean$\pm$std).}")
    print(r"\label{tab:main_comparison}")
    print(r"\centering")
    print(r"\renewcommand{\arraystretch}{1.1}")
    print(r"\begin{tabular}{lc ccc}")
    print(r"\hline")
    print(r"\textbf{Method} & \textbf{k} & \textbf{t=0} & \textbf{t=1} & \textbf{t=2} \\")
    print(r"\hline")
    for row in rows_icfs:
        print(" & ".join(row) + r" \\")
    print(r"\hline")

    base_vals = {
        "NSGA-II-MOFS": ("14", "7.05", "36.26", "33.53"),
        "Stab. Selection": ("15", "3.70", "18.47", "32.50"),
        "Boruta": ("11", "5.11", "27.67", "28.64"),
    }
    for name, (k, v0, v1, v2) in base_vals.items():
        print(f"{name} & {k} & {v0} & {v1} & {v2} " + r"\\")
    print(r"\hline")
    print(r"\multicolumn{5}{l}{\small$^\dagger$ HardFilter+DE-FS degenerates to $k_\text{eff}=1$ at $t=0$; see \S\ref{sec:hardfilter}.}\\")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating ESWA figures for IC-FS OULAD experiments...")
    print(f"  Output directory: {OUT_DIR}")
    print()

    print("[Fig 1] DRE Leakage Exposure...")
    make_fig1()

    print("[Fig 2] Ablation Study...")
    make_fig2()

    print("[Fig 3] Main Comparison vs Baselines...")
    make_fig3()

    print("[Fig 4] Budget-Constrained Metrics...")
    make_fig4()

    print_latex_table2()

    print()
    print("Done. Files saved:")
    for f in sorted(OUT_DIR.glob("fig[1-4]_*.p*")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")