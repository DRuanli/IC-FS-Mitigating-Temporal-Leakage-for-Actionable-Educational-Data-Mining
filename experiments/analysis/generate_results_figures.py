"""generate_results_figures.py
====================================================================
Regenerate every figure used in Section 5 (Results and Analysis) of
the IC-FS / ESWA manuscript.

Supports both OULAD and UCI datasets.

Input (OULAD):
  - results/oulad/k15/stat8_oulad_h{0,1,2}_k15.csv      (8 seeds × 4 methods)
  - results/oulad/k15/dre_multi_oulad_h{0,1,2}_k15.csv  (8 seeds × paired full/notemp)
  - results/oulad/k15/oulad_icfs_h{0,1,2}_k15.csv       (alpha sweep, 5 alphas)
  - results/oulad/baselines_oulad_h{0,1,2}.csv          (NSGA-II, Boruta, StabSel)

Input (UCI):
  - results/uci/{math|portuguese}/uci_{math|portuguese}_icfs_multi_h{0,1,2}.csv
  - results/uci/{math|portuguese}/uci_{math|portuguese}_icfs_h{0,1,2}.csv

Output : manuscript/figures/results/{dataset}/fig_R{1,2,3,4}_*.{pdf,png}

Typography: Times-style serif font, Wong colourblind-safe palette, ~3.4-in column width.

Usage:
    # OULAD figures (default)
    python experiments/analysis/generate_results_figures.py

    # UCI Math figures
    python experiments/analysis/generate_results_figures.py --dataset uci_math

    # UCI Portuguese figures
    python experiments/analysis/generate_results_figures.py --dataset uci_portuguese
====================================================================
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

# --------------------------------------------------------------------
# Style — kept consistent with generate_figures_eswa.py and
# generate_methodology_figures.py
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
C_YELLOW  = "#F0E442"
C_SKY     = "#56B4E9"
C_GREY    = "#666666"

# --------------------------------------------------------------------
# Paths and Configuration
# --------------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()

HORIZONS   = [0, 1, 2]
KFIX       = 15        # primary budget for OULAD headline figures
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def setup_paths(dataset: str = "oulad"):
    """
    Setup paths based on dataset.

    Args:
        dataset: "oulad", "uci_math", or "uci_portuguese"

    Returns:
        (RESULTS_DIR, OUTDIR, k_value)
    """
    if dataset == "oulad":
        results_dir = ROOT / "results" / "oulad"
        outdir = ROOT / "manuscript" / "figures" / "results" / "oulad"
        k = KFIX
    elif dataset.startswith("uci_"):
        dataset_name = dataset.split("_")[1]  # "math" or "portuguese"
        results_dir = ROOT / "results" / "uci" / dataset_name / dataset_name
        outdir = ROOT / "manuscript" / "figures" / "results" / f"uci_{dataset_name}"
        k = 5  # UCI uses top_k=5
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'oulad', 'uci_math', or 'uci_portuguese'")

    outdir.mkdir(parents=True, exist_ok=True)
    return results_dir, outdir, k


# --------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------
def load_stat8(h: int, results_dir: Path, k: int, dataset_type: str = "oulad") -> pd.DataFrame:
    """Eight-seed paired comparisons of IC-FS(full), --temporal, --actionability, hardDEFS."""
    if dataset_type == "oulad":
        return pd.read_csv(results_dir / f"k{k}" / f"stat8_oulad_h{h}_k{k}.csv")
    else:
        # UCI doesn't have stat8 files yet - use multi files
        return pd.read_csv(results_dir / f"uci_{dataset_type}_icfs_multi_h{h}.csv")


def load_dre(h: int, results_dir: Path, k: int, dataset_type: str = "oulad") -> pd.DataFrame:
    """Eight-seed paired DRE diagnostics — full vs --temporal with τ, IUS_paper, IUS_deploy."""
    if dataset_type == "oulad":
        return pd.read_csv(results_dir / f"k{k}" / f"dre_multi_oulad_h{h}_k{k}.csv")
    else:
        # UCI may not have DRE files - return empty dataframe
        csv_path = results_dir / f"dre_multi_uci_{dataset_type}_h{h}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()


def load_alpha_sweep(h: int, results_dir: Path, k: int, dataset_type: str = "oulad") -> pd.DataFrame:
    """Five-row α-sweep with stability, AR_available, IUS_deploy."""
    if dataset_type == "oulad":
        return pd.read_csv(results_dir / f"k{k}" / f"oulad_icfs_h{h}_k{k}.csv")
    else:
        return pd.read_csv(results_dir / f"uci_{dataset_type}_icfs_h{h}.csv")


def load_baselines(h: int, results_dir: Path, dataset_type: str = "oulad") -> pd.DataFrame:
    """One-seed external baselines: NSGA-II-MOFS, StabilitySelection, Boruta."""
    if dataset_type == "oulad":
        return pd.read_csv(results_dir / f"baselines_oulad_h{h}.csv")
    else:
        csv_path = results_dir / f"baselines_uci_{dataset_type}_h{h}.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()


# --------------------------------------------------------------------
# Figure R1 (= Fig. 5) — IUS_deploy across method × horizon
# --------------------------------------------------------------------
def fig_method_horizon_comparison(results_dir: Path, outdir: Path, k: int, dataset_type: str = "oulad") -> None:
    """Grouped bar chart: IUS_deploy mean ± std across 7 methods × 3 horizons."""

    # For UCI, we may not have all ablation methods - simplify to just IC-FS
    if dataset_type in ["math", "portuguese"]:
        # UCI simplified version - only IC-FS full
        means = np.zeros((1, len(HORIZONS)))
        stds  = np.zeros_like(means)

        for j, h in enumerate(HORIZONS):
            df = load_stat8(h, results_dir, k, dataset_type)
            if "IUS_deploy" in df.columns:
                vals = df["IUS_deploy"].to_numpy()
                means[0, j] = vals.mean()
                stds[0, j]  = vals.std(ddof=1)

        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        bar_w = 0.25
        x = np.arange(len(HORIZONS))

        ax.bar(x, means[0], width=bar_w, yerr=stds[0],
               capsize=2.0, error_kw=dict(elinewidth=0.6),
               color=C_BLUE, edgecolor="white", linewidth=0.4, label="IC-FS")

        ax.set_xticks(x)
        ax.set_xticklabels([f"$h={h}$" for h in HORIZONS])
        ax.set_xlabel("Prediction horizon")
        ax.set_ylabel(r"$\mathrm{IUS}_{\mathrm{deploy}}$")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", frameon=False)

        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(outdir / f"fig_R1_method_horizon.{ext}")
        plt.close(fig)
        return

    # OULAD version with all methods
    methods = [
        ("IC-FS\n(full)",   "IUS_deploy_full",     "stat8",    C_BLUE),
        ("--temporal",      "IUS_deploy_noTemp",   "stat8",    C_ORANGE),
        ("--action.",       "IUS_deploy_noAction", "stat8",    C_PINK),
        ("HardFilt.\n+DE-FS","IUS_deploy_hardDEFS","stat8",    C_GREEN),
        ("NSGA-II",         "IUS_paper",           "baseline", C_SKY),
        ("Boruta",          "IUS_paper",           "baseline", C_YELLOW),
        ("Stab.\nSel.",     "IUS_paper",           "baseline", C_GREY),
    ]
    baseline_method_names = {"NSGA-II": "NSGA-II-MOFS",
                             "Boruta": "Boruta",
                             "Stab.\nSel.": "StabilitySelection"}

    means = np.zeros((len(methods), len(HORIZONS)))
    stds  = np.zeros_like(means)

    for j, h in enumerate(HORIZONS):
        df_stat = load_stat8(h, results_dir, k, dataset_type)
        df_base = load_baselines(h, results_dir, dataset_type)
        for i, (label, col, src, _) in enumerate(methods):
            if src == "stat8":
                if col in df_stat.columns:
                    vals = df_stat[col].to_numpy()
                    means[i, j] = vals.mean()
                    stds[i, j]  = vals.std(ddof=1)
            else:
                if not df_base.empty and "method" in df_base.columns:
                    row = df_base[df_base["method"] == baseline_method_names[label]]
                    means[i, j] = row[col].iloc[0] if len(row) else np.nan
                    stds[i, j]  = 0.0

    fig, ax = plt.subplots(figsize=(7.0, 3.2))     # full text-width, two-column figure
    bar_w     = 0.11
    n_methods = len(methods)
    x         = np.arange(len(HORIZONS))

    for i, (label, _, _, colour) in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * bar_w
        ax.bar(x + offset, means[i], width=bar_w, yerr=stds[i],
               capsize=2.0, error_kw=dict(elinewidth=0.6),
               color=colour, edgecolor="white", linewidth=0.4, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$h={h}$" for h in HORIZONS])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel(r"$\mathrm{IUS}_{\mathrm{deploy}}$")
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_ylim(0, 75)
    ax.legend(ncol=4, loc="upper left", frameon=False,
              columnspacing=1.2, handletextpad=0.5,
              bbox_to_anchor=(0.0, 1.18))

    # Annotate the structural ceiling at h=0
    ax.annotate("Tier-1 sparsity\nstructural ceiling",
                xy=(0.0, 7), xytext=(0.0, 28),
                ha="center", fontsize=7, color=C_GREY,
                arrowprops=dict(arrowstyle="->", color=C_GREY,
                                lw=0.6, shrinkA=0, shrinkB=2))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_R1_method_horizon.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------
# Figure R2 (= Fig. 6) — DRE leakage diagnostic at h=0
# --------------------------------------------------------------------
def fig_dre_leakage_diagnostic(results_dir: Path, outdir: Path, k: int, dataset_type: str = "oulad") -> None:
    """Two-panel: (a) IUS_paper vs IUS_deploy for IC-FS(full) and --temporal at h=0;
    (b) τ leakage coefficient distribution, again at h=0."""
    df_stat = load_stat8(0, results_dir, k, dataset_type)
    df_dre  = load_dre(0, results_dir, k, dataset_type)

    # Skip if DRE data not available (UCI may not have it yet)
    if df_dre.empty or "tau_full" not in df_dre.columns:
        print(f"  [fig_R2] Skipping DRE figure - data not available for {dataset_type}")
        return

    paper_full   = df_stat["IUS_paper_full"].to_numpy()
    deploy_full  = df_stat["IUS_deploy_full"].to_numpy()
    paper_temp   = df_stat["IUS_paper_noTemp"].to_numpy()
    deploy_temp  = df_stat["IUS_deploy_noTemp"].to_numpy()

    tau_full     = df_dre["tau_full"].to_numpy()
    tau_temp     = df_dre["tau_notemp"].to_numpy()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0),
                                     gridspec_kw=dict(width_ratios=[1.4, 1.0]))

    # ---- panel (a): paired bars, IUS_paper vs IUS_deploy --------------
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

    # Annotate the gap on --temporal
    gap = means_paper[1] - means_deploy[1]
    ax_a.annotate(f"{gap:.0f}-pt drop\n($\\tau \\approx {tau_temp.mean():.2f}$)",
                  xy=(1 + bar_w/2, means_deploy[1]),
                  xytext=(1.05, means_paper[1] * 0.5),
                  ha="left", fontsize=7.5, color=C_ORANGE,
                  arrowprops=dict(arrowstyle="->", color=C_ORANGE,
                                  lw=0.7, shrinkA=2, shrinkB=2))

    # ---- panel (b): leakage coefficient τ -------------------------------
    ax_b.bar([0, 1], [tau_full.mean(), tau_temp.mean()],
             yerr=[tau_full.std(ddof=1), tau_temp.std(ddof=1)],
             capsize=2.5, color=[C_BLUE, C_ORANGE], edgecolor="white", width=0.55)
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
# Figure R3 (= Fig. 7) — α-sweep with dual axes
# --------------------------------------------------------------------
def fig_alpha_sweep(results_dir: Path, outdir: Path, k: int, dataset_type: str = "oulad") -> None:
    """Three-panel α-sweep: IUS_deploy (left axis) and AR_available (right axis)
    against α, one panel per horizon. The nested-best α is marked with a star."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=True)

    for ax, h in zip(axes, HORIZONS):
        df = load_alpha_sweep(h, results_dir, k, dataset_type).sort_values("alpha")
        alphas = df["alpha"].to_numpy()
        ius    = df["IUS_deploy"].to_numpy()
        ar     = df["AR_available"].to_numpy()
        nested = df["nested_best"].to_numpy()

        # Left axis: IUS_deploy
        ax.plot(alphas, ius, marker="o", color=C_BLUE,
                label=r"$\mathrm{IUS}_{\mathrm{deploy}}$")
        ax.set_xlabel(r"Trade-off parameter $\alpha$")
        ax.set_xticks(ALPHA_GRID)
        if h == 0:
            ax.set_ylabel(r"$\mathrm{IUS}_{\mathrm{deploy}}$", color=C_BLUE)
        ax.tick_params(axis="y", labelcolor=C_BLUE)

        # Right axis: AR_available
        ax_r = ax.twinx()
        ax_r.plot(alphas, ar, marker="s", color=C_ORANGE, linestyle="--",
                  label=r"$\mathrm{AR}_{\mathrm{available}}$")
        ax_r.set_ylim(0, 1.0)
        if h == 2:
            ax_r.set_ylabel(r"$\mathrm{AR}_{\mathrm{available}}$", color=C_ORANGE)
        ax_r.tick_params(axis="y", labelcolor=C_ORANGE)
        ax_r.grid(False)

        # Highlight the nested-best α with a star
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

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_R3_alpha_sweep.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------
# Figure R4 (= Fig. 8) — Stability vs performance trade-off
# --------------------------------------------------------------------
def fig_stability_performance(results_dir: Path, outdir: Path, k: int, dataset_type: str = "oulad") -> None:
    """Scatter of (Jaccard stability, IUS_deploy) for every (α, h) combination,
    distinguishing horizons and highlighting nested-best α."""
    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    horizon_colours = {0: C_ORANGE, 1: C_BLUE, 2: C_GREEN}
    horizon_marker  = {0: "o",       1: "s",     2: "^"}

    for h in HORIZONS:
        df = load_alpha_sweep(h, results_dir, k, dataset_type)
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

    # Synthetic legend handles
    handles = []
    for h in HORIZONS:
        handles.append(plt.Line2D([0], [0], marker=horizon_marker[h], linestyle="",
                                  color=horizon_colours[h], label=f"$h={h}$",
                                  markersize=6))
    handles.append(plt.Line2D([0], [0], marker="*", linestyle="",
                              color="black", markersize=8,
                              markerfacecolor="white",
                              label=r"$\alpha^\ast$ (nested)"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=7)

    ax.set_xlabel(r"Bootstrap Jaccard stability $J$")
    ax.set_ylabel(r"$\mathrm{IUS}_{\mathrm{deploy}}$")
    ax.set_xlim(0.45, 1.05)
    ax.set_ylim(0, 70)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_R4_stability_perf.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate IC-FS results figures")
    parser.add_argument("--dataset", type=str, default="oulad",
                          help="Dataset: 'oulad', 'uci_math', or 'uci_portuguese' (default: oulad)")
    args = parser.parse_args()

    dataset = args.dataset.lower()
    results_dir, outdir, k = setup_paths(dataset)

    # Extract dataset type for loader functions
    if dataset == "oulad":
        dataset_type = "oulad"
    elif dataset.startswith("uci_"):
        dataset_type = dataset.split("_")[1]  # "math" or "portuguese"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"[results-figures] Dataset: {dataset}")
    print(f"[results-figures] Reading data from {results_dir}")
    print(f"[results-figures] Writing figures to  {outdir}")

    fig_method_horizon_comparison(results_dir, outdir, k, dataset_type)
    fig_dre_leakage_diagnostic(results_dir, outdir, k, dataset_type)
    fig_alpha_sweep(results_dir, outdir, k, dataset_type)
    fig_stability_performance(results_dir, outdir, k, dataset_type)

    print("[results-figures] done. Generated:")
    for f in sorted(outdir.glob("fig_R*.pdf")):
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()