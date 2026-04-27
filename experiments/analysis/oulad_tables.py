"""
================================================================================
OULAD Tables and Figures for Manuscript
================================================================================
Generates publication-ready tables and figures from OULAD experiment outputs.

Outputs (manuscript/figures/):
  - table5_cross_dataset.csv    Cross-dataset comparison (UCI vs OULAD)
  - table6_oulad_baselines.csv  Baseline comparison on OULAD
  - table7_oulad_dre.csv        DRE statistical evidence on OULAD
  - fig5_oulad_ius_horizons.png IUS jump h0→h1→h2 narrative
  - fig6_oulad_dre_boxplot.png  Paper-style vs DRE F1 on OULAD
  - fig7_oulad_pareto.png       F1 vs AR Pareto (UCI + OULAD overlay)

Usage:
    python experiments/analysis/oulad_tables.py
================================================================================
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 110,
})

OUT_DIR = project_root / "manuscript" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RES_OULAD = project_root / "results" / "oulad"
RES_UCI = project_root / "results" / "uci"


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────
def load_csv_safe(path):
    if not path.exists():
        print(f"  [skip] {path.name} not found")
        return None
    return pd.read_csv(path)


def fmt_mean_std(m, s):
    if pd.isna(s) or s is None: return f"{m:.2f}"
    return f"{m:.2f}±{s:.2f}"


# ────────────────────────────────────────────────────────────────────────
# TABLE 5 — Cross-dataset comparison
# ────────────────────────────────────────────────────────────────────────
def table5_cross_dataset():
    print("\n[Table 5] Cross-dataset comparison")
    rows = []

    # UCI Math, t=0 — IC-FS(full) and IC-FS(-temporal) under DRE
    uci_dre = load_csv_safe(RES_UCI / "dre_multi_math_h0.csv")
    if uci_dre is not None:
        rows.append({
            "dataset": "UCI-Math", "horizon": 0, "n": len(uci_dre),
            "method": "IC-FS (full)",
            "F1_deploy_mean": uci_dre["f1_full_deploy"].mean(),
            "F1_deploy_std": uci_dre["f1_full_deploy"].std(ddof=1),
            "IUS_deploy_mean": uci_dre["IUS_deploy_full"].mean(),
            "IUS_deploy_std": uci_dre["IUS_deploy_full"].std(ddof=1),
            "AR_mean": uci_dre["AR_full"].mean(),
        })
        rows.append({
            "dataset": "UCI-Math", "horizon": 0, "n": len(uci_dre),
            "method": "IC-FS (-temporal) ≈ DE-FS",
            "F1_deploy_mean": uci_dre["f1_notemp_deploy"].mean(),
            "F1_deploy_std": uci_dre["f1_notemp_deploy"].std(ddof=1),
            "IUS_deploy_mean": uci_dre["IUS_deploy_notemp"].mean(),
            "IUS_deploy_std": uci_dre["IUS_deploy_notemp"].std(ddof=1),
            "AR_mean": uci_dre["AR_notemp"].mean(),
        })

    # OULAD t=0,1,2 — DRE multi-seed
    for h in [0, 1, 2]:
        oulad_dre = load_csv_safe(RES_OULAD / f"dre_multi_oulad_h{h}.csv")
        if oulad_dre is None: continue
        for label, fcol, icol, arcol in [
            ("IC-FS (full)", "f1_full_deploy", "IUS_deploy_full", "AR_full"),
            ("IC-FS (-temporal) ≈ DE-FS", "f1_notemp_deploy",
             "IUS_deploy_notemp", "AR_notemp"),
        ]:
            rows.append({
                "dataset": "OULAD", "horizon": h, "n": len(oulad_dre),
                "method": label,
                "F1_deploy_mean": oulad_dre[fcol].mean(),
                "F1_deploy_std": oulad_dre[fcol].std(ddof=1),
                "IUS_deploy_mean": oulad_dre[icol].mean(),
                "IUS_deploy_std": oulad_dre[icol].std(ddof=1),
                "AR_mean": oulad_dre[arcol].mean(),
            })

    if not rows:
        print("  No data — run DRE experiments first")
        return None

    df = pd.DataFrame(rows)
    df["F1_deploy"] = df.apply(
        lambda r: fmt_mean_std(r["F1_deploy_mean"], r["F1_deploy_std"]), axis=1)
    df["IUS_deploy"] = df.apply(
        lambda r: fmt_mean_std(r["IUS_deploy_mean"], r["IUS_deploy_std"]), axis=1)
    df["AR"] = df["AR_mean"].apply(lambda x: f"{x:.3f}")

    show = df[["dataset", "horizon", "method", "F1_deploy",
                "IUS_deploy", "AR", "n"]]
    out_path = OUT_DIR / "table5_cross_dataset.csv"
    show.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")
    print(show.to_string(index=False))
    return df


# ────────────────────────────────────────────────────────────────────────
# TABLE 6 — Baselines on OULAD (single seed)
# ────────────────────────────────────────────────────────────────────────
def table6_oulad_baselines():
    print("\n[Table 6] OULAD baselines (NSGA-II, Stab.Sel., Boruta)")
    rows = []
    for h in [0, 1, 2]:
        b = load_csv_safe(RES_OULAD / f"baselines_oulad_h{h}.csv")
        if b is None: continue
        rows.append(b)

        # Add IC-FS (full) reference at this horizon — best from sweep
        icfs = load_csv_safe(RES_OULAD / f"oulad_icfs_h{h}.csv")
        if icfs is not None:
            best = icfs.loc[icfs["IUS"].idxmax()].to_dict()
            rows.append(pd.DataFrame([{
                "method": "IC-FS (full)", "horizon": h,
                "accuracy": best["accuracy"], "f1": best["f1"],
                "AR": best["AR"], "TVS": best["TVS"], "IUS": best["IUS"],
                "n_features": best["n_features"],
                "selected": best.get("selected", ""),
            }]))

    if not rows:
        print("  No data")
        return None
    df = pd.concat(rows, ignore_index=True)

    # Normalize column names: use IUS_deploy if available, otherwise IUS
    if "IUS_deploy" in df.columns and "IUS" not in df.columns:
        df["IUS"] = df["IUS_deploy"]
    if "f1_deploy" in df.columns and "f1" not in df.columns:
        df["f1"] = df["f1_deploy"]

    df = df.sort_values(["horizon", "IUS"], ascending=[True, False])

    out_path = OUT_DIR / "table6_oulad_baselines.csv"
    show_cols = ["method", "horizon", "f1", "AR", "TVS", "IUS", "n_features"]
    cols_avail = [c for c in show_cols if c in df.columns]
    df[cols_avail].to_csv(out_path, index=False)
    print(f"  Saved {out_path}")
    for h in [0, 1, 2]:
        sub = df[df["horizon"] == h][cols_avail]
        if len(sub) > 0:
            print(f"\n  --- t={h} ---")
            print(sub.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return df


# ────────────────────────────────────────────────────────────────────────
# TABLE 7 — Statistical tests on OULAD DRE
# ────────────────────────────────────────────────────────────────────────
def table7_oulad_dre_stats():
    print("\n[Table 7] OULAD DRE Wilcoxon statistical tests")
    rows = []
    for h in [0, 1, 2]:
        dre = load_csv_safe(RES_OULAD / f"dre_multi_oulad_h{h}.csv")
        if dre is None or len(dre) < 2: continue

        a_f1 = dre["f1_full_deploy"].values
        b_f1 = dre["f1_notemp_deploy"].values
        a_ius = dre["IUS_deploy_full"].values
        b_ius = dre["IUS_deploy_notemp"].values

        diff_f1 = a_f1 - b_f1
        diff_ius = a_ius - b_ius

        def safe_wilcox(a, b):
            if np.allclose(a, b):
                return np.nan, np.nan
            try:
                stat, p = wilcoxon(a, b, alternative="greater",
                                     zero_method="wilcox")
                return stat, p
            except ValueError:
                return np.nan, np.nan

        stat_f1, p_f1 = safe_wilcox(a_f1, b_f1)
        stat_ius, p_ius = safe_wilcox(a_ius, b_ius)

        d_f1 = diff_f1.mean() / diff_f1.std(ddof=1) if diff_f1.std(ddof=1) > 0 else 0
        d_ius = diff_ius.mean() / diff_ius.std(ddof=1) if diff_ius.std(ddof=1) > 0 else 0

        n_leak = int(dre["notemp_has_T3"].sum()) if "notemp_has_T3" in dre.columns else None

        rows.append({
            "horizon": h, "n_seeds": len(dre),
            "F1_full_mean": a_f1.mean(), "F1_notemp_mean": b_f1.mean(),
            "F1_diff_mean": diff_f1.mean(),
            "F1_Wilcoxon_p": p_f1, "F1_Cohens_d": d_f1,
            "IUS_full_mean": a_ius.mean(), "IUS_notemp_mean": b_ius.mean(),
            "IUS_diff_mean": diff_ius.mean(),
            "IUS_Wilcoxon_p": p_ius, "IUS_Cohens_d": d_ius,
            "notemp_T3_leaks": n_leak,
        })

    if not rows:
        print("  No data")
        return None
    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "table7_oulad_dre.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")

    # Pretty print
    bonf = 0.05 / 3
    for _, r in df.iterrows():
        sig_f1 = ("***" if r["F1_Wilcoxon_p"] < 0.001
                   else "**" if r["F1_Wilcoxon_p"] < 0.01
                   else "*" if r["F1_Wilcoxon_p"] < bonf else "ns")
        sig_ius = ("***" if r["IUS_Wilcoxon_p"] < 0.001
                    else "**" if r["IUS_Wilcoxon_p"] < 0.01
                    else "*" if r["IUS_Wilcoxon_p"] < bonf else "ns")
        print(f"\n  --- t={int(r['horizon'])} (n={int(r['n_seeds'])} seeds) ---")
        print(f"    F1_deploy: full={r['F1_full_mean']:.2f} vs "
               f"-temp={r['F1_notemp_mean']:.2f} | "
               f"diff={r['F1_diff_mean']:+.2f} d={r['F1_Cohens_d']:+.2f} "
               f"p={r['F1_Wilcoxon_p']:.5f} [{sig_f1}]")
        print(f"    IUS_deploy: full={r['IUS_full_mean']:.2f} vs "
               f"-temp={r['IUS_notemp_mean']:.2f} | "
               f"diff={r['IUS_diff_mean']:+.2f} d={r['IUS_Cohens_d']:+.2f} "
               f"p={r['IUS_Wilcoxon_p']:.5f} [{sig_ius}]")
        if r["notemp_T3_leaks"] is not None:
            print(f"    Tier-3 leakage in -temporal: "
                   f"{int(r['notemp_T3_leaks'])}/{int(r['n_seeds'])} seeds")
    return df


# ────────────────────────────────────────────────────────────────────────
# FIG 5 — IUS jump across horizons (OULAD narrative)
# ────────────────────────────────────────────────────────────────────────
def fig5_oulad_ius_horizons():
    print("\n[Fig 5] OULAD IUS across horizons")
    horizons = []
    f1_means, f1_stds = [], []
    ar_means = []
    ius_means, ius_stds = [], []

    for h in [0, 1, 2]:
        # Multi-seed if available, else single-seed
        multi = load_csv_safe(RES_OULAD / f"oulad_icfs_multi_h{h}.csv")
        if multi is not None and len(multi) > 1:
            horizons.append(h)
            f1_means.append(multi["f1"].mean())
            f1_stds.append(multi["f1"].std(ddof=1))
            ar_means.append(multi["AR"].mean())
            ius_means.append(multi["IUS"].mean())
            ius_stds.append(multi["IUS"].std(ddof=1))
        else:
            single = load_csv_safe(RES_OULAD / f"oulad_icfs_h{h}.csv")
            if single is not None:
                best = single.loc[single["IUS"].idxmax()]
                horizons.append(h)
                f1_means.append(best["f1"])
                f1_stds.append(0)
                ar_means.append(best["AR"])
                ius_means.append(best["IUS"])
                ius_stds.append(0)

    if not horizons:
        print("  No data")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5.5))

    x = np.array(horizons)
    width = 0.25

    b1 = ax1.bar(x - width, f1_means, width, yerr=f1_stds,
                   label="F1 (%)", color="#1f77b4", capsize=4)
    b2 = ax1.bar(x, [a * 100 for a in ar_means], width,
                   label="AR (×100)", color="#ff7f0e")
    b3 = ax1.bar(x + width, ius_means, width, yerr=ius_stds,
                   label="IUS (%)", color="#d62728", capsize=4,
                   edgecolor='black', linewidth=1.5)

    for bar, v in zip(b3, ius_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                  f"{v:.1f}", ha='center', fontweight='bold', fontsize=11)

    ax1.set_xlabel("Prediction horizon")
    ax1.set_ylabel("Metric value")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"t={h}" for h in horizons])
    ax1.set_ylim(0, 100)
    ax1.set_title("Figure 5 — IC-FS metric evolution across prediction horizons (OULAD)\n"
                   "IUS jumps from t=0 (no Tier-2 signal) to t=1 (clickstream available)")
    ax1.legend(loc='upper left', framealpha=0.95)

    # Annotate jump
    if len(horizons) >= 2 and ius_means[0] > 0:
        jump = (ius_means[1] / ius_means[0] - 1) * 100
        ax1.annotate(f"+{jump:.0f}%\nIUS jump",
                       xy=(x[1] + width, ius_means[1]),
                       xytext=(x[1] + width + 0.15, ius_means[1] - 15),
                       arrowprops=dict(arrowstyle="->", color="darkred", lw=2),
                       fontsize=11, fontweight="bold", color="darkred")

    plt.tight_layout()
    out_path = OUT_DIR / "fig5_oulad_ius_horizons.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# ────────────────────────────────────────────────────────────────────────
# FIG 6 — DRE boxplot OULAD (mirrors UCI fig3)
# ────────────────────────────────────────────────────────────────────────
def fig6_oulad_dre_boxplot():
    print("\n[Fig 6] OULAD DRE boxplot")
    available = []
    for h in [0, 1, 2]:
        d = load_csv_safe(RES_OULAD / f"dre_multi_oulad_h{h}.csv")
        if d is not None and len(d) > 1:
            available.append((h, d))

    if not available:
        print("  No DRE data")
        return

    n_hor = len(available)
    fig, axes = plt.subplots(1, n_hor, figsize=(5 * n_hor, 5.5), sharey=False)
    if n_hor == 1:
        axes = [axes]

    for ax, (h, df) in zip(axes, available):
        data = [
            df["f1_full_paper"].values,
            df["f1_notemp_paper"].values,
            df["f1_full_deploy"].values,
            df["f1_notemp_deploy"].values,
        ]
        positions = [1, 2, 4, 5]
        labels = ["IC-FS\nfull\n(paper)",
                   "IC-FS\n-temporal\n(paper)",
                   "IC-FS\nfull\n(DRE)",
                   "IC-FS\n-temporal\n(DRE)"]
        colors = ["#2ca02c", "#d62728", "#2ca02c", "#d62728"]

        bp = ax.boxplot(data, positions=positions, widths=0.6,
                          patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for pos, vals in zip(positions, data):
            ax.scatter([pos] * len(vals), vals, s=30, color="k",
                        alpha=0.6, zorder=3)

        # Paired lines
        for i in range(len(df)):
            ax.plot([positions[0], positions[1]],
                     [data[0][i], data[1][i]], "-", alpha=0.3, color="gray")
            ax.plot([positions[2], positions[3]],
                     [data[2][i], data[3][i]], "-", alpha=0.3, color="gray")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("F1 score (%)")
        ax.set_title(f"OULAD, t={h}")
        ax.axvline(3, color="k", linestyle="--", alpha=0.5)

        ymin, ymax = ax.get_ylim()
        ax.text(1.5, ymax * 0.99, "Paper eval",
                 ha="center", fontsize=9, fontweight="bold")
        ax.text(4.5, ymax * 0.99, "Deployment eval",
                 ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(f"Figure 6 — OULAD: Paper-style vs Deployment-Realistic F1 "
                   f"({len(available[0][1])} seeds per horizon)",
                   fontsize=13)
    plt.tight_layout()
    out_path = OUT_DIR / "fig6_oulad_dre_boxplot.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


# ────────────────────────────────────────────────────────────────────────
# FIG 7 — Pareto F1 vs AR (UCI + OULAD overlay)
# ────────────────────────────────────────────────────────────────────────
def fig7_oulad_pareto():
    print("\n[Fig 7] Cross-dataset Pareto F1 vs AR")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, h in zip(axes, [0, 1, 2]):
        # OULAD α-sweep
        oulad = load_csv_safe(RES_OULAD / f"oulad_icfs_h{h}.csv")
        if oulad is not None:
            sc = ax.scatter(oulad["AR"], oulad["f1"], c=oulad["alpha"],
                              cmap="viridis", s=120, edgecolor="k", zorder=3,
                              label="IC-FS α-sweep")

        # Baselines
        b = load_csv_safe(RES_OULAD / f"baselines_oulad_h{h}.csv")
        if b is not None:
            colors = {"NSGA-II-MOFS": "#d62728",
                       "StabilitySelection": "#ff7f0e",
                       "Boruta": "#9467bd"}
            markers = {"NSGA-II-MOFS": "s",
                        "StabilitySelection": "^",
                        "Boruta": "D"}
            for _, r in b.iterrows():
                if pd.notna(r.get("f1", np.nan)):
                    ax.scatter([r["AR"]], [r["f1"]], s=180,
                                marker=markers.get(r["method"], "o"),
                                color=colors.get(r["method"], "gray"),
                                edgecolor="k", linewidth=1.5, zorder=5,
                                label=r["method"])

        ax.set_xlabel("Actionability Ratio (AR)")
        ax.set_ylabel("F1 score (%)")
        ax.set_title(f"OULAD, t={h}")
        ax.set_xlim(-0.05, 1.05)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

        if oulad is not None and len(oulad) > 0:
            cbar = plt.colorbar(sc, ax=ax, label="α")

    fig.suptitle("Figure 7 — Actionability–F1 trade-off on OULAD across horizons",
                   fontsize=13)
    plt.tight_layout()
    out_path = OUT_DIR / "fig7_oulad_pareto.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def main():
    print("=" * 80)
    print("OULAD Tables and Figures")
    print("=" * 80)

    table5_cross_dataset()
    table6_oulad_baselines()
    table7_oulad_dre_stats()
    fig5_oulad_ius_horizons()
    fig6_oulad_dre_boxplot()
    fig7_oulad_pareto()

    print(f"\n{'='*80}")
    print(f"All outputs in: {OUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()