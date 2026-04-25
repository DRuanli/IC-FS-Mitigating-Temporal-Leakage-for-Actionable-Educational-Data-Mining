"""
Generate publication figures:
  Fig 1: Pareto frontier (F1 vs AR) — IC-FS sweep vs NSGA-II
  Fig 2: Sensitivity heatmap (w_mid × top_k → IUS)
  Fig 3: DRE boxplot — paper F1 vs deployment F1 across seeds
  Fig 4: IUS decomposition bar chart by method
"""
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 110,
})

OUT_DIR = "manuscript"
os.makedirs(OUT_DIR, exist_ok=True)


# ── FIG 1: Pareto — F1 vs AR across methods on Math h=0 ──────────────────
def fig1_pareto():
    # Load IC-FS alpha-sweep for seed 42 on Math
    from utils_data import load_and_split
    from ic_fs_v2 import (TAXONOMY_UCI, feature_scores_for_selection, ic_fs_select,
                            actionability_ratio, temporal_validity_score, compute_ius,
                            filter_by_horizon)
    from run_ablation_fast import _eval_subset

    X_tr, X_te, y_tr, y_te, names = load_and_split("student-mat.csv", 0.2, 42)
    available = filter_by_horizon(names, 0)
    avail_idx = [names.index(f) for f in available]
    X_tr_a = X_tr[:, avail_idx]
    X_te_a = X_te[:, avail_idx]
    sdf = feature_scores_for_selection(X_tr_a, y_tr, available)

    icfs_pts = []
    for alpha in np.linspace(0, 1, 21):
        sel = ic_fs_select(sdf, alpha, 12)
        sel_local = [available.index(f) for f in sel]
        ev = _eval_subset(X_tr_a, y_tr, X_te_a, y_te, sel_local)
        ar = actionability_ratio(sel)
        icfs_pts.append({"alpha": alpha, "f1": ev["f1"]*100, "AR": ar,
                          "IUS": ev["f1"]*ar*100})

    # Load baseline data
    b_df = pd.read_csv("baselines_math_h0.csv")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ic = pd.DataFrame(icfs_pts)
    sc = ax.scatter(ic["AR"], ic["f1"], c=ic["alpha"], cmap="viridis",
                     s=80, edgecolor="k", zorder=3, label="IC-FS α-sweep")
    cbar = plt.colorbar(sc, ax=ax, label="α (predictive weight)")

    # Highlight best-IUS IC-FS
    best = ic.loc[ic["IUS"].idxmax()]
    ax.scatter([best["AR"]], [best["f1"]], s=300, marker="*",
                edgecolor="red", facecolor="gold", linewidth=2,
                label=f"Best IUS (α={best['alpha']:.2f})", zorder=4)

    # Baselines
    colors = {"NSGA-II-MOFS":"#d62728", "StabilitySelection":"#ff7f0e", "Boruta":"#9467bd"}
    markers = {"NSGA-II-MOFS":"s", "StabilitySelection":"^", "Boruta":"D"}
    for _, r in b_df.iterrows():
        if pd.notna(r.get("f1", np.nan)):
            ax.scatter([r["AR"]], [r["f1"]], s=150,
                        marker=markers.get(r["method"],"o"),
                        color=colors.get(r["method"],"gray"),
                        edgecolor="k", linewidth=1.5, zorder=5,
                        label=r["method"])

    # IC-FS(-temporal) with DRE (realistic)
    d = pd.read_csv("dre_mat_h0.csv")
    row = d[d["config"]=="IC-FS(-temporal)"].iloc[0]
    ax.scatter([row["AR"]], [row["f1_deploy"]], s=200, marker="X",
                color="crimson", edgecolor="k", linewidth=1.5, zorder=6,
                label="IC-FS(-temporal) under DRE")
    ax.scatter([row["AR"]], [row["f1_nomask"]], s=200, marker="x",
                color="orange", linewidth=3, zorder=6,
                label="IC-FS(-temporal) paper-style (leaks)")

    # Dotted line showing the drop
    ax.annotate("", xy=(row["AR"], row["f1_deploy"]), xytext=(row["AR"], row["f1_nomask"]),
                 arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax.text(row["AR"]+0.01, (row["f1_deploy"]+row["f1_nomask"])/2,
             f"−33.3 F1\n(leakage)", color="red", fontsize=10, fontweight="bold")

    ax.set_xlabel("Actionability Ratio (AR)")
    ax.set_ylabel("F1 score (%)")
    ax.set_title("Figure 1 — Actionability–F1 trade-off on UCI-Math at t=0\n"
                  "IC-FS α-sweep vs external baselines")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_xlim(0.1, 1.02)
    ax.set_ylim(55, 95)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_pareto.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR}/fig1_pareto.png")


# ── FIG 2: Sensitivity heatmap ────────────────────────────────────────────
def fig2_sensitivity_heatmap():
    pv = pd.read_csv("sens_math_h1_pivot_IUS.csv", index_col="w_mid")
    pv_stab = pd.read_csv("sens_math_h1_pivot_Stab.csv", index_col="w_mid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(pv.values, aspect="auto", cmap="RdYlGn", vmin=40, vmax=75)
    ax1.set_xticks(range(len(pv.columns)))
    ax1.set_xticklabels(pv.columns)
    ax1.set_yticks(range(len(pv.index)))
    ax1.set_yticklabels([f"{x:.1f}" for x in pv.index])
    ax1.set_xlabel("top_k")
    ax1.set_ylabel("$w_{mid}$ (Tier-2 weight)")
    ax1.set_title("Figure 2a — IUS sensitivity (UCI-Math, t=1)")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            ax1.text(j, i, f"{pv.values[i,j]:.1f}",
                      ha="center", va="center",
                      color="black" if pv.values[i,j] > 55 else "white", fontsize=11)
    plt.colorbar(im1, ax=ax1, label="IUS (%)")

    # Mark optimum
    best_i, best_j = np.unravel_index(pv.values.argmax(), pv.values.shape)
    ax1.add_patch(plt.Rectangle((best_j-0.5, best_i-0.5), 1, 1,
                                   fill=False, edgecolor="blue", lw=3))

    im2 = ax2.imshow(pv_stab.values, aspect="auto", cmap="Blues", vmin=0.6, vmax=1.0)
    ax2.set_xticks(range(len(pv_stab.columns)))
    ax2.set_xticklabels(pv_stab.columns)
    ax2.set_yticks(range(len(pv_stab.index)))
    ax2.set_yticklabels([f"{x:.1f}" for x in pv_stab.index])
    ax2.set_xlabel("top_k")
    ax2.set_ylabel("$w_{mid}$ (Tier-2 weight)")
    ax2.set_title("Figure 2b — Jaccard stability")
    for i in range(pv_stab.shape[0]):
        for j in range(pv_stab.shape[1]):
            ax2.text(j, i, f"{pv_stab.values[i,j]:.2f}",
                      ha="center", va="center",
                      color="black" if pv_stab.values[i,j] < 0.85 else "white",
                      fontsize=11)
    plt.colorbar(im2, ax=ax2, label="Jaccard")

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR}/fig2_sensitivity.png")


# ── FIG 3: DRE paired boxplot ─────────────────────────────────────────────
def fig3_dre_boxplot():
    m = pd.read_csv("dre_multi_math_h0.csv")
    p = pd.read_csv("dre_multi_por_h0.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, df, title in [(ax1, m, "UCI-Math, t=0"),
                            (ax2, p, "UCI-Portuguese, t=0")]:
        data = [
            df["f1_full_paper"].values,   df["f1_notemp_paper"].values,
            df["f1_full_deploy"].values,  df["f1_notemp_deploy"].values,
        ]
        positions = [1, 2, 4, 5]
        labels = ["IC-FS\nfull\n(paper)", "IC-FS\n-temporal\n(paper, LEAKS)",
                   "IC-FS\nfull\n(DRE)", "IC-FS\n-temporal\n(DRE)"]
        colors = ["#2ca02c", "#d62728", "#2ca02c", "#d62728"]
        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                          showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        # Overlay points
        for pos, vals in zip(positions, data):
            ax.scatter([pos]*len(vals), vals, s=30, color="k", alpha=0.6, zorder=3)
        # Paired lines
        for i in range(len(df)):
            ax.plot([positions[0], positions[1]],
                     [data[0][i], data[1][i]], "-", alpha=0.3, color="gray")
            ax.plot([positions[2], positions[3]],
                     [data[2][i], data[3][i]], "-", alpha=0.3, color="gray")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("F1 score (%)")
        ax.set_title(title)
        ax.axvline(3, color="k", linestyle="--", alpha=0.5)
        ax.text(1.5, ax.get_ylim()[1]*0.98, "Paper evaluation",
                 ha="center", fontsize=9, fontweight="bold")
        ax.text(4.5, ax.get_ylim()[1]*0.98, "Deployment evaluation",
                 ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Figure 3 — Paper-style vs Deployment-Realistic F1 (n=8 seeds)",
                   fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_dre_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR}/fig3_dre_boxplot.png")


# ── FIG 4: IUS decomposition stacked/grouped bar ────────────────────────
def fig4_ius_decomposition():
    # Combine ablation + baselines at horizon=0 for both datasets
    rows = []
    abl = pd.read_csv("ablation_consolidated.csv")
    abl0 = abl[abl["horizon"]==0]
    for _, r in abl0.iterrows():
        if r["config"] in ["IC-FS(full)", "IC-FS(-temporal)", "HardFilter+DE-FS"]:
            # IUS = F1 × AR × TVS — but use the CORRECT TVS from ablation CSV
            ius_check = (r["f1"]/100) * r["AR"] * r["TVS"]
            rows.append({"dataset": r["dataset"], "method": r["config"],
                          "F1": r["f1"]/100, "AR": r["AR"], "TVS": r["TVS"],
                          "IUS": ius_check})
    for ds, fn in [("math","baselines_math_h0.csv"), ("por","baselines_por_h0.csv")]:
        bdf = pd.read_csv(fn)
        for _, r in bdf.iterrows():
            if pd.notna(r.get("f1", np.nan)):
                ius_check = (r["f1"]/100) * r["AR"] * r["TVS"]
                rows.append({"dataset": ds, "method": r["method"],
                              "F1": r["f1"]/100, "AR": r["AR"], "TVS": r["TVS"],
                              "IUS": ius_check})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, ds_key, title in [(axes[0], "math", "UCI-Math at t=0"),
                                (axes[1], "por", "UCI-Portuguese at t=0")]:
        sub = df[df["dataset"]==ds_key].sort_values("IUS", ascending=False)
        methods = sub["method"].tolist()
        n = len(methods)
        x = np.arange(n)
        w = 0.22
        b1 = ax.bar(x - 1.5*w, sub["F1"], w, label="F1", color="#1f77b4")
        b2 = ax.bar(x - 0.5*w, sub["AR"], w, label="AR", color="#ff7f0e")
        b3 = ax.bar(x + 0.5*w, sub["TVS"], w, label="TVS", color="#2ca02c")
        b4 = ax.bar(x + 1.5*w, sub["IUS"], w, label="IUS = F1·AR·TVS",
                     color="#d62728", edgecolor="k", linewidth=1.5)

        for bar, v in zip(b4, sub["IUS"]):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.015, f"{v*100:.1f}",
                     ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("IC-FS","IC").replace("(-","\n(-").replace("HardFilter+DE-FS","HardFilter\n+DE-FS")
                              .replace("StabilitySelection","Stability\nSelection")
                              .replace("NSGA-II-MOFS","NSGA-II")
                             for m in methods], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Metric value (0-1)")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=9)

    fig.suptitle("Figure 4 — IUS decomposition: F1, AR, TVS, and their product",
                   fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig4_ius_decomp.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT_DIR}/fig4_ius_decomp.png")


if __name__ == "__main__":
    fig1_pareto()
    fig2_sensitivity_heatmap()
    fig3_dre_boxplot()
    fig4_ius_decomposition()
    print("\nAll figures saved to:", OUT_DIR)
