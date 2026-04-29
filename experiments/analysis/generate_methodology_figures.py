"""
experiments/analysis/generate_methodology_figures.py
=====================================================
Generate publication-ready methodology figures (Section 3) for the IC-FS
ESWA submission.  This script complements ``generate_figures_eswa.py``
(which produces the *results* figures) and follows identical typography
and palette conventions.

Outputs (saved to manuscript/figures/methodology/):
    fig_M1_architecture.pdf      — IC-FS pipeline architecture (block diagram)
    fig_M2_availability_matrix.pdf — Tier × Horizon availability heat map
    fig_M3_dre_protocol.pdf      — Paper-style vs DRE evaluation protocol
    fig_M4_ic_tradeoff.pdf       — IC-score trade-off surface across α

Each figure is also written as PNG (300 dpi) for slide use.

Usage
-----
    cd <project_root>
    python experiments/analysis/generate_methodology_figures.py

The script imports nothing from the project source tree other than what is
required for ``stat_data`` parsing (when available); it is otherwise
self-contained so that it can run before ``results/`` is populated.

Requirements: matplotlib >= 3.7, numpy.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

warnings.filterwarnings("ignore")

# ── Path setup (mirrors generate_figures_eswa.py) ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "manuscript" / "figures" / "methodology"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Typography constants (identical to generate_figures_eswa.py) ──────────────
FONT_FAMILY = "Times New Roman"
FONT_SIZE_TITLE = 10
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK = 9
FONT_SIZE_LEGEND = 9
FONT_SIZE_CAPTION = 8

LINE_WIDTH = 1.5
GRID_COLOR = "#d8d8d8"
GRID_LW = 0.6

# Wong colour-blind-safe palette (consistent with results figures)
C_BLUE = "#0072B2"
C_ORANGE = "#D55E00"
C_GREEN = "#009E73"
C_PINK = "#CC79A7"
C_YELLOW = "#F0E442"
C_GREY = "#777777"
C_LIGHTGREY = "#E0E0E0"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE_TICK,
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.labelsize": FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Helper — save figure as both PDF and PNG with consistent options
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, stem: str) -> None:
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {pdf_path.name} and {png_path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE M1 — IC-FS Pipeline Architecture (block diagram)
# ═════════════════════════════════════════════════════════════════════════════

def make_fig_architecture() -> None:
    """Single-column architecture block diagram of the IC-FS framework.

    Six numbered stages flow vertically; inputs and outputs flank the pipeline.
    The figure is dataset-agnostic — references to OULAD/UCI are deliberately
    excluded so the diagram can serve as the canonical Section 3 visual.
    """
    fig, ax = plt.subplots(figsize=(7.0, 9.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    def block(x, y, w, h, text, *, fc, ec="#000", text_fs=8.5, lw=1.0):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                             linewidth=lw, edgecolor=ec, facecolor=fc)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=text_fs, family=FONT_FAMILY)

    def arrow(x1, y1, x2, y2, *, lw=1.0, color="#000"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=12,
                                     linewidth=lw, color=color))

    # --- Inputs (left) ---
    block(0.2, 11.7, 2.4, 1.0, "Training set\n$(X_{train}, y_{train})$",
          fc="#FFFFFF", text_fs=8.2)
    block(0.2, 10.3, 2.4, 1.0, "Test set\n$(X_{test}, y_{test})$",
          fc="#FFFFFF", text_fs=8.2)
    block(0.2, 8.9, 2.4, 1.0, "Feature taxonomy\n$\\delta(\\cdot,h),\\; \\omega(\\cdot)$",
          fc=C_LIGHTGREY, text_fs=8.2)
    block(0.2, 7.5, 2.4, 1.0, "Horizon $h$",
          fc="#FFFFFF", text_fs=8.5)

    # --- Pipeline (centre) ---
    cx = 5.0
    bw = 5.4
    bh = 1.05
    stages = [
        ("STAGE 1 — Temporal filtering\n"
         "$\\mathcal{F}_{avail}(h) = \\{f \\in \\mathcal{F} : \\delta(f, h) = 1\\}$",
         12.6),
        ("STAGE 2 — Nested validation split\n"
         "$X_{train}[\\mathcal{F}_{avail}] \\to X_{inner}\\,(80\\%) + X_{val}\\,(20\\%)$",
         11.2),
        ("STAGE 3 — Ensemble predictive scoring on $X_{inner}$\n"
         "$\\mathrm{pred\\_score}(j) = \\tfrac{1}{4}\\,(\\hat\\phi_{\\chi^2}\n"
         "+\\hat\\phi_{MI}+\\hat\\phi_{corr}+\\hat\\phi_{RF})$",
         9.7),
        ("STAGE 4 — IC-score $\\alpha$-sweep, validate on $X_{val}$\n"
         "$\\mathrm{IC}(j,\\alpha) = \\alpha\\cdot\\mathrm{pred\\_score}(j) + "
         "(1-\\alpha)\\,\\omega(f_j);\\; "
         "\\alpha^* = \\arg\\max_\\alpha \\mathrm{IUS}_{val}$",
         8.1),
        ("STAGE 5 — Final training on full $X_{train}[\\mathcal{F}_{avail}]$\n"
         "Re-score, select $S^* = $ top-$k$ by $\\mathrm{IC}(\\cdot,\\alpha^*)$, fit $g$",
         6.5),
        ("STAGE 6 — Deployment-Realistic Evaluation (DRE) on $X_{test}[S^*]$\n"
         "Asymmetric mask + predict $\\Rightarrow$ "
         "$F1_{deploy},\\; AR_{available},\\; IUS_{deploy}$",
         5.0),
    ]
    stage_colors = [C_LIGHTGREY, "#FFFFFF", "#FFFFFF",
                    "#FFFFFF", "#FFFFFF", C_LIGHTGREY]
    for (txt, y), col in zip(stages, stage_colors):
        block(cx - bw / 2, y - bh / 2, bw, bh, txt, fc=col, text_fs=7.6, lw=1.1)

    # Inter-stage arrows (centre line)
    for y_top, y_bot in zip([s[1] - bh / 2 for s in stages[:-1]],
                            [s[1] + bh / 2 for s in stages[1:]]):
        arrow(cx, y_top, cx, y_bot, lw=1.0)

    # Inputs → pipeline arrows
    arrow(2.6, 12.2, cx - bw / 2, 12.6, lw=0.9, color=C_GREY)
    arrow(2.6, 10.8, cx - bw / 2, 11.2, lw=0.9, color=C_GREY)
    arrow(2.6, 9.4, cx - bw / 2, 9.7,  lw=0.9, color=C_GREY)
    arrow(2.6, 8.0, cx - bw / 2, 8.1,  lw=0.9, color=C_GREY)
    # Test set also feeds Stage 6
    arrow(2.6, 10.5, cx - bw / 2, 5.0, lw=0.9, color=C_GREY)

    # --- Outputs (bottom) ---
    out_y = 2.7
    block(0.2, out_y - 0.55, 9.6, 1.1,
          "$S^*,\\; \\alpha^*,\\; F1_{deploy},\\; AR_{available},\\; "
          "IUS_{deploy},\\; \\tau,\\; \\mathrm{Precision}@20\\%,\\; "
          "\\mathrm{Recall}@20\\%$",
          fc=C_BLUE + "33", ec=C_BLUE, lw=1.4, text_fs=9.0)
    arrow(cx, 5.0 - bh / 2, cx, out_y + 0.55, lw=1.0)

    ax.text(cx, out_y - 0.95, "Outputs", ha="center", va="top",
            fontsize=9, fontweight="bold", color=C_BLUE)

    # --- Section labels ---
    ax.text(0.2, 13.5, "Inputs", fontsize=10, fontweight="bold")
    ax.text(cx - bw / 2, 13.5, "IC-FS pipeline", fontsize=10,
            fontweight="bold")

    _save(fig, "fig_M1_architecture")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE M2 — Tier × Horizon availability matrix δ(f, h) and ω(f)
# ═════════════════════════════════════════════════════════════════════════════

def make_fig_availability_matrix() -> None:
    """Heat-map style visualisation of δ(f, h) for the four taxonomy tiers,
    annotated with the actionability weight ω of each tier.

    Dataset-agnostic by design: rows are *tiers*, not individual feature names.
    """
    tiers = ["Tier 0\n(non-actionable)",
             "Tier 1\n(pre-semester)",
             "Tier 2\n(mid-semester behavioural)",
             "Tier 3\n(past assessment scores)"]
    omegas = [0.0, 1.0, 0.7, 0.0]
    horizons = ["$h = 0$\n(course start)",
                "$h = 1$\n(25% elapsed)",
                "$h = 2$\n(50% elapsed)"]

    # δ matrix (rows: tiers, cols: horizons)  — generic pattern
    delta = np.array([
        [1, 1, 1],  # Tier 0
        [1, 1, 1],  # Tier 1
        [0, 1, 1],  # Tier 2
        [0, 1, 1],  # Tier 3 (assessment-deadline-dependent; here shown as ≥1)
    ], dtype=float)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    # Custom 2-colour map: white = 0, blue = 1
    cmap = plt.matplotlib.colors.ListedColormap(["#FFFFFF", C_BLUE + "AA"])
    ax.imshow(delta, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    # Cell annotations
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            val = int(delta[i, j])
            txt_color = "#FFFFFF" if val == 1 else "#222222"
            ax.text(j, i, f"$\\delta = {val}$",
                    ha="center", va="center",
                    fontsize=10, color=txt_color, family=FONT_FAMILY)

    # Tick labels
    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(horizons, fontsize=FONT_SIZE_TICK)
    ax.set_yticks(range(len(tiers)))
    ax.set_yticklabels(tiers, fontsize=FONT_SIZE_TICK)

    # Right-hand secondary axis: ω weights
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(tiers)))
    ax2.set_yticklabels([f"$\\omega = {w:.1f}$" for w in omegas],
                        fontsize=FONT_SIZE_TICK, color=C_ORANGE)
    ax2.tick_params(axis="y", colors=C_ORANGE, length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_ylabel("Actionability weight", color=C_ORANGE,
                   fontsize=FONT_SIZE_LABEL, rotation=270, labelpad=15)

    ax.set_xlabel("Prediction horizon", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("Taxonomy tier", fontsize=FONT_SIZE_LABEL)

    # Footnote
    fig.text(0.5, -0.02,
             "Cell value: temporal availability $\\delta(f,h)\\in\\{0,1\\}$. "
             "Right axis: actionability weight $\\omega$. "
             "Tier-3 features require the assessment deadline to have passed, "
             "which is captured per-feature in the instantiated taxonomy.",
             ha="center", va="top", fontsize=FONT_SIZE_CAPTION,
             style="italic", color="#444444")

    fig.tight_layout()
    _save(fig, "fig_M2_availability_matrix")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE M3 — Paper-style vs DRE evaluation protocol (side-by-side flow)
# ═════════════════════════════════════════════════════════════════════════════

def make_fig_dre_protocol() -> None:
    """Side-by-side flow diagram contrasting paper-style and DRE evaluation."""
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 11)
    ax.axis("off")

    def block(x, y, w, h, text, *, fc, ec="#000", text_fs=8.5, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                    linewidth=lw, edgecolor=ec, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=text_fs, family=FONT_FAMILY)

    def arrow(x1, y1, x2, y2, *, lw=1.0, color="#000"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="-|>", mutation_scale=11,
                                     linewidth=lw, color=color))

    # --- Headers ---
    ax.text(2.75, 10.5, "Paper-style evaluation",
            fontsize=10.5, fontweight="bold", ha="center")
    ax.text(2.75, 10.05, "(prior practice — $F1_{paper}$)",
            fontsize=8.5, ha="center", color="#555555", style="italic")
    ax.text(9.25, 10.5, "Deployment-Realistic Evaluation (DRE)",
            fontsize=10.5, fontweight="bold", ha="center", color=C_BLUE)
    ax.text(9.25, 10.05, "(this paper — $F1_{deploy}$)",
            fontsize=8.5, ha="center", color=C_BLUE, style="italic")

    # --- LEFT column: paper-style ---
    LX, LW = 0.5, 4.5
    block(LX, 8.6, LW, 0.9, "$X_{train}[S]$ — UNMASKED",
          fc="#FFFFFF", text_fs=9)
    block(LX, 7.1, LW, 0.9, "Train classifier $g$",
          fc=C_LIGHTGREY, text_fs=9)
    block(LX, 5.6, LW, 0.9, "$X_{test}[S]$ — UNMASKED",
          fc="#FFFFFF", text_fs=9)
    block(LX, 4.1, LW, 0.9, "Predict $\\hat{y} = g(X_{test}[S])$",
          fc=C_LIGHTGREY, text_fs=9)
    block(LX, 2.4, LW, 1.1,
          "$F1_{paper} = F1(y_{test},\\,\\hat{y})$\n"
          "Assumes every $f \\in S$ exists at deployment",
          fc=C_ORANGE + "33", ec=C_ORANGE, lw=1.3, text_fs=8.5)
    arrow(LX + LW / 2, 8.6, LX + LW / 2, 8.0)
    arrow(LX + LW / 2, 7.1, LX + LW / 2, 6.5)
    arrow(LX + LW / 2, 5.6, LX + LW / 2, 5.0)
    arrow(LX + LW / 2, 4.1, LX + LW / 2, 3.5)

    # --- RIGHT column: DRE ---
    RX, RW = 7.0, 4.5
    block(RX, 8.6, RW, 0.9, "$X_{train}[S]$ — UNMASKED",
          fc="#FFFFFF", text_fs=9)
    block(RX, 7.1, RW, 0.9, "Train classifier $g$ (identical step)",
          fc=C_LIGHTGREY, text_fs=9)
    block(RX, 5.6, RW, 0.9, "$X_{test}[S]$",
          fc="#FFFFFF", text_fs=9)
    block(RX, 4.1, RW, 1.1,
          "ASYMMETRIC MASK\n"
          "if $\\delta(f, h) = 0$:  $\\;\\tilde{X}_{test}[:,f] \\leftarrow "
          "\\bar{x}_f^{train}$",
          fc=C_BLUE + "33", ec=C_BLUE, lw=1.3, text_fs=8.5)
    block(RX, 2.6, RW, 0.9,
          "Predict $\\hat{y} = g(\\tilde{X}_{test})$",
          fc=C_LIGHTGREY, text_fs=9)
    block(RX, 0.7, RW, 1.4,
          "$F1_{deploy} = F1(y_{test},\\,\\hat{y})$\n"
          "$AR_{available}(S, h) = \\frac{1}{|S|}\\sum_{f \\in S}\n"
          "\\delta(f,h)\\,\\omega(f)$\n"
          "$IUS_{deploy} = F1_{deploy} \\cdot AR_{available}$",
          fc=C_BLUE + "33", ec=C_BLUE, lw=1.3, text_fs=8.0)
    arrow(RX + RW / 2, 8.6, RX + RW / 2, 8.0)
    arrow(RX + RW / 2, 7.1, RX + RW / 2, 6.5)
    arrow(RX + RW / 2, 5.6, RX + RW / 2, 5.2)
    arrow(RX + RW / 2, 4.1, RX + RW / 2, 3.5)
    arrow(RX + RW / 2, 2.6, RX + RW / 2, 2.1)

    # --- Equivalence note ---
    fig.text(0.5, -0.02,
             "When the temporal filter (Stage 1 of Algorithm 1) is applied "
             "upstream, $S_{unavail} = \\emptyset$ and the masking step is a "
             "no-op; consequently $F1_{deploy} \\equiv F1_{paper}$ for "
             "IC-FS (full).",
             ha="center", va="top", fontsize=FONT_SIZE_CAPTION,
             style="italic", color="#444444")

    _save(fig, "fig_M3_dre_protocol")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE M4 — IC-score trade-off surface across α
# ═════════════════════════════════════════════════════════════════════════════

def make_fig_ic_tradeoff() -> None:
    """Visualise IC(j, α) = α · pred_score + (1−α) · ω for four representative
    feature profiles, illustrating how the trade-off parameter α navigates the
    predictive–actionable continuum.
    """
    alphas = np.linspace(0.0, 1.0, 101)

    # Four representative feature profiles drawn on the [0, 1] × [0, 1] plane.
    # Each profile is (pred_score, ω) — values are illustrative archetypes.
    profiles = [
        {"name": "High-pred, low-act\n(e.g., past grade)",
         "pred": 0.95, "omega": 0.0,
         "color": C_ORANGE, "ls": "-",  "marker": "o"},
        {"name": "High-pred, high-act\n(e.g., engagement aggregate)",
         "pred": 0.85, "omega": 0.7,
         "color": C_BLUE,   "ls": "--", "marker": "s"},
        {"name": "Low-pred, high-act\n(e.g., registration timing)",
         "pred": 0.30, "omega": 1.0,
         "color": C_GREEN,  "ls": ":",  "marker": "^"},
        {"name": "Low-pred, low-act\n(e.g., demographic)",
         "pred": 0.20, "omega": 0.0,
         "color": C_PINK,   "ls": "-.", "marker": "D"},
    ]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    for p in profiles:
        ic = alphas * p["pred"] + (1.0 - alphas) * p["omega"]
        ax.plot(alphas, ic, color=p["color"], linestyle=p["ls"],
                linewidth=LINE_WIDTH, marker=p["marker"], markevery=20,
                markersize=6, label=p["name"])

    # Highlight the α-grid actually used in IC-FS
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    for a in grid:
        ax.axvline(a, color="#bbbbbb", linewidth=0.6, linestyle=":")
    ax.text(0.5, 0.04, "Discrete $\\alpha$-grid evaluated by IC-FS",
            fontsize=8, ha="center", color="#555555", style="italic",
            transform=ax.transAxes)

    ax.set_xlabel("Trade-off parameter $\\alpha$", fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel("IC score = $\\alpha\\cdot\\mathrm{pred\\_score} + "
                  "(1-\\alpha)\\,\\omega$",
                  fontsize=FONT_SIZE_LABEL)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linewidth=GRID_LW, color=GRID_COLOR, alpha=0.7)

    ax.text(-0.01, 1.04, "$\\alpha = 0$ : pure actionability",
            fontsize=8, ha="left", va="top", color=C_GREY)
    ax.text(1.01, 1.04, "$\\alpha = 1$ : pure prediction",
            fontsize=8, ha="right", va="top", color=C_GREY)

    ax.legend(loc="center right", fontsize=8, framealpha=0.9, ncol=1,
              handlelength=2.6, borderpad=0.5)

    fig.tight_layout()
    _save(fig, "fig_M4_ic_tradeoff")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating ESWA Section-3 methodology figures...")
    print(f"  Output directory: {OUT_DIR}")
    print()

    print("[Fig M1] IC-FS pipeline architecture...")
    make_fig_architecture()

    print("[Fig M2] Tier × Horizon availability matrix...")
    make_fig_availability_matrix()

    print("[Fig M3] DRE protocol (paper-style vs deployment-realistic)...")
    make_fig_dre_protocol()

    print("[Fig M4] IC-score trade-off surface...")
    make_fig_ic_tradeoff()

    print()
    print("Done. Files saved:")
    for f in sorted(OUT_DIR.glob("fig_M*.p*")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")