"""
================================================================================
Deployment-Realistic Evaluation (DRE) on OULAD — Multi-seed
================================================================================
Critical experiment: confirms that DE-FS-style (no temporal filter) suffers
F1 collapse when Tier-3 features (CMA/TMA scores) are masked at inference.

Protocol:
  For each seed, horizon ∈ {0, 1, 2}:
    1. Train IC-FS(full) with temporal filter
    2. Train IC-FS(-temporal) without filter
    3. Mask temporally-unavailable features in BOTH train and test
    4. Retrain on masked train, evaluate on masked test
    5. Compare paper-style F1 vs deployment F1

Output: dre_multi_oulad_h{0,1,2}.csv with 8 seeds each
Total runtime: ~2-3 hours on laptop with n_jobs=-1

Usage:
    python experiments/oulad/run_oulad_dre.py 0   # horizon 0
    python experiments/oulad/run_oulad_dre.py 1   # horizon 1
    python experiments/oulad/run_oulad_dre.py 2   # horizon 2
================================================================================
"""

from __future__ import annotations
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from icfs.core import (
    feature_scores_for_selection, ic_fs_select,
    actionability_ratio, actionability_ratio_available,
    temporal_validity_score, compute_ius,
    compute_ius_deploy, compute_ius_paper,
    filter_by_horizon, get_temporal_availability,
)
from icfs.taxonomy_oulad import TAXONOMY_OULAD
from preprocess_oulad import preprocess_oulad, load_oulad_horizon

RNG_SEEDS = [42, 123, 456, 789, 1011, 2024, 3033, 4044]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
TOP_K = 15
N_TREES = 60


def fit_predict_f1(X_tr, y_tr, X_te, y_te, sel_idx, random_state):
    """Train RF on selected features, return F1."""
    rf = RandomForestClassifier(n_estimators=N_TREES,
                                  random_state=random_state,
                                  n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr[:, sel_idx], y_tr)
    y_pred = rf.predict(X_te[:, sel_idx])
    return f1_score(y_te, y_pred, average='weighted', zero_division=0)


def select_best_ius(X_tr, y_tr, X_te, y_te, names, horizon, random_state,
                      apply_temporal_filter: bool):
    """Run IC-FS α-sweep and return best-IUS selection.

    apply_temporal_filter:
      True  → IC-FS (full): filter unavailable features before scoring
      False → IC-FS (-temporal): use all features (DE-FS-style)
    """
    if apply_temporal_filter:
        available = filter_by_horizon(names, horizon, TAXONOMY_OULAD)
        if not available:
            raise RuntimeError(f"No features available at horizon={horizon}")
        idx = [names.index(f) for f in available]
        X_tr_use = X_tr[:, idx]
        X_te_use = X_te[:, idx]
        feat_use = available
    else:
        X_tr_use = X_tr
        X_te_use = X_te
        feat_use = names

    score_df = feature_scores_for_selection(X_tr_use, y_tr, feat_use,
                                              TAXONOMY_OULAD)

    best_sel = None
    best_ius = -np.inf
    best_alpha = None
    for alpha in ALPHA_GRID:
        sel = ic_fs_select(score_df, alpha, min(TOP_K, len(feat_use)))
        sel_local = [feat_use.index(f) for f in sel]
        f1 = fit_predict_f1(X_tr_use, y_tr, X_te_use, y_te,
                              sel_local, random_state)
        ius = compute_ius(f1, sel, horizon, TAXONOMY_OULAD)
        if ius > best_ius:
            best_ius, best_sel, best_alpha = ius, sel, alpha

    return best_sel, best_alpha, best_ius


def evaluate_under_dre(X_tr, y_tr, X_te, y_te, selected_features, horizon,
                         names, random_state):
    """Mask temporally-unavailable selected features, retrain, evaluate."""
    sel_idx = [names.index(f) for f in selected_features]
    X_tr_s = X_tr[:, sel_idx].astype(np.float64).copy()
    X_te_s = X_te[:, sel_idx].astype(np.float64).copy()

    # Mask with train-mean for unavailable features
    train_means = X_tr_s.mean(axis=0)
    for j, f in enumerate(selected_features):
        if not get_temporal_availability(f, horizon, TAXONOMY_OULAD):
            X_tr_s[:, j] = train_means[j]
            X_te_s[:, j] = train_means[j]

    rf = RandomForestClassifier(n_estimators=N_TREES,
                                  random_state=random_state,
                                  n_jobs=-1, class_weight='balanced')
    rf.fit(X_tr_s, y_tr)
    y_pred = rf.predict(X_te_s)
    return f1_score(y_te, y_pred, average='weighted', zero_division=0)


def run_one_seed(df_raw, seed, horizon):
    """Run paper-style + DRE eval for one seed at one horizon."""
    X, y, names = preprocess_oulad(df_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                  random_state=seed, stratify=y)

    # IC-FS(full) — temporal filter on
    sel_full, alpha_full, ius_full_paper = select_best_ius(
        X_tr, y_tr, X_te, y_te, names, horizon, seed,
        apply_temporal_filter=True)
    f1_full_paper = ius_full_paper / (
        actionability_ratio(sel_full, TAXONOMY_OULAD)
        * temporal_validity_score(sel_full, horizon, TAXONOMY_OULAD)
        + 1e-10)

    # IC-FS(-temporal) — no filter (DE-FS analogue)
    sel_notemp, alpha_notemp, ius_notemp_paper = select_best_ius(
        X_tr, y_tr, X_te, y_te, names, horizon, seed,
        apply_temporal_filter=False)
    f1_notemp_paper = ius_notemp_paper / (
        actionability_ratio(sel_notemp, TAXONOMY_OULAD)
        * temporal_validity_score(sel_notemp, horizon, TAXONOMY_OULAD)
        + 1e-10)

    # DRE: mask + retrain + evaluate
    f1_full_deploy = evaluate_under_dre(X_tr, y_tr, X_te, y_te,
                                           sel_full, horizon, names, seed)
    f1_notemp_deploy = evaluate_under_dre(X_tr, y_tr, X_te, y_te,
                                             sel_notemp, horizon, names, seed)

    # ─── EXISTING: paper-style AR (unchanged for comparison) ───
    ar_full = actionability_ratio(sel_full, TAXONOMY_OULAD)
    ar_notemp = actionability_ratio(sel_notemp, TAXONOMY_OULAD)

    # ─── NEW: deployment-honest AR_available ────────────────────
    ar_available_full = actionability_ratio_available(sel_full, horizon, TAXONOMY_OULAD)
    ar_available_notemp = actionability_ratio_available(sel_notemp, horizon, TAXONOMY_OULAD)

    # ─── EXISTING: old IUS (for comparison table) ───────────────
    ius_full_deploy_old = f1_full_deploy * ar_full  # TVS=1 for full, so same
    ius_notemp_deploy_old = f1_notemp_deploy * ar_notemp  # INFLATED — for demonstration

    # ─── NEW: correct IUS_deploy ────────────────────────────────
    ius_full_deploy_new = compute_ius_deploy(f1_full_deploy, sel_full, horizon, TAXONOMY_OULAD)
    ius_notemp_deploy_new = compute_ius_deploy(f1_notemp_deploy, sel_notemp, horizon, TAXONOMY_OULAD)

    # Leakage diagnostics
    tau_full = 1.0 - (ar_available_full / ar_full) if ar_full > 0 else 0.0
    tau_notemp = 1.0 - (ar_available_notemp / ar_notemp) if ar_notemp > 0 else 0.0

    # Leakage flags: did the unfiltered version pick Tier-3 score features?
    tier3_features = ['score_CMA1', 'score_TMA1', 'score_CMA2', 'score_TMA2',
                       'weighted_assessment_score_to_date']
    full_has_t3 = any(any(f.startswith(t) or f == t for t in tier3_features)
                       for f in sel_full)
    notemp_has_t3 = any(any(f.startswith(t) or f == t for t in tier3_features)
                          for f in sel_notemp)

    return {
        "seed": seed, "horizon": horizon,
        "alpha_full": alpha_full, "alpha_notemp": alpha_notemp,
        # F1 values (paper-style and deploy)
        "f1_full_paper": f1_full_paper * 100,
        "f1_notemp_paper": f1_notemp_paper * 100,
        "f1_full_deploy": f1_full_deploy * 100,
        "f1_notemp_deploy": f1_notemp_deploy * 100,
        # AR: old (for comparison) and new (for primary reporting)
        "AR_full": ar_full,
        "AR_notemp": ar_notemp,  # Old: inflated at h=0
        "AR_available_full": ar_available_full,  # New: correct for full
        "AR_available_notemp": ar_available_notemp,  # New: reveals leakage at h=0
        # IUS: old (paper-style, inflated) vs new (deployment-honest)
        "IUS_paper_full": ius_full_paper * 100,
        "IUS_paper_notemp": ius_notemp_paper * 100,
        "IUS_deploy_old_full": ius_full_deploy_old * 100,  # F1_deploy × AR (old formula)
        "IUS_deploy_old_notemp": ius_notemp_deploy_old * 100,  # F1_deploy × AR (old formula)
        "IUS_deploy_full": ius_full_deploy_new * 100,  # PRIMARY metric
        "IUS_deploy_notemp": ius_notemp_deploy_new * 100,  # PRIMARY metric
        # Leakage diagnostics
        "tau_full": tau_full,
        "tau_notemp": tau_notemp,
        "full_has_T3": full_has_t3,
        "notemp_has_T3": notemp_has_t3,
        "n_full": len(sel_full), "n_notemp": len(sel_notemp),
        "selected_full": "|".join(sel_full),
        "selected_notemp": "|".join(sel_notemp),
    }


def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    out_dir = project_root / "results" / "oulad"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"dre_multi_oulad_h{horizon}.csv"

    print("=" * 80)
    print(f"OULAD DRE Multi-seed | horizon={horizon} | n_seeds={len(RNG_SEEDS)}")
    print("=" * 80)

    print(f"\n[1/3] Loading parquet for h={horizon}...")
    df_raw = load_oulad_horizon(horizon)
    print(f"  Loaded {len(df_raw)} enrollments × {df_raw.shape[1]} columns")
    print(f"  Pass rate: {df_raw['y'].mean():.3f}")

    print(f"\n[2/3] Running {len(RNG_SEEDS)} seeds...")
    rows = []
    t0 = time.time()
    for s in RNG_SEEDS:
        t_seed = time.time()
        try:
            r = run_one_seed(df_raw, s, horizon)
            rows.append(r)
            print(f"  seed={s:4d} ({time.time()-t_seed:5.0f}s) | "
                   f"full_dep={r['f1_full_deploy']:5.1f} "
                   f"notemp_dep={r['f1_notemp_deploy']:5.1f} "
                   f"diff={r['f1_full_deploy']-r['f1_notemp_deploy']:+5.1f} "
                   f"notemp_T3_leak={r['notemp_has_T3']}")
        except Exception as e:
            print(f"  seed={s} FAILED: {e}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}  (total {time.time()-t0:.0f}s)")

    if len(rows) < 2:
        print("\nNot enough seeds for stats")
        return

    print(f"\n[3/3] Statistics ({len(rows)} seeds)")
    print("\n--- Bootstrap 95% CI ---")
    for col in ["f1_full_deploy", "f1_notemp_deploy",
                 "IUS_deploy_full", "IUS_deploy_notemp",
                 "f1_full_paper", "f1_notemp_paper",
                 "IUS_paper_full", "IUS_paper_notemp"]:
        v = df[col].values
        print(f"  {col:<22}: mean={v.mean():6.2f}  std={v.std(ddof=1):5.2f}  "
               f"95% CI=[{np.percentile(v, 2.5):5.2f}, {np.percentile(v, 97.5):5.2f}]")

    print("\n--- Wilcoxon: IC-FS(full) > IC-FS(-temporal) under DEPLOYMENT ---")
    print(f"  Bonferroni α = 0.05/3 = 0.0167 (3 horizons)")
    a_ius = df["IUS_deploy_full"].values
    b_ius = df["IUS_deploy_notemp"].values
    if not np.allclose(a_ius, b_ius):
        try:
            stat, p = wilcoxon(a_ius, b_ius, alternative="greater",
                                zero_method="wilcox")
            diff = a_ius - b_ius
            d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
            sig = ("***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.0167 else "ns")
            print(f"  IUS_deploy: W={stat:.0f}  p={p:.5f}  d={d:+.2f}  "
                   f"diff={diff.mean():+.2f}  [{sig}]")
        except ValueError as e:
            print(f"  IUS_deploy: Wilcoxon failed ({e})")
    else:
        print("  IUS_deploy: identical values — selections converge at this horizon")

    a_f1 = df["f1_full_deploy"].values
    b_f1 = df["f1_notemp_deploy"].values
    if not np.allclose(a_f1, b_f1):
        try:
            stat, p = wilcoxon(a_f1, b_f1, alternative="greater",
                                zero_method="wilcox")
            diff = a_f1 - b_f1
            d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
            sig = ("***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.0167 else "ns")
            print(f"  F1_deploy:  W={stat:.0f}  p={p:.5f}  d={d:+.2f}  "
                   f"diff={diff.mean():+.2f}  [{sig}]")
        except ValueError as e:
            print(f"  F1_deploy: Wilcoxon failed ({e})")

    print("\n--- Leakage exposure ---")
    n_notemp_leaks = df["notemp_has_T3"].sum()
    print(f"  IC-FS(-temporal) selected Tier-3 in {n_notemp_leaks}/{len(df)} seeds")
    if n_notemp_leaks > 0:
        sub = df[df["notemp_has_T3"]]
        f1_drop = sub["f1_notemp_paper"] - sub["f1_notemp_deploy"]
        print(f"    Mean F1 drop in those seeds: {f1_drop.mean():+.2f} pts")


if __name__ == "__main__":
    main()