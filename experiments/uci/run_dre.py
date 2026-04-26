"""
Multi-seed DRE on UCI Math h=0 — definitive test of IC-FS(full) vs IC-FS(-temporal)
under realistic deployment (G1/G2 masked at inference).
"""
import sys, warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from utils_data import load_uci, preprocess_uci
from ic_fs_v2 import (
    TAXONOMY_UCI, Tier, _resolve_parent,
    feature_scores_for_selection, ic_fs_select,
    actionability_ratio, actionability_ratio_available,
    temporal_validity_score, compute_ius,
    compute_ius_deploy, compute_ius_paper,
    filter_by_horizon, get_temporal_availability,
)

RNG_SEEDS = [42, 123, 456, 789, 1011, 2024, 3033, 4044]
RANDOM_STATE = 42


def mask_and_eval(X_tr, y_tr, X_te, y_te, selected, horizon, names):
    """Train model with unavailable features masked (train & test consistent)."""
    sel_idx = [names.index(f) for f in selected]
    X_tr_s = X_tr[:, sel_idx]; X_te_s = X_te[:, sel_idx]
    tm = X_tr_s.mean(axis=0)
    X_tr_m = X_tr_s.copy(); X_te_m = X_te_s.copy()
    for j, f in enumerate(selected):
        if not get_temporal_availability(f, horizon):
            X_tr_m[:, j] = tm[j]
            X_te_m[:, j] = tm[j]
    rf = RandomForestClassifier(n_estimators=60, random_state=RANDOM_STATE, n_jobs=1)
    rf.fit(X_tr_m, y_tr)
    y_pred = rf.predict(X_te_m)
    return f1_score(y_te, y_pred, average="weighted", zero_division=0)


def run_one_seed(df_raw, seed, horizon):
    X, y, names = preprocess_uci(df_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                 random_state=seed, stratify=y)

    # IC-FS(full): pick via IUS sweep
    available = filter_by_horizon(names, horizon)
    avail_idx = [names.index(f) for f in available]
    X_tr_a = X_tr[:, avail_idx]; X_te_a = X_te[:, avail_idx]
    score_df = feature_scores_for_selection(X_tr_a, y_tr, available)

    best_full = None; best_ius = -np.inf
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sel = ic_fs_select(score_df, alpha, 12)
        sel_local = [available.index(f) for f in sel]
        rf = RandomForestClassifier(n_estimators=60, random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(X_tr_a[:, sel_local], y_tr)
        f1 = f1_score(y_te, rf.predict(X_te_a[:, sel_local]),
                       average="weighted", zero_division=0)
        ius = compute_ius(f1, sel, horizon)
        if ius > best_ius:
            best_ius = ius; best_full = sel

    # IC-FS(-temporal): pick without filter
    score_df_all = feature_scores_for_selection(X_tr, y_tr, names)
    best_notemp = None; best_ius_nt = -np.inf
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        sel = ic_fs_select(score_df_all, alpha, 12)
        sel_idx = [names.index(f) for f in sel]
        rf = RandomForestClassifier(n_estimators=60, random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(X_tr[:, sel_idx], y_tr)
        f1 = f1_score(y_te, rf.predict(X_te[:, sel_idx]),
                       average="weighted", zero_division=0)
        ius = compute_ius(f1, sel, horizon)
        if ius > best_ius_nt:
            best_ius_nt = ius; best_notemp = sel

    # Evaluate both under DRE masking
    f1_full_dep = mask_and_eval(X_tr, y_tr, X_te, y_te, best_full, horizon, names)
    f1_notemp_dep = mask_and_eval(X_tr, y_tr, X_te, y_te, best_notemp, horizon, names)

    # ─── EXISTING: paper-style AR (unchanged for comparison) ───
    ar_full = actionability_ratio(best_full)
    ar_notemp = actionability_ratio(best_notemp)

    # ─── NEW: deployment-honest AR_available ────────────────────
    ar_available_full = actionability_ratio_available(best_full, horizon)
    ar_available_notemp = actionability_ratio_available(best_notemp, horizon)

    # ─── EXISTING: old IUS (for comparison table) ───────────────
    ius_full_dep_old = f1_full_dep * ar_full
    ius_notemp_dep_old = f1_notemp_dep * ar_notemp

    # ─── NEW: correct IUS_deploy ────────────────────────────────
    ius_full_dep_new = compute_ius_deploy(f1_full_dep, best_full, horizon)
    ius_notemp_dep_new = compute_ius_deploy(f1_notemp_dep, best_notemp, horizon)

    # Leakage diagnostics
    tau_full = 1.0 - (ar_available_full / ar_full) if ar_full > 0 else 0.0
    tau_notemp = 1.0 - (ar_available_notemp / ar_notemp) if ar_notemp > 0 else 0.0

    # Also paper-style (no masking) for comparison
    f1_full_paper = best_ius / ar_full / temporal_validity_score(best_full, horizon) if ar_full * temporal_validity_score(best_full, horizon) > 0 else 0
    f1_notemp_paper = best_ius_nt / ar_notemp / temporal_validity_score(best_notemp, horizon) if ar_notemp * temporal_validity_score(best_notemp, horizon) > 0 else 0

    return {"seed": seed, "horizon": horizon,
             # F1 values (paper-style and deploy)
             "f1_full_paper": f1_full_paper * 100,
             "f1_notemp_paper": f1_notemp_paper * 100,
             "f1_full_deploy": f1_full_dep * 100,
             "f1_notemp_deploy": f1_notemp_dep * 100,
             # AR: old (for comparison) and new (for primary reporting)
             "AR_full": ar_full,
             "AR_notemp": ar_notemp,
             "AR_available_full": ar_available_full,
             "AR_available_notemp": ar_available_notemp,
             # IUS: old (paper-style, inflated) vs new (deployment-honest)
             "IUS_paper_full": best_ius * 100,
             "IUS_paper_notemp": best_ius_nt * 100,
             "IUS_deploy_old_full": ius_full_dep_old * 100,
             "IUS_deploy_old_notemp": ius_notemp_dep_old * 100,
             "IUS_deploy_full": ius_full_dep_new * 100,
             "IUS_deploy_notemp": ius_notemp_dep_new * 100,
             # Leakage diagnostics
             "tau_full": tau_full,
             "tau_notemp": tau_notemp,
             "full_has_G1G2": any(f in best_full for f in ["G1","G2"]),
             "notemp_has_G1G2": any(f in best_notemp for f in ["G1","G2"])}


def main():
    dataset = sys.argv[1] if len(sys.argv)>1 else "student-mat.csv"
    horizon = int(sys.argv[2]) if len(sys.argv)>2 else 0
    out = sys.argv[3] if len(sys.argv)>3 else f"dre_multi_{dataset.split('-')[1].split('.')[0]}_h{horizon}.csv"
    df_raw = load_uci(dataset)

    print(f"[{dataset}] h={horizon}, {len(RNG_SEEDS)} seeds")
    rows = []
    for s in RNG_SEEDS:
        t0 = time.time()
        r = run_one_seed(df_raw, s, horizon)
        rows.append(r)
        print(f"  seed={s} ({time.time()-t0:.0f}s): "
               f"full_dep_F1={r['f1_full_deploy']:.1f}  "
               f"notemp_dep_F1={r['f1_notemp_deploy']:.1f}  "
               f"diff={r['f1_full_deploy']-r['f1_notemp_deploy']:+.1f}  "
               f"notemp_leaks={r['notemp_has_G1G2']}")

    df = pd.DataFrame(rows); df.to_csv(out, index=False)

    print("\n--- Paper-style IUS (no masking, reveals overstated claim) ---")
    for col in ["IUS_paper_full","IUS_paper_notemp"]:
        v = df[col].values
        print(f"  {col:<20}: mean={v.mean():.2f}  95%CI=[{np.percentile(v,2.5):.2f},{np.percentile(v,97.5):.2f}]")

    print("\n--- Deployment-realistic IUS (G1/G2 masked at inference) ---")
    print("  NEW METRIC (IUS_deploy = F1_deploy × AR_available):")
    for col in ["IUS_deploy_full","IUS_deploy_notemp"]:
        v = df[col].values
        print(f"    {col:<24}: mean={v.mean():.2f}  95%CI=[{np.percentile(v,2.5):.2f},{np.percentile(v,97.5):.2f}]")

    print("\n  AR_available (deployment-honest actionability ratio):")
    for col in ["AR_available_full","AR_available_notemp"]:
        v = df[col].values
        print(f"    {col:<24}: mean={v.mean():.3f}  95%CI=[{np.percentile(v,2.5):.3f},{np.percentile(v,97.5):.3f}]")

    print("\n--- Wilcoxon: IC-FS(full) > IC-FS(-temporal) under DEPLOYMENT ---")
    a = df["IUS_deploy_full"].values
    b = df["IUS_deploy_notemp"].values
    stat, p = wilcoxon(a, b, alternative="greater", zero_method="wilcox")
    diff = a - b
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
    print(f"  W={stat:.1f}  p(one-sided greater)={p:.5f}  Cohen's d={d:+.2f}  "
           f"mean_diff={diff.mean():+.2f}")

    print("\n--- Wilcoxon: IC-FS(full) F1 > IC-FS(-temporal) F1 under DEPLOYMENT ---")
    a = df["f1_full_deploy"].values
    b = df["f1_notemp_deploy"].values
    stat, p = wilcoxon(a, b, alternative="greater", zero_method="wilcox")
    diff = a - b
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0
    print(f"  W={stat:.1f}  p(one-sided greater)={p:.5f}  Cohen's d={d:+.2f}  "
           f"mean_diff={diff.mean():+.2f}")


if __name__=="__main__": main()
