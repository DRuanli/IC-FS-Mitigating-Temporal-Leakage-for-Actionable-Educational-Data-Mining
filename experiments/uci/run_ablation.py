"""
Fast ablation runner:
  - α-sweep WITHOUT stability (fast)
  - Identify best-IUS α
  - Compute stability ONLY at best α (per config, per horizon)
"""
import warnings
warnings.filterwarnings("ignore")

import time
import itertools
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

from utils_data import load_and_split
from ic_fs_v2 import (
    TAXONOMY_UCI, Tier, _resolve_parent,
    feature_scores_for_selection, ic_fs_select,
    actionability_ratio, actionability_ratio_available,
    temporal_validity_score,
    compute_ius, compute_ius_geo,
    compute_ius_deploy, compute_ius_paper,
    filter_by_horizon,
)

RANDOM_STATE = 42


def _eval_subset(X_tr, y_tr, X_te, y_te, sel_idx, cv_folds=5):
    rf = RandomForestClassifier(n_estimators=60, random_state=RANDOM_STATE, n_jobs=1)
    X_tr_s = X_tr[:, sel_idx]
    X_te_s = X_te[:, sel_idx]
    rf.fit(X_tr_s, y_tr)
    y_pred = rf.predict(X_te_s)
    y_prob = rf.predict_proba(X_te_s)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="weighted", zero_division=0)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv = cross_val_score(clone(rf), X_tr_s, y_tr, cv=skf,
                          scoring="f1_weighted", n_jobs=1)
    return {"accuracy": acc, "f1": f1, "y_prob": y_prob,
             "cv_mean": cv.mean(), "cv_std": cv.std()}


def bootstrap_stability_single(X, y, feature_names, alpha, top_k,
                                 n_bootstrap=15, seed=2026, taxonomy=None):
    """Compute Jaccard stability at a SINGLE α via bootstrap."""
    tax = taxonomy if taxonomy is not None else TAXONOMY_UCI
    rng = np.random.RandomState(seed)
    selected_sets = []
    n = len(y)
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y[idx])) < 2:
            continue
        sdf = feature_scores_for_selection(X[idx], y[idx], feature_names, tax)
        selected_sets.append(set(ic_fs_select(sdf, alpha, top_k)))
    if len(selected_sets) < 2:
        return 0.0
    pairs = list(itertools.combinations(range(len(selected_sets)), 2))
    jacc = []
    for i, j in pairs:
        A, B = selected_sets[i], selected_sets[j]
        inter, union = len(A & B), len(A | B)
        jacc.append(inter/union if union > 0 else 1.0)
    return float(np.mean(jacc))


def run_config_full_fast(X_tr, y_tr, X_te, y_te, names, horizon, top_k=12,
                            alpha_values=None, taxonomy=None, n_bootstrap=15):
    if alpha_values is None:
        alpha_values = np.linspace(0, 1, 11).tolist()
    tax = taxonomy if taxonomy is not None else TAXONOMY_UCI

    available = filter_by_horizon(names, horizon, tax)
    avail_idx = [names.index(f) for f in available]
    X_tr_a = X_tr[:, avail_idx]
    X_te_a = X_te[:, avail_idx]

    score_df = feature_scores_for_selection(X_tr_a, y_tr, available, tax)

    best_row = None
    for alpha in alpha_values:
        selected = ic_fs_select(score_df, alpha, min(top_k, len(available)))
        sel_local = [available.index(f) for f in selected]
        ev = _eval_subset(X_tr_a, y_tr, X_te_a, y_te, sel_local)
        ar  = actionability_ratio(selected, tax)
        ar_avail = actionability_ratio_available(selected, horizon, tax)
        tvs = temporal_validity_score(selected, horizon, tax)
        ius_paper = compute_ius_paper(ev["f1"], selected, horizon, tax)
        ius_deploy = compute_ius_deploy(ev["f1"], selected, horizon, tax)
        row = {
            "config": "IC-FS(full)", "horizon": horizon, "alpha_best": alpha,
            "accuracy": ev["accuracy"]*100, "f1": ev["f1"]*100,
            "AR": ar, "AR_available": ar_avail, "TVS": tvs,
            "IUS_paper": ius_paper*100, "IUS_deploy": ius_deploy*100,
            "IUS": ius_deploy*100,  # Use new metric as primary
            "n_features": len(selected),
            "cv_mean": ev["cv_mean"]*100, "cv_std": ev["cv_std"]*100,
            "selected": "|".join(selected), "_alpha": alpha,
            "_selected_list": selected,
        }
        if best_row is None or row["IUS_deploy"] > best_row["IUS_deploy"]:
            best_row = row

    # Stability only at best α
    best_row["stability"] = bootstrap_stability_single(
        X_tr_a, y_tr, available, best_row["_alpha"], top_k, n_bootstrap,
        seed=RANDOM_STATE + horizon, taxonomy=tax)
    return best_row


def run_config_no_temporal_fast(X_tr, y_tr, X_te, y_te, names, horizon, top_k=12,
                                  alpha_values=None, taxonomy=None, n_bootstrap=15):
    if alpha_values is None:
        alpha_values = np.linspace(0, 1, 11).tolist()
    tax = taxonomy if taxonomy is not None else TAXONOMY_UCI

    # NO temporal filter
    score_df = feature_scores_for_selection(X_tr, y_tr, names, tax)

    best_row = None
    for alpha in alpha_values:
        selected = ic_fs_select(score_df, alpha, top_k)
        sel_local = [names.index(f) for f in selected]
        ev = _eval_subset(X_tr, y_tr, X_te, y_te, sel_local)
        ar  = actionability_ratio(selected, tax)
        ar_avail = actionability_ratio_available(selected, horizon, tax)
        tvs = temporal_validity_score(selected, horizon, tax)
        ius_paper = compute_ius_paper(ev["f1"], selected, horizon, tax)
        ius_deploy = compute_ius_deploy(ev["f1"], selected, horizon, tax)
        row = {
            "config": "IC-FS(-temporal)", "horizon": horizon, "alpha_best": alpha,
            "accuracy": ev["accuracy"]*100, "f1": ev["f1"]*100,
            "AR": ar, "AR_available": ar_avail, "TVS": tvs,
            "IUS_paper": ius_paper*100, "IUS_deploy": ius_deploy*100,
            "IUS": ius_deploy*100,  # Use new metric as primary
            "n_features": len(selected),
            "cv_mean": ev["cv_mean"]*100, "cv_std": ev["cv_std"]*100,
            "has_G1_G2": any(f in selected for f in ["G1", "G2"]),
            "selected": "|".join(selected),
            "_alpha": alpha, "_selected_list": selected,
        }
        if best_row is None or row["IUS_deploy"] > best_row["IUS_deploy"]:
            best_row = row
    # For this config, stability on full feature space (no filter)
    best_row["stability"] = bootstrap_stability_single(
        X_tr, y_tr, names, best_row["_alpha"], top_k, n_bootstrap,
        seed=RANDOM_STATE + 100 + horizon, taxonomy=tax)
    return best_row


def run_config_no_action_fast(X_tr, y_tr, X_te, y_te, names, horizon, top_k=12,
                                 taxonomy=None, n_bootstrap=15):
    tax = taxonomy if taxonomy is not None else TAXONOMY_UCI
    available = filter_by_horizon(names, horizon, tax)
    avail_idx = [names.index(f) for f in available]
    X_tr_a = X_tr[:, avail_idx]
    X_te_a = X_te[:, avail_idx]

    score_df = feature_scores_for_selection(X_tr_a, y_tr, available, tax).copy()
    score_df["actionability"] = 1.0  # degenerate — selection = pure pred

    selected = ic_fs_select(score_df, alpha=1.0, top_k=min(top_k, len(available)))
    sel_local = [available.index(f) for f in selected]
    ev = _eval_subset(X_tr_a, y_tr, X_te_a, y_te, sel_local)
    ar  = actionability_ratio(selected, tax)
    ar_avail = actionability_ratio_available(selected, horizon, tax)
    tvs = temporal_validity_score(selected, horizon, tax)
    ius_paper = compute_ius_paper(ev["f1"], selected, horizon, tax)
    ius_deploy = compute_ius_deploy(ev["f1"], selected, horizon, tax)

    # Stability — still measure (with actionability=1.0)
    rng = np.random.RandomState(RANDOM_STATE + 200 + horizon)
    selected_sets = []
    n = len(y_tr)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_tr[idx])) < 2:
            continue
        sdf = feature_scores_for_selection(X_tr_a[idx], y_tr[idx], available, tax)
        sdf = sdf.copy(); sdf["actionability"] = 1.0
        selected_sets.append(set(ic_fs_select(sdf, 1.0, top_k)))
    pairs = list(itertools.combinations(range(len(selected_sets)), 2))
    jacc = [len(A & B)/len(A | B) if len(A | B) > 0 else 1.0
             for i, j in pairs for A, B in [(selected_sets[i], selected_sets[j])]]
    stab = float(np.mean(jacc)) if jacc else 0.0

    return {
        "config": "IC-FS(-action)", "horizon": horizon, "alpha_best": np.nan,
        "accuracy": ev["accuracy"]*100, "f1": ev["f1"]*100,
        "AR": ar, "AR_available": ar_avail, "TVS": tvs,
        "IUS_paper": ius_paper*100, "IUS_deploy": ius_deploy*100,
        "IUS": ius_deploy*100,  # Use new metric as primary
        "n_features": len(selected),
        "cv_mean": ev["cv_mean"]*100, "cv_std": ev["cv_std"]*100,
        "stability": stab,
        "selected": "|".join(selected),
    }


def run_config_hardfilter_fast(X_tr, y_tr, X_te, y_te, names, horizon, top_k=12,
                                  taxonomy=None, n_bootstrap=15):
    tax = taxonomy if taxonomy is not None else TAXONOMY_UCI
    available = filter_by_horizon(names, horizon, tax)
    allowed = {Tier.PRE_SEMESTER, Tier.MID_SEMESTER}
    filtered = []
    for f in available:
        prof = _resolve_parent(f, tax)
        if prof is not None and prof.tier in allowed:
            filtered.append(f)
    if not filtered:
        return {"config":"HardFilter+DE-FS","horizon":horizon,
                 "IUS":np.nan,"note":"empty feature set"}
    top_k_eff = min(top_k, len(filtered))
    avail_idx = [names.index(f) for f in filtered]
    X_tr_a = X_tr[:, avail_idx]
    X_te_a = X_te[:, avail_idx]

    def _defs_select(Xtr, ytr, nm):
        scaler = MinMaxScaler(); Xnn = scaler.fit_transform(Xtr)
        c,_ = chi2(Xnn, ytr); c = np.nan_to_num(c, nan=0.0)
        mi = mutual_info_classif(Xtr, ytr, random_state=RANDOM_STATE)
        co = np.array([abs(np.corrcoef(Xtr[:,j], ytr)[0,1]) for j in range(Xtr.shape[1])])
        co = np.nan_to_num(co, nan=0.0)
        def nm_(v): return (v-v.min())/(v.max()-v.min()+1e-10)
        ens = (nm_(c)+nm_(mi)+nm_(co))/3
        top = np.argsort(ens)[::-1][:top_k_eff]
        return [nm[i] for i in top], top

    selected, top_idx = _defs_select(X_tr_a, y_tr, filtered)
    ev = _eval_subset(X_tr_a, y_tr, X_te_a, y_te, list(top_idx))
    ar  = actionability_ratio(selected, tax)
    ar_avail = actionability_ratio_available(selected, horizon, tax)
    tvs = temporal_validity_score(selected, horizon, tax)
    ius_paper = compute_ius_paper(ev["f1"], selected, horizon, tax)
    ius_deploy = compute_ius_deploy(ev["f1"], selected, horizon, tax)

    # Stability
    rng = np.random.RandomState(RANDOM_STATE + 300 + horizon)
    sel_sets = []
    n = len(y_tr)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_tr[idx])) < 2: continue
        sel_b, _ = _defs_select(X_tr_a[idx], y_tr[idx], filtered)
        sel_sets.append(set(sel_b))
    pairs = list(itertools.combinations(range(len(sel_sets)), 2))
    jacc = [len(A & B)/len(A | B) if len(A|B)>0 else 1.0
             for i,j in pairs for A,B in [(sel_sets[i], sel_sets[j])]]
    stab = float(np.mean(jacc)) if jacc else 0.0

    return {
        "config": "HardFilter+DE-FS", "horizon": horizon, "alpha_best": np.nan,
        "accuracy": ev["accuracy"]*100, "f1": ev["f1"]*100,
        "AR": ar, "AR_available": ar_avail, "TVS": tvs,
        "IUS_paper": ius_paper*100, "IUS_deploy": ius_deploy*100,
        "IUS": ius_deploy*100,  # Use new metric as primary
        "n_features": len(selected),
        "cv_mean": ev["cv_mean"]*100, "cv_std": ev["cv_std"]*100,
        "stability": stab,
        "selected": "|".join(selected),
    }


# ─── Orchestration ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*90)
    print("FAST ABLATION — UCI Math dataset")
    print("="*90)
    X_tr, X_te, y_tr, y_te, names = load_and_split("student-mat.csv",
                                                      test_size=0.2, random_state=42)
    print(f"N_train={len(y_tr)} N_test={len(y_te)} N_features={len(names)}")
    alpha_grid = np.linspace(0, 1, 11).tolist()
    N_BOOT = 15

    all_rows = []
    t0 = time.time()
    for h in [0, 1, 2]:
        print(f"\n>>> Horizon t={h}")
        t_h = time.time()
        r1 = run_config_full_fast(X_tr, y_tr, X_te, y_te, names, h, 12,
                                    alpha_grid, n_bootstrap=N_BOOT)
        print(f"  C1 full done in {time.time()-t_h:.1f}s | IUS={r1['IUS']:.2f} α*={r1['_alpha']:.2f}")

        t_h = time.time()
        r2 = run_config_no_temporal_fast(X_tr, y_tr, X_te, y_te, names, h, 12,
                                            alpha_grid, n_bootstrap=N_BOOT)
        print(f"  C2 -temporal done in {time.time()-t_h:.1f}s | IUS={r2['IUS']:.2f} G1/G2={r2['has_G1_G2']}")

        t_h = time.time()
        r3 = run_config_no_action_fast(X_tr, y_tr, X_te, y_te, names, h, 12,
                                         n_bootstrap=N_BOOT)
        print(f"  C3 -action done in {time.time()-t_h:.1f}s | IUS={r3['IUS']:.2f}")

        t_h = time.time()
        r4 = run_config_hardfilter_fast(X_tr, y_tr, X_te, y_te, names, h, 12,
                                           n_bootstrap=N_BOOT)
        print(f"  C4 hardfilter done in {time.time()-t_h:.1f}s | IUS={r4['IUS']:.2f}")

        for r in [r1, r2, r3, r4]:
            r.pop("_alpha", None); r.pop("_selected_list", None)
            all_rows.append(r)

    print(f"\nTotal runtime: {time.time()-t0:.1f}s")
    df = pd.DataFrame(all_rows)
    df.to_csv("ablation_uci_math_fast.csv", index=False)

    print("\n" + "="*90)
    print("ABLATION RESULTS — UCI Math (Table 1)")
    print("="*90)
    cols = ["config","horizon","alpha_best","accuracy","f1","AR","TVS","IUS",
             "n_features","stability","cv_mean"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}" if isinstance(x,float) else str(x)))
