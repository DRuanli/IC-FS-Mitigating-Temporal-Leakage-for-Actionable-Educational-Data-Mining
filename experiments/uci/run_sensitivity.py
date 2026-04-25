"""
================================================================================
Sensitivity Analysis
================================================================================
Grid: w_mid ∈ {0.3, 0.5, 0.7, 0.9} × top_k ∈ {8, 12, 16, 20}
      for each (w_mid, top_k), run IC-FS α-sweep at horizon=1 (most useful)
      report: optimal IUS, corresponding α, Tier-2 composition rate, stability

Horizon 1 chosen because:
  - At t=0, Tier-2 features (attendance etc.) mostly unavailable → weights moot
  - At t=2, G2 dominates → weights moot for best-IUS
  - At t=1, Tier-1 + Tier-2 compete → weights matter
================================================================================
"""
import sys, warnings, time, json, os, itertools
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from utils_data import load_and_split
from ic_fs_v2 import (
    TAXONOMY_UCI, Tier, ACTIONABILITY_WEIGHTS, _resolve_parent,
    feature_scores_for_selection, ic_fs_select,
    actionability_ratio, temporal_validity_score, compute_ius,
    filter_by_horizon,
)
from run_ablation_fast import _eval_subset, bootstrap_stability_single

DATASET = sys.argv[1] if len(sys.argv) > 1 else "student-mat.csv"
HORIZON = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT = sys.argv[3] if len(sys.argv) > 3 else f"sens_{DATASET.split('-')[1].split('.')[0]}_h{HORIZON}.csv"

W_MID_GRID = [0.3, 0.5, 0.7, 0.9]
TOP_K_GRID = [8, 12, 16, 20]
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
N_BOOT = 6   # smaller boot for sensitivity — noise tolerated

X_tr, X_te, y_tr, y_te, names = load_and_split(DATASET, test_size=0.2, random_state=42)
print(f"[{DATASET}] h={HORIZON} N_tr={len(y_tr)} N_feat={len(names)}")

available = filter_by_horizon(names, HORIZON)
avail_idx = [names.index(f) for f in available]
X_tr_a = X_tr[:, avail_idx]
X_te_a = X_te[:, avail_idx]


def tier2_ratio(selected_features, taxonomy=None):
    """Tỉ lệ features Tier 2 (mid-semester) trong selected set."""
    tax = taxonomy if taxonomy is not None else TAXONOMY_UCI
    if not selected_features:
        return 0.0
    cnt = 0
    for f in selected_features:
        prof = _resolve_parent(f, tax)
        if prof is not None and prof.tier == Tier.MID_SEMESTER:
            cnt += 1
    return cnt / len(selected_features)


def run_sens_cell(w_mid, top_k):
    """One (w_mid, top_k) cell — returns best-IUS α and metrics."""
    # Temporarily override MID_SEMESTER weight
    weights_override = dict(ACTIONABILITY_WEIGHTS)
    weights_override[Tier.MID_SEMESTER] = w_mid

    # Re-compute score_df with overridden weights
    from ic_fs_v2 import get_actionability_score as _gas
    # Manual scoring with custom weights
    tax = TAXONOMY_UCI
    action_scores = []
    for f in available:
        prof = _resolve_parent(f, tax)
        if prof is None:
            action_scores.append(0.0)
        else:
            action_scores.append(weights_override[prof.tier])
    action_scores = np.array(action_scores)

    # Use existing feature_scores_for_selection for pred part
    score_df = feature_scores_for_selection(X_tr_a, y_tr, available, tax)
    score_df = score_df.copy()
    score_df["actionability"] = action_scores  # override

    best = None
    for alpha in ALPHA_GRID:
        k = min(top_k, len(available))
        selected = ic_fs_select(score_df, alpha, k)
        sel_local = [available.index(f) for f in selected]
        ev = _eval_subset(X_tr_a, y_tr, X_te_a, y_te, sel_local)
        ar  = float(np.mean([action_scores[available.index(f)] for f in selected]))
        tvs = temporal_validity_score(selected, HORIZON, tax)
        ius = ev["f1"] * ar * tvs
        t2r = tier2_ratio(selected, tax)
        row = dict(w_mid=w_mid, top_k=top_k, alpha=alpha,
                    accuracy=ev["accuracy"]*100, f1=ev["f1"]*100,
                    AR=ar, TVS=tvs, IUS=ius*100,
                    tier2_ratio=t2r, selected="|".join(selected))
        if (best is None) or (row["IUS"] > best["IUS"]):
            best = row

    # Stability at best α (smaller n_boot for speed)
    def _stab(alpha_val, k_val):
        rng = np.random.RandomState(2026 + int(w_mid*100) + k_val)
        sel_sets = []
        n = len(y_tr)
        for _ in range(N_BOOT):
            idx = rng.choice(n, size=n, replace=True)
            if len(np.unique(y_tr[idx])) < 2: continue
            sdf = feature_scores_for_selection(X_tr_a[idx], y_tr[idx], available, tax)
            sdf = sdf.copy(); sdf["actionability"] = action_scores
            sel_sets.append(set(ic_fs_select(sdf, alpha_val, min(k_val, len(available)))))
        pairs = list(itertools.combinations(range(len(sel_sets)), 2))
        jacc = [len(A & B)/len(A | B) if len(A|B)>0 else 1.0
                 for i,j in pairs for A,B in [(sel_sets[i], sel_sets[j])]]
        return float(np.mean(jacc)) if jacc else 0.0

    best["stability"] = _stab(best["alpha"], best["top_k"])
    return best


t0 = time.time()
rows = []
for w_mid in W_MID_GRID:
    for top_k in TOP_K_GRID:
        r = run_sens_cell(w_mid, top_k)
        rows.append(r)
        print(f"  w_mid={w_mid} top_k={top_k:2d} | α*={r['alpha']:.2f} "
               f"IUS={r['IUS']:5.2f} F1={r['f1']:5.2f} AR={r['AR']:.2f} "
               f"T2r={r['tier2_ratio']:.2f} Stab={r['stability']:.2f}")

print(f"\nRuntime: {time.time()-t0:.1f}s")
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"Saved {OUT}")

# Spearman: w_mid vs tier2_ratio at each top_k
print("\n--- Spearman rank correlation (w_mid, tier2_ratio) ---")
from scipy.stats import spearmanr
for k in TOP_K_GRID:
    sub = df[df["top_k"]==k]
    if len(sub) >= 2 and sub["tier2_ratio"].std() > 0:
        rho, p = spearmanr(sub["w_mid"], sub["tier2_ratio"])
        print(f"  top_k={k}: ρ={rho:+.3f} (p={p:.3f})")
    else:
        print(f"  top_k={k}: constant tier2_ratio={sub['tier2_ratio'].iloc[0]:.2f}")
