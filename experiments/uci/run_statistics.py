"""
Extended multi-seed analysis with 8 seeds.
With n=8, Wilcoxon minimum p = 2^-8 / 2 ≈ 0.004 (one-sided), enough for Bonferroni.
"""
import sys, warnings, time
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split

from utils_data import load_uci, preprocess_uci
from run_ablation import (run_config_full_fast, run_config_no_temporal_fast,
                           run_config_no_action_fast, run_config_hardfilter_fast)

RNG_SEEDS = [42, 123, 456, 789, 1011, 2024, 3033, 4044]  # 8 seeds
ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
N_BOOT_STAB = 3  # minimal within-seed bootstrap (we're averaging over seeds anyway)


def bootstrap_ci(values, ci=0.95):
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0: return (np.nan, np.nan, np.nan)
    lo, hi = (1-ci)/2*100, (1+ci)/2*100
    return (float(np.mean(arr)), float(np.percentile(arr, lo)),
             float(np.percentile(arr, hi)))


def cohens_d_paired(x, y):
    x, y = np.asarray(x), np.asarray(y)
    diff = x - y
    s = np.std(diff, ddof=1) if len(diff) > 1 else 0
    return float(np.mean(diff) / s) if s > 1e-10 else 0.0


def run_seed(df_raw, seed, horizon, top_k=12):
    X, y, names = preprocess_uci(df_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                  random_state=seed, stratify=y)
    r1 = run_config_full_fast(X_tr,y_tr,X_te,y_te,names, horizon, top_k, ALPHA_GRID, n_bootstrap=N_BOOT_STAB)
    r2 = run_config_no_temporal_fast(X_tr,y_tr,X_te,y_te,names, horizon, top_k, ALPHA_GRID, n_bootstrap=N_BOOT_STAB)
    r3 = run_config_no_action_fast(X_tr,y_tr,X_te,y_te,names, horizon, top_k, n_bootstrap=N_BOOT_STAB)
    r4 = run_config_hardfilter_fast(X_tr,y_tr,X_te,y_te,names, horizon, top_k, n_bootstrap=N_BOOT_STAB)
    return {"seed":seed, "horizon":horizon,
             # New deployment-honest metrics (primary)
             "IUS_deploy_full":r1["IUS_deploy"], "IUS_deploy_noTemp":r2["IUS_deploy"],
             "IUS_deploy_noAction":r3["IUS_deploy"], "IUS_deploy_hardDEFS":r4["IUS_deploy"],
             # Old paper-style metrics (for comparison)
             "IUS_paper_full":r1["IUS_paper"], "IUS_paper_noTemp":r2["IUS_paper"],
             "IUS_paper_noAction":r3["IUS_paper"], "IUS_paper_hardDEFS":r4["IUS_paper"],
             # Other metrics
             "F1_full":r1["f1"], "F1_noTemp":r2["f1"],
             "AR_full":r1["AR"], "AR_available_full":r1["AR_available"],
             "AR_available_noTemp":r2["AR_available"],
             "TVS_full":r1["TVS"], "TVS_noTemp":r2["TVS"],
             # Legacy IUS (same as IUS_deploy)
             "IUS_full":r1["IUS"], "IUS_noTemp":r2["IUS"],
             "IUS_noAction":r3["IUS"], "IUS_hardDEFS":r4["IUS"]}


def main():
    dataset = sys.argv[1] if len(sys.argv)>1 else "student-mat.csv"
    horizon = int(sys.argv[2]) if len(sys.argv)>2 else 0
    out = sys.argv[3] if len(sys.argv)>3 else f"stat8_{dataset.split('-')[1].split('.')[0]}_h{horizon}.csv"
    df_raw = load_uci(dataset)
    rows = []
    t0 = time.time()
    for s in RNG_SEEDS:
        r = run_seed(df_raw, s, horizon)
        rows.append(r)
        print(f"  seed={s:4d}: IUS_deploy_full={r['IUS_deploy_full']:.2f} "
               f"noTemp={r['IUS_deploy_noTemp']:.2f} "
               f"noAction={r['IUS_deploy_noAction']:.2f} "
               f"hardDEFS={r['IUS_deploy_hardDEFS']:.2f}")
    df = pd.DataFrame(rows); df.to_csv(out, index=False)

    print(f"\n--- Bootstrap 95% CI - Deployment-Honest Metrics (across {len(rows)} seeds) ---")
    for col in ["IUS_deploy_full","IUS_deploy_noTemp","IUS_deploy_noAction","IUS_deploy_hardDEFS",
                "AR_available_full","AR_available_noTemp","F1_full","TVS_full","TVS_noTemp"]:
        m, lo, hi = bootstrap_ci(df[col].values, 0.95)
        print(f"  {col:<24}: mean={m:6.2f}  95% CI=[{lo:5.2f}, {hi:5.2f}]")

    print(f"\n--- Bootstrap 95% CI - Paper-Style Metrics (for comparison) ---")
    for col in ["IUS_paper_full","IUS_paper_noTemp","IUS_paper_noAction","IUS_paper_hardDEFS"]:
        m, lo, hi = bootstrap_ci(df[col].values, 0.95)
        print(f"  {col:<24}: mean={m:6.2f}  95% CI=[{lo:5.2f}, {hi:5.2f}]")

    print(f"\n--- Wilcoxon signed-rank (one-sided 'greater'), Bonferroni α=0.0167 ---")
    print("Using IUS_deploy (deployment-honest metric):")
    a = df["IUS_deploy_full"].values
    for col in ["IUS_deploy_noTemp","IUS_deploy_noAction","IUS_deploy_hardDEFS"]:
        b = df[col].values
        if np.all(a == b):
            print(f"  full vs {col}: identical, skip")
            continue
        try:
            stat,p = wilcoxon(a,b,alternative="greater",zero_method="wilcox")
            d = cohens_d_paired(a,b)
            sig = "***" if p<0.001 else "**" if p<0.01 else "*" if p<0.0167 else "ns"
            print(f"  full vs {col:<24}: W={stat:5.1f}  p={p:.5f}  Cohen's d={d:+.3f}  [{sig}]")
        except ValueError as e:
            print(f"  full vs {col}: {e}")

    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__=="__main__": main()
