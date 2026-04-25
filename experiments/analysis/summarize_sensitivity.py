import pandas as pd, numpy as np
from scipy.stats import spearmanr

df = pd.read_csv("sens_math_h1.csv")

print("="*90)
print("TABLE 2 — Sensitivity analysis (UCI Math, horizon=1)")
print("="*90)
print("\nPivot: IUS by (w_mid × top_k) with best α*")
pv_ius = df.pivot_table(index="w_mid", columns="top_k", values="IUS", aggfunc="first").round(2)
print(pv_ius.to_string())

print("\nPivot: Stability by (w_mid × top_k)")
pv_stab = df.pivot_table(index="w_mid", columns="top_k", values="stability", aggfunc="first").round(2)
print(pv_stab.to_string())

print("\nPivot: Best α* by (w_mid × top_k)")
pv_alpha = df.pivot_table(index="w_mid", columns="top_k", values="alpha", aggfunc="first").round(2)
print(pv_alpha.to_string())

print("\nPivot: Tier-2 feature ratio by (w_mid × top_k)")
pv_t2 = df.pivot_table(index="w_mid", columns="top_k", values="tier2_ratio", aggfunc="first").round(3)
print(pv_t2.to_string())

print("\n--- Spearman correlation between w_mid and Tier-2 ratio ---")
for k in sorted(df["top_k"].unique()):
    sub = df[df["top_k"]==k]
    if sub["tier2_ratio"].std() > 0 and len(sub) >= 2:
        rho, p = spearmanr(sub["w_mid"], sub["tier2_ratio"])
        print(f"  top_k={k:2d}: ρ={rho:+.3f} p={p:.3f}  (expected: ρ > 0 since higher w_mid → prefer Tier-2)")
    else:
        print(f"  top_k={k:2d}: Tier-2 ratio constant at {sub['tier2_ratio'].iloc[0]:.3f}")

print("\n--- Spearman correlation between w_mid and IUS ---")
for k in sorted(df["top_k"].unique()):
    sub = df[df["top_k"]==k]
    if sub["IUS"].std() > 0 and len(sub) >= 2:
        rho, p = spearmanr(sub["w_mid"], sub["IUS"])
        print(f"  top_k={k:2d}: ρ={rho:+.3f} p={p:.3f}")
    else:
        print(f"  top_k={k:2d}: IUS constant at {sub['IUS'].iloc[0]:.2f}")

# Best overall cell
best_row = df.loc[df["IUS"].idxmax()]
print("\n--- Best cell overall ---")
print(f"  w_mid={best_row['w_mid']} top_k={best_row['top_k']} α*={best_row['alpha']}")
print(f"  IUS={best_row['IUS']:.2f} F1={best_row['f1']:.2f} AR={best_row['AR']:.3f}")
print(f"  Stability={best_row['stability']:.3f}")

# Stability is plot-worthy; save pivot for heatmap
pv_ius.to_csv("sens_math_h1_pivot_IUS.csv")
pv_stab.to_csv("sens_math_h1_pivot_Stab.csv")
print("\nSaved pivots.")
