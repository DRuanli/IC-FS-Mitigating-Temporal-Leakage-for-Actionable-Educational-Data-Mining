"""Consolidate ablation results into a Markdown/CSV table."""
import json, glob, os
import pandas as pd

rows = []
for path in sorted(glob.glob("abl_*.json")):
    with open(path) as f:
        data = json.load(f)
    basename = os.path.basename(path)
    dataset = "math" if "mat" in basename else "por"
    for r in data:
        r["dataset"] = dataset
        rows.append(r)

df = pd.DataFrame(rows)
# Reorder columns
cols = ["dataset", "config", "horizon", "alpha_best",
         "accuracy", "f1", "AR", "TVS", "IUS",
         "n_features", "stability", "cv_mean", "cv_std"]
cols_avail = [c for c in cols if c in df.columns]
other_cols = [c for c in df.columns if c not in cols_avail]
df = df[cols_avail + other_cols]

# Round numerics
for c in ["accuracy", "f1", "AR", "TVS", "IUS", "stability", "cv_mean", "cv_std", "alpha_best"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce').round(3)

df.to_csv("ablation_consolidated.csv", index=False)
print("Saved ablation_consolidated.csv\n")

# Print Math table
print("="*100)
print("TABLE 1a — Ablation on UCI Math dataset (N=395, test=79)")
print("="*100)
sub = df[df["dataset"]=="math"][["config","horizon","alpha_best","accuracy","f1",
                                    "AR","TVS","IUS","n_features","stability"]]
print(sub.to_string(index=False))

print()
print("="*100)
print("TABLE 1b — Ablation on UCI Portuguese dataset (N=649, test=130)")
print("="*100)
sub = df[df["dataset"]=="por"][["config","horizon","alpha_best","accuracy","f1",
                                   "AR","TVS","IUS","n_features","stability"]]
print(sub.to_string(index=False))

print()
print("="*100)
print("TABLE 1c — Selected features at horizon=0 (most interesting case)")
print("="*100)
sub = df[df["horizon"]==0]
for _, r in sub.iterrows():
    sel = r.get("selected","")
    if isinstance(sel, str):
        sel_list = sel.split("|")
        print(f"[{r['dataset']:<4}] {r['config']:<24} ({len(sel_list):2d} feat): {sel_list[:8]}{'...' if len(sel_list)>8 else ''}")

print()
print("="*100)
print("KEY COMPARISON: DE-FS trap vs IC-FS (horizon=0, UCI Math)")
print("="*100)
sub = df[(df["horizon"]==0) & (df["dataset"]=="math")]
for _, r in sub.iterrows():
    leak = "LEAK" if (r["TVS"] < 1.0) else "ok"
    action_quality = "HIGH" if (r["AR"] >= 0.7) else ("MED" if r["AR"] >= 0.5 else "LOW")
    print(f"  {r['config']:<24} F1={r['f1']:5.2f}  AR={r['AR']:.2f} ({action_quality})  "
          f"TVS={r['TVS']:.2f} ({leak})  IUS={r['IUS']:5.2f}")
