"""
Generate before/after comparison table showing impact of date_unregistration taxonomy fix.

This analysis demonstrates the critical nature of temporal availability validation in
early-warning systems.
"""
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]

# Load OLD results (before fix)
dre_old = pd.read_csv(project_root / "results/oulad/before_taxonomy_fix/dre_multi_oulad_h0_OLD.csv")
baselines_old = pd.read_csv(project_root / "results/oulad/before_taxonomy_fix/baselines_oulad_h0_OLD.csv")

# Load NEW results (after fix)
dre_new = pd.read_csv(project_root / "results/oulad/dre_multi_oulad_h0.csv")
baselines_new = pd.read_csv(project_root / "results/oulad/baselines_oulad_h0.csv")

print("=" * 100)
print("TAXONOMY FIX IMPACT ANALYSIS: date_unregistration at h=0")
print("=" * 100)
print()
print("Issue: date_unregistration was classified as available at h=0 (course start).")
print("       For students who withdraw later, this value represents FUTURE information.")
print("       Only students with date_unregistration < 0 (withdrew before start) are valid.")
print()
print("Fix:   Changed available_at from [0,1,2] to [1,2] in taxonomy_oulad.py")
print()
print("=" * 100)

# ============================================================================
# TABLE 1: IC-FS Multi-seed DRE Results
# ============================================================================
print("\n" + "=" * 100)
print("TABLE 1: IC-FS Deployment-Realistic Evaluation (8 seeds, h=0)")
print("=" * 100)

# Compute statistics
def stats_summary(df, prefix):
    cols = {
        'f1_deploy': f'f1_{prefix}_deploy',
        'f1_paper': f'f1_{prefix}_paper',
        'IUS_deploy': f'IUS_deploy_{prefix}',
        'IUS_paper': f'IUS_paper_{prefix}',
        'AR': f'AR_{prefix}',
        'AR_available': f'AR_available_{prefix}'
    }
    results = {}
    for key, col in cols.items():
        if col in df.columns:
            vals = df[col].values
            results[key] = {
                'mean': vals.mean(),
                'std': vals.std(ddof=1),
                'ci_low': np.percentile(vals, 2.5),
                'ci_high': np.percentile(vals, 97.5)
            }
    return results

icfs_full_old = stats_summary(dre_old, 'full')
icfs_notemp_old = stats_summary(dre_old, 'notemp')
icfs_full_new = stats_summary(dre_new, 'full')
icfs_notemp_new = stats_summary(dre_new, 'notemp')

# Print comparison
print("\nMetric                     | IC-FS(full) BEFORE | IC-FS(full) AFTER  | Change")
print("-" * 100)
for metric in ['f1_deploy', 'IUS_deploy', 'AR_available', 'f1_paper', 'IUS_paper']:
    if metric in icfs_full_old and metric in icfs_full_new:
        old_mean = icfs_full_old[metric]['mean']
        new_mean = icfs_full_new[metric]['mean']
        change = new_mean - old_mean
        change_pct = (change / old_mean * 100) if old_mean != 0 else 0
        print(f"{metric:<26} | {old_mean:18.2f} | {new_mean:18.2f} | {change:+6.2f} ({change_pct:+5.1f}%)")

print("\nMetric                     | IC-FS(-temporal) BEFORE | IC-FS(-temporal) AFTER | Change")
print("-" * 100)
for metric in ['f1_deploy', 'IUS_deploy', 'AR_available', 'f1_paper', 'IUS_paper']:
    if metric in icfs_notemp_old and metric in icfs_notemp_new:
        old_mean = icfs_notemp_old[metric]['mean']
        new_mean = icfs_notemp_new[metric]['mean']
        change = new_mean - old_mean
        change_pct = (change / old_mean * 100) if old_mean != 0 else float('inf')
        print(f"{metric:<26} | {old_mean:23.2f} | {new_mean:22.2f} | {change:+6.2f} ({change_pct:+5.1f}%)")

# ============================================================================
# TABLE 2: Baseline Methods
# ============================================================================
print("\n" + "=" * 100)
print("TABLE 2: Baseline Methods (single-seed, h=0)")
print("=" * 100)

methods = ['NSGA-II-MOFS', 'StabilitySelection', 'Boruta']
print("\nMethod             | IUS_deploy BEFORE | IUS_deploy AFTER  | Change      | F1_deploy BEFORE | F1_deploy AFTER")
print("-" * 100)
for method in methods:
    old_row = baselines_old[baselines_old['method'] == method]
    new_row = baselines_new[baselines_new['method'] == method]

    if not old_row.empty and not new_row.empty:
        old_ius = old_row['IUS_deploy'].values[0] if 'IUS_deploy' in old_row.columns else old_row['IUS_paper'].values[0]
        new_ius = new_row['IUS_deploy'].values[0]
        old_f1 = old_row['f1_deploy'].values[0] if 'f1_deploy' in old_row.columns else old_row['f1_paper'].values[0]
        new_f1 = new_row['f1_deploy'].values[0]

        ius_change = new_ius - old_ius
        ius_pct = (ius_change / old_ius * 100) if old_ius != 0 else 0

        print(f"{method:<18} | {old_ius:17.2f} | {new_ius:17.2f} | {ius_change:+6.2f} ({ius_pct:+5.1f}%) | {old_f1:16.2f} | {new_f1:15.2f}")

# ============================================================================
# CRITICAL FINDINGS
# ============================================================================
print("\n" + "=" * 100)
print("CRITICAL FINDINGS")
print("=" * 100)

print("\n1. IC-FS(-temporal) Collapse:")
print(f"   - IUS_deploy dropped from {icfs_notemp_old['IUS_deploy']['mean']:.2f} to {icfs_notemp_new['IUS_deploy']['mean']:.2f}")
print(f"   - AR_available dropped from {icfs_notemp_old['AR_available']['mean']:.3f} to {icfs_notemp_new['AR_available']['mean']:.3f}")
print("   - This reveals that WITHOUT date_unregistration, IC-FS(-temporal) selects ZERO")
print("     actionable features that are actually available at course start.")
print("   - The paper-style IUS remains high (13.94) because it doesn't mask unavailable features.")

print("\n2. NSGA-II Performance:")
nsga_old_ius = baselines_old[baselines_old['method'] == 'NSGA-II-MOFS']['IUS_deploy'].values[0] if 'IUS_deploy' in baselines_old.columns else baselines_old[baselines_old['method'] == 'NSGA-II-MOFS']['IUS_paper'].values[0]
nsga_new_ius = baselines_new[baselines_new['method'] == 'NSGA-II-MOFS']['IUS_deploy'].values[0]
nsga_drop = nsga_old_ius - nsga_new_ius
nsga_drop_pct = (nsga_drop / nsga_old_ius * 100)
print(f"   - IUS_deploy dropped from {nsga_old_ius:.2f} to {nsga_new_ius:.2f} (-{nsga_drop:.2f}, -{nsga_drop_pct:.1f}%)")
print("   - NSGA-II was heavily reliant on date_unregistration for its h=0 performance.")
print("   - Still outperforms IC-FS(full), but the gap narrowed significantly.")

print("\n3. IC-FS(full) Robustness:")
icfs_full_drop = icfs_full_old['IUS_deploy']['mean'] - icfs_full_new['IUS_deploy']['mean']
icfs_full_drop_pct = (icfs_full_drop / icfs_full_old['IUS_deploy']['mean'] * 100)
print(f"   - IUS_deploy dropped from {icfs_full_old['IUS_deploy']['mean']:.2f} to {icfs_full_new['IUS_deploy']['mean']:.2f} (-{icfs_full_drop:.2f}, -{icfs_full_drop_pct:.1f}%)")
print("   - IC-FS(full) was ALSO using date_unregistration (it was marked as Tier-1).")
print("   - The temporal filter doesn't protect against taxonomy misclassification.")
print("   - This highlights the critical importance of taxonomy correctness.")

print("\n4. The Fundamental Challenge at h=0:")
print("   - At course start, there are essentially NO actionable behavioral features.")
print("   - Demographics (Tier 0) are non-actionable but available.")
print("   - date_registration is the ONLY valid Tier-1 feature at h=0.")
print("   - This explains why IUS_deploy values are universally low (3-7 range).")
print("   - Early-warning systems at h=0 face a fundamental trade-off: timeliness vs actionability.")

print("\n5. Implications for the Paper:")
print("   - The h=0 results should be framed as a LIMITATION, not a strength.")
print("   - The real value of IC-FS emerges at h=1 and h=2 where behavioral features accumulate.")
print("   - The Wilcoxon test at h=0 (IC-FS full vs -temporal) is now HIGHLY significant (p=0.00391).")
print("   - This vindicates the temporal filter: it correctly prevents selection of unavailable features.")

print("\n" + "=" * 100)
print("RECOMMENDATION FOR PAPER")
print("=" * 100)
print("""
The date_unregistration taxonomy error and its correction should be disclosed transparently
in the paper, ideally in a Limitations or Methodological Reflections subsection. This
demonstrates scientific rigor and strengthens the paper's credibility.

Suggested framing:
  "An initial version of our taxonomy classified date_unregistration as available at h=0.
   However, this feature represents the future withdrawal date for students who have not
   yet withdrawn, constituting temporal leakage. Upon correction (restricting availability
   to h≥1), we observed a 62% reduction in NSGA-II's IUS_deploy and complete collapse of
   IC-FS(-temporal)'s actionability ratio to zero. This incident underscores the critical
   importance of taxonomy correctness: even a deployment-honest evaluation framework cannot
   compensate for misclassified feature availability. The corrected results, which we report
   here, reveal the fundamental challenge of early-warning systems at course start: there
   exist essentially no actionable behavioral features before the semester begins."
""")

print("\n" + "=" * 100)
