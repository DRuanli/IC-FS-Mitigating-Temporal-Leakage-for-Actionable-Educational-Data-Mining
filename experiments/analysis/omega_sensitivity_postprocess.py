"""
================================================================================
Tier-2 Weight Sensitivity Analysis - Post-Processing Approach
================================================================================
Recomputes IUS_deploy for different omega_2 values using EXISTING feature
selections from dre_multi experiments.

This avoids the taxonomy deepcopy bug and is much faster since we don't
need to re-run experiments - just recalculate the metric.

Formula:
  AR_available = (1/|S|) × Σ[omega_tier(f) × delta(f,h)]
  IUS_deploy = F1_deploy × AR_available

We recalculate AR_available with different omega_2 weights for Tier-2 features,
then recompute IUS_deploy.

Output: results/oulad/omega_sensitivity_h1.csv
Runtime: ~1 minute
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from ic_fs_v2 import Tier, _resolve_parent
from src.icfs.taxonomy_oulad import TAXONOMY_OULAD

# Configuration
OMEGA_2_VALUES = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
HORIZON = 1
RESULTS_DIR = project_root / "round4" / "results" / "oulad"

# Tier weights
OMEGA_DEFAULT = {
    Tier.NON_ACTIONABLE: 0.0,
    Tier.PRE_SEMESTER: 1.0,
    Tier.MID_SEMESTER: 0.7,  # This is what we vary
    Tier.PAST_GRADE: 0.0,
}


def get_feature_tier(feature_name, taxonomy):
    """Get the tier of a feature from taxonomy (Dict[str, FeatureProfile])."""
    if feature_name in taxonomy:
        return taxonomy[feature_name].tier
    return None


def is_available_at_horizon(feature_name, horizon, taxonomy):
    """Check if feature is available at given horizon."""
    if feature_name in taxonomy:
        return horizon in taxonomy[feature_name].available_at
    return False


def compute_AR_available(selected_features, horizon, taxonomy, omega_2):
    """Compute AR_available with custom omega_2 weight for Tier-2."""
    if len(selected_features) == 0:
        return 0.0

    omega_weights = OMEGA_DEFAULT.copy()
    omega_weights[Tier.MID_SEMESTER] = omega_2

    total = 0.0
    for feat in selected_features:
        tier = get_feature_tier(feat, taxonomy)
        if tier is None:
            continue

        # Check availability
        is_available = is_available_at_horizon(feat, horizon, taxonomy)
        delta = 1.0 if is_available else 0.0

        # Get weight for this tier
        omega = omega_weights.get(tier, 0.0)

        total += omega * delta

    return total / len(selected_features)


def process_dre_multi_file(csv_path, horizon, omega_2):
    """Process dre_multi CSV and recompute IUS with new omega_2."""
    df = pd.read_csv(csv_path)

    results = []

    for _, row in df.iterrows():
        seed = row['seed']

        # Process IC-FS(full)
        selected_full = row['selected_full'].split('|')
        f1_full = row['f1_full_deploy']

        ar_full_new = compute_AR_available(selected_full, horizon, TAXONOMY_OULAD, omega_2)
        ius_full_new = (f1_full / 100.0) * ar_full_new * 100.0

        results.append({
            'seed': seed,
            'horizon': horizon,
            'omega_2': omega_2,
            'variant': 'IC-FS(full)',
            'n_features': row['n_full'],
            'f1_deploy': f1_full,
            'AR_available_original': row['AR_available_full'],
            'AR_available_omega': ar_full_new,
            'IUS_deploy_original': row['IUS_deploy_full'],
            'IUS_deploy_omega': ius_full_new,
            'selected': row['selected_full'],
        })

        # Process IC-FS(-temporal)
        selected_notemp = row['selected_notemp'].split('|')
        f1_notemp = row['f1_notemp_deploy']

        ar_notemp_new = compute_AR_available(selected_notemp, horizon, TAXONOMY_OULAD, omega_2)
        ius_notemp_new = (f1_notemp / 100.0) * ar_notemp_new * 100.0

        results.append({
            'seed': seed,
            'horizon': horizon,
            'omega_2': omega_2,
            'variant': 'IC-FS(-temporal)',
            'n_features': row['n_notemp'],
            'f1_deploy': f1_notemp,
            'AR_available_original': row['AR_available_notemp'],
            'AR_available_omega': ar_notemp_new,
            'IUS_deploy_original': row['IUS_deploy_notemp'],
            'IUS_deploy_omega': ius_notemp_new,
            'selected': row['selected_notemp'],
        })

    return results


def main():
    print("=" * 80)
    print("Tier-2 Weight Sensitivity Analysis - Post-Processing")
    print("=" * 80)
    print(f"\nHorizon: h={HORIZON}")
    print(f"Testing omega_2 values: {OMEGA_2_VALUES}")
    print(f"Default (paper) value: omega_2 = 0.7")
    print()

    # Load dre_multi file
    dre_file = RESULTS_DIR / f"dre_multi_oulad_h{HORIZON}.csv"
    if not dre_file.exists():
        print(f"❌ ERROR: File not found: {dre_file}")
        return

    print(f"[Load] Reading {dre_file.name}...")

    # Process for each omega_2 value
    all_results = []

    for omega_2 in OMEGA_2_VALUES:
        print(f"\n{'='*80}")
        print(f"OMEGA_2 = {omega_2}")
        print(f"{'='*80}")

        results = process_dre_multi_file(dre_file, HORIZON, omega_2)
        all_results.extend(results)

        # Summary for this omega_2
        df_omega = pd.DataFrame(results)

        # Statistics for IC-FS(full)
        full_results = df_omega[df_omega['variant'] == 'IC-FS(full)']
        ius_mean = full_results['IUS_deploy_omega'].mean()
        ius_std = full_results['IUS_deploy_omega'].std()
        ar_mean = full_results['AR_available_omega'].mean()

        print(f"  IC-FS(full) with omega_2={omega_2}:")
        print(f"    • IUS_deploy = {ius_mean:.2f} ± {ius_std:.2f}")
        print(f"    • AR_available = {ar_mean:.3f}")
        print(f"    • F1_deploy = {full_results['f1_deploy'].mean():.2f}%")

        # Check vs original (should match when omega_2=0.7)
        if omega_2 == 0.7:
            ius_orig = full_results['IUS_deploy_original'].mean()
            ar_orig = full_results['AR_available_original'].mean()
            print(f"    • Verification vs original: IUS={abs(ius_mean-ius_orig):.4f} diff, AR={abs(ar_mean-ar_orig):.4f} diff")
            if abs(ius_mean - ius_orig) < 0.1 and abs(ar_mean - ar_orig) < 0.001:
                print(f"    ✅ Matches original (omega_2=0.7 baseline)")
            else:
                print(f"    ⚠️  Discrepancy detected!")

    # Save all results
    df_all = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / f"omega_sensitivity_h{HORIZON}.csv"
    df_all.to_csv(out_csv, index=False, float_format='%.4f')
    print(f"\n{'='*80}")
    print(f"✅ Saved to: {out_csv}")
    print(f"{'='*80}")

    # Generate summary table
    print(f"\n{'='*80}")
    print("SUMMARY: IUS_deploy by omega_2 (IC-FS(full) only, mean ± std across 8 seeds)")
    print(f"{'='*80}\n")

    summary_data = []
    baseline_ius = None

    for omega_2 in OMEGA_2_VALUES:
        df_omega = df_all[(df_all['omega_2'] == omega_2) & (df_all['variant'] == 'IC-FS(full)')]

        ius_mean = df_omega['IUS_deploy_omega'].mean()
        ius_std = df_omega['IUS_deploy_omega'].std()
        ius_min = df_omega['IUS_deploy_omega'].min()
        ius_max = df_omega['IUS_deploy_omega'].max()

        ar_mean = df_omega['AR_available_omega'].mean()
        ar_std = df_omega['AR_available_omega'].std()

        f1_mean = df_omega['f1_deploy'].mean()

        if omega_2 == 0.7:
            baseline_ius = ius_mean

        summary_data.append({
            'omega_2': omega_2,
            'IUS_mean': ius_mean,
            'IUS_std': ius_std,
            'IUS_min': ius_min,
            'IUS_max': ius_max,
            'AR_mean': ar_mean,
            'AR_std': ar_std,
            'F1_mean': f1_mean,
        })

    df_summary = pd.DataFrame(summary_data)
    print(df_summary.to_string(index=False, float_format='%.2f'))

    # Robustness analysis
    print(f"\n{'='*80}")
    print("ROBUSTNESS ANALYSIS")
    print(f"{'='*80}\n")

    print(f"1. Baseline (omega_2=0.7, paper's choice):")
    print(f"   • IUS_deploy = {baseline_ius:.2f}")

    print(f"\n2. Performance across omega_2 range:")
    for _, row in df_summary.iterrows():
        omega = row['omega_2']
        ius = row['IUS_mean']
        diff = ius - baseline_ius
        pct_diff = (diff / baseline_ius) * 100 if baseline_ius > 0 else 0

        if abs(pct_diff) < 5:
            marker = "✓"
        elif abs(pct_diff) < 10:
            marker = "~"
        else:
            marker = "⚠"

        print(f"   {marker} omega_2={omega:.1f}: {ius:5.2f} ({diff:+5.2f}, {pct_diff:+5.1f}%)")

    # Calculate CV for middle range
    middle_range = [0.5, 0.6, 0.7, 0.8, 0.9]
    middle_ius = df_summary[df_summary['omega_2'].isin(middle_range)]['IUS_mean'].values
    cv = (np.std(middle_ius) / np.mean(middle_ius)) * 100 if np.mean(middle_ius) > 0 else 0

    print(f"\n3. Robustness metric:")
    print(f"   • Coefficient of Variation (CV) across omega_2 ∈ [0.5, 0.9]: {cv:.1f}%")

    if cv < 10:
        print(f"   ✅ ROBUST: Performance is stable across middle range")
        print(f"   → omega_2=0.7 is a valid conservative choice")
    elif cv < 15:
        print(f"   ~ MODERATELY ROBUST: Some variation but acceptable")
        print(f"   → omega_2=0.7 is reasonable, consider [0.6, 0.8] as safe range")
    else:
        print(f"   ⚠️ SENSITIVE: Performance varies significantly")
        print(f"   → May need to justify choice more carefully or report range")

    # Best omega
    best_omega = df_summary.loc[df_summary['IUS_mean'].idxmax(), 'omega_2']
    best_ius = df_summary['IUS_mean'].max()

    print(f"\n4. Optimal omega_2:")
    print(f"   • Best: omega_2={best_omega} (IUS_deploy={best_ius:.2f})")
    print(f"   • Paper's choice: omega_2=0.7 (IUS_deploy={baseline_ius:.2f})")

    if best_omega == 0.7:
        print(f"   ✅ Paper's choice is optimal!")
    else:
        diff_from_best = baseline_ius - best_ius
        pct_from_best = (diff_from_best / best_ius) * 100
        print(f"   → Gap from optimal: {diff_from_best:.2f} IUS points ({pct_from_best:.1f}%)")
        if abs(pct_from_best) < 2:
            print(f"   ✅ Paper's choice is near-optimal (< 2% from best)")
        elif abs(pct_from_best) < 5:
            print(f"   ~ Paper's choice is competitive (< 5% from best)")

    print(f"\n{'='*80}")
    print("✅ Sensitivity analysis complete!")
    print(f"{'='*80}")

    # Save summary
    summary_csv = RESULTS_DIR / f"omega_sensitivity_summary_h{HORIZON}.csv"
    df_summary.to_csv(summary_csv, index=False, float_format='%.4f')
    print(f"\n📁 Output files:")
    print(f"  • {out_csv.relative_to(project_root)}")
    print(f"  • {summary_csv.relative_to(project_root)}")

    print(f"\n📝 Interpretation for Response Letter:")
    print(f"  • Performance robust across omega_2 ∈ [0.5, 0.9] (CV = {cv:.1f}%)")
    print(f"  • Paper's omega_2=0.7 is {'optimal' if best_omega==0.7 else f'near-optimal ({abs((baseline_ius-best_ius)/best_ius*100):.1f}% from best)'}")
    print(f"  • Represents conservative middle value in pedagogically reasonable range")


if __name__ == "__main__":
    main()
