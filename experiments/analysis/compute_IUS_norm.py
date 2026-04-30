"""
Compute normalized IUS metric (IUS_norm) from existing experimental results.

IUS_norm = F1_deploy × (count_actionable_available / k_max)

where:
- count_actionable_available = AR_available × n_features
- k_max = 15 (fixed reference)

This makes cross-method comparison fair regardless of |S| (feature set size).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
K_MAX = 15  # Fixed reference for normalization
RESULTS_DIR = Path("results/oulad")
OUTPUT_DIR = Path("results/oulad/supplementary")
OUTPUT_DIR.mkdir(exist_ok=True)

def compute_IUS_norm_baselines(csv_path, k_max=15):
    """Compute IUS_norm for baseline methods."""
    df = pd.read_csv(csv_path)

    # Compute count_actionable_available
    df['count_actionable'] = df['AR_available'] * df['n_features']

    # Compute IUS_norm
    df['IUS_norm'] = (df['f1_deploy'] / 100.0) * (df['count_actionable'] / k_max)

    return df

def compute_IUS_norm_dre_multi(csv_path, k_max=15):
    """Compute IUS_norm for IC-FS variants (dre_multi)."""
    df = pd.read_csv(csv_path)

    # For IC-FS(full)
    df['count_actionable_full'] = df['AR_available_full'] * df['n_full']
    df['IUS_norm_full'] = (df['f1_full_deploy'] / 100.0) * (df['count_actionable_full'] / k_max)

    # For IC-FS(-temporal)
    df['count_actionable_notemp'] = df['AR_available_notemp'] * df['n_notemp']
    df['IUS_norm_notemp'] = (df['f1_notemp_deploy'] / 100.0) * (df['count_actionable_notemp'] / k_max)

    return df

def compute_IUS_norm_stat8(csv_path, k_max=15):
    """Compute IUS_norm for stat8 ablation study."""
    df = pd.read_csv(csv_path)

    # Columns: IUS_deploy_noTemp, IUS_deploy_noAction, IUS_deploy_hardDEFS, IUS_deploy_IC_FS
    # Need to reconstruct from AR and F1

    # For IC-FS(full)
    if 'AR_available_IC_FS' in df.columns and 'f1_deploy_IC_FS' in df.columns and 'n_IC_FS' in df.columns:
        df['count_actionable_IC_FS'] = df['AR_available_IC_FS'] * df['n_IC_FS']
        df['IUS_norm_IC_FS'] = (df['f1_deploy_IC_FS'] / 100.0) * (df['count_actionable_IC_FS'] / k_max)

    # For IC-FS(-temporal)
    if 'AR_available_noTemp' in df.columns and 'f1_deploy_noTemp' in df.columns and 'n_noTemp' in df.columns:
        df['count_actionable_noTemp'] = df['AR_available_noTemp'] * df['n_noTemp']
        df['IUS_norm_noTemp'] = (df['f1_deploy_noTemp'] / 100.0) * (df['count_actionable_noTemp'] / k_max)

    # For IC-FS(-action)
    if 'AR_available_noAction' in df.columns and 'f1_deploy_noAction' in df.columns and 'n_noAction' in df.columns:
        df['count_actionable_noAction'] = df['AR_available_noAction'] * df['n_noAction']
        df['IUS_norm_noAction'] = (df['f1_deploy_noAction'] / 100.0) * (df['count_actionable_noAction'] / k_max)

    # For HardFilter+DE-FS
    if 'AR_available_hardDEFS' in df.columns and 'f1_deploy_hardDEFS' in df.columns and 'n_hardDEFS' in df.columns:
        df['count_actionable_hardDEFS'] = df['AR_available_hardDEFS'] * df['n_hardDEFS']
        df['IUS_norm_hardDEFS'] = (df['f1_deploy_hardDEFS'] / 100.0) * (df['count_actionable_hardDEFS'] / k_max)

    return df

def create_summary_table_by_horizon(horizon):
    """Create summary table comparing IUS_deploy vs IUS_norm for given horizon."""

    results = []

    # Load baselines
    baselines_path = RESULTS_DIR / f"baselines_oulad_h{horizon}.csv"
    if baselines_path.exists():
        df_baselines = compute_IUS_norm_baselines(baselines_path)

        for _, row in df_baselines.iterrows():
            results.append({
                'Method': row['method'],
                'Horizon': horizon,
                'k': int(row['n_features']),
                'F1_deploy': row['f1_deploy'],
                'AR_available': row['AR_available'],
                'Count_Actionable': row['count_actionable'],
                'IUS_deploy': row['IUS_deploy'],
                'IUS_norm': row['IUS_norm'] * 100,  # Convert to percentage
            })

    # Load IC-FS variants
    dre_path = RESULTS_DIR / f"dre_multi_oulad_h{horizon}.csv"
    if dre_path.exists():
        df_dre = compute_IUS_norm_dre_multi(dre_path)

        # IC-FS(full) - aggregate across seeds
        results.append({
            'Method': 'IC-FS(full)',
            'Horizon': horizon,
            'k': int(df_dre['n_full'].iloc[0]),
            'F1_deploy': df_dre['f1_full_deploy'].mean(),
            'AR_available': df_dre['AR_available_full'].mean(),
            'Count_Actionable': df_dre['count_actionable_full'].mean(),
            'IUS_deploy': df_dre['IUS_deploy_full'].mean(),
            'IUS_norm': df_dre['IUS_norm_full'].mean() * 100,
        })

        # IC-FS(-temporal)
        results.append({
            'Method': 'IC-FS(-temporal)',
            'Horizon': horizon,
            'k': int(df_dre['n_notemp'].iloc[0]),
            'F1_deploy': df_dre['f1_notemp_deploy'].mean(),
            'AR_available': df_dre['AR_available_notemp'].mean(),
            'Count_Actionable': df_dre['count_actionable_notemp'].mean(),
            'IUS_deploy': df_dre['IUS_deploy_notemp'].mean(),
            'IUS_norm': df_dre['IUS_norm_notemp'].mean() * 100,
        })

    # Load stat8 ablation
    stat8_path = RESULTS_DIR / f"stat8_oulad_h{horizon}.csv"
    if stat8_path.exists():
        df_stat8 = compute_IUS_norm_stat8(stat8_path)

        # HardFilter+DE-FS
        if 'IUS_norm_hardDEFS' in df_stat8.columns:
            results.append({
                'Method': 'HardFilter+DE-FS',
                'Horizon': horizon,
                'k': int(df_stat8['n_hardDEFS'].mean()),
                'F1_deploy': df_stat8['f1_deploy_hardDEFS'].mean(),
                'AR_available': df_stat8['AR_available_hardDEFS'].mean(),
                'Count_Actionable': df_stat8['count_actionable_hardDEFS'].mean(),
                'IUS_deploy': df_stat8['IUS_deploy_hardDEFS'].mean(),
                'IUS_norm': df_stat8['IUS_norm_hardDEFS'].mean() * 100,
            })

        # IC-FS(-action)
        if 'IUS_norm_noAction' in df_stat8.columns:
            results.append({
                'Method': 'IC-FS(-action)',
                'Horizon': horizon,
                'k': int(df_stat8['n_noAction'].mean()),
                'F1_deploy': df_stat8['f1_deploy_noAction'].mean(),
                'AR_available': df_stat8['AR_available_noAction'].mean(),
                'Count_Actionable': df_stat8['count_actionable_noAction'].mean(),
                'IUS_deploy': df_stat8['IUS_deploy_noAction'].mean(),
                'IUS_norm': df_stat8['IUS_norm_noAction'].mean() * 100,
            })

    df_summary = pd.DataFrame(results)

    # Sort by IUS_norm descending
    df_summary = df_summary.sort_values('IUS_norm', ascending=False)

    return df_summary

def main():
    """Main execution."""

    print("=" * 80)
    print("Computing IUS_norm (Normalized Intervention Utility Score)")
    print("=" * 80)
    print(f"\nFormula: IUS_norm = F1_deploy × (count_actionable / k_max)")
    print(f"Reference k_max = {K_MAX}")
    print()

    all_summaries = []

    for horizon in [0, 1, 2]:
        print(f"\n{'='*80}")
        print(f"HORIZON h={horizon}")
        print(f"{'='*80}\n")

        df_summary = create_summary_table_by_horizon(horizon)
        all_summaries.append(df_summary)

        # Display table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.float_format', '{:.2f}'.format)

        print(df_summary.to_string(index=False))

        # Save to CSV
        output_path = OUTPUT_DIR / f"IUS_norm_comparison_h{horizon}.csv"
        df_summary.to_csv(output_path, index=False, float_format='%.4f')
        print(f"\n✅ Saved to: {output_path}")

        # Key insights
        print(f"\n📊 Key Insights at h={horizon}:")

        if len(df_summary) > 0:
            top_method = df_summary.iloc[0]
            print(f"  • Top method by IUS_norm: {top_method['Method']} = {top_method['IUS_norm']:.2f}")

            # Compare IC-FS(full) vs baselines
            ic_fs_full = df_summary[df_summary['Method'] == 'IC-FS(full)']
            if not ic_fs_full.empty:
                ic_fs_ius_norm = ic_fs_full['IUS_norm'].iloc[0]
                print(f"  • IC-FS(full) IUS_norm: {ic_fs_ius_norm:.2f}")

                # Compare with NSGA-II
                nsga_ii = df_summary[df_summary['Method'] == 'NSGA-II-MOFS']
                if not nsga_ii.empty:
                    nsga_ius_norm = nsga_ii['IUS_norm'].iloc[0]
                    improvement = ic_fs_ius_norm - nsga_ius_norm
                    pct_improvement = (improvement / nsga_ius_norm) * 100 if nsga_ius_norm > 0 else 0
                    print(f"  • IC-FS(full) vs NSGA-II: +{improvement:.2f} IUS_norm points ({pct_improvement:.1f}% improvement)")

                # Compare with Boruta
                boruta = df_summary[df_summary['Method'] == 'Boruta']
                if not boruta.empty:
                    boruta_ius_norm = boruta['IUS_norm'].iloc[0]
                    improvement = ic_fs_ius_norm - boruta_ius_norm
                    pct_improvement = (improvement / boruta_ius_norm) * 100 if boruta_ius_norm > 0 else 0
                    print(f"  • IC-FS(full) vs Boruta: +{improvement:.2f} IUS_norm points ({pct_improvement:.1f}% improvement)")

            # At h=0, check HardFilter paradox
            if horizon == 0:
                hardfilter = df_summary[df_summary['Method'] == 'HardFilter+DE-FS']
                ic_fs_full = df_summary[df_summary['Method'] == 'IC-FS(full)']

                if not hardfilter.empty and not ic_fs_full.empty:
                    hf_ius_deploy = hardfilter['IUS_deploy'].iloc[0]
                    hf_ius_norm = hardfilter['IUS_norm'].iloc[0]
                    ic_ius_deploy = ic_fs_full['IUS_deploy'].iloc[0]
                    ic_ius_norm = ic_fs_full['IUS_norm'].iloc[0]

                    print(f"\n  ⚠️  HardFilter k-dependence demonstration:")
                    print(f"      IUS_deploy: HardFilter={hf_ius_deploy:.2f} vs IC-FS={ic_ius_deploy:.2f} (ratio: {hf_ius_deploy/ic_ius_deploy:.1f}×)")
                    print(f"      IUS_norm:   HardFilter={hf_ius_norm:.2f} vs IC-FS={ic_ius_norm:.2f} (ratio: {hf_ius_norm/ic_ius_norm:.1f}×)")
                    print(f"      → Normalization corrects the k-dependence artifact")

    # Create combined summary table
    print(f"\n{'='*80}")
    print("COMBINED SUMMARY - ALL HORIZONS")
    print(f"{'='*80}\n")

    df_combined = pd.concat(all_summaries, ignore_index=True)

    # Pivot for easier reading
    df_pivot = df_combined.pivot_table(
        index='Method',
        columns='Horizon',
        values=['IUS_deploy', 'IUS_norm'],
        aggfunc='first'
    )

    print(df_pivot.to_string())

    # Save combined
    output_path = OUTPUT_DIR / "IUS_norm_comparison_all_horizons.csv"
    df_combined.to_csv(output_path, index=False, float_format='%.4f')
    print(f"\n✅ Combined results saved to: {output_path}")

    print("\n" + "="*80)
    print("✅ IUS_norm computation complete!")
    print("="*80)
    print("\n📁 Output files:")
    print(f"  • {OUTPUT_DIR}/IUS_norm_comparison_h0.csv")
    print(f"  • {OUTPUT_DIR}/IUS_norm_comparison_h1.csv")
    print(f"  • {OUTPUT_DIR}/IUS_norm_comparison_h2.csv")
    print(f"  • {OUTPUT_DIR}/IUS_norm_comparison_all_horizons.csv")

    print("\n📝 Interpretation:")
    print("  • IUS_norm accounts for feature set size k")
    print("  • Fair comparison across methods with different k values")
    print("  • If primary conclusions hold under IUS_norm, k-dependence is not fatal")

if __name__ == "__main__":
    main()
