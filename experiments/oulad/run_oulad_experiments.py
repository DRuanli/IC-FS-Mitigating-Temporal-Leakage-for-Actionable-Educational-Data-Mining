"""
Run IC-FS on OULAD dataset for all prediction horizons.

This script:
1. Loads pre-processed OULAD features from parquet files
2. One-hot encodes categorical variables
3. Runs IC-FS with OULAD taxonomy
4. Saves results to CSV files

Expected runtime: ~10-15 minutes per horizon on laptop.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from oulad_taxonomy import TAXONOMY_OULAD
from ic_fs_v2 import ICFSPipeline

def preprocess_oulad(df, horizon):
    """
    Preprocess OULAD features for IC-FS.

    Args:
        df: DataFrame from oulad_features_h{horizon}.parquet
        horizon: 0, 1, or 2

    Returns:
        X: numpy array of features
        y: numpy array of targets
        feature_names: list of feature names after one-hot encoding
    """
    df = df.copy()

    # Extract target
    y = df.pop('y').values

    # Drop metadata columns not used for prediction
    drop_cols = ['id_student', 'student_key', 'horizon_cutoff',
                  'module_presentation_length', 'final_result']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # One-hot encode categorical variables
    # These will create features like gender_M, gender_F, region_Scotland, etc.
    cat_cols = ['gender', 'region', 'highest_education', 'imd_band',
                'age_band', 'disability', 'code_module', 'code_presentation']

    df = pd.get_dummies(df, columns=[c for c in cat_cols if c in df.columns])

    # Convert date_registration and date_unregistration
    # '?' means not available - convert to NaN then fill
    for col in ['date_registration', 'date_unregistration']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN values with 0 (conservative approach)
    # For Tier-3 assessment scores, NaN at early horizons is expected
    df = df.fillna(0)

    # Remove zero-variance features (can cause numerical issues)
    variances = df.var()
    zero_var_features = variances[variances == 0].index.tolist()
    if zero_var_features:
        print(f"  Removing {len(zero_var_features)} zero-variance features")
        df = df.drop(columns=zero_var_features)

    print(f"  Preprocessed shape: {df.shape}")
    print(f"  Feature count after one-hot: {df.shape[1]}")

    return df.values, y, df.columns.tolist()


def main():
    """Run IC-FS on OULAD for all horizons."""

    print("="*80)
    print("IC-FS on OULAD Dataset")
    print("="*80)

    for h in [0, 1, 2]:
        print(f"\n{'─'*80}")
        print(f"HORIZON {h}")
        print(f"{'─'*80}")

        # Load pre-processed features
        input_file = project_root / f'results/oulad/oulad_features_h{h}.parquet'
        print(f"\n[1/4] Loading {input_file}...")
        df = pd.read_parquet(input_file)
        print(f"  Loaded: {len(df)} students, {df.shape[1]} raw features")
        print(f"  Target distribution: {df['y'].value_counts().to_dict()}")

        # Preprocess
        print(f"\n[2/4] Preprocessing...")
        X, y, feature_names = preprocess_oulad(df, h)

        # Train/test split with stratification
        print(f"\n[3/4] Splitting data...")
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"  Train: {len(X_tr)} samples")
        print(f"  Test: {len(X_te)} samples")

        # Run IC-FS
        print(f"\n[4/4] Running IC-FS...")
        print(f"  Horizon: {h}")
        print(f"  top_k: 15")
        print(f"  n_bootstrap: 20")
        print(f"  alpha_values: [0.0, 0.25, 0.5, 0.75, 1.0]")

        pipe = ICFSPipeline(
            horizon=h,
            top_k=15,                    # OULAD has more features than UCI
            n_bootstrap=20,              # Balance between stability and runtime
            taxonomy=TAXONOMY_OULAD,     # Critical: use OULAD taxonomy
            alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0]
        )

        pipe.fit(X_tr, y_tr, X_te, y_te, feature_names, verbose=True)

        # Save results
        output_file = project_root / f'results/oulad/oulad_icfs_h{h}.csv'
        results_df = pipe.to_dataframe()
        results_df.to_csv(output_file, index=False)

        print(f"\n✓ Saved results to {output_file}")
        print(f"  Best IUS: {results_df['IUS'].max():.2f}")
        print(f"  Best α: {results_df.loc[results_df['IUS'].idxmax(), 'alpha']}")

    print(f"\n{'='*80}")
    print("DONE: IC-FS completed for all horizons")
    print("="*80)
    print("\nOutput files:")
    print(f"  - {project_root}/results/oulad/oulad_icfs_h0.csv")
    print(f"  - {project_root}/results/oulad/oulad_icfs_h1.csv")
    print(f"  - {project_root}/results/oulad/oulad_icfs_h2.csv")


if __name__ == '__main__':
    main()
