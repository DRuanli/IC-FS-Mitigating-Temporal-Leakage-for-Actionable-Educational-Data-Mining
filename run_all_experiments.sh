#!/bin/bash
# IC-FS Complete Experiment Pipeline
# Run from project root: ./run_all_experiments.sh

set -e  # Exit on error

echo "========================================="
echo "IC-FS Complete Experiment Pipeline"
echo "========================================="
echo "Start time: $(date)"
echo ""

# Check we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "Error: Must run from project root directory"
    exit 1
fi

# Step 1: OULAD Feature Engineering
echo "========================================="
echo "[1/3] Generating OULAD features..."
echo "Expected time: ~5 minutes"
echo "========================================="
python src/icfs/oulad_pipeline.py

echo ""
echo "✓ OULAD features generated"
ls -lh results/oulad/oulad_features_h*.parquet
echo ""

# Step 2: OULAD IC-FS
echo "========================================="
echo "[2/3] Running IC-FS on OULAD..."
echo "Expected time: ~45-60 minutes"
echo "========================================="
python experiments/oulad/run_oulad_experiments.py

echo ""
echo "✓ OULAD IC-FS complete"
ls -lh results/oulad/oulad_icfs_h*.csv
echo ""

# Step 3: Validation
echo "========================================="
echo "[3/3] Validating results..."
echo "========================================="
python -c "
import pandas as pd

print('\\n' + '='*60)
print('OULAD IC-FS Results Summary')
print('='*60)

for h in [0, 1, 2]:
    df = pd.read_csv(f'results/oulad/oulad_icfs_h{h}.csv')
    best_idx = df['IUS'].idxmax()
    best = df.loc[best_idx]

    print(f'\\nHorizon {h} (t={h}):')
    print(f'  Best α:     {best[\"alpha\"]:.2f}')
    print(f'  F1:         {best[\"f1\"]:.3f} ({best[\"f1\"]*100:.1f}%)')
    print(f'  AR:         {best[\"AR\"]:.3f}')
    print(f'  IUS:        {best[\"IUS\"]:.2f}')
    print(f'  Stability:  {best[\"stability\"]:.3f}')
    print(f'  Features:   {int(best[\"n_features\"])}')

print('\\n' + '='*60)
print('Cross-Horizon Comparison')
print('='*60)

comparison = []
for h in [0, 1, 2]:
    df = pd.read_csv(f'results/oulad/oulad_icfs_h{h}.csv')
    best = df.loc[df['IUS'].idxmax()]
    comparison.append({
        'H': h,
        'F1': f\"{best['f1']:.3f}\",
        'AR': f\"{best['AR']:.3f}\",
        'IUS': f\"{best['IUS']:.2f}\",
    })

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))

# Calculate improvements
ius_values = [pd.read_csv(f'results/oulad/oulad_icfs_h{h}.csv')['IUS'].max() for h in [0,1,2]]
ar_values = [pd.read_csv(f'results/oulad/oulad_icfs_h{h}.csv').loc[
    pd.read_csv(f'results/oulad/oulad_icfs_h{h}.csv')['IUS'].idxmax(), 'AR'] for h in [0,1,2]]

print(f'\\nKey Insights:')
print(f'  IUS improvement H0→H1: +{((ius_values[1]/ius_values[0]-1)*100):.1f}%')
print(f'  IUS improvement H0→H2: +{((ius_values[2]/ius_values[0]-1)*100):.1f}%')
print(f'  AR improvement H0→H1: +{((ar_values[1]/ar_values[0]-1)*100):.1f}%')
print(f'  Best intervention window: Horizon {ius_values.index(max(ius_values))}')
print()
"

echo "========================================="
echo "✓ All experiments complete!"
echo "========================================="
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  - results/oulad/oulad_features_h{0,1,2}.parquet"
echo "  - results/oulad/oulad_icfs_h{0,1,2}.csv"
echo ""
echo "Next steps:"
echo "  1. Review results in results/oulad/"
echo "  2. Run UCI experiments if needed"
echo "  3. Generate figures: cd experiments/analysis && python make_figures.py"
