# UCI Student Performance Experiments

This directory contains IC-FS experiments on the UCI Student Performance datasets (Mathematics and Portuguese).

## Quick Start

### Run single-seed experiments (all horizons)

```bash
# Math dataset
python experiments/uci/run_uci_experiments.py --dataset math

# Portuguese dataset
python experiments/uci/run_uci_experiments.py --dataset portuguese
```

### Run multi-seed experiments (8 seeds for statistical analysis)

```bash
# Math dataset
python experiments/uci/run_uci_experiments.py --dataset math --multi-seed

# Portuguese dataset
python experiments/uci/run_uci_experiments.py --dataset portuguese --multi-seed
```

### Run single horizon

```bash
# Math, horizon 0 only
python experiments/uci/run_uci_experiments.py --dataset math --horizon 0

# Portuguese, horizon 1 only
python experiments/uci/run_uci_experiments.py --dataset portuguese --horizon 1
```

## Files

- `preprocess_uci.py` - UCI data preprocessing utilities
- `run_uci_experiments.py` - Main IC-FS α-sweep experiments
- `run_uci_baselines.py` - Baseline comparisons (NSGA-II, Boruta, Stability Selection)
- `run_uci_statistics.py` - Multi-seed ablation study with statistical tests
- `README.md` - This file

## Output Structure

Results are saved to `results/uci/{dataset}/`:

```
results/uci/
├── math/
│   ├── uci_math_icfs_h0.csv          # Single-seed α-sweep, horizon 0
│   ├── uci_math_icfs_h1.csv          # Single-seed α-sweep, horizon 1
│   ├── uci_math_icfs_h2.csv          # Single-seed α-sweep, horizon 2
│   ├── uci_math_icfs_multi_h0.csv    # Multi-seed best-IUS, horizon 0
│   ├── uci_math_icfs_multi_h1.csv    # Multi-seed best-IUS, horizon 1
│   └── uci_math_icfs_multi_h2.csv    # Multi-seed best-IUS, horizon 2
└── portuguese/
    ├── uci_portuguese_icfs_h0.csv
    ├── uci_portuguese_icfs_h1.csv
    ├── uci_portuguese_icfs_h2.csv
    ├── uci_portuguese_icfs_multi_h0.csv
    ├── uci_portuguese_icfs_multi_h1.csv
    └── uci_portuguese_icfs_multi_h2.csv
```

## Dataset Information

### Math Dataset
- **Students**: 395
- **Pass rate**: ~67%
- **Features**: 32 (varies by horizon)

### Portuguese Dataset
- **Students**: 649
- **Pass rate**: ~85%
- **Features**: 32 (varies by horizon)

## Horizons

- **h=0**: Beginning of semester (29 features: demographics + Tier-1 actionable)
- **h=1**: Mid-semester (31 features: + G1 grade + absences)
- **h=2**: Late-semester (32 features: + G2 grade)

## Tier Structure

- **Tier 0 (14)**: Non-actionable demographics & SES
- **Tier 1 (9)**: Pre-semester actionable (KEY intervention targets)
  - studytime, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic
- **Tier 2 (7)**: Mid-semester observable
  - famrel, freetime, goout, Dalc, Walc, health, absences
- **Tier 3 (2)**: Past grades
  - G1, G2

## Key Results (h=0, seed=42)

### Math Dataset
- **Best α**: 0.00 (full actionability priority)
- **F1_deploy**: 65.62%
- **AR_available**: 1.000
- **IUS_deploy**: 65.62
- **Stability**: 1.000
- **Top-5 features**: studytime, schoolsup, famsup, paid, activities

All selected features are Tier-1 (pre-semester actionable), making them ideal for early intervention design.

## Running Baselines and Ablations

### Baseline Comparisons

Compare IC-FS against established methods:

```bash
# Math dataset
python experiments/uci/run_uci_baselines.py --dataset math --horizon 0

# Portuguese dataset
python experiments/uci/run_uci_baselines.py --dataset portuguese --horizon 0
```

**Baselines included:**
- **NSGA-II-MOFS**: Multi-objective evolutionary (F1 vs AR)
- **Stability Selection**: Bootstrap L1-logistic regression
- **Boruta**: Permutation-based RF importance

**Runtime**: ~2 minutes per (dataset, horizon)

### Statistical Ablation Study

Compare IC-FS variants across 8 random seeds:

```bash
# Math dataset, horizon 0, 8 seeds
python experiments/uci/run_uci_statistics.py --dataset math --horizon 0
```

**Variants tested:**
1. **IC-FS (full)** - Complete method
2. **IC-FS (--temporal)** - No temporal filter (shows leakage)
3. **IC-FS (--actionability)** - Pure prediction (ignores actionability)
4. **HardFilter + DE-FS** - Heuristic Tier filtering

**Includes:**
- Paired Wilcoxon signed-rank tests
- Cohen's d effect sizes
- Bootstrap 95% confidence intervals

**Runtime**: ~15-20 minutes per (dataset, horizon)

### Quick Baseline Results (Math h=0)

| Method | IUS_deploy | AR_available | Features |
|--------|-----------|--------------|----------|
| **IC-FS (full)** | **65.62** | **1.000** | 5 |
| NSGA-II-MOFS | 62.45 | 0.957 | 7 |
| Stability Selection | 32.96 | 0.540 | 5 |
| Boruta | 22.12 | 0.350 | 2 |

**IC-FS achieves:**
- ✅ Highest intervention utility (IUS_deploy)
- ✅ Perfect actionability (all features modifiable)
- ✅ Compact feature set (5 features)

See `UCI_BASELINES_AND_ABLATIONS.md` for detailed analysis.

## Generating Figures

After running experiments, generate results figures:

```bash
# Math dataset figures
python experiments/analysis/generate_results_figures.py --dataset uci_math

# Portuguese dataset figures
python experiments/analysis/generate_results_figures.py --dataset uci_portuguese
```

Figures are saved to `manuscript/figures/results/uci_{dataset}/`.

## Runtime

- **Single horizon, single seed**: ~30 seconds
- **All horizons, single seed**: ~2 minutes
- **All horizons, 8 seeds**: ~15-20 minutes

## Comparison with OULAD

| Aspect | UCI | OULAD |
|--------|-----|-------|
| Sample size | ~400-650 | ~32,000 |
| Features | 32 | 31 |
| Tier-1 richness | **High** (9 features) | Low (1 feature) |
| Data type | Survey/static | Clickstream/dynamic |
| Best for | Intervention design | Scale/generalization |

UCI's rich Tier-1 features make it ideal for demonstrating IC-FS's intervention-focused feature selection at early horizons.
