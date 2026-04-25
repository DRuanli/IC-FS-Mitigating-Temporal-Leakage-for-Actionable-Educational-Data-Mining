# IC-FS Project Structure

This document describes the reorganized project structure for the IC-FS (Intervention-Constrained Feature Selection) research project.

## Directory Layout

```
icfs-jla/
├── README.md                      # Project overview and quick start
├── paper.MD                       # Detailed guide for Claude Code/LLM agents
├── PROJECT_STRUCTURE.md           # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation configuration
│
├── src/icfs/                      # Core IC-FS package
│   ├── __init__.py
│   ├── core.py                    # Main IC-FS implementation (from ic_fs_v2.py)
│   ├── taxonomy_oulad.py          # OULAD 4-tier feature taxonomy
│   ├── taxonomy_uci.py            # UCI taxonomy (extracted from core.py)
│   ├── data_loaders.py            # UCI data loading utilities
│   └── oulad_pipeline.py          # OULAD feature engineering pipeline
│
├── experiments/                   # Organized experiment scripts
│   ├── uci/
│   │   ├── run_ablation.py        # 4-config ablation study
│   │   ├── run_sensitivity.py     # (w_mid, k) sensitivity analysis
│   │   ├── run_statistics.py      # Multi-seed statistical tests
│   │   └── run_dre.py             # Deployment-Realistic Evaluation
│   ├── oulad/
│   │   └── run_oulad_experiments.py  # IC-FS on OULAD (all horizons)
│   └── analysis/
│       ├── consolidate_ablation.py   # Aggregate ablation results
│       ├── summarize_sensitivity.py  # Sensitivity pivot tables
│       ├── final_tables.py           # Generate paper tables
│       ├── master_table.py           # Publication-ready master table
│       └── make_figures.py           # All 4 publication figures
│
├── data/                          # Data files
│   ├── uci/
│   │   ├── student-mat.csv
│   │   └── student-por.csv
│   └── oulad_raw/                 # Raw OULAD CSV files (7 tables)
│       ├── studentInfo.csv
│       ├── studentVle.csv
│       ├── studentAssessment.csv
│       ├── assessments.csv
│       ├── vle.csv
│       ├── courses.csv
│       └── studentRegistration.csv
│
├── results/                       # All experimental outputs
│   ├── uci/                       # UCI experiment results
│   ├── oulad/                     # OULAD results
│   │   ├── oulad_features_h0.parquet  # Preprocessed features (t=0)
│   │   ├── oulad_features_h1.parquet  # Preprocessed features (t=1)
│   │   ├── oulad_features_h2.parquet  # Preprocessed features (t=2)
│   │   └── oulad_icfs_h0_test.csv     # IC-FS results (test run)
│   └── figures/                   # Generated plots
│
├── manuscript/                    # Paper drafts and figures
│   ├── icfs_manuscript.md
│   └── figures/
│
└── tests/                         # Unit tests (placeholder for future)
    └── test_core.py
```

## Key Changes from Original Structure

### Before (Flat Structure)
- All 20+ Python files in root directory
- Data scattered across multiple locations
- Duplicate files (`ic_fs_v2 copy.py`, `oulad_taxonomy copy.py`)
- No clear separation between source code and experiments
- Results mixed with source code

### After (Modular Structure)
- **Source code** organized in `src/icfs/` as installable package
- **Experiments** separated by dataset (UCI vs OULAD) and purpose
- **Data** centralized in `data/` with subdirectories
- **Results** isolated in `results/` to avoid clutter
- **Clean dependencies** via `requirements.txt` and `setup.py`

## Installation

### Quick Setup
```bash
# Clone repository
cd /path/to/DE-FS

# Install dependencies
pip install -r requirements.txt

# Install IC-FS package in development mode
pip install -e .
```

### Verify Installation
```python
from icfs.core import ICFSPipeline
from icfs.taxonomy_oulad import TAXONOMY_OULAD
print("IC-FS imported successfully!")
```

## Running Experiments

### OULAD Pipeline
```bash
# Generate features for all horizons (t=0, t=1, t=2)
python src/icfs/oulad_pipeline.py

# Run IC-FS on OULAD
python experiments/oulad/run_oulad_experiments.py
```

### UCI Experiments
```bash
# Ablation study
python experiments/uci/run_ablation.py

# Sensitivity analysis
python experiments/uci/run_sensitivity.py

# Statistical tests (8 seeds)
python experiments/uci/run_statistics.py

# Deployment-Realistic Evaluation
python experiments/uci/run_dre.py
```

### Analysis and Visualization
```bash
cd experiments/analysis

# Generate tables
python final_tables.py
python master_table.py

# Generate figures
python make_figures.py
```

## Reproducibility Notes

### Random Seeds
All experiments use deterministic random seeds for reproducibility:
- Default seeds: `[42, 123, 456, 789, 1011, 2024, 3033, 4044]`
- Single-seed experiments use `42`

### Expected Runtimes
- **OULAD feature engineering**: ~5 minutes (10.6M clickstream records)
- **IC-FS on OULAD** (1 horizon): ~10-15 minutes
- **IC-FS on UCI** (1 horizon): ~2-3 minutes
- **Full ablation study**: ~15 minutes
- **8-seed statistics**: ~20 minutes per dataset

### Data Sizes
- UCI Math: 395 students, 33 features → 45 after one-hot
- UCI Portuguese: 649 students, 33 features → 45 after one-hot
- OULAD: 32,593 enrollments, 31 base features → ~70 after one-hot

## File Migration Map

### Core Source Files
- `ic_fs_v2.py` → `src/icfs/core.py`
- `oulad_taxonomy.py` → `src/icfs/taxonomy_oulad.py`
- `oulad_pipeline.py` → `src/icfs/oulad_pipeline.py`
- `utils_data.py` → `src/icfs/data_loaders.py`

### Experiment Scripts
- `run_ablation_fast.py` → `experiments/uci/run_ablation.py`
- `run_stat_ext.py` → `experiments/uci/run_statistics.py`
- `run_dre_multi.py` → `experiments/uci/run_dre.py`
- `run_sensitivity.py` → `experiments/uci/run_sensitivity.py`
- `run_oulad.py` → `experiments/oulad/run_oulad_experiments.py`

### Analysis Scripts
- `consolidate_ablation.py` → `experiments/analysis/consolidate_ablation.py`
- `final_tables.py` → `experiments/analysis/final_tables.py`
- `make_figures.py` → `experiments/analysis/make_figures.py`
- `master_table.py` → `experiments/analysis/master_table.py`
- `summarize_sensitivity.py` → `experiments/analysis/summarize_sensitivity.py`

### Data Files
- `oulad_data/*` → `data/oulad_raw/*`
- `student-mat.csv`, `student-por.csv` → `data/uci/` (to be added)

### Results
- `oulad_features_h*.parquet` → `results/oulad/`
- `oulad_icfs_h*.csv` → `results/oulad/`
- Generated CSVs from UCI experiments → `results/uci/`
- Generated figures → `results/figures/`

## Files Marked for Removal

These duplicate/obsolete files can be safely deleted:
- `ic_fs_v2 copy.py` (duplicate)
- `oulad_taxonomy copy.py` (duplicate)
- `de_fs_implementation.py` (reference implementation from prior work)
- `run_ablation_small.py` (superseded by run_ablation.py)
- `run_statistics.py` (superseded by run_stat_ext.py)
- `run_dre.py` (single-seed version, superseded by run_dre_multi.py)

## Next Steps

1. **Complete OULAD Experiments**: Run full IC-FS on all 3 horizons
2. **Extract UCI Taxonomy**: Move taxonomy from `core.py` to `taxonomy_uci.py`
3. **Add Unit Tests**: Create `tests/test_core.py`, `tests/test_taxonomy.py`
4. **Add UCI Data**: Move UCI CSVs to `data/uci/`
5. **Update Import Statements**: Update all experiment scripts to use `from icfs.core import ...`
6. **Clean Up Root**: Remove old files after confirming new structure works
7. **Documentation**: Update Readme.MD with new structure

## Contact & Attribution

**Project**: IC-FS (Intervention-Constrained Feature Selection)
**Target**: Journal of Learning Analytics, Round-2 Major Revision
**Author**: Le Nguyen
**Date**: April 2026

For detailed technical documentation, see `paper.MD`.
For manuscript draft, see `manuscript/icfs_manuscript.md`.
