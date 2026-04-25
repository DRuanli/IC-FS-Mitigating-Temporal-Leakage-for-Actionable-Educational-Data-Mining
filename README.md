# IC-FS: Intervention-Constrained Feature Selection — JLA Revision Package

**Response to Major Revision for Journal of Learning Analytics**

This package contains the complete empirical response to the Round-1 review of the IC-FS manuscript, plus the hardened codebase, reproducible experiment scripts, and manuscript draft.

---

## Executive Summary

| Review concern | Response | Evidence file |
|---|---|---|
| Dataset generalisation (OULAD needed) | Taxonomy for 31 OULAD features built with 4-tier structure + pipeline guide | `oulad_taxonomy.py` |
| Ablation study missing | 4 configs × 3 horizons × 2 datasets, 24 runs total | `ablation_consolidated.csv`, `run_ablation_fast.py` |
| Sensitivity analysis missing | 4×4 grid on $(w_{mid}, k)$, Spearman correlations | `sens_math_h1.csv`, `fig2_sensitivity.png` |
| Statistical tests missing | Wilcoxon signed-rank + Bonferroni + Cohen's d, 8 seeds | `stat8_math_h0.csv`, `dre_multi_*.csv` |
| Causal framing missing | Dedicated §2 "Causal Assumptions and Limitations" with DAG | `manuscript/icfs_manuscript.md` |
| Stronger baselines | NSGA-II-MOFS, Stability Selection, Boruta implemented | `baselines_*.csv`, `run_baselines.py` |
| Circular evaluation on IUS | Precision@top-k% reported as non-circular metric | §A.4 in manuscript, `ic_fs_v2.py::evaluate_with_external_metric` |
| Code hardening (silent leakage) | All fixes applied: unknown features default to unavailable; conservative actionability = 0 | `ic_fs_v2.py` |

**New contribution added: Deployment-Realistic Evaluation (DRE)**. Beyond the reviewer's requests, we introduce a DRE protocol that masks temporally-unavailable features at inference and retrains, exposing the gap between paper-style F1 (with unintended G1/G2 access) and deployment F1 (the real-world early-warning scenario). This was motivated by the reviewer's critique that DE-FS's 95.8% accuracy is unattainable at *t=0* due to G2 leakage, and is our most empirically decisive contribution.

---

## Headline Numbers

At UCI-Math, horizon t=0 (hardest early-warning case), 8 random seeds:

- **IC-FS (full)** deployment IUS = 58.7 ± 3.6
- **IC-FS (−temporal)** ≈ DE-FS methodology, deployment IUS = 50.4 ± 3.3
- **Wilcoxon p = 0.00391, Cohen's d = +7.47** (Bonferroni-adjusted α = 0.0167)
- **F1 drops 33.3 pts** for IC-FS(−temporal) when G1/G2 are masked at deployment

At UCI-Portuguese, horizon t=0:
- IC-FS (full) deployment IUS = 74.1 ± 1.0 > IC-FS(−temporal) 72.1 ± 2.6
- Wilcoxon p = 0.0156, Cohen's d = +0.91 (borderline Bonferroni)
- Effect smaller because Portuguese has weaker G1/G2 leakage naturally

---

## Directory Contents

### Hardened core
- `ic_fs_v2.py` — IC-FS framework with all critical fixes (Bước A):
  - Unknown feature → unavailable (prevents silent leakage)
  - Unknown feature → actionability 0.0 (conservative, not 0.5)
  - Bootstrap seeds independent per α
  - External metrics for non-circular evaluation
- `ic_fs_core.py` — original (Round-0) for comparison

### Taxonomies
- `oulad_taxonomy.py` — 31 OULAD features mapped to 4 tiers, with feature-engineering guide
- Taxonomy for UCI is embedded in `ic_fs_v2.py::build_uci_taxonomy()`

### Experiment drivers (reproducible)
- `run_ablation_fast.py` — 4-config ablation pipeline
- `run_ablation_small.py` — per-horizon wrapper (sandbox-friendly)
- `run_sensitivity.py` — 4×4 $(w_{mid}, k)$ grid
- `run_statistics.py` — 3-seed statistical framework
- `run_stat_ext.py` — 8-seed extended statistics
- `run_dre.py` / `run_dre_multi.py` — Deployment-Realistic Evaluation
- `run_baselines.py` — NSGA-II-MOFS, Stability Selection, Boruta
- `utils_data.py` — UCI data loading

### Data
- `student-mat.csv`, `student-por.csv` — UCI Student Performance datasets

### Results (csv)
- `ablation_consolidated.csv` — ablation master table
- `sens_math_h1.csv` — sensitivity detailed
- `sens_math_h1_pivot_IUS.csv` / `sens_math_h1_pivot_Stab.csv` — heatmap-ready
- `stat_{math,por}_h{0,1,2}.csv` — 3-seed statistics per dataset × horizon
- `stat8_math_h0.csv` — 8-seed Math h=0 for definitive Wilcoxon
- `dre_mat_h{0,1}.csv`, `dre_por_h0.csv` — single-seed DRE
- `dre_multi_{math,por}_h0.csv` — 8-seed DRE for paired Wilcoxon
- `baselines_{math,por}_h0.csv` — external baselines

### Reporting
- `consolidate_ablation.py`, `final_tables.py`, `master_table.py` — table generators
- `summarize_sensitivity.py` — Spearman + pivots
- `make_figures.py` — publication figures

### Manuscript (`manuscript/`)
- `icfs_manuscript.md` — full draft: 7 sections + appendix, ~5,500 words
- `fig1_pareto.png` — F1 vs AR with α-sweep + baselines + DRE leakage arrow
- `fig2_sensitivity.png` — $(w_{mid}, k)$ heatmap, IUS and Stability
- `fig3_dre_boxplot.png` — paired boxplot paper-style vs deployment
- `fig4_ius_decomp.png` — F1·AR·TVS bar decomposition by method

---

## Reproducibility

### Environment
```
python >= 3.10
scikit-learn >= 1.4
pandas, numpy, scipy, matplotlib
pymoo 0.6.1    # for NSGA-II baseline
boruta 0.4     # for Boruta baseline
```

### Reproduce main results
```bash
# Ablation (24 runs, ~2 min)
python run_ablation_small.py student-mat.csv 0 abl_mat_h0.json
python run_ablation_small.py student-mat.csv 1 abl_mat_h1.json
python run_ablation_small.py student-mat.csv 2 abl_mat_h2.json
python run_ablation_small.py student-por.csv 0 abl_por_h0.json
python run_ablation_small.py student-por.csv 1 abl_por_h1.json
python run_ablation_small.py student-por.csv 2 abl_por_h2.json
python consolidate_ablation.py

# Sensitivity
python run_sensitivity.py student-mat.csv 1 sens_math_h1.csv
python summarize_sensitivity.py

# Statistics (8 seeds)
python run_stat_ext.py student-mat.csv 0 stat8_math_h0.csv

# DRE (8 seeds)
python run_dre_multi.py student-mat.csv 0 dre_multi_math_h0.csv
python run_dre_multi.py student-por.csv 0 dre_multi_por_h0.csv

# Baselines
python run_baselines.py student-mat.csv 0 baselines_math_h0.csv
python run_baselines.py student-por.csv 0 baselines_por_h0.csv

# Final tables & figures
python final_tables.py
python master_table.py
python make_figures.py
```

Seeds used: 42, 123, 456, 789, 1011, 2024, 3033, 4044.
RNG is fully deterministic; re-running any script yields byte-identical CSVs.

---

## What Remains Outside This Package

1. **OULAD empirical validation** — taxonomy is built, pipeline is documented; running it requires downloading ~100MB from https://analyse.kmi.open.ac.uk/open_dataset and ~300 lines of aggregation code outside the scope of this revision round. The manuscript §A.2 describes the exact pipeline for a colleague to execute.

2. **Delphi teacher survey for taxonomy validation** — noted as future work in the Discussion. A small 5-10 teacher study would substantially strengthen construct validity but was not feasible in the revision timeframe.

3. **Multi-institution causal intervention study** — the natural next step after IC-FS identifies intervention candidates. Requires institutional partnership and is out of scope here.

These limitations are discussed transparently in the manuscript §6.# IC-FS-Mitigating-Temporal-Leakage-for-Actionable-Educational-Data-Mining
# IC-FS-Mitigating-Temporal-Leakage-for-Actionable-Educational-Data-Mining
# IC-FS-Mitigating-Temporal-Leakage-for-Actionable-Educational-Data-Mining
