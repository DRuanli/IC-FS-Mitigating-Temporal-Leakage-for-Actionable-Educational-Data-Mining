# IC-FS: Intervention-Constrained Feature Selection
## Project Guide for Claude Code

> **Purpose**: This is a self-contained guide that lets Claude Code (or any LLM coding agent) understand the project, run experiments, extend the codebase, and complete the JLA submission. Read top-to-bottom before doing anything.

---

## 1. Project Overview

### 1.1 Origin
This project is a **Major Revision response** to a Q1 review (Journal of Learning Analytics) for the IC-FS algorithm. The Round-1 reviewer flagged 8 critical issues; Round-2 must address them all empirically before the paper is acceptable.

### 1.2 What is IC-FS?
Intervention-Constrained Feature Selection — a framework for picking features that are useful for **early-warning student performance prediction**. Unlike prior work (DE-FS, Malik et al. 2025, *Sci. Rep.*), IC-FS enforces two constraints:

1. **Temporal availability**: feature must exist at the prediction horizon (e.g., G2 grade is unavailable at start-of-term)
2. **Actionability**: feature must be modifiable by educators (e.g., `studytime` is actionable, `sex` is not)

### 1.3 Core artifacts (non-negotiable design)

```python
# 4-tier actionability taxonomy
Tier.NON_ACTIONABLE = 0   # demographics, SES, past failures
Tier.PRE_SEMESTER   = 1   # studytime, schoolsup, internet, higher
Tier.MID_SEMESTER   = 2   # absences, alcohol, family relations
Tier.PAST_GRADE     = 3   # G1, G2 — predictive but non-retroactive

ACTIONABILITY_WEIGHTS = {
    Tier.NON_ACTIONABLE: 0.0,
    Tier.PRE_SEMESTER:   1.0,
    Tier.MID_SEMESTER:   0.7,
    Tier.PAST_GRADE:     0.0,
}

# Selection objective (scalarized multi-objective)
ic_score(f, α) = α · pred_score(f) + (1-α) · actionability(f)

# Evaluation metric (multiplicative hard-gate)
IUS = F1 × AR × TVS
where:
  AR  = mean actionability weight of selected features
  TVS = fraction of selected features available at horizon t
```

### 1.4 The DRE finding (most important contribution)
**Deployment-Realistic Evaluation** masks temporally-unavailable features at inference time and retrains. Reveals that DE-FS-style methods report 87.7% F1 in the paper but achieve only 54.4% F1 in actual t=0 deployment (UCI-Math, 8 seeds, **Cohen's d = +7.47, p = 0.004**).

This is what saves the paper from being "yet another FS variant" — it **exposes a systemic measurement bias** in the EDM literature.

---

## 2. Repository Layout

```
icfs-jla/
├── README.md                         # User-facing summary
├── paper.md                          # ← THIS FILE (Claude Code guide)
├── manuscript/
│   ├── icfs_manuscript.md            # Full draft (5,556 words, 7 sections)
│   ├── fig1_pareto.png               # F1 vs AR with α-sweep + leakage arrow
│   ├── fig2_sensitivity.png          # (w_mid × top_k) heatmap
│   ├── fig3_dre_boxplot.png          # Paper-style vs DRE F1 boxplot
│   └── fig4_ius_decomp.png           # F1·AR·TVS bar chart per method
│
├── ic_fs_v2.py                       # ★ Hardened core (Step A fixes applied)
├── ic_fs_core.py                     # Original Round-0 (kept for diff)
├── oulad_taxonomy.py                 # 31 OULAD features, 4-tier
├── utils_data.py                     # UCI loaders
│
├── ablation.py                       # Slow ablation (reference impl)
├── run_ablation_fast.py              # ★ Fast 4-config runner
├── run_ablation_small.py             # Per-horizon CLI wrapper
├── run_sensitivity.py                # 4×4 (w_mid, top_k) grid
├── run_statistics.py                 # 3-seed statistics
├── run_stat_ext.py                   # ★ 8-seed Wilcoxon
├── run_dre.py                        # Single-seed DRE
├── run_dre_multi.py                  # ★ 8-seed DRE (paired Wilcoxon)
├── run_baselines.py                  # NSGA-II, Stability Sel., Boruta
│
├── consolidate_ablation.py           # JSONs → unified CSV
├── summarize_sensitivity.py          # Pivots + Spearman
├── final_tables.py                   # Paper Tables 1-4
├── master_table.py                   # Publication-ready master table
├── make_figures.py                   # All 4 figures
│
├── student-mat.csv                   # UCI Math (n=395)
├── student-por.csv                   # UCI Portuguese (n=649)
│
└── *.csv                             # Cached experiment results (~20 files)
```

★ = files Claude Code should touch most often.

---

## 3. Critical Code Conventions

### 3.1 Things that MUST NOT change
- **Taxonomy lookup**: `_resolve_parent()` does prefix matching for one-hot encoded features (`Mjob_teacher` → parent `Mjob`). Do not break this.
- **Conservative defaults** (Step A fixes from Round-1 reviewer):
  - Unknown feature in `filter_by_horizon` → `False` + warning (NEVER True)
  - Unknown feature in `get_actionability_score` → `0.0` + warning (NEVER 0.5)
- **Bootstrap seed independence**: `_bootstrap_stability` must accept `bootstrap_seed` per call. Sharing seeds across α values introduces correlation that invalidates Wilcoxon.
- **n_jobs=1 in sandbox**: joblib parallelism crashes the sandbox. On local with multiple cores, change `n_jobs=1` → `n_jobs=-1` for 3–5× speedup.

### 3.2 Things that CAN change carefully
- `RNG_SEEDS` — increase to 20+ for stronger Wilcoxon power
- `ALPHA_GRID` — denser sweep gives smoother Pareto
- `top_k` default 12 — sensitivity-validated optimum
- `n_bootstrap` — 15 for ablation, 50+ for final stability claims

### 3.3 Naming conventions
- Methods labeled `IC-FS(-temporal)` ≈ DE-FS methodology (no leakage filter)
- "Paper-style" = test set retains G1/G2 (current literature convention)
- "DRE" = deployment-realistic (mask unavailable features at inference)
- "h" or "horizon" ∈ {0, 1, 2} corresponds to {start-of-term, after-G1, after-G2}

---

## 4. How to Run Experiments

### 4.1 Environment setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install scikit-learn pandas numpy scipy matplotlib pymoo boruta
```

### 4.2 Smoke test (~20s, validates install)
```bash
python run_ablation_small.py student-mat.csv 0 abl_mat_h0.json
```

Expected output:
```
[student-mat.csv] h=0 N_tr=316 N_te=79 N_feat=45
  C1 ~7s IUS=55.79 α*=0.75
  C2 ~7s IUS=63.32 hasG1G2=True
  C3 ~4s IUS=28.61
  C4 ~1s IUS=53.94
```

If IUS values diverge by >5% from these, something broke — diff `ic_fs_v2.py` against the canonical version.

### 4.3 Full reproducibility pipeline (~15 min on laptop)
```bash
# Ablation: 6 runs (2 datasets × 3 horizons)
for ds in student-mat.csv student-por.csv; do
  for h in 0 1 2; do
    name="${ds%.csv}"; name="${name#student-}"
    python run_ablation_small.py "$ds" "$h" "abl_${name}_h${h}.json"
  done
done
python consolidate_ablation.py        # → ablation_consolidated.csv

# Sensitivity grid
python run_sensitivity.py student-mat.csv 1 sens_math_h1.csv
python summarize_sensitivity.py       # → pivots + Spearman

# Multi-seed statistics (8 seeds)
python run_stat_ext.py student-mat.csv 0 stat8_math_h0.csv

# Deployment-realistic eval (8 seeds, the paper's headline result)
python run_dre_multi.py student-mat.csv 0 dre_multi_math_h0.csv
python run_dre_multi.py student-por.csv 0 dre_multi_por_h0.csv

# Additional baselines
python run_baselines.py student-mat.csv 0 baselines_math_h0.csv
python run_baselines.py student-por.csv 0 baselines_por_h0.csv

# Final tables and figures
python final_tables.py
python master_table.py
python make_figures.py                 # → manuscript/fig{1,2,3,4}.png
```

### 4.4 Expected key numbers (sanity check)

UCI-Math, t=0, 8 seeds (`stat8_math_h0.csv` and `dre_multi_math_h0.csv`):

| Metric | IC-FS(full) | IC-FS(-temporal) | Δ | Test |
|---|---|---|---|---|
| F1_paper | 66.8±2.9 | 90.7±2.4 | +23.9 | (leakage) |
| F1_deploy | 66.8±2.9 | 58.2±3.8 | **−8.6** | Wilcoxon p=0.004 |
| IUS_deploy | 58.7±3.6 | 50.4±3.3 | **+8.3** | Cohen's d=+7.47 |

If reproduction shows different signs or much smaller effect sizes, investigate before proceeding.

---

## 5. Manuscript Status

### 5.1 What's done
- 7 sections drafted in `manuscript/icfs_manuscript.md` (~5,500 words)
- 4 publication-quality figures
- 6 tables embedded in markdown
- Causal Assumptions section with DAG (responds to reviewer Q3)
- Statistical framework with Bonferroni correction (Q4)
- Non-circular metric (precision@top-k%) in §A.4 (Q7)

### 5.2 What's missing (blockers for JLA submit)
1. **OULAD empirical validation** — only taxonomy is built. Reviewer will reject without it.
2. **Power analysis** for n=8 seeds — narrative is in §4.7 but no formal calculation
3. **LaTeX conversion** — currently markdown only

### 5.3 What's nice-to-have
- Subgroup fairness check (SPD, EOD, ABROCA by sex/SES)
- Delphi teacher survey for taxonomy validation (out of scope here)
- Multi-classifier robustness (RF only currently)

---

## 6. OULAD Extension Roadmap

### 6.1 Why it matters
Reviewer demanded a second dataset. UCI is small (n≤649) and old (2008). OULAD (n=32k, 2013–14) is the standard scale-up validation in EDM.

### 6.2 What's already in repo
- `oulad_taxonomy.py`: 31 features mapped to 4 tiers
  - Tier 0: 10 (gender, region, imd_band, age_band, ...)
  - Tier 1: 2 (date_registration, date_unregistration)
  - Tier 2: 14 (sum_click_to_date, forum_clicks, ...)
  - Tier 3: 5 (score_CMA1, score_TMA1, ...)
- `oulad_feature_engineering_guide()` returns SQL-style pipeline pseudocode

### 6.3 What's NOT in repo (must be built)
A `oulad_pipeline.py` (~300 LOC) that:
1. Loads 7 OULAD CSVs (~450MB) from https://analyse.kmi.open.ac.uk/open_dataset
2. For each prediction horizon h ∈ {0, 1, 2}:
   - Compute `horizon_cutoff_days = module_length × {0.0, 0.25, 0.50}` per student
   - Filter `studentVle.date <= cutoff` → aggregate clicks per activity_type
   - Filter `studentAssessment.date_submitted <= cutoff` → past CMA/TMA scores
3. Output one parquet per horizon with target `y = (final_result ∈ {Pass, Distinction})`

### 6.4 Critical OULAD pitfalls
- **Per-student horizon cutoff** — different `code_module` have different `module_length` (240 to 269 days). Don't use a global cutoff.
- **Tier-3 NaN sanity check** — at h=0, all CMA/TMA scores must be NaN (no leakage). If any non-NaN, pipeline is wrong.
- **Group-aware CV** — use `GroupKFold` on `code_module + code_presentation`, not random split. Different presentations are different cohorts.
- **Class imbalance** — ~30% Withdrawn. Use `class_weight='balanced'` in RF or stratified sampling.
- **One-hot blowup** — after `pd.get_dummies` on `region`, `code_module`, etc., feature count ~60. Verify `_resolve_parent` correctly maps `gender_M` → parent `gender`.

### 6.5 Integration test
```python
from ic_fs_v2 import get_actionability_score, get_temporal_availability
from oulad_taxonomy import TAXONOMY_OULAD

assert get_actionability_score('gender_M', TAXONOMY_OULAD) == 0.0
assert get_actionability_score('sum_click_to_date', TAXONOMY_OULAD) == 0.7
assert get_temporal_availability('score_CMA1', 0, TAXONOMY_OULAD) == False
assert get_temporal_availability('score_CMA1', 1, TAXONOMY_OULAD) == True
```

### 6.6 Running IC-FS on OULAD
```python
from ic_fs_v2 import ICFSPipeline
from oulad_taxonomy import TAXONOMY_OULAD

# After oulad_pipeline.py produces oulad_features_h{0,1,2}.parquet:
df = pd.read_parquet('oulad_features_h0.parquet')
X, y, names = preprocess_oulad(df)  # one-hot, fillna(0), pop 'y'
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=42, stratify=y)

pipe = ICFSPipeline(
    horizon=0, top_k=15, n_bootstrap=20,
    taxonomy=TAXONOMY_OULAD,
    alpha_values=[0.0, 0.25, 0.5, 0.75, 1.0]
)
pipe.fit(X_tr, y_tr, X_te, y_te, names, verbose=True)
```

Expected OULAD runtime per horizon: 15–30 min on laptop. Use parquet cache; don't re-aggregate.

---

## 7. Known Issues & Workarounds

### 7.1 Sandbox timeouts on bash_tool
If running in Anthropic sandbox, individual commands timeout at ~2 min. Use `run_ablation_small.py` (one horizon at a time) instead of `run_ablation_fast.py` (all horizons).

### 7.2 Joblib worker leaks
After interrupted runs: `pkill -9 -f 'joblib\|popen_loky\|multiprocessing'` to clean up.

### 7.3 Wilcoxon with n=8
Minimum p-value = 1/2^8 = 0.0039 (one-sided). Bonferroni-adjusted α=0.0167 is achievable but tight. For stronger claims, raise to 20+ seeds.

### 7.4 RF non-determinism
Even with `random_state=42`, multi-threaded RF produces tiny variation. Set `n_jobs=1` for byte-exact reproducibility.

### 7.5 NSGA-II underperforms IC-FS
With `pop_size=30, n_gen=20`, NSGA-II is in cold-start. Larger budget (pop=100, gen=100) likely closes the gap. Mention this caveat in §5.5.

---

## 8. Outstanding Reviewer Concerns (Round-2 Checklist)

| # | Concern | Status | Action |
|---|---|---|---|
| 1 | OULAD validation | **BLOCKER** | Build `oulad_pipeline.py`; run IC-FS at 3 horizons |
| 2 | Ablation study | ✅ Done | `ablation_consolidated.csv` |
| 3 | Sensitivity analysis | ✅ Done | `sens_math_h1.csv`, fig2 |
| 4 | Statistical tests | ✅ Done | `stat8_math_h0.csv`, p<0.0167 |
| 5 | Causal framing | ✅ Done | Manuscript §2 |
| 6 | Stronger baselines | ✅ Done | NSGA-II, Stab.Sel., Boruta |
| 7 | Circular evaluation | ✅ Done | precision@top-k in §A.4 |
| 8 | Code hardening | ✅ Done | `ic_fs_v2.py` Step A fixes |
| 9 | Multi-seed (≥20) | ⚠️ Partial | 8 seeds; raise to 20 |
| 10 | LaTeX format | ⚠️ Pending | Convert .md → .tex |

---

## 9. Tasks Claude Code Should Be Asked To Do

### 9.1 Easy (≤1 hour)
- "Re-run statistics with 20 seeds": edit `RNG_SEEDS` in `run_stat_ext.py` and `run_dre_multi.py`
- "Add F1-macro alongside F1-weighted": modify `_eval_subset()` in `run_ablation_fast.py`
- "Try logistic regression instead of RF": replace classifier in `_eval_subset()`
- "Add precision/recall@10% in addition to @20%": extend `evaluate_with_external_metric()` in `ic_fs_v2.py`
- "Generate LaTeX tables from CSVs": add `df.to_latex()` calls in `final_tables.py`

### 9.2 Medium (1–4 hours)
- "Build OULAD pipeline" (see §6.3) — most valuable single task
- "Add subgroup fairness metrics": new `fairness.py` computing SPD/EOD/ABROCA per sex, Medu band
- "Convert manuscript .md → JLA LaTeX template": pandoc + manual table fixes
- "Stratified CV instead of single split": modify all `train_test_split` calls
- "Cost-weighted IUS variant": add `compute_ius_cost(f1, sel, h, costs)` to `ic_fs_v2.py`

### 9.3 Hard (>4 hours)
- "Implement causal recourse selection (Karimi et al. 2021)" — requires DoWhy or causalml
- "Replace scalarized α-sweep with proper NSGA-II at population 100": tune pymoo
- "Run on third dataset (xAPI-Edu, ASSISTments)": new taxonomy + pipeline + integration
- "Delphi teacher validation study simulator": Bayesian aggregation of expert ratings

---

## 10. Single Source of Truth Summary

If anything contradicts this paper.md elsewhere in the repo, this file wins. Key invariants:

```
Project    : IC-FS (Intervention-Constrained Feature Selection)
Target     : Journal of Learning Analytics, Round-2 revision
Datasets   : UCI Student Performance (math, por); OULAD pending
Core idea  : Temporal filter + actionability soft penalty + multiplicative IUS
Key result : IC-FS(full) > DE-FS-style on DRE-IUS, p=0.004, d=+7.47 (Math, t=0, n=8)
Status     : Pha 1 (experiments) ✅; Pha 2 (manuscript) drafted; OULAD ✗
Bottleneck : Build oulad_pipeline.py (~300 LOC, ~30 min compute)
```

When given a new task, Claude Code should:
1. Read this file first
2. Check `manuscript/icfs_manuscript.md` to understand current narrative
3. Run smoke test (§4.2) before modifying core files
4. Diff `ic_fs_v2.py` after changes — Step A fixes are non-negotiable
5. Verify expected numbers (§4.4) didn't drift after changes
6. Update `manuscript/icfs_manuscript.md` if results changed materially

End of guide.