"""
Compute Precision@20% and Recall@20% for h=0 results to provide operational context
for the 56% F1 score.

These metrics answer: "At a fixed intervention budget of 20% of students (realistic
institutional constraint), what fraction of at-risk students are correctly identified?"
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from ic_fs_v2 import (
    filter_by_horizon,
    actionability_ratio_available,
    compute_ius_deploy,
    get_temporal_availability,
)
from src.icfs.taxonomy_oulad import TAXONOMY_OULAD
from experiments.oulad.preprocess_oulad import load_oulad_horizon, preprocess_oulad

RNG_SEEDS = [42, 123, 456, 789, 1011, 2024, 3033, 4044]
N_TREES = 100  # Issue 6 fix: match ICFSPipeline n_estimators=100 (paper §3.7)
HORIZON = 0
TOP_K_PCT = 0.20  # Top 20% for intervention


def precision_recall_at_top_k(y_true, y_proba, top_k_pct=0.20):
    """
    Compute precision and recall at top k% of predictions.

    Args:
        y_true: True labels (1 = at-risk, 0 = not at-risk)
        y_proba: Predicted probabilities for at-risk class
        top_k_pct: Fraction of population to intervene on (e.g., 0.20 = top 20%)

    Returns:
        precision, recall at top k%
    """
    n = len(y_true)
    k = int(n * top_k_pct)

    # Issue 2 fix: sort ASCENDING by P(Pass) → lowest P(Pass) = highest at-risk.
    # Previously sorted descending, which ranked PASSING students first.
    top_k_idx = np.argsort(y_proba)[:k]

    # Issue 2 fix: count Fail/Withdrawn (y=0) in top-k.
    # Previously used y_true.sum() which counted y=1 (Pass) — the wrong class.
    tp = (y_true[top_k_idx] == 0).sum()
    precision = tp / k if k > 0 else 0.0

    # Recall: of all at-risk students (y=0), what fraction did we capture?
    total_at_risk = (y_true == 0).sum()
    recall = tp / total_at_risk if total_at_risk > 0 else 0.0

    return precision, recall


def evaluate_one_seed(df_raw, seed):
    """Evaluate IC-FS(full) and IC-FS(-temporal) for one seed at h=0."""
    X, y, names = preprocess_oulad(df_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # ─── IC-FS(full): temporal filter ON ───
    available = filter_by_horizon(names, HORIZON, TAXONOMY_OULAD)
    if not available:
        raise RuntimeError(f"No features available at h={HORIZON}")
    avail_idx = [names.index(f) for f in available]
    X_tr_full = X_tr[:, avail_idx]
    X_te_full = X_te[:, avail_idx]

    # Train simple RF on available features
    rf_full = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                      n_jobs=-1, class_weight='balanced')
    rf_full.fit(X_tr_full, y_tr)
    y_proba_full = rf_full.predict_proba(X_te_full)[:, 1]
    y_pred_full = rf_full.predict(X_te_full)

    f1_full = f1_score(y_te, y_pred_full, average='weighted', zero_division=0)
    prec_full, rec_full = precision_recall_at_top_k(y_te, y_proba_full, TOP_K_PCT)

    # ─── IC-FS(-temporal): no filter, use all features ───
    rf_notemp = RandomForestClassifier(n_estimators=N_TREES, random_state=seed,
                                        n_jobs=-1, class_weight='balanced')
    rf_notemp.fit(X_tr, y_tr)
    y_proba_notemp = rf_notemp.predict_proba(X_te)[:, 1]
    y_pred_notemp = rf_notemp.predict(X_te)

    f1_notemp = f1_score(y_te, y_pred_notemp, average='weighted', zero_division=0)
    prec_notemp, rec_notemp = precision_recall_at_top_k(y_te, y_proba_notemp, TOP_K_PCT)

    # ─── Majority-class baseline ───
    majority_class = 1 if y_tr.mean() > 0.5 else 0
    y_pred_majority = np.full_like(y_te, majority_class)
    f1_majority = f1_score(y_te, y_pred_majority, average='weighted', zero_division=0)
    # For majority-class, "probability" is just 1.0 for predicted class
    y_proba_majority = np.full(len(y_te), 0.5)  # No information
    prec_majority, rec_majority = precision_recall_at_top_k(y_te, y_proba_majority, TOP_K_PCT)

    return {
        'seed': seed,
        'f1_full': f1_full * 100,
        'prec20_full': prec_full * 100,
        'rec20_full': rec_full * 100,
        'f1_notemp': f1_notemp * 100,
        'prec20_notemp': prec_notemp * 100,
        'rec20_notemp': rec_notemp * 100,
        'f1_majority': f1_majority * 100,
        'pass_rate': y_tr.mean() * 100,
    }


def main():
    print("=" * 100)
    print(f"INTERVENTION METRICS AT h=0: Precision@{int(TOP_K_PCT*100)}% and Recall@{int(TOP_K_PCT*100)}%")
    print("=" * 100)
    print()
    print("Question: At a fixed intervention budget (top 20% of students by predicted risk),")
    print("          what fraction of truly at-risk students are captured?")
    print()

    # Load data
    df_raw = load_oulad_horizon(HORIZON)
    print(f"Loaded {len(df_raw)} enrollments")
    print(f"Pass rate: {df_raw['y'].mean():.3f}")
    print()

    # Run across all seeds
    results = []
    for seed in RNG_SEEDS:
        print(f"Running seed {seed}...", end=' ')
        r = evaluate_one_seed(df_raw, seed)
        results.append(r)
        print(f"✓ F1={r['f1_full']:.1f}% Prec@20%={r['prec20_full']:.1f}% Rec@20%={r['rec20_full']:.1f}%")

    df = pd.DataFrame(results)

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary Statistics
    # ═══════════════════════════════════════════════════════════════════════════
    print()
    print("=" * 100)
    print("RESULTS SUMMARY (8 seeds)")
    print("=" * 100)
    print()

    print("IC-FS(full) - Temporal filter ON:")
    print(f"  F1 (weighted):      {df['f1_full'].mean():5.1f}% ± {df['f1_full'].std():4.1f}%")
    print(f"  Precision@20%:      {df['prec20_full'].mean():5.1f}% ± {df['prec20_full'].std():4.1f}%")
    print(f"  Recall@20%:         {df['rec20_full'].mean():5.1f}% ± {df['rec20_full'].std():4.1f}%")
    print()

    print("IC-FS(-temporal) - No filter:")
    print(f"  F1 (weighted):      {df['f1_notemp'].mean():5.1f}% ± {df['f1_notemp'].std():4.1f}%")
    print(f"  Precision@20%:      {df['prec20_notemp'].mean():5.1f}% ± {df['prec20_notemp'].std():4.1f}%")
    print(f"  Recall@20%:         {df['rec20_notemp'].mean():5.1f}% ± {df['rec20_notemp'].std():4.1f}%")
    print()

    print("Majority-class baseline:")
    print(f"  F1 (weighted):      {df['f1_majority'].mean():5.1f}%")
    print(f"  Precision@20%:      ~{df['pass_rate'].mean():5.1f}% (random selection from population)")
    print(f"  Recall@20%:         ~20.0% (by definition, top 20% of random selection)")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # Interpretation
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("INTERPRETATION FOR PAPER")
    print("=" * 100)
    print()

    prec_mean = df['prec20_full'].mean()
    rec_mean = df['rec20_full'].mean()
    baseline_prec = 100 - df['pass_rate'].mean()  # Fraction of population that's at-risk
    lift = prec_mean / baseline_prec if baseline_prec > 0 else 0

    print(f"At course start (h=0), IC-FS(full) achieves F1 = {df['f1_full'].mean():.1f}%, which is modest")
    print(f"compared to the majority-class baseline (F1 = {df['f1_majority'].mean():.1f}%). However, this")
    print(f"understates intervention utility.")
    print()
    print(f"If an institution can intervene on 20% of students (a realistic budget constraint),")
    print(f"IC-FS(full) achieves:")
    print(f"  • Precision@20% = {prec_mean:.1f}%: Of students we intervene on, {prec_mean:.0f}% are truly at-risk")
    print(f"  • Recall@20% = {rec_mean:.1f}%: We capture {rec_mean:.0f}% of all at-risk students")
    print()
    print(f"Random selection from the population would yield Precision ≈ {baseline_prec:.1f}% (the base rate")
    print(f"of at-risk students). IC-FS(full) achieves {lift:.2f}× lift over random selection.")
    print()
    print(f"For an early-warning system triggered at course start, this represents actionable predictive")
    print(f"value: instructors can focus retention efforts on a targeted subset with {lift:.2f}× higher")
    print(f"probability of being at-risk, despite the low overall F1.")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════════════════════
    out_path = project_root / "results" / "oulad" / "intervention_metrics_h0.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print()

    # ═══════════════════════════════════════════════════════════════════════════
    # Suggested text for paper
    # ═══════════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("SUGGESTED TEXT FOR RESULTS SECTION")
    print("=" * 100)
    print()
    print(f'''
IC-FS(full) achieves F1 = {df['f1_full'].mean():.1f}% at h=0, a modest improvement over the
majority-class baseline (F1 = {df['f1_majority'].mean():.1f}%) and comparable to all baselines
(NSGA-II: 56.5%, Boruta: 56.0%, StabilitySelection: 55.5%). This reflects the structural scarcity
of predictive signal in OULAD's pre-course demographic features: before any student behavior is
observed, the available feature space contains only registration timing and demographic indicators,
none of which carry strong causal signals about within-course performance independent of module effects.

However, overall F1 understates intervention utility. The critical question for early-warning
deployment is not "what is the global classification accuracy?" but "at a fixed intervention
budget, what fraction of at-risk students are correctly identified?" If an institution can
intervene on 20% of students—a realistic resource constraint—IC-FS(full) achieves Precision@20% =
{prec_mean:.1f}% and Recall@20% = {rec_mean:.1f}%, correctly identifying {rec_mean:.0f}% of all
at-risk students. This represents {lift:.2f}× lift over random selection from the population
(baseline precision ≈ {baseline_prec:.1f}%, the base rate of at-risk students). For an advisor
deploying retention interventions at course start, this targeted prediction—concentrating efforts
on a subset with {lift:.2f}× higher risk—is operationally valuable despite the low overall F1.
'''.strip())
    print()


if __name__ == "__main__":
    main()
