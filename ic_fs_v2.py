"""
================================================================================
IC-FS v2: Intervention-Constrained Feature Selection (HARDENED)
================================================================================
"""

from __future__ import annotations
import warnings
import itertools
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: ACTIONABILITY TAXONOMY (unchanged structure)
# ─────────────────────────────────────────────────────────────────────────────

class Tier(IntEnum):
    NON_ACTIONABLE = 0
    PRE_SEMESTER   = 1
    MID_SEMESTER   = 2
    PAST_GRADE     = 3


@dataclass
class FeatureProfile:
    name: str
    tier: Tier
    available_at: List[int]
    description: str
    educational_rationale: str


def build_uci_taxonomy() -> Dict[str, FeatureProfile]:
    """Taxonomy cho UCI Student Performance dataset."""
    profiles = [
        # Tier 0 — non-actionable demographics/SES
        FeatureProfile("school",     Tier.NON_ACTIONABLE, [0,1,2], "Binary school id", "Institution-level; non-modifiable"),
        FeatureProfile("sex",        Tier.NON_ACTIONABLE, [0,1,2], "Sex F/M", "Protected attribute"),
        FeatureProfile("age",        Tier.NON_ACTIONABLE, [0,1,2], "Age 15-22", "Grade-repetition proxy"),
        FeatureProfile("address",    Tier.NON_ACTIONABLE, [0,1,2], "Urban/rural", "SES proxy"),
        FeatureProfile("famsize",    Tier.NON_ACTIONABLE, [0,1,2], "Family size", "Structure"),
        FeatureProfile("Pstatus",    Tier.NON_ACTIONABLE, [0,1,2], "Parents cohab", "Household"),
        FeatureProfile("Medu",       Tier.NON_ACTIONABLE, [0,1,2], "Mother edu 0-4", "SES"),
        FeatureProfile("Fedu",       Tier.NON_ACTIONABLE, [0,1,2], "Father edu 0-4", "SES"),
        FeatureProfile("Mjob",       Tier.NON_ACTIONABLE, [0,1,2], "Mother job", "Occupational SES"),
        FeatureProfile("Fjob",       Tier.NON_ACTIONABLE, [0,1,2], "Father job", "Occupational SES"),
        FeatureProfile("reason",     Tier.NON_ACTIONABLE, [0,1,2], "Reason for school", "Retrospective"),
        FeatureProfile("guardian",   Tier.NON_ACTIONABLE, [0,1,2], "Guardian", "Family structure"),
        FeatureProfile("traveltime", Tier.NON_ACTIONABLE, [0,1,2], "Travel time", "Geographic"),
        FeatureProfile("failures",   Tier.NON_ACTIONABLE, [0,1,2], "Past failures", "Historical; non-retroactive"),

        # Tier 1 — pre-semester actionable
        FeatureProfile("studytime",  Tier.PRE_SEMESTER,   [0,1,2], "Weekly study 1-4", "KEY target: study hours"),
        FeatureProfile("schoolsup",  Tier.PRE_SEMESTER,   [0,1,2], "School support", "Direct intervention"),
        FeatureProfile("famsup",     Tier.PRE_SEMESTER,   [0,1,2], "Family support", "Parent engagement"),
        FeatureProfile("paid",       Tier.PRE_SEMESTER,   [0,1,2], "Paid classes", "Scholarship target"),
        FeatureProfile("activities", Tier.PRE_SEMESTER,   [0,1,2], "Extracurricular", "Schedule balance"),
        FeatureProfile("nursery",    Tier.PRE_SEMESTER,   [0,1,2], "Nursery attended", "Early-ed history"),
        FeatureProfile("higher",     Tier.PRE_SEMESTER,   [0,1,2], "Wants higher ed", "Aspiration: goal-setting"),
        FeatureProfile("internet",   Tier.PRE_SEMESTER,   [0,1,2], "Internet access", "Digital equity"),
        FeatureProfile("romantic",   Tier.PRE_SEMESTER,   [0,1,2], "Romantic relation", "Counselor-addressable"),

        # Tier 2 — mid-semester observable
        FeatureProfile("famrel",     Tier.MID_SEMESTER,   [0,1,2], "Family quality 1-5", "Family counseling"),
        FeatureProfile("freetime",   Tier.MID_SEMESTER,   [0,1,2], "Free time 1-5", "Time management"),
        FeatureProfile("goout",      Tier.MID_SEMESTER,   [0,1,2], "Going out 1-5", "Social balance"),
        FeatureProfile("Dalc",       Tier.MID_SEMESTER,   [0,1,2], "Workday alcohol", "Health intervention"),
        FeatureProfile("Walc",       Tier.MID_SEMESTER,   [0,1,2], "Weekend alcohol", "Health intervention"),
        FeatureProfile("health",     Tier.MID_SEMESTER,   [0,1,2], "Health 1-5", "Nurse referral"),
        FeatureProfile("absences",   Tier.MID_SEMESTER,   [1,2],   "Absences", "Attendance alert"),

        # Tier 3 — past grades (non-retroactive)
        FeatureProfile("G1",         Tier.PAST_GRADE,     [1,2],   "G1 (0-20)", "Predictive, non-retroactive"),
        FeatureProfile("G2",         Tier.PAST_GRADE,     [2],     "G2 (0-20)", "Near-tautological with G3"),
    ]
    return {p.name: p for p in profiles}


TAXONOMY_UCI = build_uci_taxonomy()

ACTIONABILITY_WEIGHTS = {
    Tier.NON_ACTIONABLE: 0.0,
    Tier.PRE_SEMESTER:   1.0,
    Tier.MID_SEMESTER:   0.7,
    Tier.PAST_GRADE:     0.0,
}


def _resolve_parent(feature_name: str, taxonomy: Dict[str, FeatureProfile]
                    ) -> Optional[FeatureProfile]:
    """Resolve one-hot encoded child (e.g. 'Mjob_teacher') back to parent profile."""
    if feature_name in taxonomy:
        return taxonomy[feature_name]
    for parent_name, profile in taxonomy.items():
        if feature_name.startswith(parent_name + "_"):
            return profile
    return None


def get_actionability_score(feature_name: str,
                             taxonomy: Dict[str, FeatureProfile] = None,
                             weights: Dict[Tier, float] = None,
                             strict: bool = False) -> float:
    """
    FIX A2: Unknown feature => 0.0 (conservative) + warning, không còn 0.5 silent.

    Args:
        strict: nếu True, raise KeyError cho unknown feature thay vì warn.
    """
    tax = taxonomy if taxonomy is not None else TAXONOMY_UCI
    w   = weights  if weights  is not None else ACTIONABILITY_WEIGHTS
    profile = _resolve_parent(feature_name, tax)
    if profile is not None:
        return w[profile.tier]
    msg = f"[IC-FS] Unknown feature '{feature_name}' in taxonomy; defaulting to 0.0"
    if strict:
        raise KeyError(msg)
    warnings.warn(msg, stacklevel=2)
    return 0.0


def get_temporal_availability(feature_name: str,
                               horizon: int,
                               taxonomy: Dict[str, FeatureProfile] = None,
                               strict: bool = False) -> bool:
    """
    FIX A1: Unknown feature => False (+ warning), không còn silent True.
    """
    tax = taxonomy if taxonomy is not None else TAXONOMY_UCI
    profile = _resolve_parent(feature_name, tax)
    if profile is not None:
        return horizon in profile.available_at
    msg = (f"[IC-FS] Unknown feature '{feature_name}' in taxonomy at "
           f"horizon={horizon}; defaulting to UNAVAILABLE to prevent leakage")
    if strict:
        raise KeyError(msg)
    warnings.warn(msg, stacklevel=2)
    return False


def filter_by_horizon(feature_names: List[str],
                       horizon: int,
                       taxonomy: Dict[str, FeatureProfile] = None,
                       strict: bool = False) -> List[str]:
    """Hard temporal filter. Với fix A1, unknown features bị loại."""
    return [f for f in feature_names
            if get_temporal_availability(f, horizon, taxonomy, strict)]


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: METRICS (IUS + external metrics for non-circular eval)
# ─────────────────────────────────────────────────────────────────────────────

def actionability_ratio(selected_features: List[str],
                         taxonomy: Dict[str, FeatureProfile] = None) -> float:
    if not selected_features:
        return 0.0
    weights = [get_actionability_score(f, taxonomy) for f in selected_features]
    return float(np.mean(weights))


def actionability_ratio_available(
    selected_features: List[str],
    horizon: int,
    taxonomy: Dict[str, FeatureProfile] = None,
    weights: Dict[Tier, float] = None,
) -> float:
    """
    AR_available = |S_actionable ∩ S_available_at_h| / |S|

    The CORRECT intervention utility ratio for deployment.
    Unlike actionability_ratio(), this function zeros out the
    contribution of any feature that is temporally unavailable at `horizon`,
    regardless of its pedagogical actionability in theory.

    This eliminates the TVS tautology: a feature that is actionable but
    not yet observable at horizon h has zero practical intervention value.

    Args:
        selected_features: List of selected feature names.
        horizon: Deployment time point (0, 1, 2).
        taxonomy: Feature taxonomy (defaults to TAXONOMY_UCI).
        weights: Actionability weights per Tier (defaults to ACTIONABILITY_WEIGHTS).

    Returns:
        Float in [0, 1]. Returns 0.0 if selected_features is empty.
    """
    if not selected_features:
        return 0.0
    tax = taxonomy if taxonomy is not None else TAXONOMY_UCI
    w   = weights  if weights  is not None else ACTIONABILITY_WEIGHTS

    scores = []
    for f in selected_features:
        # Gate 1: Is the feature available at this horizon?
        if not get_temporal_availability(f, horizon, tax):
            scores.append(0.0)
            continue
        # Gate 2: Is it pedagogically actionable?
        scores.append(get_actionability_score(f, tax, w))

    return float(np.mean(scores))


def temporal_validity_score(selected_features: List[str],
                             horizon: int,
                             taxonomy: Dict[str, FeatureProfile] = None) -> float:
    if not selected_features:
        return 0.0
    valid = sum(1 for f in selected_features
                if get_temporal_availability(f, horizon, taxonomy))
    return valid / len(selected_features)


def compute_ius(f1: float,
                selected_features: List[str],
                horizon: int,
                taxonomy: Dict[str, FeatureProfile] = None) -> float:
    """
    DEPRECATED — NOT USED IN ICFSPipeline.

    Original formula: IUS = F1 × AR × TVS  (three-way multiplicative).

    Why retired:
      - TVS is always 1.0 for IC-FS(full) because filter_by_horizon() removes
        unavailable features before selection, making the TVS gate a no-op.
      - AR ignores temporal availability, so it double-counts features that are
        actionable in principle but not yet observable at horizon h.
      - The successor metrics are:
          compute_ius_paper()  → F1_paper × AR        (inflated, kept for demos)
          compute_ius_deploy() → F1_deploy × AR_avail (primary, deployment-honest)

    Retained only so that unit-test suites can demonstrate the value gap
    between old and new formulations.  Raises DeprecationWarning on every call
    to prevent accidental use in new experiment scripts.
    """
    warnings.warn(
        "compute_ius() (F1 × AR × TVS) is deprecated and not used in "
        "ICFSPipeline.  Use compute_ius_deploy() for deployment-honest "
        "evaluation, or compute_ius_paper() for the inflated comparison metric.",
        DeprecationWarning,
        stacklevel=2,
    )
    ar  = actionability_ratio(selected_features, taxonomy)
    tvs = temporal_validity_score(selected_features, horizon, taxonomy)
    return f1 * ar * tvs


def compute_ius_geo(f1: float,
                     selected_features: List[str],
                     horizon: int,
                     taxonomy: Dict[str, FeatureProfile] = None) -> float:
    """Geometric-mean variant (for robustness comparison in sensitivity study)."""
    ar  = actionability_ratio(selected_features, taxonomy)
    tvs = temporal_validity_score(selected_features, horizon, taxonomy)
    prod = max(f1 * ar * tvs, 0.0)
    return prod ** (1/3)


def compute_ius_deploy(
    f1_deploy: float,
    selected_features: List[str],
    horizon: int,
    taxonomy: Dict[str, FeatureProfile] = None,
) -> float:
    """
    IUS_deploy = F1_deploy × AR_available   (Path A — the clean formulation)

    Both components are independent:
      - f1_deploy: predictive outcome under DRE masking protocol
      - AR_available: structural audit via set intersection (actionable ∩ available)

    This replaces the original compute_ius() which used F1 × AR × TVS,
    where TVS was always 1.0 due to the hard temporal filter applied upstream.

    Args:
        f1_deploy: F1-score from DRE evaluation (features masked if unavailable).
        selected_features: Final selected feature set.
        horizon: Deployment time point.
        taxonomy: Feature taxonomy.

    Returns:
        Float in [0, 1].
    """
    ar_avail = actionability_ratio_available(selected_features, horizon, taxonomy)
    return f1_deploy * ar_avail


def compute_ius_paper(
    f1_paper: float,
    selected_features: List[str],
    horizon: int,
    taxonomy: Dict[str, FeatureProfile] = None,
) -> float:
    """
    IUS_paper = F1_paper × AR   (old metric, retained for comparison logging)

    Kept to allow direct demonstration of the gap between the inflated
    paper-style IUS and the deployment-honest IUS in the results tables.
    TVS is intentionally excluded here (it was always 1.0 anyway).
    """
    ar = actionability_ratio(selected_features, taxonomy)
    return f1_paper * ar


def apply_dre_mask(
    X_tr_sel: np.ndarray,
    X_te_sel: np.ndarray,
    selected: List[str],
    horizon: int,
    taxonomy: Dict[str, FeatureProfile] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shared asymmetric DRE masking utility.

    This is the single authoritative implementation of the DRE protocol.
    All experiment scripts (run_oulad_baselines.py, run_oulad_statistics.py,
    run_oulad_dre.py) must call this function instead of duplicating the
    masking loop, to guarantee protocol uniformity.

    Protocol:
      - Training data is left UNMASKED (historical records have full observations).
      - Columns in X_te that are temporally unavailable at `horizon` are replaced
        with the corresponding training-set column means.
      - The returned X_tr_sel is the same object (unchanged); X_te_deploy is a copy.

    For methods that applied filter_by_horizon() upstream (IC-FS full,
    NSGA-II, Boruta, Stability Selection), every selected feature is already
    temporally available, so the inner loop is a no-op and X_te_deploy ≡ X_te_sel.
    The function is still called for code uniformity and future safety.

    Args:
        X_tr_sel: Training slice, shape (n_train, |selected|). Not modified.
        X_te_sel: Test slice, shape (n_test, |selected|). Source for copy.
        selected: Feature names aligned with columns of X_tr_sel / X_te_sel.
        horizon:  Deployment time point (0, 1, 2).
        taxonomy: Feature taxonomy (defaults to TAXONOMY_UCI).

    Returns:
        (X_tr_sel, X_te_deploy): X_tr unchanged; X_te_deploy has masked columns.
    """
    tax = taxonomy if taxonomy is not None else TAXONOMY_UCI
    train_means = X_tr_sel.mean(axis=0)
    X_te_deploy = X_te_sel.copy().astype(np.float64)
    for j, feat_name in enumerate(selected):
        if not get_temporal_availability(feat_name, horizon, tax):
            X_te_deploy[:, j] = train_means[j]
    return X_tr_sel, X_te_deploy

def precision_at_top_k_actionable(y_true: np.ndarray,
                                    y_prob: np.ndarray,
                                    k_pct: float = 0.2) -> float:
    """
    Precision@top-k% — không chứa AR, tránh circular eval.

    Đo: trong k% sinh viên model flag là rủi ro cao nhất, bao nhiêu % thực sự fail.
    Đây là metric quan trọng cho Early Warning System thực tế.
    """
    n = len(y_true)
    k = max(1, int(n * k_pct))
    # Top-k theo prob của class "fail" (giả định fail=0 trong nhị phân)
    # y_prob là prob của class positive (pass=1), nên bottom-k = most at-risk
    top_idx = np.argsort(y_prob)[:k]  # k học sinh dự đoán rủi ro cao nhất (prob pass thấp)
    # y_true[top_idx] — fraction thực sự fail = (1 - mean)
    actual_fail_rate = 1 - np.mean(y_true[top_idx])
    return float(actual_fail_rate)


def top_k_intervention_hit_rate(y_true: np.ndarray,
                                  y_prob: np.ndarray,
                                  budget_pct: float = 0.2) -> float:
    """
    Recall@top-k% — trong học sinh fail thực tế, bao nhiêu % được flag?
    Đây là metric "intervention coverage" cho policy realistic với budget hạn chế.
    """
    n = len(y_true)
    k = max(1, int(n * budget_pct))
    top_idx = np.argsort(y_prob)[:k]
    flagged_as_fail = np.sum(1 - y_true[top_idx])
    total_fail = np.sum(1 - y_true)
    return float(flagged_as_fail / total_fail) if total_fail > 0 else 0.0


def evaluate_with_external_metric(y_true: np.ndarray,
                                    y_prob: np.ndarray,
                                    selected_features: List[str],
                                    horizon: int,
                                    taxonomy: Dict[str, FeatureProfile] = None,
                                    budget_pct: float = 0.2) -> dict:
    """
    Returns dict với:
      - precision_at_topk: độ chính xác khi flag top-k% rủi ro cao
      - recall_at_topk   : recall/coverage trong fail thực tế
      - tvs              : (reported for transparency, không dùng làm selection obj)
    """
    p_at_k = precision_at_top_k_actionable(y_true, y_prob, budget_pct)
    r_at_k = top_k_intervention_hit_rate(y_true, y_prob, budget_pct)
    tvs    = temporal_validity_score(selected_features, horizon, taxonomy)
    return {
        "precision_at_topk": p_at_k,
        "recall_at_topk":    r_at_k,
        "temporal_validity": tvs,
        "budget_pct":        budget_pct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: FEATURE SCORING & IC-FS SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def feature_scores_for_selection(X: np.ndarray,
                                   y: np.ndarray,
                                   feature_names: List[str],
                                   taxonomy: Dict[str, FeatureProfile] = None,
                                   random_state: int = RANDOM_STATE,
                                   ) -> pd.DataFrame:
    """
    Compute four-component ensemble feature scores for IC-FS selection.

    Note on asymmetric scaling: chi-squared is computed on MinMax-scaled
    features (required — chi2 needs non-negative inputs); MI, Pearson
    correlation, and RF importance are computed on the original unscaled X.
    This asymmetry is intentional and documented here explicitly.
    """
    scaler = MinMaxScaler()
    X_nn = scaler.fit_transform(X)

    chi2_raw, _ = chi2(X_nn, y)            # needs non-negative → scaled
    chi2_raw = np.nan_to_num(chi2_raw, nan=0.0)

    mi_raw = mutual_info_classif(X, y, random_state=random_state)   # FIX #6

    corr_raw = np.array([abs(np.corrcoef(X[:, j], y)[0, 1])
                          for j in range(X.shape[1])])
    corr_raw = np.nan_to_num(corr_raw, nan=0.0)

    rf = RandomForestClassifier(n_estimators=100,
                                 random_state=random_state, n_jobs=1)  # FIX #6
    rf.fit(X, y)
    rf_imp = rf.feature_importances_

    def norm(v):
        mn, mx = v.min(), v.max()
        return (v - mn) / (mx - mn + 1e-10)

    action_scores = np.array([get_actionability_score(f, taxonomy)
                               for f in feature_names])

    df = pd.DataFrame({
        "feature":      feature_names,
        "chi2":         norm(chi2_raw),
        "mutual_info":  norm(mi_raw),
        "correlation":  norm(corr_raw),
        "rf_importance":norm(rf_imp),
        "actionability":action_scores,
    })
    df["pred_score"] = (df["chi2"] + df["mutual_info"] +
                        df["correlation"] + df["rf_importance"]) / 4
    return df


def ic_fs_select(score_df: pd.DataFrame,
                  alpha: float,
                  top_k: int) -> List[str]:
    """α·pred_score + (1-α)·actionability, chọn top-k."""
    beta = 1.0 - alpha
    df = score_df.copy()
    df["ic_score"] = alpha * df["pred_score"] + beta * df["actionability"]
    return df.nlargest(top_k, "ic_score")["feature"].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: SOLUTION / PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SolutionPoint:
    alpha: float
    beta: float
    selected_features: List[str]
    accuracy: float
    f1: float           # F1 from standard evaluation (paper-style)
    f1_deploy: float    # F1 from DRE protocol (honest evaluation)
    ar: float           # AR (old: ignores temporal availability)
    ar_available: float # AR_available = |S_action ∩ S_avail| / |S|
    tvs: float          # Kept for backward-compatibility logging only
    ius: float          # Old IUS = F1 × AR × TVS (for comparison tables)
    ius_deploy: float   # New IUS_deploy = F1_deploy × AR_available
    ius_geo: float
    n_features: int
    stability: float = 0.0
    cv_mean: float = 0.0
    cv_std: float = 0.0
    # External metrics (non-circular)
    precision_at_topk: float = 0.0
    recall_at_topk:    float = 0.0


class ICFSPipeline:
    """Pipeline IC-FS v2 với bootstrap seed độc lập + external metrics."""

    def __init__(self,
                  horizon: int = 0,
                  alpha_values: Optional[List[float]] = None,
                  top_k: int = 12,
                  n_bootstrap: int = 50,
                  cv_folds: int = 10,
                  taxonomy: Dict[str, FeatureProfile] = None,
                  bootstrap_base_seed: int = 2026,
                  random_state: int = RANDOM_STATE):
        self.horizon = horizon
        # FIX #3: Default α-grid matches Algorithm 1 (5 values, not linspace(0,1,11)).
        # Callers may override via alpha_values= for finer sweeps.
        self.alpha_values = alpha_values or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.top_k = top_k
        self.n_bootstrap = n_bootstrap
        self.cv_folds = cv_folds
        self.taxonomy = taxonomy if taxonomy is not None else TAXONOMY_UCI
        self.bootstrap_base_seed = bootstrap_base_seed
        # Issue 3 fix: store random_state so val_seed varies with the outer seed
        self.random_state = random_state
        self.solutions_: List[SolutionPoint] = []
        self.score_df_: Optional[pd.DataFrame] = None

    # FIX A3: Bootstrap seed độc lập cho mỗi alpha
    def _bootstrap_stability(self,
                              X: np.ndarray, y: np.ndarray,
                              feature_names: List[str],
                              alpha: float, k: int,
                              bootstrap_seed: int) -> float:
        """
        Jaccard stability across B bootstrap resamples.

        The index-sampling RNG (np.random.RandomState(bootstrap_seed)) is kept
        separate from the scorer seeds so the two sources of randomness are
        independently controllable.
        """
        n = len(y)
        selected_sets = []
        rng = np.random.RandomState(bootstrap_seed)
        for b_i in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_bs, y_bs = X[idx], y[idx]
            if len(np.unique(y_bs)) < 2:
                continue
            score_df = feature_scores_for_selection(
                X_bs, y_bs, feature_names, self.taxonomy,
                random_state=bootstrap_seed + b_i,
            )
            selected = ic_fs_select(score_df, alpha, k)
            selected_sets.append(set(selected))

        if len(selected_sets) < 2:
            return 0.0
        pairs = list(itertools.combinations(range(len(selected_sets)), 2))
        jacc = []
        for i, j in pairs:
            A, B = selected_sets[i], selected_sets[j]
            inter, union = len(A & B), len(A | B)
            jacc.append(inter / union if union > 0 else 1.0)
        return float(np.mean(jacc))

    def _compute_dre_f1(self,
                         clf,
                         X_tr_sel: np.ndarray,
                         y_tr: np.ndarray,
                         X_te_sel: np.ndarray,
                         y_te: np.ndarray,
                         selected: List[str]) -> float:
        """
        Deployment-Realistic Evaluation (DRE) with FIXED asymmetric masking.

        Protocol (matches Algorithm 1, Steps 2–5 in the paper):
          - Compute training-set column means from the UNMASKED X_tr_sel.
          - Train clf_deploy on the UNMASKED training matrix (historical records
            have all features fully observed at model training time).
          - Substitute unavailable columns ONLY in X_te_deploy with training
            means (simulating inference time when future features do not exist).
          - Return F1 on the masked test matrix.

        For IC-FS(full), filter_by_horizon() guarantees all selected features
        are temporally available, so the masking loop is a no-op and this
        function returns the same F1 as standard evaluation.  The protocol
        becomes active for IC-FS(-temporal) and baselines where |S_unavail|>0.

        Args:
            clf:       Unfitted base classifier (will be cloned).
            X_tr_sel:  Training slice, shape (n_train, |selected|), UNMASKED.
            y_tr:      Training labels.
            X_te_sel:  Test slice, shape (n_test, |selected|), before masking.
            y_te:      Test labels.
            selected:  Feature names aligned with columns of X_tr_sel/X_te_sel.

        Returns:
            Weighted F1 score under deployment-realistic masking.
        """
        # Step 1: means from UNMASKED training data
        train_means = X_tr_sel.mean(axis=0)

        # Step 2: mask ONLY the test matrix
        X_te_deploy = X_te_sel.copy().astype(np.float64)
        for j, feat_name in enumerate(selected):
            if not get_temporal_availability(feat_name, self.horizon, self.taxonomy):
                X_te_deploy[:, j] = train_means[j]

        # Step 3: train on COMPLETE, UNMASKED training data
        clf_deploy = clone(clf)
        clf_deploy.fit(X_tr_sel, y_tr)

        # Step 4: predict on deployment-realistic masked test data
        y_pred_deploy = clf_deploy.predict(X_te_deploy)
        return f1_score(y_te, y_pred_deploy, average="weighted", zero_division=0)
    
    def fit(self,
             X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             feature_names: List[str],
             verbose: bool = True,
             skip_stability: bool = False) -> "ICFSPipeline":
        # ── Temporal filter ───────────────────────────────────────────────────
        available = filter_by_horizon(feature_names, self.horizon, self.taxonomy)
        if not available:
            raise ValueError(f"No features available at horizon t={self.horizon}")
        avail_idx = [feature_names.index(f) for f in available]
        X_tr = X_train[:, avail_idx]
        X_te = X_test[:,  avail_idx]
        k = min(self.top_k, len(available))

        if verbose:
            removed = set(feature_names) - set(available)
            print(f"[horizon={self.horizon}] Available {len(available)}/{len(feature_names)}; "
                   f"removed={sorted(removed)[:5]}{'...' if len(removed) > 5 else ''}")

        # ── Base classifier and CV splitter ───────────────────────────────────
        clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state,
                                       class_weight='balanced', n_jobs=1)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                                random_state=self.random_state)

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 1 — Nested validation for α* selection  (FIX #2)
        #
        # Split X_tr 80/20.  α is swept on the INNER split; IUS_val is
        # computed on X_val.  The test set is untouched in this phase.
        # α* = argmax IUS_val(α) over the α-grid.
        # ═════════════════════════════════════════════════════════════════════
        val_random_state = self.random_state + 1000  # Issue 3 fix: matches standalone script formula (seed+1000)
        X_inner, X_val, y_inner, y_val = train_test_split(
            X_tr, y_train,
            test_size=0.2,
            random_state=val_random_state,
            stratify=y_train,
        )

        # Feature scores computed on INNER training data only
        score_df_inner = feature_scores_for_selection(
            X_inner, y_inner, available, self.taxonomy)

        best_alpha_nested: Optional[float] = None
        best_ius_val: float = -np.inf

        if verbose:
            print(f"  [Phase 1] Nested α-selection on {len(y_inner)}/{len(y_val)} "
                   f"inner/val split (val_seed={val_random_state})")

        for alpha in self.alpha_values:
            sel_inner = ic_fs_select(score_df_inner, alpha, k)
            sel_loc_inner = [available.index(f) for f in sel_inner]

            # DRE F1 on validation using the fixed asymmetric protocol
            f1_val = self._compute_dre_f1(
                clf,
                X_inner[:, sel_loc_inner], y_inner,
                X_val[:, sel_loc_inner],   y_val,
                sel_inner,
            )
            ar_avail_val = actionability_ratio_available(
                sel_inner, self.horizon, self.taxonomy)
            ius_val = compute_ius_deploy(f1_val, sel_inner, self.horizon, self.taxonomy)

            if verbose:
                print(f"    α={alpha:.2f}  F1_val={f1_val:.3f}  "
                       f"AR_avail={ar_avail_val:.3f}  IUS_val={ius_val:.3f}")

            if ius_val > best_ius_val:
                best_ius_val   = ius_val
                best_alpha_nested = alpha

        self.best_alpha_nested_: Optional[float] = best_alpha_nested

        if verbose:
            print(f"  [Phase 1] α* = {best_alpha_nested}  "
                   f"(IUS_val = {best_ius_val:.3f})")

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 2 — Full α-sweep on complete train+test  (logging + ablation)
        #
        # Feature scores recomputed on the FULL X_tr (maximises sample size
        # for the final reported model, per Algorithm 1 Step 5).
        # Each α is evaluated on X_test for the results table / to_dataframe().
        # FIX #1 is applied here: clf_deploy trains on unmasked X_tr_s.
        # ═════════════════════════════════════════════════════════════════════
        self.score_df_ = feature_scores_for_selection(
            X_tr, y_train, available, self.taxonomy)

        self.solutions_ = []

        if verbose:
            print(f"  [Phase 2] Full α-sweep on {len(y_train)} train / "
                   f"{len(y_test)} test samples")

        for ai, alpha in enumerate(self.alpha_values):
            selected  = ic_fs_select(self.score_df_, alpha, k)
            sel_local = [available.index(f) for f in selected]
            X_tr_s = X_tr[:, sel_local]
            X_te_s = X_te[:, sel_local]

            # Standard (paper-style) evaluation
            clf_f = clone(clf)
            clf_f.fit(X_tr_s, y_train)
            y_pred = clf_f.predict(X_te_s)
            y_prob = (clf_f.predict_proba(X_te_s)[:, 1]
                      if hasattr(clf_f, "predict_proba") else y_pred.astype(float))
            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            # FIX #1 — asymmetric DRE: train on unmasked, mask only test
            f1_deploy = self._compute_dre_f1(
                clf,
                X_tr_s, y_train,
                X_te_s, y_test,
                selected,
            )

            # Cross-validation F1 on training set (no test data involved)
            cv_scores = cross_val_score(clone(clf), X_tr_s, y_train,
                                         cv=skf, scoring="f1_weighted", n_jobs=1)

            # Metrics
            ar       = actionability_ratio(selected, self.taxonomy)
            ar_avail = actionability_ratio_available(selected, self.horizon, self.taxonomy)
            tvs      = temporal_validity_score(selected, self.horizon, self.taxonomy)
            ius_old    = compute_ius_paper(f1, selected, self.horizon, self.taxonomy)
            ius_deploy = compute_ius_deploy(f1_deploy, selected, self.horizon, self.taxonomy)
            ius_geo    = compute_ius_geo(f1, selected, self.horizon, self.taxonomy)
            ext = evaluate_with_external_metric(
                y_test, y_prob, selected, self.horizon, self.taxonomy)

            # Bootstrap stability (FIX A3: independent seed per alpha)
            if skip_stability:
                stab = 0.0
            else:
                stab = self._bootstrap_stability(
                    X_tr, y_train, available, alpha, k,
                    bootstrap_seed=self.bootstrap_base_seed + ai)

            is_nested_best = (best_alpha_nested is not None and
                               np.isclose(alpha, best_alpha_nested))

            self.solutions_.append(SolutionPoint(
                alpha=alpha, beta=1 - alpha, selected_features=selected,
                accuracy=acc, f1=f1, f1_deploy=f1_deploy,
                ar=ar, ar_available=ar_avail,
                tvs=tvs, ius=ius_old, ius_deploy=ius_deploy, ius_geo=ius_geo,
                n_features=len(selected), stability=stab,
                cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
                precision_at_topk=ext["precision_at_topk"],
                recall_at_topk=ext["recall_at_topk"],
            ))

            if verbose:
                marker = " ← α*" if is_nested_best else ""
                print(f"  α={alpha:.2f}  F1={f1:.3f}  F1_deploy={f1_deploy:.3f}  "
                       f"AR={ar:.3f}  AR_avail={ar_avail:.3f}  TVS={tvs:.2f}  "
                       f"IUS_deploy={ius_deploy:.3f}  "
                       f"Prec@20%={ext['precision_at_topk']:.3f}  "
                       f"Stab={stab:.3f}{marker}")
        return self

    def best_by_ius(self) -> SolutionPoint:
        """
        Best solution by IUS_deploy, selected via nested validation (no leakage).

        Returns the SolutionPoint whose alpha matches best_alpha_nested_ — the
        alpha chosen in Phase 1 using only the inner/val split.  The returned
        solution's reported metrics (F1_deploy, IUS_deploy, etc.) come from
        Phase 2 on the full train+test split, but the alpha CHOICE itself never
        saw the test set.

        Falls back to argmax IUS_deploy over solutions_ if Phase 1 was skipped
        or best_alpha_nested_ is not found (with a warning).
        """
        if hasattr(self, 'best_alpha_nested_') and self.best_alpha_nested_ is not None:
            for s in self.solutions_:
                if np.isclose(s.alpha, self.best_alpha_nested_):
                    return s
            raise RuntimeError(
                f"[IC-FS] best_alpha_nested_={self.best_alpha_nested_} not found "
                f"in solutions_. Verify that alpha_values passed to fit() are "
                f"identical to those swept in Phase 1. Falling back to test-set "
                f"argmax would silently introduce leakage and is not permitted."
            )
        raise RuntimeError(
            "[IC-FS] best_alpha_nested_ is not set. Call fit() before best_by_ius()."
        )

    def best_by_ius_paper(self) -> SolutionPoint:
        """Best by IUS_paper (old metric); retained for comparison logging."""
        return max(self.solutions_, key=lambda s: s.ius)

    def best_by_f1(self) -> SolutionPoint:
        """Best by standard F1 (paper-style, no DRE masking)."""
        return max(self.solutions_, key=lambda s: s.f1)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Export full α-sweep results as DataFrame.

        Column 'nested_best' marks the row selected by Phase 1 nested
        validation — this is the row reported as the primary result.
        All other rows are retained for the ablation sweep table.

        Issue 4 WARNING: IUS_deploy values for rows where nested_best=False are
        computed from Phase 2 test-set evaluation and MUST NOT be used for model
        selection. Always use best_by_ius() or filter to nested_best=True for the
        reported model. Sorting non-best rows by IUS_deploy silently re-introduces
        test-set leakage that Phase 1 nested validation was designed to prevent.
        """
        rows = []
        
        for s in self.solutions_:
            is_best = (hasattr(self, 'best_alpha_nested_') and
                        self.best_alpha_nested_ is not None and
                        np.isclose(s.alpha, self.best_alpha_nested_))
            rows.append({
                "alpha":        s.alpha,
                "nested_best":  is_best,              # NEW: marks the α* row
                "accuracy":     round(s.accuracy * 100, 2),
                "f1_paper":     round(s.f1 * 100, 2),
                "f1_deploy":    round(s.f1_deploy * 100, 2),
                "AR":           round(s.ar, 3),
                "AR_available": round(s.ar_available, 3),
                "TVS":          round(s.tvs, 3),
                "IUS_paper":    round(s.ius * 100, 3),
                "IUS_deploy":   round(s.ius_deploy * 100, 3),   # KEY COLUMN
                "IUS_geo":      round(s.ius_geo * 100, 3),
                "n_features":   s.n_features,
                "stability":    round(s.stability, 3),
                "prec@20%":     round(s.precision_at_topk, 3),
                "recall@20%":   round(s.recall_at_topk, 3),
                "cv_mean":      round(s.cv_mean * 100, 2),
                "cv_std":       round(s.cv_std * 100, 2),
                "selected":     "|".join(s.selected_features),
            })
        return pd.DataFrame(rows)