"""
================================================================================
IC-FS v2: Intervention-Constrained Feature Selection (HARDENED)
================================================================================
Thay đổi so với v1 (ic_fs_core.py):
  A1. filter_by_horizon: unknown feature -> False (+ warning), không còn silent leakage
  A2. get_actionability_score: unknown feature -> 0.0 conservative (+ warning)
  A3. _bootstrap_stability: nhận bootstrap_seed độc lập cho mỗi lần gọi
  A4. Thêm Wilcoxon test + bootstrap CI (statistical_tests.py)
  A5. evaluate_with_external_metric(): metric không chứa AR để tránh circular eval
  A6. Thêm get_actionability_score với tùy chọn strict=True (raise thay vì warn)
  A7. Thêm precision_at_top_k_actionable() và top_k_intervention_hit_rate()
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
from sklearn.model_selection import StratifiedKFold, cross_val_score
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


# ─────────────────────────────────────────────────────────────────────────────
# FIX A1 + A2: Safer taxonomy lookup helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    """IUS = F1 × AR × TVS (multiplicative hard-gate)."""
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


# ── FIX A5: External metrics to avoid circular evaluation ────────────────────

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
    FIX A5: Đánh giá KHÔNG chứa AR — tránh circular.

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
                                   taxonomy: Dict[str, FeatureProfile] = None
                                   ) -> pd.DataFrame:
    scaler = MinMaxScaler()
    X_nn = scaler.fit_transform(X)

    chi2_raw, _ = chi2(X_nn, y)
    chi2_raw = np.nan_to_num(chi2_raw, nan=0.0)

    mi_raw = mutual_info_classif(X, y, random_state=RANDOM_STATE)

    # Correlation - handle zero-variance features
    def safe_corr(x_col, y_col):
        if np.std(x_col) < 1e-10 or np.std(y_col) < 1e-10:
            return 0.0
        try:
            return abs(np.corrcoef(x_col, y_col)[0, 1])
        except:
            return 0.0

    corr_raw = np.array([safe_corr(X[:, j], y) for j in range(X.shape[1])])
    corr_raw = np.nan_to_num(corr_raw, nan=0.0)

    rf = RandomForestClassifier(n_estimators=100,
                                 random_state=RANDOM_STATE, n_jobs=1)
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
    f1: float
    ar: float
    tvs: float
    ius: float
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
                  bootstrap_base_seed: int = 2026):
        self.horizon = horizon
        self.alpha_values = alpha_values or np.linspace(0, 1, 11).tolist()
        self.top_k = top_k
        self.n_bootstrap = n_bootstrap
        self.cv_folds = cv_folds
        self.taxonomy = taxonomy if taxonomy is not None else TAXONOMY_UCI
        self.bootstrap_base_seed = bootstrap_base_seed
        self.solutions_: List[SolutionPoint] = []
        self.score_df_: Optional[pd.DataFrame] = None

    # FIX A3: Bootstrap seed độc lập cho mỗi alpha
    def _bootstrap_stability(self,
                              X: np.ndarray, y: np.ndarray,
                              feature_names: List[str],
                              alpha: float, k: int,
                              bootstrap_seed: int) -> float:
        n = len(y)
        selected_sets = []
        rng = np.random.RandomState(bootstrap_seed)
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            X_bs, y_bs = X[idx], y[idx]
            if len(np.unique(y_bs)) < 2:
                continue
            score_df = feature_scores_for_selection(X_bs, y_bs, feature_names,
                                                     self.taxonomy)
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

    def fit(self,
             X_train: np.ndarray, y_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray,
             feature_names: List[str],
             verbose: bool = True,
             skip_stability: bool = False) -> "ICFSPipeline":
        available = filter_by_horizon(feature_names, self.horizon, self.taxonomy)
        if not available:
            raise ValueError(f"No features available at horizon t={self.horizon}")
        avail_idx = [feature_names.index(f) for f in available]
        X_tr = X_train[:, avail_idx]
        X_te = X_test[:,  avail_idx]

        if verbose:
            removed = set(feature_names) - set(available)
            print(f"[horizon={self.horizon}] Available {len(available)}/{len(feature_names)}; "
                   f"removed={sorted(removed)[:5]}{'...' if len(removed)>5 else ''}")

        self.score_df_ = feature_scores_for_selection(X_tr, y_train, available,
                                                       self.taxonomy)
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                       n_jobs=1)
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                                random_state=RANDOM_STATE)

        self.solutions_ = []
        k = min(self.top_k, len(available))

        for ai, alpha in enumerate(self.alpha_values):
            selected = ic_fs_select(self.score_df_, alpha, k)
            sel_local = [available.index(f) for f in selected]
            X_tr_s = X_tr[:, sel_local]
            X_te_s = X_te[:, sel_local]

            clf_f = clone(clf)
            clf_f.fit(X_tr_s, y_train)
            y_pred = clf_f.predict(X_te_s)
            y_prob = clf_f.predict_proba(X_te_s)[:, 1] if hasattr(clf_f, "predict_proba") else y_pred.astype(float)

            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average="weighted", zero_division=0)

            cv_scores = cross_val_score(clone(clf), X_tr_s, y_train,
                                         cv=skf, scoring="f1_weighted", n_jobs=1)

            ar  = actionability_ratio(selected, self.taxonomy)
            tvs = temporal_validity_score(selected, self.horizon, self.taxonomy)
            ius = compute_ius(f1, selected, self.horizon, self.taxonomy)
            ius_geo = compute_ius_geo(f1, selected, self.horizon, self.taxonomy)

            ext = evaluate_with_external_metric(y_test, y_prob, selected,
                                                 self.horizon, self.taxonomy)

            if skip_stability:
                stab = 0.0
            else:
                # FIX A3: seed độc lập cho mỗi alpha (base + ai)
                stab = self._bootstrap_stability(
                    X_tr, y_train, available, alpha, k,
                    bootstrap_seed=self.bootstrap_base_seed + ai)

            self.solutions_.append(SolutionPoint(
                alpha=alpha, beta=1-alpha, selected_features=selected,
                accuracy=acc, f1=f1, ar=ar, tvs=tvs, ius=ius, ius_geo=ius_geo,
                n_features=len(selected), stability=stab,
                cv_mean=cv_scores.mean(), cv_std=cv_scores.std(),
                precision_at_topk=ext["precision_at_topk"],
                recall_at_topk=ext["recall_at_topk"],
            ))

            if verbose:
                print(f"  α={alpha:.2f} F1={f1:.3f} AR={ar:.3f} TVS={tvs:.2f} "
                       f"IUS={ius:.3f} Prec@20%={ext['precision_at_topk']:.3f} "
                       f"Stab={stab:.3f}")
        return self

    def best_by_ius(self) -> SolutionPoint:
        return max(self.solutions_, key=lambda s: s.ius)

    def best_by_f1(self) -> SolutionPoint:
        return max(self.solutions_, key=lambda s: s.f1)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for s in self.solutions_:
            rows.append({
                "alpha": s.alpha, "accuracy": round(s.accuracy*100,2),
                "f1": round(s.f1*100,2), "AR": round(s.ar,3),
                "TVS": round(s.tvs,3), "IUS": round(s.ius*100,3),
                "IUS_geo": round(s.ius_geo*100,3),
                "n_features": s.n_features, "stability": round(s.stability,3),
                "prec@20%": round(s.precision_at_topk,3),
                "recall@20%": round(s.recall_at_topk,3),
                "cv_mean": round(s.cv_mean*100,2), "cv_std": round(s.cv_std*100,2),
                "selected": "|".join(s.selected_features),
            })
        return pd.DataFrame(rows)
