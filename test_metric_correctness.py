"""
Comprehensive Algorithm Correctness Test for IUS Metric Upgrade
Tests all critical properties from the blueprint specification.
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "src"))

from icfs.core import (
    actionability_ratio, actionability_ratio_available,
    compute_ius_paper, compute_ius_deploy,
    filter_by_horizon, get_temporal_availability,
    TAXONOMY_UCI, ACTIONABILITY_WEIGHTS, Tier
)

def test_ar_available_algorithm():
    """
    Test AR_available = |S_actionable ∩ S_available_at_h| / |S|

    Validates the two-gate algorithm:
    Gate 1: Is feature available at horizon?
    Gate 2: Is it pedagogically actionable?
    """
    print("=" * 80)
    print("TEST 1: AR_available Algorithm Correctness")
    print("=" * 80)

    # Test case: Mixed selection at h=0
    # - 'school', 'sex', 'age': Available at h=0, Tier 0 (weight=0.0)
    # - 'studytime', 'schoolsup': Available at h=0, Tier 1 (weight=1.0)
    # - 'absences': NOT available at h=0 (only [1,2]), Tier 2 (weight=0.7)
    # - 'G1': NOT available at h=0 (only [1,2]), Tier 3 (weight=0.0)

    selected = ['school', 'sex', 'age', 'studytime', 'schoolsup', 'absences', 'G1']
    horizon = 0

    # Manual calculation:
    # Available at h=0: school, sex, age, studytime, schoolsup (5 features)
    # Actionable at h=0: studytime (1.0), schoolsup (1.0) = 2 features
    # AR_available = (0.0 + 0.0 + 0.0 + 1.0 + 1.0 + 0.0 + 0.0) / 7 = 2.0/7 ≈ 0.2857

    expected_ar_avail = 2.0 / 7.0
    computed_ar_avail = actionability_ratio_available(selected, horizon, TAXONOMY_UCI)

    print(f"  Selected features: {selected}")
    print(f"  Horizon: {horizon}")
    print(f"\n  Manual calculation:")
    print(f"    - school (Tier 0, avail): 0.0")
    print(f"    - sex (Tier 0, avail): 0.0")
    print(f"    - age (Tier 0, avail): 0.0")
    print(f"    - studytime (Tier 1, avail): 1.0")
    print(f"    - schoolsup (Tier 1, avail): 1.0")
    print(f"    - absences (Tier 2, NOT avail): 0.0")
    print(f"    - G1 (Tier 3, NOT avail): 0.0")
    print(f"    Sum = 2.0, Count = 7")
    print(f"\n  Expected AR_available: {expected_ar_avail:.6f}")
    print(f"  Computed AR_available: {computed_ar_avail:.6f}")
    print(f"  Difference: {abs(expected_ar_avail - computed_ar_avail):.9f}")

    assert np.isclose(computed_ar_avail, expected_ar_avail), \
        f"AR_available mismatch: expected {expected_ar_avail}, got {computed_ar_avail}"

    print(f"  ✓ PASS: Two-gate algorithm correctly implemented")
    return True


def test_temporal_validity_invariant():
    """
    Test critical invariant: For IC-FS(full), AR_available = AR
    Because filter_by_horizon guarantees all selected features are available.
    """
    print("\n" + "=" * 80)
    print("TEST 2: IC-FS(full) Invariant - AR_available = AR")
    print("=" * 80)

    all_features = list(TAXONOMY_UCI.keys())

    for horizon in [0, 1, 2]:
        # Simulate IC-FS(full): filter first, then select
        available = filter_by_horizon(all_features, horizon, TAXONOMY_UCI)
        selected = available[:10]  # Take first 10 available

        ar = actionability_ratio(selected, TAXONOMY_UCI)
        ar_avail = actionability_ratio_available(selected, horizon, TAXONOMY_UCI)

        diff = abs(ar - ar_avail)

        print(f"\n  Horizon {horizon}:")
        print(f"    Available features: {len(available)}")
        print(f"    Selected features: {len(selected)}")
        print(f"    AR (old): {ar:.6f}")
        print(f"    AR_available (new): {ar_avail:.6f}")
        print(f"    Difference: {diff:.9f}")

        assert diff < 1e-9, \
            f"Invariant violated at h={horizon}: AR={ar}, AR_available={ar_avail}"

        print(f"    ✓ PASS: AR_available = AR (within floating-point tolerance)")

    return True


def test_temporal_leakage_penalty():
    """
    Test that AR_available < AR when selection contains unavailable features.
    This is the key property that reveals temporal leakage in IC-FS(-temporal).
    """
    print("\n" + "=" * 80)
    print("TEST 3: Temporal Leakage Penalty - AR_available < AR")
    print("=" * 80)

    # IC-FS(-temporal) at h=0: includes G1 and absences (unavailable at h=0)
    # These have actionability weights but should be zeroed in AR_available
    leaky_selection = [
        'school',      # Tier 0 (0.0), available
        'studytime',   # Tier 1 (1.0), available
        'schoolsup',   # Tier 1 (1.0), available
        'famrel',      # Tier 2 (0.7), available at h=0 (mid-semester observable)
        'absences',    # Tier 2 (0.7), NOT available at h=0 (only [1,2])
        'G1',          # Tier 3 (0.0), NOT available at h=0 (only [1,2])
    ]

    horizon = 0

    ar = actionability_ratio(leaky_selection, TAXONOMY_UCI)
    ar_avail = actionability_ratio_available(leaky_selection, horizon, TAXONOMY_UCI)

    # Manual calculation for AR (ignores temporal availability):
    # (0.0 + 1.0 + 1.0 + 0.7 + 0.7 + 0.0) / 6 = 3.4 / 6 ≈ 0.5667
    expected_ar = 3.4 / 6.0

    # Manual calculation for AR_available (zeros unavailable):
    # absences: NOT available at h=0 → 0.0
    # G1: NOT available at h=0 → 0.0
    # (0.0 + 1.0 + 1.0 + 0.7 + 0.0 + 0.0) / 6 = 2.7 / 6 = 0.45
    expected_ar_avail = 2.7 / 6.0

    print(f"  Selected features: {leaky_selection}")
    print(f"  Horizon: {horizon}")
    print(f"\n  AR (old, ignores temporal):")
    print(f"    Expected: {expected_ar:.6f}")
    print(f"    Computed: {ar:.6f}")
    print(f"    Diff: {abs(ar - expected_ar):.9f}")

    print(f"\n  AR_available (new, applies temporal penalty):")
    print(f"    Expected: {expected_ar_avail:.6f}")
    print(f"    Computed: {ar_avail:.6f}")
    print(f"    Diff: {abs(ar_avail - expected_ar_avail):.9f}")

    penalty_pct = (1 - ar_avail/ar) * 100 if ar > 0 else 0
    print(f"\n  Temporal leakage penalty: {penalty_pct:.1f}%")
    print(f"  AR_available < AR: {ar_avail < ar}")

    assert np.isclose(ar, expected_ar, atol=1e-6), \
        f"AR mismatch: expected {expected_ar}, got {ar}"
    assert np.isclose(ar_avail, expected_ar_avail, atol=1e-6), \
        f"AR_available mismatch: expected {expected_ar_avail}, got {ar_avail}"
    assert ar_avail < ar, \
        f"Expected AR_available < AR for leaky selection, got {ar_avail} >= {ar}"

    print(f"  ✓ PASS: Temporal leakage correctly penalized")
    return True


def test_ius_deploy_formula():
    """
    Test IUS_deploy = F1_deploy × AR_available
    Verify it produces correct values and differs from IUS_paper when there's leakage.
    """
    print("\n" + "=" * 80)
    print("TEST 4: IUS_deploy Formula Correctness")
    print("=" * 80)

    f1_deploy = 0.75

    # Case 1: Clean selection (all available)
    clean_selection = ['school', 'sex', 'studytime', 'schoolsup', 'famsup']
    horizon = 0

    ar = actionability_ratio(clean_selection, TAXONOMY_UCI)
    ar_avail = actionability_ratio_available(clean_selection, horizon, TAXONOMY_UCI)
    ius_paper = compute_ius_paper(f1_deploy, clean_selection, horizon, TAXONOMY_UCI)
    ius_deploy = compute_ius_deploy(f1_deploy, clean_selection, horizon, TAXONOMY_UCI)

    expected_ius_deploy = f1_deploy * ar_avail

    print(f"\n  Case 1: Clean selection (all available at h={horizon})")
    print(f"    Selected: {clean_selection}")
    print(f"    F1_deploy: {f1_deploy}")
    print(f"    AR: {ar:.6f}")
    print(f"    AR_available: {ar_avail:.6f}")
    print(f"    IUS_paper: {ius_paper:.6f}")
    print(f"    IUS_deploy (expected): {expected_ius_deploy:.6f}")
    print(f"    IUS_deploy (computed): {ius_deploy:.6f}")

    assert np.isclose(ius_deploy, expected_ius_deploy, atol=1e-9), \
        f"IUS_deploy formula error: expected {expected_ius_deploy}, got {ius_deploy}"
    assert np.isclose(ar, ar_avail, atol=1e-9), \
        f"For clean selection, AR should equal AR_available"
    assert np.isclose(ius_paper, ius_deploy, atol=1e-9), \
        f"For clean selection, IUS_paper should equal IUS_deploy"

    print(f"    ✓ PASS: Clean selection → IUS_paper ≈ IUS_deploy")

    # Case 2: Leaky selection (contains unavailable features)
    leaky_selection = ['studytime', 'schoolsup', 'absences', 'G1']  # absences, G1 not at h=0

    ar_leaky = actionability_ratio(leaky_selection, TAXONOMY_UCI)
    ar_avail_leaky = actionability_ratio_available(leaky_selection, horizon, TAXONOMY_UCI)
    ius_paper_leaky = compute_ius_paper(f1_deploy, leaky_selection, horizon, TAXONOMY_UCI)
    ius_deploy_leaky = compute_ius_deploy(f1_deploy, leaky_selection, horizon, TAXONOMY_UCI)

    expected_ius_deploy_leaky = f1_deploy * ar_avail_leaky

    print(f"\n  Case 2: Leaky selection (unavailable features at h={horizon})")
    print(f"    Selected: {leaky_selection}")
    print(f"    F1_deploy: {f1_deploy}")
    print(f"    AR: {ar_leaky:.6f}")
    print(f"    AR_available: {ar_avail_leaky:.6f}")
    print(f"    IUS_paper: {ius_paper_leaky:.6f}")
    print(f"    IUS_deploy (expected): {expected_ius_deploy_leaky:.6f}")
    print(f"    IUS_deploy (computed): {ius_deploy_leaky:.6f}")

    collapse_pct = (1 - ius_deploy_leaky/ius_paper_leaky) * 100 if ius_paper_leaky > 0 else 0
    print(f"    Metric collapse: {collapse_pct:.1f}%")

    assert np.isclose(ius_deploy_leaky, expected_ius_deploy_leaky, atol=1e-9), \
        f"IUS_deploy formula error: expected {expected_ius_deploy_leaky}, got {ius_deploy_leaky}"
    assert ar_avail_leaky < ar_leaky, \
        f"For leaky selection, AR_available should be < AR"
    assert ius_deploy_leaky < ius_paper_leaky, \
        f"For leaky selection, IUS_deploy should be < IUS_paper"

    print(f"    ✓ PASS: Leaky selection → IUS_deploy < IUS_paper (reveals leakage)")

    return True


def test_edge_cases():
    """Test edge cases: empty selection, all unavailable, all non-actionable."""
    print("\n" + "=" * 80)
    print("TEST 5: Edge Cases")
    print("=" * 80)

    # Edge case 1: Empty selection
    ar_empty = actionability_ratio_available([], horizon=0, taxonomy=TAXONOMY_UCI)
    ius_empty = compute_ius_deploy(0.75, [], horizon=0, taxonomy=TAXONOMY_UCI)

    print(f"\n  Edge case 1: Empty selection")
    print(f"    AR_available: {ar_empty:.6f}")
    print(f"    IUS_deploy: {ius_empty:.6f}")

    assert ar_empty == 0.0, "Empty selection should have AR_available = 0.0"
    assert ius_empty == 0.0, "Empty selection should have IUS_deploy = 0.0"
    print(f"    ✓ PASS")

    # Edge case 2: All unavailable
    all_unavailable = ['absences', 'G1', 'G2']  # None available at h=0
    ar_unavail = actionability_ratio_available(all_unavailable, horizon=0, taxonomy=TAXONOMY_UCI)
    ius_unavail = compute_ius_deploy(0.75, all_unavailable, horizon=0, taxonomy=TAXONOMY_UCI)

    print(f"\n  Edge case 2: All features unavailable at h=0")
    print(f"    Selected: {all_unavailable}")
    print(f"    AR_available: {ar_unavail:.6f}")
    print(f"    IUS_deploy: {ius_unavail:.6f}")

    assert ar_unavail == 0.0, "All unavailable should have AR_available = 0.0"
    assert ius_unavail == 0.0, "All unavailable should have IUS_deploy = 0.0"
    print(f"    ✓ PASS")

    # Edge case 3: All non-actionable (but available)
    all_non_actionable = ['school', 'sex', 'age', 'address']  # All Tier 0
    ar_non_act = actionability_ratio_available(all_non_actionable, horizon=0, taxonomy=TAXONOMY_UCI)
    ius_non_act = compute_ius_deploy(0.75, all_non_actionable, horizon=0, taxonomy=TAXONOMY_UCI)

    print(f"\n  Edge case 3: All features non-actionable (Tier 0)")
    print(f"    Selected: {all_non_actionable}")
    print(f"    AR_available: {ar_non_act:.6f}")
    print(f"    IUS_deploy: {ius_non_act:.6f}")

    assert ar_non_act == 0.0, "All non-actionable should have AR_available = 0.0"
    assert ius_non_act == 0.0, "All non-actionable should have IUS_deploy = 0.0"
    print(f"    ✓ PASS")

    return True


def main():
    """Run all algorithm correctness tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 16 + "IUS METRIC UPGRADE - ALGORITHM CORRECTNESS TEST" + " " * 15 + "║")
    print("╚" + "═" * 78 + "╝")

    tests = [
        test_ar_available_algorithm,
        test_temporal_validity_invariant,
        test_temporal_leakage_penalty,
        test_ius_deploy_formula,
        test_edge_cases,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(("PASS", test.__name__))
        except AssertionError as e:
            print(f"\n  ✗ FAIL: {e}")
            results.append(("FAIL", test.__name__))
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            results.append(("ERROR", test.__name__))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for status, name in results:
        symbol = "✓" if status == "PASS" else "✗"
        print(f"  {symbol} {status:5s} | {name}")

    n_pass = sum(1 for s, _ in results if s == "PASS")
    n_total = len(results)

    print(f"\n  Tests passed: {n_pass}/{n_total}")

    if n_pass == n_total:
        print("\n  ✅ ALL ALGORITHM CORRECTNESS TESTS PASSED!")
        print("  The implementation correctly follows the blueprint specification.")
        return 0
    else:
        print("\n  ❌ SOME TESTS FAILED - Review implementation")
        return 1


if __name__ == "__main__":
    exit(main())
