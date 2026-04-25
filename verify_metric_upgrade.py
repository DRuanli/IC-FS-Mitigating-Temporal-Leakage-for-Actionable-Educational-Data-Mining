"""
Verification script for IUS metric upgrade.
Tests the critical invariant: IUS_deploy ≈ IUS_paper for IC-FS(full).
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ic_fs_v2 import (
    actionability_ratio, actionability_ratio_available,
    compute_ius_paper, compute_ius_deploy,
    filter_by_horizon, TAXONOMY_UCI
)

def test_invariant():
    """Test that for temporally-filtered selections, AR_available = AR."""
    print("=" * 80)
    print("VERIFICATION: IUS Metric Upgrade Invariant")
    print("=" * 80)

    # Test case 1: All features available at horizon
    print("\n[Test 1] Features all available at horizon h=0:")
    selected_h0 = ["school", "sex", "age", "address", "studytime", "schoolsup"]

    ar = actionability_ratio(selected_h0, TAXONOMY_UCI)
    ar_avail = actionability_ratio_available(selected_h0, horizon=0, taxonomy=TAXONOMY_UCI)

    print(f"  Selected: {selected_h0}")
    print(f"  AR (old):           {ar:.6f}")
    print(f"  AR_available (new): {ar_avail:.6f}")
    print(f"  Difference:         {abs(ar - ar_avail):.9f}")
    print(f"  ✓ PASS" if np.isclose(ar, ar_avail) else f"  ✗ FAIL")

    # Test case 2: Using filter_by_horizon (IC-FS(full) protocol)
    print("\n[Test 2] After filter_by_horizon (IC-FS full protocol):")
    all_features = list(TAXONOMY_UCI.keys())
    available_h1 = filter_by_horizon(all_features, horizon=1, taxonomy=TAXONOMY_UCI)
    selected_filtered = available_h1[:10]  # Select first 10 available

    ar = actionability_ratio(selected_filtered, TAXONOMY_UCI)
    ar_avail = actionability_ratio_available(selected_filtered, horizon=1, taxonomy=TAXONOMY_UCI)

    print(f"  Total features: {len(all_features)}")
    print(f"  Available at h=1: {len(available_h1)}")
    print(f"  Selected (first 10): {len(selected_filtered)}")
    print(f"  AR (old):           {ar:.6f}")
    print(f"  AR_available (new): {ar_avail:.6f}")
    print(f"  Difference:         {abs(ar - ar_avail):.9f}")
    print(f"  ✓ PASS" if np.isclose(ar, ar_avail) else f"  ✗ FAIL")

    # Test case 3: Mixed selection (some unavailable) — IC-FS(-temporal) case
    print("\n[Test 3] Mixed selection with unavailable features at h=0:")
    # absences is only available at h=1,2 (not h=0)
    # G1 is only available at h=1,2 (not h=0)
    mixed_selection = ["school", "sex", "studytime", "absences", "G1"]

    ar = actionability_ratio(mixed_selection, TAXONOMY_UCI)
    ar_avail = actionability_ratio_available(mixed_selection, horizon=0, taxonomy=TAXONOMY_UCI)

    print(f"  Selected: {mixed_selection}")
    print(f"  AR (old, ignores temporal):  {ar:.6f}")
    print(f"  AR_available (new, temporal): {ar_avail:.6f}")
    print(f"  Difference:                   {abs(ar - ar_avail):.6f}")
    print(f"  Expected: ar_avail < ar (temporal penalty)")
    print(f"  ✓ PASS" if ar_avail < ar else f"  ✗ FAIL")

    # Test case 4: IUS computation equivalence for filtered selection
    print("\n[Test 4] IUS_deploy ≈ IUS_paper for temporally-valid selection:")
    f1 = 0.75
    selected = available_h1[:12]  # All guaranteed available at h=1

    ius_paper = compute_ius_paper(f1, selected, horizon=1, taxonomy=TAXONOMY_UCI)
    ius_deploy = compute_ius_deploy(f1, selected, horizon=1, taxonomy=TAXONOMY_UCI)

    print(f"  F1: {f1}")
    print(f"  Selected: {len(selected)} features (all available at h=1)")
    print(f"  IUS_paper:  {ius_paper:.6f}")
    print(f"  IUS_deploy: {ius_deploy:.6f}")
    print(f"  Difference: {abs(ius_paper - ius_deploy):.9f}")
    print(f"  ✓ PASS" if np.isclose(ius_paper, ius_deploy) else f"  ✗ FAIL")

    # Test case 5: IUS divergence for leaky selection
    print("\n[Test 5] IUS_deploy < IUS_paper for temporally-leaky selection:")
    # Mixed: some available, some not at h=0
    leaky_selection = ["school", "sex", "studytime", "absences", "G1", "G2"]

    ius_paper = compute_ius_paper(f1, leaky_selection, horizon=0, taxonomy=TAXONOMY_UCI)
    ius_deploy = compute_ius_deploy(f1, leaky_selection, horizon=0, taxonomy=TAXONOMY_UCI)

    print(f"  F1: {f1}")
    print(f"  Selected: {leaky_selection}")
    print(f"  IUS_paper:  {ius_paper:.6f}")
    print(f"  IUS_deploy: {ius_deploy:.6f}")
    print(f"  Difference: {(ius_paper - ius_deploy):.6f}")
    print(f"  Expected: ius_deploy < ius_paper (temporal leakage penalty)")
    print(f"  ✓ PASS" if ius_deploy < ius_paper else f"  ✗ FAIL")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("All invariants should PASS for correct implementation.")
    print("=" * 80)

if __name__ == "__main__":
    test_invariant()
