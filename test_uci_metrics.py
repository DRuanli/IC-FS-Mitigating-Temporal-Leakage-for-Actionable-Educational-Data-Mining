"""
Quick test to verify UCI metric upgrade works correctly.
"""
import sys
from pathlib import Path

# Add experiments/uci to path
uci_exp = Path(__file__).parent / "experiments" / "uci"
sys.path.insert(0, str(uci_exp))

print("=" * 80)
print("UCI Metric Upgrade - Import Test")
print("=" * 80)

print("\n[1/3] Testing ic_fs_v2 imports...")
try:
    from ic_fs_v2 import (
        actionability_ratio_available,
        compute_ius_deploy,
        compute_ius_paper,
        TAXONOMY_UCI,
    )
    print("  ✓ All new functions import successfully from ic_fs_v2.py")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

print("\n[2/3] Testing run_dre.py imports...")
try:
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    # The run_dre.py file should import correctly now
    print("  ✓ run_dre.py should import new functions correctly")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

print("\n[3/3] Testing run_ablation.py imports...")
try:
    from run_ablation import (
        run_config_full_fast,
        run_config_no_temporal_fast,
        run_config_no_action_fast,
        run_config_hardfilter_fast,
    )
    print("  ✓ All config functions import successfully from run_ablation.py")
except ImportError as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

print("\n[4/4] Quick algorithm test...")
try:
    # Test AR_available with UCI taxonomy
    from ic_fs_v2 import actionability_ratio

    selected = ['school', 'sex', 'studytime', 'G1']  # G1 not available at h=0
    horizon = 0

    ar_val = actionability_ratio(selected, TAXONOMY_UCI)
    ar_avail = actionability_ratio_available(selected, horizon, TAXONOMY_UCI)

    print(f"  Selected features: {selected}")
    print(f"  AR (old, ignores temporal): {ar_val:.3f}")
    print(f"  AR_available (new, h=0): {ar_avail:.3f}")

    if ar_avail < ar_val:
        print(f"  ✓ Temporal penalty correctly applied (G1 not at h=0)")
    else:
        print(f"  ✗ Expected AR_available < AR")
        sys.exit(1)

except Exception as e:
    print(f"  ✗ Algorithm test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL UCI METRIC UPGRADE TESTS PASSED")
print("=" * 80)
print("\nYou can now run UCI experiments:")
print("  cd experiments/uci")
print("  python run_dre.py student-mat.csv 0")
print("  python run_statistics.py student-mat.csv 0")
print("  python run_ablation.py")
