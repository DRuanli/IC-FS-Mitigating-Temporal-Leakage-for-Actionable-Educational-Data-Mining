#!/usr/bin/env python3
"""
Verify IC-FS project setup before running experiments
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check required Python packages."""
    print("="*60)
    print("Checking Dependencies...")
    print("="*60)

    required = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False

    print("\n✓ All dependencies installed")
    return True


def check_data_files():
    """Check OULAD data files exist."""
    print("\n" + "="*60)
    print("Checking Data Files...")
    print("="*60)

    data_dir = Path('data/oulad_raw')

    required_files = [
        'studentInfo.csv',
        'studentVle.csv',
        'studentAssessment.csv',
        'assessments.csv',
        'vle.csv',
        'courses.csv',
        'studentRegistration.csv',
    ]

    missing = []
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ {filename:30s} ({size:6.1f} MB)")
        else:
            print(f"✗ {filename:30s} - MISSING")
            missing.append(filename)

    if missing:
        print(f"\n❌ Missing data files: {', '.join(missing)}")
        print("Download from: https://analyse.kmi.open.ac.uk/open_dataset")
        return False

    print("\n✓ All data files present")
    return True

    """Check source code files exist."""
    print("\n" + "="*60)
    print("Checking Source Files...")
    print("="*60)

    required_files = {
        'src/icfs/core.py': 'IC-FS core algorithm',
        'src/icfs/oulad_pipeline.py': 'OULAD feature pipeline',
        'src/icfs/taxonomy_oulad.py': 'OULAD taxonomy',
        'experiments/oulad/run_oulad_experiments.py': 'OULAD experiment runner',
    }

    missing = []
    for filepath, description in required_files.items():
        path = Path(filepath)
        if path.exists():
            print(f"✓ {description:40s} ({filepath})")
        else:
            print(f"✗ {description:40s} - MISSING")
            missing.append(filepath)

    if missing:
        print(f"\n❌ Missing source files: {', '.join(missing)}")
        return False

    print("\n✓ All source files present")
    return True


    """Check required directories exist."""
    print("\n" + "="*60)
    print("Checking Directories...")
    print("="*60)

    required_dirs = [
        'src/icfs',
        'experiments/oulad',
        'experiments/uci',
        'experiments/analysis',
        'data/oulad_raw',
        'results/oulad',
        'results/uci',
    ]

    missing = []
    for dirname in required_dirs:
        dirpath = Path(dirname)
        if dirpath.exists() and dirpath.is_dir():
            print(f"✓ {dirname}")
        else:
            print(f"✗ {dirname} - MISSING")
            missing.append(dirname)

    if missing:
        print(f"\n❌ Missing directories: {', '.join(missing)}")
        print("Create with: mkdir -p " + " ".join(missing))
        return False

    print("\n✓ All directories exist")
    return True


def main():
    """Run all verification checks."""
    print("\n" + "="*60)
    print("IC-FS Project Setup Verification")
    print("="*60)
    print()

    checks = [
        check_dependencies(),
        check_data_files(),
    ]

    print("\n" + "="*60)
    if all(checks):
        print("✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("="*60)
        print("\nYou're ready to run experiments!")
        print("\nQuick start:")
        print("  ./run_all_experiments.sh")
        print("\nOR step by step:")
        print("  1. python src/icfs/oulad_pipeline.py")
        print("  2. python experiments/oulad/run_oulad_experiments.py")
        return 0
    else:
        print("❌❌❌ SOME CHECKS FAILED ❌❌❌")
        print("="*60)
        print("\nPlease fix the issues above before running experiments.")
        return 1


if __name__ == '__main__':
    sys.exit(main())