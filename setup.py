"""
Setup configuration for IC-FS package.

Install in development mode:
    pip install -e .

Install from source:
    pip install .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="icfs",
    version="2.0.0",
    author="Le Nguyen",
    description="Intervention-Constrained Feature Selection for Educational Data Mining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/icfs-jla",  # Update with actual repo URL
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.4.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pymoo>=0.6.1",
        "boruta>=0.4",
        "pyarrow>=14.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "icfs-oulad=icfs.oulad_pipeline:main",
        ],
    },
)
