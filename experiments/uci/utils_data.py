"""
Utility wrapper for UCI data loading.
Imports from src.icfs.data_loaders for compatibility.
"""
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from src.icfs.data_loaders import load_uci, preprocess_uci, load_and_split

__all__ = ['load_uci', 'preprocess_uci', 'load_and_split']
