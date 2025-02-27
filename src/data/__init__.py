"""
BiteBase Data Processing and Loading
"""

from pathlib import Path

# Define data paths
DATA_DIR = Path(__file__).parent
RAW_DATA_PATH = DATA_DIR / "raw"
DYNAMIC_DATA_PATH = DATA_DIR / "dynamic"

# Create directories if they don't exist
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
DYNAMIC_DATA_PATH.mkdir(parents=True, exist_ok=True)

STATIC_DATA_PATH = RAW_DATA_PATH / 'Statics'

__version__ = '0.1.0'