"""
BiteBase Data Processing and Loading
"""

from pathlib import Path

# Define base data paths
BASE_PATH = Path(__file__).parent.parent.parent
RAW_DATA_PATH = BASE_PATH / 'RAW'
DYNAMIC_DATA_PATH = RAW_DATA_PATH / 'Dynamics'
STATIC_DATA_PATH = RAW_DATA_PATH / 'Statics'

__version__ = '0.1.0'