# src/my_package/__init__.py
__version__ = "0.1.0"


import warnings
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
