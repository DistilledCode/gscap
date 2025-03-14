# src/my_package/__init__.py
__version__ = "0.1.0"


import warnings
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=UserWarning)


DAYS_IN_YEAR = 252
DAYS_IN_WEEK = 5
DAYS_IN_MONTH = 22
HOURS_IN_DAY = 23
MINUTES_IN_YEAR = DAYS_IN_YEAR * HOURS_IN_DAY * 60
FDM_RESAMPLE = "W"
IDM_RESAMPLE = "W"
# Assuming daily data, lookback period is one week, for 5 min data it'll be one hour (12)
VOL_SCLR_LBACK_SPAN_SLOW = None
VOL_SCLR_LBACK_SPAN_FAST = None
