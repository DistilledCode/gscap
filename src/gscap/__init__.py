__version__ = "0.1.0"


import warnings
from pathlib import Path

import pandas as pd

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
VOL_SCLR_LBACK_SPAN_SLOW = None
VOL_SCLR_LBACK_SPAN_FAST = None


@pd.api.extensions.register_series_accessor("interval")
class SeriesTimeSeriesInterval:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self):
        ts = self._obj
        intv = (ts.index[1:] - ts.index[:-1]).total_seconds()
        return intv.to_series().mode()[0]


@pd.api.extensions.register_dataframe_accessor("interval")
class DataFrameTimeSeriesInterval:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def __call__(self):
        ts = self._obj
        intv = (ts.index[1:] - ts.index[:-1]).total_seconds()
        return intv.to_series().mode()[0]
