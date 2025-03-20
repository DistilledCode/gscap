from __future__ import annotations

import pandas as pd
from typing import TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from gscap.framework.instruments import Instrument


def get_th_series(instrument: Instrument):
    _index = pd.DatetimeIndex(
        [
            "2014-10-31 00:00:00+00:00",
            "2014-11-01 00:00:00+00:00",
        ]
    )
    series = pd.Series(
        [23, 4.588888],
        index=_index,
    )
    return series.reindex(instrument.close_price().index, method="nearest")


def interval_of_time_series(ts: pd.Series):
    _t_delta = ts.index[1:] - ts.index[:-1]
    _t_deltas_in_secs = pd.Series(f.total_seconds() for f in _t_delta)
    return _t_deltas_in_secs.mode()[0]


@lru_cache(maxsize=128)
def days_look_back(instrument: Instrument):
    """
    Number of observation we have to include to cover data equal to one trading day
    """
    adj: pd.Series = instrument.close_price().adjusted.squeeze()
    if isinstance(instrument.meta.trading_hours, pd.Series):
        th = instrument.meta.trading_hours[-1]
    else:
        th = instrument.meta.trading_hours
    return max(int(th * 3600 / interval_of_time_series(adj)), 1)


def unit_vol_perc(instrument: Instrument):
    """
    returns non-forward looking interval unit volatality in terms 
    of percentage.
    Automatically handles negative underlying price by forward 
    filling last valid price. Might not work effectively for spreads
    but can handle outrights
    """
    adj: pd.Series = instrument.close_price().adjusted.squeeze()
    und: pd.Series = instrument.close_price().underlying.squeeze()
    _neg_values = {i: None for i in und[und.le(0)].values}
    if len(_neg_values) > 0:
        print(f"forward filling {len(_neg_values)} negative values.")
    ffiled_neg_und = und.replace(_neg_values)

    dlb = days_look_back(instrument)
    unit_vol_perc = adj.diff().ewm(span=22 * dlb).std() / ffiled_neg_und
    return unit_vol_perc.shift(1)
