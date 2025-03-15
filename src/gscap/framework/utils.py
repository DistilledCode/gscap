import pandas as pd

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
