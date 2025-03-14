from dataclasses import dataclass

import pandas as pd
from pandas import Series

from gscap import plot
from gscap.framework.instruments import Instrument


@dataclass
class Cost:
    commission_currency: Series = None
    slippage_currency: Series = None
    total_currency: Series = None
    # risk_adj_per_lot: Series = None
    return_series: Series = None


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


def _resample_series(series: pd.Series, resample):
    _s = series.resample(resample).sum()
    return _s[~_s.eq(0.0)]


def analyse_cost(return_series, cost_return_series, show=True):
    """
    Assuming that we'll always keep `include_cost=True`
    Soon we'll make it non-optional to remove cost
    """
    _name = return_series.name
    rtr_without_cost = return_series + cost_return_series
    rtr_with_cost = return_series
    cost_return_series = cost_return_series
    rtr_without_cost.name = f"{_name} (without cost)"
    rtr_with_cost.name = f"{_name} (with cost)"
    cost_return_series.name = f"{_name} cost"
    # rtr_without_cost = _resample_series(rtr_without_cost, "D")
    # rtr_with_cost = _resample_series(rtr_with_cost, "D")
    # cost_return_series = _resample_series(cost_return_series, "D")
    rtr_without_cost = rtr_without_cost.resample("D").sum(min_count=1).dropna()
    rtr_with_cost = rtr_with_cost.resample("D").sum(min_count=1).dropna()
    cost_return_series = cost_return_series.resample("D").sum(min_count=1).dropna()
    plot.returns(
        rtr_with_cost,
        rtr_without_cost,
        cumulative=True,
        show=show,
        title="Cost Effect On Returns",
    )
    plot.returns(
        cost_return_series,
        cumulative=True,
        show=show,
        title=r"Cost (% of capital)",
    )
    plot.return_histogram(
        cost_return_series,
        granular_returns=True,
        show=show,
        title=r"Cost (% of capital)",
    )
