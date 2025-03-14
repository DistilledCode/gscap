import numpy as np
import pandas as pd

import gscap

_NORMAL_DIVIDER = 5.854340606494375
_1_DEV = 0.682689492137086
_3_DEV = 0.997300203936740


def ewma_vol(
    return_series: pd.Series,
    span=22,
    annualize=True,
) -> pd.DataFrame:

    span = span
    _vol = return_series.ewm(span=span, min_periods=span // 2).std()
    if annualize:
        _vol *= np.sqrt(gscap.DAYS_IN_YEAR)
    return _vol


def rolling_sharpe(
    return_series: pd.Series,
    span=22 * 6,
    annualize=True,
):
    _wseries = return_series.ewm(span=span, min_periods=span // 2)
    _rs = _wseries.mean() / _wseries.std()
    if annualize:
        _rs *= np.sqrt(gscap.DAYS_IN_YEAR)
    return _rs


def _cumulative_drawdown(return_series: pd.Series):
    cumulative_sum_series = return_series.cumsum()
    max_cumul_series = cumulative_sum_series.cummax()
    return cumulative_sum_series - max_cumul_series


def _compounding_drawdown(return_series: pd.Series):
    cumulative_prod_series = return_series.add(1).cumprod()
    max_cumul_return = cumulative_prod_series.cummax()
    return cumulative_prod_series / max_cumul_return - 1


def drawdown_series(return_series: pd.Series, compound=False, cumulative=True):
    """calculates drawdown series given a return series

    Args:
        return_series (pd.Series): return series in fraction (`.pct_change()`)
            (where 0.01 denotes 1% return)
        compound (bool, False):
            - whether to calaculate drawdown by compounding returns.
        cumulative (bool, False):
            - whether to calaculate drawdown by cumulating returns.

    Raises:
        ValueError:
            - Either of compound or cumulative must be True
            - Neither of them can be True at the same time
    """
    if cumulative ^ compound is False:
        raise ValueError("Either of `compound` or `cumulative` must be True, not both")
    if compound:
        return _compounding_drawdown(return_series)
    else:
        return _cumulative_drawdown(return_series)


def tail_ratios(return_series: pd.Series):
    """
    * Measures fat tail ratios in comparison to Gaussian Distribution.
    * Left Tail: Ratio of (-3 SD thrshld / -1 SD thrshld of demeaned returns) and
    (-3 SD thrshld / -1 SD thrshld of Gaussian Distribution)
    * Right Tail: Ratio of (+3 SD thrshld / +1 SD thrshld of demeaned returns) and
    (+3 SD thrshld / +1 SD thrshld of Gaussian Distribution)
    * (3 SD / 1 SD) for Gaussian Distribution is same on either side
    * (3 SD / 1 SD) for Gaussian Distribution = 5.854146875469564

    Adopted from "Advaned Futures Trading Strategies" by Robert Carver,
    Strategy 01, "Measuring Fat Tails"

    Args:
        return_series (pd.Series): return series in fraction (`.pct_change()`)
            (where 0.01 denotes 1% return)
    Returns:
        dict: ratios for left and right tail
    """
    if not isinstance(pd.Series, type(return_series)):
        return_series = pd.Series(return_series)
    demeaned_series = return_series - return_series.mean()
    _qtiles = demeaned_series.quantile(
        [
            (1 - _3_DEV),
            (1 - _1_DEV),
            _1_DEV,
            _3_DEV,
        ]
    )
    _01p, _32p, _68p, _99p = _qtiles
    return {
        "left": (_01p / _32p) / _NORMAL_DIVIDER,
        "right": (_99p / _68p) / _NORMAL_DIVIDER,
    }


def turnover_series(position_series: pd.Series) -> pd.Series:
    # _annualize_factor = 1 if annualize is False else 252
    _tseries = position_series.diff().abs() / position_series.shift(1)
    # return _tseries * _annualize_factor
    return _tseries
