from typing import Literal

import numpy as np
import pandas as pd

_NORMAL_DIVIDER = 5.854340606494375
_1_DEV = 0.682689492137086
_3_DEV = 0.997300203936740


def equi_vol(
    return_series: pd.Series,
    rolling_window=22,
    min_periods=None,
    annualize=True,
    periods_per_year=252,
) -> pd.DataFrame:
    """
    Returns equi-weghted volatility for given rolling window.
    Assumes input to be daily data
    Args:
        return_series (pd.Series): return series in fraction
        rolling_window (int, 22): Window for equally-weighted averaging.
        annualized (bool, True): Whether to annualize the volatility.

    Returns:
        pd.DataFrame
    """
    _rw = rolling_window
    _mp = _rw if min_periods is None else min_periods
    _vol = return_series.rolling(_rw, min_periods=_mp).std()
    if annualize:
        _vol *= np.sqrt(periods_per_year)
    return _vol


def ewma_vol(
    return_series: pd.Series,
    halflife=11,
    min_periods=None,
    annualize=True,
    periods_per_year=252,
) -> pd.DataFrame:
    """
    Returns exponentially-weghted volatility for given half-life.
    Assumes input to be daily data
    Args:
        return_series (pd.Series): return series in fraction
        half_life (int, 11): Half-life for weighing.
        annualized (bool, True): Whether to annualize the volatility.

    Returns:
        pd.DataFrame
    """
    _hl = halflife
    _mp = _hl * 2 if min_periods is None else min_periods
    _vol = return_series.ewm(halflife=_hl, min_periods=_mp).std()
    if annualize:
        _vol *= np.sqrt(periods_per_year)
    return _vol


def sharpe(return_series: pd.Series, annualize=True, period_per_year=252):
    sharpe = return_series.mean() / return_series.std()
    return sharpe * np.sqrt(period_per_year) if annualize else sharpe


def rolling_sharpe(
    return_series: pd.Series,
    periods=22 * 6,
    weights: Literal["equi", "ewma"] = "equi",
    annualize=True,
    periods_per_year=252,
):
    if weights == "equi":
        _rw = periods
        _mp = _rw
        _wseries = return_series.rolling(window=_rw, min_periods=_mp)

    elif weights == "ewma":
        _hl = periods // 2
        _mp = _hl * 2
        _wseries = return_series.ewm(halflife=_hl, min_periods=_mp)

    _rs = _wseries.mean() / _wseries.std()

    if annualize:
        _rs *= np.sqrt(periods_per_year)

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


def turnover_series(position_series: pd.Series, annualize=True) -> pd.Series:
    _annualize_factor = 1 if annualize is False else 252
    _tseries = position_series.diff().abs() / position_series.shift(1)
    return _tseries * _annualize_factor
