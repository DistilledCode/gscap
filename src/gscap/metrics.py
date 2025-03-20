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


def turnover_series(
    positions_data: pd.Series | pd.DataFrame,
    hrs_in_day: int | pd.Series = 24,
) -> pd.Series:

    if isinstance(positions_data, pd.Series):
        return _turnover_series_from_series(positions_data, hrs_in_day)
    elif isinstance(positions_data, pd.DataFrame):
        return _turnover_series_from_df(positions_data, hrs_in_day)


def _turnover_series_from_series(
    positions_series: pd.Series,
    hrs_in_day: int | pd.Series = 24,
) -> pd.Series:

    intv = positions_series.interval()
    annualizing_factor = (252 * hrs_in_day * 3600) / intv
    abv_pos = positions_series.abs().expanding().mean().shift(1)
    delta_pos = positions_series.diff().abs()
    _tseries = delta_pos / abv_pos
    _tseries.name = positions_series.name
    return _tseries * annualizing_factor


def _turnover_series_from_df(
    positions_df: pd.DataFrame,
    hrs_in_day: int | pd.Series = 24,
) -> pd.Series:

    intv = positions_df.interval()
    annualizing_factor = (252 * hrs_in_day * 3600) / intv
    abv_pos = positions_df.abs().sum(axis=1).expanding().mean().shift(1)
    delta_pos = positions_df.diff().abs().sum(axis=1)
    _tseries = delta_pos / abv_pos
    return _tseries * annualizing_factor


def long_trades(position_df: pd.DataFrame):

    total = np.sum((~position_df.diff().isna()).astype(int).values)
    long = np.sum(position_df.diff().ge(0).astype(int).values)
    zero = np.sum((position_df.diff().eq(0)).astype(int).values)
    if total == 0:
        return np.nan
    return (long - zero) / (total - zero)
