import numpy as np
import pandas as pd

_CURR = 0


def _buffer_row(row: pd.Series):
    global _CURR
    if _CURR < row.lower:
        _CURR += row.lower - _CURR
    elif _CURR > row.upper:
        _CURR += row.upper - _CURR
    row.buffer = _CURR
    _CURR = row.buffer
    return row


def buffer(
    raw_position_series: pd.Series,
    fraction: float = 0.10,
    return_df=False,
    integrity_check=True,
):
    global _CURR
    rps = raw_position_series
    N = pd.DataFrame(
        {
            "raw": rps,
            "lower": (np.sign(rps) * (np.abs(rps) - rps * fraction)).round(),
            "upper": (np.sign(rps) * (np.abs(rps) + rps * fraction)).round(),
            "buffer": 0.0,
        }
    )
    # N = N.fillna(0)
    N = N.apply(_buffer_row, axis=1)
    _CURR = 0
    if integrity_check:
        if len(N[N["buffer"] < N["lower"]]) != 0:
            raise AssertionError(f'{len(N[N["buffer"] < N["lower"]])=} != 0')
        if len(N[N["buffer"] > N["upper"]]) != 0:
            raise AssertionError(f'{len(N[N["buffer"] > N["upper"]])=} != 0')
    assert _CURR == 0
    return N["buffer"].astype(int) if return_df is False else N


def percentage_returns_series(
    position_series: pd.Series,
    adjusted_price_series: pd.Series,
    multiplier: float,
    capital_series: pd.Series = None,
    fx_series: pd.Series = None,
) -> pd.Series:

    # same as  adjusted_price_series - adjusted_price_series.shift(1)
    price_delta = adjusted_price_series.diff()
    return_price_points = price_delta * position_series.shift(1)

    rtrn_instrmnt_currency = return_price_points * multiplier

    if not isinstance(type(capital_series), pd.Series):
        capital_series = 100_000 if capital_series is None else capital_series
        capital_series = pd.Series(capital_series, index=position_series.index)
    if fx_series is None:
        fx_series_aligned = pd.Series(1, index=rtrn_instrmnt_currency.index)
    else:
        fx_series_aligned = fx_series.reindex(rtrn_instrmnt_currency.index)

    fx_series_aligned.ffill(inplace=True)

    return_base_currency = rtrn_instrmnt_currency * fx_series_aligned
    perc_return = return_base_currency / capital_series

    return perc_return
