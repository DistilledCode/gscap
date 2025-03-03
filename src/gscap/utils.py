import numpy as np
import pandas as pd
from numba import jit

# def buffer(
#     raw_position_series: pd.Series,
#     fraction: float = 0.10,
#     return_df=False,
#     integrity_check=True,
# ):
#     _curr = 0

#     def _buffer_row(row: pd.Series):
#         nonlocal _curr
#         if _curr < row.lower:
#             _curr += row.lower - _curr
#         elif _curr > row.upper:
#             _curr += row.upper - _curr
#         row.buffer = _curr
#         _curr = row.buffer
#         return row

#     rps = raw_position_series
#     N = pd.DataFrame(
#         {
#             "raw": rps,
#             "lower": (np.sign(rps) * (np.abs(rps) - rps * fraction)).round(),
#             "upper": (np.sign(rps) * (np.abs(rps) + rps * fraction)).round(),
#             "buffer": 0.0,
#         }
#     )
#     # N = N.fillna(0)
#     assert _curr == 0
#     N = N.apply(_buffer_row, axis=1)

#     if integrity_check:
#         if len(N[N["buffer"] < N["lower"]]) != 0:
#             raise AssertionError(f'{len(N[N["buffer"] < N["lower"]])=} != 0')
#         if len(N[N["buffer"] > N["upper"]]) != 0:
#             raise AssertionError(f'{len(N[N["buffer"] > N["upper"]])=} != 0')

#     return N["buffer"].astype(int) if return_df is False else N


@jit(forceobj=True)
def buffer(
    raw_position_series: pd.Series,
    fraction: float = 0.10,
    return_df=False,
    integrity_check=True,
):
    rps = raw_position_series.to_numpy()  # Convert to NumPy for fast operations

    lower = (np.sign(rps) * (np.abs(rps) - rps * fraction)).round()
    upper = (np.sign(rps) * (np.abs(rps) + rps * fraction)).round()

    buffer = np.zeros_like(rps)  # Initialize buffer array
    buffer[0] = 0.0  # Start at lower bound

    # Vectorized loop instead of `apply()`
    for i in range(1, len(rps)):
        buffer[i] = buffer[i - 1]
        if buffer[i] < lower[i]:
            buffer[i] = lower[i]
        elif buffer[i] > upper[i]:
            buffer[i] = upper[i]

    # Integrity check
    if integrity_check:
        if (buffer < lower).any():
            raise AssertionError(f"buffer values below lower bound")
        if (buffer > upper).any():
            raise AssertionError(f"buffer values above upper bound")

    # Return DataFrame if required
    if return_df:
        return pd.DataFrame(
            {"raw": rps, "lower": lower, "upper": upper, "buffer": buffer},
            index=raw_position_series.index,
        )
    return pd.Series(buffer.astype(int), index=raw_position_series.index)


def percentage_returns_series(
    position_series: pd.Series,
    adjusted_price_series: pd.Series,
    multiplier: float,
    capital_series: pd.Series = None,
    fx_series: pd.Series = None,
) -> pd.Series:

    # Calculating SS positions
    # KE, CME, 1d, ins, 8:	first time
    # ZC, CME, 1d, ins, 8:	first time
    # ZL, CME, 1d, ins, 8:	first time
    # Calculating initial SS returns
    # ! len(position_series.index)=6540; len(adjusted_price_series.index)=6540
    # ! len(position_series.index)=6563; len(adjusted_price_series.index)=6563
    # ! len(position_series.index)=6572; len(adjusted_price_series.index)=6572
    # Calculating Instrument Weights
    # Calculating IDM
    # 2.0926734000677243
    # Recalculating SS positions
    # KE, CME, 1d, ins, 8:	scaling position with IDM
    # ZC, CME, 1d, ins, 8:	scaling position with IDM
    # ZL, CME, 1d, ins, 8:	scaling position with IDM
    # Recalculating SS returns
    # ! len(position_series.index)=6681; len(adjusted_price_series.index)=6540
    # ! len(position_series.index)=6681; len(adjusted_price_series.index)=6563
    # ! len(position_series.index)=6681; len(adjusted_price_series.index)=6572
    if all(
        (
            isinstance(adjusted_price_series, pd.DataFrame),
            adjusted_price_series.shape[-1] == 1,
        )
    ):
        adjusted_price_series = adjusted_price_series.squeeze()
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
