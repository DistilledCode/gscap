import numpy as np
import pandas as pd
from fractions import Fraction
import math
from functools import reduce
from gscbt.data import get_spread, RollMethod
from gscbt import Ticker, Cache
from synthetic_builder_py import wrapper

cme = Ticker.TICKERS.cme
iceec = Ticker.TICKERS.iceec
ice = Ticker.TICKERS.ice
icefu = Ticker.TICKERS.icefu


def minfluc(_data: pd.DataFrame | pd.Series | float, tick_size: float):
    return round(_data / tick_size) * tick_size


def lcm(a, b):
    """Compute the least common multiple of two integers."""
    return a * b // math.gcd(a, b)


def hcf_of_fractions(*fractions):
    """
    Compute the HCF of n fractions represented as floats.
    Returns the result as a float, consistent with the original function.
    """
    # Convert each float to an integer ratio (numerator, denominator)
    fractions = fractions[0]
    ratios = [Fraction(f).limit_denominator(1e10).as_integer_ratio() for f in fractions]

    # Extract numerators and denominators
    numerators = [n for n, d in ratios]
    denominators = [d for n, d in ratios]

    # Compute GCD of all numerators
    gcd_num = reduce(math.gcd, numerators)

    # Compute LCM of all denominators
    lcm_den = reduce(lcm, denominators)

    # Simplify the resulting fraction gcd_num / lcm_den
    common = math.gcd(gcd_num, lcm_den)
    simplified_num = gcd_num // common
    simplified_den = lcm_den // common

    # Return the result as a float, matching the original function's behavior

    return simplified_num / simplified_den


def get_data(
    exp: str,
    tickers: list,
    sd: str,
    ed: str,
    badj: bool,
    roll_over: int,
    styr: int = 2010,
    badjmode: int = 1,
):

    _data = wrapper(
        exp=exp,
        back_adjustd=badj,
        start_year=styr,
        offset=roll_over,
        max_lookback_for_back_adjust=10,
        back_adjust_mode=badjmode,
    )

    # _data = get_spread(
    #     expression=exp,
    #     ohlcv="ohlc",
    #     back_adjusted=badj,
    #     start=sd,
    #     end=ed,
    #     roll_method=RollMethod.contractwise,
    #     cache_mode=Cache.Mode.market_api,
    #     max_lookback_for_back_adjust=10,
    #     verbose=False,
    # )
    _data.dropna(inplace=True)
    if tickers is None:
        return _data
    else:
        hcf = hcf_of_fractions([i.min_price_fluctuation for i in tickers])
        # for c in ("open", "high", "low", "close"):
        # for c in ("open", "high", "low", "close"):
        _data.close = minfluc(_data.close, hcf)
        return _data


def asymmetric_filter_iterative(series, rise_factor=1, fall_factor=0.01):
    if not (0 < rise_factor <= 1 and 0 < fall_factor <= 1):
        raise ValueError("Rise and fall factors must be between 0 and 1.")

    filtered_series = np.zeros_like(series, dtype=float)
    if len(series) == 0:
        return filtered_series

    filtered_series[0] = series[0]  # Initialize the first value

    for i in range(1, len(series)):
        current_value = series[i]
        previous_filtered_value = filtered_series[i - 1]

        if current_value > previous_filtered_value:
            filtered_series[i] = previous_filtered_value + rise_factor * (
                current_value - previous_filtered_value
            )
        else:
            filtered_series[i] = previous_filtered_value + fall_factor * (
                current_value - previous_filtered_value
            )

    return filtered_series


def get_ratcheted(_margin: pd.Series, fall_factor=0.01):

    _margin = _margin.dropna()
    return pd.Series(
        asymmetric_filter_iterative(_margin, fall_factor=fall_factor),
        index=_margin.index,
        name=f"{_margin.name}_ratcheted",
    )


def get_synth_slices(data: pd.DataFrame):

    index_slices = [
        data.index[0],
        *data[data.days_to_roll.dt.days == 0].index,
        data.index[-1],
    ]
    for i in range(len(index_slices) - 1):
        if index_slices[i] == index_slices[i + 1]:
            # for the case when last row has days_to_roll == 0
            continue
        # This will give us inclusive of both date, eg: 30 June 2005 to 30 June 2006
        _d = data.loc[index_slices[i] : index_slices[i + 1]]
        # But we actually need 01 July 2005 to 30 June 2006
        _d = _d.iloc[1:]
        yield ((_d.index[0], _d.index[-1]), _d)


def get_index_map(synthetic_sliced, data_nbadj):

    roll_index_date_map = (
        pd.DataFrame(
            [
                [ind, pd.date_range(start=i[0][0], end=i[0][1]).to_list()]
                for ind, i in enumerate(synthetic_sliced)
            ],
            columns=["roll_index", "date"],
        )
        .explode("date")
        .set_index("date")
    )
    roll_index_date_map = roll_index_date_map.reindex(
        roll_index_date_map.index.union(data_nbadj.index)
    )
    roll_index_date_map.fillna(0, inplace=True)
    roll_index_date_map.roll_index = roll_index_date_map.roll_index.astype(int)

    skip_index = [
        index
        for index, count in (
            roll_index_date_map.reset_index()
            .groupby("roll_index")
            .agg("count")["index"]
        ).items()
        if count < 300
    ]

    roll_index_date_map = roll_index_date_map.to_dict(orient="index")
    return roll_index_date_map, skip_index
