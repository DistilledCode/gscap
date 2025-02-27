#
# ! ########################################### ! #
# !                  VERSION 3.0                ! #
# ! ########################################### ! #


import pandas as pd
from gscbt import DataPipeline
from gscbt.utils import Dotdict


def _get_prices(contracts: list[Dotdict], ohlcv: str, interval="1d") -> pd.DataFrame:
    pipe = DataPipeline()

    _map = {
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
    }
    if len(ohlcv) == 1:
        _l = _map[ohlcv]

    else:
        _l = [_map[i] for i in ohlcv]

    # contracts = [i.upper() for i in contracts]
    adjusted = pipe.get(contracts, ohlcv, interval=interval)[_l]
    underlying = pipe.get(contracts, ohlcv, interval=interval, back_adjusted=False)[_l]

    assert all(adjusted.columns == underlying.columns)
    cols = adjusted.columns

    if isinstance(cols, pd.MultiIndex):
        if len(contracts) == 1:
            _adj_col = [("adjusted", i[0].lower()) for i in cols]
            _undrlyng_col = [("underlying", i[0].lower()) for i in cols]
        else:
            _adj_col = [("adjusted", *[k.lower() for k in i]) for i in cols]
            _undrlyng_col = [("underlying", *[k.lower() for k in i]) for i in cols]

    elif isinstance(cols, pd.Index):
        _adj_col = [("adjusted", i.lower()) for i in cols]
        _undrlyng_col = [("underlying", i.lower()) for i in cols]
    multi_columns = _adj_col + _undrlyng_col
    multi_index = pd.MultiIndex.from_tuples(multi_columns)
    df = pd.concat((adjusted, underlying), axis=1)
    df.columns = multi_index
    return df


def open_price(contracts: list[Dotdict], interval="1d") -> pd.DataFrame:
    return _get_prices(contracts=contracts, ohlcv="o", interval=interval)


def high_price(contracts: list[Dotdict], interval="1d") -> pd.DataFrame:
    return _get_prices(contracts=contracts, ohlcv="h", interval=interval)


def low_price(contracts: list[Dotdict], interval="1d") -> pd.DataFrame:
    return _get_prices(contracts=contracts, ohlcv="l", interval=interval)


def close_price(contracts: list[Dotdict], interval="1d") -> pd.DataFrame:
    return _get_prices(contracts=contracts, ohlcv="c", interval=interval)


def ohlcv(contracts: list[Dotdict], ohlcv: str = None, interval="1d") -> pd.DataFrame:
    ohlcv = "ohlcv" if ohlcv is None else ohlcv
    return _get_prices(contracts=contracts, ohlcv=ohlcv, interval=interval)


def volume(contracts: list[Dotdict], interval="1d") -> pd.DataFrame:
    pipe = DataPipeline()

    return pipe.get(contracts, ohclv="v", interval=interval)
