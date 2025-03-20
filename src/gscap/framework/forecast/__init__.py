from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pandas import DataFrame

import gscap

if TYPE_CHECKING:
    from gscap.framework.instruments import Instrument

span2hl = lambda x: x * np.log(0.5) / np.log(2 / (x + 1))


def get_vol_regime_multiplier(instrument: Instrument):
    # uvp =  unit_vol_perc(instrument)
    uvp = instrument.unit_vol_perc()
    # for relative vol our lookback period is 5 years
    td = timedelta(days=gscap.DAYS_IN_YEAR * 5)
    rel_vol = uvp / uvp.rolling(window=td).mean().shift(1)
    td = timedelta(days=10)
    # min is 2 - (1.75 * 1.0) = 0.25; max is 2 - (1.75 * 0.0) = 2
    # multiplier = rel_vol.rank(pct=True).mul(-1.75).add(2)
    multiplier = rel_vol.rank(pct=True).mul(-1.50).add(2)
    _hl = timedelta(days=span2hl(10))
    return multiplier.ewm(halflife=_hl, times=multiplier.index.to_series()).mean()


class Forecast(ABC):
    """
    Abstract base class for all forecast types.
    Subclasses must implement the `generate_forecast` method.
    """

    def __init__(self):
        self.is_looking_ahead = True
        self.is_generated = False
        self.ABS_AVG = 10
        self.CLIP = 20
        self.raw_forecast_value: pd.Series = None
        self.forecast_value: pd.Series = None
        self.forecast_scalar: pd.Series = None

    @abstractmethod
    def generate_forecast(self) -> DataFrame:
        """
        Abstract method to generate the forecast.
        Subclasses must implement this method to define their specific logic.
        """
        pass

    def clone(self):
        """Return a deep copy of the forecast instance."""
        return deepcopy(self)

    def scale_and_clip(self, clip=True):
        _rfv = self.raw_forecast_value

        self.forecast_scalar = self.ABS_AVG / _rfv.abs().expanding().mean()
        scaled_fcast = self.raw_forecast_value * self.forecast_scalar

        return scaled_fcast.clip(-self.CLIP, self.CLIP) if clip else scaled_fcast

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, value):
        if isinstance(value, Forecast):
            return hash(self) == hash(value)
        warnings.warn(f"{repr(value)} is not of type Forecast, returning False")
        return False


class VolTargetLongOnly(Forecast):

    def __init__(self):
        super().__init__()

    def generate_forecast(self, instrument: Instrument) -> pd.DataFrame:

        data = instrument.close_price()

        _forecast = pd.DataFrame(
            {instrument.meta.symbol: self.ABS_AVG},
            index=data.index,
        )
        self.is_looking_ahead = False
        self.raw_forecast_value = _forecast
        self.forecast_value = _forecast
        self.is_generated = True

    def __repr__(self):
        return f"BuyAndHold(abs_avg={self.ABS_AVG})"


class EWMACForecast(Forecast):
    """
    Forecast based on Exponentially Weighted Moving Average Crossover (EWMAC).
    """

    def __init__(
        self,
        fast_window_days: int,
        slow_window_days: int = None,
        vol_span_days=35,
    ):
        super().__init__()
        self.fast_window = fast_window_days
        if slow_window_days is not None:
            self.slow_window = slow_window_days
        else:
            self.slow_window = fast_window_days * 4
        self.vol_span = vol_span_days

    def generate_forecast(self, instrument: Instrument) -> pd.DataFrame:
        data = instrument.close_price()
        fw = self.fast_window * instrument.days_look_back()
        sw = self.slow_window * instrument.days_look_back()
        ftrend = data.adjusted.ewm(span=fw).mean()
        strend = data.adjusted.ewm(span=sw).mean()
        _span = self.vol_span * instrument.days_look_back()
        price_volatility = data.adjusted.diff().ewm(span=_span).std().shift(1)
        self.raw_forecast_value = (ftrend - strend) / price_volatility

        processed_fcast = self.scale_and_clip()
        self.forecast_value = processed_fcast.shift(1)
        self.is_looking_ahead = False
        self.is_generated = True

    def __repr__(self):
        return (
            f"EWMACForecast({self.fast_window:>03}, {self.slow_window:>03}, "
            f"{self.CLIP}, {self.ABS_AVG}, {self.vol_span})"
        )

    def __lt__(self, other):
        return self.fast_window < other.fast_window

    def __gt__(self, other):
        return self.fast_window > other.fast_window


class MeanRevForecast(Forecast):
    """
    Forecast based on Mean Reversion.
    """

    def __init__(self, equi_lb_days: int, vol_span=35):
        super().__init__()
        self.equi_lb_days = equi_lb_days
        self.equi_price: pd.Series = None
        self.price_vol: pd.Series = None
        self.vol_span = vol_span

    def generate_forecast(self, instrument: Instrument) -> pd.DataFrame:

        data = instrument.close_price()
        # ! Use Daily EOD price here?
        # daily_close = data.adjusted.resample("D").last().dropna()

        _instrument = instrument.clone()
        _instrument.interval = "1d"
        daily_close_adj = _instrument.close_price().adjusted
        # daily_close_und = _instrument.close_price().underlying

        equi_price = daily_close_adj.ewm(span=self.equi_lb_days).mean().shift(1)
        _ci = equi_price.index.union(data.index)
        equi_price = equi_price.reindex(_ci).ffill().reindex(data.index)
        _span = self.vol_span * instrument.days_look_back()
        price_volatility = daily_close_adj.diff().ewm(span=_span).std().shift(1)
        price_volatility = price_volatility.reindex(_ci).ffill().reindex(data.index)
        self.equi_price = equi_price
        self.price_vol = price_volatility
        self.raw_forecast_value = (equi_price - data.adjusted) / price_volatility

        processed_fcast = self.scale_and_clip(clip=False)
        self.forecast_value = processed_fcast.shift(1)
        self.is_looking_ahead = False
        self.is_generated = True

    def __repr__(self):
        return f"MeanRevForecast({self.equi_lb_days}, {self.vol_span})"

    def __lt__(self, other):
        return self.equi_lb_days < other.equi_lb_days

    def __gt__(self, other):
        return self.equi_lb_days > other.equi_lb_days


# class SafeMeanRevForecast(Forecast):
#     """
#     Forecast based on Safe Mean Reversion.
#     """

#     def __init__(
#         self,
#         equi_lb_days: int,
#         trend_lb_days: int = 16,
#         vol_span: int = 35,
#     ):
#         super().__init__()
#         self.equi_lb_days = equi_lb_days
#         self.trend_lb_days = trend_lb_days
#         self.vol_span = vol_span
#         self.trend = None
#         self.fmrev = None

#     def generate_forecast(self, instrument: Instrument) -> pd.DataFrame:
#         """
#         Big Assumption: Start Date of "1d" data is farther (or equal to)
#         start date of interval of `instrument`
#         """
#         self.fmrev = MeanRevForecast(self.equi_lb_days, self.vol_span)
#         self.fmrev.generate_forecast(instrument)
#         self.trend = EWMACForecast(fast_window_days=self.trend_lb_days)
#         instrument_cloned = instrument.clone()
#         # instrument_cloned.interval = "5m"
#         self.trend.generate_forecast(instrument=instrument_cloned)

#         _trend_sd = self.trend.forecast_value.index[0]
#         _fmrev_sd = self.fmrev.forecast_value.index[0]

#         if not _trend_sd <= _fmrev_sd:
#             _str = (
#                 f"`1d` data for EWMA is of shorter span than "
#                 f"`{instrument.interval}` data for MeanReversion of {instrument}"
#             )
#             raise AssertionError(_str)

#         _ci = self.fmrev.forecast_value.index.union(self.trend.forecast_value.index)
#         _trend = (
#             self.trend.forecast_value.reindex(_ci)
#             .ffill()
#             .reindex_like(self.fmrev.forecast_value)
#         )
#         _overlay = _trend.apply(np.sign) * self.fmrev.forecast_value.apply(np.sign)
#         _overlay = _overlay.replace(-1.0, 0)

#         self.raw_forecast_value = self.fmrev.raw_forecast_value * _overlay
#         self.raw_forecast_value = self.raw_forecast_value.squeeze()
#         _vol_multiplier = get_vol_regime_multiplier(instrument)
#         self.raw_forecast_value = self.raw_forecast_value * _vol_multiplier

#         processed_fcast = self.scale_and_clip()
#         self.forecast_value = processed_fcast  # ! No .shift(1) as already done by FMR
#         self.is_looking_ahead = False
#         self.is_generated = True

#     def __repr__(self):
#         return f"SafeMeanRevForecast({self.equi_lb_days}, {self.vol_span})"

#     def __lt__(self, other):
#         return self.equi_lb_days < other.equilibriuim_lookback

#     def __gt__(self, other):
#         return self.equi_lb_days > other.equilibriuim_lookback
