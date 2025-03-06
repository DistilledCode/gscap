from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

import pandas as pd
from pandas import DataFrame

if TYPE_CHECKING:
    from gscap.framework import Instrument


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
        self.forecast_value = None

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

    def __hash__(self):
        return hash(repr(self))

    def __eq__(self, value):
        if isinstance(value, Forecast):
            return hash(self) == hash(value)
        warnings.warn(f"{repr(value)} is not of type Forecast, returning False")
        return False


class EWMACForecast(Forecast):
    """
    Forecast based on Exponentially Weighted Moving Average Crossover (EWMAC).
    """

    def __init__(
        self,
        fast_window: int,
        slow_window: int = None,
        vol_span=35,
    ):
        super().__init__()
        self.fast_window = fast_window
        self.slow_window = slow_window if slow_window is not None else fast_window * 4
        self.vol_span = vol_span

    def generate_forecast(self, instrument: Instrument) -> pd.DataFrame:

        data = instrument.close_price()

        ftrend = data.adjusted.ewm(span=self.fast_window).mean()
        strend = data.adjusted.ewm(span=self.slow_window).mean()
        price_volatility = data.adjusted.diff().ewm(span=self.vol_span).std()
        raw_fcast = (ftrend - strend) / price_volatility

        scaled_fcast = raw_fcast * self.ABS_AVG / raw_fcast.abs().expanding().mean()
        clipped_fcast = scaled_fcast.clip(-self.CLIP, self.CLIP)
        self.forecast_value = clipped_fcast.shift(1)
        self.is_looking_ahead = False
        self.is_generated = True

    def __repr__(self):
        return (
            f"EWMACForecast({self.fast_window}, {self.slow_window}, "
            f"{self.CLIP}, {self.ABS_AVG}, {self.vol_span})"
        )


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
        self.forecast_value = _forecast
        self.is_generated = True

    def __repr__(self):
        return f"BuyAndHold(abs_avg={self.ABS_AVG})"
