from typing import Literal

import numpy as np
from gscbt.utils import Dotdict
from pandas import DataFrame, Series

import gscap.pipe as pipe
from gscap.framework import Forecast
from gscap.framework.forecast.utils import combined_forecast


class Instrument:
    def __init__(
        self,
        meta: Dotdict,
        interval: Literal["1d", "5m"] = "1d",
        period: Literal["ins", "oos", "fbd"] = "ins",
    ):
        self.meta = meta

        self.forecast: list[Forecast] = []
        self.cfs: Series = None
        self.fw: np.array = None
        self.fdm: Series = None
        self.period = period
        self.interval = interval

        self._open_price = None
        self._high_price = None
        self._low_price = None
        self._close_price = None
        self._volume = None
        self._ohlcv = None

        self._last_open = None
        self._last_high = None
        self._last_low = None
        self._last_close = None
        self._last_vol = None
        self._last_ohlcv = None

    def _normalize_params(self, interval=None, period=None):
        """Normalize interval and period parameters to use defaults if None."""
        _intv = interval if interval is not None else self.interval
        _period = period if period is not None else self.period
        return _intv, _period

    def _fetch_cached_data(
        self,
        pipe_func,
        cache_attr,
        last_attr,
        interval=None,
        period=None,
    ):
        """
        Fetch data from cache if available, otherwise fetch from pipe and cache it.

        Args:
            pipe_func: Function to fetch data from pipe
            cache_attr: Attribute name to store cached data
            last_attr: Attribute name to store last used params
            interval: Time interval for data
            period: Time period for data

        Returns:
            DataFrame: The requested data
        """
        interval, period = self._normalize_params(interval, period)

        # Check if cache is valid
        if any(
            (
                getattr(self, last_attr) is None,
                getattr(self, last_attr) != (interval, period),
            )
        ):
            # Fetch new data
            _data = pipe_func([self.meta], interval=interval)
            setattr(self, cache_attr, self._sampled_data(_data, period))
            setattr(self, last_attr, (interval, period))

        return getattr(self, cache_attr)

    def open_price(
        self,
        interval: Literal["1d", "5m"] = None,
        period: Literal["ins", "oos", "fbd"] = None,
    ) -> DataFrame:
        return self._fetch_cached_data(
            pipe.open_price,
            "_open_price",
            "_last_open",
            interval,
            period,
        )

    def high_price(
        self,
        interval: Literal["1d", "5m"] = None,
        period: Literal["ins", "oos", "fbd"] = None,
    ) -> DataFrame:
        return self._fetch_cached_data(
            pipe.high_price,
            "_high_price",
            "_last_high",
            interval,
            period,
        )

    def low_price(
        self,
        interval: Literal["1d", "5m"] = None,
        period: Literal["ins", "oos", "fbd"] = None,
    ) -> DataFrame:
        return self._fetch_cached_data(
            pipe.low_price,
            "_low_price",
            "_last_low",
            interval,
            period,
        )

    def close_price(
        self,
        interval: Literal["1d", "5m"] = None,
        period: Literal["ins", "oos", "fbd"] = None,
    ) -> DataFrame:
        return self._fetch_cached_data(
            pipe.close_price,
            "_close_price",
            "_last_close",
            interval,
            period,
        )

    def volume(
        self,
        interval: Literal["1d", "5m"] = None,
        period: Literal["ins", "oos", "fbd"] = None,
    ) -> DataFrame:
        return self._fetch_cached_data(
            pipe.volume,
            "_volume",
            "_last_vol",
            interval,
            period,
        )

    def ohlcv(
        self,
        interval: Literal["1d", "5m"] = None,
        period: Literal["ins", "oos", "fbd"] = None,
    ) -> DataFrame:
        return self._fetch_cached_data(
            pipe.ohlcv,
            "_ohlcv",
            "_last_ohlcv",
            interval,
            period,
        )

    def _sampled_data(self, data: DataFrame, period: Literal["ins", "oos", "fbd"]):
        if period == "ins":
            _data = data[data.index.year <= 2020]
        elif period == "oos":
            _data = data[data.index.year.isin([2021, 2022])]
        elif period == "fbd":
            _data = data[data.index.year >= 2023]
        if _data.empty:
            raise ValueError(
                f"No data found for {self.meta.symbol} for {period=}."
                f" Range of available data: {data.index[0]} to {data.index[-1]}"
            )
        return _data

    def add_forecast(self, forecast: Forecast | list[Forecast]):
        if isinstance(forecast, list):
            self.forecast.extend(list(set(forecast)))

        elif forecast not in self.forecast:
            self.forecast.append(forecast)

    def generate_forecasts(self):
        for forecast in self.forecast:
            if forecast.is_generated is True:
                continue
            forecast.generate_forecast(self)

    def combine_forecast(self, weights="corr", resample="W") -> DataFrame:
        if self.cfs is None:
            self.cfs = combined_forecast(
                self,
                weights=weights,
                fdm_resample=resample,
            )
            return self.cfs
        return self.cfs

    def __repr__(self):
        return f"{self.meta.symbol}, {self.meta.exchange}"
        # return (
        #     f"{self.meta.product} ({self.meta.symbol}, "
        #     f"{self.meta.exchange}, {self.meta.currency})"
        # )

    def __str__(self):
        return repr(self)

    def __eq__(self, value):
        if isinstance(value, Instrument):
            return hash(self) == hash(value)
        return False

    def __hash__(self):
        return hash(
            (
                self.meta.product,
                self.meta.symbol,
                self.meta.exchange,
                self.meta.currency,
                self.interval,
                self.period,
            )
        )
