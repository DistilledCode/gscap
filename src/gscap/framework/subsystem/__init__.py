import pandas as pd
from gscbt.framework import volatility_scalar

import gscap
from gscap.framework.forecast import Forecast
from gscap.framework.instruments import Instrument
from gscap.utils import percentage_returns_series


def get_returns(price_df=pd.DataFrame):
    return price_df.adjusted.diff() / price_df.underlying.shift(1)


class SubSystem:
    def __init__(
        self,
        instruement: Instrument,
        forecasts: list[Forecast] | Forecast,
        unit_cash_vol_tgt: float,
        capital: float | int = 1_000_000,
        fdm_resample="B",
    ):
        self.instrument = instruement
        self.forecasts = (
            [i.clone() for i in forecasts]
            if isinstance(forecasts, list)
            else forecasts.clone()
        )
        self.unit_cv_target = unit_cash_vol_tgt
        self.fdm_resample = fdm_resample
        self.capital = capital
        self._volatility_scalar = None
        self._position = None
        self.return_series = None

    @property
    def volatility_scalar(self) -> pd.Series:
        if self._volatility_scalar is None:
            # print(gscap.VOL_LOOKBACK_SPAN)
            self._volatility_scalar = volatility_scalar(
                price_df=self.instrument.close_price(),
                return_callable=get_returns,
                ticker=self.instrument.meta,
                unit_cash_volatility_target=self.unit_cv_target,
                span=gscap.VOL_LOOKBACK_SPAN,
            )
        return self._volatility_scalar

    @property
    def position(self) -> pd.Series:
        if self._position is None:
            if self.instrument.cfs is None:
                self._process_forecasts()
            self._position = self.instrument.cfs * self.volatility_scalar
            if not isinstance(self.forecasts, list):
                _abs_avg = self.forecasts.ABS_AVG
            else:
                _abs_avg = self.forecasts[0].ABS_AVG
            self._position /= _abs_avg
            self._position.name = self.instrument.meta.symbol.lower()
        return self._position

    @position.setter
    def position(self, value: pd.Series) -> None:

        self._position = value
        self._position.name = self.instrument.meta.symbol.lower()

    def _process_forecasts(self):
        self.instrument.add_forecast(self.forecasts)
        self.instrument.generate_forecasts()
        self.instrument.combine_forecast(resample=self.fdm_resample)

    def calculate_return_series(self) -> pd.Series:
        _prs = percentage_returns_series(
            self.position,
            self.instrument.close_price().adjusted,
            multiplier=self.instrument.meta.dollar_equivalent,
            capital_series=self.capital,
            fx_series=None,
        )
        self.return_series = _prs.fillna(0)
        self.return_series.name = self.instrument.meta.symbol.lower()
        return self.return_series

    def __repr__(self):
        _n = 1 if not isinstance(self.forecasts, list) else len(self.forecasts)
        return (
            f"{self.instrument}, {self.instrument.interval}, "
            f"{self.instrument.period}, {_n}"
        )
