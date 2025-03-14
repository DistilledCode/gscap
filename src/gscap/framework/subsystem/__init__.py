import numpy as np
import pandas as pd
from gscbt.framework import volatility_scalar

import gscap
from gscap.framework.forecast import Forecast
from gscap.framework.instruments import Instrument
from gscap.framework.subsystem.calculations import prs_with_cost
from gscap.framework.utils import Cost, analyse_cost, get_th_series


class SubSystem:
    def __init__(
        self,
        instrument: Instrument,
        forecasts: list[Forecast] | Forecast,
        annual_cash_vol_tgt: float,
        capital: float | int = 1_000_000,
        fdm_resample=gscap.FDM_RESAMPLE,
    ):
        self.instrument = instrument.clone()
        self.forecasts = (
            [i.clone() for i in forecasts]
            if isinstance(forecasts, list)
            else forecasts.clone()
        )
        self.annual_cv_target = annual_cash_vol_tgt
        self.fdm_resample = fdm_resample
        self.capital = capital
        self._volatility_scalar = None
        self._position = None
        self.return_series = None
        self.cost = Cost()
        self.vol_scl_log: pd.DataFrame = None
        if self.instrument.interval == "5m":
            if self.instrument.meta.trading_hours < 0:
                self.instrument.meta.trading_hours = get_th_series(self.instrument)
            intv_num = gscap.DAYS_IN_YEAR * self.instrument.meta.trading_hours * 12
        elif self.instrument.interval == "1d":
            intv_num = gscap.DAYS_IN_YEAR
        self.unit_cv_target = self.annual_cv_target / np.sqrt(intv_num)

    @property
    def volatility_scalar(self) -> pd.Series:
        if self._volatility_scalar is None:
            self.vol_scl_log = volatility_scalar(
                price_df=self.instrument.close_price(),
                # return_callable=get_returns,
                ticker=self.instrument.meta,
                unit_cash_volatility_target=self.unit_cv_target,
                slow_span=gscap.VOL_SCLR_LBACK_SPAN_SLOW,
                fast_span=gscap.VOL_SCLR_LBACK_SPAN_FAST,
            )
            self._volatility_scalar = self.vol_scl_log.vol_scalar.squeeze()
        return self._volatility_scalar

    @property
    def position(self) -> pd.Series:
        """
        # Regarding rounding off
        - SubSystem positions will stay unrounded as buffering requires unrounded positions
        - If the Strategy class doesn't buffer the positions then SubSystem postions will
            stay unrouded, else they'll be rounded
        - Positions of all instrument in Strategy attribute `positions` are rounded
        - For return_series calculation, we'll explicitly round the postion inside the func

        # Regarding scaling with risk weights
        - in `_calculate_ss_positions()` method of Strategy the `.position` attribute of
            SubSystem get scaled down with the risk weight and then we calculate the returns.
        """
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
        # if self.include_cost is False:
        #     _prs = percentage_returns_series(
        #         self.position,
        #         self.instrument.close_price().adjusted,
        #         multiplier=self.instrument.meta.dollar_equivalent,
        #         capital_series=self.capital,
        #         fx_series=None,
        #     )
        #     self._set_cost_zero()
        # else:
        _prs = prs_with_cost(self, capital_series=self.capital, fx_series=None)
        self.return_series = _prs
        # print(_prs.isna().value_counts() / len(_prs))
        # self.return_series = _prs.fillna(0.0)
        self.return_series.name = self.instrument.meta.symbol.lower()
        return self.return_series

    def __repr__(self):
        _n = 1 if not isinstance(self.forecasts, list) else len(self.forecasts)
        return (
            f"{self.instrument}, {self.instrument.interval}, "
            f"{self.instrument.period}, {_n}"
        )

    def _set_cost_zero(self):
        self.cost.slippage_currency = pd.Series(0.0, index=self.position.index)
        self.cost.commission_currency = pd.Series(0.0, index=self.position.index)
        self.cost.total_currency = pd.Series(0.0, index=self.position.index)
        self.cost.risk_adj_per_lot = pd.Series(0.0, index=self.position.index)
        self.cost.return_series = pd.Series(0.0, index=self.position.index)

    def analyse_cost(self, show=True):
        analyse_cost(
            return_series=self.return_series,
            cost_return_series=self.cost.return_series,
            show=show,
        )
