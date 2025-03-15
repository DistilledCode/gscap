from dataclasses import dataclass

import numpy as np
import pandas as pd
from gscbt.framework import volatility_scalar

import gscap
from gscap.framework.forecast import Forecast
from gscap.framework.instruments import Instrument
from gscap.framework.utils import get_th_series


@dataclass
class Cost:
    commission_currency: pd.Series = None
    slippage_currency: pd.Series = None
    total_currency: pd.Series = None
    # risk_adj_per_lot: pd.Series = None
    return_series: pd.Series = None


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

        if self.instrument.meta.trading_hours < 0:
            self.instrument.meta.trading_hours = get_th_series(self.instrument)

        if self.instrument.interval == "5m":
            intv_num = gscap.DAYS_IN_YEAR * self.instrument.meta.trading_hours * 12
        elif self.instrument.interval == "1h":
            intv_num = gscap.DAYS_IN_YEAR * self.instrument.meta.trading_hours
        elif self.instrument.interval == "1d":
            intv_num = gscap.DAYS_IN_YEAR
        self.unit_cv_target = self.annual_cv_target / np.sqrt(intv_num)

    @property
    def volatility_scalar(self) -> pd.Series:
        if self._volatility_scalar is None:
            self.vol_scl_log = volatility_scalar(
                price_df=self.instrument.close_price(),
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
        _prs = price_return_series(self, capital_series=self.capital, fx_series=None)
        self.return_series = _prs
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


def price_return_series(
    ss: SubSystem,
    capital_series: pd.Series = None,
    fx_series: pd.Series = None,
) -> pd.Series:
    """
    Calculating SS positions
    KE, CME, 1d, ins, 8:	first time
    ZC, CME, 1d, ins, 8:	first time
    ZL, CME, 1d, ins, 8:	first time
    Calculating initial SS returns
    ! len(position_series.index)=6540; len(adjusted_price_series.index)=6540
    ! len(position_series.index)=6563; len(adjusted_price_series.index)=6563
    ! len(position_series.index)=6572; len(adjusted_price_series.index)=6572
    Calculating Instrument Weights
    Calculating IDM
    2.0926734000677243
    Recalculating SS positions
    KE, CME, 1d, ins, 8:	scaling position with IDM
    ZC, CME, 1d, ins, 8:	scaling position with IDM
    ZL, CME, 1d, ins, 8:	scaling position with IDM
    Recalculating SS returns
    ! len(position_series.index)=6681; len(adjusted_price_series.index)=6540
    ! len(position_series.index)=6681; len(adjusted_price_series.index)=6563
    ! len(position_series.index)=6681; len(adjusted_price_series.index)=6572
    """
    instrument = ss.instrument
    positions = ss.position.round()
    adjusted_price_series = instrument.close_price().adjusted.squeeze()
    price_delta = adjusted_price_series.diff()
    return_price_points = price_delta * positions.shift(1)
    lots_traded = positions.diff().abs()

    slippage_cost_currency = (
        lots_traded
        * instrument.meta.currency_tick_value
        * instrument.meta.cost_in_ticks
    )
    commission_cost_currency = lots_traded * instrument.meta.commission_cost
    total_cost_currency = slippage_cost_currency + commission_cost_currency

    rtrn_instrmnt_currency = return_price_points * instrument.meta.dollar_equivalent

    if not isinstance(type(capital_series), pd.Series):
        capital_series = 100_000 if capital_series is None else capital_series
        capital_series = pd.Series(capital_series, index=positions.index)
    if fx_series is None:
        fx_series_aligned = pd.Series(1.0, index=positions.index)
    else:
        fx_series_aligned = fx_series.reindex(positions.index)
    fx_series_aligned.ffill(inplace=True)

    return_base_currency = rtrn_instrmnt_currency * fx_series_aligned
    tcost_base_currency = total_cost_currency * fx_series_aligned

    gross_perc_return = return_base_currency / capital_series
    cost_in_perc = tcost_base_currency / capital_series

    net_perc_return = gross_perc_return - cost_in_perc

    # _annual_rolling_price_vol = adjusted_price_series.diff().ewm(
    #     span=gscap.DAYS_IN_MONTH
    # ).std() * np.sqrt(gscap.DAYS_IN_YEAR)

    ss.cost.slippage_currency = slippage_cost_currency * fx_series_aligned
    ss.cost.commission_currency = commission_cost_currency * fx_series_aligned
    ss.cost.total_currency = tcost_base_currency
    ss.cost.return_series = cost_in_perc
    ss.cost.return_series.name = ss.instrument.meta.symbol.lower()

    # ss.cost.risk_adj_per_lot = (tcost_base_currency / lots_traded) / (
    #     _annual_rolling_price_vol * instrument.meta.dollar_equivalent
    # )
    return net_perc_return
