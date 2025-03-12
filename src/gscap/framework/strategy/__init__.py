# from time import perf_counter
from typing import Any, Literal, Optional

# from gscbt.framework import instrument_weight
import numpy as np
import pandas as pd
from gscbt.utils import Dotdict

import gscap

# from gscap import plot
from gscap.framework.forecast import Forecast
from gscap.framework.instruments import Instrument

# import matplotlib.pyplot as plt
from gscap.framework.strategy.analyse import _metric_table, _strategy_plots
from gscap.framework.strategy.calculations import buffer, calculate_idm
from gscap.framework.subsystem import SubSystem
from gscap.framework.utils import analyse_cost


def instrument_weight(returns: pd.DataFrame, resample="YE", n_itr=100, frac=0.1):
    return pd.DataFrame(
        1 / len(returns.columns),
        columns=returns.columns,
        index=returns.index,
    )


class Strategy:

    def __init__(
        self,
        forecasts: Forecast | list[Forecast] | dict[Any, Forecast | list[Forecast]],
        contracts: Optional[list[Dotdict]] = None,
        capital: float | int = 1_000_000,
        tau: float = 0.2,
        interval: Literal["1d", "5m"] = "1d",
        period: Literal["ins", "oos", "fbd"] = "ins",
        fdm_resample=None,
        risk_weights=None,
        buffer_fraction: Optional[float] = None,
        name: Optional[str] = "strategy_xx",
        # include_cost: bool = True,
    ):
        self.name = name
        self.contracts = contracts
        self.forecasts = forecasts
        self.capital = capital
        self.tau = tau
        self.annual_cvt = self.capital * self.tau
        self.interval = interval
        self.period = period
        self.fdm_resample = gscap.FDM_RESAMPLE if fdm_resample is None else fdm_resample
        self.risk_weights = risk_weights
        self.buffer_fraction = buffer_fraction
        # self.include_cost = include_cost
        self.fmapping = None
        self.subsystems = None
        self.indv_return_series: pd.DataFrame = None
        self.aggr_return_series: pd.Series = None
        self.indv_cost_return_series: pd.DataFrame = None
        self.aggr_cost_return_series: pd.Series = None
        self.instrument_weights: pd.DataFrame = None
        self.idm: Optional[pd.Series] = None

        if self.interval == "5m":
            gscap.VOL_LOOKBACK_SPAN = 12
        elif self.interval == "1d":
            gscap.VOL_LOOKBACK_SPAN = 5

    def init(self):

        self._process_fmapping()
        self._sanity_check()

        self.subsystems = [
            SubSystem(
                instrument=inst,
                forecasts=fcasts,
                annual_cash_vol_tgt=self.annual_cvt,
                fdm_resample=self.fdm_resample,
                capital=self.capital,
                # include_cost=self.include_cost,
            )
            for inst, fcasts in self.fmapping.items()
        ]

        if self.risk_weights is None:
            _n = len(self.fmapping)
            self.risk_weights = {inst: 1 / _n for inst in self.fmapping}

    # ! This bad boy will get parallelized!!
    def _calculate_ss_positions(self):
        for ss in self.subsystems:
            print(ss, end="")
            if self.idm is None:
                # As IDM series is None, we are calculating ss positions for the first time
                print(":\tfirst time")
                ss.position = ss.position * self.risk_weights.get(ss.instrument)
            elif isinstance(self.idm, pd.Series):
                # We got IDM, we are just scaling our positions
                print(":\tscaling position with IDM")
                ss.position = ss.position.multiply(self.idm, axis=0)
            if self.buffer_fraction:
                ss.position = buffer(ss.position, fraction=self.buffer_fraction)
        self.positions = pd.concat((ss.position for ss in self.subsystems), axis=1)
        self.positions = self.positions.round()

    def _calculate_ss_return_series(self):
        for ss in self.subsystems:
            ss.calculate_return_series()
        self.indv_return_series = pd.concat(
            [ss.return_series for ss in self.subsystems],
            axis=1,
        )
        self.aggr_return_series = self.indv_return_series.sum(axis=1)
        self.aggr_return_series.name = self.name

        self.indv_cost_return_series = pd.concat(
            [ss.cost.return_series for ss in self.subsystems],
            axis=1,
        )
        self.aggr_cost_return_series = self.indv_cost_return_series.sum(axis=1)
        self.aggr_cost_return_series.name = f"{self.name} Cost"

    def _calculate_instrument_weights(self):
        self.instrument_weights = instrument_weight(self.indv_return_series)

    def _calculate_idm(self):
        if len(self.fmapping) > 1:
            self.idm = calculate_idm(self.indv_return_series, self.instrument_weights)
        elif len(self.fmapping) == 1:
            self.idm = pd.Series(1.0, index=self.instrument_weights.index)

        self.idm.name = self.name

    def compile(self):
        self.init()
        print("Calculating SS positions")
        self._calculate_ss_positions()
        print("Calculating initial SS returns")
        self._calculate_ss_return_series()
        print("Calculating Instrument Weights")
        self._calculate_instrument_weights()
        # return
        print("Calculating IDM")
        # s = perf_counter()
        self._calculate_idm()
        # print(perf_counter() - s)
        print("Recalculating SS positions")
        self._calculate_ss_positions()
        print("Recalculating SS returns")
        self._calculate_ss_return_series()

    def analyse(self, show=True):
        _strategy_plots(self, show=show)
        _metric_table(self)

    def compare(self, other_strategy, show=True):
        if not isinstance(other_strategy, Strategy):
            raise ValueError("Pass another `Strategy` instance for comaparison")

        if self.name == other_strategy.name:
            _err = f"Conflicting Strategy names: {repr(self.name)} & {repr(other_strategy.name)}"
            raise RuntimeError(_err)
        _strategy_plots(self, benchmark=other_strategy, show=show)
        _metric_table(self, benchmark=other_strategy)

    def _process_fmapping(self):

        if isinstance(self.forecasts, dict) and self.contracts is None:
            self.fmapping = {
                Instrument(
                    meta=cnt_meta,
                    period=self.period,
                    interval=self.interval,
                ): fcast
                for cnt_meta, fcast in self.forecasts.items()
            }
        elif not isinstance(self.forecasts, dict) and self.contracts is not None:
            self.fmapping = {
                Instrument(
                    meta=cnt_meta,
                    period=self.period,
                    interval=self.interval,
                ): self.forecasts
                for cnt_meta in self.contracts
            }
        else:
            raise ValueError(
                "If `forecasts` is dict then `contracts` must be None.",
                "If `contracts` is not None then `forecasts` cannot be dict;"
                "it must be `Forecast | list[Forecast]`",
            )

    def _sanity_check(self):
        for inst in self.fmapping:
            if any(
                (
                    np.isnan(inst.meta.dollar_equivalent) is np.True_,
                    np.isnan(inst.meta.currency_multiplier) is np.True_,
                )
            ):
                raise AttributeError(
                    f"NaN values for multipliers of {inst}: "
                    f"{inst.meta.dollar_equivalent=}; {inst.meta.currency_multiplier=}"
                )
            if np.isnan(inst.meta.currency_tick_value) is np.True_:
                raise AttributeError(f"NaN values for tick value of {inst}")

        _symbols = [instrmnt.meta.symbol for instrmnt in self.fmapping]
        assert len(set(_symbols)) == len(_symbols), "Symbols not unique!"

        if isinstance(self.buffer_fraction, bool):
            _str = (
                f"Invalid type for {type(self.buffer_fraction)=}; ",
                f"Can only be `None` or `float`",
            )
            raise ValueError(_str)

    def analyse_cost(self, show=True):
        analyse_cost(
            return_series=self.aggr_return_series,
            cost_return_series=self.aggr_cost_return_series,
            show=show,
        )
