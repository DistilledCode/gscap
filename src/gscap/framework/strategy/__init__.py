# from time import perf_counter
from typing import Any, Literal, Optional

# from gscbt.framework import instrument_weight
import numpy as np
import pandas as pd
from gscbt.utils import Dotdict

import gscap
import gscap.framework.strategy.analyse as analysis
from gscap.framework.forecast import Forecast
from gscap.framework.instruments import Instrument
from gscap.framework.strategy.calculations import buffer, calculate_idm
from gscap.framework.subsystem import SubSystem


def instrument_weight(returns: pd.DataFrame, resample="YE", n_itr=100, frac=0.1):
    return pd.DataFrame(
        1 / len(returns.columns),
        columns=returns.columns,
        index=returns.index,
    )


def top_down_rw(
    fmapping: dict[Instrument, Forecast | list[Forecast]],
) -> dict[Instrument, float]:
    instruments = fmapping.keys()
    classification = dict()
    for i in instruments:
        classification.setdefault(i.meta.asset_class, dict())
        classification[i.meta.asset_class].setdefault(i.meta.product_group, [])
        classification[i.meta.asset_class][i.meta.product_group].append(i)
    VOL_W = 0.05
    NORM_FACTOR = 1 - VOL_W
    weights = {classification.pop("Volatility")["Volatility"][0]: VOL_W}
    _ac_weights = {i: 1 / len(classification) for i in classification}
    _ac_weights
    _pg_weights = {}
    for k, v in classification.items():
        _pg_weights |= {i: (1 / len(v)) * _ac_weights[k] for i in v}
        for prod_type, cntrcts in v.items():
            _n = len(cntrcts)
            for contract in cntrcts:
                weights[contract] = 1 / _n * _pg_weights[prod_type] * NORM_FACTOR
    assert np.allclose(sum(weights.values(), 0), 1), "weight not summing up to 1"
    return weights


class Strategy:

    def __init__(
        self,
        forecasts: Forecast | list[Forecast] | dict[Any, Forecast | list[Forecast]],
        contracts: Optional[list[Dotdict]] = None,
        capital: float | int = 1_000_000,
        tau: float = 0.2,
        interval: Literal["5m", "1h", "1d"] = "1d",
        period: Literal["ins", "oos", "fbd"] = "ins",
        fdm_resample=None,
        risk_weights: Optional[dict[Dotdict, float] | str] = None,
        buffer_fraction: Optional[float] = None,
        name: Optional[str] = "strategy_xx",
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
            )
            for inst, fcasts in self.fmapping.items()
        ]

        if self.risk_weights is None:
            _n = len(self.fmapping)
            self.risk_weights = {inst: 1 / _n for inst in self.fmapping}
        elif self.risk_weights == "td":
            self.risk_weights = top_down_rw(self.fmapping)

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
        self.idm = calculate_idm(self.indv_return_series, self.instrument_weights)
        # self.idm /= 2

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
        analysis.metric_table(self)
        analysis.strategy_plots(self, show=show)
        analysis.analyse_cost(self, show=show)

    def compare(self, other_strategy, show=True):
        if not isinstance(other_strategy, Strategy):
            raise ValueError("Pass another `Strategy` instance for comaparison")

        if self.name == other_strategy.name:
            _err = f"Conflicting Strategy names: {repr(self.name)} & {repr(other_strategy.name)}"
            raise RuntimeError(_err)

        analysis.metric_table(self, benchmark=other_strategy)
        analysis.strategy_plots(self, benchmark=other_strategy, show=show)
        analysis.analyse_cost(self, benchmark=other_strategy, show=show)

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
                "Can only be `None` or `float`",
            )
            raise ValueError(_str)
