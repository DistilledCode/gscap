from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from prettytable import PrettyTable

import gscap
from gscap import plot
from gscap.metrics import drawdown_series, tail_ratios

if TYPE_CHECKING:
    from gscap.framework.strategy import Strategy


def _strategy_plots(main_strat: Strategy, benchmark: Strategy = None, show=True):
    main_rs = main_strat.aggr_return_series
    bench_rs = benchmark.aggr_return_series if benchmark is not None else None
    plot.returns(main_rs, benchmark=bench_rs, cumulative=False, show=show)
    plot.returns(main_rs, benchmark=bench_rs, cumulative=True, show=show)
    plot.return_histogram(main_rs, benchmark=bench_rs, show=show)
    plot.eoy_returns(main_rs, benchmark=bench_rs, show=show)
    plot.rolling_volatility(main_rs, benchmark=bench_rs, show=show)
    plot.rolling_sharpe(main_rs, benchmark=bench_rs, show=show)
    plot.drawdown(main_rs, benchmark=bench_rs, show=show)
    plot.monthly_heatmap(main_rs, benchmark=bench_rs, show=show)


def stats(strat: Strategy):

    daily_rtr = strat.aggr_return_series.resample("D").sum()
    daily_rtr = daily_rtr[~daily_rtr.eq(0.0)]
    monthly_rtr = strat.aggr_return_series.resample("ME").sum()
    monthly_rtr = monthly_rtr[~monthly_rtr.eq(0.0)]
    yearly_rtr = strat.aggr_return_series.resample("YE").sum()
    yearly_rtr = yearly_rtr[~yearly_rtr.eq(0.0)]

    annualized_daily_rtr = daily_rtr.mean() * gscap.DAYS_IN_YEAR
    annualized_daily_std = daily_rtr.std() * np.sqrt(gscap.DAYS_IN_YEAR)
    daily_sharpe = annualized_daily_rtr / annualized_daily_std

    annualized_monthly_rtr = monthly_rtr.mean() * 12
    annualized_monthly_std = monthly_rtr.std() * np.sqrt(12)
    monthly_sharpe = annualized_monthly_rtr / annualized_monthly_std

    dd = drawdown_series(daily_rtr, cumulative=True)
    max_dd = dd.min() * 100
    mean_dd = dd.mean() * 100
    tratios_daily = tail_ratios(daily_rtr)
    tratios_monthly = tail_ratios(monthly_rtr)
    daily_skew = daily_rtr.skew()
    monthly_skew = monthly_rtr.skew()
    expected_daily_rtr = daily_rtr.mean() * 100
    expected_monthly_rtr = monthly_rtr.mean() * 100
    expected_yearly_rtr = yearly_rtr.mean() * 100
    _1p_d, _5p_d = daily_rtr.quantile([0.01, 0.05])
    _1p_m, _5p_m = monthly_rtr.quantile([0.01, 0.05])
    daily_1p_var = _1p_d * 100
    daily_5p_var = _5p_d * 100
    monthly_1p_var = _1p_m * 100
    monthly_5p_var = _5p_m * 100
    daily_1p_cvar = daily_rtr[daily_rtr <= _1p_d].mean() * 100
    daily_5p_cvar = daily_rtr[daily_rtr <= _5p_d].mean() * 100
    monthly_1p_cvar = monthly_rtr[monthly_rtr <= _1p_m].mean() * 100
    monthly_5p_cvar = monthly_rtr[monthly_rtr <= _5p_m].mean() * 100
    best_day = daily_rtr.max() * 100
    best_month = monthly_rtr.max() * 100
    best_year = yearly_rtr.max() * 100
    worst_day = daily_rtr.min() * 100
    worst_month = monthly_rtr.min() * 100
    worst_year = yearly_rtr.min() * 100
    win_days = daily_rtr[daily_rtr > 0].count() / len(daily_rtr) * 100
    win_months = monthly_rtr[monthly_rtr > 0].count() / len(monthly_rtr) * 100
    win_years = yearly_rtr[yearly_rtr > 0].count() / len(yearly_rtr) * 100

    return {
        "top00": {
            "Annualized Daily Return": annualized_daily_rtr * 100,
            "Annualized Daily Volatility": annualized_daily_std * 100,
            "Annualized Daily SR": daily_sharpe,
        },
        "top01": {
            "Annualized Monthly Return": annualized_monthly_rtr * 100,
            "Annualized Monthly Volatility": annualized_monthly_std * 100,
            "Annualized Monthly SR": monthly_sharpe,
        },
        "risk": {
            "Max Drawdown": max_dd,
            "Mean Drawdown": mean_dd,
            "Skew (Daily Return)": daily_skew,
            "Skew (Monthly Return)": monthly_skew,
        },
        "skew_kurt": {
            "Left Tail Ratio (Daily)": tratios_daily["left"],
            "Right Tail Ratio (Daily)": tratios_daily["right"],
            "Left Tail Ratio (Monthly)": tratios_monthly["left"],
            "Right Tail Ratio (Monthly)": tratios_monthly["right"],
        },
        "expected": {
            "Expected Daily Return": expected_daily_rtr,
            "Expected Monthly Return": expected_monthly_rtr,
            "Expected Yearly Return": expected_yearly_rtr,
        },
        "daily_var": {
            "Daily VaR (1%-ile)": daily_1p_var,
            "Daily CVaR (1%-ile)": daily_1p_cvar,
            "Daily VaR (5%-ile)": daily_5p_var,
            "Daily CVaR (5%-ile)": daily_5p_cvar,
        },
        "monthly_cvar": {
            "Monthly VaR (1%-ile)": monthly_1p_var,
            "Monthly CVaR (1%-ile)": monthly_1p_cvar,
            "Monthly VaR (5%-ile)": monthly_5p_var,
            "Monthly CVaR (5%-ile)": monthly_5p_cvar,
        },
        "best": {
            "Best Day Return": best_day,
            "Best Month Return": best_month,
            "Best Year Return": best_year,
        },
        "best": {
            "Worst Day Return": worst_day,
            "Worst Month Return": worst_month,
            "Worst Year Return": worst_year,
        },
        "wd": {
            "Win Days (%)": win_days,
            "Win Month (%)": win_months,
            "Win Year (%)": win_years,
        },
    }


#! https://posit-dev.github.io/great-tables/get-started/
#! https://github.com/astanin/python-tabulate


def _metric_table(main_strat: Strategy, benchmark: Strategy = None):

    main_stats = stats(main_strat)
    if benchmark is not None:
        bench_stats = stats(benchmark)
        combined = {i: [main_stats[i], bench_stats[i]] for i in main_stats}
        fields = ["Metrics", main_strat.name, benchmark.name]
    else:
        combined = {i: [main_stats[i]] for i in main_stats}
        fields = ["Metrics", main_strat.name]

    table = PrettyTable(float_format=".3", align="c")
    table.field_names = fields
    for k, v in combined.items():
        _to_add = {i: [k[i] for k in v] for i in v[0]}
        table.add_rows([i, *list(j)] for i, j in _to_add.items())
        table.add_divider()
    print(table)
