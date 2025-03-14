from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
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

    gross_returns = strat.aggr_return_series + strat.aggr_cost_return_series
    net_returns = strat.aggr_return_series

    gross_d_rtr = gross_returns.resample("D").sum(min_count=1).dropna()
    # gross_d_rtr = _remove_zeroes(gross_d_rtr)

    net_d_rtr = net_returns.resample("D").sum(min_count=1).dropna()
    # net_d_rtr = _remove_zeroes(net_d_rtr)

    gross_m_rtr = gross_returns.resample("ME").sum(min_count=1).dropna()
    # gross_m_rtr = _remove_zeroes(gross_m_rtr)

    net_m_rtr = net_returns.resample("ME").sum(min_count=1).dropna()
    # net_m_rtr = _remove_zeroes(net_m_rtr)

    net_y_rtr = net_returns.resample("YE").sum(min_count=1).dropna()
    # net_y_rtr = _remove_zeroes(net_y_rtr)

    annualized_net_daily_rtr = net_d_rtr.mean() * gscap.DAYS_IN_YEAR
    annualized_gross_daily_std = gross_d_rtr.std() * np.sqrt(gscap.DAYS_IN_YEAR)
    daily_sharpe = annualized_net_daily_rtr / annualized_gross_daily_std

    annualized_net_monthly_rtr = net_m_rtr.mean() * 12
    annualized_gross_monthly_std = gross_m_rtr.std() * np.sqrt(12)
    monthly_sharpe = annualized_net_monthly_rtr / annualized_gross_monthly_std

    dd = drawdown_series(net_d_rtr, cumulative=True)
    max_dd = dd.min() * 100
    mean_dd = dd.mean() * 100
    tratios_daily = tail_ratios(net_d_rtr)
    tratios_monthly = tail_ratios(net_m_rtr)
    daily_skew = net_d_rtr.skew()
    monthly_skew = net_m_rtr.skew()
    expected_daily_rtr = net_d_rtr.mean() * 100
    expected_monthly_rtr = net_m_rtr.mean() * 100
    expected_yearly_rtr = net_y_rtr.mean() * 100
    _1p_d, _5p_d = net_d_rtr.quantile([0.01, 0.05])
    _1p_m, _5p_m = net_m_rtr.quantile([0.01, 0.05])
    daily_1p_var = _1p_d * 100
    daily_5p_var = _5p_d * 100
    monthly_1p_var = _1p_m * 100
    monthly_5p_var = _5p_m * 100
    daily_1p_cvar = net_d_rtr[net_d_rtr <= _1p_d].mean() * 100
    daily_5p_cvar = net_d_rtr[net_d_rtr <= _5p_d].mean() * 100
    monthly_1p_cvar = net_m_rtr[net_m_rtr <= _1p_m].mean() * 100
    monthly_5p_cvar = net_m_rtr[net_m_rtr <= _5p_m].mean() * 100
    best_day = net_d_rtr.max() * 100
    best_month = net_m_rtr.max() * 100
    best_year = net_y_rtr.max() * 100
    worst_day = net_d_rtr.min() * 100
    worst_month = net_m_rtr.min() * 100
    worst_year = net_y_rtr.min() * 100
    win_days = net_d_rtr[net_d_rtr > 0].count() / len(net_d_rtr) * 100
    win_months = net_m_rtr[net_m_rtr > 0].count() / len(net_m_rtr) * 100
    win_years = net_y_rtr[net_y_rtr > 0].count() / len(net_y_rtr) * 100

    return {
        "top00": {
            "Annualized Net Daily Return": annualized_net_daily_rtr * 100,
            "Annualized Gross Daily Volatility": annualized_gross_daily_std * 100,
            "Annualized Daily SR*": daily_sharpe,
        },
        "top01": {
            "Annualized Net Monthly Return": annualized_net_monthly_rtr * 100,
            "Annualized Gross Monthly Volatility": annualized_gross_monthly_std * 100,
            "Annualized Monthly SR*": monthly_sharpe,
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


# ! https://posit-dev.github.io/great-tables/get-started/
# ! https://github.com/astanin/python-tabulate


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
