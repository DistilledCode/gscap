from typing import Literal

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import quantstats._plotting.core as qsplot

import gscap.metrics as metrics
from gscap import SRC_DIR

LINE_WIDTH = 1.25


def set_style():
    font_path = SRC_DIR / "../FiraCode-Medium.ttf"
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.serif"] = ["Fira Code"]
    plt.rcParams["font.family"] = ["Fira Code"]
    plt.rcParams["font.fantasy"] = ["Fira Code"]
    plt.rcParams["font.monospace"] = ["Fira Code"]
    plt.rcParams["font.sans-serif"] = ["Fira Code"]


set_style()


def drawdown(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    compound=False,
    cumulative=False,
    fill=True,
):
    if cumulative ^ compound is False:
        raise ValueError("Either of `compound` or `cumulative` must be True, not both")
    _ds = metrics.drawdown_series(return_series, compound, cumulative)
    if benchmark is not None:
        _bds = metrics.drawdown_series(benchmark, compound, cumulative)

    _type = "Compounding" if compound else "Cumulative"

    return qsplot.plot_timeseries(
        _ds,
        None if benchmark is None else _bds,
        cumulative=False,
        compound=False,
        fill=fill,
        hline=_ds.mean(),
        hlw=1.5,
        lw=LINE_WIDTH,
        title=f"Drawdown Graph ({_type})",
        fontname="Fira Code",
        ylabel="Drawdown",
        hllabel="Average",
        show=False,
    )


def position(position_series: pd.Series, benchmark: pd.Series = None, fill=False):
    position_series = position_series.dropna()
    if benchmark is not False:
        benchmark = benchmark.dropna()

    return qsplot.plot_timeseries(
        position_series,
        benchmark,
        cumulative=False,
        compound=False,
        fill=fill,
        # hline=_ds.mean(),
        # hlw=1.5,
        lw=LINE_WIDTH,
        title="Position Size",
        fontname="Fira Code",
        ylabel="Position Size",
        show=False,
        percent=False,
    )


def turnover(
    position_series: pd.Series,
    benchmark: pd.Series = None,
    fill=False,
    period=22 * 6,
):
    _ts = metrics.turnover_series(position_series)
    _ts = _ts.rolling(period, min_periods=period).mean()
    if benchmark is not None:
        _bts = metrics.turnover_series(benchmark)
        _bts = _bts.rolling(period, min_periods=period).mean()

    _ts.dropna(inplace=True)
    _bts.dropna(inplace=True)
    return qsplot.plot_timeseries(
        _ts,
        _bts,
        cumulative=False,
        compound=False,
        fill=fill,
        hline=_ts.mean(),
        hllabel="Average Turnover",
        hlw=1.5,
        lw=LINE_WIDTH,
        title=f"Turnover; rolling {period=}",
        fontname="Fira Code",
        ylabel="Annualized Turnover",
        show=False,
        percent=False,
    )


def return_histogram(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    resample="ME",
    bins=30,
):

    title = f"Returns Histogram; {resample=}"
    return qsplot.plot_histogram(
        return_series,
        benchmark=benchmark,
        fontname="Fira Code",
        compounded=False,
        bins=bins,
        resample=resample,
        title=title,
    )


def returns(
    returns: pd.Series,
    benchmark: pd.Series = None,
    compound=False,
    cumulative=False,
):
    if cumulative and compound is True:
        raise ValueError("Both `compound` and `cumulative` cannot be True")

    _title = "Returns"

    if compound:
        _title = "Returns (Compounding)"
        returns = returns.add(1).cumprod() - 1
        if benchmark is not None:
            benchmark = benchmark.add(1).cumprod() - 1
    if cumulative:
        _title = "Returns (Cumulative)"
        returns = returns.cumsum()
        if benchmark is not None:
            benchmark = benchmark.cumsum()

    return qsplot.plot_timeseries(
        returns,
        benchmark,
        cumulative=False,
        compound=False,
        fill=False,
        lw=LINE_WIDTH,
        title=_title,
        fontname="Fira Code",
        ylabel=_title,
        show=False,
    )


def rolling_volatility(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    periods=14,
    weights: Literal["equi", "ewma"] = "equi",
    annualize=True,
    periods_per_year=252,
):

    if weights == "ewma":
        halflife = periods // 2
        title = f"Rolling Volatility ({weights=}; {halflife=})"
        _vol = metrics.ewma_vol(
            return_series,
            halflife=halflife,
            annualize=annualize,
            periods_per_year=periods_per_year,
        )
        if benchmark is not None:
            _bvol = metrics.ewma_vol(
                benchmark,
                halflife=halflife,
                annualize=annualize,
                periods_per_year=periods_per_year,
            )
    elif weights == "equi":
        title = f"Rolling Volatility ({weights=}; {periods=})"
        _vol = metrics.equi_vol(
            return_series,
            rolling_window=periods,
            annualize=annualize,
            periods_per_year=periods_per_year,
        )
        if benchmark is not None:
            _bvol = metrics.equi_vol(
                benchmark,
                rolling_window=periods,
                annualize=annualize,
                periods_per_year=periods_per_year,
            )
    else:
        raise ValueError("Wrong weighing scheme dedi!! Either `equi` or `ewma`")
    ylabel = "Non-" if not annualize else ""
    ylabel += "Annualized Volatility"
    return qsplot.plot_rolling_stats(
        _vol,
        None if benchmark is None else _bvol,
        hline=_vol.mean(),
        hlw=1.5,
        title=title,
        ylabel=ylabel,
        fontname="Fira Code",
        hllabel="Average Volatility",
        lw=LINE_WIDTH,
        figsize=(10, 6),
        subtitle=True,
        show=False,
    )


def rolling_sharpe(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    periods=22 * 6,
    weights: Literal["equi", "ewma"] = "equi",
    annualize=True,
    periods_per_year=252,
):
    if weights == "equi":
        title = f"Rolling Sharpe ({weights=}; {periods=})"
        _rs = metrics.rolling_sharpe(
            return_series,
            periods=periods,
            annualize=annualize,
            periods_per_year=periods_per_year,
        )
        if benchmark is not None:
            _brs = metrics.rolling_sharpe(
                benchmark,
                periods=periods,
                annualize=annualize,
                periods_per_year=periods_per_year,
            )

    elif weights == "ewma":
        _hl = periods // 2
        title = f"Rolling Sharpe ({weights=}; halflife={_hl})"
        _rs = metrics.rolling_sharpe(
            return_series,
            weights="ewma",
            periods=periods,
            annualize=annualize,
            periods_per_year=periods_per_year,
        )
        if benchmark is not None:
            _brs = metrics.rolling_sharpe(
                return_series,
                weights="ewma",
                periods=periods,
                annualize=annualize,
                periods_per_year=periods_per_year,
            )

    ylabel = "Non-" if not annualize else ""
    ylabel += "Annualized Sharpe"
    return qsplot.plot_rolling_stats(
        _rs,
        None if benchmark is None else _brs,
        hline=_rs.mean(),
        hlw=1.5,
        title=title,
        ylabel=ylabel,
        fontname="Fira Code",
        hllabel="Average Sharpe",
        lw=LINE_WIDTH,
        figsize=(10, 6),
        subtitle=True,
        show=False,
    )


def volatility(df, rolling_window=14, annualize=True):

    fig = go.Figure()

    for contract in df:
        fig.add_trace(
            go.Scatter(
                x=df[contract].index,
                y=metrics.equi_vol(
                    df[contract],
                    rolling_window=rolling_window,
                    annualize=annualize,
                ),
                name=contract,
            )
        )

    if annualize:
        _title = f"Annualized Rolling Volatility ({rolling_window} days)"
    else:
        _title = f"Daily Rolling Volatility ({rolling_window} days)"
    fig.update_layout(
        title=_title,
        yaxis_title="Volatility",
        # height=600,
        # width=1400,
        legend=dict(
            title="Instruments",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            orientation="v",
        ),
    )
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()


def vol_of_vol(df, rolling_window=14, annualize=True):

    _vol = metrics.equi_vol(
        df,
        rolling_window=rolling_window,
        annualize=annualize,
    )

    fig = go.Figure()

    for contract in _vol.columns:
        fig.add_trace(
            go.Scatter(
                x=_vol[contract].index,
                y=metrics.equi_vol(
                    _vol[contract],
                    rolling_window=rolling_window,
                    annualize=annualize,
                ),
                name=contract,
            )
        )

    if annualize:
        _title = f"Volatility (Annualized {rolling_window} days rolling)"
    else:
        _title = f"Volatility (Daily {rolling_window} days rolling)"
    _title = "Volatility of " + _title
    fig.update_layout(
        title=_title,
        yaxis_title="Volatility of Volatility",
        # height=600,
        # width=1400,
        legend=dict(
            title="Instruments",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            orientation="v",
        ),
    )
    fig.update_xaxes(rangeslider_visible=True)
    fig.show()
