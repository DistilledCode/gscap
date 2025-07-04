import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import quantstats._plotting.core as qsplot
from matplotlib.ticker import FuncFormatter
from quantstats._plotting.core import format_pct_axis_dollar
from quantstats._plotting.wrappers import monthly_heatmap as _monthly_heatmap

import gscap.metrics as metrics
from gscap import SRC_DIR

LINE_WIDTH = 1.25
DEFAULT_FIGSIZE = (10, 6)


def set_style():
    font_path = SRC_DIR / "../FiraCode-Medium.ttf"
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.serif"] = ["Fira Code"]
    plt.rcParams["font.family"] = ["Fira Code"]
    plt.rcParams["font.fantasy"] = ["Fira Code"]
    plt.rcParams["font.monospace"] = ["Fira Code"]
    plt.rcParams["font.sans-serif"] = ["Fira Code"]


set_style()


def monthly_heatmap(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    title="Strategy",
    figsize=(10, 7),
    show=True,
):
    _ftitle = title if return_series.name is None else return_series.name
    if benchmark is None:
        active = False
        title = _ftitle
    else:
        active = True
        title = f"{_ftitle} - {benchmark.name}"
    return _monthly_heatmap(
        return_series,
        benchmark=benchmark,
        figsize=figsize,
        annot_size=11,
        returns_label=title,
        compounded=False,
        fontname="Fira Code",
        show=show,
        active=active,
    )


def drawdown(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    compound=False,
    cumulative=True,
    fill=True,
    show=False,
):
    return_series = return_series.resample("D").sum(min_count=1).dropna()
    if benchmark is not None:
        benchmark = benchmark.resample("D").sum(min_count=1).dropna()

    if cumulative ^ compound is False:
        raise ValueError("Either of `compound` or `cumulative` must be True, not both")
    _ds = metrics.drawdown_series(return_series, compound, cumulative)
    # Smoothing for visuals
    _ds = _ds.ewm(span=22).mean()
    if benchmark is not None:
        _bds = metrics.drawdown_series(benchmark, compound, cumulative)
        # Smoothing for visuals
        _bds = _bds.ewm(span=22).mean()
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
        title=f"Drawdown Graph [{_type}; Smooth Curve]",
        fontname="Fira Code",
        ylabel="Drawdown",
        hllabel="Average",
        figsize=(10, 6),
        show=show,
    )


def position(
    position_series: pd.Series,
    benchmark: pd.Series = None,
    fill=False,
    figsize=None,
):
    position_series = position_series.dropna()
    if benchmark is not None:
        benchmark = benchmark.dropna()

    return qsplot.plot_timeseries(
        position_series,
        benchmark,
        cumulative=False,
        compound=False,
        fill=fill,
        lw=LINE_WIDTH,
        title="Position Size",
        fontname="Fira Code",
        ylabel="Position Size",
        show=False,
        figsize=figsize or (10, 6),
        percent=False,
    )


def turnover(
    position_series: pd.Series,
    benchmark: pd.Series = None,
    figsize=(10, 6),
    show=False,
    hrs_in_day: float = 24.0,
):
    _ts = metrics.turnover_series(position_series, hrs_in_day)
    _ts = _ts.resample("D").last().dropna() / 100
    _tsr = _ts.rolling(22 * 6, min_periods=22 * 3).mean().dropna()
    # Smoothing for visuals
    _tsr = _tsr.rolling(11, min_periods=11).mean().dropna()
    _tsr.name = position_series.name
    if benchmark is not None:
        _bts = metrics.turnover_series(benchmark, hrs_in_day)
        _bts = _bts.resample("D").last().dropna() / 100
        _btsr = _bts.rolling(22 * 6, min_periods=22 * 3).mean().dropna()
        # Smoothing for visuals
        _btsr = _btsr.rolling(11, min_periods=11).mean().dropna()
        _btsr.name = benchmark.name
    else:
        _btsr = None
    return qsplot.plot_timeseries(
        _tsr,
        _btsr,
        cumulative=False,
        title="Annualized Daily Turnover [rolling 6 month; Smooth Curve]",
        # fill=True,
        hline=_ts.mean(),
        hlw=1.5,
        lw=LINE_WIDTH,
        fontname="Fira Code",
        hllabel=f"Avg: {(_ts.mean() * 100):.2f}% (nr)",  # nr - non rolling
        ylabel="Annualized Turnover",
        figsize=figsize,
        show=show,
    )


def rolling_long_trades(
    position_series: pd.Series,
    benchmark: pd.Series = None,
    figsize=(10, 6),
    show=False,
    name_tuple=("Blue", "Benchmark"),
):
    intv = position_series.interval()
    anf = (252 * 24 * 3600) / intv

    def get_rolling_tl(_df: pd.DataFrame):
        _window = int(anf / 2)
        _minp = int(anf / 4)
        _tl = pd.Series(
            [
                metrics.long_trades(_position)
                for _position in _df.rolling(_window, min_periods=_minp)
            ],
            index=_df.index,
        )
        _tl = _tl.resample("D").last().dropna()
        return _tl, _tl.rolling(22, min_periods=22).mean().dropna()

    _tl, _tlr = get_rolling_tl(position_series)
    _tlr.name = name_tuple[0]
    if benchmark is not None:
        _, _btlr = get_rolling_tl(benchmark)
        _btlr.name = name_tuple[-1]
    else:
        _btlr = None
    return qsplot.plot_timeseries(
        _tlr,
        _btlr,
        cumulative=False,
        title="Long Trades [rolling 6 month; Smooth Curve]",
        hline=_tl.mean(),
        hlw=1.5,
        lw=LINE_WIDTH,
        fontname="Fira Code",
        hllabel="Average",
        ylabel="Percentage of Long Trades",
        figsize=figsize,
        show=show,
    )


def return_histogram(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    resample="ME",
    bins=30,
    figsize=(10, 6),
    show=False,
    granular_returns=False,
    title="Returns Histogram",
):

    title = f"{title}; {resample=}"
    return qsplot.plot_histogram(
        return_series,
        benchmark=benchmark,
        fontname="Fira Code",
        compounded=False,
        bins=bins,
        resample=resample,
        title=title,
        figsize=figsize,
        show=show,
        granular_returns=granular_returns,
    )


def eoy_returns(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    figsize=(12, 6),
    show=False,
):
    if benchmark is not None:
        cmb_rtrs = pd.concat([return_series, benchmark], axis=1)
        color = ["#3698BF", "#FFB917"]
    else:
        cmb_rtrs = return_series
        color = "#3698BF"
    cmb_yrly_rtrs = cmb_rtrs.resample("YE").sum()
    x = cmb_yrly_rtrs.plot.bar(
        color=color,
        width=0.75,
        figsize=figsize,
        alpha=0.75,
    )
    xticks = [
        year if index % 3 == 0 else ""
        for index, year in enumerate(cmb_yrly_rtrs.index.year)
    ]
    x.set_xticklabels(xticks)
    plt.xticks(rotation=45, ha="right")  # 'ha' is horizontal alignment

    def percentage_formatter(y, pos):
        return f"{y * 100:.00f}%"  # Format as percentage with no decimal places

    x.margins(x=0.10)  # Adds 5% padding on the x-axis
    x.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    x.set_title(
        f"End of Year Cumulative Returns: "
        f"{cmb_yrly_rtrs.index.year[0]} - {cmb_yrly_rtrs.index.year[-1]}"
    )
    if show:
        plt.show()
    else:
        return x


def returns(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    compound=False,
    cumulative=False,
    show=False,
    title="Returns",
    starting_capital=None,
    figsize=(10, 6),
):
    return_series = return_series.resample("D").sum(min_count=1).dropna()
    if benchmark is not None:
        benchmark = benchmark.resample("D").sum(min_count=1).dropna()
    if cumulative and compound is True:
        raise ValueError("Both `compound` and `cumulative` cannot be True")

    _title = title

    if compound:
        _title = f"{title} (Compounding)"
        return_series = return_series.add(1).cumprod() - 1
        if benchmark is not None:
            benchmark = benchmark.add(1).cumprod() - 1
    if cumulative:
        _title = f"{title} (Cumulative)"
        return_series = return_series.cumsum()
        if benchmark is not None:
            benchmark = benchmark.cumsum()

    if starting_capital is not None:
        percent = False
        return_series *= starting_capital
        if benchmark is not None:
            benchmark *= starting_capital
        sc = format_pct_axis_dollar(starting_capital, "")
        _title = f"{_title[:-1]} Returns); Starting Capital: {sc}"
        ylabel = "Value"
    else:
        ylabel = title
        percent = True
    return qsplot.plot_timeseries(
        return_series,
        benchmark,
        cumulative=False,
        compound=False,
        fill=False,
        lw=1.5,
        title=_title,
        fontname="Fira Code",
        ylabel=ylabel,
        figsize=figsize or (10, 6),
        show=show,
        percent=percent,
    )


def rolling_volatility(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    span=22 * 3,
    annualize=True,
    show=False,
):
    return_series = return_series.resample("D").sum(min_count=1).dropna()
    _vol = metrics.ewma_vol(return_series, span=span, annualize=annualize)
    # Smoothing for visuals
    _vol = _vol.ewm(span=22).mean()

    if benchmark is not None:
        benchmark = benchmark.resample("D").sum(min_count=1).dropna()
        _bvol = metrics.ewma_vol(benchmark, span=span, annualize=annualize)
        # Smoothing for visuals
        _bvol = _bvol.ewm(span=22).mean()

    else:
        _bvol = None

    title = f"Rolling Volatility [EWMA; lookback={span}; Smooth Curve]"
    ylabel = "Non-" if not annualize else ""
    ylabel += "Annualized Volatility"

    return qsplot.plot_rolling_stats(
        _vol,
        _bvol,
        hline=_vol.mean(),
        hlw=1.5,
        title=title,
        ylabel=ylabel,
        fontname="Fira Code",
        hllabel="Average Volatility",
        lw=LINE_WIDTH,
        figsize=(10, 6),
        subtitle=True,
        show=show,
    )


def rolling_sharpe(
    return_series: pd.Series,
    benchmark: pd.Series = None,
    span=22 * 3,
    annualize=True,
    show=False,
):
    return_series = return_series.resample("D").sum(min_count=1).dropna()
    if benchmark is not None:
        benchmark = benchmark.resample("D").sum(min_count=1).dropna()
        _brs = metrics.rolling_sharpe(benchmark, span=span, annualize=annualize)
        # Smoothing for visuals
        _brs = _brs.ewm(span=22).mean()
    else:
        _brs = None

    _rs = metrics.rolling_sharpe(return_series, span=span, annualize=annualize)
    # Smoothing for visuals
    _rs = _rs.ewm(span=22).mean()

    ylabel = "Non-" if not annualize else ""
    ylabel += "Annualized Sharpe"
    title = f"Rolling Sharpe [EWMA; lookback={span}; Smooth Curve]"

    return qsplot.plot_rolling_stats(
        _rs,
        _brs,
        # hline=_rs.mean(),
        hlw=1.5,
        title=title,
        ylabel=ylabel,
        fontname="Fira Code",
        hllabel="Average Sharpe",
        lw=LINE_WIDTH,
        figsize=(10, 6),
        subtitle=True,
        show=show,
    )
