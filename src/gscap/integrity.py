import calplot
import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
import plotly.graph_objects as go


def _plot_msno(df, ptype):
    msno.heatmap(df[ptype], figsize=(8, 6), fontsize=12)
    plt.title("Correlation between missing data points\n", pad=20)
    plt.show()

    msno.matrix(df[ptype], figsize=(8, 6), fontsize=12)
    plt.title("Missing data in Chronological Order", pad=20)
    plt.show()


def _plot_calplot(df, ptype):

    calplot.calplot(
        df[ptype].isna().sum(axis=1).astype(bool).astype(int),
        yearlabel_kws={"fontsize": 30, "fontname": "Arial"},
        colorbar=False,
        suptitle="Dates with atleast one missing data",
    )
    plt.show()


def _plot_plotly(df, ptype):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df[ptype].isna().sum(axis=1).index,
            y=df[ptype].isna().sum(axis=1),
            name="combined",
            line=dict(color="black", width=1),
        )
    )

    for contract in df[ptype]:
        fig.add_trace(
            go.Scatter(
                x=df[ptype][contract].isna().index,
                y=df[ptype][contract].isna().astype(int),
                name=contract,
            )
        )

    fig.update_layout(
        title="Count of Missing Data",
        yaxis_title="Count",
        height=600,
        width=1400,
    )
    fig.update_xaxes(rangeslider_visible=True)

    fig.show()


def missing_data(df: pd.DataFrame, ptype: str):
    _n = len(df)
    _min_date = df[ptype].index.min().date()
    _max_date = df[ptype].index.max().date()
    print(f"Data  Range: {_min_date} to {_max_date}\nData Points: {_n}\n")
    print("Missing Data Count".center(26, "="))
    for k, v in df[ptype].isna().sum().to_dict().items():
        print(f"{k:<7} {v:>5} \t({(v/_n*100):05.2f}%)")

    _plot_msno(df, ptype)
    _plot_calplot(df, ptype)
    _plot_plotly(df, ptype)
