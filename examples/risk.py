import pandas as pd
import plotly.graph_objects as go
from gscbt import Ticker
from gscap.risk.helper import get_data, get_synth_slices, get_index_map, get_ratcheted
from gscap.risk.models import spline_models, get_risk_vol, get_risk_ext, final_risk

from plotly.subplots import make_subplots
import plotly.graph_objects as go

cme = Ticker.TICKERS.cme
iceec = Ticker.TICKERS.iceec
ice = Ticker.TICKERS.ice
icefu = Ticker.TICKERS.icefu


tickers = [cme.zc.f]
tickers = [cme.cl.f, cme.ho.f, cme.rb.f]
exp = "ZCK25-2*ZCN25+ZCU25"
exp = "-2*CLN25 + 2*CLQ25 + RBN25-RBQ25 + HON25-HOQ25"
start = "2009-01-01"
end = "2024-12-31"
data_nbadj = get_data(exp, tickers, start, end, badj=False)
data_badj = get_data(exp, tickers, start, end, badj=True)


synthetic_sliced = list(get_synth_slices(data_nbadj))

print(f"\nSynthethic: {exp}\n\nStart Date={start}; End Date={end}\n")
for ind, (date, sliced_df) in enumerate(synthetic_sliced, start=1):
    print(f"{ind:>02}: from {date[0].date()} to {date[-1].date()}", len(sliced_df))

ywise_hstacked = pd.concat(
    [
        sliced_df.set_index(sliced_df.days_to_roll).close[::-1]
        for _, sliced_df in synthetic_sliced
    ],
    axis=1,
    keys=[date[-1].year for date, _ in synthetic_sliced],
)
ywise_hstacked = ywise_hstacked[::-1].sort_index(ascending=False)


lb = 20

fig = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("Yearly Stacked", f"Rolling Vol of Different Years (LB={lb})"),
    vertical_spacing=0.08,
    shared_xaxes=True,
)


for col in ywise_hstacked.columns:
    fig.add_trace(
        go.Scatter(
            y=ywise_hstacked[col].interpolate(),
            name=col,
            legendgroup=col,
            showlegend=True,
        ),
        row=1,
        col=1,
    )


for col in ywise_hstacked.columns:
    fig.add_trace(
        go.Scatter(
            y=ywise_hstacked[col].interpolate().rolling(lb, min_periods=lb // 2).std(),
            name=col,
            legendgroup=col,
            showlegend=False,
        ),
        row=2,
        col=1,
    )


fig.update_layout(
    title_text=f"Combined Analysis: {exp}",
    title_x=0.5,
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
)


fig.update_xaxes(tickformat="%b %d", tickmode="auto", row=1, col=1)
fig.update_xaxes(tickformat="%b %d", tickmode="auto", row=2, col=1)
fig.show()


risk_vol = get_risk_vol(data_badj)
risk_ext = get_risk_ext(data_badj)

rolling_std = ywise_hstacked.interpolate().rolling(lb, min_periods=lb // 2).std()

ywise_vstacked = rolling_std.reset_index()
ywise_vstacked = ywise_vstacked.melt(
    id_vars="days_to_roll",
    var_name="year",
    value_name="value",
)


ywise_vstacked["days_to_roll"] = ywise_vstacked["days_to_roll"].dt.days
ywise_vstacked.dropna(inplace=True)
ywise_vstacked = ywise_vstacked.sort_values(["days_to_roll", "year"])

spline_models_list = spline_models(ywise_vstacked)

roll_wise_vstacked_vol = pd.concat(
    [
        roll_slice.close.rolling(lb, min_periods=10).std()
        for _, roll_slice in synthetic_sliced
    ]
)
roll_wise_vstacked_vol = roll_wise_vstacked_vol.squeeze()
roll_wise_vstacked_vol.name = "roll_wise_vol"

data_nbadj = pd.concat([data_nbadj, risk_ext, roll_wise_vstacked_vol], axis=1)

roll_index_date_map, skip_index = get_index_map(synthetic_sliced, data_nbadj)

frisk = final_risk(data_nbadj, roll_index_date_map, skip_index, spline_models_list)
frisk_ratcheted = get_ratcheted(frisk, fall_factor=0.0625)
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("Different Risk Measures", "Extra Risk Added"),
    vertical_spacing=0.08,
)

# First subplot - Risk components
fig.add_trace(
    go.Scatter(x=risk_vol.index, y=risk_vol.values, name="Risk Vol"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=risk_ext.index, y=risk_ext.values, name="Risk Ext"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(x=frisk.index, y=frisk.values, name="Final Risk"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=frisk_ratcheted.index,
        y=frisk_ratcheted.values,
        name="Final Risk ratcheted",
    ),
    row=1,
    col=1,
)

# Second subplot - Delta offsets
fig.add_trace(
    go.Scatter(x=frisk.index, y=frisk - risk_ext, name="Extra Riks Added"),
    row=2,
    col=1,
)

# Update layout
fig.update_layout(showlegend=True, title_text="Risk Analysis Dashboard")
fig.update_xaxes(title_text="Date", row=2, col=1)

fig.update_yaxes(title_text="Risk Value", row=1, col=1)
fig.update_yaxes(title_text="Delta", row=2, col=1)

fig.show()
