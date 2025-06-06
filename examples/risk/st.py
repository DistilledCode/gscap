# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# from gscbt import Ticker
# from gscap.risk.helper import get_data, get_synth_slices, get_index_map, get_ratcheted
# from gscap.risk.models import spline_models, get_risk_vol, get_risk_ext, final_risk
# from plotly.subplots import make_subplots
# import numpy as np

# # Title and description
# # Set to light theme
# st.set_page_config(
#     page_title="My App",
#     layout="wide",
#     page_icon="🔍",
#     initial_sidebar_state="expanded",
# )

# # Optional: You can also force the light theme with this
# st._config.set_option("theme.base", "light")
# st.title("Financial Risk Analysis")
# st.write(
#     "This app analyzes risk measures for futures contracts based on a synthetic expression."
# )

# # Hardcoded tickers and expression
# cme = Ticker.TICKERS.cme
# tickers = [cme.cl.f, cme.ho.f, cme.rb.f]
# all_season = "ZMN25-ZMQ25-ZMV25-3*ZMZ25 + 8*ZMF26-4*ZMH26 + ZSN25-ZSQ25-ZSU25 + ZSN26 + ZLN25-4*ZLU25 + 5*ZLZ25-2*ZLH26"
# # exp =

# # User inputs
# start_date = st.date_input("Start date", value=pd.to_datetime("2009-01-01"))
# end_date = st.date_input("End date", value=pd.to_datetime("2024-12-31"))
# exp = st.text_input("Expression", value=all_season)
# lb = st.slider("Lookback period (days)", min_value=5, max_value=100, value=20)

# start = start_date.strftime("%Y-%m-%d")
# end = end_date.strftime("%Y-%m-%d")


# # Cache the data processing
# @st.cache_data
# def process_data(_tickers, exp, start, end, lb):
#     data_nbadj = get_data(exp, _tickers, start, end, badj=False)
#     data_badj = get_data(exp, _tickers, start, end, badj=True)
#     synthetic_sliced = list(get_synth_slices(data_nbadj))
#     st.dataframe(
#         pd.DataFrame(
#             np.vstack(
#                 [
#                     (ind, date[0].date(), date[-1].date(), len(sliced_df))
#                     for ind, (date, sliced_df) in enumerate(
#                         get_synth_slices(data_nbadj), start=1
#                     )
#                 ]
#             ),
#             columns=["slice #", "Start Date", "End Date", "# of days"],
#         )
#     )
#     ywise_hstacked = pd.concat(
#         [
#             sliced_df.set_index(sliced_df.days_to_roll).close[::-1]
#             for _, sliced_df in synthetic_sliced
#         ],
#         axis=1,
#         keys=[date[-1].year for date, _ in synthetic_sliced],
#     )
#     ywise_hstacked = ywise_hstacked[::-1].sort_index(ascending=False)

#     rolling_std = ywise_hstacked.interpolate().rolling(lb, min_periods=lb // 2).std()

#     ywise_vstacked = rolling_std.reset_index().melt(
#         id_vars="days_to_roll",
#         var_name="year",
#         value_name="value",
#     )
#     ywise_vstacked["days_to_roll"] = ywise_vstacked["days_to_roll"].dt.days
#     ywise_vstacked.dropna(inplace=True)
#     ywise_vstacked = ywise_vstacked.sort_values(["days_to_roll", "year"])

#     spline_models_list = spline_models(ywise_vstacked)

#     roll_wise_vstacked_vol = pd.concat(
#         [
#             roll_slice.close.rolling(lb, min_periods=10).std()
#             for _, roll_slice in synthetic_sliced
#         ]
#     )
#     roll_wise_vstacked_vol = roll_wise_vstacked_vol.squeeze()
#     roll_wise_vstacked_vol.name = "roll_wise_vol"

#     risk_vol = get_risk_vol(data_badj)
#     risk_ext = get_risk_ext(data_badj)

#     data_nbadj = pd.concat([data_nbadj, risk_ext, roll_wise_vstacked_vol], axis=1)

#     roll_index_date_map, skip_index = get_index_map(synthetic_sliced, data_nbadj)

#     frisk = final_risk(data_nbadj, roll_index_date_map, skip_index, spline_models_list)
#     frisk_ratcheted = get_ratcheted(frisk, fall_factor=0.0625)

#     return {
#         "synthetic_sliced": synthetic_sliced,
#         "ywise_hstacked": ywise_hstacked,
#         "rolling_std": rolling_std,
#         "risk_vol": risk_vol,
#         "risk_ext": risk_ext,
#         "frisk": frisk,
#         "frisk_ratcheted": frisk_ratcheted,
#     }


# # Process data with a spinner
# with st.spinner("Processing data..."):
#     processed_data = process_data(tickers, exp, start, end, lb)

# # Display synthetic slices info
# # st.write(f"**Synthetic Expression:** {exp}")
# # st.write(f"**Start Date:** {start}; **End Date:** {end}")

# # for ind, (date, sliced_df) in enumerate(processed_data["synthetic_sliced"], start=1):

# #     st.write(
# #         f"{ind:>02}: from {date[0].date()} to {date[-1].date()}, length={len(sliced_df)}"
# #     )

# # Create first figure: Yearly Stacked and Rolling Volatility
# fig1 = make_subplots(
#     rows=2,
#     cols=1,
#     subplot_titles=("Yearly Stacked", f"Rolling Vol of Different Years (LB={lb})"),
#     vertical_spacing=0.08,
#     shared_xaxes=True,
# )

# for col in processed_data["ywise_hstacked"].columns:
#     fig1.add_trace(
#         go.Scatter(
#             y=processed_data["ywise_hstacked"][col].interpolate(),
#             name=str(col),
#             legendgroup=str(col),
#             showlegend=True,
#         ),
#         row=1,
#         col=1,
#     )

# for col in processed_data["ywise_hstacked"].columns:
#     fig1.add_trace(
#         go.Scatter(
#             y=processed_data["rolling_std"][col],
#             name=str(col),
#             legendgroup=str(col),
#             showlegend=False,
#         ),
#         row=2,
#         col=1,
#     )

# fig1.update_layout(
#     title_text=f"Analysis",
#     title_x=0.5,
#     legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
#     height=850,
# )
# fig1.update_xaxes(tickformat="%b %d", tickmode="auto", row=1, col=1)
# fig1.update_xaxes(tickformat="%b %d", tickmode="auto", row=2, col=1)

# # Create second figure: Risk Measures and Extra Risk
# fig2 = make_subplots(
#     rows=2,
#     cols=1,
#     shared_xaxes=True,
#     subplot_titles=("Different Risk Measures", "Extra Risk Added"),
#     vertical_spacing=0.08,
# )

# fig2.add_trace(
#     go.Scatter(
#         x=processed_data["risk_vol"].index,
#         y=processed_data["risk_vol"].values,
#         name="Risk Vol",
#     ),
#     row=1,
#     col=1,
# )
# fig2.add_trace(
#     go.Scatter(
#         x=processed_data["risk_ext"].index,
#         y=processed_data["risk_ext"].values,
#         name="Risk Ext",
#     ),
#     row=1,
#     col=1,
# )
# fig2.add_trace(
#     go.Scatter(
#         x=processed_data["frisk"].index,
#         y=processed_data["frisk"].values,
#         name="Final Risk",
#     ),
#     row=1,
#     col=1,
# )
# fig2.add_trace(
#     go.Scatter(
#         x=processed_data["frisk_ratcheted"].index,
#         y=processed_data["frisk_ratcheted"].values,
#         name="Final Risk Ratcheted",
#     ),
#     row=1,
#     col=1,
# )
# fig2.add_trace(
#     go.Scatter(
#         x=processed_data["frisk"].index,
#         y=processed_data["frisk"] - processed_data["risk_ext"],
#         name="Extra Risk Added",
#     ),
#     row=2,
#     col=1,
# )

# fig2.update_layout(showlegend=True, title_text="Risk Analysis Dashboard", height=850)
# fig2.update_xaxes(title_text="Date", row=2, col=1)
# fig2.update_yaxes(title_text="Risk Value", row=1, col=1)
# fig2.update_yaxes(title_text="Delta", row=2, col=1)

# # Display figures
# st.plotly_chart(fig1, use_container_width=True)
# st.plotly_chart(fig2, use_container_width=True)


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from gscbt import Ticker
from gscap.risk.helper import get_data, get_synth_slices, get_index_map, get_ratcheted
from gscap.risk.models import spline_models, get_risk_vol, get_risk_ext, final_risk
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Financial Risk Analysis",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="expanded",
)

# Optional: You can also force the light theme with this
st._config.set_option("theme.base", "light")

# Main title
st.title("Financial Risk Analysis")
st.write("This app analyzes risk measures for futures contracts based on a synthetic expression.")

# Hardcoded tickers and expression
cme = Ticker.TICKERS.cme
tickers = [cme.cl.f, cme.ho.f, cme.rb.f]
all_season = "ZMN25-ZMQ25-ZMV25-3*ZMZ25 + 8*ZMF26-4*ZMH26 + ZSN25-ZSQ25-ZSU25 + ZSN26 + ZLN25-4*ZLU25 + 5*ZLZ25-2*ZLH26"

# SIDEBAR - Input Panel
with st.sidebar:
    st.header("📊 Input Parameters")
    
    start_date = st.date_input(
        "Start Date", 
        value=pd.to_datetime("2009-01-01"),
        help="Select the start date for analysis"
    )
    
    end_date = st.date_input(
        "End Date", 
        value=pd.to_datetime("2024-12-31"),
        help="Select the end date for analysis"
    )
    
    exp = st.text_area(
        "Expression", 
        value=all_season,
        height=100,
        help="Enter the synthetic expression for analysis"
    )
    
    lb = st.slider(
        "Lookback Period (days)", 
        min_value=5, 
        max_value=100, 
        value=20,
        help="Rolling window size for volatility calculations"
    )
    
    # Add a process button for better UX
    process_button = st.button("🔄 Process Data", type="primary")

# Convert dates to strings
start = start_date.strftime("%Y-%m-%d")
end = end_date.strftime("%Y-%m-%d")

# Cache the data processing
@st.cache_data
def process_data(_tickers, exp, start, end, lb):
    data_nbadj = get_data(exp, _tickers, start, end, badj=False)
    data_badj = get_data(exp, _tickers, start, end, badj=True)
    synthetic_sliced = list(get_synth_slices(data_nbadj))
    
    # Create dataframe for synthetic slices
    slices_df = pd.DataFrame(
        np.vstack(
            [
                (ind, date[0].date(), date[-1].date(), len(sliced_df))
                for ind, (date, sliced_df) in enumerate(
                    get_synth_slices(data_nbadj), start=1
                )
            ]
        ),
        columns=["Slice #", "Start Date", "End Date", "# of Days"],
    )
    
    ywise_hstacked = pd.concat(
        [
            sliced_df.set_index(sliced_df.days_to_roll).close[::-1]
            for _, sliced_df in synthetic_sliced
        ],
        axis=1,
        keys=[date[-1].year for date, _ in synthetic_sliced],
    )
    ywise_hstacked = ywise_hstacked[::-1].sort_index(ascending=False)

    rolling_std = ywise_hstacked.interpolate().rolling(lb, min_periods=lb // 2).std()

    ywise_vstacked = rolling_std.reset_index().melt(
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

    risk_vol = get_risk_vol(data_badj)
    risk_ext = get_risk_ext(data_badj)

    data_nbadj = pd.concat([data_nbadj, risk_ext, roll_wise_vstacked_vol], axis=1)

    roll_index_date_map, skip_index = get_index_map(synthetic_sliced, data_nbadj)

    frisk = final_risk(data_nbadj, roll_index_date_map, skip_index, spline_models_list)
    frisk_ratcheted = get_ratcheted(frisk, fall_factor=0.0625)

    return {
        "slices_df": slices_df,
        "synthetic_sliced": synthetic_sliced,
        "ywise_hstacked": ywise_hstacked,
        "rolling_std": rolling_std,
        "risk_vol": risk_vol,
        "risk_ext": risk_ext,
        "frisk": frisk,
        "frisk_ratcheted": frisk_ratcheted,
    }

# MAIN CONTENT AREA
# Process data automatically or when button is clicked
if process_button or 'processed_data' not in st.session_state:
    with st.spinner("Processing data..."):
        processed_data = process_data(tickers, exp, start, end, lb)
        st.session_state.processed_data = processed_data
else:
    processed_data = st.session_state.processed_data

# Display synthetic slices dataframe in the main area
st.header("📋 Synthetic Slices Summary")
st.dataframe(
    processed_data["slices_df"], 
    use_container_width=True,
    hide_index=True
)

# Add some metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Slices", len(processed_data["slices_df"]))
with col2:
    st.metric("Total Days", processed_data["slices_df"]["# of Days"].sum())
with col3:
    st.metric("Lookback Period", f"{lb} days")
with col4:
    st.metric("Date Range", f"{(end_date - start_date).days} days")

# PLOTS SECTION
st.header("📈 Analysis Charts")

# Create first figure: Yearly Stacked and Rolling Volatility
fig1 = make_subplots(
    rows=2,
    cols=1,
    subplot_titles=("Yearly Stacked", f"Rolling Vol of Different Years (LB={lb})"),
    vertical_spacing=0.08,
    shared_xaxes=True,
)

for col in processed_data["ywise_hstacked"].columns:
    fig1.add_trace(
        go.Scatter(
            y=processed_data["ywise_hstacked"][col].interpolate(),
            name=str(col),
            legendgroup=str(col),
            showlegend=True,
        ),
        row=1,
        col=1,
    )

for col in processed_data["ywise_hstacked"].columns:
    fig1.add_trace(
        go.Scatter(
            y=processed_data["rolling_std"][col],
            name=str(col),
            legendgroup=str(col),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

fig1.update_layout(
    title_text="Yearly Analysis",
    title_x=0.5,
    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.01),
    height=850,
)
fig1.update_xaxes(tickformat="%b %d", tickmode="auto", row=1, col=1)
fig1.update_xaxes(tickformat="%b %d", tickmode="auto", row=2, col=1)

# Create second figure: Risk Measures and Extra Risk
fig2 = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("Different Risk Measures", "Extra Risk Added"),
    vertical_spacing=0.08,
)

fig2.add_trace(
    go.Scatter(
        x=processed_data["risk_vol"].index,
        y=processed_data["risk_vol"].values,
        name="Risk Vol",
    ),
    row=1,
    col=1,
)
fig2.add_trace(
    go.Scatter(
        x=processed_data["risk_ext"].index,
        y=processed_data["risk_ext"].values,
        name="Risk Ext",
    ),
    row=1,
    col=1,
)
fig2.add_trace(
    go.Scatter(
        x=processed_data["frisk"].index,
        y=processed_data["frisk"].values,
        name="Final Risk",
    ),
    row=1,
    col=1,
)
fig2.add_trace(
    go.Scatter(
        x=processed_data["frisk_ratcheted"].index,
        y=processed_data["frisk_ratcheted"].values,
        name="Final Risk Ratcheted",
    ),
    row=1,
    col=1,
)
fig2.add_trace(
    go.Scatter(
        x=processed_data["frisk"].index,
        y=processed_data["frisk"] - processed_data["risk_ext"],
        name="Extra Risk Added",
    ),
    row=2,
    col=1,
)

fig2.update_layout(showlegend=True, title_text="Risk Analysis Dashboard", height=850)
fig2.update_xaxes(title_text="Date", row=2, col=1)
fig2.update_yaxes(title_text="Risk Value", row=1, col=1)
fig2.update_yaxes(title_text="Delta", row=2, col=1)

# Display figures
st.plotly_chart(fig1, use_container_width=True)
st.plotly_chart(fig2, use_container_width=True)