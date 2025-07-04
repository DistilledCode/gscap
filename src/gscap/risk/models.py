"""
Risk modeling utilities for financial data analysis.

This module provides functions for calculating risk measures, volatility estimates,
and spline-based risk models for financial time series data. The primary focus is
on generating robust risk estimates that incorporate both historical data patterns
and forward-looking adjustments.

Key Components:
- Extreme value calculations using normal distribution assumptions
- Multi-timeframe volatility risk calculations
- Spline-based modeling for roll-dependent risk adjustments
- Final risk calculations combining multiple methodologies
"""

import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
import scipy.stats as stats


def get_ext(series: pd.Series, tail_pct: float = 0.975) -> float:
    """
    Calculate extreme value estimate using normal distribution assumption.

    This function estimates the extreme value (97.5th percentile) of a series
    by fitting a normal distribution to the data and calculating the 2.5% tail
    value in absolute terms.

    Args:
        series (pd.Series): Input time series data for which to calculate
                           the extreme value estimate.

    Returns:
        float: Absolute value of the 97.5th percentile estimate based on
               normal distribution fit (mean and standard deviation).

    Note:
        This approach assumes the input series follows a normal distribution.
        The commented alternative approach uses empirical quantiles directly.
    """
    loc = series.mean()
    scale = series.std()

    #! If we get all same values, then scale will be 0 and norm.ppf will return NaN
    #! Highest chance of this happening is with smallest rolling window!
    #! But according to the model downstream flow, if the current extreme value is
    #! too low then other longer window rolling risk will take over. So it should be
    #! safe to forward fill any NaN

    return np.abs(stats.norm.ppf(tail_pct, loc=loc, scale=scale))


def get_risk_vol(data: pd.DataFrame) -> pd.Series:
    """
    Calculate composite risk volatility using multiple timeframes.

    This function combines volatility estimates from different time horizons
    to create a robust risk volatility measure. It uses a weighted combination
    of expanding, rolling, and exponentially weighted moving averages.

    The weighting scheme prioritizes shorter-term volatility (10-day EWM) while
    incorporating longer-term structural volatility patterns.

    Args:
        data (pd.DataFrame): DataFrame containing price data with a 'close' column.
                            Must have sufficient data points for calculations
                            (minimum 20 observations recommended).

    Returns:
        pd.Series: Composite risk volatility series with name 'risk_vol'.
                   Values represent the volatility estimate for each time point.

    Weighting Formula:
        - Expanding window: 10%
        - 2-year rolling: 25%
        - 6-month rolling: 25%
        - 10-day exponential: 40%

    Note:
        The final result takes the maximum between the 10-day EWM and the
        weighted composite to ensure minimum volatility floor.
    """
    expnd = data.close.diff().expanding(min_periods=20).std()
    two_yr = data.close.diff().rolling(252 * 2, min_periods=20).std()
    six_m = data.close.diff().rolling(22 * 6, min_periods=20).std()
    ten_d = data.close.diff().rolling(10).std()

    ult = expnd * 0.10 + two_yr * 0.25 + six_m * 0.25 + ten_d * 0.40
    ult = pd.concat([ten_d, ult], axis=1).max(axis=1).squeeze()
    ult.name = "risk_vol"
    return ult


from functools import partial


def get_risk_ext(data: pd.DataFrame, tail_pct: float = 0.975) -> pd.Series:
    """
    Calculate composite risk extreme values using multiple timeframes.

    Similar to get_risk_vol, but calculates extreme value estimates instead
    of standard deviations. This function applies the get_ext function across
    different time windows and combines them using the same weighting scheme.

    Args:
        data (pd.DataFrame): DataFrame containing price data with a 'close' column.
                            Must have sufficient data points for calculations
                            (minimum 20 observations recommended).

    Returns:
        pd.Series: Composite risk extreme value series with name 'risk_ext'.
                   Values represent extreme value estimates for each time point.

    Weighting Formula:
        - Expanding window: 10%
        - 2-year rolling: 25%
        - 6-month rolling: 25%
        - 10-day rolling: 40%

    Note:
        The final result takes the maximum between the 10-day estimate and the
        weighted composite to ensure a minimum extreme value floor.
        Uses rolling (not EWM) for the 10-day window unlike get_risk_vol.
    """
    extreme_val = partial(get_ext, tail_pct=tail_pct)
    expnd = data.close.diff().expanding(min_periods=20).apply(extreme_val)
    two_yr = data.close.diff().rolling(252 * 2, min_periods=20).apply(extreme_val)
    six_m = data.close.diff().rolling(22 * 6, min_periods=20).apply(extreme_val)
    ten_d = data.close.diff().rolling(10).apply(extreme_val)

    ten_d = ten_d.ffill()

    ult = expnd * 0.10 + two_yr * 0.25 + six_m * 0.25 + ten_d * 0.40
    ult = pd.concat([ten_d, ult], axis=1).max(axis=1).squeeze()
    ult.name = "risk_ext"
    return ult


def spline_models_v2(df: pd.DataFrame, lb: int, qtile: float) -> list[UnivariateSpline]:
    """
    Create spline models for risk estimation using quantile-based approach (Version 2).

    This function builds a series of univariate spline models, each trained on
    progressively more historical data. Each spline models the relationship between
    days-to-roll and volatility quantiles, allowing for forward-looking risk estimates.

    Args:
        df (pd.DataFrame): DataFrame with time series data organized by year columns.
                          Expected to have datetime index and year-based columns.
        lb (int): Lookback period for rolling window calculations.
                 Must be positive integer.
        qtile (float): Quantile level to use for volatility estimation.
                      Should be between 0 and 1 (e.g., 0.95 for 95th percentile).

    Returns:
        list[UnivariateSpline]: List of fitted spline models, one for each year
                               in the dataset. Each spline maps days-to-roll to
                               expected volatility quantile values.

    Algorithm:
        1. For each year, include data up to that year
        2. Calculate rolling standard deviations with specified lookback
        3. Compute quantiles across the rolling windows
        4. Fit univariate spline with degree 5 and adaptive smoothing
        5. Smoothing parameter scales with data variance and length

    Note:
        The function includes commented plotting code for visualization.
        Splines use degree 5 (quintic) with variance-adjusted smoothing.
    """
    spline_models = []
    for yr in range(df.shape[-1]):
        kk = df.iloc[:, : (yr + 1)]
        qtile_df = kk.interpolate().rolling(lb, min_periods=lb // 2)
        qtile_df = qtile_df.std().quantile(qtile, axis=1).dropna().iloc[::-1]
        x = qtile_df.index.days
        y = qtile_df.values
        spline = UnivariateSpline(x, y, k=5, s=len(x) * y.var() ** 2)
        spline_models.append(spline)
        # x_smooth = np.linspace(0, 365, 500)
        # y_smooth = spline(x_smooth)

        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        # ax[0].scatter(x, y, alpha=0.1, label="Raw data")
        # ax[0].plot(x_smooth, y_smooth, color="red", label="Spline")
        # ax[0].set_xlabel("Days to Roll")
        # ax[0].set_ylabel("Quantile Value")
        # ax[0].set_title(
        #     f"Spline Fitted to {quantile} quantile Yearly stacked upto {kk.columns[-1]}"
        # )
        # ax[0].legend()
        # ax[0].invert_xaxis()  # optional, if lower days are later

        # ax[1].plot(x_smooth, y_smooth, color="red", label="Spline")
        # ax[1].set_xlabel("Days to Roll")
        # ax[1].set_ylabel("Quantile Value")
        # ax[1].set_title(
        #     f"Spline Fitted to {quantile} quantile Yearly stacked upto {kk.columns[-1]}"
        # )
        # ax[1].legend()
        # ax[1].invert_xaxis()  # optional, if lower days are later

        # plt.show()
    return spline_models


def spline_models_v1(df: pd.DataFrame) -> list[UnivariateSpline]:
    """
    Create spline models for risk estimation using unified approach (Version 1).

    This function builds univariate spline models using a different data structure
    than v2. It expects data in long format with specific columns and applies
    weighted fitting with emphasis on more recent observations.

    Args:
        df (pd.DataFrame): DataFrame in long format with columns:
                          - 'last_date': Date identifier for temporal grouping
                          - 'days_to_roll': X-axis values (days until roll)
                          - 'value': Y-axis values (risk/volatility measures)

    Returns:
        list[UnivariateSpline]: List of fitted spline models, one for each
                               unique last_date in chronological order.
                               Each spline maps days-to-roll to risk values.

    Algorithm:
        1. Group data by last_date in chronological order
        2. For each group, fit spline with weighted observations
        3. Apply higher weights (5x) to first 20 observations
        4. Use variance-adjusted smoothing parameter
        5. Fit degree-5 splines for smooth interpolation

    Weighting Strategy:
        - First 20 points: weight = 5 (emphasized recent data)
        - Remaining points: weight = 1 (standard weighting)

    Note:
        This version includes commented visualization code.
        The weighting scheme prioritizes recent observations for better
        forward-looking risk estimates.
    """

    spline_models = []

    for yr in sorted(df.last_date.unique()):
        gg = df[df.last_date <= yr]
        x = gg["days_to_roll"].values
        y = gg["value"].values
        weights = [5] * 20 + [1] * (len(x) - 20)
        smoothing = len(x) * gg["value"].var() ** 2
        spline = UnivariateSpline(x, y, k=5, s=smoothing, w=weights)
        spline_models.append(spline)

        # x_smooth = np.linspace(0, 365, 500)
        # y_smooth = spline(x_smooth)
        # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6))
        # ax[0].scatter(x, y, alpha=0.1, label="Raw data")
        # ax[0].plot(x_smooth, y_smooth, color="red", label="Unified spline")
        # ax[0].set_xlabel("Days to Roll")
        # ax[0].set_ylabel("Value")
        # ax[0].set_title(f"Unified Spline upto(including) {yr}; points {len(x)}")
        # ax[0].legend()
        # ax[0].invert_xaxis()  # optional, if lower days are later
        # ax[1].plot(x_smooth, y_smooth, color="red", label="Unified spline")
        # ax[1].set_xlabel("Days to Roll")
        # ax[1].set_ylabel("Value")
        # ax[1].set_title(f"Unified Spline upto(including) {yr}; points {len(x)}")
        # ax[1].invert_xaxis()  # optional, if lower days are later
        # plt.show()
    return spline_models


def final_risk(data_nbadj, roll_index_date_map, skip_index, spline_models) -> pd.Series:
    """
    Calculate final risk estimates combining baseline risk with spline-based adjustments.

    This function produces the ultimate risk estimate by comparing current volatility
    conditions against spline-modeled expectations and applying adjustments when
    current volatility appears insufficient relative to historical patterns.

    Args:
        data_nbadj: DataFrame with risk data containing columns:
                   - 'risk_ext': Baseline risk extreme values
                   - 'days_to_roll': Days until contract roll (timedelta)
                   - 'roll_wise_vol': Current rolling volatility measure
        roll_index_date_map (dict): Mapping of dates to roll indices for spline selection.
                                   Each entry should have 'roll_index' key.
        skip_index (set/list): Roll indices to skip (use baseline risk only).
        spline_models (list): Pre-fitted spline models from spline_models_v1/v2.

    Returns:
        pd.Series: Final risk estimates with original datetime index.
                   Values represent adjusted risk measures incorporating
                   both current conditions and modeled expectations.

    Algorithm:
        1. For each observation, determine appropriate spline model
        2. Skip if no historical data or index in skip_index
        3. Get modeled risk expectation from spline based on days_to_roll
        4. If current volatility >= modeled expectation: use baseline risk
        5. If current volatility < modeled expectation: adjust risk upward
        6. Adjustment uses quadratic combination: sqrt(baseline² + adjustment²)

    Risk Adjustment Logic:
        - No adjustment needed: current_vol >= spline_prediction
        - Adjustment required: current_vol < spline_prediction
        - New risk = sqrt(baseline_risk² + (spline_prediction - current_vol)²)

    Note:
        This approach ensures risk estimates never fall below modeled expectations
        while preserving baseline risk characteristics when conditions are normal.
    """
    if len(skip_index) != 0:
        print(
            "[!] We'll be skipping some roll-wise spline models (len(skip_index) != 0)"
        )
        print("\t[!]", skip_index)
    frisk = []
    for ts in data_nbadj.itertuples():
        roll_index = roll_index_date_map[ts.Index]["roll_index"] - 1
        if roll_index == -1:
            # No backward looking data!
            frisk.append((ts.Index, ts.risk_ext))
            continue
        if roll_index in skip_index:
            frisk.append((ts.Index, ts.risk_ext))
            continue
        else:
            dte = ts.days_to_roll.days
            model = spline_models[roll_index_date_map[ts.Index]["roll_index"]]
            modeled_risk = model(dte)
            if np.isnan(ts.roll_wise_vol):
                frisk.append((ts.Index, ts.risk_ext))
                continue
            if ts.roll_wise_vol >= modeled_risk:
                frisk.append((ts.Index, ts.risk_ext))
            else:
                to_add = modeled_risk - ts.roll_wise_vol
                new_vol = np.sqrt(ts.risk_ext**2 + to_add**2)
                frisk.append((ts.Index, new_vol))

    frisk = np.vstack(frisk)
    frisk = pd.Series(frisk[:, 1], index=frisk[:, 0].flatten(), name="frisk")
    return frisk.astype(float)
