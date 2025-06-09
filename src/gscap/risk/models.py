import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline


def get_ext(series: pd.Series):
    return max(np.abs(series.quantile(0.025)), np.abs(series.quantile(1 - 0.025)))


def get_risk_vol(data: pd.DataFrame):
    expnd = data.close.diff().expanding(min_periods=20).std()
    two_yr = data.close.diff().rolling(252 * 2, min_periods=20).std()
    six_m = data.close.diff().rolling(22 * 6, min_periods=20).std()
    ten_d = data.close.diff().ewm(10).std()

    ult = expnd * 0.10 + two_yr * 0.25 + six_m * 0.25 + ten_d * 0.40
    ult = pd.concat([ten_d, ult], axis=1).max(axis=1).squeeze()
    ult.name = "risk_vol"
    return ult


def get_risk_ext(data: pd.DataFrame):
    expnd = data.close.diff().expanding(min_periods=20).apply(get_ext)
    two_yr = data.close.diff().rolling(252 * 2, min_periods=20).apply(get_ext)
    six_m = data.close.diff().rolling(22 * 6, min_periods=20).apply(get_ext)
    ten_d = data.close.diff().rolling(10).apply(get_ext)

    ult = expnd * 0.10 + two_yr * 0.25 + six_m * 0.25 + ten_d * 0.40
    ult = pd.concat([ten_d, ult], axis=1).max(axis=1).squeeze()
    ult.name = "risk_ext"
    return ult


def spline_models_v2(df: pd.DataFrame, lb: int, qtile: float) -> list[UnivariateSpline]:
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

    spline_models = []

    for yr in sorted(df.year.unique()):
        gg = df[df.year <= yr]
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


def final_risk(data_nbadj, roll_index_date_map, skip_index, spline_models):
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
    frisk = pd.Series(frisk[:, 1], index=frisk[:, 0].flatten())
    return frisk
