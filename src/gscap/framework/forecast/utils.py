from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from gscap.framework import Instrument


import numpy as np


def _fw_equi(instrument: Instrument):
    _n = len(instrument.forecast)
    return np.full(_n, 1 / _n)


# def _fw_corr(corr_matrix: np.ndarray):
#     """
#     ! This function moderately punishes forecasts with higher correlation

#     EWMACForecast(fast=002, slow=0008, clip=20, abs_avg=10, vol_span=35): 0.3077
#     EWMACForecast(fast=008, slow=0032, clip=20, abs_avg=10, vol_span=35): 0.1686
#     EWMACForecast(fast=032, slow=0128, clip=20, abs_avg=10, vol_span=35): 0.1431
#     EWMACForecast(fast=064, slow=0256, clip=20, abs_avg=10, vol_span=35): 0.1463
#     EWMACForecast(fast=256, slow=1024, clip=20, abs_avg=10, vol_span=35): 0.2341
#     """
#     _corr = np.copy(corr_matrix)
#     np.fill_diagonal(_corr, 0)
#     corr_matrix = pd.DataFrame(_corr)
#     _mean = corr_matrix.mean(axis=1)
#     _norm = _mean.min() / _mean
#     return _norm / _norm.sum()


def _fw_corr(corr_matrix: np.ndarray):
    """
    ! This function severely punishes forecasts with higher correlation

    EWMACForecast(fast=002, slow=0008, clip=20, abs_avg=10, vol_span=35): 0.4606
    EWMACForecast(fast=008, slow=0032, clip=20, abs_avg=10, vol_span=35): 0.1438
    EWMACForecast(fast=032, slow=0128, clip=20, abs_avg=10, vol_span=35): 0.0188
    EWMACForecast(fast=064, slow=0256, clip=20, abs_avg=10, vol_span=35): 0.0367
    EWMACForecast(fast=256, slow=1024, clip=20, abs_avg=10, vol_span=35): 0.3398
    """
    corr_matrix = pd.DataFrame(corr_matrix).sum(axis=1).add(-1)
    _y = corr_matrix.max() - corr_matrix + 0.05
    return _y / _y.sum()


def inv_sqrt_fn(ser: pd.Series, fw: np.array):
    return [np.nan if np.allclose(i, 0) else 1 / np.sqrt(fw @ i @ fw.T) for i in ser]


def _get_fdm_calculate_fw(instrument: Instrument, resample="W", weights: str = "corr"):

    rho_fc = []
    _ts = []

    _all_forecasts = pd.concat([f.forecast_value for f in instrument.forecast], axis=1)

    resampled = _all_forecasts.resample(resample).last()
    _ts = resampled.index
    rho_fc = [i.corr(min_periods=2).values.clip(0) for i in resampled.expanding()]

    rho_fc = pd.DataFrame({"fdm": rho_fc}, index=_ts).rename_axis("Timestamp")

    # ! We need forecast weights to calculate FDM
    if weights == "equi":
        fw = _fw_equi(instrument)
    elif weights == "corr":
        fw = _fw_corr(rho_fc.iloc[-1].iloc[0])
    instrument.fw = {k: v for k, v in zip(instrument.forecast, fw)}

    _fdm = rho_fc.apply(lambda x: inv_sqrt_fn(x, fw))
    _fdm = _fdm.reindex(rho_fc.index.union(_all_forecasts.index))
    _fdm = _fdm.bfill().ffill()
    _fdm = _fdm.reindex(_all_forecasts.index)

    instrument.fdm = _fdm.squeeze()
    return instrument.fdm


def combined_forecast(instrument: Instrument, weights="corr", fdm_resample="W"):
    if instrument.forecast is None:
        raise AttributeError(f"No forecast(s) found for {instrument}")
    if any([f.forecast_value is None for f in instrument.forecast]):
        raise ValueError("One or more Forecast initiated but not generated!")
    if len(instrument.forecast) == 1:
        instrument.fw = np.array([1.0])
        _index = instrument.forecast[0].forecast_value.index
        instrument.fdm = pd.Series(1.0, index=_index, name="fdm")
        return instrument.forecast[0].forecast_value

    fdm = _get_fdm_calculate_fw(instrument, resample=fdm_resample, weights=weights)

    assert instrument.fw is not None
    assert instrument.fdm is not None

    _combined_fcast = sum(
        fweight * fcast.forecast_value
        for fweight, fcast in zip(instrument.fw.values(), instrument.forecast)
    )
    _combined_fcast = _combined_fcast.squeeze() * fdm
    _combined_clip = np.mean([i.clip for i in instrument.forecast])
    return _combined_fcast.clip(-_combined_clip, _combined_clip)
