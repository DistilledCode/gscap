import numpy as np
import pandas as pd
from numba import jit
from scipy.optimize import minimize


def volatility_scalar(
    price_df: pd.DataFrame,
    ticker,
    unit_cash_volatility_target,
    slow_span_days=None,
    fast_span_days=None,
):

    und: pd.Series = price_df.underlying.squeeze()

    # ! Considering only day end close price for std calculation
    # ! and forward filling for vps_slow will not work.
    # ! It will decimate the returns
    vps_slow = und.diff().ewm(span=slow_span_days).std()
    vps_fast = und.diff().ewm(span=fast_span_days).std()
    vol_price_term = vps_slow * 0.30 + vps_fast * 0.70

    # It can be inf when the first n-differences are same.
    _z = vol_price_term[vol_price_term == 0.00]
    if len(_z) > 0:
        print(f"Removing {len(_z)} instances of zero price diff vol")
        print("\t", *list(_z.index), sep="\n\t",end="\n\n")
        vol_price_term.replace(0.00, np.nan, inplace=True)

    vol_exposure_terms = vol_price_term * ticker.dollar_equivalent
    vol_exposure_terms = vol_exposure_terms.shift(1)
    vol_scalar = unit_cash_volatility_target / vol_exposure_terms

    return pd.DataFrame(
        {
            "unit_vol_exp_term": vol_exposure_terms,
            "vol_scalar": vol_scalar,
        },
        index=vol_scalar.index,
    )


# def volatility_scalar(
#     price_df: pd.DataFrame,
#     return_callable,
#     ticker,
#     unit_cash_volatility_target,
#     span=None,
# ):

#     adj = price_df.adjusted.squeeze()
#     und = price_df.underlying.squeeze()
#     _neg_values = {i: None for i in und[und.le(0)].values}
#     print(f"forward filling {len(_neg_values)} negative values.")
#     ffiled_neg_und = und.replace(_neg_values)

#     # ! Refer to page 666 of AFTS
#     # ! Author recommends same values for num and denom but we calculate
#     # ! returns with adj in numberator and und in denominator
#     # ! Thus our analogous would be std(price diff of adj) divided by
#     # ! underlying (ffiled negative value with latest positive value)
#     # ! ############################################################
#     # !           BEHAVIOR OF THIS METHOD WITH PROLONGED
#     # !       NEGATIVE UNDERLYING VALUE IS NOT ANALYZED YET
#     # ! ############################################################

#     unit_vol_perc = adj.diff().ewm(span=span).std() / ffiled_neg_und
#     unit_vol_perc = unit_vol_perc.shift(1)
#     block_val = adj * ticker.dollar_equivalent
#     inst_c_vol = unit_vol_perc * block_val
#     vs = unit_cash_volatility_target / inst_c_vol

#     return pd.DataFrame(
#         {
#             "inst_curr_vol": inst_c_vol,
#             "vol_scalar": vs,
#         },
#         index=vs.index,
#     )

# @jit(forceobj=True)
# def volatility_scalar(
#     price_df,
#     return_callable,
#     ticker,
#     unit_cash_volatility_target,
#     span=None,
# ):

#     temp = pd.DataFrame()
#     temp["unit_returns"] = return_callable(price_df)
#     # temp["unit_returns"] = temp["unit_returns"].fillna(0)
#     temp["unit_returns"] = temp["unit_returns"]
#     temp["unit_vol"] = temp["unit_returns"].ewm(span=span).std()

#     n = len(temp)
#     underlying_price = price_df.underlying[ticker.symbol.lower()].values
#     unit_vol = temp["unit_vol"].values
#     equal_price_counter = np.zeros(n, dtype=int)
#     vol_resume = span

#     for i in range(1, n):
#         if underlying_price[i] == underlying_price[i - 1]:
#             equal_price_counter[i] = equal_price_counter[i - 1] + 1
#         if equal_price_counter[i] >= span:
#             copy_vol_idx = int(i - equal_price_counter[i])
#             if copy_vol_idx >= 0:
#                 unit_vol[i] = unit_vol[copy_vol_idx]
#         if equal_price_counter[i] == 0 and equal_price_counter[i - 1] >= span:
#             for j in range(vol_resume):
#                 if i + j < n:
#                     unit_vol[i + j] = unit_vol[i - 1]

#     for i in range(1, n):
#         if unit_vol[i] == 0:
#             unit_vol[i] = np.nan

#     temp["unit_vol"] = unit_vol
#     temp["unit_vol"] = temp["unit_vol"].shift(1)
#     temp["block_value"] = price_df.underlying[ticker.symbol.lower()]
#     temp["block_value"] *= ticker.currency_multiplier

#     temp["instrument_currency_vol"] = temp["unit_vol"] * temp["block_value"]
#     temp["vol_scalar"] = unit_cash_volatility_target / temp["instrument_currency_vol"]
#     # temp["vol_scalar"] = temp["vol_scalar"].shift(1)
#     return temp
#     # return temp[["vol_scalar"]].squeeze()


def correlation(returns):
    return returns.corr().clip(lower=0)


def sharpe_ratio(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(returns.cov() * 252, weights))
    )
    return -portfolio_return / portfolio_volatility  # Negative for minimization


def optimize_portfolio(sample_returns):

    n_assets = sample_returns.shape[1]
    init_weights = np.ones(n_assets) / n_assets
    bounds = [(0, 1) for _ in range(n_assets)]
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

    result = minimize(
        sharpe_ratio,
        init_weights,
        args=(sample_returns,),
        bounds=bounds,
        constraints=constraints,
    )

    return result.x


def bootstrap(returns, n_itr, frac):
    optimized_weights = []
    for _ in range(n_itr):
        sample = returns.sample(frac=frac, replace=True)
        weight = optimize_portfolio(sample)
        optimized_weights.append(weight)

    avg_weights = np.mean(optimized_weights, axis=0)
    final_weights = avg_weights / np.sum(avg_weights)

    # risk scaling
    volatility = returns.std() * np.sqrt(252)
    risk_scaled_weights = final_weights / volatility
    risk_scaled_weights /= risk_scaled_weights.sum()

    return risk_scaled_weights


def instrument_weight(returns, resample="YE", n_itr=100, frac=0.1):
    # returns = price_df.adjusted.diff() / price_df.underlying.shift(1)
    resample_returns = returns.resample(resample).last()

    list_weights = []
    for timestamp in resample_returns.index:
        weights = bootstrap(returns.loc[:timestamp], n_itr, frac)
        weights.name = timestamp
        list_weights.append(weights)

    df = pd.concat(list_weights, axis=1).T
    df = df.fillna(0)
    temp_index = df.index.union(returns.index)
    df = df.reindex(temp_index).bfill().ffill().reindex(returns.index)

    return df


def _IDM(returns, weights):
    W = weights
    H = correlation(returns)

    if np.allclose(H, 0):
        return np.nan

    idm = 1 / np.sqrt(W.T @ H @ W)
    return idm


def IDM(returns, weights, resample="W"):
    # returns = price_df.adjusted.diff() / price_df.underlying.shift(1)
    resample_returns = returns.resample(resample).last()

    resample_returns["idm"] = None
    for timestamp in resample_returns.index:
        # timestamp of weight can be included?
        _r = returns.loc[:timestamp]
        _w1 = weights.loc[:timestamp]
        if _r.empty or _w1.empty:
            continue
        _w2 = _w1.iloc[-1]
        resample_returns.loc[timestamp, "idm"] = _IDM(_r, _w2)

    df = resample_returns["idm"].to_frame()
    temp_index = df.index.union(returns.index)
    df = df.reindex(temp_index).bfill().ffill().reindex(returns.index)

    return df
