import numpy as np
import pandas as pd
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
        print("\t", *list(_z.index), sep="\n\t", end="\n\n")
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
