# import numpy as np
# import pandas as pd
# from scipy.optimize import minimize
# import gscbt

# # ------------------------------
# # 1. GENERATE SAMPLE RETURN DATA
# # ------------------------------


# # np.random.seed(42)
# # dates = pd.date_range(start='2020-01-01', periods=500)

# # Simulate returns for 3 instruments (mean ~0.05%, std ~1%)
# # data = pd.DataFrame({
# #     'Stock_A': np.random.normal(0.0005, 0.01, size=500),
# #     'Bond_B': np.random.normal(0.0003, 0.005, size=500),
# #     'Commodity_C': np.random.normal(0.0004, 0.015, size=500)
# # }, index=dates)

# dp = gscbt.DataPipeline()
# cme = gscbt.get_tickers().cme


# df = dp.get_pandas([cme.zb.f, cme.es.f, cme.nq.f])
# # print(data)
# print(df)

# # print(np.log(df/df.shift(1)).dropna())
# # data = df.pct_change().dropna()
# data = np.log(df/df.shift(1)).dropna()
# # data = data.loc["2000-01-01":"2014-12-31"]
# data = data.loc["2000-01-01":]

# print(data)

# # ------------------------------
# # 2. BOOTSTRAPPING FUNCTION
# # ------------------------------
# def bootstrap_returns(data, n_iterations=1000):
#     bootstrapped_samples = []
#     for _ in range(n_iterations):
#         sample = data.sample(frac=0.1, replace=True)
#         bootstrapped_samples.append(sample)
#     return bootstrapped_samples

# # ------------------------------
# # 3. PORTFOLIO OPTIMIZATION
# # ------------------------------
# def sharpe_ratio(weights, returns):
#     portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized
#     portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
#     return -portfolio_return / portfolio_volatility  # Negative for minimization

# # Constraint: Weights sum to 1
# def optimize_portfolio(returns):
#     # print("RETURN")
#     # print(returns)
#     n_assets = returns.shape[1]
#     # print("n-assets")
#     # print(n_assets)
#     init_weights = np.ones(n_assets) / n_assets
#     # print(init_weights)
#     bounds = [(0, 1) for _ in range(n_assets)]
#     # print(bounds)
#     constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
#     # print(constraints)

#     result = minimize(sharpe_ratio, init_weights, args=(returns,), bounds=bounds, constraints=constraints)
#     # print(result)
#     return result.x

# # ------------------------------
# # 4. RUN BOOTSTRAPPING & OPTIMIZATION
# # ------------------------------
# boot_samples = bootstrap_returns(data, n_iterations=200)
# # print(boot_samples)
# optimized_weights = []

# for sample in boot_samples:
#     weights = optimize_portfolio(sample)
#     optimized_weights.append(weights)
#     # break

# # Average weights across all bootstrapped samples
# avg_weights = np.mean(optimized_weights, axis=0)
# final_weights = avg_weights / np.sum(avg_weights)  # Normalize

# # ------------------------------
# # 5. RISK SCALING
# # ------------------------------
# volatility = data.std() * np.sqrt(252)
# risk_scaled_weights = final_weights / volatility
# risk_scaled_weights /= risk_scaled_weights.sum()  # Normalize again

# # ------------------------------
# # 6. RESULTS
# # ------------------------------
# assets = data.columns

# print("\nAverage Optimized Weights (Pre-Risk Scaling):")
# for asset, weight in zip(assets, final_weights):
#     print(f"{asset}: {weight:.2%}")

# print("\nFinal Risk-Scaled Weights:")
# for asset, weight in zip(assets, risk_scaled_weights):
#     print(f"{asset}: {weight:.2%}")



# def single_volatility_scalar(df_back_adjusted, df_non_back_adjusted, ticker, daily_cash_volatility_target, halflife):
#     df = pd.DataFrame()
#     df["daily_returns"] = (df_back_adjusted[ticker.symbol].diff() / df_non_back_adjusted[ticker.symbol].shift(1))*100
#     df["daily_returns"] = df["daily_returns"].fillna(0)

#     df["daily_volatility"] = df["daily_returns"].ewm(halflife = halflife).std() #* np.sqrt(252)

#     df["block_value"] = (df_back_adjusted[ticker.symbol] * ticker.currency_multiplier) / 100
#     df["instrument_currency_volatility"] = df["daily_volatility"] * df["block_value"]

#     df["volatility_scalar"] = daily_cash_volatility_target / df["instrument_currency_volatility"]
#     df[ticker.symbol] = df["volatility_scalar"].shift(1)

#     return df[[ticker.symbol]]


# def volatility_scalar(portfolio, daily_cash_volatility_target, halflife=11, interval="1d"):
#     pipe = DataPipeline()
#     df = pd.DataFrame()
#     for ticker in portfolio:
#         df_b = pipe.get_pandas([ticker], interval=interval)
#         df_nb = pipe.get_pandas([ticker], back_adjusted=False, interval=interval)
#         temp = single_volatility_scalar(df_b, df_nb, ticker, daily_cash_volatility_target, halflife)

#         if df.empty:
#             df = temp
#         else:
#             df = pd.merge(df, temp,  how="outer", left_index=True, right_index=True)

#     return df        
