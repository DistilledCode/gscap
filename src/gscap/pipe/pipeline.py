#
# ! ########################################### ! #
# !                  DEPRECIATED                ! #
# ! ########################################### ! #


# import os
# import urllib.request
# from datetime import datetime
# from pathlib import Path

# import pandas as pd


# class DataPipeline:
#     def __init__(self):
#         self.data_dir = Path().home() / ".mydata"
#         self.data_dir.mkdir(exist_ok=True)

#     def get(self, tickers, ohclv="ohclv", back_adjusted=True):
#         df = pd.DataFrame()
#         for ticker in tickers:
#             temp_df = self._get_one(ticker, ohclv, back_adjusted)

#             if df.empty:
#                 df = temp_df
#             else:
#                 df = pd.merge(
#                     df, temp_df, how="outer", left_index=True, right_index=True
#                 )

#         df = df.swaplevel(axis=1).sort_index(axis=1)
#         return df

#     def _get_one(self, ticker, ohclv, back_adjusted):
#         today = datetime.today().strftime("%Y-%m-%d")
#         date = f"from=1990-01-01&to={today}"

#         type = "cd1" if back_adjusted else "c1"

#         url = f"http://192.168.0.25:8080/api/v1/data/download?symbols={ticker}{type}&{date}"
#         filename = self.data_dir / f"{ticker}{type}.csv"

#         if not os.path.exists(filename):
#             urllib.request.urlretrieve(url, filename)

#         df = pd.read_csv(filename)

#         df.Timestamp = df.Timestamp.map(self._convert_date)
#         df.set_index(["Timestamp"], inplace=True)

#         # Create MultiIndex for column
#         columns = pd.MultiIndex.from_tuples(
#             [
#                 (ticker, "Sym"),
#                 (ticker, "Open"),
#                 (ticker, "High"),
#                 (ticker, "Low"),
#                 (ticker, "Close"),
#                 (ticker, "Volume"),
#                 (ticker, "OpenInterest"),
#             ]
#         )
#         df.columns = columns

#         column_drop_list = ["Sym"]
#         if "o" not in ohclv:
#             column_drop_list.append("Open")
#         if "h" not in ohclv:
#             column_drop_list.append("High")
#         if "c" not in ohclv:
#             column_drop_list.append("Close")
#         if "l" not in ohclv:
#             column_drop_list.append("Low")
#         if "v" not in ohclv:
#             column_drop_list.append("Volume")
#         # if "h" not in ohclv:
#         column_drop_list.append("OpenInterest")

#         df = df.drop(column_drop_list, axis=1, level=1)

#         return df

#     # convert date for pandas dataframe object to datetime
#     def _convert_date(self, date, date_format="%Y-%m-%d %H:%M:%S %z %Z"):
#         return datetime.strptime(str(date), date_format)


# if __name__ == "__main__":
#     pipe = DataPipeline()
#     df = pipe.get(["CL", "RC"], back_adjusted=False)
#     print(df.columns)
#     print(df.head())
