from datetime import datetime
from pathlib import Path

import pandas as pd

from gscbt import utils

CACHE_PATH = utils.LOCAL_STORAGE_PATH / "cache"


class DataPipeline:

    def __init__(self):
        # this will create folder for cache store if not exist
        Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)

        self.interval_to_second_map = {
            "1m": 60,
            "2m": 120,
            "3m": 180,
            "4m": 240,
            "5m": 300,
            "10m": 600,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
        }

    def get_pandas(self, tickers=[], back_adjusted=True, interval="1d"):
        df = pd.DataFrame()
        for ticker in tickers:

            file_path = (
                CACHE_PATH / ticker.exchange / ticker.symbol / ticker.type / interval
            )
            contract_type = "cd1" if back_adjusted else "c1"
            file_type = ".parquet"
            ticker.filename = ticker.symbol + contract_type + file_type
            path = file_path / ticker.filename

            if not path.exists():
                self._cache_data(ticker, interval)

            temp = pd.read_parquet(path, columns=["timestamp", "close"])

            # Convert 'timestamp' to datetime
            temp["timestamp"] = pd.to_datetime(temp["timestamp"])

            # Rename 'close' column
            temp = temp.rename(columns={"close": ticker.symbol})

            # temp["timestamp"] = temp["timestamp"].map(utils.convert_date_for_json)

            temp["timestamp"] = pd.to_datetime(temp["timestamp"], format="%Y-%m-%d")
            temp.set_index(["timestamp"], inplace=True)

            # Initialize df if empty, else perform full outer join
            if df.empty:
                df = temp
            else:
                df = pd.merge(df, temp, how="outer", left_index=True, right_index=True)

        return df

    def get(
        self,
        tickers=[],
        ohclv="ohclv",
        back_adjusted=True,
        start=None,
        end=None,
        interval="1d",
    ):
        df = pd.DataFrame()

        for ticker in tickers:
            temp_df = self._get_one(ticker, ohclv, back_adjusted, start, end, interval)
            if df.empty:
                df = temp_df
            else:
                df = pd.merge(
                    df, temp_df, how="outer", left_index=True, right_index=True
                )

        # convert df into
        # Close          Low
        # CL    RC      CL    RC
        df = df.swaplevel(axis=1).sort_index(axis=1)
        return df

    def _get_one(self, ticker, ohclv, back_adjusted, start, end, interval):
        file_path = (
            CACHE_PATH / ticker.exchange / ticker.symbol / ticker.type / interval
        )

        # c1 is for non back adjusted
        # cd1 is for back adjusted
        contract_type = "cd1" if back_adjusted else "c1"
        file_type = ".parquet"
        ticker.filename = ticker.symbol + contract_type + file_type

        path = file_path / ticker.filename
        # if no cache available than cache it
        if not path.exists():
            self._cache_data(ticker, interval, back_adjusted)

        column_list = ["timestamp"] if interval == "1d" else ["timeutc"]
        if "o" in ohclv:
            column_list.append("open")
        if "h" in ohclv:
            column_list.append("high")
        if "c" in ohclv:
            column_list.append("close")
        if "l" in ohclv:
            column_list.append("low")
        if "v" in ohclv:
            column_list.append("volume")

        # df = pd.read_csv(path)
        df = pd.read_parquet(path, columns=column_list)

        # droping open interest column
        # bcz 5 min data don't have that column
        # df.drop(columns="open_int", inplace=True, errors="ignore")

        if interval != "1d":
            df = df.rename(columns={"timeutc": "timestamp"})

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index(["timestamp"], inplace=True)

        # Create MultiIndex for column
        columns = pd.MultiIndex.from_tuples(
            [
                (ticker.symbol, i)
                for i in column_list
                if (i != "timestamp" and i != "timeutc")
            ]
        )
        df.columns = columns

        return df.loc[start:end]

    # this function will cache file if not exits
    # CACHE_PATH / exchange / symbol / type / filename
    # filename : c1 if back adjusted
    #          : h23 for single
    # this function assume ticker have one added attribute/key "filename"
    def _cache_data(self, ticker, interval, back_adjusted):
        file_path = (
            CACHE_PATH / ticker.exchange / ticker.symbol / ticker.type / interval
        )
        Path(file_path).mkdir(parents=True, exist_ok=True)

        # ?symbols={ticker}c1&{date}

        # if interval == "5m":
        #     url = utils.API["QUANT_APIS"]

        #     path = file_path / Path(ticker.filename).with_suffix(".json")
        #     # download csv
        #     utils.download_file(url, path, params)
        #     # convert csv into parquet
        #     utils.json_to_parquet(path)

        if interval == "1d":
            url = utils.API["GET_MARKET_DATA"]
            params = {
                # removeing .csv with assumption that there is no "." in file name
                "symbols": ticker.filename.split(".")[0],
                "from": ticker.data_from_date.strftime("%Y-%m-%d"),
                "to": datetime.today().strftime("%Y-%m-%d"),
            }

            path = file_path / Path(ticker.filename).with_suffix(".json")
            # download csv
            utils.download_file(url, path, params)
            # convert csv into parquet
            utils.json_to_parquet(path)

        else:
            url = utils.API["GET_IQFEED_DATA"]
            params = {
                "symbols": ticker.iqfeed_symbol + ("#C" if back_adjusted else "#"),
                "start_date": ticker.data_from_date.strftime("%Y-%m-%d"),
                "end_date": datetime.today().strftime("%Y-%m-%d"),
                "type": "ohlcv",
                "duration": self.interval_to_second_map[interval],
            }

            path = file_path / Path(ticker.filename).with_suffix(".json")
            # download csv
            utils.download_file(url, path, params)
            # convert csv into parquet
            utils.json_to_parquet(path)

            # print("other timeframe data is currently not available/supported")

        # remove csv
        utils.remove_file(path)


if __name__ == "__main__":
    pass
