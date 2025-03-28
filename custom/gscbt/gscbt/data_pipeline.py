from datetime import datetime
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pyarrow.parquet as pq

from gscbt import utils

CACHE_PATH = utils.LOCAL_STORAGE_PATH / "cache"


class DataPipeline:

    def __init__(self):
        # this will create folder for cache store if not exist
        Path(CACHE_PATH).mkdir(parents=True, exist_ok=True)

        self.interval_to_second_map = {
            "15s": 15,
            "30s": 30,
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
            "1d": 1,  # fetch through iqfeed eod
        }

    def get(
        self,
        tickers=[],
        ohclv="ohclv",
        back_adjusted=True,
        start=None,
        end=None,
        interval="1d",
        cache_mode="data_api",
    ):
        df = pd.DataFrame()

        for ticker in tickers:
            temp_df = self._get_one(
                ticker, ohclv, back_adjusted, start, end, interval, cache_mode
            )
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

    def _get_one(self, ticker, ohclv, back_adjusted, start, end, interval, cache_mode):
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
            self._cache_data(ticker, interval, back_adjusted, cache_mode)

        column_list = ["timeutc"]
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

        # old_df = pd.read_parquet(path, columns=column_list)

        # this will read only rows which requrie + some over head
        # future TODO: direct metadata storing of each parquet file can give more faster access
        # it will more like creating database indexing stuff
        # pf : parquet file
        pf = pq.ParquetFile(path)

        # first we fetch start and end row gorud id from stored parquet format
        st_row_group_idx = 0
        end_row_group_idx = pf.num_row_groups - 1

        if start != None:
            st_row_group_idx = self.row_group_finder(pf, column_list[0], start)

        if end != None:
            end_row_group_idx = self.row_group_finder(pf, column_list[0], end)

        # join all row groups from start to end
        # convert it into pandas dataframe
        list_of_row_groups = list(range(st_row_group_idx, end_row_group_idx + 1))
        df = pf.read_row_groups(list_of_row_groups, columns=column_list).to_pandas()

        # renaming timeutc to timestamp
        df = df.rename(columns={"timeutc": "timestamp"})

        # df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        # print(df["timestamp"])
        # print(pd.to_datetime(df["timestamp"], utc=True))

        df.set_index(["timestamp"], inplace=True)

        # bcz we are accessing row groups we have some unneccessary entries at start and end
        # which requrie us to cut it
        df = df.loc[start:end]

        # re-backadjust the data from the end date
        if back_adjusted and end != None:

            # getting last row of underlying for finding diff
            # future TODO: code getting repeat
            underlying_contract_type = "c1"
            back_adjusted_contract_type = "cd1"
            # file_type = ".parquet"

            underlying_filename = ticker.symbol + underlying_contract_type + file_type
            underlying_path = file_path / underlying_filename

            back_adjusted_filename = (
                ticker.symbol + back_adjusted_contract_type + file_type
            )
            back_adjusted_path = file_path / back_adjusted_filename

            # there is case where we believe that underlying data is already cache
            # but there is possibility that underlying don't exists

            if not underlying_path.exists():
                self._cache_data(ticker, interval, back_adjusted=False, mode=cache_mode)

            u_pf = pq.ParquetFile(underlying_path)
            u_row_group_idx = self.row_group_finder(u_pf, column_list[0], end)
            u_df = u_pf.read_row_group(
                u_row_group_idx, columns=[column_list[0], "close"]
            ).to_pandas()

            b_pf = pq.ParquetFile(back_adjusted_path)
            b_row_group_idx = self.row_group_finder(b_pf, column_list[0], end)
            b_df = b_pf.read_row_group(
                b_row_group_idx, columns=[column_list[0], "close"]
            ).to_pandas()

            u_df = u_df.rename(columns={"timeutc": "timestamp"})
            b_df = b_df.rename(columns={"timeutc": "timestamp"})

            u_df["timestamp"] = pd.to_datetime(u_df["timestamp"], utc=True)
            u_df.set_index(["timestamp"], inplace=True)

            b_df["timestamp"] = pd.to_datetime(b_df["timestamp"], utc=True)
            b_df.set_index(["timestamp"], inplace=True)

            u_df = u_df.loc[:end]
            b_df = b_df.loc[:end]

            u_df = u_df.reindex(b_df.index)

            diff = u_df.iloc[-1]["close"] - b_df.iloc[-1]["close"]
            # print(diff)

            # eleminate diff but not from volume column
            if "open" in df.columns:
                df["open"] = df["open"] + diff
            if "close" in df.columns:
                df["close"] = df["close"] + diff
            if "high" in df.columns:
                df["high"] = df["high"] + diff
            if "low" in df.columns:
                df["low"] = df["low"] + diff

        # Create MultiIndex for column
        columns = pd.MultiIndex.from_tuples(
            [
                (ticker.symbol, i)
                for i in column_list
                if (i != "timestamp" and i != "timeutc")
            ]
        )
        df.columns = columns

        df = df.sort_index()
        return df
        # return df.loc[start:end]

    # this function will cache file if not exits
    # CACHE_PATH / exchange / symbol / type / filename
    # filename : c1 if back adjusted
    #          : h23 for single
    # TODO: cache can be done directly from local running iqfeed desktop application
    def _cache_data(self, ticker, interval, back_adjusted, mode="data_api"):
        file_path = (
            CACHE_PATH / ticker.exchange / ticker.symbol / ticker.type / interval
        )
        Path(file_path).mkdir(parents=True, exist_ok=True)

        _filename = ticker.symbol + ("cd1" if back_adjusted else "c1")
        url = utils.API["GET_IQFEED_DATA"]
        params = {
            "symbols": ticker.iqfeed_symbol + ("#C" if back_adjusted else "#"),
            "start_date": ticker.data_from_date.strftime("%Y-%m-%d"),
            "end_date": datetime.today().strftime("%Y-%m-%d"),
            "type": "eod" if interval == "1d" else "ohlcv",
            "duration": self.interval_to_second_map[interval],
        }

        if mode == "direct_iqfeed":
            url = utils.API["DIRECT_IQFEED_APIS"]
            path = file_path / Path(_filename).with_suffix(".parquet")
            utils.download_file(url, path, params)

        else:
            path = file_path / Path(_filename).with_suffix(".json")
            utils.download_file(url, path, params)
            utils.json_to_parquet(path)

            # remove json file
            utils.remove_file(path)

    # bs : binary search
    # target string should be able to get converted into timestamp_format
    def is_timestamp_in_row_group_bs(self, pf, mid, col_name, target):

        table = pf.read_row_group(mid, columns=[col_name])

        st = table[0][0].as_py()
        end = table[0][-1].as_py()

        st = pd.to_datetime(st, utc=True)
        end = pd.to_datetime(end, utc=True)
        target = pd.to_datetime(target, utc=True)

        # if st.tz:
        #     target = pd.to_datetime(target, utc=True)
        # else:
        #     target = pd.to_datetime(target)

        if target < st:
            return -1
        if end < target:
            return 1
        return 0

    def row_group_finder(self, pf, col_name, target):

        num_row_groups = pf.num_row_groups

        left = 0
        right = num_row_groups - 1
        while left <= right:
            mid = left + (right - left) // 2

            temp_res = self.is_timestamp_in_row_group_bs(pf, mid, col_name, target)
            if temp_res == 0:
                return mid
            elif temp_res > 0:
                left = mid + 1
            else:
                right = mid - 1

        return -1

    # this will delete cache folder it self
    def remove_cache(self):
        cache_path = Path(CACHE_PATH)
        if cache_path.exists() and cache_path.is_dir():
            shutil.rmtree(cache_path)

    def cache_underlying_n_back_adjusted(
        self,
        tickers=[],
        intervals=[],
        interval_first=False,
        verbose=False,
        n_workers=1,
        cache_mode="data_api",
    ):
        if interval_first:
            for interval in intervals:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    for ticker in tickers:
                        executor.submit(
                            self._cache_data,
                            ticker,
                            interval,
                            back_adjusted=False,
                            mode=cache_mode,
                        )
                        executor.submit(
                            self._cache_data,
                            ticker,
                            interval,
                            back_adjusted=True,
                            mode=cache_mode,
                        )

                if verbose:
                    print(f"{interval} caching done.")

        else:
            for ticker in tickers:
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    for interval in intervals:
                        executor.submit(
                            self._cache_data,
                            ticker,
                            interval,
                            back_adjusted=False,
                            mode=cache_mode,
                        )
                        executor.submit(
                            self._cache_data,
                            ticker,
                            interval,
                            back_adjusted=True,
                            mode=cache_mode,
                        )

                if verbose:
                    print(f"{ticker.symbol} caching done.")


if __name__ == "__main__":
    pass
