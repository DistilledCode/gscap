from datetime import datetime
from pathlib import Path

import requests
from requests.exceptions import HTTPError, Timeout, RequestException
import pandas as pd
import polars as pl

# cosnts
SERVER_IP_PORT = "192.168.0.25:8080"
# LOCAL_WIN_DIRECT_IQFEED_IP_PORT = "192.168.0.xx:5675"
LOCAL_WIN_DIRECT_IQFEED_IP_PORT = "192.168.0.68:5675"

# application local data in user's space
LOCAL_STORAGE_PATH = Path.home() / ".gscbt"

# this data is packed with library
PACKAGE_DIR_PATH = Path(__file__).parent
LOCAL_DATA_PATH = PACKAGE_DIR_PATH / "data"


# format is same as title of swagger api docs
# all caps and space with '_'
API = {
    "GET_USD_CONVERSION": f"http://{SERVER_IP_PORT}/api/v1/data/dollarequivalent",
    "DOWNLOAD_MARKET_DATA": f"http://{SERVER_IP_PORT}/api/v1/data/download",
    "GET_IQFEED_DATA": f"http://{SERVER_IP_PORT}/api/v2/data/iqfeed",
    "GET_MARKET_DATA": f"http://{SERVER_IP_PORT}/api/v1/data/ohlcv",
    "QUANT_APIS": f"http://{SERVER_IP_PORT}/api/v1/quant/data/ohlcv",  # for 5 min data
    "DIRECT_IQFEED_APIS": f"http://{LOCAL_WIN_DIRECT_IQFEED_IP_PORT}/api/v1/data_parquet/iqfeed",
}


# this function is bind with api
def get_USD_conversion(ticker) -> float:
    params = {"sym": ticker.currency + "USDc1"}

    # res value : [{'timestamp': '2025-02-10T00:00:00Z', 'sym': 'EURUSDc1', 'settlement': 1.0324}]
    res = requests.get(API["GET_USD_CONVERSION"], params=params).json()

    # extract float price and return
    return res[0]["settlement"]


# this function will give dollar equivalent multiplier as per current rate
def get_dollar_equivalent(ticker) -> float:
    mul = 1.0
    if ticker.currency != "USD":
        mul = get_USD_conversion(ticker)

    return ticker.currency_multiplier * mul


def download_file(url, filename_with_path, params=None, allow_redirect=False):
    try:
        # Send a GET request with stream=True to download the file in chunks
        response = requests.get(
            url,
            params=params,
            stream=True,
            timeout=600,
            allow_redirects=allow_redirect,
        )  # Timeout set to 30 seconds

        # Check if the request was successful
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        # Open the file in write-binary mode
        with open(filename_with_path, "wb") as file:
            # Download the file in chunks
            for chunk in response.iter_content(chunk_size=8192):  # 8 KB chunks
                file.write(chunk)

    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Timeout as timeout_err:
        print(f"Request timed out: {timeout_err}")
    except RequestException as req_err:
        print(f"Error during request: {req_err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")


# this function make python dictionary dot accessable
# it's recursive function <costly>
class Dotdict:
    def __init__(self, data):
        for key, value in data.items():
            setattr(self, key, Dotdict(value) if isinstance(value, dict) else value)

    def __getattr__(self, item):
        raise AttributeError(f"DotDict object has no attribute {item}")


# convert date for pandas dataframe object to datetime
# handle problem related to timezone
# data source : inhouse api csv downloadable
def convert_date_for_csv(date, date_format="%Y-%m-%d %H:%M:%S %z %Z"):
    return datetime.strptime(str(date), date_format)


# data source : inhouse api JSON response
def convert_date_for_json(date, date_format="%Y-%m-%dT%H:%M:%SZ"):
    return datetime.strptime(str(date), date_format)


# this function support lazy scan
# implies it can convert data which is larger than ram
def csv_to_parquet(csv_path, parquet_path=None):
    if parquet_path is None:
        parquet_path = csv_path.with_suffix(".parquet")

    df = pl.scan_csv(csv_path)
    df.sink_parquet(parquet_path)


# this function don't support lazy scan
# bcz json is not in NDJSON format
def json_to_parquet(json_path, parquet_path=None):
    if parquet_path is None:
        parquet_path = json_path.with_suffix(".parquet")

    df = pl.read_json(json_path)
    df.write_parquet(parquet_path)


# store api json response inMemory
# convert it into parquet file
def json_stream_to_parquet(url, params, parquet_path):
    json_data = requests.get(url, params)
    pl.DataFrame(json_data).write_parquet(parquet_path)


def remove_file(path):
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    pass
