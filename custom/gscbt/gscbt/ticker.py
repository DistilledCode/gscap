from collections import defaultdict
from functools import lru_cache

import pandas as pd

from gscbt import utils

EXCEL_NAME = "iqfeed_data.xlsx"

# this function is couple with iqfeed_data.excel(which is maintain by dev team for internal usage)
# any change in that file leads to change in this function
# this function return ticker with it's metadata in dictionary format
def _parse_excel_to_dict():

    path = utils.LOCAL_DATA_PATH / EXCEL_NAME

    # Load Excel file into a DataFrame
    # skip first row "IQ FEED SYMBOL MAPPING"
    df = pd.read_excel(path, engine="calamine", skiprows=1)

    ###
    #  Breaking It Down:
    #  defaultdict(dict)

    #  This means that whenever a key is accessed that does not exist, it automatically creates an empty dictionary {}.
    #  lambda: defaultdict(dict)

    #  This means that each missing key in the first defaultdict will return another defaultdict(dict), allowing nested storage.
    #  Final Structure:

    #  result is a defaultdict where each value is another defaultdict of dictionaries.
    #  This allows automatic creation of deeply nested dictionaries.
    ###
    result = defaultdict(lambda: defaultdict(dict))  # Nested defaultdict to store the result

    # Iterate through the rows of the DataFrame
    for _, row in df.iterrows():
        # you will be using class.
        exchange = row["Exchange"].lower()
        symbol = row["Symbol"].lower()
        
        # Construct the dictionary structure for "Future" data
        # Make all column a field 
        future_data = {
            "product": row["Product"],
            "symbol": row["Symbol"],
            "type" : "futures", # custom added not part of excel file 
            "iqfeed_symbol": row["IQFeed symbol"],
            "exchange": row["Exchange"],
            "data_from_date": row["Data From Date"],
            "category": row["Category"],
            "currency_multiplier": row["Currency Multiplier"],
            "currency": row["Currency"],
            "exchange_rate": row["Exchange rate"],
            "dollar_equivalent": row["Dollar equivalent"],
            "contract_months": row["Contract Months"],
            "last_contract": row["Last Contract"],
            "tick_value" : row.get("Tick Value", None)
        }
        
        # Populate the nested dictionary with exchange and symbol
        # "f" is used for futures
        result[exchange][symbol]["f"] = future_data

    # Convert defaultdict to normal dict before returning
    # and normal dict to dotaccessdict(DotDict)
    return utils.Dotdict(dict(result))

# this function will cache the result and only run once
@lru_cache(maxsize=1)
def get_tickers():
    return _parse_excel_to_dict()

if __name__ == "__main__":
    pass