import pandas as pd
from twelvedata import TDClient
import os
import json
import requests



##############################
###### Global Variables ######
##############################

WORKING_DIRECTORY = 'C:/Users/Matt/trading/dev/'
CONFIG_DIRECTORY = WORKING_DIRECTORY + 'configs/'

os.chdir(WORKING_DIRECTORY + 'scripts/')
from dev.scripts.trading_utils import *

os.chdir(CONFIG_DIRECTORY)
from dev.configs.pull_data_config import *

pd.set_option('display.max_columns', 10)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', 3)

td_api = TDClient(**TWELVEDATA_HEADERS)


ts1 = td_api.time_series(
    symbol= ["AAPL", 'SPY', 'TSLA'],
    interval="1min",
    outputsize=5000,
    timezone="America/Chicago",
)

df = ts1.as_pandas() # as pandas df
df.head()

stocks = requests.get(TWELVEDATA_BASE_URL + '/stocks', TWELVEDATA_HEADERS['apikey'])
if json.loads(stocks.content)['status'] == 'ok':
    df_stocks = pd.DataFrame.from_dict(json.loads(stocks.content)['data'])

df_stocks[df_stocks['country'] == 'United States']['type'].value_counts()

etfs = requests.get(TWELVEDATA_BASE_URL + '/etf', TWELVEDATA_HEADERS['apikey'])
df_etfs = pd.DataFrame.from_dict(json.loads(etfs.content)['data'])

stocks_and_etf_symbols = list(df_stocks['symbol']) + list(df_etfs['symbol'])

symbols_to_pull_US = list(df_stocks[df_stocks['country'] == 'United States']['symbol']) + \
                         list(df_etfs[df_etfs['currency'] == 'USD']['symbol'])

# time_series pull using requests

ts = requests.get(TWELVEDATA_BASE_URL + "/time_series?symbol=AAPL&interval=1min&apikey=" + TWELVEDATA_HEADERS['apikey'])