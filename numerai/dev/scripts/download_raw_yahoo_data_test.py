import os
from configparser import ConfigParser
import sys

import yfinance
from IPython.display import display
from datetime import datetime
import time
import numerapi

start_time = time.time()

if not os.getcwd().endswith('trading'): os.chdir('../../..') # local machine

assert os.getcwd().endswith('trading'), 'Wrong path!'
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

sys.path.append(os.getcwd())
from dev.scripts.ML_utils import * # run if on local machine
from dev.scripts.trading_utils import * # run if on local machine
from numerai.dev.scripts.numerai_utils import *
from numerai.dev.configs.build_numerai_dataset_cfg import *
config = ConfigParser()
config.read('numerai/numerai_keys.ini')

napi = numerapi.SignalsAPI(config['KEYS']['NUMERAI_PUBLIC_KEY'], config['KEYS']['NUMERAI_SECRET_KEY'])

ticker_map = download_ticker_map(napi, 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv', main_ticker_col='bloomberg_ticker')

# download the data
n = 1
chunk_df = [ticker_map['yahoo'].iloc[i:i+n] for i in range(0, len(ticker_map['yahoo']), n)]

column_order = ['Date', 'yahoo_ticker', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

### 1d ###

concat_dfs = []
for df in chunk_df:
    try:
        if n == 1:
            temp_df = yfinance.download(df.iloc[0], start='1990-01-01', threads=False, interval='1d', progress=False).reset_index()
            temp_df['yahoo_ticker'] = df.iloc[0]
            temp_df = temp_df[column_order]
        else:
            temp_df = yfinance.download(df.str.cat(sep=' '), start='1990-01-01', threads=False)
            temp_df = temp_df[[i for i in column_order if i in temp_df.columns]]
        concat_dfs.append(temp_df)
    except simplejson.errors.JSONDecodeError:
        pass
df_yahoo_1d = pd.concat(concat_dfs)
df_yahoo_1d.reset_index(drop=True).to_feather('/media/melgazar9/HDD_10TB/trading/data/numerai/datasets/raw_yahoo_dfs/df_yahoo_1d_' + str(datetime.datetime.today().date()) + '.feather')

print('*** done with 1d! ***')
# 1h
concat_dfs = []
for df in chunk_df:
    try:
        if n == 1:
            temp_df = yfinance.download(df.iloc[0], start=str(datetime.datetime.today().date() - datetime.timedelta(days=363)), threads=False, interval='1h', progress=False).reset_index().rename(columns={'index': 'Date'})
            temp_df['yahoo_ticker'] = df.iloc[0]
            temp_df = temp_df[column_order]
        else:
            temp_df = yfinance.download(df.str.cat(sep=' '), start='1990-01-01', threads=False)
            temp_df = temp_df[[i for i in column_order if i in temp_df.columns]]
        concat_dfs.append(temp_df)
    except simplejson.errors.JSONDecodeError:
        pass
df_yahoo_1h = pd.concat(concat_dfs)
df_yahoo_1h.reset_index(drop=True).to_feather('/media/melgazar9/HDD_10TB/trading/data/numerai/datasets/raw_yahoo_dfs/df_yahoo_1h_' + str(datetime.datetime.today().date()) + '.feather')
