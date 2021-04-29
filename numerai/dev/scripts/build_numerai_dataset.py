#####################
###### Imports ######
#####################

import os
from configparser import ConfigParser
import sys
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


###  pd options / configs ###

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_columns', 10)
config = ConfigParser()
config.read('numerai/numerai_keys.ini')

### connect to the numerai signals API ###

napi = numerapi.SignalsAPI(config['KEYS']['NUMERAI_PUBLIC_KEY'], config['KEYS']['NUMERAI_SECRET_KEY'])

### Load eligible tickers ###

ticker_map = download_ticker_map(napi, **DOWNLOAD_VALID_TICKERS_PARAMS)

### Download or load in yahoo finance data in the expected numerai format using the yfinance library ###

# Yahoo Finance wrappers: https://github.com/ranaroussi/yfinance and https://pypi.org/project/yfinance/.
# Downloading ~2 hours on a single-thread

if DOWNLOAD_YAHOO_DATA:
    if VERBOSE: print('****** Downloading yfinance data ******')
    if 'tickers' not in DOWNLOAD_YFINANCE_DATA_PARAMS.keys():
        df_yahoo = download_yfinance_data(tickers=ticker_map['yahoo'].tolist(), **DOWNLOAD_YFINANCE_DATA_PARAMS) # all valid yahoo tickers
    else:
        df_yahoo = download_yfinance_data(**DOWNLOAD_YFINANCE_DATA_PARAMS)
else:
    # read in file
    if YAHOO_READ_FILEPATH.lower().endswith('pq') or YAHOO_READ_FILEPATH.lower().endswith('parquet'):
        df_yahoo = dd.read_parquet(YAHOO_READ_FILEPATH, DASK_NPARTITIONS=DASK_NPARTITIONS).compute()
    elif YAHOO_READ_FILEPATH.lower().endswith('feather'):
        df_yahoo = pd.read_feather(YAHOO_READ_FILEPATH)
# df_yahoo = df_yahoo.tail(1000000)# debugging

if VERBOSE: print(df_yahoo.info())
gc.collect()

if CREATE_BLOOMBERG_TICKER_FROM_YAHOO or DOWNLOAD_YAHOO_DATA:
    if 'ticker' in df_yahoo.columns:
        df_yahoo.rename(columns={'ticker': 'yahoo_ticker'}, inplace=True)
    df_yahoo.loc[:, 'bloomberg_ticker'] = df_yahoo['yahoo_ticker'].map(dict(zip(ticker_map['yahoo'], ticker_map['bloomberg_ticker'])))

### Ensure no [DATETIME_COL, TICKER_COL] are duplicated. If so then there is an issue. ###

print('\nvalidating unique date + ticker index...\n')
if DROP_NULL_TICKERS: df_yahoo.dropna(subset=[TICKER_COL], inplace=True)

datetime_ticker_cat = (df_yahoo[DATETIME_COL].astype(str) + ' ' + df_yahoo[TICKER_COL].astype(str)).tolist()
assert len(datetime_ticker_cat) == len(set(datetime_ticker_cat)), 'TICKER_COL and DATETIME_COL do not make a unique index!'
del datetime_ticker_cat

print('\nreading targets...\n')

targets = pd.read_csv(NUMERAI_TARGETS_URL).assign(date=lambda df: pd.to_datetime(df['friday_date'], format='%Y%m%d'))

if VERBOSE: targets['target'].value_counts(), targets['target'].value_counts(normalize=True)

### Merge targets into df_yahoo ###

# - From an inner join on `['date', 'bloomberg_ticker']` we lose about 85% of rows as of 2021-03-30.
# - If we drop rows with NAs we have 0 rows left
# - The best bet seems to be an outer join without dropping NA rows.

print('\nmerging numerai target...\n')

df_yahoo = pd.merge(df_yahoo, targets, on=TARGET_JOIN_COLS, how=TARGET_JOIN_METHOD)

del targets # reduce memory

### save memory ###

if CONVERT_DF_DTYPES:
    print('\nconverting dtypes...\n')
    df_yahoo = convert_df_dtypes(df_yahoo, **CONVERT_DTYPE_PARAMS)

### Save df ###

print('\nsaving df build...\n')
if FINAL_SAVE_FILEPATH.endswith('feather'):
    if 'date' in df_yahoo.index.names or 'ticker' in df_yahoo.index.names:
        df_yahoo.reset_index().to_feather(FINAL_SAVE_FILEPATH)
    else:
        df_yahoo.reset_index(drop=True).to_feather(FINAL_SAVE_FILEPATH)
elif FINAL_SAVE_FILEPATH.endswith('pq') or FINAL_SAVE_FILEPATH.endswith('parquet'):
    df_yahoo.to_parquet(FINAL_SAVE_FILEPATH)

end_time = time.time()

if VERBOSE: print('Script took:', round((end_time - start_time) / 60, 3), 'minutes')
