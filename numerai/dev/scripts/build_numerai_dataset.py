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

### download data ###

if DOWNLOAD_NUMERAI_COMPETITION_DATA:
    # napi = numerapi.NumerAPI(NUMERAI_PUBLIC_KEY, NUMERAI_SECRET_KEY)
    napi.download_current_dataset(unzip=True)
if LOAD_NUMERAI_COMPETITION_DATA:
    df_numerai_comp = dd.read_csv(DF_NUMERAI_COMP_TRAIN_PATH).compute()


### Load eligible tickers ###

eligible_tickers = pd.Series(napi.ticker_universe(), name='ticker')

ticker_map = pd.read_csv('https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv')
ticker_map = ticker_map[ticker_map[TICKER_COL].isin(eligible_tickers)]

if VERBOSE:
    print(f"Number of eligible tickers: {len(eligible_tickers)}")
    print(f"Number of eligible tickers in map: {len(ticker_map)}")

# Remove null / empty tickers from the yahoo tickers
valid_tickers = [i for i in ticker_map['yahoo']
     if not pd.isnull(i)
     and not str(i).lower()=='nan' \
     and not str(i).lower()=='null' \
     and not str(i).lower()==''\
]

if VERBOSE: print('tickers before:', ticker_map.shape) # before removing bad tickers
ticker_map = ticker_map[ticker_map['yahoo'].isin(valid_tickers)]
if VERBOSE: print('tickers after:', ticker_map.shape)

### Download or load in yahoo finance data in the expected numerai format using the yfinance library ###
# Yahoo Finance wrappers: https://github.com/ranaroussi/yfinance and https://pypi.org/project/yfinance/.
# Downloading ~2 hours on a single-thread

if DOWNLOAD_YAHOO_DATA:
    df_yahoo = download_yfinance_data(list(ticker_map['yahoo']), **DOWNLOAD_YFINANCE_DATA_PARAMS) # all valid yahoo tickers
else:
    if DF_YAHOO_FILEPATH.lower().endswith('pq') or DF_YAHOO_FILEPATH.lower().endswith('parquet'):
        df_yahoo = dd.read_parquet(DF_YAHOO_FILEPATH,
                                    DASK_NPARTITIONS=DASK_NPARTITIONS).compute()
    elif DF_YAHOO_FILEPATH.lower().endswith('feather'):
        df_yahoo = dd.from_pandas(delayed(feather.read_dataframe)(DF_YAHOO_FILEPATH).compute(),
                                   npartitions=DASK_NPARTITIONS).compute()

df_yahoo.sort_index(inplace=True)
df_yahoo.reset_index(inplace=True)

if VERBOSE: print(df_yahoo.info())

if CREATE_BLOOMBERG_TICKER_FROM_YAHOO or DOWNLOAD_YAHOO_DATA:
    if ('yahoo_ticker' not in df_yahoo.columns) or ('ticker' in df_yahoo.columns):
        df_yahoo.rename(columns={'ticker': 'yahoo_ticker'}, inplace=True)
    df_yahoo.loc[:, 'bloomberg_ticker'] = df_yahoo['yahoo_ticker'].map(dict(zip(ticker_map['yahoo'], ticker_map['bloomberg_ticker'])))

### Ensure no [DATETIME_COL, TICKER_COL] are duplicated. If so then there is an issue. ###

datetime_ticker_cat = (df_yahoo[DATETIME_COL].astype(str) + ' ' + df_yahoo[TICKER_COL].astype(str)).tolist()

assert len(datetime_ticker_cat) == len(set(datetime_ticker_cat)), 'TICKER_COL and DATETIME_COL do not make a unique index!'
del datetime_ticker_cat

if SAVE_DF_YAHOO_TO_FEATHER:
    df_yahoo.to_feather(DF_YAHOO_OUTPATH)

if SAVE_DF_YAHOO_TO_PARQUET:
    df_yahoo.to_parquet(DF_YAHOO_OUTPATH)

targets = pd.read_csv(NUMERAI_TARGETS_URL)\
            .assign(date=lambda df: pd.to_datetime(df['friday_date'], format='%Y%m%d'))

if VERBOSE: targets['target'].value_counts(), targets['target'].value_counts(normalize=True)



### Merge targets into df_yahoo ###

# - From an inner join on `['date', 'bloomberg_ticker']` we lose about 85% of rows.
# - If we drop rows with NAs we have 0 rows left no matter what.
# - The best bet seems to be an outer join without dropping NA rows.
# df_yahoo.set_index(DATETIME_COL, inplace=True)
# df_yahoo.sort_index(inplace=True)

df_yahoo = pd.merge(df_yahoo, targets, on=TARGET_JOIN_COLS, how=TARGET_JOIN_METHOD)

TICKERS = df_yahoo[TICKER_COL].unique().tolist()

### conditionally drop NAs ###

if RUN_CONDITIONAL_DROPNA:
    df_yahoo = drop_nas(df_yahoo, **DROPNA_PARAMS)


### create naive features ###

df_yahoo = df_yahoo.groupby(GROUPBY_COLS, group_keys=False).apply(create_naive_features_single_symbol) # Create naive features (e.g. moves, ranges, etc...)

### create manual targets ###

df_yahoo = CreateTargets(df_yahoo).create_targets_HL3(**TARGETS_HL3_PARAMS) # create target_HL3
df_yahoo = CreateTargets(df_yahoo).create_targets_HL5(**TARGETS_HL5_PARAMS) # create target_HL5


if VERBOSE:
    display(df_yahoo[TARGETS_HL3_PARAMS['target_suffix']].value_counts()), display(df_yahoo[TARGETS_HL3_PARAMS['target_suffix']].value_counts(normalize=True))
    display(df_yahoo[TARGETS_HL5_PARAMS['target_suffix']].value_counts()), display(df_yahoo[TARGETS_HL5_PARAMS['target_suffix']].value_counts(normalize=True))

### For each ticker, shift the target backwards one timestamp, where each row is the unit of measure (e.g. each row is a day) ###

if SHIFT_TARGET_HL_UP_TO_PRED_FUTURE:
    df_yahoo[TARGETS_HL3_PARAMS['target_suffix']] = df_yahoo.groupby(GROUPBY_COLS)[TARGETS_HL3_PARAMS['target_suffix']].transform(lambda col: col.shift(-1))
    df_yahoo[TARGETS_HL5_PARAMS['target_suffix']] = df_yahoo.groupby(GROUPBY_COLS)[TARGETS_HL5_PARAMS['target_suffix']].transform(lambda col: col.shift(-1))

### Create lagging features ###

df_yahoo = create_lagging_features(df_yahoo, **LAGGING_FEATURES_PARAMS)

gc.collect()

### Create rolling features ###

df_yahoo = create_rolling_features(df_yahoo, **ROLLING_FEATURES_PARAMS)

### Create move_iar features ###

df_yahoo = df_yahoo.groupby(GROUPBY_COLS).apply(lambda df: calc_move_iar(df, iar_cols=IAR_COLS))


### Save df ###

if DF_YAHOO_OUTPATH.endswith('pq') or DF_YAHOO_OUTPATH.endswith('parquet'):
    df_yahoo.to_parquet(DF_YAHOO_OUTPATH)
elif DF_YAHOO_OUTPATH.endswith('feather'):
    df_yahoo.reset_index(drop=True).to_feather(DF_YAHOO_OUTPATH)

end_time = time.time()

if VERBOSE: print('Script took:', round((end_time - start_time) / 60, 3), 'minutes')
