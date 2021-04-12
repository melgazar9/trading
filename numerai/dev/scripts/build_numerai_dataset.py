#####################
###### Imports ######
#####################

import os
from configparser import ConfigParser
import sys

if not os.getcwd().endswith('trading'):
    os.chdir('../../..') # local machine
assert os.getcwd().endswith('trading'), 'Wrong path!'
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

sys.path.append(os.getcwd())
from dev.scripts.ML_utils import * # run if on local machine
from dev.scripts.numerai_utils import *
from dev.scripts.trading_utils import * # run if on local machine
from numerai.dev.scripts.numerai_fns import *
from numerai.dev.configs.build_numerai_dataset_cfg import *

pd.set_option('display.float_format', lambda x: '%.5f' % x)
config = ConfigParser()
config.read('numerai/numerai_keys.ini')

# Download Numerai data
napi = numerapi.SignalsAPI(config['KEYS']['NUMERAI_PUBLIC_KEY'], config['KEYS']['NUMERAI_SECRET_KEY'])

# download data
if DOWNLOAD_NUMERAI_COMPETITION_DATA:
    # napi = numerapi.NumerAPI(NUMERAI_PUBLIC_KEY, NUMERAI_SECRET_KEY)
    napi.download_current_dataset(unzip=True)
if LOAD_NUMERAI_COMPETITION_DATA:
    df_numerai_comp = dd.read_csv(DF_NUMERAI_COMP_TRAIN_PATH).compute()

# Load eligible tickers
eligible_tickers = pd.Series(napi.ticker_universe(), name='ticker')
if VERBOSE: print(f"Number of eligible tickers: {len(eligible_tickers)}")
ticker_map = pd.read_csv('https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv')
ticker_map = ticker_map[ticker_map['bloomberg_ticker'].isin(eligible_tickers)]
if VERBOSE: print(f"Number of eligible tickers in map: {len(ticker_map)}")

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
    df_yahoo = dd.from_pandas(download_yfinance_data(list(ticker_map['yahoo']), start='2006-01-01')) # all valid yahoo tickers
else:

    if DF_YAHOO_FILEPATH.lower().endswith('pq') or DF_YAHOO_FILEPATH.lower().endswith('parquet'):
        df_yahoo = dd.read_parquet(DF_YAHOO_FILEPATH,
                                    DASK_NPARTITIONS=DASK_NPARTITIONS).compute()
        # df_yahoo = pd.read_parquet(DF_YAHOO_FILEPATH)
    elif DF_YAHOO_FILEPATH.lower().endswith('feather'):
        df_yahoo = dd.from_pandas(delayed(feather.read_dataframe)(DF_YAHOO_FILEPATH).compute(),
                                   DASK_NPARTITIONS=DASK_NPARTITIONS).compute()
if VERBOSE: print(df_yahoo.info())

if CREATE_BLOOMBERG_TICKER_FROM_YAHOO:
    df_yahoo.loc[:, 'bloomberg_ticker'] = df_yahoo['yahoo_ticker'].map(dict(zip(ticker_map['yahoo'], ticker_map['bloomberg_ticker'])))


if SAVE_DF_YAHOO_TO_FEATHER:
    df_yahoo.reset_index().to_feather(DF_YAHOO_OUTPATH + '.feather')
if SAVE_DF_YAHOO_TO_PARQUET:
    df_yahoo.to_parquet(DF_YAHOO_OUTPATH + '.pq')

targets = pd.read_csv(NUMERAI_TARGETS_ADDRESS)\
            .assign(date = lambda df: pd.to_datetime(df['friday_date'], format='%Y%m%d'))

if VERBOSE: targets['target'].value_counts(), targets['target'].value_counts(normalize=True)

### Merge targets into ddf_yahoo ###
# - From an inner join on `['date', 'bloomberg_ticker']` we lose about 85% of rows. <br>
# - If we drop rows with NAs we have 0 rows left no matter what. <br>
# - The best bet seems to be an outer join without dropping NA rows.

# By doing an inner join we lose ~85% of the rows
df_yahoo = pd.merge(df_yahoo, targets, on=['date', 'bloomberg_ticker'], how='inner')
df_yahoo.set_index('date', inplace=True)
df_yahoo.sort_index(inplace=True)


if DROP_1D_NAS:
    df_yahoo = drop_suffix_nas(df_yahoo, col_suffix='1d')
if DROP_1H_NAS:
    df_yahoo = drop_suffix_nas(df_yahoo, col_suffix='1h')

# Create naive features
TICKERS = df_yahoo['bloomberg_ticker'].unique().tolist()
df_yahoo = df_yahoo.groupby(GROUPBY_COLS, group_keys=False).apply(create_naive_features_single_symbol)

df_yahoo = CreateTargets(df_yahoo).create_targets_HL3(**TARGETS_HL3_PARAMS)

if VERBOSE: print(df_yahoo['target_HL3'].value_counts()), display(df_yahoo['target_HL3'].value_counts(normalize=True))

df_yahoo = CreateTargets(df_yahoo).create_targets_HL5(**TARGETS_HL5_PARAMS)

if VERBOSE: display(df_yahoo['target_HL5'].value_counts()), display(df_yahoo['target_HL5'].value_counts(normalize=True))


### Create lagging features ###

df_yahoo = create_lagging_features(df_yahoo, groupby_cols=GROUPBY_COLS, lagging_map=LAGGING_MAP)
gc.collect()

### Create rolling features ###

df_yahoo = create_rolling_features(df_yahoo, **ROLLING_FEATURES_PARAMS)


### Save df ###

if DF_YAHOO_OUTPATH.endswith('pq') or DF_YAHOO_OUTPATH.endswith('parquet'):
    df_yahoo.to_parquet(DF_YAHOO_OUTPATH)
elif DF_YAHOO_OUTPATH.endswith('feather'):
    df_yahoo.reset_index(drop=True).to_feather(DF_YAHOO_OUTPATH)
