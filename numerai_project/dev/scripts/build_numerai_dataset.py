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
from numerai_project.dev.scripts.numerai_utils import *
from numerai_project.dev.configs.build_numerai_dataset_cfg import *


###  pd options / configs ###

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_columns', 10)
config = ConfigParser()
config.read('numerai_project/numerai_keys.ini')

### connect to the numerai signals API ###

napi = numerapi.SignalsAPI(config['KEYS']['NUMERAI_PUBLIC_KEY'], config['KEYS']['NUMERAI_SECRET_KEY'])

if napi.HISTORICAL_DATA_URL != NUMERAI_TARGETS_URL or napi.HISTORICAL_DATA_URL != DOWNLOAD_VALID_TICKERS_PARAMS['numerai_ticker_link']:
    warnings.warn("napi.HISTORICAL_DATA_URL is not equal to config set HISTORICAL_DATA_URL")

### Load eligible tickers ###

ticker_map = download_ticker_map(napi, **DOWNLOAD_VALID_TICKERS_PARAMS)

if APPEND_OLD_DATA:
    # load in data
    if OLD_FULL_NUMERAI_BUILD_FILEPATH.lower().endswith('pq') or OLD_FULL_NUMERAI_BUILD_FILEPATH.lower().endswith(
            'parquet'):
        df_numerai_old = dd.read_parquet(OLD_FULL_NUMERAI_BUILD_FILEPATH, DASK_NPARTITIONS=DASK_NPARTITIONS).compute()
    elif OLD_FULL_NUMERAI_BUILD_FILEPATH.lower().endswith('feather'):
        df_numerai_old = pd.read_feather(OLD_FULL_NUMERAI_BUILD_FILEPATH)

    DOWNLOAD_YFINANCE_DATA_PARAMS['yfinance_params']['start'] = str(df_numerai_old[DATETIME_COL].max().date())
    df_numerai_old.set_index([TICKER_COL, DATETIME_COL], inplace=True)


### Download or load in yahoo finance data in the expected numerai format using the yfinance library ###

# Yahoo Finance wrappers: https://github.com/ranaroussi/yfinance and https://pypi.org/project/yfinance/.
# Downloading ~2 hours on a single-thread

if DOWNLOAD_YAHOO_DATA:

    if VERBOSE: print('****** Downloading yfinance data ******')

    # DOWNLOAD_YFINANCE_DATA_PARAMS['tickers'] = ['FB', 'AMZN', 'AAPL', 'TSLA', 'MSFT', 'NVDA'] # for debugging

    if 'tickers' not in DOWNLOAD_YFINANCE_DATA_PARAMS.keys():
        DOWNLOAD_YFINANCE_DATA_PARAMS['tickers'] = ticker_map['yahoo'].tolist() # all valid yahoo finance tickers

    dfs = download_yfinance_data(**DOWNLOAD_YFINANCE_DATA_PARAMS)

else:
    # load in data
    if YAHOO_READ_FILEPATH.lower().endswith('pq') or YAHOO_READ_FILEPATH.lower().endswith('parquet'):
        df_yahoo = dd.read_parquet(YAHOO_READ_FILEPATH, DASK_NPARTITIONS=DASK_NPARTITIONS).compute()
    elif YAHOO_READ_FILEPATH.lower().endswith('feather'):
        df_yahoo = pd.read_feather(YAHOO_READ_FILEPATH)
    elif YAHOO_READ_FILEPATH.lower().lower().endswith('pkl'):
        dfs = dill.load(open(YAHOO_READ_FILEPATH, 'rb'))

### save dfs after initial download for backup ###

if DOWNLOAD_YAHOO_DATA and INIT_SAVE_FILEPATH is not None and (INIT_SAVE_FILEPATH.lower().endswith('pkl') or INIT_SAVE_FILEPATH.lower().endswith('pickle')):
    dill.dump(dfs, open(INIT_SAVE_FILEPATH, 'wb'))

if ('dfs' in locals()):

    ### join the dfs in the dfs dictionary ###

    # if NUM_WORKERS == 1 or os.name == 'nt': # unsurprisingly, multiprocessing does not work on Windows
    for i in dfs.keys():
        dfs[i].sort_values(by=DATETIME_COL, inplace=True)

    gc.collect()

    index_flatten_col = DATETIME_COL + '_localized' if FLATTEN_GRANULAR_DATA_PARAMS['flatten_localized'] else DATETIME_COL
    dfs['1d'][index_flatten_col] = pd.to_datetime(dfs['1d'][index_flatten_col], utc=True)\
                                     .dt.tz_convert(inspect.signature(download_yfinance_data)\
                                                    .parameters['tz_localize_location']\
                                                    .default)

    if FLATTEN_GRANULAR_DATA_PARAMS:

        for i in FLATTEN_GRANULAR_DATA_PARAMS['timeseries_to_flatten']:

            if 'index' in dfs[i].columns: dfs[i].drop('index', axis=1, inplace=True)

            dfs[i] = dfs[i].pivot_table(index=[pd.to_datetime(dfs[i][index_flatten_col], utc=True)\
                                                 .dt.tz_convert(inspect.signature(download_yfinance_data)\
                                                    .parameters['tz_localize_location']\
                                                    .default).dt.date, YAHOO_TICKER_COL],
                                        columns=[pd.to_datetime(dfs[i][index_flatten_col], utc=True)\
                                                   .dt.tz_convert(inspect.signature(download_yfinance_data)\
                                                   .parameters['tz_localize_location']\
                                                   .default).dt.hour],
                                        aggfunc=FLATTEN_GRANULAR_DATA_PARAMS['aggfunc'],
                                        values=[i for i in dfs[i].columns if not i in [index_flatten_col, YAHOO_TICKER_COL]])

            dfs[i].columns = list(pd.Index([str(e[0]).lower() + '_' + str(e[1]).lower() for e in dfs[i].columns.tolist()]).str.replace(' ', '_'))
            dfs[i].reset_index(inplace=True)
            dfs[i][index_flatten_col] = pd.to_datetime(dfs[i][index_flatten_col], utc=True)\
                                          .dt.tz_convert(inspect.signature(download_yfinance_data)\
                                                                .parameters['tz_localize_location']\
                                                                .default)

    df_yahoo = reduce(lambda df1, df2: \
                          JOIN_DFS_PARAMS['join_function'](df1,
                                                           df2,
                                                           **{k: JOIN_DFS_PARAMS[k] for k in
                                                              [x for x in JOIN_DFS_PARAMS.keys() if
                                                               not x == 'join_function']}),
                      list(dfs.values()))
gc.collect()

if VERBOSE: print(df_yahoo.info())
gc.collect()


### test if [yahoo_ticker_col + datetime] makes a unique index ###

datetime_ticker_cat_init = (df_yahoo[DATETIME_COL].astype(str) + ' ' + df_yahoo[YAHOO_TICKER_COL].astype(str)).tolist()

if VERBOSE: print('datetime_ticker_cat before: ' + str(len(datetime_ticker_cat_init)))

groupby_params = {'observed': True}

if GROUPBY_TICKER_DATE_AFTER_DOWNLOAD and len(datetime_ticker_cat_init) != len(set(datetime_ticker_cat_init)):

    gc.collect()
    if N_GROUPBY_CHUNKS > 1:

        # since the above two approaches kill the memory, I will chunk the groupby into yahoo_ticker groups
        unique_tickers = df_yahoo[YAHOO_TICKER_COL].unique()
        ticker_chunks = [i for i in split_list(lst=unique_tickers, n=int(len(unique_tickers) / N_GROUPBY_CHUNKS))][0]

        list_of_dfs = []

        for chunk in ticker_chunks:
            list_of_dfs.append(df_yahoo[df_yahoo[YAHOO_TICKER_COL].isin(chunk)]\
                               .groupby([DATETIME_COL, YAHOO_TICKER_COL], **groupby_params)\
                               .first()\
                               .reset_index())

        df_yahoo = pd.concat(list_of_dfs, axis=0)
        del list_of_dfs

    else:
        df_yahoo = df_yahoo.groupby(by=[DATETIME_COL, YAHOO_TICKER_COL], **groupby_params).first().reset_index()

del datetime_ticker_cat_init



### create bloomberg ticker ###

if CREATE_BLOOMBERG_TICKER_FROM_YAHOO or DOWNLOAD_YAHOO_DATA:
    if 'ticker' in df_yahoo.columns:
        df_yahoo.rename(columns={'ticker': YAHOO_TICKER_COL}, inplace=True)
    df_yahoo.loc[:, 'bloomberg_ticker'] = df_yahoo[YAHOO_TICKER_COL].map(dict(zip(ticker_map['yahoo'], ticker_map['bloomberg_ticker'])))




### ensure no [DATETIME_COL, TICKER_COL] are duplicated ###

# If there are any duplicates in datetime_ticker_cat_after at this point, and when GROUPBY_TICKER_DATE_AFTER_DOWNLOAD is set to True),then there is a bug in at least one of the above functions

if VERBOSE: print('\nvalidating unique date + ticker index...\n')

if DROP_NULL_TICKERS: df_yahoo.dropna(subset=[TICKER_COL], inplace=True)

datetime_ticker_cat_after = (df_yahoo[DATETIME_COL].astype(str) + ' ' + df_yahoo[TICKER_COL].astype(str)).tolist()
assert len(datetime_ticker_cat_after) == len(set(datetime_ticker_cat_after)), 'TICKER_COL and DATETIME_COL do not make a unique index!'
del datetime_ticker_cat_after

gc.collect()
if VERBOSE: print('\nreading targets...\n')

targets = pd.read_csv(NUMERAI_TARGETS_URL).assign(date=lambda df: pd.to_datetime(df['friday_date'], format='%Y%m%d'))

if VERBOSE:
    for t in NUMERAI_TARGET_NAMES:
        print('\n', targets[t].value_counts(),'\n')
        print('\n', targets[t].value_counts(normalize=True), '\n')



### Merge targets into df_yahoo ###

# - From an inner join on `['date', 'bloomberg_ticker']` we lose about 85% of rows as of 2021-03-30.
# - If we drop rows with NAs we have 0 rows left
# - The best bet seems to be an outer join without dropping NA rows.

if VERBOSE: print('\nmerging numerai target...\n')

df_yahoo = pd.merge(df_yahoo, targets, on=TARGET_JOIN_COLS, how=TARGET_JOIN_METHOD)
del targets

df_yahoo.dropna(axis=0, how='all',
                subset=[i for i in df_yahoo.columns if not i in [DATETIME_COL, TICKER_COL, YAHOO_TICKER_COL] + NUMERAI_TARGET_NAMES + ['friday_date', 'data_type']],
                inplace=True)
gc.collect()

### save memory ###

if CONVERT_DF_DTYPES:
    if VERBOSE: print('\nconverting dtypes...\n')
    df_yahoo = convert_df_dtypes(df_yahoo, **CONVERT_DTYPE_PARAMS)

### append the new df to the previous df ###

if APPEND_OLD_DATA:
    if VERBOSE: print('\nAppending old data...\n')
    df_yahoo.set_index([TICKER_COL, DATETIME_COL], inplace=True)
    matching_cols = np.intersect1d(df_numerai_old.columns, df_yahoo.columns)
    new_missing_cols = list(set(df_numerai_old.columns) - set(df_yahoo.columns))
    old_missing_cols = list(set(df_yahoo.columns) - set(df_numerai_old.columns))
    df_numerai_old.loc[:, old_missing_cols] = np.nan
    df_yahoo.loc[:, new_missing_cols] = np.nan
    assert len(np.intersect1d(df_yahoo.columns, df_numerai_old.columns)) == len(df_yahoo.columns)
    assert len(np.intersect1d(df_yahoo.columns, df_numerai_old.columns)) == len(df_numerai_old.columns)
    old_indices_to_keep = [i for i in df_numerai_old.index if i not in df_yahoo.index]
    assert(len(old_indices_to_keep) == len(set(old_indices_to_keep))), 'Old indices do not make a unique index!'
    df_numerai_old = df_numerai_old.loc[old_indices_to_keep][matching_cols]
    matching_cols = list(set([TICKER_COL, DATETIME_COL] + list(matching_cols)))
    df_yahoo.reset_index(inplace=True)
    df_numerai_old.reset_index(inplace=True)
    df_numerai_old = df_numerai_old[matching_cols]
    df_yahoo = df_yahoo[matching_cols]
    df_yahoo = pd.concat([df_numerai_old, df_yahoo], axis=0)

    ### test if [yahoo_ticker_col + datetime] makes a unique index ###

    datetime_ticker_cat_final = (df_yahoo[DATETIME_COL].astype(str) + ' ' + df_yahoo[TICKER_COL].astype(str)).tolist()

    if VERBOSE: print('datetime_ticker_cat_final: ' + str(len(datetime_ticker_cat_final)))
    assert(len(datetime_ticker_cat_final) == len(set(datetime_ticker_cat_final))), 'The final output index is not unique!'
    del datetime_ticker_cat_final

### Save df ###

if VERBOSE: print('\nsaving df build...\n')

gc.collect()

if FINAL_SAVE_FILEPATH.endswith('feather'):
    if 'date' in df_yahoo.index.names or YAHOO_TICKER_COL in df_yahoo.index.names:
        df_yahoo.reset_index(inplace=True)
        df_yahoo.to_feather(FINAL_SAVE_FILEPATH)
    else:
        df_yahoo.reset_index(drop=True).to_feather(FINAL_SAVE_FILEPATH)
elif FINAL_SAVE_FILEPATH.endswith('pq') or FINAL_SAVE_FILEPATH.endswith('parquet'):
    df_yahoo.to_parquet(FINAL_SAVE_FILEPATH)

end_time = time.time()

if VERBOSE: print('Script took:', round((end_time - start_time) / 60, 3), 'minutes')
