import datetime

""" general params """

VERBOSE = True
DASK_NPARTITIONS=16

DATETIME_COL = 'date'
TICKER_COL = 'bloomberg_ticker'

CREATE_BLOOMBERG_TICKER_FROM_YAHOO = True
DROP_NULL_TICKERS = False
SHIFT_TARGET_HL_UP_TO_PRED_FUTURE = False

CONVERT_DF_DTYPES = False
CONVERT_DTYPE_PARAMS = {'new_float_dtype': 'float32',
                        'new_int_dtype': 'int8',
                        'new_obj_dtype': 'category',
                        'exclude_cols': ['friday_date', 'bloomberg_ticker', 'data_type']}

""" download_yfinance_data params """

DOWNLOAD_VALID_TICKERS_PARAMS = {'numerai_ticker_link': 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv',
                                 'main_ticker_col': TICKER_COL,
                                 'verbose': True}

DOWNLOAD_YAHOO_DATA = True
# 'tickers': ['SPY', 'AAPL', 'AMZN', 'TSLA', 'FB', 'MSFT', 'IWM'],
DOWNLOAD_YFINANCE_DATA_PARAMS = {
                                 'intervals_to_download': ['1d', '1h'],
                                 'join_method': 'outer',
                                 'max_intraday_lookback_days': 363,
                                 'n_chunks': 1, # n_chunks=1 is the most reliable, but slow
                                 # set progress to False in yfinance_params or set verbose = False in the main params to turn off progress bar per download
                                 'yfinance_params': {'start': '1990-01-01', 'threads': False, 'progress': False}}

# the below filepath reads the df into memory if DOWNLOAD_YAHOO_DATA == False
YAHOO_READ_FILEPATH = '/media/melgazar9/HDD_10TB/trading/data/numerai/datasets/build_dataset_dfs/df_numerai_init_2021-04-18.feather'


""" target params """

TARGET_JOIN_METHOD = 'outer' # Do not set this to inner! An inner join will result in the rolling / lagging features not making any sense!
TARGET_JOIN_COLS = [DATETIME_COL, TICKER_COL]

NUMERAI_TARGETS_URL = 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_train_val_bbg.csv'

""" path params """

INIT_SAVE_FILEPATH = '/media/melgazar9/HDD_10TB/trading/data/numerai/datasets/build_dataset_dfs/df_numerai_init_' + str(datetime.datetime.today().date()) + '.feather'
FINAL_SAVE_FILEPATH = '/media/melgazar9/HDD_10TB/trading/data/numerai/datasets/build_dataset_dfs/df_numerai_build_' + str(datetime.datetime.today().date()) + '.feather'

""" numerai_competition_data params """

DOWNLOAD_NUMERAI_COMPETITION_DATA = False
LOAD_NUMERAI_COMPETITION_DATA = False
DF_NUMERAI_COMP_TRAIN_PATH = '/media/melgazar9/HDD_10TB/trading/data/numerai_dataset_255/numerai_training_data.csv' # local
