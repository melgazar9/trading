import datetime

""" general params """

VERBOSE = True
DASK_NPARTITIONS=16

DATETIME_COL = 'date'
TICKER_COL = 'bloomberg_ticker'

CREATE_BLOOMBERG_TICKER_FROM_YAHOO = True

SHIFT_TARGET_HL_UP_TO_PRED_FUTURE = False

CONVERT_DF_DTYPES = True
CONVERT_DTYPE_PARAMS = {'new_float_dtype': 'float32',
                        'new_int_dtype': 'int8',
                        'new_obj_dtype': 'category',
                        'exclude_cols': ['friday_date', 'bloomberg_ticker', 'data_type']}

""" download_yfinance_data params """

DOWNLOAD_YAHOO_DATA = True

DOWNLOAD_YFINANCE_DATA_PARAMS = {'intervals_to_download': ['1d', '1h'],
                                 'join_method': 'outer',
                                 'max_intraday_lookback_days': 363,
                                 'n_chunks': 500,
                                 'yfinance_params': {'start': '2021-03-01', 'threads': False}}

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

""" drop_nas params """

RUN_CONDITIONAL_DROPNA = False

DROPNA_PARAMS = {'col_contains': ['1d'],
                 'exception_cols': ['target']}


NAIVE_FEATURES_PARAMS = {'open_col': 'open_1d',
                         'high_col': 'high_1d',
                         'low_col': 'low_1d',
                         'close_col': 'adj_close_1d',
                         'volume_col': 'volume_1d',
                         'new_col_suffix': '_1d',
                         'copy': False}

DIFF_PARAMS_STRING = "{'diff_cols': list(set([i for i in df_yahoo.columns for j in ['move', 'pct', 'chg', 'minus'] if j in i])), \
                       'copy': False}"

PCT_CHG_PARAMS_STRING = "{'pct_change_cols': list(set([i for i in df_yahoo.columns for j in ['move', 'pct', 'chg', 'minus', 'diff'] if j in i])),\
                          'copy': False}"

DROP_DUPLICATE_ROWS = False

""" CreateTargets params """

TARGETS_HL3_PARAMS = {'buy': 0.025,
                      'sell': 0.025,
                      'threshold': 0.25,
                      'stop': .01,
                      'move_col': 'move_pct_1d',
                      'lm_col': 'low_move_pct_1d',
                      'hm_col': 'high_move_pct_1d',
                      'target_suffix': 'target_HL3'
                      }

TARGETS_HL5_PARAMS = {'strong_buy': 0.035,
                      'med_buy': 0.015,
                      'med_sell': 0.015,
                      'strong_sell': 0.035,
                      'threshold': 0.25,
                      'stop': .025,
                      'move_col': 'move_pct_1d',
                      'lm_col': 'low_move_pct_1d',
                      'hm_col': 'high_move_pct_1d',
                      'target_suffix': 'target_HL5'
                      }
""" lagging_features params """

LAGGING_FEATURES_PARAMS = {
    # 'groupby_cols': TICKER_COL,

    'lagging_map': {'target': [1, 2, 3, 4, 5],
                    'target_HL3': [1, 2, 3, 4, 5],
                    'target_HL5': [1, 2, 3, 4, 5],
                    'volume_1d': [1, 2, 3, 4, 5],
                    'adj_close_1d': [1, 2, 3, 4, 5],
                    'move_1d': [1, 2, 3, 4, 5]
                    },
    'copy': False
    }


""" rolling_features params """

ROLLING_FEATURES_PARAMS = {'rolling_params' : {'window': 30},
                           'rolling_fn': 'mean',
                           'ewm_fn': 'mean',
                           'ewm_params': {'com':.5},
                           'rolling_cols': ['open_1d', 'high_1d', 'low_1d', 'adj_close_1d', 'volume_1d', 'prev1_target', 'prev1_target_HL5'],
                           'ewm_cols': ['open_1d', 'high_1d', 'low_1d', 'adj_close_1d', 'volume_1d', 'prev1_target', 'prev1_target_HL5'],
                           'join_method': 'outer',
                           # 'groupby_cols': TICKER_COL,
                           'copy': False
                           }

""" move_iar params """

IAR_PARAMS = {'iar_cols': ['move_1d', 'high_move_1d', 'low_move_1d'], 'copy': False}

""" timeseries groupby params """

TIMESERIES_GROUPBY_PERIODS = ['7d', '21d', '30d']
