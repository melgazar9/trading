import datetime

### general params ###

VERBOSE = True
DASK_NPARTITIONS=16

DATETIME_COL = 'date'
TICKER_COL = 'bloomberg_ticker'

CREATE_BLOOMBERG_TICKER_FROM_YAHOO = True

SHIFT_TARGET_HL_UP_TO_PRED_FUTURE = False

### download_yfinance_data params ###

DOWNLOAD_NUMERAI_DATA = True

DOWNLOAD_YFINANCE_DATA_PARAMS = {'intervals_to_download': ['1d', '1h'],
                                 'join_method': 'outer',
                                 'max_intraday_lookback_days': 363,
                                 'n_chunks': 500,
                                 'yfinance_params': {'start': '1990-01-01', 'threads': False}}

### target params ###

TARGET_JOIN_METHOD = 'outer' # Do not set this to inner! An inner join will result in the rolling / lagging features not making any sense!
TARGET_JOIN_COLS = [DATETIME_COL, TICKER_COL]
GROUPBY_COLS = 'bloomberg_ticker'

NUMERAI_TARGETS_URL = 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_train_val_bbg.csv'

### path params ###

DF_INIT_FILEPATH = '/media/melgazar9/HDD_10TB/trading/data/yfinance/df_numerai_init_' + str(datetime.datetime.today().date()) + '.feather'
DF_BUILD_FILEPATH = '/media/melgazar9/HDD_10TB/trading/data/yfinance/df_numerai_build_' + str(datetime.datetime.today().date()) + '.feather'

### numerai_competition_data params ###

DOWNLOAD_NUMERAI_COMPETITION_DATA = False
LOAD_NUMERAI_COMPETITION_DATA = False
DF_NUMERAI_COMP_TRAIN_PATH = '/media/melgazar9/HDD_10TB/trading/data/numerai_dataset_255/numerai_training_data.csv' # local

### drop_nas params ###

RUN_CONDITIONAL_DROPNA = False
DROPNA_PARAMS = {'col_contains': ['1d'],
                 'exception_cols': ['target']}

### CreateTargets params ###

TARGETS_HL3_PARAMS = {'buy': 0.03,
                      'sell': 0.03,
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


### lagging_features params ###

LAGGING_FEATURES_PARAMS = {
    'groupby_cols': GROUPBY_COLS,

    'lagging_map': {'target': [1, 2, 3, 4, 5],
                    'target_HL3': [1, 2, 3, 4, 5],
                    'target_HL5': [1, 2, 3, 4, 5],
                    'volume_1d': [1, 2, 3, 4, 5],
                    'adj_close_1d': [1, 2, 3, 4, 5],
                    'move_1d': [1, 2, 3, 4, 5]
                    }
    }


### rolling_features params ###

ROLLING_FEATURES_PARAMS = {'rolling_params' : {'window': 30},
                           'rolling_fn': 'mean',
                           'ewm_fn': 'mean',
                           'ewm_params': {'com':.5},
                           'rolling_cols': ['open_1d', 'high_1d', 'low_1d', 'adj_close_1d', 'volume_1d', 'prev1_target', 'prev1_target_HL5'],
                           'ewm_cols': ['open_1d', 'high_1d', 'low_1d', 'adj_close_1d', 'volume_1d', 'prev1_target', 'prev1_target_HL5'],
                           'join_method': 'outer',
                           'groupby_cols': GROUPBY_COLS,
                           'create_diff_cols': True
                           }

### move_iar params ###

IAR_COLS = ['move_1d', 'high_move_1d', 'low_move_1d']
