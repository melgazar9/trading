import datetime

DOWNLOAD_YAHOO_DATA = False

DOWNLOAD_NUMERAI_COMPETITION_DATA = False
LOAD_NUMERAI_COMPETITION_DATA = False
DF_NUMERAI_COMP_TRAIN_PATH = '/media/melgazar9/HDD_10TB/trading/data/numerai_dataset_255/numerai_training_data.csv' # local

VERBOSE = False

DF_YAHOO_FILEPATH = '/media/melgazar9/HDD_10TB/trading/data/yfinance/df_yahoo_2021-04-07.pq'
DF_YAHOO_OUTPATH = '/media/melgazar9/HDD_10TB/trading/data/numerai/df_numerai_' + str(datetime.datetime.today().date()) + '.feather'
DASK_NPARTITIONS=16

CREATE_BLOOMBERG_TICKER_FROM_YAHOO = False
SAVE_DF_YAHOO_TO_FEATHER = False
SAVE_DF_YAHOO_TO_PARQUET = False


NUMERAI_TARGETS_ADDRESS = 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_train_val_bbg.csv'

DROP_1D_NAS = False
DROP_1H_NAS = False
GROUPBY_COLS = 'bloomberg_ticker'



TARGETS_HL3_PARAMS = {'buy': 0.03,
                      'sell': 0.03,
                      'threshold': 0.25,
                      'stop': .01,
                      'move_col': 'move_pct_1d',
                      'lm_col': 'low_move_pct_1d',
                      'hm_col': 'high_move_pct_1d'
                      }

TARGETS_HL5_PARAMS = {'strong_buy': 0.035,
                      'med_buy': 0.015,
                      'med_sell': 0.015,
                      'strong_sell': 0.035,
                      'threshold': 0.25,
                      'stop': .025,
                      'move_col': 'move_pct_1d',
                      'lm_col': 'low_move_pct_1d',
                      'hm_col': 'high_move_pct_1d'
                      }

LAGGING_MAP = {'target': [1, 2, 3, 4, 5],
               'target_HL3': [1, 2, 3, 4, 5],
               'target_HL5': [1, 2, 3, 4, 5],
               'volume_1d': [1, 2, 3, 4, 5],
               'adj_close_1d': [1, 2, 3, 4, 5],
               'move_1d':[1, 2, 3, 4, 5]
               }


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
