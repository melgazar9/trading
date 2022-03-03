from dev.scripts.ML_utils import *
from numerai.dev.scripts.numerai_utils import *
from numerai.dev.configs.prep_and_train_cfg import *

### general params ###

NUM_WORKERS = 32
SPLIT_COLNAME = 'dataset_split'
TARGET_COL = 'target_20d'
PRESERVE_VARS = ['friday_date', 'date_localized', 'data_type', 'dataset_split']
DATE_COL = 'date'
TICKER_COL = 'bloomberg_ticker'
DROP_NA_TARGETS = True
VERBOSE = True
START_DATE = '2015-01-01'

### path params ###

SAVE_OBJECT = True
OBJECT_OUTPATH = '/media/melgazar9/HDD_10TB/trading/objects/' # 'D:/trading/objects/'
LOAD_DATA_FILEPATH = '/media/melgazar9/HDD_10TB/trading/data/numerai/datasets/processed_data/df_numerai_build_2022-02-19.feather' # 'D:/trading/data/numerai/datasets/processed_data/df_numerai_build_2022-02-19.feather' # windows

### feature params ###

INPUT_FEATURES_STRING = """list(set([col for col in df_numerai.columns
                                    for name in
                                    ['ticker', 'prev', 'pct', 'move', 'minus', 'diff', 'range', 'volume', 'day', 'end', 'is_', 'start', 'month', 'quarter', 'week', 'TARGET_HL3', 'TARGET_HL5']
                                    if name in col]))"""

PRESERVE_VARS_STRING = "list(set([col for col in df_numerai.columns if col not in input_features and ('target' in col or 'open' in col or 'close' in col or 'high' in col or 'low' in col)]))"

DROP_VARS_STRING = "list(set([col for col in df_numerai.columns if col not in input_features and col not in preserve_vars]))"


TIMESERIES_SPLIT_PARAMS = {'train_prop': .7,
                           'val_prop': .15,
                           'return_single_df': True,
                           'split_colname': SPLIT_COLNAME,
                           'sort_df_params': {}}


CONVERT_DTYPE_PARAMS = {'verbose': False}

########################################
###### Feature Engineering Params ######
########################################

DROPNA_PARAMS = {'col_contains': ['1d'], 'exception_cols': [TARGET_COL]}

NAIVE_FEATURES_PARAMS = {'symbol_sep': '',
                         'loop1': {'open_col': 'open_1d', 'high_col': 'high_1d', 'low_col': 'low_1d', 'close_col': 'adj_close_1d', 'volume_col': 'volume_1d'},
                         'loop2': {'open_col': 'open_1h_0', 'high_col': 'high_1h_0', 'low_col': 'low_1h_0', 'close_col': 'adj_close_1h_0', 'volume_col': 'volume_1h_0'}
                         }



DROP_DUPLICATE_ROWS = False

""" CreateTargets params """

TARGETS_HL3_PARAMS = {'buy': 0.025,
                      'sell': 0.025,
                      'stop': .01,
                      'lm_col': 'low_move_pct_1d',
                      'hm_col': 'high_move_pct_1d',
                      'target_suffix': 'target_HL3'
                      }

TARGETS_HL5_PARAMS = {'strong_buy': 0.035,
                      'med_buy': 0.015,
                      'med_sell': 0.015,
                      'strong_sell': 0.035,
                      'stop': .025,
                      'lm_col': 'low_move_pct_1d',
                      'hm_col': 'high_move_pct_1d',
                      'target_suffix': 'target_HL5'
                      }

""" lagging_features params """

LAGGING_FEATURES_PARAMS = {
    # 'groupby_cols': TICKER_COL,

    'lagging_map': {TARGET_COL: [1, 2, 3, 4, 5],
                    'target_HL3': [1, 2, 3, 4, 5],
                    'target_HL5': [1, 2, 3, 4, 5],
                    'volume_1d': [1, 2, 3, 4, 5],
                    'adj_close_1d': [1, 2, 3, 4, 5],
                    'move_1d': [1, 2, 3, 4, 5]
                    },
    'copy': False
    }


""" rolling_features params """

DROP_INIT_ROLLING_WINDOW_ROWS = True

ROLLING_FEATURES_PARAMS = {'rolling_params' : {'window': 30},
                           'rolling_fn': 'mean',
                           'ewm_fn': 'mean',
                           'ewm_params': {'com': .5},
                           'join_method': 'outer',
                           'index_cols': [DATE_COL, TICKER_COL],
                           'copy': False
                           }

DIFF_COLS_TO_SELECT_STRING = "[i for i in df_numerai.columns if i not in [DATE_COL, TICKER_COL, TARGET_COL] and not ('dat' in i.lower() or 'target' in i.lower() or 'ticker' in i.lower())]"
PCT_CHG_COLS_TO_SELECT_STRING = "[i for i in df_numerai.columns if i not in [DATE_COL, TICKER_COL, TARGET_COL] and not ('dat' in i.lower() or 'target' in i.lower() or 'ticker' in i.lower())]"



""" feature creation pipeline """

# this will be done in the main source code until I can figure out a better way to make it easily configurable

# FEATURE_CREATION_PARAMS = Pipeline(\
#     steps=[\
#
#         # ('create_naive_features', FunctionTransformer(lambda df: CreateFeatures(**NAIVE_FEATURES_PARAMS).compute_naive_features(use_symbol_prefix=False))),
#
#         ('create_targets_HL3', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
#                                             .apply(lambda df: CreateTargets(df, copy=False)\
#                                             .create_targets_HL3(**TARGETS_HL3_PARAMS)))),\
#
#         ('create_targets_HL5', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
#                                             .apply(lambda df: CreateTargets(df, copy=False)\
#                                             .create_targets_HL5(**TARGETS_HL5_PARAMS)))),\
#
#         ('lagging_features', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
#                                           .apply(lambda df: create_lagging_features(df, **LAGGING_FEATURES_PARAMS)))),\
#
#         ('rolling_features', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
#                                           .apply(lambda df: create_rolling_features(df, **ROLLING_FEATURES_PARAMS)))),\
#
#         ('calc_move_iar', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
#                                        .apply(lambda df: calc_move_iar(df, **IAR_PARAMS)))),\
#
#         ('convert_dtypes', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
#                                         .apply(lambda df: convert_df_dtypes(df, **CONVERT_DTYPE_PARAMS))))\
#         ]
#     )

### Main machine-learning feature engineering pipeline ###

FE_pipeline = {

    'numeric_pipe': make_pipeline(
                        FunctionTransformer(lambda df: df.replace([np.inf, -np.inf], np.nan)),
                        # Winsorizer(capping_method='gaussian', tail='both', fold=3, missing_values='ignore')
                        # StandardScaler()
                        SimpleImputer(strategy='median', add_indicator=True)
                        ),

     'hc_pipe': make_pipeline(
                         FunctionTransformer(lambda x: x.astype(str)),
                         SimpleImputer(strategy='constant'),
                         TargetEncoder(return_df=True,
                                       handle_missing='value',
                                       handle_unknown='value',
                                       min_samples_leaf=10)
                        ),

     'oh_pipe': make_pipeline(
         FunctionTransformer(lambda x: x.astype(str)),
         SimpleImputer(strategy='constant'),
         OneHotEncoder(drop='if_binary', sparse=False, handle_unknown='ignore')
     ),

     'custom_pipeline': {Pipeline(steps=[('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]): [TICKER_COL]}
    }

PREPROCESS_FEATURES_PARAMS = {'target': TARGET_COL, 'FE_pipeline_dict': FE_pipeline, 'max_oh_cardinality': 10, 'n_jobs': 16, 'copy': False}

""" move_iar params """

IAR_PARAMS = {'iar_cols': ['move_1d', 'high_move_1d', 'low_move_1d'], 'copy': False}

""" timeseries groupby params """

# TIMESERIES_GROUPBY_PERIODS = ['7d', '21d', '30d']



### machine learning params ###

# ALGORITHMS = [
#     RandomForestRegressor(n_estimators= 100,
#                           max_features= .8,
#                           min_samples_leaf= 35,
#                           criterion= "mse",
#                           n_jobs= NUM_WORKERS,
#                           random_state= 100),
#     XGBRegressor(colsample_bytree=0.8,
#                  gamma=0.01,
#                  learning_rate=0.05,
#                  max_depth=5,
#                  min_child_weight=1,
#                  n_estimators=1000,
#                  n_jobs=NUM_WORKERS,
#                  random_state=0,
#                  subsample=0.7)
#     ]

ALGORITHM = XGBRegressor(colsample_bytree=0.8,
                         gamma=0.01,
                         learning_rate=0.05,
                         max_depth=5,
                         min_child_weight=1,
                         n_estimators=100,
                         n_jobs=NUM_WORKERS,
                         random_state=0,
                         subsample=0.7)

RUN_MODEL_PARAMS = {
    'copy': True,
    'prediction_colname': type(ALGORITHM).__name__ + '_pred',
    'seed': 100,
    'convert_float32': False,
    'bypass_all_numeric': False,
    'map_predictions_to_df_full': True,
    'predict_proba': False,
    'use_eval_set_when_possible': True,
    'fit_params': {'early_stopping': 5}
}
