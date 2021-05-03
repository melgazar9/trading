from dev.scripts.ML_utils import *
from numerai.dev.scripts.numerai_utils import *
### general params ###

NUM_WORKERS = 32
SPLIT_COLNAME = 'dataset_split'
TARGET = 'target'
PRESERVE_VARS = []
DATE_COL = 'date'
TICKER_COL = 'bloomberg_ticker'
DROP_NA_TARGETS = True
VERBOSE = True
START_DATE = '2005-01-01'

### path params ###

SAVE_FEATURE_TRANSFORMER = False
SAVE_OBJECT = False

LOAD_DATA_FILEPATH = '/media/melgazar9/HDD_10TB/trading/data/numerai/datasets/build_dataset_dfs/df_numerai_build_2021-04-28.feather'
OBJECT_OUTPATH = '/media/melgazar9/HDD_10TB/trading/objects/'

### feature params ###

INPUT_FEATURES_STRING = "list(set([col for col in df_numerai.columns\
                            for prefix in\
                            ['prev', 'pct', 'move', 'minus', 'diff', 'range', 'TARGET_HL3', 'TARGET_HL5']\
                            if prefix in col]))"

PRESERVE_VARS_STRING = "list(set([col for col in df_numerai.columns if not col in input_features]))"

TIMESERIES_SPLIT_PARAMS = {'train_prop': .7,
                           'val_prop': .15,
                           'return_single_df': True,
                           'split_colname': SPLIT_COLNAME,
                           'sort_df_params': {}}





########################################
###### Feature Engineering Params ######
########################################

DROPNA_PARAMS = {'col_contains': ['1d'], 'exception_cols': ['target']}

NAIVE_FEATURES_PARAMS = {'open_col': 'open_1d',
                         'high_col': 'high_1d',
                         'low_col': 'low_1d',
                         'close_col': 'adj_close_1d',
                         'volume_col': 'volume_1d',
                         'new_col_suffix': '_1d',
                         'copy': False}

DIFF_COLS_STRING = "list(set([i for i in df.columns for j in ['move', 'pct', 'chg', 'minus'] if j in i]))"
PCT_CHG_COLS_STRING = "list(set([i for i in df.columns for j in ['move', 'pct', 'chg', 'minus', 'diff'] if j in i]))"

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

# TIMESERIES_GROUPBY_PERIODS = ['7d', '21d', '30d']



""" feature creation pipeline """

DATA_CLEANER_PIPE = Pipeline(\
    steps=[\
            # ('drop_null_yahoo_tickers', FunctionTransformer(lambda df: df.dropna(subset=['yahoo_ticker'], how='any'))),\

            ('dropna_targets', FunctionTransformer(lambda df: df.dropna(subset=[TARGET, 'yahoo_ticker'], how='any'))),\

            ('dropna_cols', FunctionTransformer(lambda df: df.dropna(axis=1, how='any'))),\

            # ('conditional_feature_dropna', FunctionTransformer(lambda df: drop_nas(df, **DROPNA_PARAMS))),\

            ('sort_df', FunctionTransformer(lambda df: df.sort_values(by=[DATE_COL, TICKER_COL])))\
        ]\
)

FEATURE_CREATION_PARAMS = Pipeline(\

    steps=[\

        ('create_targets_HL3', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                            .apply(lambda df: CreateTargets(df, copy=False)\
                                            .create_targets_HL3(**TARGETS_HL3_PARAMS)))),\

        ('create_targets_HL5', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                            .apply(lambda df: CreateTargets(df, copy=False)\
                                            .create_targets_HL5(**TARGETS_HL5_PARAMS)))),\

        ('lagging_features', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                          .apply(lambda df: create_lagging_features(df, **LAGGING_FEATURES_PARAMS)))),\

        ('rolling_features', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                          .apply(lambda df: create_rolling_features(df, **ROLLING_FEATURES_PARAMS)))),\

        ('calc_move_iar', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                       .apply(lambda df: calc_move_iar(df, **IAR_PARAMS)))),\

        ('convert_dtypes', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                        .apply(lambda df: convert_df_dtypes(df, **CONVERT_DTYPE_PARAMS))))\
    ]
)



### Main machine-learning feature engineering pipeline ###

FE_pipeline = {

    'numeric_pipe': make_pipeline(
                        FunctionTransformer(lambda x: x.replace([np.inf, -np.inf], np.nan)),
                        Winsorizer(capping_method='gaussian', tail='both', fold=3, missing_values = 'ignore'),
                        MinMaxScaler(),
                        SimpleImputer(strategy='median', add_indicator=True),
                        ),

     'hc_pipe': make_pipeline(
                         FunctionTransformer(lambda x: x.astype(str)),
                         SimpleImputer(strategy='constant'),
                         TargetEncoder(return_df = True,
                                       handle_missing = 'value',
                                       handle_unknown = 'value',
                                       min_samples_leaf = 10)
                        ),

     'oh_pipe': make_pipeline(
         FunctionTransformer(lambda x: x.astype(str)),
         SimpleImputer(strategy='constant'),
         OneHotEncoder(handle_unknown='ignore')
 )
}

PREPROCESS_FEATURES_PARAMS = {'target': TARGET, 'FE_pipeline_dict': FE_pipeline, 'max_oh_cardinality': 10}


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
    'use_eval_set_when_possible': True
}
