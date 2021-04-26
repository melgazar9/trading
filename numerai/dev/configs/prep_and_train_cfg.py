from dev.scripts.ML_utils import *

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

LOAD_DATA_FILEPATH = '/media/melgazar9/HDD_10TB/trading/data/numerai/datasets/build_dataset_dfs/df_numerai_build_2021-04-24.feather'
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


### feature transformation params ###

FE_pipeline = {

    'custom_fns': make_pipeline(

        FunctionTransformer(lambda df: drop_nas(df, **DROPNA_PARAMS)), # drop_nas
        # FunctionTransformer(lambda df: drop_duplicates(df)), # drop_duplicates
        FunctionTransformer(lambda df: df.sort_values(by=[DATETIME_COL, TICKER_COL])), # sort values

        ### create naive features ###
        FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                         .apply(lambda df: create_naive_features(df, **NAIVE_FEATURES_PARAMS))),
        ### calc_diffs ###
        # FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False) \
        #                     .apply(lambda df: calc_diffs(df, **diff_params)),

        ### calc_pct_chg ###
        # FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False) \
        #                     .apply(lambda df: calc_pct_changes(df, **pct_change_params)),

        ### CreateTargets ###
        FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                         .apply(lambda df: CreateTargets(df, copy=False)\
                                         .create_targets_HL3(**TARGETS_HL3_PARAMS))),

        FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                         .apply(lambda df: CreateTargets(df, copy=False)\
                                         .create_targets_HL3(**TARGETS_HL5_PARAMS))),

        ### lagging features ###
        FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                         .apply(lambda df: create_lagging_features(df, **LAGGING_FEATURES_PARAMS))),

        ### rolling features ###
        FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False)\
                                         .apply(lambda df: create_rolling_features(df, **ROLLING_FEATURES_PARAMS))),

        ### calc_move_iar ###
        FunctionTransformer(lambda df: df_yahoo.groupby(TICKER_COL, group_keys=False)\
                                               .apply(lambda df: calc_move_iar(df, **IAR_PARAMS))),

        ### convert dtypes ###
        FunctionTransformer(lambda df: df_yahoo.groupby(TICKER_COL, group_keys=False)\
                                                .apply(lambda df: convert_df_dtypes(df_yahoo, **CONVERT_DTYPE_PARAMS)))
    ),

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
