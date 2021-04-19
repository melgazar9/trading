from dev.scripts.ML_utils import *

NUM_WORKERS = 32

LOAD_DATA_FILEPATH = '/media/melgazar9/HDD_10TB/trading/data/yfinance/df_numerai_build_2021-04-16.feather'

DF_OUTPATH = '/media/melgazar9/HDD_10TB/trading/data/numerai/'

SPLIT_COLNAME = 'data_type'
TARGET = 'target'
PRESERVE_VARS = []

TICKER_COL = 'bloomberg_ticker'

INPUT_FEATURES_STRING = "list(set([col for col in df_numerai.columns\
                            for prefix in\
                            ['prev', 'pct', 'move', 'minus', 'diff', 'range', 'TARGET_HL3', 'TARGET_HL5']\
                            if prefix in col]))"

PRESERVE_VARS_STRING = "list(set([col for col in df_numerai.columns if not col in input_features]))"

TIMESERIES_SPLIT_PARAMS = {'train_prop': .7,
                           'val_prop': .15,
                           'feature_prefixes': [],
                           'feature_suffixes': [],
                           'target_name': 'target',
                           'return_single_df': True,
                           'split_colname': 'dataset_split',
                           'sort_df_params': {}}

FE_pipeline = {

    'numeric_pipe' : make_pipeline(
       Winsorizer(capping_method='gaussian', tail='both', fold=3, missing_values = 'ignore'),
       MinMaxScaler(feature_range = (0,1)),
       SimpleImputer(strategy='median', add_indicator=True)
       ),

     'hc_pipe' : make_pipeline(
                         FunctionTransformer(lambda x: x.astype(str)),
                         SimpleImputer(strategy='constant'),
                         TargetEncoder(return_df = True,
                                       handle_missing = 'value',
                                       handle_unknown = 'value',
                                       min_samples_leaf = 10)
                        ),

     'oh_pipe' : make_pipeline(
         FunctionTransformer(lambda x: x.astype(str)),
         SimpleImputer(strategy='constant'),
         OneHotEncoder(handle_unknown='ignore')
 )
}

PREPROCESS_FEATURES_PARAMS = {'target': TARGET, 'FE_pipeline_dict': FE_pipeline, 'max_oh_cardinality': 10}


algorithms = [
    RandomForestRegressor(n_estimators = 100,
                          max_features = .8,
                          min_samples_leaf = 35,
                          criterion = "mse",
                          n_jobs = NUM_WORKERS,
                          random_state = 100),
    XGBRegressor(colsample_bytree=0.8,
                 gamma=0.01,
                 learning_rate=0.05,
                 max_depth=5,
                 min_child_weight=1,
                 n_estimators=1000,
                 n_jobs=NUM_WORKERS,
                 random_state=0,
                 subsample=0.7)

    ]
