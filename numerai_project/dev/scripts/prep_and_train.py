#####################
###### Imports ######
#####################

import os
from configparser import ConfigParser
import sys
import re
if not os.getcwd().endswith('trading'): os.chdir('../../..') # local machine
assert os.getcwd().endswith('trading'), 'Wrong path!'
import numerapi
from collections import Counter
from pandarallel import pandarallel  # parallel pandas

import platform
import time

sys.path.append(os.getcwd())

from dev.scripts.ML_utils import * # run if on local machine
from dev.scripts.trading_utils import * # run if on local machine
from numerai_project.dev.scripts.numerai_utils import *
from numerai_project.dev.configs.prep_and_train_cfg import *

os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

start_time = time.time()

pd.set_option('display.float_format', lambda x: '%.5f' % x, 'display.max_columns', 7)

config = ConfigParser()
config.read('numerai_project/numerai_keys.ini')

# Connect to the Numerai API
napi = numerapi.SignalsAPI(config['KEYS']['NUMERAI_PUBLIC_KEY'], config['KEYS']['NUMERAI_SECRET_KEY'])


### Load in the data created from build_numerai_dataset.py ###

if LOAD_DATA_FILEPATH.endswith('feather'):
    df_numerai = pd.read_feather(LOAD_DATA_FILEPATH)
elif LOAD_DATA_FILEPATH.endswith('pq') or LOAD_DATA_FILEPATH.endswith('parquet'):
    df_numerai = pd.read_parquet(LOAD_DATA_FILEPATH)
elif LOAD_DATA_FILEPATH.endswith('csv'):
    df_numerai = pd.read_csv(LOAD_DATA_FILEPATH)


### dropnas ###

if DROP_NA_TARGETS:
    df_numerai.dropna(subset=[TARGET_COL], inplace=True)


if START_DATE:
    df_numerai = df_numerai[df_numerai[DATE_COL] >= START_DATE]

gc.collect()

if 'data_type' in df_numerai.columns and USE_NUMERAI_TRAIN_TEST_SPLIT:
    vc = df_numerai['data_type'].value_counts()
    assert ('training' in vc.index or 'train' in vc.index) and ('val' in vc.index or 'validation' in vc.index)


###########################################
###### feature manipulation pipeline ######
###########################################

# 1. data cleaning
# 2. feature creation
# 3. feature transformation

basic_move_params = merge_dicts(
    {'loop1_1d': {'open_col': 'open_1d',
                  'high_col': 'high_1d',
                  'low_col': 'low_1d',
                  'close_col': 'close_1d',
                  'volume_col': 'volume_1d',
                  'suffix': '_1d'}},

    {'loop' + str(i) + '_1h': {
        'open_col': 'open_1h_' + i,
        'high_col': 'high_1h_' + i,
        'low_col': 'low_1h_' + i,
        'close_col': 'close_1h_' + i,
        'volume_col': 'volume_1h_' + i,
        'suffix': '_1h_' + i} for i in
        [str(i) for i in range(0, 24)
         if 'open_1h_' + str(i) in df_numerai.columns
         and 'high_1h_' + str(i) in df_numerai.columns
         and 'low_1h_' + str(i) in df_numerai.columns
         and 'close_1h_' + str(i) in df_numerai.columns
         and 'volume_1h_' + str(i) in df_numerai.columns]
    }
)

# df_numerai = df_numerai.tail(300000) # for debugging purposes

FEATURE_CREATOR_PIPE = Pipeline(
    steps=[
        ('drop_null_yahoo_tickers', FunctionTransformer(lambda df: df.dropna(subset=['yahoo_ticker'], how='any'))),
        ('dropna_targets', FunctionTransformer(lambda df: df.dropna(subset=[TARGET_COL, 'yahoo_ticker'], how='any'))),
        ('dropna_features', FunctionTransformer(lambda df: df.dropna(axis=1, how='all'))),

        ('calc_moves', FunctionTransformer(
            lambda df: df.groupby(TICKER_COL, group_keys=False).parallel_apply(
                lambda x: CalcMoves(copy=False).compute_multi_basic_moves(x,
                basic_move_params={k:v for k,v in basic_move_params.items() if all(col in df.columns for col in list(basic_move_params[k].values())[0:-1])},
                dask_join_cols=[DATE_COL, TICKER_COL])))),

        ('calc_diffs', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False).parallel_apply(
            lambda x: calc_diffs(x,
                                 diff_cols=np.intersect1d(x.columns, [i for i in eval(DIFF_COLS_TO_SELECT_STRING) if i in x.columns]),
                                 index_cols=[DATE_COL, TICKER_COL])))),

        ('calc_pct_chgs', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False).parallel_apply(
            lambda df: calc_pct_changes(df,
                                        pct_change_cols=np.intersect1d(df.columns, [i for i in eval(PCT_CHG_COLS_TO_SELECT_STRING) if i in df.columns]),
                                        index_cols=[DATE_COL, TICKER_COL])))),

        ('calc_move_iar', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False).parallel_apply(
            lambda df: calc_move_iar(df, **IAR_PARAMS)))),


        # when calling rolling features below, it is difficult to parameterize this as part of the config because the above lagging features
        # pipeline creates features that we want to take rolling features for, which are not columns in the original df
        ('rolling_features', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False).parallel_apply(
            lambda x: create_rolling_features(x,
                                              **ROLLING_FEATURES_PARAMS,
                                              rolling_cols=[i for i in x.columns if 'move' in i],
                                              ewm_cols=[i for i in x.columns if 'move' in i])
        ))),


        # ('convert_dtypes', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False).parallel_apply(
        #     lambda df: convert_df_dtypes(df, **CONVERT_DTYPE_PARAMS))))
        # ('conditional_feature_dropna', FunctionTransformer(lambda df: drop_nas(df, **DROPNA_PARAMS))),\
        # ('sort_df', FunctionTransformer(lambda df: df.sort_values(by=[DATE_COL, TICKER_COL]))) # has to be outside of the pipeline
        ]
    )

use_memory_fs = False if platform.system() == 'Windows' else True # pandarallel fails on windows in pycharm with use_memory_fs set to the default True
pandarallel.initialize(nb_workers=NUM_WORKERS, use_memory_fs=use_memory_fs)

input_df = df_numerai.copy()

start_feature_creation = time.time()

if VERBOSE: print('\nRunning feature creation...\n')

FEATURE_CREATOR_PIPE.steps = list(dict(FEATURE_CREATOR_PIPE.steps).items()) # just in case there is a user error containing duplicated transformation steps
feature_creator = FEATURE_CREATOR_PIPE.fit(df_numerai)
df_numerai = feature_creator.transform(df_numerai)

if VERBOSE: ('\nFeature creation took {} minutes.\n'.format(round((time.time() - start_feature_creation) / 60, 3)))

gc.collect()

if DROP_INIT_ROLLING_WINDOW_ROWS:
    df_numerai = df_numerai[df_numerai[DATE_COL] >= df_numerai[DATE_COL].min() + datetime.timedelta(ROLLING_FEATURES_PARAMS['rolling_params']['window'])]


### Timeseries Split ###

df_numerai = df_numerai.groupby([TICKER_COL]).apply(lambda df: timeseries_split(df, **TIMESERIES_SPLIT_PARAMS))

if 'data_type' in df_numerai.columns and USE_NUMERAI_TRAIN_TEST_SPLIT:
    df_numerai.loc[(df_numerai[SPLIT_COLNAME] == 'train') & (df_numerai['data_type'] == 'validation'), SPLIT_COLNAME] = 'val'
    df_numerai.loc[(df_numerai[SPLIT_COLNAME] == 'val') & (df_numerai['data_type'] == 'train'), SPLIT_COLNAME] = 'train'

input_features = eval(INPUT_FEATURES_STRING)
preserve_vars = list(set(PRESERVE_VARS + eval(PRESERVE_VARS_STRING)))
drop_vars = eval(DROP_VARS_STRING)

if TRAIN_ON_FULL_DATA:
    df_numerai.loc[df_numerai[SPLIT_COLNAME] == 'val', SPLIT_COLNAME] = 'train'
    df_numerai.loc[df_numerai[SPLIT_COLNAME] == 'test', SPLIT_COLNAME] = 'train'
    X_train, y_train = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('train')][input_features], \
                       df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('train')][TARGET_COL]
    X_val, y_val = X_train, y_train
else:
    X_train, y_train = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('train')][input_features], \
                       df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('train')][TARGET_COL]

    X_val, y_val = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('val')][input_features], \
                   df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('val')][TARGET_COL]

    # X_test, y_test = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('test')][input_features], \
    #                  df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('test')][TARGET_COL]


gc.collect()


### Preprocessing ###

start_feature_transformation = time.time()

if VERBOSE: ('\nRunning feature transformations...\n')

PREPROCESS_FEATURES_PARAMS['preserve_vars'] = preserve_vars

feature_transformer = PreprocessFeatures(**PREPROCESS_FEATURES_PARAMS).fit(X_train, y_train)

if VERBOSE: print('\nFeature transformation took {} minutes.\n'.format(round((time.time() - start_feature_transformation) / 60, 3)))

assert len([item for item, count in Counter(feature_transformer.output_cols).items() if count > 1]) == 0, 'output_cols has duplicate column names!'

X_train_transformed = feature_transformer.transform(X_train)\
    .drop(feature_transformer.feature_types['discarded_features'], axis=1)\
    .set_index(y_train.index)

final_features = [i for i in X_train_transformed.columns if i not in PRESERVE_VARS]

X_train_transformed = X_train_transformed[final_features]

if TRAIN_ON_FULL_DATA:
    X_transformed = X_train_transformed[final_features]
    X_val_transformed = X_train_transformed[final_features]
else:
    X_transformed = pd.concat([X_train_transformed, X_val_transformed, X_test_transformed])[final_features]
    X_val_transformed = feature_transformer.transform(X_val).set_index(y_val.index)[final_features]
    # X_test_transformed = feature_transformer.transform(X_test).set_index(y_test.index)[final_features]

gc.collect()


### Train model ###

start_model_training = time.time()
model_obj = RunModel(X_test=X_transformed,
                     features=list(X_transformed.columns),
                     X_train=X_train_transformed,
                     y_train=y_train,
                     algorithm=ALGORITHM,
                     eval_set=[(X_val_transformed, y_val)],
                     df_full=df_numerai,
                     **RUN_MODEL_PARAMS).train_and_predict()

if VERBOSE: print(f"\nModel training took {round((time.time() - start_model_training) / 60, 3)} minutes.\n")

model_obj['input_df'] = input_df
del input_df

model_obj['feature_creator'] = feature_creator
del feature_creator

model_obj['feature_transformer'] = feature_transformer
del feature_transformer

model_obj['input_features'] = input_features
model_obj['final_features'] = list(X_train_transformed.columns)
model_obj['final_dtype_mapping'] = model_obj['df_pred'].dtypes.to_dict()
model_obj['dropped_features'] = drop_vars

gc.collect()

if SAVE_OBJECT:
    save_start_time = time.time()
    dill.dump(model_obj, open(OBJECT_OUTPATH + \
                               type(model_obj['model']).__name__ + '_' + \
                               str(datetime.datetime.today()\
                                   .replace(second=0, microsecond=0))\
                                   .replace(' ', '_')\
                                   .replace(':', '_') + '.pkl',\
                               'wb'))

    save_end_time = time.time()
    print("\nIt took %s minutes to save the object\n" % round((save_end_time - save_start_time) / 60, 3))

end_time = time.time()

print("\nThe whole process took %s minutes\n" % round((end_time - start_time)/60, 3))
