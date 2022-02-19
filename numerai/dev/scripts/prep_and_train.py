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
import plotly.express as px
from skimpy import clean_columns

os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'

sys.path.append(os.getcwd())
from dev.scripts.ML_utils import * # run if on local machine
from dev.scripts.trading_utils import * # run if on local machine
from numerai.dev.scripts.numerai_utils import *
from numerai.dev.configs.prep_and_train_cfg import *


pd.set_option('display.float_format', lambda x: '%.5f' % x)
config = ConfigParser()
config.read('numerai/numerai_keys.ini')

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

if START_DATE:
    df_numerai = df_numerai[df_numerai[DATE_COL] >= START_DATE]

######################
###### pipeline ######
######################

# 1. data cleaning
# 2. feature creation
# 3. feature transformation

if CALC_DIFF_PIPE:
    diff_cols_to_select = eval(DIFF_COLS_TO_SELECT_STRING)
    diff_step = ('calc_diffs', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False).apply(lambda df: calc_diffs(df, diff_cols=np.intersect1d(df.columns, diff_cols_to_select)))))
    DATA_MANIPULATION_PIPE.steps.append(diff_step)

if CALC_PCT_CHG_PIPE:
    pct_chg_cols_to_select = eval(PCT_CHG_COLS_TO_SELECT_STRING)
    pct_chg_step = ('calc_pct_chgs', FunctionTransformer(lambda df: df.groupby(TICKER_COL, group_keys=False).apply(lambda df: calc_pct_changes(df, pct_change_cols=np.intersect1d(df.columns, pct_chg_cols_to_select)))))
    DATA_MANIPULATION_PIPE.steps.append(pct_chg_step)

DATA_MANIPULATION_PIPE.steps = list(dict(DATA_MANIPULATION_PIPE.steps).items())
data_manipulator = DATA_MANIPULATION_PIPE.fit(df_numerai)

df_numerai = data_manipulator.transform(df_numerai)


### Timeseries Split ###

df_numerai = timeseries_split(df_numerai, **TIMESERIES_SPLIT_PARAMS)

input_features = eval(INPUT_FEATURES_STRING)
preserve_vars = list(set(PRESERVE_VARS + eval(PRESERVE_VARS_STRING)))
drop_vars = eval(DROP_VARS_STRING)

X_train, y_train = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('train')][input_features], df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('train')][TARGET_COL]
X_val, y_val = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('val')][input_features], df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('val')][TARGET_COL]
X_test, y_test = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('test')][input_features], df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('test')][TARGET_COL]



### Preprocessing ###

PREPROCESS_FEATURES_PARAMS['preserve_vars'] = preserve_vars
feature_transformer = PreprocessFeatures(**PREPROCESS_FEATURES_PARAMS).fit(X_train, y_train)
final_features = FeatureImportance(feature_transformer, verbose=VERBOSE).get_feature_names() # [i.replace('.', '__').replace(' ', '_').lower().replace('-', '_') for i in FeatureImportance(feature_transformer, verbose=VERBOSE).get_feature_names()]
assert len([item for item, count in collections.Counter(final_features).items() if count > 1]) == 0

X_train_transformed = clean_columns(pd.DataFrame(feature_transformer.transform(X_train), columns=final_features))
X_val_transformed = clean_columns(pd.DataFrame(feature_transformer.transform(X_val), columns=final_features))
X_test_transformed = clean_columns(pd.DataFrame(feature_transformer.transform(X_test), columns=final_features))
X_transformed = clean_columns(pd.concat([X_train_transformed, X_val_transformed, X_test_transformed]))
final_features = X_train_transformed.columns

### Train model ###

model_obj = RunModel(X_test=X_transformed,
                      features=final_features,
                      X_train=X_train_transformed.tail(10000),
                      y_train=y_train.tail(10000),
                      algorithm=ALGORITHM,
                      eval_set=[(X_val_transformed, y_val)],
                      df_full=df_numerai,
                      **RUN_MODEL_PARAMS).train_and_predict()

model_obj['data_manipulator'] = data_manipulator
del data_manipulator

model_obj['feature_transformer'] = feature_transformer
del feature_transformer

model_obj['input_features'] = input_features
model_obj['final_features'] = final_features
model_obj['final_dtype_mapping'] = model_obj['df_pred'].dtypes.to_dict()

if SAVE_OBJECT:
    dill.dump(output_dict, open(OBJECT_OUTPATH + \
                               type(model_obj['model']).__name__ + '_' + \
                               str(datetime.datetime.today()\
                                   .replace(second=0, microsecond=0))\
                                   .replace(' ', '_')\
                                   .replace(':', '_') + '.pkl',\
                               'wb'))


########################
###### evaluation ######
########################

### run numerai analytics ###

# place holder - this will be part of another script

# try:
#     pred_colname = RUN_MODEL_PARAMS['prediction_colname']
# except:
#     pred_colname = 'prediction'

# importances = pd.DataFrame({(f, imp) for f, imp in zip(X_train_transformed.columns, output_dict['model'].feature_importances_)})\
#                 .rename(columns={0: 'feature', 1: 'importance'})\
#                 .sort_values(by='importance', ascending=False)
#
# ### corr_coefs for train / val / test ###
#
# train_era_scores = output_dict['df_pred'][output_dict['df_pred'][SPLIT_COLNAME].str.startswith('train')]\
#                     .groupby(DATE_COL)\
#                     .apply(calc_coef, TARGET, pred_colname)
#
# val_era_scores = output_dict['df_pred'][output_dict['df_pred'][SPLIT_COLNAME].str.startswith('val')]\
#                     .groupby(DATE_COL)\
#                     .apply(calc_coef, TARGET, pred_colname)
# test_era_scores = output_dict['df_pred'][output_dict['df_pred'][SPLIT_COLNAME].str.startswith('test')]\
#                     .groupby(DATE_COL)\
#                     .apply(calc_coef, TARGET, pred_colname)
#
#
# ### plot the coef scores / print the hit rates ###
#
# train_era_scores = pd.DataFrame(train_era_scores, columns=['era_score']).assign(era='train')
# val_era_scores = pd.DataFrame(val_era_scores, columns=['era_score']).assign(era='val')
# test_era_scores = pd.DataFrame(test_era_scores, columns=['era_score']).assign(era='test')
# era_scores = pd.concat([train_era_scores, val_era_scores, test_era_scores])
#
# fig = px.line(era_scores.reset_index(), x="date", y="era_score", line_group='era')
# fig.show()
#
#
# if __name__ == '__main__':
#     print('Done!')
