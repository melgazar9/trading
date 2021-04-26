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

if DROP_NA_TARGETS:
    df_numerai.dropna(subset=[TARGET], how='any', inplace=True)
    df_numerai.reset_index(inplace=True)


### Timeseries Split ###

df_numerai = timeseries_split(df_numerai, **TIMESERIES_SPLIT_PARAMS)

if START_DATE:
    df_numerai = df_numerai[df_numerai[DATE_COL] >= START_DATE]


### train test split ###

input_features = eval(INPUT_FEATURES_STRING)
preserve_vars = eval(PRESERVE_VARS_STRING)

X_train, y_train = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('train')][input_features], df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('train')][TARGET]
X_val, y_val = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('val')][input_features], df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('val')][TARGET]
X_test, y_test = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('test')][input_features], df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('test')][TARGET]


### Preprocessing ###

# df_train = df_numerai[list(set(input_features + preserve_vars + [TARGET]))]

PREPROCESS_FEATURES_PARAMS['preserve_vars'] = preserve_vars
feature_transformer = PreprocessFeatures(**PREPROCESS_FEATURES_PARAMS).fit(X_train, y_train)

final_features = FeatureImportance(feature_transformer, verbose=VERBOSE).get_feature_names()
X_train_transformed = pd.DataFrame(feature_transformer.transform(X_train), columns=final_features)
X_val_transformed = pd.DataFrame(feature_transformer.transform(X_val), columns=final_features)
X_test_transformed = pd.DataFrame(feature_transformer.transform(X_test), columns=final_features)
X_transformed = pd.concat([X_train_transformed, X_val_transformed, X_test_transformed])

### Train model ###

model_dict = RunModel(X_test=X_transformed,
                      features=final_features,
                      X_train=X_train_transformed.tail(10000),
                      y_train=y_train.tail(10000),
                      algorithm=ALGORITHM,
                      eval_set=[(X_val_transformed, y_val)],
                      df_full=df_numerai,
                      **RUN_MODEL_PARAMS).run_everything()

try:
    pred_colname = RUN_MODEL_PARAMS['prediction_colname']
except:
    pred_colname = 'prediction'


model_dict['feature_transformer'] = feature_transformer
del feature_transformer

model_dict['input_features'] = input_features
model_dict['final_features'] = final_features
model_dict['data_type_mapping'] = model_dict['df_pred'].dtypes.to_dict()

if SAVE_OBJECT:
    dill.dump(model_dict, open(OBJECT_OUTPATH + \
                               type(model_dict['model']).__name__ + '_' + \
                               str(datetime.datetime.today()\
                                   .replace(second=0, microsecond=0))\
                                   .replace(' ', '_')\
                                   .replace(':', '_') + '.pkl',\
                               'wb'))


########################
###### evaluation ######
########################

### run numerai analytics ###

importances = pd.DataFrame({(f, imp) for f, imp in zip(X_train_transformed.columns, model_dict['model'].feature_importances_)})\
                .rename(columns={0: 'feature', 1: 'importance'})\
                .sort_values(by='importance', ascending=False)

### corr_coefs for train / val / test ###

train_era_scores = model_dict['df_pred'][model_dict['df_pred'][SPLIT_COLNAME].str.startswith('train')]\
                    .groupby(DATE_COL)\
                    .apply(calc_coef, TARGET, pred_colname)

val_era_scores = model_dict['df_pred'][model_dict['df_pred'][SPLIT_COLNAME].str.startswith('val')]\
                    .groupby(DATE_COL)\
                    .apply(calc_coef, TARGET, pred_colname)
test_era_scores = model_dict['df_pred'][model_dict['df_pred'][SPLIT_COLNAME].str.startswith('test')]\
                    .groupby(DATE_COL)\
                    .apply(calc_coef, TARGET, pred_colname)


### plot the coef scores / print the hit rates ###

train_era_scores = pd.DataFrame(train_era_scores, columns=['era_score']).assign(era='train')
val_era_scores = pd.DataFrame(val_era_scores, columns=['era_score']).assign(era='val')
test_era_scores = pd.DataFrame(test_era_scores, columns=['era_score']).assign(era='test')
era_scores = pd.concat([train_era_scores, val_era_scores, test_era_scores])

fig = px.line(era_scores.reset_index(), x="date", y="era_score", line_group='era')
fig.show()


if __name__ == '__main__':
    print('Done!')
