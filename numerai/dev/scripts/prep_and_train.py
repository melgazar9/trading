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

### Select only the input features ###

input_features = eval(INPUT_FEATURES_STRING)
preserve_vars = eval(PRESERVE_VARS_STRING)
# df_train = df_numerai[list(set(input_features + preserve_vars + [TARGET]))]

### Timeseries Split ###

df_numerai = timeseries_split(df_numerai, **TIMESERIES_SPLIT_PARAMS)

X_train, y_train = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('train')][input_features], df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('train')][TARGET]
X_val, y_val = df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('val')].drop(TARGET, axis=1), df_numerai[df_numerai[SPLIT_COLNAME].str.startswith('val')][TARGET]
# X_test, y_test = df_numerai[df_numerai[SPLIT_COLNAME] == 'test'].drop(TARGET, axis=1), df_numerai[df_numerai[SPLIT_COLNAME] == 'test'][TARGET]


PREPROCESS_FEATURES_PARAMS['preserve_vars'] = preserve_vars
feature_transformer = PreprocessFeatures(**PREPROCESS_FEATURES_PARAMS).fit(df_numerai[input_features], df_numerai[TARGET])
