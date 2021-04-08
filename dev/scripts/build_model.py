# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:55:18 2021

@author: Matt
"""

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os


WORKING_DIRECTORY = 'C:/Users/Matt/trading/dev/'
CONFIG_DIRECTORY = WORKING_DIRECTORY + 'configs/'

WORKING_DIRECTORY = 'C:/Users/Matt/trading/dev/'
CONFIG_DIRECTORY = WORKING_DIRECTORY + 'configs/'

os.chdir(WORKING_DIRECTORY + 'scripts/')
from dev.scripts.trading_utils import *
from dev.scripts.ML_utils import *

os.chdir(CONFIG_DIRECTORY)
from dev.configs.pull_data_config import *
import dask
from dask import delayed
import feather


df_symbols_path = 'C:/Users/Matt/trading/dev/outputs/data/df_symbols_daily_features_and_target.feather'
df_symbols = pd.read_feather(df_symbols_path)

if 'timestamp' in df_symbols.columns:
    df_symbols.set_index('timestamp', inplace=True)


# Train a simple model and calc the pnl

X = df_symbols.drop('TSLA_target_HL5', axis = 1)
y = df_symbols['TSLA_target_HL5'].shift(-1).iloc[0:-1]
X[X==np.inf] = np.nan
y[y==np.inf] = np.nan

X = X[[i for i in X.columns if not 'target' in i]]
X.fillna(0, inplace = True)
y.fillna(0, inplace = True)
y = y.astype(int)

X = X[[i for i in X.columns if i.endswith('pct') or i.endswith('move')]]

X_train = X.loc[X.index < '2020-06-01']
X_val = X.loc[(X.index >= '2020-06-01') & (X.index < '2020-12-31')]
X_test = X.loc[X.index >= '2021-01-01']
y_train = y.loc[y.index < '2020-06-01']
y_val = y.loc[(y.index >= '2020-06-01') & (y.index < '2020-12-31')]
y_test = y.loc[y.index >= '2021-01-01']



run_model_params = {'convert_float32' : False,
                   'map_predictions_to_df_full' : True,
                   'predict_proba' : False,
                   'use_eval_set_when_possible' : True,
                   'fit_params' : {'early_stopping_rounds' : 5},
                   'predict_params' : {}
                    }

model_dict = RunModel(features = X_train.columns,
                      X_test=X,
                      X_train=X_train,
                      y_train=y_train,
                      algorithm = CatBoostClassifier(depth = 6, \
                                                     iterations = 500, \
                                                     loss_function='MultiClass', \
                                                     l2_leaf_reg = 9, \
                                                     learning_rate = 0.15, \
                                                     border_count = 100, \
                                                     random_strength = .2, \
                                                     max_ctr_complexity = 3, \
                                                     #subsample = .7, \
                                                     bagging_temperature = .2, \
                                                     thread_count = 16, \
                                                     verbose = False, \
                                                     random_state = 0),
                      #algorithm = RandomForestClassifier(),
                      eval_set = [(X_val, y_val)],
                      copy=True,
                      prediction_colname='pred',
                      seed=100,
                      bypass_all_numeric=False,
                      df_full=X,
                      **run_model_params).run_everything()





class Tmp():

    def __init__(self, a, **kwargs):
        self.a = a
        self.kwargs = kwargs

    def print_kwargs(self):
        print(self.kwargs)
        return self.a





cb_model = CatBoostClassifier(depth = 6, \
                              iterations = 500, \
                              loss_function='MultiClass', \
                              l2_leaf_reg = 9, \
                              learning_rate = 0.15, \
                              border_count = 100, \
                              random_strength = .2, \
                              max_ctr_complexity = 3, \
                              #subsample = .7, \
                              bagging_temperature = .2, \
                              thread_count = 16, \
                              verbose = False, \
                              random_state = 0).fit(X_train, y_train, eval_set = Pool(X_val, y_val), early_stopping_rounds = 5)


X_train['pred_HL5'] = cb_model.predict(X_train)
X_val['pred_HL5'] = cb_model.predict(X_val)
X_test['pred_HL5'] = cb_model.predict(X_test)


accuracy_score(X_train['pred_HL5'], y_train)
accuracy_score(X_val['pred_HL5'], y_val)
accuracy_score(X_test['pred_HL5'].iloc[0:-1], y_test)

precision_score(X_val['pred_HL5'], y_val, average = 'weighted')
precision_score(X_test['pred_HL5'].iloc[0:-1], y_test, average = 'weighted')
recall_score(X_val['pred_HL5'], y_val, average = 'weighted')
recall_score(X_test['pred_HL5'].iloc[0:-1], y_test, average = 'weighted')




X_train['pred_HL5_shift'] = X_train['pred_HL5'].shift()
X_val['pred_HL5_shift'] = X_val['pred_HL5'].shift()
X_test['pred_HL5_shift'] = X_test['pred_HL5'].shift()


X_train = calc_dummy_pnl(X_train, 
                         prediction_colname = 'pred_HL5_shift', 
                         actual_open_colname = 'TSLA_open', 
                         actual_close_colname = 'TSLA_close', 
                         pnl_colname = 'PnL')

X_val = calc_dummy_pnl(X_val, 
                       prediction_colname = 'pred_HL5_shift', 
                       actual_open_colname = 'TSLA_open', 
                       actual_close_colname = 'TSLA_close', 
                       pnl_colname = 'PnL')

X_test = calc_dummy_pnl(X_test,
                        prediction_colname = 'pred_HL5_shift',
                        actual_open_colname = 'TSLA_open',
                        actual_close_colname = 'TSLA_close',
                        pnl_colname = 'PnL')


X_train['PnL'].sum() * 100
X_val['PnL'].sum() * 100
X_test['PnL'].sum() * 100

X_train['PnL'].min() * 100
X_val['PnL'].min() * 100
X_test['PnL'].min() * 100

X_train['PnL'].max() * 100
X_val['PnL'].max() * 100
X_test['PnL'].max() * 100

X_train['PnL'].mean() * 100
X_val['PnL'].mean() * 100
X_test['PnL'].mean() * 100


X_val.loc[X_val['PnL'] > 0].shape[0] / X_val.shape[0]
X_test.loc[X_test['PnL'] > 0].shape[0] / X_test.shape[0]


calc_sharpe(X_train, 'pred_HL5', 2, 'PnL')
calc_sharpe(X_val, 'pred_HL5', 2, 'PnL')
calc_sharpe(X_test, 'pred_HL5', 2, 'PnL')

cb_model.get_feature_importance(prettified = True)

X_test


# How accurate is the model?

from sklearn.metrics import confusion_matrix
cb_cm_train = confusion_matrix(X_train['pred_HL5'], y_train)
cb_cm_val = confusion_matrix(X_val['pred_HL5'], y_val)
cb_cm_test = confusion_matrix(X_test['pred_HL5'].iloc[0:-1], y_test)

cb_cm_test.astype('float') / cb_cm_test.sum(axis=1)[:, np.newaxis]

tmp = pd.merge(X_test, y_test, left_index = True, right_index = True)
tmp[(tmp['pred_HL5'].isin([0,1])) & (tmp['TSLA_target_HL5'].isin([0,1]))].shape[0]
tmp[(tmp['pred_HL5'].isin([3,4])) & (tmp['TSLA_target_HL5'].isin([3,4]))].shape[0]

print('True accuracy: ', (tmp[(tmp['pred_HL5'].isin([0,1])) & (tmp['TSLA_target_HL5'].isin([0,1]))].shape[0] + tmp[(tmp['pred_HL5'].isin([3,4])) & (tmp['TSLA_target_HL5'].isin([3,4]))].shape[0]) / tmp.shape[0])











# Calc rules

# Calc pnl for rules
