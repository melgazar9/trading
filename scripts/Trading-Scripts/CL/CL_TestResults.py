import feather
import time
start_time = time.time()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from imblearn.over_sampling import SMOTE
from sklearn.metrics import *
from imblearn.metrics import classification_report_imbalanced
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, plot_importance
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import configparser
import ast
import pickle
import sys
from multiprocessing import Pool, Process
# import keras
# from keras.layers import *
# from keras.models import *
# from keras.preprocessing.sequence import *
# from keras.regularizers import *
# from keras.optimizers import *
# from keras.callbacks import EarlyStopping
import warnings

config = configparser.ConfigParser()
# config.read('/home/melgazar9/Trading/TD/Scripts/Trading-Scripts/CL/scripts/CL_30min_TRAIN-MODEL.ini')

config_path = config.read('/home/melgazar9/Trading/TD/Scripts/Trading-Scripts/CL/scripts/CL_{}_TestResults.ini'.format(sys.argv[1]))

print('**************** RUNNING', config['NAME']['product'], ' ****************')

train_start_date = config['PARAMS']['train_start_date']
train_end_date = config['PARAMS']['train_end_date']
val_start_date = config['PARAMS']['val_start_date']
val_end_date = config['PARAMS']['val_end_date']
test_start_date = config['PARAMS']['test_start_date']

threshold = float(config['PARAMS']['threshold'])
multiplier = float(config['PARAMS']['multiplier'])

strong_buy_actual = float(config['PARAMS']['strong_buy_actual'])
med_buy_actual = float(config['PARAMS']['med_buy_actual'])
no_trade_actual = float(config['PARAMS']['no_trade_actual'])
med_sell_actual = float(config['PARAMS']['med_sell_actual'])
strong_sell_actual = float(config['PARAMS']['strong_sell_actual'])
stop_actual = float(config['PARAMS']['stop_actual'])

strong_buy_HL = float(config['PARAMS']['strong_buy_HL'])
med_buy_HL = float(config['PARAMS']['med_buy_HL'])
no_trade_HL = float(config['PARAMS']['no_trade_HL'])
med_sell_HL = float(config['PARAMS']['med_sell_HL'])
strong_sell_HL = float(config['PARAMS']['strong_sell_HL'])
stop_HL = float(config['PARAMS']['stop_HL'])

strong_cap_actual = float(config['PARAMS']['strong_cap_actual'])
med_cap_actual = float(config['PARAMS']['med_cap_actual'])
strong_cap_HL = float(config['PARAMS']['strong_cap_HL'])
med_cap_HL = float(config['PARAMS']['med_cap_actual'])

min_prob0 = float(config['PARAMS']['min_prob0'])
min_prob1 = float(config['PARAMS']['min_prob1'])
min_prob3 = float(config['PARAMS']['min_prob3'])
min_prob4 = float(config['PARAMS']['min_prob4'])

min_prob_override0 = float(config['PARAMS']['min_prob_override0'])
min_prob_override1 = float(config['PARAMS']['min_prob_override1'])
min_prob_override3 = float(config['PARAMS']['min_prob_override3'])
min_prob_override4 = float(config['PARAMS']['min_prob_override4'])

Actual_Move = config['PARAMS']['Actual_Move_Name']
Actual_HighMove = config['PARAMS']['Actual_HighMove_Name']
Actual_LowMove = config['PARAMS']['Actual_LowMove_Name']


round_trip_fee = float(config['PARAMS']['round_trip_fee'])


if config['PARAMS']['raw_Actual_ON'] == 'TRUE':
    raw_model_Actual = pickle.load(open(config['PARAMS']['raw_model_filename_Actual'], 'rb'))
if config['PARAMS']['raw_HL_ON'] == 'TRUE':
    raw_model_HL = pickle.load(open(config['PARAMS']['raw_model_filename_HL'], 'rb'))
if config['PARAMS']['smote_Actual_ON'] == 'TRUE':
    smote_model_Actual = pickle.load(open(config['PARAMS']['smote_model_filename_Actual'], 'rb'))
if config['PARAMS']['smote_HL_ON'] == 'TRUE':
    smote_model_HL = pickle.load(open(config['PARAMS']['smote_model_filename_HL'], 'rb'))


if config['PARAMS']['read_feather'] == 'FALSE':
    df = pd.read_csv(config['PATH']['filename'])
elif config['PARAMS']['read_feather'] == 'TRUE':
    # df = feather.read_dataframe(config['PATH']['filename'], nthreads=32)
    print('Reading Feather File')
    df = feather.read_dataframe(config['PATH']['filename'])
df.set_index('Datetime', inplace=True)
df.index = pd.to_datetime(df.index)
df.index = df.index.tz_localize('utc').tz_convert('US/Central')

df.dropna(axis=0, inplace=True)


df[Actual_Move] = df[['Prev' + Actual_Move.strip('Actual')]].resample(config['NAME']['product'][3:5] + 'min').first().rename(columns={'Prev' + Actual_Move.strip('Actual') : Actual_Move}).shift(-1)
df[Actual_HighMove] = df[['Prev' + Actual_HighMove.strip('Actual')]].resample(config['NAME']['product'][3:5] + 'min').first().rename(columns={'Prev' + Actual_HighMove.strip('Actual') : Actual_HighMove}).shift(-1)
df[Actual_LowMove] = df[['Prev' + Actual_LowMove.strip('Actual')]].resample(config['NAME']['product'][3:5] + 'min').first().rename(columns={'Prev' + Actual_LowMove.strip('Actual'): Actual_LowMove}).shift(-1)


# Create categorical feature for Overnight_or_Intraday
df.loc[df.between_time('06:00:00','15:00:00', include_start=False).index, 'Overnight_or_Intraday'] = 1
df['Overnight_or_Intraday'].fillna(0, inplace=True)


class CalcTarget():

    def __init__(self, df, strong_buy, med_buy, no_trade, med_sell, strong_sell, threshold, stop):

        self.df = df
        self.strong_buy = strong_buy
        self.med_buy = med_buy
        self.no_trade = no_trade
        self.med_sell = med_sell
        self.strong_sell = strong_sell
        self.threshold = threshold # to prevent data errors
        self.stop = stop

    def calc_target_actual(self):

        super().__init__()

#         self.df[Actual_Move] = self.df['Prev' + Actual_Move.strip('Actual')].shift(-1)

        lst = []
        i=0
        while i < len(df):
            if np.isnan(self.df[Actual_LowMove][i]) or np.isnan(self.df[Actual_HighMove][i]):
                i+=1

            # strong buy
            elif self.df[Actual_Move][i] >= self.strong_buy and self.df[Actual_Move][i] <= self.threshold and self.df[Actual_LowMove][i] > (-1)*self.stop:
                lst.append(4)
                i+=1

            # medium buy
            elif self.df[Actual_Move][i] >= self.med_buy and self.df[Actual_Move][i] <= self.strong_buy and self.df[Actual_LowMove][i] > (-1)*self.stop:
                lst.append(3)
                i+=1

            # medium sell
            elif self.df[Actual_Move][i] <= (-1) * self.med_sell and self.df[Actual_Move][i] >= (-1) * self.strong_sell and self.df[Actual_LowMove][i] < self.stop:
                lst.append(1)
                i+=1

            # strong sell
            elif self.df[Actual_Move][i] <= (-1) * self.strong_sell and self.df[Actual_Move][i] >= (-1) * self.threshold and self.df[Actual_LowMove][i] < self.stop:
                lst.append(0)
                i+=1

            # no trade
            else:
                lst.append(2)
                i+=1

#         return pd.DataFrame(lst, index=self.df.index).rename(columns={0:'Target_Actual'})
        return pd.DataFrame(lst, index=self.df[[Actual_Move]].dropna().index).rename(columns={0:'Target_Actual'})


    def calc_target_HL(self):

        # stop means how much heat I am willing to take per trade
        # i.e. if the move went up in my favor $50 but I took $1000 worth of heat that isn't good
        # hm stands for high move, lm stands for low move

        lst = []

        i = 0
        while i < len(self.df):
            if np.isnan(self.df[Actual_LowMove][i]) or np.isnan(self.df[Actual_HighMove][i]):
                i+=1
            # if ActualHM >= buy signal AND ActualLM doesn't go below stop
            elif self.df[Actual_HighMove][i] >= self.strong_buy and self.df[Actual_LowMove][i] >= (-1)*self.stop:
                lst.append(4)
                i+=1
            elif self.df[Actual_LowMove][i] <= (-1)*self.strong_sell and self.df[Actual_HighMove][i] <= self.stop:
                lst.append(0)
                i+=1
            elif self.df[Actual_HighMove][i] >= self.med_buy and self.df[Actual_LowMove][i] >= (-1)*self.stop:
                lst.append(3)
                i+=1
            elif self.df[Actual_LowMove][i] <= (-1)*self.med_sell and self.df[Actual_HighMove][i] <= self.stop:
                lst.append(1)
                i+=1
            else:
                lst.append(2)
                i+=1
        print(len(lst))

#         return pd.DataFrame(lst, index=self.df.resample('60min').first().index).rename(columns={0:'Target_HL'})
        return pd.DataFrame(lst, index=self.df[[Actual_Move]].dropna().index).rename(columns={0:'Target_HL'})


# print('Calculating Target...')
target_actual = CalcTarget(df, strong_buy=strong_buy_actual, med_buy=med_buy_actual, no_trade=no_trade_actual,
                            med_sell=med_sell_actual, strong_sell=strong_sell_actual, threshold=threshold,
                            stop=stop_actual).calc_target_actual()

target_HL = CalcTarget(df, strong_buy=strong_buy_HL, med_buy=med_buy_HL, no_trade=no_trade_HL,
                        med_sell=med_sell_HL, strong_sell=strong_sell_HL, threshold=threshold,
                        stop=stop_HL).calc_target_HL()

print(target_actual['Target_Actual'].value_counts())
print(target_HL['Target_HL'].value_counts())

for i in range(int(config['PARAMS']['min_target_lookback']), int(config['PARAMS']['max_target_lookback']), int(config['PARAMS']['target_lookback_increment'])):
    target_HL['Prev_Target_HL' + str(i)] = target_HL['Target_HL'].shift(i)

for i in range(int(config['PARAMS']['min_target_lookback']), int(config['PARAMS']['max_target_lookback']), int(config['PARAMS']['target_lookback_increment'])):
    target_HL['Prev_Target_HL' + str(i)] = target_HL['Target_HL'].shift(i)

target_HL = target_HL.fillna(2).astype('int')
target_actual = target_actual.fillna(2).astype('int')

df['Target_Actual'] = target_actual['Target_Actual']
df['Target_HL'] = target_HL['Target_HL']

# Set categorical variables in an array
cat_vars = ['Year', 'Month', 'Week', 'Day', 'DayofWeek', 'DayofYear', 'IsMonthEnd',
            'IsMonthStart', 'IsQuarterEnd', 'IsQuarterStart', 'IsYearEnd', 'IsHoliday',
            'IsYearStart', 'Overnight_or_Intraday', 'Hour', 'Quarter']

cat_vars = cat_vars + [i for i in df.columns if i.endswith('Binned') or i.endswith('Opinion') or i.startswith('PrevTarget')]
cont_vars = [i for i in df.columns if not i in cat_vars]

print(len(df.columns) == len(cat_vars)+len(cont_vars)) # must be True!

joined = df[train_start_date:val_end_date][cat_vars+cont_vars].copy()
joined_test = df[test_start_date:][cat_vars+cont_vars].copy()

# for v in cat_vars:
#     joined[v] = joined[v].astype('category').cat.as_ordered()
#     joined_test[v] = joined_test[v].astype('category').cat.as_ordered()

# Create categorical variables for joined_test set on datetime-type columns: Year, Month, etc..
# 'Year', 'Month', 'Week', 'Day', 'DayofWeek', 'DayofYear', 'Hour', 'Quarter'
print('Filling Categories...')
joined_test['Year'] = joined_test.index.year.astype('category')
joined_test['Month'] = joined_test.index.month.astype('category')
joined_test['Week'] = joined_test.index.week.astype('category')
joined_test['Day'] = joined_test.index.day.astype('category')
joined_test['DayofWeek'] = joined_test.index.dayofweek.astype('category')
joined_test['DayofYear'] = joined_test.index.dayofyear.astype('category')
joined_test['Hour'] = joined_test.index.hour.astype('category')
joined_test['Quarter'] = joined_test.index.quarter.astype('category')

# Fill continuous variables using backfill
for v in cont_vars:
    joined[v] = joined[v].bfill().astype('float32')
    joined_test[v] = joined_test[v].bfill().astype('float32')
# print(joined.isnull().any()[joined.isnull().any()!=False], '\n')
# print(joined_test.isnull().any()[joined_test.isnull().any()!=False])
# print(joined.get_dtype_counts())
# print(joined_test.get_dtype_counts())

print('Concatenating joined and joined_test...')
df = pd.concat([joined, joined_test], axis=0)
print(df)

# One-hot encode categorical features
print('Getting Dummies...')
df = pd.get_dummies(df, columns=[i for i in cat_vars], drop_first=True)
df.rename(columns={'Overnight_or_Intraday_1.0':'Overnight_or_Intraday'}, inplace=True)
cat_vars = [i for i in df.columns if not i in cont_vars]

print('Converting dtypes...')
for col in cat_vars:
    df[col] = df[col].astype('category').cat.as_ordered()

for col in cont_vars:
    df[col] = df[col].astype('float32')


def time_series_split(train_start_date, train_end_date, val_start_date, val_end_date, test_start_date, HL):

    X = df.drop([Actual_Move, Actual_HighMove, Actual_LowMove, 'Target_Actual', 'Target_HL'], axis=1)
    y_actual = df['Target_Actual']
    y_HL = df['Target_HL']

    X.sort_index(inplace=True)
    y_actual.sort_index(inplace=True)
    y_HL.sort_index(inplace=True)

    y_HL.fillna(2, inplace=True)
    y_actual.fillna(2, inplace=True)

    X_train = X[train_start_date:train_end_date]
    X_val = X[val_start_date:val_end_date]
    X_test = X[test_start_date:]

    y_train_actual = y_actual[train_start_date:train_end_date]
    y_val_actual = y_actual[val_start_date:val_end_date]
    y_test_actual = y_actual[test_start_date:]

    y_train_HL = y_HL[train_start_date:train_end_date]
    y_val_HL = y_HL[val_start_date:val_end_date]
    y_test_HL = y_HL[test_start_date:]

    y_test_actual.fillna(2, inplace=True)
    y_test_HL.fillna(2, inplace=True)


    if config['PARAMS']['HL_ON']=='FALSE':
        print(X_train.shape, X_val.shape, X_test.shape, y_train_actual.shape, y_val_actual.shape, y_test_actual.shape)
        return X_train, X_val, X_test, y_train_actual, y_val_actual, y_test_actual

    elif config['PARAMS']['HL_ON']=='TRUE':
        print(X_train.shape, X_val.shape, X_test.shape, y_train_HL.shape, y_val_HL.shape, y_test_HL.shape)
        return X_train, X_val, X_test, y_train_HL, y_val_HL, y_test_HL


print('Time Series Split...')
# Actual
if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
    X_train, X_val, X_test, y_train_actual, y_val_actual, y_test_actual = time_series_split(train_start_date=train_start_date, train_end_date=train_end_date,
                                                                                            val_start_date=val_start_date, val_end_date=val_end_date,
                                                                                            test_start_date=test_start_date, HL=False)
# HL
if config['PARAMS']['HL_ON'] == 'TRUE':
    X_train, X_val, X_test, y_train_HL, y_val_HL, y_test_HL = time_series_split(train_start_date=train_start_date, train_end_date=train_end_date,
                                                                                val_start_date=val_start_date, val_end_date=val_end_date,
                                                                                test_start_date=test_start_date, HL=True)
print(X_train.get_dtype_counts(), X_val.get_dtype_counts(), X_test.get_dtype_counts())

cont_vars = [i for i in cont_vars if not i.startswith('Target') and not i.startswith('Actual')]





# Scale Data
if config['PARAMS']['scale'] == 'TRUE':

    print('Scaling Data...')
    scaler = pickle.load(open(config['PATH']['scaler_path'], 'rb'))
    # scaler = MinMaxScaler().fit(np.array(X_train))

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

    # for col in df.select_dtypes('uint8').columns:
    for col in cat_vars:
        #print(col)
        X_train_scaled[col] = X_train_scaled[col].astype('category').cat.as_ordered()
        X_val_scaled[col] = X_val_scaled[col].astype('category').cat.as_ordered()
        X_test_scaled[col] = X_test_scaled[col].astype('category').cat.as_ordered()

    for col in cont_vars:
        X_train_scaled[col] = X_train_scaled[col].astype('float32')
        X_val_scaled[col] = X_val_scaled[col].astype('float32')
        X_test_scaled[col] = X_test_scaled[col].astype('float32')

if config['PARAMS']['one_hot_targets'] == 'TRUE':
    print('Converting categorical targets to one-hot encoding targets')
    y_train_oh_actual = keras.utils.to_categorical(y_train_actual, num_class=5)
    y_val_oh_actual = keras.utils.to_categorical(y_val_actual, num_class=5)
    y_test_oh_actual = keras.utils.to_categorical(y_test_actual, num_class=5)

    y_train_oh_HL = keras.utils.to_categorical(y_train_HL, num_class=5)
    y_val_oh_HL = keras.utils.to_categorical(y_val_HL, num_class=5)
    y_test_oh_HL = keras.utils.to_categorical(y_test_HL, num_class=5)




class CalcResults():

    def __init__(self, model, X, y, predictions, stop, strong_cap, med_cap, multiplier, need_cont_vars, HL, NN=False):

        self.model = model
        self.X = X
        self.y = y
        self.predictions = predictions
        self.stop = stop
        self.strong_cap = strong_cap
        self.med_cap = med_cap
        self.multiplier = multiplier
        self.need_cont_vars = need_cont_vars
        self.HL = HL
        self.NN = NN

    def calc_predictions(self):

        # Calculates which class to predict based on the predicted probability for decision making
        super().__init__()
        probs = self.model.predict_proba(self.X)

        lst=[]

        # for p in probs:
        #
        #     if p[2] == max(p): # if no trade is max probaility
        #         lst.append(2)
        #
        #     elif (p[0] >= min_prob0) and (p[0] >= max(p)): # if strong sell is max probability and >= min_prob_threshold
        #         if p[1] == sorted(p)[-2]: # if med_sell is the second highest probability
        #             lst.append(0)
        #         # elif (p[0] - sorted(p)[-2]) >= float(config['PARAMS']['min_prob_diff']): # strong sell - second highest probability which isn't med_sell (difference)
        #         #     lst.append(2)
        #         else:
        #             lst.append(2)
        #
        #     elif (p[1] >= min_prob1) and (p[1] >= max(p)): # if med sell is max probability and >= min_prob_threshold
        #         if p[0] == sorted(p)[-2]:
        #             lst.append(1)
        #         # elif (p[1] - sorted(p)[-2] >= float(config['PARAMS']['min_prob_diff'])):
        #         #     lst.append(2)
        #         else:
        #             lst.append(2)
        #
        #     elif (p[3] >= min_prob3) and (p[3] >= max(p)): # if med buy is max probability and >= min_prob_threshold
        #         if p[4] == sorted(p)[-2]:
        #             lst.append(3)
        #         # elif (p[3] - sorted(p)[-2] >= float(config['PARAMS']['min_prob_diff'])):
        #         #     lst.append(2)
        #         else:
        #             lst.append(2)
        #
        #     elif (p[4] >= min_prob4) and (p[4] >= max(p)): # if strong buy is max probability and >= min_prob_threshold
        #         if (p[3] == sorted(p)[-2]): # med_buy is second highest probability
        #             lst.append(4)
        #         # elif (p[4] - sorted(p)[-2] >= float(config['PARAMS']['min_prob_diff'])):
        #         #     lst.append(2)
        #         else:
        #             lst.append(2)
        #
        #     else:
        #         lst.append(2)

        for p in probs:

            if p[2] == max(p): # if no trade is max probaility
                lst.append(2)

            elif (p[0] >= min_prob0) and (p[0] >= max(p)): # if strong sell is max probability and >= min_prob_threshold
                if config['PARAMS']['PredProb_OrderedSell_ON'] == 'TRUE': # ON means secodn highest probability for sell must be med_sell or strong_sell.. the buy/sell probabilities (signals) can't contradict each other!
                    if p[1] == sorted(p)[-2]: # if med_sell is the second highest probability
                        lst.append(0)
                    # elif (p[0] - sorted(p)[-2]) >= float(config['PARAMS']['min_prob_diff']): # strong sell - second highest probability which isn't med_sell (difference)
                    #     lst.append(2)
                    elif config['PARAMS']['PredProb_OVERRIDE_ON'] == 'TRUE':
                        if p[0] >= min_prob_override0:
                            lst.append(0)
                        else:
                            lst.append(2)
                    else:
                        lst.append(2)

                elif config['PARAMS']['PredProb_OrderedSell_ON'] == 'FALSE':
                    lst.append(0)


            elif (p[1] >= min_prob1) and (p[1] >= max(p)): # if med sell is max probability and >= min_prob_threshold
                if config['PARAMS']['PredProb_OrderedSell_ON'] == 'TRUE':
                    if p[0] == sorted(p)[-2]: # if strong_sell is the second highest probability
                        lst.append(1)
                    # elif (p[1] - sorted(p)[-2] >= float(config['PARAMS']['min_prob_diff'])):
                    #     lst.append(2)
                    elif config['PARAMS']['PredProb_OVERRIDE_ON'] == 'TRUE':
                        if p[1] >= min_prob_override1:
                            lst.append(1)
                        else:
                            lst.append(2)
                    else:
                        lst.append(2)

                elif config['PARAMS']['PredProb_OrderedSell_ON'] == 'FALSE':
                    lst.append(1)

            elif (p[3] >= min_prob3) and (p[3] >= max(p)): # if med buy is max probability and >= min_prob_threshold
                if config['PARAMS']['PredProb_OrderedBuy_ON'] == 'TRUE':
                    if p[4] == sorted(p)[-2]: # if strong_buy is the second highest probability
                        lst.append(3)
                    # elif (p[3] - sorted(p)[-2] >= float(config['PARAMS']['min_prob_diff'])):
                    #     lst.append(2)
                    elif config['PARAMS']['PredProb_OVERRIDE_ON'] == 'TRUE':
                        if p[3] >= min_prob_override3:
                            lst.append(3)
                        else:
                            lst.append(2)
                    else:
                        lst.append(2)
                elif config['PARAMS']['PredProb_OrderedBuy_ON'] == 'FALSE':
                    lst.append(3)

            elif (p[4] >= min_prob4) and (p[4] >= max(p)): # if strong buy is max probability and >= min_prob_threshold
                if config['PARAMS']['PredProb_OrderedBuy_ON'] == 'TRUE':
                    if (p[3] == sorted(p)[-2]): # med_buy is second highest probability
                        lst.append(4)
                    # elif (p[4] - sorted(p)[-2] >= float(config['PARAMS']['min_prob_diff'])):
                    #     lst.append(2)
                    elif config['PARAMS']['PredProb_OVERRIDE_ON'] == 'TRUE':
                        if p[4] >= min_prob_override4:
                            lst.append(4)
                        else:
                            lst.append(2)
                    else:
                        lst.append(2)
                elif config['PARAMS']['PredProb_OrderedBuy_ON'] == 'FALSE':
                    lst.append(4)

            else:
                lst.append(2)

        return lst

    def initialize_df(self):

        super().__init__()

        if self.need_cont_vars == True:
            self.X = self.X.astype('float32')
        print(self.X.get_dtype_counts())

        if config['PARAMS']['PredProb_ON'] == 'FALSE':
            pred = pd.DataFrame(self.model.predict(self.X), index=self.X.index).rename(columns={0:self.predictions})
        elif config['PARAMS']['PredProb_ON'] == 'TRUE':
            pred = pd.DataFrame(self.calc_predictions(), index=self.X.index).rename(columns={0:self.predictions})

        if self.HL == False:
            results = pd.concat([pd.DataFrame(self.y, index=self.X.index), pred], axis=1).rename(columns={0:'Target_Actual'})
        elif self.HL == True:
            results = pd.concat([pd.DataFrame(self.y, index=self.X.index), pred], axis=1).rename(columns={0:'Target_HL'})

        results = pd.concat([results, pd.DataFrame(df[Actual_HighMove], index=pred.index),
                             pd.DataFrame(df[Actual_LowMove], index=pred.index),
                             pd.DataFrame(df[Actual_Move], index=pred.index)], axis=1)

        return results

    def calc_pnl_traditional_onelot_actual(self):

        super().__init__()

        results = self.initialize_df()

        print('Actual Move Results')

        lst=[]

        i=0
        while i < len(results):

            # strong buy -> 2 contracts traded
            if results[self.predictions][i] == 4:
                if results[Actual_LowMove][i] > (-1)*self.stop: # not stopped out
                    if results[Actual_HighMove][i] >= self.strong_cap:
                        lst.append(1*self.strong_cap)
                        i+=1
                    else:
                        lst.append(1*results[Actual_Move][i])
                        i+=1
                elif results[Actual_LowMove][i] <= (-1)*self.stop: # stopped out
                    lst.append((-1)*self.stop) # -.02 for assuming a trade out -> i.e. selling at the bid horribly (worst case testing)
                    i+=1
                else:
                    print('Error1')
                    lst.append(np.nan)
                    i+=1

            # medium buy
            elif results[self.predictions][i] == 3:
                if results[Actual_LowMove][i] >= (-1)*self.stop:
                    if results[Actual_HighMove][i] > self.med_cap:
                        lst.append(self.med_cap)
                        i+=1
                    else:
                        lst.append(results[Actual_Move][i])
                        i+=1
                elif results[Actual_LowMove][i] <= (-1)*self.stop:
                    lst.append((-1)*self.stop)
                    i+=1
                else:
                    print('Error2')
                    lst.append(np.nan)
                    i+=1

            # no trade
            elif results[self.predictions][i] == 2:
                lst.append(0)
                i+=1

            # medium sell
            elif results[self.predictions][i] == 1:
                if results[Actual_HighMove][i] < self.stop:
                    if results[Actual_LowMove][i] < (-1)*self.med_cap:
                        lst.append(self.med_cap)
                        i+=1
                    else:
                        lst.append((-1) * results[Actual_Move][i])
                        i+=1
                elif results[Actual_HighMove][i] >= self.stop:
                    lst.append((-1)*self.stop)
                    i+=1
                else:
                    print('Error3')
                    lst.append(np.nan)
                    i+=1

            # strong sell
            elif results[self.predictions][i] == 0:
                if results[Actual_HighMove][i] < self.stop:
                    if results[Actual_LowMove][i] < (-1)*self.strong_cap:
                        lst.append(1*self.strong_cap)
                        i+=1
                    else:
                        lst.append((-1) * results[Actual_Move][i])
                        i+=1
                elif results[Actual_HighMove][i] >= self.stop:
                    lst.append((-1)*self.stop)
                    i+=1
                else:
                    print('Error5')
                    lst.append(np.nan)
                    i+=1
            else:
                lst2.append('Error6')
                lst.append(np.nan)
                i+=1

        pnl = pd.DataFrame(lst, index=results.index).rename(columns={0:self.predictions.split(' ')[0] + ' P/L Actual'})
        results = pd.concat([results, pnl], axis=1)
        results.sort_index(inplace=True)

        print('Trade Pct Predicted Actual:', float(len(results[results[self.predictions]!=2])) / len(results))
        print('Trade Pct Actual Target Actual', float(len(results[results['Target_Actual']!=2])) / len(results))
        print(self.predictions + ' P/L BEFORE FEES Actual:', float(np.sum(results[self.predictions.split(' ')[0] + ' P/L Actual'])*self.multiplier))
        print(self.predictions + ' P/L AFTER FEES Actual:', float(np.sum(results[self.predictions.split(' ')[0] + ' P/L Actual'])*self.multiplier) - float((round_trip_fee)*len(results.loc[results[self.predictions]!=2]))) # subtract fees

        dataset_name = self.predictions.split(' ')[0]

        pnl_4 = results[results[dataset_name + ' Predictions']==4]
        pnl_3 = results[results[dataset_name + ' Predictions']==3]
        pnl_1 = results[results[dataset_name + ' Predictions']==1]
        pnl_0 = results[results[dataset_name + ' Predictions']==0]

        print(dataset_name + ' Class 4:', pnl_4[self.predictions.split(' ')[0] + ' P/L Actual'].sum()*self.multiplier)
        print(dataset_name + ' Class 3:', pnl_3[self.predictions.split(' ')[0] + ' P/L Actual'].sum()*self.multiplier)
        print(dataset_name + ' Class 1:', pnl_1[self.predictions.split(' ')[0] + ' P/L Actual'].sum()*self.multiplier)
        print(dataset_name + ' Class 0:', pnl_0[self.predictions.split(' ')[0] + ' P/L Actual'].sum()*self.multiplier)

        return results


    def calc_pnl_traditional_2to1_actual(self):

        super().__init__()
        results = self.initialize_df()

        print('Actual Move Results')
        # Concat results into dataframe with columns: Target, Predictions, Actual Move

        pred = pd.DataFrame(model.predict(X), index=X.index).rename(columns={0:self.predictions})
        results = pd.concat([pd.DataFrame(y, index=X.index), pred], axis=1).rename(columns={0:'Target_Actual'})
        results = pd.concat([results, pd.DataFrame(df[Actual_Move], index=X.index)], axis=1)

        # Calculate P/L and concat it as an additional column
        lst=[]

        i=0
        while i < len(results):

            # strong buy -> 2 contracts traded
            if results[self.predictions][i] == 4:
                lst.append(1*results[Actual_Move][i])
                i+=1

            # medium buy
            elif results[self.predictions][i] == 3:
                lst.append(results[Actual_Move][i])
                i+=1

            # no trade
            elif results[self.predictions][i] == 2:
                lst.append(0)
                i+=1

            # medium sell
            elif results[self.predictions][i] == 1:
                lst.append((-1) * results[Actual_Move][i])
                i+=1

            # strong sell
            elif results[self.predictions][i] == 0:
                lst.append((-1) * results[Actual_Move][i])
                i+=1

            else:
                lst.append('Error')
                i+=1

        pnl = pd.DataFrame(lst, index=results.index).rename(columns={0:self.predictions.split(' ')[0] + ' P/L HL'})
        results = pd.concat([results, pnl], axis=1)

        print(self.predictions + ' P/L HL BEFORE FEES:', float(np.sum(results[self.predictions.split(' ')[0] + ' P/L HL'])*self.multiplier))
        print(self.predictions + ' P/L HL AFTER FEES:', float(np.sum(results[self.predictions.split(' ')[0] + ' P/L HL'])*self.multiplier) - float((round_trip_fee)*len(results.loc[results[self.predictions]!=2]))) # subtract fees

        return results.sort_index()



    def calc_pnl_traditional_onelot_HL(self):

        super().__init__()
        results = self.initialize_df()

        print('HL Results')
        # Calculate P/L and concat it as an additional column
        lst=[]
#         lst2=[]
        i=0
        while i < len(results):

            # strong buy -> 2 contracts traded
            if results[self.predictions][i] == 4:
                if results[Actual_LowMove][i] > (-1)*self.stop: # not stopped out
                    if results[Actual_HighMove][i] > self.strong_cap:
#                         lst2.append(1)
                        lst.append(self.strong_cap) # take profit at strong_cap
                        i+=1
                    else:
#                         lst2.append(2)
                        lst.append(results[Actual_Move][i])
                        i+=1
                else: # stopped out
#                     lst2.append(3)
                    lst.append((-1)*self.stop) # Assume additional loss of 2 ticks for being stopped out
                    i+=1

            # med buy -> 1 contract traded
            elif results[self.predictions][i] == 3:
                if results[Actual_LowMove][i] > (-1)*self.stop: # not stopped out
                    if results[Actual_HighMove][i] > self.med_cap:
#                         lst2.append(4)
                        lst.append(self.med_cap) # take profit at med_cap
                        i+=1
                    else:
#                         lst2.append(5)
                        lst.append(results[Actual_Move][i])
                        i+=1
                else: # stopped out
#                     lst2.append(6)
                    lst.append((-1)*self.stop) # Assume additional loss of 1 tick for being stopped out
                    i+=1


            elif results[self.predictions][i] == 2:
#                 lst2.append(7)
                lst.append(0)
                i+=1

            # med sell
            elif results[self.predictions][i] == 1:
                if results[Actual_HighMove][i] < self.stop: # not stopped out
                    if results[Actual_LowMove][i] < (-1)*self.med_cap:
#                         lst2.append(8)
                        lst.append(self.med_cap) # take profit at med_cap
                        i+=1
                    else:
                        lst.append((-1)*results[Actual_Move][i])
#                         lst2.append(9)
                        i+=1
                else:
                    lst.append((-1)*self.stop)
#                     lst2.append(10)
                    i+=1

            # strong sell
            elif results[self.predictions][i] == 0:
                if results[Actual_HighMove][i] < self.stop:
                    if results[Actual_LowMove][i] < (-1)*self.strong_cap:
#                         lst2.append(11)
                        lst.append((1)*self.strong_cap) # take profit
                        i+=1
                    else:
#                         lst2.append(12)
                        lst.append((-1)*results[Actual_Move][i])
                        i+=1

                else:
#                     lst2.append(13)
                    lst.append((-1)*self.stop)
                    i+=1

            else:
#                 lst2.append('error')
                lst.append('Error')
                i+=1

        pnl = pd.DataFrame(lst, index=results.index).rename(columns={0:self.predictions.split(' ')[0] + ' P/L HL'})
#         results = pd.concat([results, pnl, pd.DataFrame(lst2, index=results.index)], axis=1)
        results = pd.concat([results, pnl], axis=1)
        results.sort_index(inplace=True)

        print('Trade Pct Predicted HL:', float(len(results[results[self.predictions]!=2])) / len(results))
        print('Trade Pct Actual Target HL', float(len(results[results['Target_HL']!=2])) / len(results))
        print(self.predictions + ' P/L HL BEFORE FEES:', np.sum(results[self.predictions.split(' ')[0] + ' P/L HL'])*self.multiplier)
        print(self.predictions + ' P/L HL AFTER FEES:', np.sum(results[self.predictions.split(' ')[0] + ' P/L HL'])*self.multiplier - (round_trip_fee)*len(results.loc[results[self.predictions]!=2])) # subtract fees

        dataset_name = self.predictions.split(' ')[0]


        pnl_4 = results[results[dataset_name + ' Predictions']==4]
        pnl_3 = results[results[dataset_name + ' Predictions']==3]
        pnl_1 = results[results[dataset_name + ' Predictions']==1]
        pnl_0 = results[results[dataset_name + ' Predictions']==0]

        print(dataset_name + ' Class 4:', pnl_4[dataset_name + ' P/L HL'].sum()*self.multiplier)
        print(dataset_name + ' Class 3:', pnl_3[dataset_name + ' P/L HL'].sum()*self.multiplier)
        print(dataset_name + ' Class 1:', pnl_1[dataset_name + ' P/L HL'].sum()*self.multiplier)
        print(dataset_name + ' Class 0:', pnl_0[dataset_name + ' P/L HL'].sum()*self.multiplier)

        return results


    def calc_profitability(self):

        if self.HL == False:
            if self.NN == False:
                prediction_df = self.calc_pnl_traditional_onelot_actual()
#             if self.NN == True:
#                 pred = pd.DataFrame(np.argmax(self.model.predict(self.X), axis=1)).set_index(self.X_df.index).rename(columns={0:predictions})
#                 prediction_df = pd.concat([pd.DataFrame(self.y, index=pred.index), pred], axis=1).rename(columns={0:'Target'})
#                 prediction_df = pd.concat([prediction_df, pd.DataFrame(df[Actual_Move], index=pred.index)], axis=1)

        elif self.HL == True:
            if self.NN == False:
                prediction_df = self.calc_pnl_traditional_onelot_HL()

#             elif self.NN == True:
#                 if self.NN == True:
#                     pred = pd.DataFrame(np.argmax(self.model.predict(self.X), axis=1)).set_index(self.X_df.index).rename(columns={0:predictions})
#                     prediction_df = pd.concat([pd.DataFrame(self.y, index=pred.index), pred], axis=1).rename(columns={0:'Target'})

        results = prediction_df[prediction_df[self.predictions] != 2]

        sell_lst = []
        buy_lst = []

        if self.HL == False:


            i=0
            while i < len(results):
                if results[self.predictions][i] >= 3 and results[Actual_Move][i] > 0:
                    buy_lst.append(1)
                    i+=1
                elif results[self.predictions][i] >= 3 and results[Actual_Move][i] <= 0:
                    buy_lst.append(0)
                    i+=1
                elif results[self.predictions][i] <= 1 and results[Actual_Move][i] < 0:
                    sell_lst.append(1)
                    i+=1
                elif results[self.predictions][i] <= 1 and results[Actual_Move][i] >= 0:
                    sell_lst.append(0)
                    i+=1
                else:
                    sell_lst.append('Error')
                    i+=1

            if config['PARAMS']['plot_pnl_cumsum'] == 'TRUE':
                plt.figure(figsize=(26,8))
                plt.plot(results.index, results[self.predictions.split(' ')[0] + ' P/L Actual'].cumsum())
                plt.show()


        if self.HL == True:

            # Below: the result is not really too accurate but it will suffice.
            # The actual rate will be better because the low move can happen before we get stopped out.
            # I just assume if the stop target is hit we get stopped out, but that's not reality


            i=0
            while i < len(results):
                if results[self.predictions][i] >= 3 and ((results[Actual_Move][i] > 0 and results[Actual_LowMove][i] > (-1)*self.stop) or
                                                          (results[Actual_HighMove][i] > self.strong_cap and results[Actual_LowMove][i] > (-1)*self.stop)):
                    buy_lst.append(1)
                    i+=1
                elif results[self.predictions][i] >= 3 and (results[Actual_Move][i] <= 0 or results[Actual_LowMove][i] <= (-1)*self.stop):
                    buy_lst.append(0)
                    i+=1
                elif results[self.predictions][i] <= 1 and ((results[Actual_Move][i] < 0 and results[Actual_HighMove][i] <= self.stop) or
                                                            (results[Actual_LowMove][i] <= (-1)*self.strong_cap and results[Actual_HighMove][i] <= self.stop)):
                    sell_lst.append(1)
                    i+=1
                elif results[self.predictions][i] <= 1 and (results[Actual_Move][i] >= 0 or results[Actual_HighMove][i] >= self.stop):
                    sell_lst.append(0)
                    i+=1
                else:
                    sell_lst.append('Error')
                    i+=1

            if config['PARAMS']['plot_pnl_cumsum'] == 'TRUE':
                plt.figure(figsize=(26,8))
                plt.plot(results.index, results[self.predictions.split(' ')[0] + ' P/L HL'].cumsum())
                plt.show()


        try:
            print('Buy Accuracy:', float(Counter(buy_lst)[1]) / len(buy_lst))
            print('Sell Accuracy:', float(Counter(sell_lst)[1]) / len(sell_lst))
            # print('\n')
        except ZeroDivisionError:
            try:
                print('Sell Accuracy:', float(Counter(sell_lst)[1]) / len(sell_lst))
                print('Could not calculate buy accuracy -> Could not divide by 0.')
            except ZeroDivisionError:
                try:
                    print('Buy Accuracy:', float(Counter(buy_lst)[1]) / len(buy_lst))
                    print('Could not calculate sell accuracy -> Could not divide by 0.')
                except ZeroDivisionError:
                    print('Zero Division Error. Could not divide by 0.')
            # print('\n')

        return results


    def calc_sharpe(self, results):

        super().__init__()

        dataset_name = self.predictions.split(' ')[0]

        if config['PARAMS']['HL_ON'] == 'TRUE':
            sharpe_HL = np.sum(results[results[self.predictions]!=2][dataset_name + ' P/L HL'] / (np.sqrt(len(results[results[self.predictions] != 2])*results[results[self.predictions] !=2][dataset_name + ' P/L HL'].std())))
            print('HL Sharpe:', sharpe_HL)
            print('\n')
        if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
            sharpe_actual = np.sum(results[results[self.predictions]!=2][dataset_name + ' P/L Actual'] / (np.sqrt(len(results[results[self.predictions] != 2])*results[results[self.predictions] !=2][dataset_name + ' P/L Actual'].std())))
            print('Actual Sharpe:', sharpe_actual)
            print('\n')

        return

    def calc_results(self):

        super().__init__()

        self.initialize_df()
        results = self.calc_profitability().astype('float32')
        self.calc_sharpe(results)

        return results

if config['PARAMS']['need_cont_vars'] == 'TRUE':
    need_cont_vars_param = True
elif config['PARAMS']['need_cont_vars'] == 'FALSE':
    need_cont_vars_param = False
if config['PARAMS']['plot_importances'] == 'TRUE':
    plot_importances_param = True
elif config['PARAMS']['plot_importances'] == 'FALSE':
    plot_importances_param = False







if config['PARAMS']['HL_ON'] == 'TRUE':

  if config['PARAMS']['RAW_ON'] == 'TRUE':
      # HL RAW
      if config['MODEL']['xgbc'] == 'TRUE':
          print('Running XGBClassifier RAW-HL')
          raw_model = XGBClassifier(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_HL']))

      if config['MODEL']['lgbmc'] == 'TRUE':
          print('Running LGBMClassifier RAW-HL')
          raw_model = LGBMClassifier(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_HL']))

      if config['MODEL']['gbmc'] == 'TRUE':
          print('Running GradientBoostingClassifier RAW-HL')
          raw_model = GradientBoostingClassifier(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_HL']))

      if config['MODEL']['lrc'] == 'TRUE':
          print('Running LogisticRegressionCV RAW-HL')
          raw_model = LogisticRegressionCV(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_HL']))

      if config['MODEL']['svc'] == 'TRUE':
          print('Running SVC RAW-HL')
          raw_model = SVC(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_HL']))

      if config['MODEL']['rfc'] == 'TRUE':
          print('Running RandomForestClassifier RAW-HL')
          raw_model = RandomForestClassifier(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_HL']))

      # raw_model_HL = pickle.load(open(raw_model_filename_HL, 'rb'))


      CalcResults(model=raw_model_HL, X=X_train, y=y_train_HL, predictions='Train Predictions',
                  stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                  multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                  HL=False, NN=False).calc_results()

      CalcResults(model=raw_model_HL, X=X_val, y=y_val_HL, predictions='Val Predictions',
                  stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                  multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                  HL=False, NN=False).calc_results()

      CalcResults(model=raw_model_HL, X=X_test, y=y_test_HL, predictions='Test Predictions',
                  stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                  multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                  HL=False, NN=False).calc_results()


  # HL SMOTE
  if config['PARAMS']['SMOTE_ON'] == 'TRUE':


      if config['MODEL']['xgbc'] == 'TRUE':
          print('Running XGBClassifier SMOTE-HL')
          model_name = 'XGBC'
          smote_model = XGBClassifier(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_HL']))

      if config['MODEL']['lgbmc'] == 'TRUE':
          print('Running LGBMClassifier SMOTE-HL')
          model_name = 'LGBMC'
          smote_model = LGBMClassifier(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_HL']))

      if config['MODEL']['gbmc'] == 'TRUE':
          print('Running GradientBoostingClassifier SMOTE-HL')
          model_name = 'GBMC'
          smote_model = GradientBoostingClassifier(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_HL']))

      if config['MODEL']['lrc'] == 'TRUE':
          print('Running LogisticRegressionCV SMOTE-HL')
          model_name = 'LRC'
          smote_model = LogisticRegressionCV(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_HL']))

      if config['MODEL']['svc'] == 'TRUE':
          print('Running SVC SMOTE-HL')
          model_name = 'SVC'
          smote_model = SVC(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_HL']))

      if config['MODEL']['rfc'] == 'TRUE':
          print('Running RandomForestClassifier SMOTE-HL')
          model_name = 'RFC'
          smote_model = RandomForestClassifier(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_HL']))

      # smote_model_HL = pickle.load(open(smote_model_filename_HL, 'rb'))


      CalcResults(smote_model_HL, X_train, y_train_HL, predictions='Train Predictions',
                  stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                  multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                  HL=True, NN=False).calc_results()
      CalcResults(smote_model_HL, X_val, y_val_HL, predictions='Val Predictions',
                  stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                  multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                  HL=True, NN=False).calc_results()
      CalcResults(smote_model_HL, X_test, y_test_HL, predictions='Test Predictions',
                  stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                  multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                  HL=True, NN=False).calc_results()





if config['PARAMS']['ACTUAL_ON'] == 'TRUE':

    if config['PARAMS']['RAW_ON'] == 'TRUE':
        # Actual RAW

        if config['MODEL']['xgbc'] == 'TRUE':
            print('Running XGBClassifier RAW-ACTUAL')
            raw_model = XGBClassifier(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_ACTUAL']))

        if config['MODEL']['lgbmc'] == 'TRUE':
            print('Running LGBMClassifier RAW-ACTUAL')
            raw_model = LGBMClassifier(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_ACTUAL']))

        if config['MODEL']['gbmc'] == 'TRUE':
            print('Running GradientBoostingClassifier RAW-ACTUAL')
            raw_model = GradientBoostingClassifier(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_ACTUAL']))

        if config['MODEL']['lrc'] == 'TRUE':
            print('Running LogisticRegressionCV RAW-ACTUAL')
            raw_model = LogisticRegressionCV(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_ACTUAL']))

        if config['MODEL']['svc'] == 'TRUE':
            print('Running SVC RAW-ACTUAL')
            raw_model = SVC(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_ACTUAL']))

        if config['MODEL']['rfc'] == 'TRUE':
            print('Running RandomForestClassifier RAW-ACTUAL')
            raw_model = RandomForestClassifier(**ast.literal_eval(config['PARAMS']['raw_model_hyperparams_ACTUAL']))


        # raw_model_actual = pickle.load(open(raw_model_actual_filename, 'rb'))


        CalcResults(model=raw_model_actual, X=X_train, y=y_train_actual, predictions='Train Predictions',
                    stop=stop_actual, strong_cap=strong_cap_actual, med_cap=med_cap_actual,
                    multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                    HL=False, NN=False).calc_results()

        CalcResults(model=raw_model_actual, X=X_val, y=y_val_actual, predictions='Val Predictions',
                    stop=stop_actual, strong_cap=strong_cap_actual, med_cap=med_cap_actual,
                    multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                    HL=False, NN=False).calc_results()

        CalcResults(model=raw_model_actual, X=X_test, y=y_test_actual, predictions='Test Predictions',
                    stop=stop_actual, strong_cap=strong_cap_actual, med_cap=med_cap_actual,
                    multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                    HL=False, NN=False).calc_results()

    # Actual SMOTE
    if config['PARAMS']['SMOTE_ON'] == 'TRUE':


        if config['MODEL']['xgbc'] == 'TRUE':
            print('Running XGBClassifier SMOTE-ACTUAL')
            model_name = 'XGBC'
            smote_model = XGBClassifier(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_ACTUAL']))

        if config['MODEL']['lgbmc'] == 'TRUE':
            print('Running LGBMClassifier SMOTE-ACTUAL')
            model_name = 'LGBMC'
            smote_model = LGBMClassifier(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_ACTUAL']))

        if config['MODEL']['gbmc'] == 'TRUE':
            print('Running GradientBoostingClassifier SMOTE-ACTUAL')
            model_name = 'GBMC'
            smote_model = GradientBoostingClassifier(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_ACTUAL']))

        if config['MODEL']['lrc'] == 'TRUE':
            print('Running LogisticRegressionCV SMOTE-ACTUAL')
            model_name = 'LRC'
            smote_model = LogisticRegressionCV(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_ACTUAL']))

        if config['MODEL']['svc'] == 'TRUE':
            print('Running SVC SMOTE-ACTUAL')
            model_name = 'SVC'
            smote_model = SVC(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_ACTUAL']))

        if config['MODEL']['rfc'] == 'TRUE':
            print('Running RandomForestClassifier SMOTE-ACTUAL')
            model_name = 'RFC'
            smote_model = RandomForestClassifier(**ast.literal_eval(config['PARAMS']['smote_model_hyperparams_ACTUAL']))

        # smote_model_actual = pickle.load(open(smote_model_actual_filename, 'rb'))

        CalcResults(smote_model_actual, X_train, y_train_actual, predictions='Train Predictions',
                    stop=stop_actual, strong_cap=strong_cap_actual, med_cap=med_cap_actual,
                    multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                    HL=False, NN=False).calc_results()
        CalcResults(smote_model_actual, X_val, y_val_actual, predictions='Val Predictions',
                    stop=stop_actual, strong_cap=strong_cap_actual, med_cap=med_cap_actual,
                    multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                    HL=False, NN=False).calc_results()
        CalcResults(smote_model_actual, X_test, y_test_actual, predictions='Test Predictions',
                    stop=stop_actual, strong_cap=strong_cap_actual, med_cap=med_cap_actual,
                    multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                    HL=False, NN=False).calc_results()

if config['PARAMS']['HL_ON'] == 'TRUE':
    if config['PARAMS']['RAW_ON'] == 'TRUE':
        print(config['PARAMS']['raw_model_hyperparams_HL'])
    if config['PARAMS']['SMOTE_ON'] == 'TRUE':
        print(config['PARAMS']['smote_model_hyperparams_HL'])
        print('strong_buy_HL:', strong_buy_HL)
        print('med_buy_HL:', med_buy_HL)
        print('no_trade_HL:', no_trade_HL)
        print('med_sell_HL:', med_sell_HL)
        print('strong_sell_HL:', strong_sell_HL)
        print('strong_cap_HL:', strong_cap_HL)
        print('med_cap_HL:', med_cap_HL)
        print('stop_HL:', stop_HL)
        if config['PARAMS']['PredProb_ON'] == 'TRUE':
            print('PredProb_ON')
            print('min_prob0', min_prob0)
            print('min_prob1', min_prob1)
            print('min_prob3', min_prob3)
            print('min_prob4', min_prob4)

if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
    if config['PARAMS']['RAW_ON'] == 'TRUE':
        print(config['PARAMS']['raw_model_hyperparams_ACTUAL'])
    if config['PARAMS']['SMOTE_ON'] == 'TRUE':
        print(config['PARAMS']['smote_model_hyperparams_ACTUAL'])
