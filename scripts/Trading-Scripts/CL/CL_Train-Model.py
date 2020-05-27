import time
start_time = time.time()
import pandas as pd
import numpy as np
import feather
import matplotlib.pyplot as plt
import gc
from imblearn.over_sampling import SMOTE, SMOTENC, SVMSMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from sklearn.metrics import *
from imblearn.metrics import classification_report_imbalanced
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier, plot_importance
from catboost import CatBoostClassifier
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import configparser
import ast
import pickle
import sys
from multiprocessing import Pool, Process
import keras
# from keras.layers import *
# from keras.models import *
# from keras.preprocessing.sequence import *
# from keras.regularizers import *
# from keras.optimizers import *
# from keras.callbacks import EarlyStopping
import warnings

config = configparser.ConfigParser()
# config.read('/home/melgazar9/Trading/TD/Scripts/Trading-Scripts/CL/scripts/CL_30min_TRAIN-MODEL.ini')

config_path = config.read('/home/melgazar9/Trading/TD/Scripts/Trading-Scripts/CL/scripts/CL_{}_TRAIN-MODEL.ini'.format(sys.argv[1]))

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


if config['PARAMS']['HL_ON'] == 'TRUE':
    HL_ON = True
else:
    HL_ON = False

if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
    ACTUAL_ON = True
else:
    ACTUAL_ON = False




if config['PARAMS']['read_csv'] == 'TRUE':
    print('Reading CSV...')
    df = pd.read_csv(config['PATH']['filename'])
elif config['PARAMS']['read_feather'] == 'TRUE':
    print('Reading Feather File...')
    # df = pd.read_feather(config['PATH']['filename'], use_threads=32)
    df = feather.read_dataframe(config['PATH']['filename'])
elif config['PARAMS']['read_parquet'] == 'TRUE':
    df = pd.read_parquet(config['PATH']['filename'])


try:
    df.set_index('Datetime', inplace=True)
except KeyError:
    print('Datetime index is already set!')
    pass

df.index = pd.to_datetime(df.index)
df.index = df.index.tz_localize('utc').tz_convert('US/Central')
df.sort_index(inplace=True)
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
        while i < len(self.df):
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


print('Calculating Target...')
target_actual = CalcTarget(df, strong_buy=strong_buy_actual, med_buy=med_buy_actual, no_trade=no_trade_actual,
                            med_sell=med_sell_actual, strong_sell=strong_sell_actual, threshold=threshold,
                            stop=stop_actual).calc_target_actual()

target_HL = CalcTarget(df, strong_buy=strong_buy_HL, med_buy=med_buy_HL, no_trade=no_trade_HL,
                        med_sell=med_sell_HL, strong_sell=strong_sell_HL, threshold=threshold,
                        stop=stop_HL).calc_target_HL()

print(target_actual['Target_Actual'].value_counts())
print(target_HL['Target_HL'].value_counts())

for i in range(int(config['PARAMS']['min_target_lookback']), int(config['PARAMS']['max_target_lookback']), int(config['PARAMS']['target_lookback_increment'])):
    target_actual['PrevTarget_ActMove' + str(i)] = target_actual['Target_Actual'].shift(i)

for i in range(int(config['PARAMS']['min_target_lookback']), int(config['PARAMS']['max_target_lookback']), int(config['PARAMS']['target_lookback_increment'])):
    target_HL['PrevTarget_HL' + str(i)] = target_HL['Target_HL'].shift(i)

target_HL = target_HL.fillna(2).astype('int')
target_actual = target_actual.fillna(2).astype('int')

# df['Target_Actual'] = target_actual['Target_Actual']
# df['Target_HL'] = target_HL['Target_HL']
if config['DIM_REDUCTION_PARAMS']['dim_reduction_ON'] == 'FALSE':
    df = pd.concat([df, target_actual, target_HL], axis=1)

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

    joined_test['Year'] = joined_test.index.year.astype('category')
    joined_test['Month'] = joined_test.index.month.astype('category')
    joined_test['Week'] = joined_test.index.week.astype('category')
    joined_test['Day'] = joined_test.index.day.astype('category')
    joined_test['DayofWeek'] = joined_test.index.dayofweek.astype('category')
    joined_test['DayofYear'] = joined_test.index.dayofyear.astype('category')
    joined_test['Hour'] = joined_test.index.hour.astype('category')
    joined_test['Quarter'] = joined_test.index.quarter.astype('category')

    print('Converting continuous vars to float32 in joined and joined_test...')
    # Fill continuous variables using backfill
    # for v in cont_vars:
    #     joined[v] = joined[v].bfill().astype('float32')
    #     joined_test[v] = joined_test[v].bfill().astype('float32')
    joined[cont_vars] = joined[cont_vars].astype('float32')
    joined_test[cont_vars] = joined_test[cont_vars].astype('float32')
    print(joined.get_dtype_counts())

    # print(joined.isnull().any()[joined.isnull().any()!=False], '\n')
    # print(joined_test.isnull().any()[joined_test.isnull().any()!=False])
    # print(joined.get_dtype_counts())
    # print(joined_test.get_dtype_counts())

    print('Concatenating joined and joined_test...')
    df = pd.concat([joined, joined_test], axis=0)
    del joined, joined_test
    gc.collect()
    print('Joined df: \n', df)

    # One-hot encode categorical features
    if config['PARAMS']['one_hot_features_ON'] == 'TRUE':
        print('Getting Dummies...')
        df = pd.get_dummies(df, columns=[i for i in cat_vars], drop_first=True)
        df.rename(columns={'Overnight_or_Intraday_1.0':'Overnight_or_Intraday'}, inplace=True)

    cat_vars = [i for i in df.columns if not i in cont_vars]

    if config['PARAMS']['need_cont_vars'] == 'FALSE':
        print('Converting dtypes...')
        for col in cat_vars:
            df[col] = df[col].astype('category').cat.as_ordered()
            gc.collect()
        for col in cont_vars:
            df[col] = df[col].astype('float32')
            gc.collect()

elif config['DIM_REDUCTION_PARAMS']['dim_reduction_ON'] == 'TRUE' and config['DIM_REDUCTION_PARAMS']['read_dim_reduction_ON'] == 'TRUE':
    if config['DIM_REDUCTION_PARAMS']['read_csv'] == 'TRUE':
        print('Reading Dimensionality-Reduction CSV...')
        df2 = pd.read_csv(config['PATH']['dim_reduction_filename'])
    elif config['DIM_REDUCTION_PARAMS']['read_feather'] == 'TRUE':
        print('Reading Dimensionality-Reduction Feather File...')
        df2 = feather.read_dataframe(config['PATH']['dim_reduction_filename'])
    elif config['DIM_REDUCTION_PARAMS']['read_parquet'] == 'TRUE':
        print('Reading Dimensionality-Reduction Parquet File...')
        df2 = pd.read_parquet(config['PATH']['dim_reduction_filename'])

    # df = pd.concat([df, target_actual, target_HL], axis=1)
    target_cols = pd.concat([df[[Actual_Move, Actual_HighMove, Actual_LowMove]], target_actual['Target_Actual'], target_HL['Target_HL']], axis=1)
    df = pd.merge_asof(df2, target_cols, left_index=True, right_index=True)
    # df = pd.concat([df2, target_cols], axis=1)
    print(df)
    del target_cols, df2
    gc.collect()


#################################################################
#       TRAIN TEST SPLIT FOR ACTUAL-MOVE and HL STRATEGIES
#################################################################

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
    # print(list(X_test.columns))

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

if config['DIM_REDUCTION_PARAMS']['dim_reduction_ON'] == 'FALSE':

    cont_vars = [i for i in cont_vars if not i.startswith('Target') and not i.startswith('Actual')]

    X_train[[i for i in X_train.columns if i.startswith('PrevTarget')]] = X_train[[i for i in X_train.columns if i.startswith('PrevTarget')]].ffill()
    X_val[[i for i in X_train.columns if i.startswith('PrevTarget')]] = X_val[[i for i in X_val.columns if i.startswith('PrevTarget')]].ffill()
    X_test[[i for i in X_train.columns if i.startswith('PrevTarget')]] = X_test[[i for i in X_test.columns if i.startswith('PrevTarget')]].ffill()

    # Scale Data
    if config['PARAMS']['scale_ON'] == 'TRUE':

        print('Scaling Data...')
        scaler = eval(config['PARAMS']['scaler']).fit(np.array(X_train))

        X_train_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), index=X_val.index, columns=X_val.columns)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        X_train_scaled[[i for i in X_train.columns if i.startswith('PrevTarget')]] = X_train_scaled[[i for i in X_train_scaled.columns if i.startswith('PrevTarget')]].ffill()
        X_val_scaled[[i for i in X_train.columns if i.startswith('PrevTarget')]] = X_val_scaled[[i for i in X_val_scaled.columns if i.startswith('PrevTarget')]].ffill()
        X_test_scaled[[i for i in X_train.columns if i.startswith('PrevTarget')]] = X_test_scaled[[i for i in X_test_scaled.columns if i.startswith('PrevTarget')]].ffill()

        # for col in df.select_dtypes('uint8').columns:
        if config['PARAMS']['need_cont_vars'] == 'FALSE':
            print('Converting datatypes on scaled dataset...')
            for col in cat_vars:
                #print(col)
                X_train_scaled[col] = X_train_scaled[col].astype('category').cat.as_ordered()
                X_val_scaled[col] = X_val_scaled[col].astype('category').cat.as_ordered()
                X_test_scaled[col] = X_test_scaled[col].astype('category').cat.as_ordered()

            for col in cont_vars:
                X_train_scaled[col] = X_train_scaled[col].astype('float32')
                X_val_scaled[col] = X_val_scaled[col].astype('float32')
                X_test_scaled[col] = X_test_scaled[col].astype('float32')

        elif config['PARAMS']['need_cont_vars'] == 'TRUE':
            X_train_scaled = X_train_scaled.astype('float32')
            X_val_scaled = X_val_scaled.astype('float32')
            X_test_scaled = X_test_scaled.astype('float32')

        if config['PARAMS']['save_scaler'] == 'TRUE':
            pickle.dump(scaler, open(config['PATH']['scaler_outpath'] + config['PARAMS']['scaler']  + str(datetime.datetime.today().date()) + '.pickle.dat', 'wb'))

if config['PARAMS']['one_hot_targets_ON'] == 'TRUE':
    print('Converting categorical targets to one-hot encoding targets')
    y_train_oh_actual = keras.utils.to_categorical(y_train_actual, num_class=5)
    y_val_oh_actual = keras.utils.to_categorical(y_val_actual, num_class=5)
    y_test_oh_actual = keras.utils.to_categorical(y_test_actual, num_class=5)

    y_train_oh_HL = keras.utils.to_categorical(y_train_HL, num_class=5)
    y_val_oh_HL = keras.utils.to_categorical(y_val_HL, num_class=5)
    y_test_oh_HL = keras.utils.to_categorical(y_test_HL, num_class=5)

    gc.collect()

##################################################################
#                       MACHINE LEARNING
##################################################################


class TrainModel():

    def __init__(self, model, X_train_fit, y_train_fit, X_val_fit, y_val_fit,
                 X_test_unseen, y_test_unseen, need_cont_vars, plot_importances=True):

        self.model = model
        self.X_train_fit = X_train_fit
        self.y_train_fit = y_train_fit
        self.X_val_fit = X_val_fit
        self.y_val_fit = y_val_fit
        self.X_test_unseen = X_test_unseen
        self.y_test_unseen = y_test_unseen
        self.need_cont_vars = need_cont_vars
        self.plot_importances = plot_importances


    def plot_feature_importances_traditional(self):

        super().__init__()

        features = self.X_train_fit.columns
        importances = self.model.feature_importances_
        indices = np.argsort(importances)

        plt.figure(figsize=(30, 150))
        plt.title('Feature Importances')
        plt.barh(y=range(len(indices)), width=importances[indices], height=.5, color='c', align='center')
        plt.yticks(ticks=range(len(indices)), labels=[features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

        return

    def train_traditional_ML_model(self):

        super().__init__()

        if self.need_cont_vars == True:
            self.X_train_fit = self.X_train_fit.astype('float32')
            self.X_val_fit = self.X_val_fit.astype('float32')
            self.X_test_unseen = self.X_test_unseen.astype('float32')


        try:
            self.model.fit(self.X_train.fit, self.y_train_fit, eval_set={self.X_val_fit, self.y_val_fit},
                           early_stopping_rounds=int(config['PARAMS']['early_stopping_rounds']),
                           eval_metric=ast.literal_eval(config['PARAMS']['eval_metric']))
        except:
            self.model.fit(self.X_train_fit, self.y_train_fit)

        # if config['MODEL']['predict_concatenated_array'] == 'FALSE':
        print('Train Accuracy:', accuracy_score(self.model.predict(self.X_train_fit), self.y_train_fit))
        print('Val Accuracy:', accuracy_score(self.model.predict(self.X_val_fit), self.y_val_fit))
        print('Test Accuracy:', accuracy_score(self.model.predict(self.X_test_unseen), self.y_test_unseen))

        print('Train F1:', f1_score(self.model.predict(self.X_train_fit), self.y_train_fit, average='weighted'))
        print('Val F1:', f1_score(self.model.predict(self.X_val_fit), self.y_val_fit, average='weighted'))
        print('Test F1:', f1_score(self.model.predict(self.X_test_unseen), self.y_test_unseen, average='weighted'))

        print('Train:', classification_report_imbalanced(self.model.predict(self.X_train_fit), self.y_train_fit))
        print('Val:', classification_report_imbalanced(self.model.predict(self.X_val_fit), self.y_val_fit))
        print('Test:', classification_report_imbalanced(self.model.predict(self.X_test_unseen), self.y_test_unseen))

        if self.plot_importances == True:
            self.plot_feature_importances_traditional()

        return self.model

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
        try:
            print(self.X.get_dtype_counts())
        except AttributeError:
            print('Cannot print dtypes of numpy array. Dimensionality Reduction is probably running')
            pass

        if config['PARAMS']['PredProb_ON'] == 'FALSE':
            # if config['MODEL']['predict_concatenated_array'] == 'FALSE':
            pred = pd.DataFrame(self.model.predict(self.X), index=self.X.index).rename(columns={0:self.predictions})
            # elif config['MODEL']['predict_concatenated_array'] == 'TRUE':
            #     pred = pd.DataFrame(np.concatenate(self.model.predict(self.X)).ravel(), index=self.X.index).rename(columns={0:self.predictions})
        elif config['PARAMS']['PredProb_ON'] == 'TRUE':
            # if config['MODEL']['predict_concatenated_array'] == 'FALSE':
            pred = pd.DataFrame(self.calc_predictions(), index=self.X.index).rename(columns={0:self.predictions})
            # elif config['MODEL']['predict_concatenated_array'] == 'TRUE':
            #     pred = pd.DataFrame(np.concatenate(self.calc_predictions()).ravel(), index=self.X.index).rename(columns={0:self.predictions})
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

        pred = pd.DataFrame(np.concatenate(model.predict(X)).ravel(), index=X.index).rename(columns={0:self.predictions})
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


def create_oversampled_df(X_train_fit, X_val_fit, HL, scaler, need_cont_vars=False, ratio=None):

    if config['OVERSAMPLING_PARAMS']['OverSampling_algorithm'] == 'SMOTE':
        print('Applying SMOTE...')
        sm = SMOTE(n_jobs=32, random_state=int(config['OVERSAMPLING_PARAMS']['random_state']), ratio=ratio)
    elif config['OVERSAMPLING_PARAMS']['OverSampling_algorithm'] == 'SMOTENC':
        print('Applying SMOTENC...')
        sm = SMOTENC(categorical_features=[X_train_fit.columns.get_loc(c) for c in cat_vars], n_jobs=32, k_neighbors=int(config['OVERSAMPLING_PARAMS']['kneighbors']), random_state=int(config['OVERSAMPLING_PARAMS']['random_state']))
        # print('Categorical Features:', list(df.iloc[:, [X_train_fit.columns.get_loc(c) for c in cat_vars]].columns), len([X_train_fit.columns.get_loc(c) for c in cat_vars]))
    elif config['OVERSAMPLING_PARAMS']['OverSampling_algorithm'] == 'SVMSMOTE':
        print('Applying SVMSMOTE...')
        sm = SVMSMOTE(out_step=float(config['OVERSAMPLING_PARAMS']['out_step']), n_jobs=32, sampling_strategy=config['OVERSAMPLING_PARAMS']['sampling_strategy'], random_state=int(config['OVERSAMPLING_PARAMS']['random_state']))
    elif config['OVERSAMPLING_PARAMS']['OverSampling_algorithm'] == 'RandomOverSampler':
        print('Applying RandomOverSampler...')
        sm = RandomOverSampler(config['OVERSAMPLING_PARAMS']['sampling_strategy'], random_state=int(config['OVERSAMPLING_PARAMS']['random_state']), return_indices=ast.literal_eval(config['OVERSAMPLING_PARAMS']['return_indices']))
    else:
        print('OVERSAMPLED_ON and CREATE_OVERSAMPLED_DF_ON must be set to either TRUE or FALSE')
        sys.exit()

    if config['DIM_REDUCTION_PARAMS']['dim_reduction_ON'] == 'TRUE' and config['DIM_REDUCTION_PARAMS']['read_dim_reduction_ON'] and config['DIM_REDUCTION_PARAMS']['dim_reduction_scaler_ON'] == 'TRUE':
        scaler = pickle.load(open(config['PATH']['dim_reduction_scaler'], 'rb'))


    X_temp = pd.concat([X_train_fit, X_val_fit], axis=0).reset_index().drop('Datetime', axis=1)
    gc.collect()

    if config['PARAMS']['HL_ON'] == 'FALSE':

        y_temp = pd.concat([y_train_actual, y_val_actual], axis=0)
        X_resampled, y_resampled = sm.fit_resample(X_temp, y_temp.ravel())
        gc.collect()
        print('Original dataset shape:', Counter(y_temp))

        X_train_oversampled = pd.DataFrame(X_resampled, columns=X_train_fit.columns).iloc[0:int(len(X_resampled)*.7)].fillna(0)
        X_val_oversampled = pd.DataFrame(X_resampled, columns=X_train_fit.columns).iloc[int(len(X_resampled)*.7):].fillna(0)
        y_train_oversampled = pd.Series(y_resampled, name='Target_Actual').iloc[0:int(len(y_resampled)*.7)].fillna(2)
        y_val_oversampled = pd.Series(y_resampled, name='Target_Actual').iloc[int(len(y_resampled)*.7):].fillna(2)

    elif config['PARAMS']['HL_ON'] == 'TRUE':

        y_temp = pd.concat([y_train_HL, y_val_HL], axis=0)
        X_resampled, y_resampled = sm.fit_resample(X_temp, y_temp.ravel())
        gc.collect()
        print('Original dataset shape:', Counter(y_temp))

        X_train_oversampled = pd.DataFrame(X_resampled, columns=X_train_fit.columns).iloc[0:int(len(X_resampled)*.7)].fillna(0)
        X_val_oversampled = pd.DataFrame(X_resampled, columns=X_train_fit.columns).iloc[int(len(X_resampled)*.7):].fillna(0)
        y_train_oversampled = pd.Series(y_resampled, name='Target_HL').iloc[0:int(len(y_resampled)*.7)].fillna(2)
        y_val_oversampled = pd.Series(y_resampled, name='Target_HL').iloc[int(len(y_resampled)*.7):].fillna(2)

    print('Resampled dataset shape:', Counter(y_resampled))

    if need_cont_vars == True:
        X_train_oversampled = X_train_oversampled.astype('float32')
        X_val_oversampled = X_val_oversampled.astype('float32')
    print(X_train_oversampled)
    if config['OVERSAMPLING_PARAMS']['scale_oversampled_df'] == 'TRUE':
        X_train_oversampled = pd.DataFrame(scaler.transform(X_train_oversampled), columns=X_train.columns)
        X_val_oversampled = pd.DataFrame(scaler.transform(X_val_oversampled), columns=X_val.columns)

    if config['OVERSAMPLING_PARAMS']['save_oversampled_df'] == 'TRUE' and config['OVERSAMPLING_PARAMS']['CREATE_OVERSAMPLED_DF_ON'] == 'TRUE':
        X_oversampled = pd.concat([X_train_oversampled, X_val_oversampled], axis=0)
        y_oversampled = pd.concat([y_train_oversampled, y_val_oversampled], axis=0)
        oversampled_df = pd.concat([X_oversampled.reset_index(drop=True), y_oversampled.reset_index(drop=True)], axis=1)
        oversampled_df = oversampled_df.rename(columns={'0':'Target_HL'})
        # print(oversampled_df.isnull().sum()[oversampled_df.isnull().sum() > 0])
        if config['OVERSAMPLING_PARAMS']['save_oversampled_df_as_csv'] == 'TRUE':
            print('Saving OverSampled df as CSV...')
            oversampled_df.to_csv(config['PATH']['oversampled_df_outpath'] + config['NAME']['product'][0:2] + '_' + config['OVERSAMPLING_PARAMS']['OverSampling_algorithm'] + '_DF_' + str(datetime.datetime.today().date()) + '.csv')
        elif config['OVERSAMPLING_PARAMS']['save_oversampled_df_as_feather'] == 'TRUE':
            print('Saving OverSampled df as Feather File...')
            oversampled_df.to_feather(config['PATH']['oversampled_df_outpath'] + config['NAME']['product'][0:2] + '_' + config['OVERSAMPLING_PARAMS']['OverSampling_algorithm'] + '_DF_' + str(datetime.datetime.today().date()) + '.feather')
        elif config['OVERSAMPLING_PARAMS']['save_oversampled_df_as_parquet'] == 'TRUE':
            print('Saving OverSampled df as Parquet File...')
            oversampled_df.to_parquet(config['PATH']['oversampled_df_outpath'] + config['NAME']['product'][0:2] + '_' + config['OVERSAMPLING_PARAMS']['OverSampling_algorithm'] + '_DF_' + str(datetime.datetime.today().date()) + '.parquet')
        print('OverSampled df Saved!')
        del oversampled_df, X_oversampled, y_oversampled
        gc.collect()

    if need_cont_vars == False and config['DIM_REDUCTION_PARAMS']['dim_reduction_ON'] == 'FALSE':
        print('Converting dtypes on OverSampled df...')
        for col in [i for i in cat_vars if not 'Actual' in i]:
            X_train_oversampled[col] = X_train_oversampled[col].astype('category').cat.as_ordered()
            X_val_oversampled[col] = X_val_oversampled[col].astype('category').cat.as_ordered()
            gc.collect()
        for col in [i for i in cont_vars if not 'Actual' in i]:
            X_train_oversampled[col] = X_train_oversampled[col].astype('float32')
            X_val_oversampled[col] = X_val_oversampled[col].astype('float32')
            gc.collect()

    if config['PARAMS']['HL_ON'] == 'FALSE':
        print(X_train_oversampled.shape, X_val_oversampled.shape, X_test.shape, y_train_oversampled.shape, y_val_oversampled.shape, y_test_actual.shape)
    elif config['PARAMS']['HL_ON'] == 'TRUE':
        print(X_train_oversampled.shape, X_val_oversampled.shape, X_test.shape, y_train_oversampled.shape, y_val_oversampled.shape, y_test_HL.shape)

    print(X_train_oversampled.get_dtype_counts(), X_val_oversampled.get_dtype_counts())


    return X_train_oversampled, X_val_oversampled, y_train_oversampled, y_val_oversampled
print(df)
def main():

    # for setting all values to be either continuous or to have categorical variables
    if config['PARAMS']['need_cont_vars'] == 'TRUE':
        need_cont_vars_param = True
    elif config['PARAMS']['need_cont_vars'] == 'FALSE':
        need_cont_vars_param = False
    if config['PARAMS']['plot_importances'] == 'TRUE':
        plot_importances_param = True
    elif config['PARAMS']['plot_importances'] == 'FALSE':
        plot_importances_param = False



    if config['OVERSAMPLING_PARAMS']['SMOTE_RATIO'] == 'FALSE' or config['OVERSAMPLING_PARAMS']['SMOTE_RATIO'] == 'None':
        smote_ratio_param = None
    else:
        smote_ratio_param = ast.literal_eval(config['PARAMS']['SMOTE_RATIO'])

    # print(pd.concat([X_train, y_train_HL], axis=1))

    ####################################################
    #                   MODEL TRAINING
    ####################################################


    if config['PARAMS']['HL_ON'] == 'TRUE':

        if config['PARAMS']['RAW_ON'] == 'TRUE':

            # HL RAW
            raw_model = eval(config['MODEL']['model'])
            raw_model = raw_model.set_params(**eval(config['PARAMS']['raw_model_hyperparams_HL']))

            if config['PARAMS']['scale_ON'] == 'FALSE':

                print('******Running', type(raw_model).__name__, 'Raw-HL (not scaled)******')

                raw_model_HL = TrainModel(model=raw_model,
                                          X_train_fit=X_train, y_train_fit=y_train_HL, X_val_fit=X_val, y_val_fit=y_val_HL,
                                          X_test_unseen=X_test, y_test_unseen=y_test_HL,
                                          need_cont_vars=need_cont_vars_param,
                                          plot_importances=plot_importances_param).train_traditional_ML_model()


                CalcResults(model=raw_model_HL, X=X_train, y=y_train_HL, predictions='Train Predictions',
                            stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                            multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                            HL=True, NN=False).calc_results()

                CalcResults(model=raw_model_HL, X=X_val, y=y_val_HL, predictions='Val Predictions',
                            stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                            multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                            HL=True, NN=False).calc_results()

                CalcResults(model=raw_model_HL, X=X_test, y=y_test_HL, predictions='Test Predictions',
                            stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                            multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                            HL=True, NN=False).calc_results()

            if config['PARAMS']['scale_ON'] == 'TRUE':

                print('******Running', type(raw_model).__name__, 'Raw-HL (not scaled)******')

                raw_model_HL = TrainModel(model=raw_model,
                                          X_train_fit=X_train_scaled, y_train_fit=y_train_HL, X_val_fit=X_val_scaled, y_val_fit=y_val_HL,
                                          X_test_unseen=X_test_scaled, y_test_unseen=y_test_HL,
                                          need_cont_vars=need_cont_vars_param,
                                          plot_importances=plot_importances_param).train_traditional_ML_model()


                CalcResults(model=raw_model_HL, X=X_train_scaled, y=y_train_HL, predictions='Train Predictions',
                            stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                            multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                            HL=True, NN=False).calc_results()

                CalcResults(model=raw_model_HL, X=X_val_scaled, y=y_val_HL, predictions='Val Predictions',
                            stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                            multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                            HL=True, NN=False).calc_results()

                CalcResults(model=raw_model_HL, X=X_test_scaled, y=y_test_HL, predictions='Test Predictions',
                            stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                            multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                            HL=True, NN=False).calc_results()



        if config['OVERSAMPLING_PARAMS']['OVERSAMPLED_ON'] == 'TRUE':

            # HL OverSampled

            if config['PARAMS']['scale_ON'] == 'FALSE':
                print('Applying OverSampling (not scaled)...')
                X_train_oversampled, X_val_oversampled, y_train_oversampled_HL, y_val_oversampled_HL = create_oversampled_df(X_train_fit=X_train, X_val_fit=X_val, HL=True, scaler=scaler, need_cont_vars=need_cont_vars_param, ratio=smote_ratio_param)
                print(X_train_oversampled.shape, X_val_oversampled.shape, y_train_oversampled_HL.shape, y_val_oversampled_HL.shape)

            elif config['PARAMS']['scale_ON'] == 'TRUE':
                print('Applying OverSampling (scaled first, then oversample)...')
                X_train_oversampled, X_val_oversampled, y_train_oversampled_HL, y_val_oversampled_HL = create_oversampled_df(X_train_fit=X_train_scaled, X_val_fit=X_val_scaled, HL=True, scaler=scaler, need_cont_vars=need_cont_vars_param, ratio=smote_ratio_param)
                print(X_train_oversampled.shape, X_val_oversampled.shape, y_train_oversampled_HL.shape, y_val_oversampled_HL.shape)

            elif config['OVERSAMPLING_PARAMS']['CREATE_OVERSAMPLED_DF_ON'] == 'FALSE':

                print("Using 'oversampled_read_df' HL as training data...")
                print('Reading OverSampled df...')

                if config['OVERSAMPLING_PARAMS']['read_feather'] == 'FALSE':
                    oversampled_df = pd.read_csv(config['PATH']['oversampled_read_df'])
                    try:
                        oversampled_df = oversampled_df.drop(['Unnamed: 0'], axis=1)
                    except KeyError:
                        pass
                elif config['OVERSAMPLING_PARAMS']['read_feather'] == 'TRUE':
                    oversampled_df = pd.read_feather(config['PATH']['oversampled_read_df'], use_threads=32)
                    try:
                        oversampled_df = oversampled_df.drop(['Unnamed: 0'], axis=1)
                    except KeyError:
                        pass


                X_train_oversampled = oversampled_df.iloc[0:int(len(oversampled_df)*.7)].fillna(0).drop('Target_HL', axis=1)
                X_val_oversampled = oversampled_df.iloc[int(len(oversampled_df)*.7):].fillna(0).drop('Target_HL', axis=1)
                y_train_oversampled_HL = oversampled_df['Target_HL'].iloc[0:int(len(oversampled_df)*.7)].fillna(2)
                y_val_oversampled_HL = oversampled_df['Target_HL'].iloc[int(len(oversampled_df)*.7):].fillna(2)

                if need_cont_vars_param == False:
                    print('Converting dtypes on OverSampled df...')
                    for col in [i for i in cat_vars if not 'Actual' in i]:
                        X_train_oversampled[col] = X_train_oversampled[col].astype('category').cat.as_ordered()
                        X_val_oversampled[col] = X_val_oversampled[col].astype('category').cat.as_ordered()
                    for col in [i for i in cont_vars if not 'Actual' in i]:
                        X_train_oversampled[col] = X_train_oversampled[col].astype('float32')
                        X_val_oversampled[col] = X_val_oversampled[col].astype('float32')

                print(X_train_oversampled.shape, X_val_oversampled.shape, y_train_oversampled_HL.shape, y_val_oversampled_HL.shape)


        ########################################################
            oversampled_model = eval(config['MODEL']['model'])
            oversampled_model = oversampled_model.set_params(**eval(config['PARAMS']['oversampled_model_hyperparams_HL']))
            model_name = str(oversampled_model)
            print('******Running', type(oversampled_model).__name__, 'OverSampled-HL******')

            oversampled_model_HL = TrainModel(oversampled_model,
                                        X_train_oversampled, y_train_oversampled_HL, X_val_oversampled, y_val_oversampled_HL,
                                        X_test, y_test_HL,
                                        need_cont_vars=need_cont_vars_param,
                                        plot_importances=plot_importances_param).train_traditional_ML_model()


            CalcResults(oversampled_model_HL, X_train, y_train_HL, predictions='Train Predictions',
                        stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                        multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                        HL=True, NN=False).calc_results()
            CalcResults(oversampled_model_HL, X_val, y_val_HL, predictions='Val Predictions',
                        stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                        multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                        HL=True, NN=False).calc_results()
            CalcResults(oversampled_model_HL, X_test, y_test_HL, predictions='Test Predictions',
                        stop=stop_HL, strong_cap=strong_cap_HL, med_cap=med_cap_HL,
                        multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                        HL=True, NN=False).calc_results()


    if config['PARAMS']['ACTUAL_ON'] == 'TRUE':

        if config['PARAMS']['RAW_ON'] == 'TRUE':

            # Actual RAW
            raw_model = eval(config['MODEL']['model'])
            raw_model = raw_model.set_params(**eval(config['PARAMS']['raw_model_hyperparams_ACTUAL']))
            model_name = str(raw_model)
            print('******Running', type(raw_model).__name__, 'Raw-Actual******')

            raw_model_actual = TrainModel(raw_model,
                                          X_train, y_train_actual, X_val, y_val_actual,
                                          X_test, y_test_actual,
                                          need_cont_vars=need_cont_vars_param,
                                          plot_importances=plot_importances_param).train_traditional_ML_model()


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

        if config['OVERSAMPLING_PARAMS']['OVERSAMPLED_ON'] == 'TRUE':
            # Actual SMOTE
            if config['OVERSAMPLING_PARAMS']['CREATE_OVERSAMPLED_DF_ON'] == 'TRUE':
                if config['PARAMS']['scale_ON'] == 'FALSE':
                    X_train_oversampled, X_val_oversampled, y_train_oversampled_actual, y_val_oversampled_actual = create_oversampled_df(X_train_fit=X_train, X_val_fit=X_val, HL=False, scaler=scaler, need_cont_vars=need_cont_vars_param, ratio=smote_ratio_param)
                    print(X_train_oversampled.shape, X_val_oversampled.shape, y_train_oversampled_actual.shape, y_val_oversampled_actual.shape)
                elif config['PARAMS']['scale_ON'] == 'TRUE':
                    X_train_oversampled, X_val_oversampled, y_train_oversampled_actual, y_val_oversampled_actual = create_oversampled_df(X_train_fit=X_train_scaled, X_val_fit=X_val_scaled, HL=False, scaler=scaler, need_cont_vars=need_cont_vars_param, ratio=smote_ratio_param)
                    print(X_train_oversampled.shape, X_val_oversampled.shape, y_train_oversampled_actual.shape, y_val_oversampled_actual.shape)

            elif config['OVERSAMPLING_PARAMS']['CREATE_OVERSAMPLED_DF_ON'] == 'FALSE':

                print("Using 'oversampled_read_df' ACTUAL-MOVE as training data...")
                print('Reading SMOTE df...')

                if config['OVERSAMPLING_PARAMS']['read_feather'] == 'FALSE':
                    oversampled_df = pd.read_csv(config['PATH']['oversampled_read_df'])
                    try:
                        oversampled_df = oversampled_df.drop(['Unnamed: 0'], axis=1)
                    except KeyError:
                        pass
                elif config['OVERSAMPLING_PARAMS']['read_feather'] == 'TRUE':
                    oversampled_df = pd.read_feather(config['PATH']['oversampled_read_df'], use_threads=32)
                    try:
                        oversampled_df = oversampled_df.drop(['Unnamed: 0'], axis=1)
                    except KeyError:
                        pass

                X_train_oversampled = oversampled_df.iloc[0:int(len(oversampled_df)*.7)].fillna(0).drop('Target_Actual', axis=1)
                X_val_oversampled = oversampled_df.iloc[int(len(oversampled_df)*.7):].fillna(0).drop('Target_Actual', axis=1)
                y_train_oversampled_actual = oversampled_df['Target_Actual'].iloc[0:int(len(oversampled_df)*.7)].fillna(2)
                y_val_oversampled_actual = oversampled_df['Target_Actual'].iloc[int(len(oversampled_df)*.7):].fillna(2)

                print(X_train_oversampled.shape, X_val_oversampled.shape, y_train_oversampled_actual.shape, y_val_oversampled_actual.shape)

            oversampled_model = eval(config['MODEL']['model'])
            oversampled_model = oversampled_model.set_params(**eval(config['PARAMS']['oversampled_model_hyperparams_ACTUAL']))
            model_name = str(oversampled_model)
            print('******Running', type(oversampled_model).__name__, 'Oversampled-Actual******')

            oversampled_model_actual = TrainModel(oversampled_model,
                                            X_train_oversampled, y_train_oversampled_actual, X_val_oversampled, y_val_oversampled_actual,
                                            X_test, y_test_actual,
                                            need_cont_vars=need_cont_vars_param,
                                            plot_importances=plot_importances_param).train_traditional_ML_model()

            CalcResults(oversampled_model_actual, X_train, y_train_actual, predictions='Train Predictions',
                        stop=stop_actual, strong_cap=strong_cap_actual, med_cap=med_cap_actual,
                        multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                        HL=False, NN=False).calc_results()
            CalcResults(oversampled_model_actual, X_val, y_val_actual, predictions='Val Predictions',
                        stop=stop_actual, strong_cap=strong_cap_actual, med_cap=med_cap_actual,
                        multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                        HL=False, NN=False).calc_results()
            CalcResults(oversampled_model_actual, X_test, y_test_actual, predictions='Test Predictions',
                        stop=stop_actual, strong_cap=strong_cap_actual, med_cap=med_cap_actual,
                        multiplier=multiplier, need_cont_vars=need_cont_vars_param,
                        HL=False, NN=False).calc_results()

    if config['PARAMS']['HL_ON'] == 'TRUE':
        if config['PARAMS']['RAW_ON'] == 'TRUE':
            print(config['PARAMS']['raw_model_hyperparams_HL'])
        if config['OVERSAMPLING_PARAMS']['OVERSAMPLED_ON'] == 'TRUE':
            print(config['PARAMS']['oversampled_model_hyperparams_HL'])
            print('strong_buy_HL:', strong_buy_HL)
            print('med_buy_HL:', med_buy_HL)
            print('no_trade_HL:', no_trade_HL)
            print('med_sell_HL:', med_sell_HL)
            print('strong_sell_HL:', strong_sell_HL)
            print('strong_cap_HL:', strong_cap_HL)
            print('med_cap_HL:', med_cap_HL)
            print('stop_HL:', stop_HL)

    if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
        if config['PARAMS']['RAW_ON'] == 'TRUE':
            print(config['PARAMS']['raw_model_hyperparams_ACTUAL'])
        if config['OVERSAMPLING_PARAMS']['OVERSAMPLED_ON'] == 'TRUE':
            print(config['PARAMS']['oversampled_model_hyperparams_ACTUAL'])
            print('strong_buy_ACTUAL:', strong_buy_actual)
            print('med_buy_ACTUAL:', med_buy_actual)
            print('no_trade_ACTUAL:', no_trade_actual)
            print('med_sell_ACTUAL:', med_sell_actual)
            print('strong_sell_ACTUAL:', strong_sell_actual)
            print('strong_cap_ACTUAL:', strong_cap_actual)
            print('med_cap_ACTUAL:', med_cap_actual)
            print('stop_ACTUAL:', stop_actual)

    if config['PARAMS']['PredProb_ON'] == 'TRUE':
        print('PredProb_ON...')
        print('min_prob0', min_prob0)
        print('min_prob1', min_prob1)
        print('min_prob3', min_prob3)
        print('min_prob4', min_prob4)

    if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
        if config['PARAMS']['RAW_ON'] == 'TRUE':
            print(config['PARAMS']['raw_model_hyperparams_ACTUAL'])
        if config['OVERSAMPLING_PARAMS']['CREATE_OVERSAMPLED_DF_ON'] == 'TRUE':
            print(config['PARAMS']['oversampled_model_hyperparams_ACTUAL'])


    try:
        # SAVE RAW
        if config['PARAMS']['save_raw_model'] == 'TRUE' and config['PARAMS']['RAW_ON'] == 'TRUE':
            if config['PARAMS']['HL_ON'] == 'TRUE':
                pickle.dump(raw_model_HL, open(config['PATH']['model_outpath'] + config['NAME']['product'] + str(model_name)[0:4] + '_RAW-HL_' + str(datetime.datetime.today().date()) + '.pickle.dat', 'wb'))
                # np.savetxt(config['PATH']['model_outpath'] + config['NAME']['product'] + model_name + '_RAW-HL_COLUMNS.txt', np.array(X_train.columns), fmt='%s', delimiter=',')
                X_train.to_csv(config['PATH']['model_outpath'] + config['NAME']['product'] + str(model_name)[0:4] + '_RAW-HL_X_train_' + str(datetime.datetime.today().date()) + '.csv')

            if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
                pickle.dump(raw_model_actual, open(config['PATH']['model_outpath'] + config['NAME']['product'] + str(model_name)[0:4] + '_RAW-ACTUAL_' + str(datetime.datetime.today().date()) + '.pickle.dat', 'wb'))
                # np.savetxt(config['PATH']['model_outpath'] + config['NAME']['product'] + model_name + '_RAW-ACTUAL_COLUMNS.txt', np.array(X_train.columns), fmt='%s', delimiter=',')
                X_train.to_csv(config['PATH']['model_outpath'] + config['NAME']['product'] + str(model_name)[0:4] + '_RAW-ACTUAL_X_train_' + str(datetime.datetime.today().date()) + '.csv')


        # SAVE SMOTE
        if config['OVERSAMPLING_PARAMS']['save_oversampled_model'] == 'TRUE' and config['OVERSAMPLING_PARAMS']['OVERSAMPLED_ON'] == 'TRUE':
            if config['PARAMS']['HL_ON'] == 'TRUE':
                pickle.dump(oversampled_model_HL, open(config['PATH']['model_outpath'] + config['NAME']['product'] + str(model_name)[0:4] + '_oversampled-HL_' + str(datetime.datetime.today().date()) + '.pickle.dat', 'wb'))
                # np.savetxt(config['PATH']['model_outpath'] + config['NAME']['product'] + model_name + '_oversampled-HL_COLUMNS.txt', np.array(X_train.columns), fmt='%s', delimiter=',')
                X_train.to_csv(config['PATH']['model_outpath'] + config['NAME']['product'] + str(model_name)[0:4] + '_oversampled-HL_X_train_' + str(datetime.datetime.today().date()) + '.csv')

            if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
                pickle.dump(oversampled_model_actual, open(config['PATH']['model_outpath'] + config['NAME']['product'] + str(model_name)[0:4] + '_oversampled-ACTUAL_' + str(datetime.datetime.today().date()) + '.pickle.dat', 'wb'))
                # np.savetxt(config['PATH']['model_outpath'] + config['NAME']['product'] + model_name + '_oversampled-ACTUAL_COLUMNS.txt', np.array(X_train.columns), fmt='%s', delimiter=',')
                X_train.to_csv(config['PATH']['model_outpath'] + config['NAME']['product'] + str(model_name)[0:4] + '_oversampled-ACTUAL_X_train_' + str(datetime.datetime.today().date()) + '.csv')

    except UnboundLocalError:
        raise("You probably didn't specify RAW_ON or OVERSAMPLED_ON in the config file.")

    return

if __name__ == '__main__':
    p = Process(target=main)
    p.start()
    p.join()
    print('Script took:', time.time() - start_time, 'seconds')
