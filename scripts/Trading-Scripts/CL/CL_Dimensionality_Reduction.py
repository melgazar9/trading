import time
import pandas as pd
import dask.dataframe as dd
import numpy as np
import feather
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from imblearn.over_sampling import SMOTE, SMOTENC, SVMSMOTE, ADASYN, BorderlineSMOTE, RandomOverSampler
from sklearn.metrics import *
from imblearn.metrics import classification_report_imbalanced
import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from collections import Counter
import configparser
import ast
import pickle
import sys
from multiprocessing import Pool, Process
import warnings
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegressionCV

warnings.filterwarnings('ignore')

config = configparser.ConfigParser()

config_path = config.read('/home/melgazar9/Trading/TD/Scripts/Trading-Scripts/CL/scripts/CL_Dimensionality_Reduction.ini')

print('**************** RUNNING', config['NAME']['product'], '****************')

train_start_date = config['PARAMS']['train_start_date']
train_end_date = config['PARAMS']['train_end_date']
test_start_date = config['PARAMS']['test_start_date']

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

Actual_Move = config['PARAMS']['Actual_Move_Name']
Actual_HighMove = config['PARAMS']['Actual_HighMove_Name']
Actual_LowMove = config['PARAMS']['Actual_LowMove_Name']

threshold = int(config['PARAMS']['threshold'])

if config['PARAMS']['read_csv'] == 'TRUE':
    print('Reading CSV...')
    ddf = dd.read_csv(config['PATH']['filename'])
elif config['PARAMS']['read_feather'] == 'TRUE':
    print('Reading Feather File...')
    # df = pd.read_feather(config['PATH']['filename'], use_threads=32)
    df = feather.read_dataframe(config['PATH']['filename'])
elif config['PARAMS']['read_parquet'] == 'TRUE':
    print('Reading Parquet File...')
    ddf = dd.read_parquet(config['PATH']['filename'])

if config['PARAMS']['read_feather'] != 'TRUE':
    df = ddf.compute()

try:
    df.set_index('Datetime', inplace=True)
except KeyError:
    print('Datetime index is already set!')
    pass


df.index = pd.to_datetime(df.index)
df.index = df.index.tz_localize('utc').tz_convert('US/Central')

df.dropna(axis=0, inplace=True)

df[Actual_Move] = df[['Prev' + Actual_Move.strip('Actual')]].resample(config['NAME']['product'][3:5] + 'min').first().rename(columns={'Prev' + Actual_Move.strip('Actual') : Actual_Move}).shift(-1)
df[Actual_HighMove] = df[['Prev' + Actual_HighMove.strip('Actual')]].resample(config['NAME']['product'][3:5] + 'min').first().rename(columns={'Prev' + Actual_HighMove.strip('Actual') : Actual_HighMove}).shift(-1)
df[Actual_LowMove] = df[['Prev' + Actual_LowMove.strip('Actual')]].resample(config['NAME']['product'][3:5] + 'min').first().rename(columns={'Prev' + Actual_LowMove.strip('Actual'): Actual_LowMove}).shift(-1)



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
df = pd.concat([df, target_actual, target_HL], axis=1)

X = df.drop([i for i in df.columns if i=='Target_HL' or 'Actual' in i], axis=1)
y = df[config['PARAMS']['Target_Variable']]

X_train = X[train_start_date:train_end_date]
X_test = X[test_start_date:]
y_train = y[train_start_date:train_end_date]
y_test = y[test_start_date:]


if config['PARAMS']['fillna_X'] == 'ffill':
    X.ffill(inplace=True)
if config['PARAMS']['fillna_y'] == 'ffill':
    y.ffill(inplace=True)
if config['PARAMS']['fillna_X'] == 'bfill':
    X.bfill(inplace=True)
if config['PARAMS']['fillna_y'] == 'bfill':
    y.bfill(inplace=True)
else:
    X.fillna(int(config['PARAMS']['fillna_values']))
    y.fillna(int(config['PARAMS']['fillna_values']))

if config['PARAMS']['scale_ON'] == 'TRUE':

    scaler = eval(config['PARAMS']['scaler']).fit(X_train)

    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)


cat_vars = ['Year', 'Month', 'Week', 'Day', 'DayofWeek', 'DayofYear', 'IsMonthEnd',
            'IsMonthStart', 'IsQuarterEnd', 'IsQuarterStart', 'IsYearEnd', 'IsHoliday',
            'IsYearStart', 'Overnight_or_Intraday', 'Hour', 'Quarter', 'PrevTarget']

cat_vars = [col for col in X.columns for cat in cat_vars if cat in col]
cat_vars = [i for i in X.columns if i.endswith('Binned') or i.endswith('Opinion') or i.startswith('PrevTarget')]
cont_vars = [i for i in X.columns if not i in cat_vars]

print(len(X.columns) == len(cat_vars)+len(cont_vars)) # must be True!

if config['PARAMS']['scale_ON'] == 'FALSE':
    print('Converting column dtypes...')
    for col in cat_vars:
        X[col] = X[col].astype('category').cat.as_ordered()
        gc.collect()
    for col in cont_vars:
        X[col] = X[col].astype('float32')
        gc.collect()
if config['PARAMS']['scale_ON'] == 'TRUE':
    print('Converting column dtypes...')
    for col in cat_vars:
        X_scaled[col] = X_scaled[col].astype('category').cat.as_ordered()
        gc.collect()
    for col in cont_vars:
        X_scaled[col] = X_scaled[col].astype('float32')
        gc.collect()



if config['PARAMS']['dim_reduction_algo'] == 'PCA':
    dim_reduction_algo = PCA(**ast.literal_eval(config['PARAMS']['dim_reduction_algo_params']))
elif config['PARAMS']['dim_reduction_algo'] == 'SVD':
    dim_reduction_algo = TruncatedSVD(**ast.literal_eval(config['PARAMS']['dim_reduction_algo_params']))
elif config['PARAMS']['dim_reduction_algo'] == 't-SNE':
    dim_reduction_algo = TSNE(**ast.literal_eval(config['PARAMS']['dim_reduction_algo_params']))


print('Fitting', config['PARAMS']['dim_reduction_algo'] + '...')
if config['PARAMS']['scale_ON'] == 'TRUE':
    dim_reduction_algo.fit(X_train_scaled)
elif config['PARAMS']['scale_ON'] == 'FALSE':
    dim_reduction_algo.fit(X_train)
print('Fitted!')

X_embedded = pd.DataFrame(dim_reduction_algo.transform(X_scaled), index=X.index)
X_embedded.columns = [config['PARAMS']['dim_reduction_algo'] + '_' + str(i) for i in range(len(X_embedded.columns))]
X_train_embedded = pd.DataFrame(dim_reduction_algo.transform(X_train_scaled), index=X_train.index)
X_train_embedded.columns = [config['PARAMS']['dim_reduction_algo'] + '_' + str(i) for i in range(len(X_embedded.columns))]
X_test_embedded = pd.DataFrame(dim_reduction_algo.transform(X_test_scaled), index=X_test.index)
X_test_embedded.columns = [config['PARAMS']['dim_reduction_algo'] + '_' + str(i) for i in range(len(X_embedded.columns))]

# X_embedded = pd.concat([X_train_embedded, X_test_embedded], axis=0)

if config['PARAMS']['save_df_as_csv'] == 'TRUE':
    print('Saving CSV...')
    X_embedded.to_csv(config['PATH']['save_df_path'] + 'X_' + config['PARAMS']['dim_reduction_algo'] + '_' + str(ast.literal_eval(config['PARAMS']['dim_reduction_algo_params'])['n_components']) + 'D_' + str(datetime.datetime.today().date()) + '.csv')
    # X_train_embedded.to_csv(config['PATH']['save_df_path'] + 'X_train_' + config['PARAMS']['dim_reduction_algo'] + '_' + str(datetime.datetime.today().date()) + '.csv')
    # X_test_embedded.to_csv(config['PATH']['save_df_path'] + 'X_test_' + config['PARAMS']['dim_reduction_algo'] + '_' + str(datetime.datetime.today().date()) + '.csv')
    print('Saved!')
if config['PARAMS']['save_df_as_feather'] == 'TRUE':
    print('Saving Feather File...')
    X_embedded.to_feather(config['PATH']['save_df_path'] + 'X_' + config['PARAMS']['dim_reduction_algo'] + '_' + str(ast.literal_eval(config['PARAMS']['dim_reduction_algo_params'])['n_components']) + 'D_' + str(datetime.datetime.today().date()) + '.feather')
    # X_train_embedded.to_feather(config['PATH']['save_df_path'] + 'X_train' + str(datetime.datetime.today().date()) + '.feather')
    # X_test_embedded.to_feather(config['PATH']['save_df_path'] + 'X_test_' + config['PARAMS']['dim_reduction_algo'] + '_' + str(datetime.datetime.today().date()) + '.csv')
    print('Saved!')
if config['PARAMS']['save_df_as_parquet'] == 'TRUE':
    print('Saving Parquet File...')
    X_embedded.to_parquet(config['PATH']['save_df_path'] + 'X_' + config['PARAMS']['dim_reduction_algo'] + '_' + str(ast.literal_eval(config['PARAMS']['dim_reduction_algo_params'])['n_components']) + 'D_' + str(datetime.datetime.today().date()) + '.parquet')
    # X_train_embedded.to_parquet(config['PATH']['save_df_path'] + 'X_train' + str(datetime.datetime.today().date()) + '.parquet')
    # X_test_embedded.to_parquet(config['PATH']['save_df_path'] + 'X_test' + str(datetime.datetime.today().date()) + '.parquet')
    print('Saved!')


if config['PARAMS']['save_model'] == 'TRUE':
    print('Saving', config['PARAMS']['dim_reduction_algo'] + '...')
    pickle.dump(dim_reduction_algo, open(config['PATH']['save_model_path'] + config['PARAMS']['dim_reduction_algo'] + '_' + str(ast.literal_eval(config['PARAMS']['dim_reduction_algo_params'])['n_components']) + 'D_' + str(datetime.datetime.today().date())  + '.pickle.dat', 'wb'))
    print('Saved!')

if config['PARAMS']['save_scaler'] == 'TRUE':
    print('Saving', config['PARAMS']['scaler'] + '...')
    pickle.dump(scaler, open(config['PATH']['save_scaler_path'] + config['PARAMS']['scaler'] + str(ast.literal_eval(config['PARAMS']['dim_reduction_algo_params'])['n_components']) + 'D_' + str(datetime.datetime.today().date())  + '.pickle.dat', 'wb'))
    print('Saved!')

if config['PARAMS']['plot_ON'] == 'TRUE':
    if ast.literal_eval(config['PARAMS']['dim_reduction_algo_params'])['n_components'] != 2:
        print('Cannot plot - More than 2 dimensions')
    else:
        print('Plotting...')
        plt.figure(figsize=(12,8))
        plt.scatter(X_train_embedded.iloc[:, 0], X_train_embedded.iloc[:, 1], alpha=0.2, c=y_train, cmap='viridis')
        plt.show()
