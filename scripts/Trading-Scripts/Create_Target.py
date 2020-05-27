import pandas as pd
import numpy as np
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import configparser
import warnings
import feather
import time
from multiprocessing import Pool, Process
import ast
import dask.dataframe as dd
import gc
import re

import sys

start_time = time.time()


config = configparser.ConfigParser()
config.read('/home/melgazar9/Trading/TD/Scripts/Trading-Scripts/Multi-Product/scripts/CL/CL_Create_Target.ini')

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

threshold = float(config['PARAMS']['threshold'])

Actual_Move = config['PARAMS']['Actual_Move']
Actual_HighMove = config['PARAMS']['Actual_HighMove']
Actual_LowMove = config['PARAMS']['Actual_LowMove']



if config['PATH']['df_path'].endswith('.feather'):
    df = pd.read_feather(config['PATH']['df_path'])
elif config['PATH']['df_path'].endswith('.feather'):
    df = dd.read_parquet(config['PATH']['df_path'], low_memory=False).compute()
elif config['PATH']['df_path'].endswith('.csv'):
    df = pd.read_csv(config['PATH']['df_path'], low_memory=False)


df.set_index('Datetime', inplace=True)
df.sort_index(inplace=True)

resample_period = list(re.findall('\d+', Actual_Move))[0] + 'min'
df_tmp = df.resample(resample_period)
tmp_actualMove = df_tmp[Actual_Move.replace('Actual', 'Prev')].shift(-1)
tmp_actualMove.name = Actual_Move
tmp_actualHighMove = df_tmp[Actual_HighMove.replace('Actual', 'Prev')].shift(-1)
tmp_actualHighMove.name = Actual_HighMove
tmp_actualLowMove = df_tmp[Actual_LowMove.replace('Actual', 'Prev')].shift(-1)
tmp_actualLowMove.name = Actual_LowMove
tmp = pd.merge_asof(tmp_actualMove, tmp_actualHighMove, left_index = True, right_index = True)
tmp = pd.merge_asof(tmp, tmp_actualLowMove, left_index = True, right_index = True)
df = pd.merge_asof(df, tmp, left_index = True, right_index = True)
del df_tmp, resample_period, tmp_actualMove, tmp_actualHighMove, tmp_actualLowMove, tmp

print(df.head())
print(df.shape)

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

        # strong buy
        self.df.loc[(self.df[Actual_Move] >= self.strong_buy) & (self.df[Actual_Move] <= self.threshold) & (self.df[Actual_LowMove] > (-1)*self.stop), 'Target_Actual'] = 4

        # medium buy
        self.df.loc[(self.df[Actual_Move] >= self.med_buy) & (self.df[Actual_Move] <= self.strong_buy) & (self.df[Actual_LowMove] > (-1)*self.stop) & (self.df['Target_Actual'] != 4), 'Target_Actual'] = 3

        # medium sell
        self.df.loc[(self.df[Actual_Move] <= (-1) * self.med_sell) & (self.df[Actual_Move] >= (-1) * self.strong_sell) & (self.df[Actual_LowMove] < self.stop) & (self.df['Target_Actual'] != 4) & (self.df['Target_Actual'] != 3), 'Target_Actual'] = 1

        # strong sell
        self.df.loc[(self.df[Actual_Move] <= (-1) * self.strong_sell) & (self.df[Actual_Move] >= (-1) * self.threshold) & (self.df[Actual_LowMove] < self.stop) & (self.df['Target_Actual'] != 4) & (self.df['Target_Actual'] != 3) & (self.df['Target_Actual'] != 1), 'Target_Actual'] = 0

        self.df.loc[(self.df['Target_Actual'] != 0) & (self.df['Target_Actual'] != 1) & (self.df['Target_Actual'] != 3) & (self.df['Target_Actual'] != 4), 'Target_Actual'] = 2

#         return pd.DataFrame(lst, index=self.df.index).rename(columns={0:'Target_Actual'})
#         return pd.DataFrame(lst, index=self.df[[Actual_Move]].dropna().index).rename(columns={0:'Target_Actual'})
        return df


    def calc_target_HL(self):

        # stop means how much heat I am willing to take per trade
        # i.e. if the move went up in my favor $50 but I took $1000 worth of heat that isn't good
        # hm stands for high move, lm stands for low move

#             if np.isnan(self.df[Actual_LowMove]) or np.isnan(self.df[Actual_HighMove])

        # if ActualHM >= buy signal AND ActualLM doesn't go below stop
        # Strong Buy
        self.df.loc[(self.df[Actual_HighMove] >= self.strong_buy) & (self.df[Actual_LowMove] >= (-1)*self.stop), 'Target_HL'] = 4

        # Strong Sell
        self.df.loc[(self.df[Actual_LowMove] <= (-1)*self.strong_sell) & (self.df[Actual_HighMove] <= self.stop) & (self.df['Target_HL'] != 4), 'Target_HL'] = 0

        # Medium Buy
        self.df.loc[(self.df[Actual_HighMove] >= self.med_buy) & (self.df[Actual_LowMove] >= (-1)*self.stop) & (self.df['Target_HL'] != 4) & (self.df['Target_HL'] != 0), 'Target_HL'] = 3

        # Medium Sell
        self.df.loc[(self.df[Actual_LowMove] <= (-1)*self.med_sell) & (self.df[Actual_HighMove] <= self.stop) & (self.df['Target_HL'] != 4) & (self.df['Target_HL'] != 0) & (self.df['Target_HL'] != 3), 'Target_HL'] = 1

        self.df.loc[(self.df['Target_HL'] != 0) & (self.df['Target_HL'] != 1) & (self.df['Target_HL'] != 3) & (self.df['Target_HL'] != 4), 'Target_HL'] = 2
#         return pd.DataFrame(lst, index=self.df.resample('60min').first().index).rename(columns={0:'Target_HL'})
#         return pd.DataFrame(lst, index=self.df[[Actual_Move]].dropna().index).rename(columns={0:'Target_HL'})
        return df

if config['PARAMS']['create_target_Actual_ON'] == 'TRUE':

    df_target_actual = CalcTarget(df, strong_buy=strong_buy_actual, med_buy=med_buy_actual, no_trade=no_trade_actual,
                                med_sell=med_sell_actual, strong_sell=strong_sell_actual, threshold=threshold,
                                stop=stop_actual).calc_target_actual()
    for i in range(int(config['PARAMS']['min_target_lookback']), int(config['PARAMS']['max_target_lookback']), int(config['PARAMS']['target_lookback_increment'])):
        df_target_actual['PrevTarget_ActMove' + str(i)] = df_target_actual['Target_Actual'].shift(i)

    df = df_target_actual.fillna(2).astype('int')

    print(df['Target_Actual'].value_counts())

if config['PARAMS']['create_target_HL_ON'] == 'TRUE':

    df_target_HL = CalcTarget(df, strong_buy=strong_buy_HL, med_buy=med_buy_HL, no_trade=no_trade_HL,
                            med_sell=med_sell_HL, strong_sell=strong_sell_HL, threshold=threshold,
                            stop=stop_HL).calc_target_HL()


    print(df_target_HL['Target_HL'].value_counts())

    for i in range(int(config['PARAMS']['min_target_lookback']), int(config['PARAMS']['max_target_lookback']), int(config['PARAMS']['target_lookback_increment'])):
        df_target_HL['PrevTarget_HL' + str(i)] = df_target_HL['Target_HL'].shift(i)

    df = df_target_HL.fillna(2).astype('int')

    print(df['Target_HL'].value_counts())
