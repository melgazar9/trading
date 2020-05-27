import os
import time
start_time = time.time()
import json
import pandas as pd
import dask.dataframe as dd
from dask import delayed
import numpy as np
import gc
import datetime
from pandas.io.json import json_normalize
import itertools
import pickle
import warnings
import subprocess
import sys
from sklearn.preprocessing import label_binarize
from sklearn.metrics import *
from itertools import cycle
from collections import Counter
import re
from multiprocessing import Pool, Process
import matplotlib.pyplot as plt
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
# import datefinder

import configparser

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings('ignore')


###############################
#            CONFIG
###############################

config = configparser.ConfigParser()
config_path = '/home/melgazar9/Trading/TD/Live-Trading/CL/scripts/CL_10min_TD_TRADITIONAL.ini'
config.read(config_path)

print('******USING MODEL', config['MODEL']['model_name'] + '******')

multiplier = float(config['PARAMS']['multiplier'])
threshold = float(config['PARAMS']['threshold'])

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

strong_cap_HL = float(config['PARAMS']['strong_cap_HL'])
med_cap_HL = float(config['PARAMS']['med_cap_HL'])
strong_cap_actual = float(config['PARAMS']['strong_cap_actual'])
med_cap_actual = float(config['PARAMS']['med_cap_actual'])



min_prob0 = float(config['PARAMS']['min_prob0'])
min_prob1 = float(config['PARAMS']['min_prob1'])
min_prob3 = float(config['PARAMS']['min_prob3'])
min_prob4 = float(config['PARAMS']['min_prob4'])

if config['MODEL']['xgbc_on'] == 'TRUE':
    xgbc_on = True
else:
    xgbc_on = False

if config['MODEL']['gbmc_on'] == 'TRUE':
    gbmc_on = True
else:
    gbmc_on = False

if config['MODEL']['lgbmc_on'] == 'TRUE':
    lgbmc_on = True
else:
    lgbmc_on = False


if xgbc_on == True:
    xgbc_cols = dd.read_csv(config['MODEL']['xgbc_df']).columns.drop('Datetime')
if gbmc_on == True:
    gbmc_cols = dd.read_csv(config['MODEL']['gbmc_df']).columns.drop('Datetime')
if lgbmc_on == True:
    lgbmc_cols = dd.read_csv(config['MODEL']['lgbmc_df']).columns.drop('Datetime')


min_lookback = int(config['PARAMS']['min_lookback'])
max_lookback = int(config['PARAMS']['max_lookback']) + 1
lookback_increment = int(config['PARAMS']['lookback_increment'])


Actual_Move = config['PARAMS']['Actual_Move_Name']
Actual_HighMove = config['PARAMS']['Actual_HighMove_Name']
Actual_LowMove = config['PARAMS']['Actual_LowMove_Name']


###############################################################################################
# DEFINE A FUNCTION THAT ONLY READS IN THE DATA IF WE ARE LOOKING AT THE CORRECT TIME STAMP
###############################################################################################

def verify_valid_timestamp():
    return

######################################
#       READ THE CORRECT FILE        #
######################################

filename = config['PATH']['filename']
historical_filename = config['PATH']['dropped_connection_filename']


#################################################################
#           LOAD CLASSIFIERS, BINARIZERS, AND SCALERS           #
#################################################################

smote_xgbc_HL = pickle.load(open(config['MODEL']['xgbc_HL'], 'rb'))
# smote_gbmc_HL = pickle.load(open(config['MODEL']['gbmc_HL'], 'rb'))
smote_lgbmc_HL = pickle.load(open(config['MODEL']['lgbmc_HL'], 'rb'))
smote_gbmc_HL = pickle.load(open(config['MODEL']['gbmc_HL'], 'rb'))
######################################
#           READ JSON DATA           #
######################################

def read_data(filepath):
    # read the entire file into a python array
    with open(filepath, 'rb') as f:
        data = f.readlines()

    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)

    data_list = []
    try:
        #data = [json.loads(x) for x in data]
        for x in data:
            data_list.append(json.loads(x))
    except ValueError:
        print('THERE IS A NAN VALUE')

    return data_list



######################################
#        VERIFY DATA IS CORRECT      #
######################################

def verify_data_integrity(raw_data):
    json_response = raw_data[0] # raw_data = read_data(filename)
    for k1 in json_response['response']:
        if (k1['command'] == 'LOGIN') and ((list(k1['content'].keys()) == ['msg', 'code']) or list(k1['content'].keys()) == ['code', 'msg']) and (k1['service'] == 'ADMIN'):
            print(True)
        else:
            print('REASON: RESPONSE ******Disconnecting from exchange****** REASON: RESPONSE')
            sys.exit()


    json_heartbeat = raw_data[1]
    for k2 in json_heartbeat['notify']:
        for k2a in k2.keys():
            if k2a == 'heartbeat':
                print(True)
            else:
                print('REASON: NOTIFY ******Disconnecting from exchange****** REASON: NOTIFY')
                sys.exit()

    json_response2 = raw_data[2]

    for k3 in json_response2['response']:
        if (k3['service'] == 'CHART_FUTURES') and (k3['command'] == 'SUBS') and ((list(k3['content'].keys()) == ['msg', 'code']) or (list(k3['content'].keys()) == ['code', 'msg'])):
            print(True)
        else:
            print('REASON: RESPONSE2 ******Disconnecting from exchange****** REASON: RESPONSE2')
            sys.exit()

    json_data = raw_data[3]
    for k4 in json_data['data']:
        if (k4['command'] == 'SUBS') and (k4['service'] == 'CHART_FUTURES'):
            print(True)
        else:
            print('REASON: DATA ISSUE ******Disconnecting from exchange****** REASON: DATA ISSUE')
            sys.exit()

    return

def read_historical_data(historical_filepath):
    # read the entire file into a python array
    with open(historical_filepath, 'rb') as f:
        data = f.readlines()

    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)

    data_list = []
    try:
        #data = [json.loads(x) for x in data]
        for x in data:
            data_list.append(json.loads(x))
    except ValueError:
        print('THERE IS A NAN VALUE')

    return data_list

######################################
#        VERIFY DATA IS CORRECT      #
######################################

def verify_historical_data(historical_data):
    json_response = historical_data[0]
    for k1 in json_response['response']:
        if (k1['command'] == 'LOGIN') and ((list(k1['content'].keys()) == ['msg', 'code']) or list(k1['content'].keys()) == ['code', 'msg']) and (k1['service'] == 'ADMIN'):
            print(True)
        else:
            print('REASON: RESPONSE ******Disconnecting from exchange****** REASON: RESPONSE')
            sys.exit()


    json_heartbeat = historical_data[1]
    for k2 in json_heartbeat['snapshot']:
        for k2a in k2.keys():
            if k2a == 'content':
                print(True)
            elif k2a == 'timestamp':
                print(True)
            elif k2a == 'command':
                print(True)
            elif k2a == 'service':
                print(True)
            else:
                print('REASON: NOTIFY ******Disconnecting from exchange****** REASON: NOTIFY')
                sys.exit()

    return


def get_historical_df(raw_data):

    cols = {'Key':'/CL', '0':'Datetime','1':'1minOpen','2':'1minHigh','3':'1minLow','4':'1minClose','5':'1minVolume'}

    data = pd.io.json.json_normalize(raw_data)['snapshot'].dropna()
    for lst1 in data:
        for lst2 in lst1:
            for lst3 in lst2['content']:
                df = pd.DataFrame(lst3['3'])


    df = df.rename(columns=cols)
    df['1minRange'] = df['1minHigh'] - df['1minLow']
    df['1minMove'] = df['1minClose'] - df['1minOpen']
    df['1minLowMove'] = df['1minLow'] - df['1minOpen']
    df['1minHighMove'] = df['1minHigh'] - df['1minOpen']
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index, unit='ms')
    df = df.sort_index()

    return df

######################################
#            GET CURR DF             #
######################################

def get_curr_df(raw_data):

    cols = {'1':'Datetime', '2':'1minOpen', '3':'1minHigh', '4':'1minLow', '5':'1minClose', '6':'1minVolume', 'key':'Symbol'}

    data_lst = []

    for lst1 in raw_data:
        if list(lst1.keys()) == ['data']:
            for lst2 in lst1['data']:
                for lst3 in lst2['content']:
                    data_lst.append(lst3)

    df = pd.DataFrame.from_dict(data_lst).rename(columns=cols).drop(['seq'], axis=1)
    df['1minRange'] = df['1minHigh'] - df['1minLow']
    df['1minMove'] = df['1minClose'] - df['1minOpen']
    df['1minLowMove'] = df['1minLow'] - df['1minOpen']
    df['1minHighMove'] = df['1minHigh'] - df['1minOpen']
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index, unit='ms')
    df = df.sort_index()

    return df


#####################################################
#          RESAMPLE DATA BEFORE SHIFTING            #
#####################################################

def get_df_init(df_1min, timeframe):

    df_1min = df_1min.drop('Symbol', axis=1)
    df_ohlc = df_1min.resample(timeframe).ohlc()
    df_volume = df_1min['1minVolume'].resample(timeframe).sum()

    df_resampled = pd.DataFrame()
    df_resampled[timeframe + 'Open'] = df_ohlc['1minOpen']['open']
    df_resampled[timeframe + 'High'] = df_ohlc['1minHigh']['high']
    df_resampled[timeframe + 'Low'] = df_ohlc['1minLow']['low']
    df_resampled[timeframe + 'Close'] = df_ohlc['1minClose']['close']
    df_resampled[timeframe + 'Move'] = df_ohlc['1minClose']['close'] - df_ohlc['1minOpen']['open']
    df_resampled[timeframe + 'Range'] = df_ohlc['1minHigh']['high'] - df_ohlc['1minLow']['low']
    df_resampled[timeframe + 'HighMove'] = df_ohlc['1minHigh']['high'] - df_ohlc['1minOpen']['open']
    df_resampled[timeframe + 'LowMove'] = df_ohlc['1minLow']['low'] - df_ohlc['1minOpen']['open']
    df_resampled[timeframe + 'Volume'] = df_volume

    return df_resampled


def calc_target(df, strong_buy, med_buy, no_trade, med_sell, strong_sell, threshold):

    lst = []

    for move in df[Actual_Move]:

        # strong buy
        if move >= strong_buy and move <= threshold:
            lst.append(4)

        # medium buy
        elif move >= med_buy and move <= strong_buy:
            lst.append(3)

        # no trade
        elif move <= med_buy and move >= med_buy * (-1):
            lst.append(2)

        # medium sell
        elif move <= med_sell * (-1) and move >= strong_sell * (-1):
            lst.append(1)

        elif move <= strong_sell * (-1) and move >= threshold * (-1):
            lst.append(0)

        else:
            lst.append('Error')

    return pd.DataFrame(lst, index=df.index).rename(columns={0:'Target'})






#################################################################################

###########################################
#            CREATE FEATURES
###########################################

#################################################################################


def get_rolling_features(df, col, window, min_periods):

    df[col + 'Rolling' + str('Sum').strip('()') + '_Window' + str(window)] = df[col].rolling(window=window, min_periods=min_periods).sum()
    df[col + 'Rolling' + str('Mean').strip('()') + '_Window' + str(window)] = df[col].rolling(window=window, min_periods=min_periods).mean()
    df[col + 'Rolling' + str('Std').strip('()') + '_Window' + str(window)] = df[col].rolling(window=window, min_periods=min_periods).std()
    df[col + 'Rolling' + str('Max').strip('()') + '_Window' + str(window)] = df[col].rolling(window=window, min_periods=min_periods).max()
    df[col + 'Rolling' + str('Min').strip('()') + '_Window' + str(window)] = df[col].rolling(window=window, min_periods=min_periods).min()

    return df



def macd(df, features, nslow, nfast):

    for feature in features:
        df[feature+'MACD_'+str(nslow)+'-'+str(nfast)] = df[feature].ewm(span=nslow, adjust=True).mean() - df[feature].ewm(span=nfast, adjust=True).mean() # 26 -12 period
        df[feature+'9dMA_'+str(nslow)+'-'+str(nfast)] = df[feature+'MACD_'+str(nslow)+'-'+str(nfast)].rolling(window=9).mean()
    return df


def RSI(series, period):

    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) # first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) # first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = u.ewm(com=period-1, adjust=False).mean() / d.ewm(com=period-1, adjust=False).mean()
    rsi = pd.DataFrame(100 - 100 / (1 + rs)).rename(columns={series.name : str(series.name).strip('Close')+'RSI'})

    #df = pd.merge_asof(df, rsi, )
    return rsi


def assign_rsi_cols(df, close_cols, rsi_cols):

    i=0
    while i < len(close_cols):
        try:
            df = pd.merge_asof(df, RSI(df[close_cols[i]], 14), left_index=True, right_index=True)
            i+=1
        except:
            df[rsi_cols[i]] = 50
            i+=1
    return df



def ATR(df, feature):
    df[feature[0:-5]+'ATR'] = df[feature].ewm(span=10).mean()
    df[feature[0:-5]+'ATR'] = (df[feature[0:-5]+'ATR'].shift(1)*13 + df[feature]) /  14

    return df



def Bollinger_Bands(df, feature, window_size, num_of_std):

    rolling_mean = df[feature].rolling(window=window_size).mean()
    rolling_std  = df[feature].rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)

    upper_band = pd.DataFrame(upper_band, index=upper_band.index).rename(columns={feature : feature+'UpperBB'})
    lower_band = pd.DataFrame(lower_band, index=lower_band.index).rename(columns={feature : feature+'LowerBB'})


    temp = pd.merge_asof(df, upper_band, left_index=True, right_index=True)
    temp = pd.merge_asof(temp, lower_band, left_index=True, right_index=True)

#     temp =  Bollinger_Bands(df, 'Prev10minClose', 20, 2)
    price_upperBB_diff = df[feature] - temp[feature+'UpperBB']
    price_lowerBB_diff = df[feature] - temp[feature+'LowerBB']
    temp[feature.strip('Close')+'Price-UpperBB_diff'] = price_upperBB_diff
    temp[feature.strip('Close')+'Price-LowerBB_diff'] = price_lowerBB_diff

    temp[feature.strip('Close')+'UpperBB_Change'] = temp[feature+'UpperBB'].diff()#.rename(columns={feature+'UpperBB':feature+'UpperBB_Change'})
    temp[feature.strip('Close')+'LowerBB_Change'] = temp[feature+'LowerBB'].diff()


    temp.drop([feature+'UpperBB'], axis=1, inplace=True)
    temp.drop([feature+'LowerBB'], axis=1, inplace=True)

    return temp


def PPSR(df, high, low, close):

    PP_10min = pd.Series((df['Prev10minHigh'] + df['Prev10minLow'] + df['Prev10minClose']) / 3)
    S1_10min = pd.Series(2 * PP_10min - df['Prev10minHigh'])
    R1_10min = pd.Series(2 * PP_10min - df['Prev10minLow'])
    R2_10min = pd.Series(PP_10min + df['Prev10minHigh'] - df['Prev10minLow'])
    S2_10min = pd.Series(PP_10min - df['Prev10minHigh'] + df['Prev10minLow'])
    R3_10min = pd.Series(df['Prev10minHigh'] + 2 * (PP_10min - df['Prev10minLow']))
    S3_10min = pd.Series(df['Prev10minLow'] - 2 * (df['Prev10minHigh'] - PP_10min))
    psr_10min = {'PP': PP_10min, 'S1': S1_10min, 'R1': R1_10min, 'S2': S2_10min, 'R2': R2_10min, 'S3': S3_10min, 'R3': R3_10min}
    PSR_10min = pd.DataFrame(psr_10min).rename(columns={'PP':'Prev10minPP',
                                                        'S1':'Prev10minS1',
                                                        'R1':'Prev10minR1',
                                                        'S2':'Prev10minS2',
                                                        'R2':'Prev10minR2',
                                                        'S3':'Prev10minS3',
                                                        'R3':'Prev10minR3'})
    if high == 'Prev10minHigh':
        return PSR_10min


    PP = pd.Series((df[high] + df[low] + df[close]) / 3)
    S1 = pd.Series(2 * PP - df[high])
    R1 = pd.Series(2 * PP - df[low])
    R2 = pd.Series(PP + df[high] - df[low])
    S2 = pd.Series(PP - df[high] + df[low])
    R3 = pd.Series(df[high] + 2 * (PP - df[low]))
    S3 = pd.Series(df[low] - 2 * (df[high] - PP))
    psr = {'PP': PP, 'S1': S1, 'R1': R1, 'S2': S2, 'R2': R2, 'S3': S3, 'R3': R3}
    PSR = pd.DataFrame(psr).rename(columns={'PP':low.strip('Low')+'PP',
                                            'S1':low.strip('Low')+'S1',
                                            'R1':low.strip('Low')+'R1',
                                            'S2':low.strip('Low')+'S2',
                                            'R2':low.strip('Low')+'R2',
                                            'S3':low.strip('Low')+'S3',
                                            'R3':low.strip('Low')+'R3'})




    temp = pd.merge_asof(PSR_10min, PSR, left_index=True, right_index=True)

    temp[low.strip('Low')+'PP_Change'] = temp[low.strip('Low')+'PP'] - temp['Prev10minPP']
    temp[low.strip('Low')+'S1_Change'] = temp[low.strip('Low')+'S1'] - temp['Prev10minS1']
    temp[low.strip('Low')+'R1_Change'] = temp[low.strip('Low')+'R1'] - temp['Prev10minR1']
    temp[low.strip('Low')+'S2_Change'] = temp[low.strip('Low')+'S2'] - temp['Prev10minS2']
    temp[low.strip('Low')+'R2_Change'] = temp[low.strip('Low')+'R2'] - temp['Prev10minR2']
    temp[low.strip('Low')+'S3_Change'] = temp[low.strip('Low')+'S3'] - temp['Prev10minS3']
    temp[low.strip('Low')+'R3_Change'] = temp[low.strip('Low')+'R3'] - temp['Prev10minR3']

    temp = temp[[i for i in temp.columns if i.endswith('Change')]]

    return temp






#################################################################################

#########################################################
#            DATA CLEANING / DATA PREP
#########################################################

#################################################################################



def calc_pnl_with_stop_with_cap(model, df, X, y, stop, strong_cap, med_cap, predictions, model_name):

    pred = pd.DataFrame(model.predict(X), index=X.index).rename(columns={0:predictions})
    results = pd.concat([pd.DataFrame(y), pred], axis=1)
    results = pd.concat([results, pd.DataFrame(df[Actual_Move], index=X.index)], axis=1)
    results = pd.concat([results, pd.DataFrame(df['Prev10minLowMove'].shift(-1), index=X.index).rename(columns={'Prev10minLowMove':Actual_LowMove})], axis=1)
    results = pd.concat([results, pd.DataFrame(df['Prev10minHighMove'].shift(-1), index=X.index).rename(columns={'Prev10minHighMove':Actual_HighMove})], axis=1)


    # Calculate P/L and concat it as an additional column
    lst=[]

    i=0
    while i < len(results):

        # strong buy -> 2 contracts traded
        if results[predictions][i] == 4:
            if results[Actual_LowMove][i] > (-1)*stop: # not stopped out
                if results[Actual_HighMove][i] >= strong_cap:
                    lst.append(1*strong_cap)
                    i+=1
                else:
                    lst.append(1*results[Actual_Move][i])
                    i+=1
            elif results[Actual_LowMove][i] <= (-1)*stop: # stopped out
                lst.append(-1*stop) # -.02 for assuming a trade out -> i.e. selling at the bid horribly (worst case testing)
                i+=1
            else:
                print('Error1')
                lst.append(np.nan)
                i+=1

        # medium buy
        elif results[predictions][i] == 3:
            if results[Actual_LowMove][i] > (-1)*stop:
                if results[Actual_HighMove][i] >= med_cap:
                    lst.append(med_cap)
                    i+=1
                else:
                    lst.append(results[Actual_Move][i])
                    i+=1
            elif results[Actual_LowMove][i] <= (-1)*stop:
                lst.append((-1)*stop)
                i+=1
            else:
                print('Error2')
                lst.append(np.nan)
                i+=1

        # no trade
        elif results[predictions][i] == 2:
            lst.append(0)
            i+=1

        # medium sell
        elif results[predictions][i] == 1:
            if results[Actual_HighMove][i] < stop:
                if results[Actual_LowMove][i] <= (-1)*med_cap:
                    lst.append(med_cap)
                    i+=1
                else:
                    lst.append((-1) * results[Actual_Move][i])
                    i+=1
            elif results[Actual_HighMove][i] >= stop:
                lst.append((-1)*stop)
                i+=1
            else:
                print('Error3')
                lst.append(np.nan)
                i+=1

        # strong sell
        elif results[predictions][i] == 0:
            if results[Actual_HighMove][i] < stop:
                if results[Actual_LowMove][i] <= (-1)*strong_cap:
                    lst.append(1*strong_cap)
                    i+=1
                else:
                    lst.append((-1) * results[Actual_Move][i])
                    i+=1
            elif results[Actual_HighMove][i] >= stop:
                lst.append((-1)*stop)
                i+=1
            else:
                print('Error5')
                lst.append(np.nan)
                i+=1
        else:
            print('Error6')
            lst.append(np.nan)
            i+=1

    pnl = pd.DataFrame(lst, index=results.index).rename(columns={0:model_name + ' P/L'})
    results = pd.concat([results, pnl], axis=1)

    print('\n')
    print('Trade Pct Predicted:', float(len(results[results[predictions]!=2])) / len(results))
    print('Trade Pct Actual Target', float(len(results[results['Target']!=2])) / len(results))

    print(predictions + ' P/L BEFORE FEES:', float(np.sum(results[model_name + ' P/L'])*1000))
    print(predictions + ' P/L AFTER FEES:', float(np.sum(results[model_name + ' P/L'])*1000) - float((7)*len(results.loc[results[predictions]!=2]))) # subtract fees

    return results.sort_index()



def calc_profitability_traditional(model, df, X, y, predictions, model_name, NN=False, X_df=None):

    if NN == False:
        if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
            prediction_df = calc_pnl_with_stop_with_cap(model, df, X, y, stop_actual, strong_cap_actual, predictions, model_name)
        if config['PARAMS']['HL_ON'] == 'TRUE':
            prediction_df = calc_pnl_with_stop_with_cap(model, df, X, y, stop_HL, strong_cap_HL, predictions, model_name)
    if NN == True:
        pred = pd.DataFrame(np.argmax(model.predict(X), axis=1)).set_index(X_df.index).rename(columns={0:predictions})
        prediction_df = pd.concat([pd.DataFrame(y, index=pred.index), pred], axis=1).rename(columns={0:'Target'})
        prediction_df = pd.concat([prediction_df, pd.DataFrame(df[Actual_Move], index=pred.index)], axis=1)

    results = prediction_df[prediction_df[predictions] != 2]

    sell_lst = []
    buy_lst = []

    i=0
    while i < len(results):
        if results[predictions][i] >= 3 and results[Actual_Move][i] > 0:
            buy_lst.append(1)
            i+=1
        elif results[predictions][i] >= 3 and results[Actual_Move][i] <= 0:
            buy_lst.append(0)
            i+=1
        elif results[predictions][i] <= 1 and results[Actual_Move][i] < 0:
            sell_lst.append(1)
            i+=1
        elif results[predictions][i] <= 1 and results[Actual_Move][i] >= 0:
            sell_lst.append(0)
            i+=1
        else:
            sell_lst.append('Error')
            i+=1

    try:
        print('Buy Accuracy:', float(Counter(buy_lst)[1]) / len(buy_lst))
        print('Sell Accuracy:', float(Counter(sell_lst)[1]) / len(sell_lst))
    except ZeroDivisionError:
        print('Zero Division Error - could not divide by zero to calculate accuracy.')

    return results


def calc_results(df, X, y, model, model_name):

    results = calc_profitability_traditional(model, df, X, y, model_name + ' Predictions', model_name, NN=False, X_df=None)
    df[model_name + ' P/L'] = results[model_name + ' P/L']
    print(model_name + ' Sharpe:', np.sum(results[results[model_name + ' Predictions']!=2][model_name + ' P/L'] / (np.sqrt(len(results[results[model_name + ' Predictions'] !=2])*results[results[model_name + ' Predictions'] !=2][model_name + ' P/L'].std()))))

    return df




def get_last_row(df_1min, timeframe):

#     df_last_row = pd.DataFrame(index=[pd.to_datetime(df_5min.index[-1] + pd.Timedelta(minutes=10))])

    df_last_row = pd.DataFrame()

    curr_prev_symbol = df_1min['Symbol'][-1]

    curr_prev_open = df_1min['1minOpen'].resample(timeframe).first().iloc[-1]

    curr_prev_high = df_1min['1minHigh'].resample(timeframe).max().iloc[-1]

    curr_prev_low = df_1min['1minLow'].resample(timeframe).min().iloc[-1]

    curr_prev_close = df_1min['1minClose'].resample(timeframe).last().iloc[-1]

    curr_prev_volume = df_1min['1minVolume'].resample(timeframe).sum().iloc[-1]

    curr_prev_range = curr_prev_high - curr_prev_low

    df_last_row = pd.DataFrame(index=[pd.to_datetime(df_1min.index[-1] + pd.Timedelta(minutes=10))])

    df_last_row['1minOpen'] = curr_prev_open
    df_last_row['1minHigh'] = curr_prev_high
    df_last_row['1minLow'] = curr_prev_low
    df_last_row['1minClose'] = curr_prev_close
    df_last_row['1minVolume'] = curr_prev_volume
    df_last_row['1minRange'] = curr_prev_range
    df_last_row['1minSymbol'] = curr_prev_symbol

    df_last_row = get_df_init(df_1min, timeframe)

    return df_last_row



# MODEL EVALUATION
class CalcTarget():

    def __init__(self, df, strong_buy, med_buy, no_trade, med_sell, strong_sell, threshold, stop):
        self.df = df
        self.strong_buy = strong_buy
        self.med_buy = med_buy
        self.no_trade = no_trade
        self.med_sell = med_sell
        self.strong_sell = strong_sell
        self.threshold = threshold
        self.stop = stop

    def calc_target_actual(self):
        super().__init__()

        self.df[Actual_Move] = self.df['Prev10minMove'].shift(-1)

        lst = []

        for move in self.df[Actual_Move]:

            # strong buy
            if move >= self.strong_buy and move <= self.threshold:
                lst.append(4)

            # medium buy
            elif move >= self.med_buy and move <= self.strong_buy:
                lst.append(3)

            # no trade
            elif move <= self.med_buy and move >= self.med_buy * (-1):
                lst.append(2)

            # medium sell
            elif move <= self.med_sell * (-1) and move >= self.strong_sell * (-1):
                lst.append(1)

            elif move <= self.strong_sell * (-1) and move >= self.threshold * (-1):
                lst.append(0)

            else:
                lst.append('Error')

        return pd.DataFrame(lst, index=self.df.index).rename(columns={0:'Target_Actual'})

    def calc_target_HL(self):

        # stop means how much heat I am willing to take per trade
        # i.e. if the move went up in my favor $50 but I took $1000 worth of heat that isn't good
        # hm stands for high move, lm stands for low move

        lst = []

        i = 0
        while i < len(self.df):
            # if ActualHM >= buy signal AND ActualLM doesn't go below stop
            if self.df[Actual_HighMove][i] >= self.strong_buy and self.df[Actual_LowMove][i] >= (-1)*self.stop:
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
        print(lst, len(lst))
        return pd.DataFrame(lst, index=self.df.index).rename(columns={0:'Target_HL'})


class CalcResults():

    def __init__(self, model, df, X, y, predictions, stop, strong_cap, med_cap, multiplier, need_cont_vars, HL, NN=False):

        self.model = model
        self.df = df
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
        print('Previous Probabilities:', probs[-2, :])
        print('Current Probabilities:', probs[-1, :])

        lst=[]

        for p in probs:

            if p[2] == max(p): # if no trade is max probaility
                lst.append(2)

            elif (p[0] >= min_prob0) and (p[0] >= max(p)): # if strong sell is max probability and >= min_prob_threshold
                if p[1] == sorted(p)[-2]: # if med_sell is the second highest probability
                    lst.append(0)
                # elif (p[0] - sorted(p)[-2]) >= float(config['PARAMS']['min_prob_diff']): # strong sell - second highest probability which isn't med_sell (difference)
                #     lst.append(2)
                else:
                    lst.append(2)

            elif (p[1] >= min_prob1) and (p[1] >= max(p)): # if med sell is max probability and >= min_prob_threshold
                if p[0] == sorted(p)[-2]:
                    lst.append(1)
                # elif (p[1] - sorted(p)[-2] >= float(config['PARAMS']['min_prob_diff'])):
                #     lst.append(2)
                else:
                    lst.append(2)

            elif (p[3] >= min_prob3) and (p[3] >= max(p)): # if med buy is max probability and >= min_prob_threshold
                if p[4] == sorted(p)[-2]:
                    lst.append(3)
                # elif (p[3] - sorted(p)[-2] >= float(config['PARAMS']['min_prob_diff'])):
                #     lst.append(2)
                else:
                    lst.append(2)

            elif (p[4] >= min_prob4) and (p[4] >= max(p)): # if strong buy is max probability and >= min_prob_threshold
                if (p[3] == sorted(p)[-2]): # med_buy is second highest probability
                    lst.append(4)
                # elif (p[4] - sorted(p)[-2] >= float(config['PARAMS']['min_prob_diff'])):
                #     lst.append(2)
                else:
                    lst.append(2)

            else:
                lst.append(2)

        return lst


    def initialize_df(self):

        super().__init__()

        if self.need_cont_vars == True:
            self.X = self.X.astype('float32')

        if config['PARAMS']['PredProb_ON'] == 'FALSE':
            pred = pd.DataFrame(self.model.predict(self.X), index=self.X.index).rename(columns={0:self.predictions})
        elif config['PARAMS']['PredProb_ON'] == 'TRUE':
            pred = pd.DataFrame(self.calc_predictions(), index=self.X.index).rename(columns={0:self.predictions})


        if self.HL == False:
            results = pd.concat([pd.DataFrame(self.y, index=self.X.index), pred], axis=1).rename(columns={0:'Target_Actual'})
        elif self.HL == True:
            results = pd.concat([pd.DataFrame(self.y, index=self.X.index), pred], axis=1).rename(columns={0:'Target_HL'})

        results = pd.concat([results, pd.DataFrame(self.df[Actual_HighMove], index=pred.index),
                             pd.DataFrame(self.df[Actual_LowMove], index=pred.index),
                             pd.DataFrame(self.df[Actual_Move], index=pred.index)], axis=1)

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
                if results[Actual_LowMove][i] > (-1)*self.stop:
                    if results[Actual_HighMove][i] >= self.med_cap:
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
                    if results[Actual_LowMove][i] <= (-1)*self.med_cap:
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
                    if results[Actual_LowMove][i] <= (-1)*self.strong_cap:
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
        print(self.predictions + ' P/L AFTER FEES Actual:', float(np.sum(results[self.predictions.split(' ')[0] + ' P/L Actual'])*self.multiplier) - float((7)*len(results.loc[results[self.predictions]!=2]))) # subtract fees

        pnl_4 = results[results['Actual Predictions']==4]
        pnl_3 = results[results['Actual Predictions']==3]
        pnl_1 = results[results['Actual Predictions']==1]
        pnl_0 = results[results['Actual Predictions']==0]

        print('Class 4:', pnl_4['Actual P/L'].sum()*self.multiplier)
        print('Class 3:', pnl_3['Actual P/L'].sum()*self.multiplier)
        print('Class 1:', pnl_1['Actual P/L'].sum()*self.multiplier)
        print('Class 0:', pnl_0['Actual P/L'].sum()*self.multiplier)

        return results


    def calc_pnl_traditional_2to1_actual(self):

        super().__init__()
        results = self.initialize_df()

        print('Actual Move Results')
        # Concat results into dataframe with columns: Target, Predictions, Actual Move

        pred = pd.DataFrame(model.predict(X), index=X.index).rename(columns={0:self.predictions})
        results = pd.concat([pd.DataFrame(y, index=X.index), pred], axis=1).rename(columns={0:'Target_Actual'})
        results = pd.concat([results, pd.DataFrame(self.df[Actual_Move], index=X.index)], axis=1)

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
        print(self.predictions + ' P/L HL AFTER FEES:', float(np.sum(results[self.predictions.split(' ')[0] + ' P/L HL'])*self.multiplier) - float((7)*len(results.loc[results[self.predictions]!=2]))) # subtract fees

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
                    if results[Actual_HighMove][i] >= self.strong_cap:
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
                    if results[Actual_HighMove][i] >= self.med_cap:
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
                    if results[Actual_LowMove][i] <= (-1)*self.med_cap:
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
                    if results[Actual_LowMove][i] <= (-1)*self.strong_cap:
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

        pnl = pd.DataFrame(lst, index=results.index).rename(columns={0:'HL P/L'})
#         results = pd.concat([results, pnl, pd.DataFrame(lst2, index=results.index)], axis=1)
        results = pd.concat([results, pnl], axis=1)
        results.sort_index(inplace=True)

        print('Trade Pct Predicted HL:', float(len(results[results[self.predictions]!=2])) / len(results))
        print('Trade Pct Actual Target HL', float(len(results[results['Target_HL']!=2])) / len(results))
        print(self.predictions + ' P/L HL BEFORE FEES:', np.sum(results['HL P/L'])*self.multiplier)
        print(self.predictions + ' P/L HL AFTER FEES:', np.sum(results['HL P/L'])*self.multiplier - (7)*len(results.loc[results[self.predictions]!=2])) # subtract fees


        pnl_4 = results[results['HL Predictions']==4]
        pnl_3 = results[results['HL Predictions']==3]
        pnl_1 = results[results['HL Predictions']==1]
        pnl_0 = results[results['HL Predictions']==0]

        print('Class 4:', pnl_4['HL P/L'].sum()*self.multiplier)
        print('Class 3:', pnl_3['HL P/L'].sum()*self.multiplier)
        print('Class 1:', pnl_1['HL P/L'].sum()*self.multiplier)
        print('Class 0:', pnl_0['HL P/L'].sum()*self.multiplier)


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

            plt.figure(figsize=(26,8))
            plt.plot(results.index, results[self.predictions.split(' ')[0] + ' P/L Actual'].cumsum())


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

            if config['PARAMS']['plot_cumsum'] == 'TRUE':
                plt.figure(figsize=(26,8))
                plt.plot(results.index, results['HL P/L'].cumsum())
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


        return prediction_df

    def calc_sharpe(self, results):

        if config['PARAMS']['HL_ON'] == 'TRUE':
            sharpe_HL = np.sum(results[results['HL Predictions']!=2]['HL P/L'] / (np.sqrt(len(results[results['HL Predictions'] !=2])*results[results['HL Predictions'] !=2]['HL P/L'].std())))
            print('HL Sharpe:', sharpe_HL)
            print('\n')
        if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
            sharpe_actual = np.sum(results[results['Actual Predictions']!=2]['Actual P/L'] / (np.sqrt(len(results[results['Actual Predictions'] !=2])*results[results['Actual Predictions'] !=2]['Actual P/L'].std())))
            print('Actual Sharpe:', sharpe_actual)
            print('\n')

        return

    def calc_results(self):

        super().__init__()

        self.initialize_df()
        results = self.calc_profitability()
        self.calc_sharpe(results)

        return results





############################################################################
#                                 MAIN
############################################################################


# MAIN
def main():

    # Raw Data
    raw_data = read_data(filename)
    verify_data_integrity(raw_data)

    # Get 1min df as pandas object
    df_1min = get_curr_df(raw_data)
    symbol = df_1min['Symbol']
    print(df_1min.get_dtype_counts())

    # Concat df_1min historical_data since internet connection was dropped
    if config['PARAMS']['dropped_connection'] == 'TRUE':
        historical_data = read_historical_data(historical_filename)
        verify_historical_data(historical_data)
        historical_df = get_historical_df(historical_data)[config['PARAMS']['dropped_connection_start_date']:config['PARAMS']['dropped_connection_end_date']]
        df_1min = pd.concat([df_1min, historical_df], axis=0).drop_duplicates().sort_index()
        df_1min['Symbol'].bfill(inplace=True)
        df_1min.drop('Symbol', axis=1, inplace=True)
        df_1min['Symbol'] = '/CL'
        # print(df_1min.get_dtype_counts())

    last_row_1min = pd.DataFrame(get_last_row(df_1min, '1min').iloc[-1]).T
    last_row_1min.index = pd.DatetimeIndex([df_1min.index[-1] + pd.Timedelta(minutes=10)])
    df_1min = pd.concat([df_1min, last_row_1min], axis=0)





    dfs = {f'{i}min' : get_df_init(df_1min, f'{i}min') for i in range(min_lookback, max_lookback, lookback_increment)}

    for i in range(min_lookback, max_lookback, lookback_increment):
        dfs[f'{i}min'].index = pd.Series(dfs[f'{i}min'].index).shift(-1)
        dfs[f'{i}min'] = dfs[f'{i}min'].loc[dfs[f'{i}min'].index.to_series().dropna()]
        dfs[f'{i}min'].sort_index(inplace=True)


    df = pd.merge_asof(dfs['5min'], dfs['10min'], left_index=True, right_index=True)
    for i in range(min_lookback + 10, max_lookback, lookback_increment):
        df = pd.merge_asof(df, dfs[f'{i}min'], left_index=True, right_index=True)

    df = df.add_prefix('Prev')







    # df_5min = get_df_init(df_1min, '5min')
    # df_10min = get_df_init(df_1min, '10min')
    # df_15min = get_df_init(df_1min, '15min')
    # df_30min = get_df_init(df_1min, '30min')
    # df_1h = get_df_init(df_1min, '1h')
    # df_2h = get_df_init(df_1min, '2h')
    # df_4h = get_df_init(df_1min, '4h')
    # df_8h = get_df_init(df_1min, '8h')
    # df_12h = get_df_init(df_1min, '12h')
    # df_1D = get_df_init(df_1min, '1D')
    # df_1W = get_df_init(df_1min, '1W')
    # # df_1M = get_df_init(df_1min, '1M')
    #
    #
    # df_5min.index = pd.Series(df_5min.index).shift(-1)
    # df_10min.index = pd.Series(df_10min.index).shift(-1)
    # df_15min.index = pd.Series(df_15min.index).shift(-1)
    # df_30min.index = pd.Series(df_30min.index).shift(-1)
    # df_1h.index = pd.Series(df_1h.index).shift(-1)
    # df_2h.index = pd.Series(df_2h.index).shift(-1)
    # df_4h.index = pd.Series(df_4h.index).shift(-1)
    # df_8h.index = pd.Series(df_8h.index).shift(-1)
    # df_12h.index = pd.Series(df_12h.index).shift(-1)
    # df_1D.index = pd.Series(df_1D.index).shift(-1)
    # df_1W.index = pd.Series(df_1W.index).shift(-1)
    # # df_1M.index = pd.Series(df_1M.index).shift(-1)
    #
    # df_5min = df_5min.loc[df_5min.index.to_series().dropna()] # drop nan datetimes
    # df_10min = df_10min.loc[df_10min.index.to_series().dropna()]
    # df_15min = df_15min.loc[df_15min.index.to_series().dropna()]
    # df_30min = df_30min.loc[df_30min.index.to_series().dropna()]
    # df_1h = df_1h.loc[df_1h.index.to_series().dropna()]
    # df_2h = df_2h.loc[df_2h.index.to_series().dropna()]
    # df_4h = df_4h.loc[df_4h.index.to_series().dropna()]
    # df_8h = df_8h.loc[df_8h.index.to_series().dropna()]
    # df_12h = df_12h.loc[df_12h.index.to_series().dropna()]
    # df_1D = df_1D.loc[df_1D.index.to_series().dropna()]
    # df_1W = df_1W.loc[df_1W.index.to_series().dropna()]
    # # df_1M = df_1M.loc[df_1M.index.to_series().dropna()]
    #
    #
    # df_5min.sort_index(inplace=True)
    # df_10min.sort_index(inplace=True)
    # df_15min.sort_index(inplace=True)
    # df_30min.sort_index(inplace=True)
    # df_1h.sort_index(inplace=True)
    # df_2h.sort_index(inplace=True)
    # df_4h.sort_index(inplace=True)
    # df_8h.sort_index(inplace=True)
    # df_12h.sort_index(inplace=True)
    # df_1D.sort_index(inplace=True)
    # df_1W.sort_index(inplace=True)
    # # df_1M.sort_index(inplace=True)
    #
    # df = pd.merge_asof(df_5min, df_10min, left_index=True, right_index=True)
    # df = pd.merge_asof(df, df_15min, left_index=True, right_index=True)
    # df = pd.merge_asof(df, df_30min, left_index=True, right_index=True)
    # df = pd.merge_asof(df, df_1h, left_index=True, right_index=True)
    # df = pd.merge_asof(df, df_2h, left_index=True, right_index=True)
    # df = pd.merge_asof(df, df_4h, left_index=True, right_index=True)
    # df = pd.merge_asof(df, df_8h, left_index=True, right_index=True)
    # df = pd.merge_asof(df, df_12h, left_index=True, right_index=True)
    # df = pd.merge_asof(df, df_1D, left_index=True, right_index=True)
    # df = pd.merge_asof(df, df_1W, left_index=True, right_index=True)
    # # df = pd.merge_asof(df, df_1M, left_index=True, right_index=True)
    #
    # df = df.add_prefix('Prev')
    #


    df = df[[i for i in df.columns if not i.startswith('Prev5min')]]
    df = df.resample('10min').first()
    df.dropna(inplace=True)

    cal = calendar()
    dr = pd.date_range(start=df.index[0], end=df.index[-1])
    holidays = cal.holidays(start=dr.min(), end=dr.max())
    df['IsHoliday'] = df.index.isin(holidays)

    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Week'] = df.index.week
    df['Day'] = df.index.day
    df['DayofWeek'] = df.index.dayofweek
    df['DayofYear'] = df.index.dayofyear
    df['IsMonthStart'] = df.index.is_month_start
    df['IsMonthEnd'] = df.index.is_month_end
    df['IsQuarterStart'] = df.index.is_quarter_start
    df['IsQuarterEnd'] = df.index.is_quarter_end
    df['IsYearStart'] = df.index.is_year_start
    df['IsYearEnd'] = df.index.is_year_end
    df['Hour'] = df.index.hour
    df['Quarter'] = df.index.quarter


    for col in df[[i for i in df.columns if i.endswith('Move') or i.endswith('Volume')]].columns:
        df = get_rolling_features(df, col, 4, 1)

    df = macd(df, [i for i in df.columns if i.endswith('Close')], 12, 26)

    for col in [i for i in df.columns if i.endswith('Close')]:
        df = pd.merge_asof(df, RSI(df[col], 14), left_index=True, right_index=True)


    # df = pd.merge_asof(df, RSI(df['Prev10minClose'], 14), left_index=True, right_index=True)
    # df = pd.merge_asof(df, RSI(df['Prev15minClose'], 14), left_index=True, right_index=True)
    # df = pd.merge_asof(df, RSI(df['Prev30minClose'], 14), left_index=True, right_index=True)
    # df = pd.merge_asof(df, RSI(df['Prev1hClose'], 14), left_index=True, right_index=True)
    # df = pd.merge_asof(df, RSI(df['Prev2hClose'], 14), left_index=True, right_index=True)
    # df = pd.merge_asof(df, RSI(df['Prev4hClose'], 14), left_index=True, right_index=True)
    # df = pd.merge_asof(df, RSI(df['Prev8hClose'], 14), left_index=True, right_index=True)
    # df = pd.merge_asof(df, RSI(df['Prev12hClose'], 14), left_index=True, right_index=True)
    # df = pd.merge_asof(df, RSI(df['Prev1DClose'], 14), left_index=True, right_index=True)
    # df = pd.merge_asof(df, RSI(df['Prev1WClose'], 14), left_index=True, right_index=True)
    # # df = pd.merge_asof(df, RSI(df['Prev1MClose'], 14), left_index=True, right_index=True)


    for i in [i for i in df.columns if i.endswith('Range')]:
        df = ATR(df, col)

    # df = ATR(df, 'Prev10minRange')
    # df = ATR(df, 'Prev15minRange')
    # df = ATR(df, 'Prev30minRange')
    # df = ATR(df, 'Prev1hRange')
    # df = ATR(df, 'Prev2hRange')
    # df = ATR(df, 'Prev4hRange')
    # df = ATR(df, 'Prev8hRange')
    # df = ATR(df, 'Prev12hRange')
    # df = ATR(df, 'Prev1DRange')
    # df = ATR(df, 'Prev1WRange')
    # df = ATR(df, 'Prev1MRange')

    for col in [i for i in df.columns if i.endswith('Close')]:
        df = Bollinger_Bands(df, col, 20, 2)

    # df = Bollinger_Bands(df, 'Prev10minClose', 20, 2)
    # df = Bollinger_Bands(df, 'Prev15minClose', 20, 2)
    # df = Bollinger_Bands(df, 'Prev30minClose', 20, 2)
    # df = Bollinger_Bands(df, 'Prev1hClose', 20, 2)
    # df = Bollinger_Bands(df, 'Prev2hClose', 20, 2)
    # df = Bollinger_Bands(df, 'Prev4hClose', 20, 2)
    # df = Bollinger_Bands(df, 'Prev8hClose', 20, 2)
    # df = Bollinger_Bands(df, 'Prev12hClose', 20, 2)
    # df = Bollinger_Bands(df, 'Prev1DClose', 20, 2)
    # df = Bollinger_Bands(df, 'Prev1WClose', 20, 2)
    # # df = Bollinger_Bands(df, 'Prev1MClose', 20, 2)




    ppsrs = {f'{i}min' : PPSR(df, 'Prev' + f'{i}min' + 'High', 'Prev' + f'{i}min' + 'Low', 'Prev' + f'{i}min' + 'Close') for i in range(min_lookback + 5, max_lookback, lookback_increment)}
    temp = pd.merge_asof(ppsrs['10min'], ppsrs['15min'], left_index=True, right_index=True)
    for i in range(min_lookback + 15, max_lookback, lookback_increment):
        temp = pd.merge_asof(temp, ppsrs[f'{i}min'], left_index=True, right_index=True)

    #
    # ppsr_10min = PPSR(df, 'Prev10minHigh', 'Prev10minLow', 'Prev10minClose')
    # ppsr_15min = PPSR(df, 'Prev15minHigh', 'Prev15minLow', 'Prev15minClose')
    # ppsr_30min = PPSR(df, 'Prev30minHigh', 'Prev30minLow', 'Prev30minClose')
    # ppsr_1h = PPSR(df, 'Prev1hHigh', 'Prev1hLow', 'Prev1hClose')
    # ppsr_2h = PPSR(df, 'Prev2hHigh', 'Prev2hLow', 'Prev2hClose')
    # ppsr_4h = PPSR(df, 'Prev4hHigh', 'Prev4hLow', 'Prev4hClose')
    # ppsr_8h = PPSR(df, 'Prev8hHigh', 'Prev8hLow', 'Prev8hClose')
    # ppsr_12h = PPSR(df, 'Prev12hHigh', 'Prev12hLow', 'Prev12hClose')
    # ppsr_1D = PPSR(df, 'Prev1DHigh', 'Prev1DLow', 'Prev1DClose')
    # ppsr_1W = PPSR(df, 'Prev1WHigh', 'Prev1WLow', 'Prev1WClose')
    # # ppsr_1M = PPSR(df, 'Prev1MHigh', 'Prev1MLow', 'Prev1MClose')
    #
    #
    # temp = pd.merge_asof(ppsr_10min, ppsr_15min, left_index=True, right_index=True)
    # temp = pd.merge_asof(temp, ppsr_30min, left_index=True, right_index=True)
    # temp = pd.merge_asof(temp, ppsr_1h, left_index=True, right_index=True)
    # temp = pd.merge_asof(temp, ppsr_2h, left_index=True, right_index=True)
    # temp = pd.merge_asof(temp, ppsr_4h, left_index=True, right_index=True)
    # temp = pd.merge_asof(temp, ppsr_8h, left_index=True, right_index=True)
    # temp = pd.merge_asof(temp, ppsr_12h, left_index=True, right_index=True)
    # temp = pd.merge_asof(temp, ppsr_1D, left_index=True, right_index=True)
    # temp = pd.merge_asof(temp, ppsr_1W, left_index=True, right_index=True)
    # # temp = pd.merge_asof(temp, ppsr_1M, left_index=True, right_index=True)

    temp_10min_change = temp[[i for i in temp.columns if i.startswith('Prev10min')]].diff().rename(columns={'Prev10minPP': 'Prev10minPP_Change',
                                                                                                            'Prev10minS1': 'Prev10minS1_Change',
                                                                                                            'Prev10minR1': 'Prev10minR1_Change',
                                                                                                            'Prev10minS2': 'Prev10minS2_Change',
                                                                                                            'Prev10minR2': 'Prev10minR2_Change',
                                                                                                            'Prev10minS3': 'Prev10minS3_Change',
                                                                                                            'Prev10minR3': 'Prev10minR3_Change'})
    temp.drop(['Prev10minPP','Prev10minS1','Prev10minR1','Prev10minS2','Prev10minR2','Prev10minS3','Prev10minR3'], axis=1, inplace=True)
    df = pd.merge_asof(df, temp, left_index=True, right_index=True)
    df = pd.merge_asof(df, temp_10min_change, left_index=True, right_index=True)

    df.drop(['Prev10minPP_Change', 'Prev10minS1_Change','Prev10minR1_Change',
             'Prev10minS2_Change','Prev10minR2_Change', 'Prev10minS3_Change','Prev10minR3_Change'],
             axis=1, inplace=True)

    important_cols = [i for i in df.columns if not i.endswith('Open') and not i.endswith('High') and not
                      i.endswith('Low') and not i.endswith('Close')]


    df = df[important_cols]
    df.dropna(inplace=True)

    try:
        df.drop([Actual_Move + 'RollingSum_Window4',
                 Actual_Move + 'RollingMean_Window4',
                 Actual_Move + 'RollingStd_Window4',
                 Actual_Move + 'RollingMax_Window4',
                 Actual_Move + 'RollingMin_Window4'], axis=1, inplace=True)
    except:
        print('Cannot drop actual cols!')

    print('Cols that start with Actual:', df[[i for i in df.columns if i.startswith('Actual')]].columns) # must only have Actual_Move!



    #################################################################################

    ###########################################
    #            CREATE FEATURES
    ###########################################

    #################################################################################

    df[Actual_LowMove] = df['Prev' + Actual_LowMove.strip('Actual')].shift(-1)
    df[Actual_HighMove] = df['Prev' + Actual_HighMove.strip('Actual')].shift(-1)

    df.index = df.index.tz_localize('utc').tz_convert('US/Central')

    df.loc[df.between_time('06:00:00','15:00:00', include_start=False).index, 'Overnight_or_Intraday'] = 1
    df['Overnight_or_Intraday'].fillna(0, inplace=True)

    target_actual = CalcTarget(df, strong_buy=strong_buy_actual, med_buy=med_buy_actual, no_trade=no_trade_actual,
                                med_sell=med_sell_actual, strong_sell=strong_sell_actual, threshold=threshold,
                                stop=stop_actual).calc_target_actual()
    target_actual = target_actual.replace('Error', 2)

    target_HL = CalcTarget(df, strong_buy=strong_buy_HL, med_buy=med_buy_HL, no_trade=no_trade_HL, med_sell=med_sell_HL,
                           strong_sell=strong_sell_HL, threshold=threshold, stop=stop_HL).calc_target_HL()#['Target'].value_counts()
    print(target_actual['Target_Actual'].value_counts())
    print(target_HL['Target_HL'].value_counts())

    df['Target_Actual'] = target_actual
    df['Target_HL'] = target_HL




    cat_vars = ['Year', 'Month', 'Week', 'Day', 'DayofWeek', 'DayofYear', 'IsMonthEnd',
                'IsMonthStart', 'IsQuarterEnd', 'IsQuarterStart', 'IsYearEnd', 'IsHoliday',
                'IsYearStart', 'Overnight_or_Intraday', 'Hour', 'Quarter']

    cat_vars = cat_vars + [i for i in df.columns if i.endswith('Binned') or i.endswith('Opinion')]

    cont_vars = [i for i in df.columns if not i in cat_vars]
    # cat_vars = [c for c in x.columns for i in cat_vars if c.startswith(i) or c.endswith(i)]
    # print(len(df.columns) == len(cat_vars)+len(cont_vars)) # must be True!


    df = pd.get_dummies(df, columns=[i for i in cat_vars], drop_first=True)
    df.rename(columns={'Overnight_or_Intraday_1.0':'Overnight_or_Intraday'}, inplace=True)
    cat_vars = [i for i in df.columns if not i in cont_vars]

    for col in cat_vars:
        df[col] = df[col].astype('category').cat.as_ordered()

    for col in cont_vars:
        df[col] = df[col].astype('float32')

    # Month one-hot
    for m in range(1, 13):
        df.loc[df.index.month == m, 'Month_'+str(m)] = 1
        df.loc[df.index.month != m, 'Month_'+str(m)] = 0

    # Quarter one-hot
    for q in range(2, 5):
        df.loc[df.index.quarter == q, 'Quarter_'+str(q)] = 1
        df.loc[df.index.quarter != q, 'Quarter_'+str(q)] = 0

    # Hour one-hot
    for h in range(1, 25):
        df.loc[df.index.hour == h, 'Hour_'+str(h)] = 1
        df.loc[df.index.hour != h, 'Hour_'+str(h)] = 0

    # Day of year one-hot
    for d in range(1, 367): # in case of leap year
        df.loc[df.index.dayofyear == d, 'DayofYear_'+str(d)] = 1
        df.loc[df.index.dayofyear != d, 'DayofYear_'+str(d)] = 0

    # Day of Week one-hot
    for dw in range(1, 7):
        df.loc[df.index.dayofweek == dw, 'DayofWeek_'+str(dw)] = 1
        df.loc[df.index.dayofweek != dw, 'DayofWeek_'+str(dw)] = 0

    # Day of Month one-hot
    for dm in range(1, 32):
        df.loc[df.index.day == dm, 'Day_'+str(dm)] = 1
        df.loc[df.index.day != dm, 'Day_'+str(dm)] = 0

    # Week one-hot
    for w in range(1, 53):
        df.loc[df.index.week == w, 'Week_'+str(w)] = 1
        df.loc[df.index.week != w, 'Week_'+str(w)] = 0


    df.loc[df.index.is_month_start, 'IsMonthStart_True'] = 1
    df['IsMonthStart_True'].fillna(0, inplace=True)
    df['IsMonthStart_True'] = df['IsMonthStart_True'].astype('category').cat.as_ordered()

    df.loc[df.index.is_month_end, 'IsMonthEnd_True'] = 1
    df['IsMonthEnd_True'].fillna(0, inplace=True)
    df['IsMonthEnd_True'] = df['IsMonthEnd_True'].astype('category').cat.as_ordered()

    df.loc[df.index.is_quarter_start, 'IsQuarterStart_True'] = 1
    df['IsQuarterStart_True'].fillna(0, inplace=True)
    df['IsQuarterStart_True'] = df['IsQuarterStart_True'].astype('category').cat.as_ordered()

    df.loc[df.index.is_quarter_end, 'IsQuarterEnd_True'] = 1
    df['IsQuarterEnd_True'].fillna(0, inplace=True)
    df['IsQuarterEnd_True'] = df['IsQuarterEnd_True'].astype('category').cat.as_ordered()

    df.loc[df.index.isin(holidays), 'IsHoliday_True'] = 1
    df['IsHoliday_True'].fillna(0, inplace=True)
    df['IsHoliday_True'] = df['IsHoliday_True'].astype('category').cat.as_ordered()


    for col in df.select_dtypes('float64').columns:
        df[col] = df[col].astype('category').cat.as_ordered()



    if config['PARAMS']['scale'] == 'TRUE':

        # load scaler
        scaler = pickle.load(open(config['MODEL']['scaler']), 'wb')
        X_scaled = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)

        for col in cat_vars:
            X_scaled[col] = X_scaled[col].astype('category').cat.as_ordered()

        for col in cont_vars:
            X_scaled[col] = X_scaled[col].astype('float32')


    if config['PARAMS']['OneHot-Target_ON'] == 'TRUE':
        if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
            y_train_oh_actual = keras.utils.to_categorical(y_train_actual, num_classes=5)
            y_val_oh_actual = keras.utils.to_categorical(y_val_actual, num_classes=5)
            y_test_oh_actual = keras.utils.to_categorical(y_test_actual, num_classes=5)
        if config['PARAMS']['HL_ON'] == 'TRUE':
            y_train_oh_HL = keras.utils.to_categorical(y_train_HL, num_classes=5)
            y_val_oh_HL = keras.utils.to_categorical(y_val_HL, num_classes=5)
            y_test_oh_HL = keras.utils.to_categorical(y_test_HL, num_classes=5)


    if lgbmc_on == True:
        temp = df[lgbmc_cols].copy()
        temp['Target_HL'] = df['Target_HL']
        temp[Actual_HighMove] = df[Actual_HighMove]
        temp[Actual_LowMove] = df[Actual_LowMove]
        temp[Actual_Move] = df[Actual_Move]
        df = temp


    X = df.drop([i for i in df.columns if i.startswith('Target') or i.startswith('Actual')], axis=1)
    if config['PARAMS']['HL_ON'] == 'TRUE':
        y_HL = df['Target_HL']
    if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
        y_actual = df['Target_Actual']

    if config['PARAMS']['HL_ON'] == 'TRUE':
        if xgbc_on == True:
            print('SMOTE HL XGBC')
            hl_results_xgbc = CalcResults(smote_xgbc_HL, df, X.astype('float32'), y_HL, 'HL Predictions', stop_HL, strong_cap_HL, med_cap_HL, multiplier, need_cont_vars=False, HL=True).calc_results()
            print('Previous XGBC_HL Time:', hl_results_xgbc.index[-2])
            print('Previous XGBC_HL Prediction:', hl_results_xgbc['HL Predictions'][-2])
            print('Current XGBC_HL Time:', hl_results_xgbc.index[-1])
            print('Current XGBC_HL Prediction:', hl_results_xgbc['HL Predictions'][-1])

        if gbmc_on == True:
            print('SMOTE HL GBMC')
            hl_results_gbmc = CalcResults(smote_gbmc_HL, df, X, y_HL, 'HL Predictions', stop_HL, strong_cap_HL, med_cap_HL, multiplier, need_cont_vars=False, HL=True).calc_results()
            print('Previous GBMC_HL Time:', hl_results_gbmc.index[-2])
            print('Previous GBMC_HL Prediction:', hl_results_gbmc['HL Predictions'][-2])
            print('Current GBMC_HL Time:', hl_results_gbmc.index[-1])
            print('Current GBMC_HL Prediction:', hl_results_gbmc['HL Predictions'][-1])

        print('\n')

        if lgbmc_on == True:
            print('SMOTE HL LGBMC')
            hl_results_lgbmc = CalcResults(smote_lgbmc_HL, df, X, y_HL, 'HL Predictions', stop_HL, strong_cap_HL, med_cap_HL, multiplier, need_cont_vars=False, HL=True).calc_results()
            hl_results_lgbmc = hl_results_lgbmc.round(decimals=2)
            print('Previous LGBMC_HL Time:', hl_results_lgbmc.index[-2])
            print('Previous LGBMC_HL Prediction:', hl_results_lgbmc['HL Predictions'][-2])
            print('Current LGBMC_HL Time:', hl_results_lgbmc.index[-1])
            print('Current LGBMC_HL Prediction:', hl_results_lgbmc['HL Predictions'][-1])


    if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
        if xgbc_on == True:
            print('SMOTE ACTUAL XGBC')
            actual_results_xgbc = CalcResults(smote_xgbc_actual, df, X, y_Actual, 'Actual Predictions', stop_actual, strong_cap_actual, med_cap_actual, multiplier, need_cont_vars=False, HL=False).calc_results()

        if gbmc_on == True:
            print('SMOTE ACTUAL GBMC')
            actual_results_gbmc = CalcResults(smote_gbmc_actual, df, X, y_Actual, 'Actual Predictions', stop_actual, strong_cap_actual, med_cap_actual, multiplier, need_cont_vars=False, HL=False).calc_results()

        if lgbmc_on == True:
            print('SMOTE ACTUAL LGBMC')
            actual_results_lgbmc = CalcResults(smote_lgbmc_actual, df, X, y_Actual, 'Actual Predictions', stop_actual, strong_cap_actual, med_cap_actual, multiplier, need_cont_vars=False, HL=False).calc_results()

    # print(df, df.get_dtype_counts())
    print(X.get_dtype_counts())

    df['Target_Actual'] = target_actual
    df['Target_HL'] = target_HL

    if config['PARAMS']['ACTUAL_ON'] == 'FALSE':
        df['Actual Predictions'] = np.nan
        df['Actual P/L'] = np.nan
    if config['PARAMS']['HL_ON'] == 'FALSE':
        df['HL Predictions'] = np.nan
        df['HL P/L'] = np.nan

    if config['PARAMS']['ACTUAL_ON'] == 'TRUE':
        if xgbc_on == True:
            df['XGBC Actual Predictions'] = actual_results_xgbc['Actual Predictions']
            df['XGBC Actual P/L'] = actual_results_xgbc['Actual P/L']

        if gbmc_on == True:
            df['GBMC Actual Predictions'] = actual_results_gbmc['Actual Predictions']
            df['GBMC Actual P/L'] = actual_results_gbmc['Actual P/L']

        if lgbmc_on == True:
            df['LGBMC Actual Predictions'] = actual_results_lgbmc['Actual Predictions']
            df['LGBMC Actual P/L'] = actual_results_lgbmc['Actual P/L']

    if config['PARAMS']['HL_ON'] == 'TRUE':
        if xgbc_on == True:
            df['XGBC HL Predictions'] = hl_results_xgbc['HL Predictions']
            df['XGBC HL P/L'] = hl_results_xgbc['HL P/L']

        if gbmc_on == True:
            df['GBMC HL Predictions'] = hl_results_gbmc['HL Predictions']
            df['GBMC HL P/L'] = hl_results_gbmc['HL P/L']

        if lgbmc_on == True:
            df['LGBMC HL Predictions'] = hl_results_lgbmc['HL Predictions']
            df['LGBMC HL P/L'] = hl_results_lgbmc['HL P/L']

    if config['SAVE']['save_df'] == 'TRUE':

        if xgbc_on == True and lgbmc_on == True and gbmc_on == True:
            column_order = ['Target_Actual', 'Target_HL', 'LGBMC HL Predictions', 'LGBMC HL P/L',
                            'GBMC HL Predictions', 'GBMC HL P/L', 'XGBC HL Predictions', 'XGBC HL P/L',
                            Actual_LowMove, Actual_HighMove,Actual_Move] + list(df.columns)

        elif lgbmc_on == True and gbmc_on == False and xgbc_on == False:
            column_order = ['Target_Actual', 'Target_HL', 'LGBMC HL Predictions', 'LGBMC HL P/L',
                            Actual_LowMove, Actual_HighMove,Actual_Move] + list(df.columns)

        elif xgbc_on == True and lgbmc_on == False and gbmc_on == False:
            column_order = ['Target_Actual', 'Target_HL', 'XGBC HL Predictions', 'XGBC HL P/L',
                            Actual_LowMove, Actual_HighMove,Actual_Move] + list(df.columns)

        elif gbmc_on == True and xgbc_on == False and lgbmc_on == False:
            column_order = ['Target_Actual', 'Target_HL', 'GBMC HL Predictions', 'GBMC HL P/L',
                            Actual_LowMove, Actual_HighMove,Actual_Move] + list(df.columns)

        elif xgbc_on == True and lgbmc_on == True and gbmc_on == False:
            column_order = ['Target_Actual', 'Target_HL', 'LGBMC HL Predictions', 'LGBMC HL P/L', 'XGBC HL Predictions', 'XGBC HL P/L',
                            Actual_LowMove, Actual_HighMove,Actual_Move] + list(df.columns)

        elif xgbc_on == True and lgbmc_on == False and gbmc_on == True:
            column_order = ['Target_Actual', 'Target_HL', 'GBMC HL Predictions', 'GBMC HL P/L', 'XGBC HL Predictions', 'XGBC HL P/L',
                            Actual_LowMove, Actual_HighMove,Actual_Move] + list(df.columns)

        elif xgbc_on == False and lgbmc_on == True and gbmc_on == True:
            column_order = ['Target_Actual', 'Target_HL', 'LGBMC HL Predictions', 'LGBMC HL P/L',
                            'GBMC HL Predictions', 'GBMC HL P/L',
                            Actual_LowMove, Actual_HighMove,Actual_Move] + list(df.columns)



        df = df[column_order]

        df.to_csv(config['SAVE']['save_path'] + 'CL_10min_FULL_results_' + str(datetime.datetime.today().date()) + '.csv')

    return



if __name__ == '__main__':

    main()
    print(time.time() - start_time)
    # p = Process(target=main)
    # p.start()
    # p.join()
    # print('Script took:', time.time() - start_time, 'seconds')

    # with Pool() as p:
    #     p.starmap(main, [(min_lookback, max_lookback, lookback_increment)])
    #     end_time = time.time()
    #     print(end_time - start_time)
    #     print('Script took:', end_time - start_time)
