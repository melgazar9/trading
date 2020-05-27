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

start_time = time.time()


config = configparser.ConfigParser()
config.read('/home/melgazar9/Trading/TD/Scripts/Trading-Scripts/CL/scripts/CL_1min_CreateFeatures.ini')

resampled_bars = config['PARAMS']['resampled_bars']

min_lookback = int(config['PARAMS']['min_lookback'])
max_lookback = int(config['PARAMS']['max_lookback']) + 1
lookback_increment = int(config['PARAMS']['lookback_increment'])


def get_df_init(df, timeframe):

    df_ohlc = df.resample(timeframe).ohlc()
    df_volume = df['1minVolume'].resample(timeframe).sum()

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
    gc.collect()

    return df_resampled


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


def move_iar(df, feature):

    lst=[]
    prev_move_iar = 0

    for move in df[feature]:
        if np.isnan(move):
            move_iar = 0
            lst.append(move_iar)
            prev_move_iar = move_iar
        else:
            if move == 0:
                move_iar = prev_move_iar
                lst.append(move_iar)
                prev_move_iar = move_iar
            elif (move >= 0 and prev_move_iar >= 0) or (move <= 0 and prev_move_iar <= 0):
                move_iar = move + prev_move_iar
                lst.append(move_iar)
                prev_move_iar = move_iar
            elif (move < 0 and prev_move_iar >= 0) or (move > 0 and prev_move_iar <= 0):
                move_iar = move
                lst.append(move_iar)
                prev_move_iar = move_iar

    return pd.DataFrame(lst, index=df.index, columns=[feature]).rename(columns={feature : feature + 'IAR'})

def calc_daily_vwap(df, price_groupby_freq, volume_groupby_freq):

    s = df['Prev1minClose'] * df['Prev1minVolume']
    s = s.groupby(pd.Grouper(freq=price_groupby_freq)).cumsum()
    volume_groupby_freq = df.groupby(pd.Grouper(freq=volume_groupby_freq))['Prev1minVolume'].cumsum()

    return s / volume_groupby_freq

def pos_neg_move(df, feature, min_move_up):
    return df[feature] > min_move_up

# Define opinionated trades

def opinion_trade_Prev5minMove(df):
    lst=[]
    for i in df['Prev5minMove']:
        if i >= float(config['PARAMS']['opinion_Prev5minMove']):
            lst.append(1)
        else:
            lst.append(0)
    return pd.DataFrame(lst, index=df.index).rename(columns={0: 'opinion_Prev5minMove'})

def opinion_trade_Prev5minHighMove(df):
    lst=[]
    for i in df['Prev5minHighMove']:
        if i >= float(config['PARAMS']['opinion_Prev5minHighMove']):
            lst.append(1)
        else:
            lst.append(0)
    return pd.DataFrame(lst, index=df.index).rename(columns={0: 'opinion_Prev5minHighMove'})

def opinion_trade_Prev5minLowMove(df):
    lst=[]
    for i in df['Prev5minLowMove']:
        if i <= (-1)*float(config['PARAMS']['opinion_Prev5minLowMove']):
            lst.append(1)
        else:
            lst.append(0)
    return pd.DataFrame(lst, index=df.index).rename(columns={0: 'opinion_Prev5minLowMove'})


def opinion_trade_Prev10minMove(df):
    lst=[]
    for i in df['Prev10minMove']:
        if i >= float(config['PARAMS']['opinion_Prev10minMove']):
            lst.append(1)
        else:
            lst.append(0)
    return pd.DataFrame(lst, index=df.index).rename(columns={0: 'opinion_Prev10minMove'})

def opinion_trade_Prev10minHighMove(df):
    lst=[]
    for i in df['Prev10minHighMove']:
        if i >= float(config['PARAMS']['opinion_Prev10minHighMove']):
            lst.append(1)
        else:
            lst.append(0)
    return pd.DataFrame(lst, index=df.index).rename(columns={0: 'opinion_Prev10minHighMove'})

def opinion_trade_Prev10minLowMove(df):
    lst=[]
    for i in df['Prev10minLowMove']:
        if i <= (-1)*float(config['PARAMS']['opinion_Prev10minLowMove']):
            lst.append(1)
        else:
            lst.append(0)
    return pd.DataFrame(lst, index=df.index).rename(columns={0: 'opinion_Prev10minLowMove'})

def opinion_trade_Prev15minMove(df):
    lst=[]
    for i in df['Prev15minMove']:
        if i >= float(config['PARAMS']['opinion_Prev15minMove']):
            lst.append(1)
        else:
            lst.append(0)
    return pd.DataFrame(lst, index=df.index).rename(columns={0: 'opinion_Prev15minMove'})

def opinion_trade_Prev15minHighMove(df):
    lst=[]
    for i in df['Prev15minHighMove']:
        if i >= float(config['PARAMS']['opinion_Prev15minHighMove']):
            lst.append(1)
        else:
            lst.append(0)
    return pd.DataFrame(lst, index=df.index).rename(columns={0: 'opinion_Prev15minHighMove'})

def opinion_trade_Prev15minLowMove(df):
    lst=[]
    for i in df['Prev15minLowMove']:
        if i <= (-1)*float(config['PARAMS']['opinion_Prev15minLowMove']):
            lst.append(1)
        else:
            lst.append(0)
    return pd.DataFrame(lst, index=df.index).rename(columns={0: 'opinion_Prev15minLowMove'})





def main():

    if config['PARAMS']['read_csv'] == 'TRUE':
        print('Reading CSV...')
        df_1min = pd.read_csv(config['PATH']['read_file'])
    elif config['PARAMS']['read_feather'] == 'TRUE':
        print('Reading Feather File...')
        df_1min = pd.read_feather(config['PATH']['read_file'])
    elif config['PARAMS']['read_csv'] == 'TRUE':
        print('Reading Parquet File...')
        df_1min = pd.read_parquet(config['PATH']['read_file'])

    df_1min.set_index('Datetime', inplace=True)
    df_1min.index = pd.to_datetime(df_1min.index)
    df_1min.sort_index(inplace=True)
    print(df_1min)
    gc.collect()

    ####################################################
    #                 Rewritten Code Here
    ####################################################
    print('Calculating Features...')
    if config['PARAMS']['lookback_increment_timesteps_ON'] == 'TRUE':

        dfs = {f'{i}min' : get_df_init(df_1min, f'{i}min') for i in range(min_lookback, max_lookback, lookback_increment)}
        for i in range(min_lookback, max_lookback, lookback_increment):
            dfs[f'{i}min'].index = pd.Series(dfs[f'{i}min'].index).shift(-1)
            dfs[f'{i}min'] = dfs[f'{i}min'].loc[dfs[f'{i}min'].index.to_series().dropna()]
            dfs[f'{i}min'].sort_index(inplace=True)


        df = pd.merge_asof(dfs['1min'], dfs['2min'], left_index=True, right_index=True)

        for i in range(min_lookback + 1, max_lookback, lookback_increment):
            df = pd.merge_asof(df, dfs[f'{i}min'], left_index=True, right_index=True)

    elif config['PARAMS']['lookback_increment_timesteps_ON'] == 'FALSE':

        dfs = {f'{i}min' : get_df_init(df_1min, f'{i}min') for i in ast.literal_eval(config['PARAMS']['lookback_timesteps_list'])}

        for i in ast.literal_eval(config['PARAMS']['lookback_timesteps_list']):
            print('Looking back:', str(i) + 'min')
            dfs[f'{i}min'].index = pd.Series(dfs[f'{i}min'].index).shift(-1)
            dfs[f'{i}min'] = dfs[f'{i}min'].loc[dfs[f'{i}min'].index.to_series().dropna()]
            dfs[f'{i}min'].sort_index(inplace=True)

            df = pd.merge_asof(dfs['1min'], dfs['2min'], left_index=True, right_index=True)
            for i in sorted(ast.literal_eval(config['PARAMS']['lookback_timesteps_list']))[2:]:
                df = pd.merge_asof(df, dfs[f'{i}min'], left_index=True, right_index=True)


    df = df.add_prefix('Prev')
    gc.collect()
    if config['PARAMS']['keep_1min_candlesticks'] == 'FALSE':
        df = df[[i for i in df.columns if not i.startswith('Prev5min')]]
        df = df.resample('1min').first()

    elif config['PARAMS']['keep_1min_candlesticks'] == 'TRUE':
        pass
        #print(df[[i for i in df.columns if i.startswith('Prev1min')]])
    df.dropna(inplace=True)
    # print(df[[i for i in df.columns if i.startswith('Prev15min')]])
    gc.collect()

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
    gc.collect()

    print('Calculating rolling features...')
    for col in df[[i for i in df.columns if i.endswith('Move') or i.endswith('Volume')]].columns:
        df = get_rolling_features(df, col, 4, 1)
        gc.collect()

    # for col in df[[i for i in df.columns if i.endswith('Close') or i.endswith('Volume')]].columns:
    #     df = get_rolling_features(df, col, 4, 1)

    print('Calculating MACD...')
    df = macd(df, [i for i in df.columns if i.endswith('Close')], 12, 26)

    print('Calculating RSI...')
    for col in [i for i in df.columns if i.endswith('Close')]:
        df = pd.merge_asof(df, RSI(df[col], 14), left_index=True, right_index=True)
        gc.collect()

    print('Calculating ATR...')
    for i in [i for i in df.columns if i.endswith('Range')]:
        df = ATR(df, col)
        gc.collect()

    print('Calculating Bollinger Bands...')
    for col in [i for i in df.columns if i.endswith('Close')]:
        df = Bollinger_Bands(df, col, 20, 2)
        gc.collect()

    print('Calculating moveIAR...')
    for col in [i for i in df.columns if i.endswith('minMove')]:
        df[col + 'IAR'] = move_iar(df.resample(re.findall(r'\d+', col)[0] + 'min').first(), col)
        gc.collect()

    # print('Calculating moveIAR change...')
    # for col in [i for i in df.columns if i.endswith('HighMove') or i.endswith('LowMove')]:
    #     df[col + 'IAR'] = move_iar(df, col).diff()
    #     gc.collect()

    print('Calculating pos_neg_move...')
    for col in [i for i in df.columns if i.endswith('Move') and not 'High' in i and not 'Low' in i]:
        df[col + '_PosNeg'] = pos_neg_move(df, col, int(config['PARAMS']['PosNegMove_MinMoveUp']))
        df[col + '_PosNegIAR'] = move_iar(np.sign(df[[col + 'IAR']].resample(re.findall(r'\d+', col)[0] + 'min').first()), col + 'IAR').rename(columns={col + 'IAR': col + '_PosNegIAR'})
        gc.collect()

    print('Calculating daily_vwap...')
    df['PrevDaily_VWAP_1minAgg'] = calc_daily_vwap(df, '1D', '1D')

    # for col in df[[i for i in df.columns if i.endswith('Close')]]:
    #     df[col + '_PctChange'] = df[col].pct_change()

    # df[df == np.inf] = 0
    # df[df == (-1)*np.inf] = 0

    # for col in [i for i in df.columns if i.endswith('Move') and not 'High' in i and not 'Low' in i]:
    #     df[col + '_PosNegIAR'] = move_iar(np.sign(df[[col]]), col + 'IAR').rename(columns = {col + 'IAR':col + '_PosNegIAR'})


    # df['Prev10minMoveIAR'] = move_iar(df, 'Prev10minMove')
    # df['Prev10minHighMoveIAR'] = move_iar(df, 'Prev10minHighMove')
    # df['Prev10minLowMoveIAR'] = move_iar(df, 'Prev10minLowMove')
    #
    # df['PosNeg_Prev10minMove'] = pos_neg_move(df, 'Prev10minMove', int(config['PARAMS']['PosNegMove_MinMoveUp']))
    #
    # df['Prev10minMove_PosNegIAR'] = move_iar(np.sign(df[['Prev10minMoveIAR']]), 'Prev10minMoveIAR').rename(columns={'Prev10minMoveIAR':'Prev10minMove_PosNegIAR'})

    if config['PARAMS']['PPSR_ON'] == 'TRUE' and config['PARAMS']['lookback_increment_timesteps_ON'] == 'TRUE':
        print('Calculating PPSR...')
        ppsrs = {f'{i}min' : PPSR(df, 'Prev' + f'{i}min' + 'High', 'Prev' + f'{i}min' + 'Low', 'Prev' + f'{i}min' + 'Close') for i in range(min_lookback + 1, max_lookback, lookback_increment)}
        temp = pd.merge_asof(ppsrs['10min'], ppsrs['15min'], left_index=True, right_index=True)
        for i in range(min_lookback + 3, max_lookback, lookback_increment):
            temp = pd.merge_asof(temp, ppsrs[f'{i}min'], left_index=True, right_index=True)



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

    print('Calculating Opinion Trades...')
    df['Prev5minMove_Opinion'] = opinion_trade_Prev5minMove(df)
    df['Prev5minHighMove_Opinion'] = opinion_trade_Prev5minHighMove(df)
    df['Prev5minLowMove_Opinion'] = opinion_trade_Prev5minLowMove(df)

    df['Prev10minMove_Opinion'] = opinion_trade_Prev10minMove(df)
    df['Prev10minHighMove_Opinion'] = opinion_trade_Prev10minHighMove(df)
    df['Prev10minLowMove_Opinion'] = opinion_trade_Prev10minLowMove(df)

    df['Prev15minMove_Opinion'] = opinion_trade_Prev15minMove(df)
    df['Prev15minHighMove_Opinion'] = opinion_trade_Prev15minHighMove(df)
    df['Prev15minLowMove_Opinion'] = opinion_trade_Prev15minLowMove(df)
    gc.collect()

    important_cols = [i for i in df.columns if not i.endswith('Open') and not i.endswith('High') and not
                      i.endswith('Low') and not i.endswith('Close')]


    df = df[important_cols]
    # df.dropna(inplace=True)
    df = df.ffill().dropna()

    try:
        df.drop(['Actual10minMoveRollingSum_Window4',
                 'Actual10minMoveRollingMean_Window4',
                 'Actual10minMoveRollingStd_Window4',
                 'Actual10minMoveRollingMax_Window4',
                 'Actual10minMoveRollingMin_Window4'], axis=1, inplace=True)

    except:
        pass

    print([i for i in df.columns if 'Actual' in i])
    print(df)
    print(df.isnull().sum()[df.isnull().sum()>0])
    # print(list(df.columns))
    if config['PARAMS']['save_df'] == 'TRUE':
        if config['PARAMS']['write_csv'] == 'TRUE':
            print('Writing CSV...')
            df.to_csv(config['PATH']['save_df_path'] + config['PARAMS']['product'] + '_1min_FULL_' + str(datetime.datetime.today().date()) + '.csv')
        if config['PARAMS']['write_feather'] == 'TRUE':
            print('Writing Feather File...')
            # feather.write_dataframe(df.reset_index(), config['PATH']['save_df_path'] + config['PARAMS']['product'] + '_10min_FULL_' + str(datetime.datetime.today().date()) + '.feather', nthreads=32)
            df.reset_index().to_feather(config['PATH']['save_df_path'] + config['PARAMS']['product'] + '_1min_FULL_' + str(datetime.datetime.today().date()) + '.feather')
        if config['PARAMS']['write_parquet'] == 'TRUE':
            print('Writing Parquet File...')
            ddf = dd.from_pandas(df, npartitions=int(config['PARAMS']['npartitions']))
            dd.to_parquet(ddf, config['PATH']['save_df_path'] + config['PARAMS']['product'] + '_1min_FULL_' + str(datetime.datetime.today().date()) + '.parquet')

    return

if __name__ == '__main__':
    main()
    print(time.time()-start_time)
    # p = Process(target=main)
    # p.start()
    # p.join()
    # print(time.time() - start_time)

    # with Pool(32) as p: # This doesn't speed up the script at all
    #     p.map(main, [1])
    #     # main()
    #     end_time = time.time()
    #     print(end_time - start_time)
    #     print('Script took:', end_time - start_time)
