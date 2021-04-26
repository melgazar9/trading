# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 12:26:20 2020

@author: Matt
"""
import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import multiprocessing as mp
import dask
from dask import delayed
import os
import time
import alpaca_trade_api as alpaca_api
from functools import reduce
import sys
from twelvedata import TDClient
import yfinance
import simplejson
import requests

###### Use alpha_vantage to get technical indicators

from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.sectorperformance import SectorPerformances



class PullHistoricalData():


    """

    Parameters
    __________

    api_headers: dictionary of alpaca api headers - it should contain the api key
    symbols: a list of symbols to pull - currently the symbols are limited to stocks only
    copy: create a copy of the dataframe before running each function (default True - setting to False may overwrite df while running certain functions)

    """

    def __init__(self,
                 api_client,
                 symbols,
                 copy = True):

        self.api_client = api_client
        self.symbols = symbols
        self.copy = copy

    def twelvedata_pull_timeseries(self,
                                   interval,
                                   join_method='left',
                                   left_table='AAPL',
                                   outputsize=5000,
                                   timezone="America/Chicago",
                                   max_requests_per_min=8,
                                   max_requests_per_day=800,
                                   num_workers=-1):

        # Can only handle 8 requests per minute with the free version - so we cannot request more than 8 symbols at once with the api wrapper

        if join_method == 'left':
            assert left_table in self.symbols, 'left table not in self.symbols!'

            self.symbols = [left_table] + [i for i in self.symbols if not i in left_table]

        num_requests = 0

        if num_workers != 1:
            if num_workers <= 0:
                num_workers = mp.cpu_count()

            def pull_single_symbol(symbol,
                                   interval=interval,
                                   outputsize=outputsize,
                                   timezone=timezone):

                df_symbol = self.api_client.time_series(
                    symbol=symbol,
                    interval=interval,
                    outputsize=outputsize,
                    timezone=timezone,
                ).as_pandas()
                df_symbol = df_symbol.add_prefix(symbol + '_')
                return df_symbol

            # Run an initial pull so we don't have to wait an extra minute on the first iteration
            symbol_chunk = self.symbols[num_requests: num_requests + max_requests_per_min]
            delayed_list = [delayed(pull_single_symbol(s)) for s in symbol_chunk]

            tuple_of_dfs = dask.compute(*delayed_list, num_workers=num_workers)
            list_of_dfs = [tuple_of_dfs[i] for i in range(len(tuple_of_dfs))]
            df_symbols = reduce(lambda x, y: pd.merge(x, y, how=join_method, left_index=True, right_index=True),
                          list_of_dfs)

            # loop through chunks of symbols in the symbol list
            i = 0

            while i < len(self.symbols) - max_requests_per_min:

                num_requests += max_requests_per_min

                symbol_chunk = self.symbols[num_requests: num_requests + max_requests_per_min]

                if num_requests % max_requests_per_day == 0:
                    time.sleep(86000)

                if num_requests % max_requests_per_min == 0:
                    time.sleep(60)
                    if num_requests % 500 == 0:
                        print('\nsleeping' + str(max_requests_per_min) + ' seconds...\n')
                        print('\nnum_requests: ' + str(num_requests) + '\n')

                delayed_list = [delayed(pull_single_symbol(s)) for s in symbol_chunk]

                tuple_of_dfs = dask.compute(*delayed_list, num_workers=num_workers)
                list_of_dfs = [tuple_of_dfs[i] for i in range(len(tuple_of_dfs))]
                df_i = reduce(lambda x, y: pd.merge(x, y, how=join_method, left_index=True, right_index=True),
                              list_of_dfs)
                df_symbols = pd.merge(df_symbols, df_i, how=join_method, left_index=True, right_index=True)

                i += max_requests_per_min

        else:
            for i in range(len(self.symbols)):

                if num_requests % max_requests_per_min == 0:
                    time.sleep(60)
                    if num_requests % 500 == 0:
                        print('\nsleeping' + str(max_requests_per_min) + ' seconds...\n')
                        print('\nnum_requests: ' + str(num_requests) + '\n')


                df_symbols_i = self.api_client.time_series(
                                                        symbol=self.symbols[num_requests],
                                                        interval=interval,
                                                        outputsize=outputsize,
                                                        timezone=timezone,
                                                        ).as_pandas()
                df_symbols_i = df_symbols_i.add_prefix(self.symbols[num_requests] + '_')

                df_symbols = pd.merge(df_symbols, df_symbols_i, how=join_method, left_index=True, right_index=True)

                num_requests += 1


        return df_symbols

    def alpaca_get_barset_data(self,
                        time_interval = 'minute',
                        limit = None):

        assert str(type(self.api_client)) == "<class 'alpaca_trade_api.rest.REST'>", 'Wrong api_client submitted'

        df_symbols = self.api_client.get_barset(self.symbols, time_interval, limit = limit).df
        df_symbols.columns = ['_'.join(col).strip() for col in df_symbols.columns.values]

        return df_symbols

    def alpaca_get_historic_agg_v2_data(self,
                                 multiplier = 1,
                                 timespan = 'minute',
                                 _from = '1900-01-01',
                                 to = '2099-12-31',
                                 join_method = 'inner',
                                 left_table = 'AAPL',
                                 num_workers = mp.cpu_count()):

        assert str(type(self.api_client)) == "<class 'alpaca_trade_api.rest.REST'>", 'Wrong api_client submitted'

        if num_workers != 1:
            #self.api_client.polygon.historic_agg_v2(symbol = symbol, multiplier = multiplier, timespan = timespan, _from = _from, to = to).df
            def pull_single_agg(symbol):
                df_i = self.api_client.polygon.historic_agg_v2(symbol = symbol, multiplier = multiplier, timespan = timespan, _from = _from, to = to).df
                df_i = df_i.add_prefix(symbol + '_')
                return df_i

            delayed_list = [delayed(pull_single_agg(s)) for s in self.symbols]
            tuple_of_dfs = dask.compute(*delayed_list)
            list_of_dfs = [tuple_of_dfs[i] for i in range(len(tuple_of_dfs))]

            if join_method == 'left' or join_method == 'right':
                self.symbols = [left_table] + [i for i in self.symbols if not i == left_table]

            df_out = reduce(lambda x, y: pd.merge(x, y, how = join_method, left_index = True, right_index = True), list_of_dfs)

            # df_out = list_of_dfs[0]
            # for i in range(len(list_of_dfs) - 1):
            #     df_out = pd.merge(df_out, list_of_dfs[i+1], how = join_method, left_index = True, right_index = True)

        else:
            df_out = self.api_client.polygon.historic_agg_v2(symbol = self.symbols[0], multiplier = multiplier, timespan = timespan, _from = _from, to = to).df
            df_out = df_out.add_prefix(self.symbols[0] + '_')
            for symbol in self.symbols[1:]:
                tmp_df = self.api_client.polygon.historic_agg_v2(symbol = symbol, multiplier = multiplier, timespan = timespan, _from = _from, to = to).df
                tmp_df = tmp_df.add_prefix(symbol + '_')
                df_out = pd.merge(df_out, tmp_df, how = join_method, left_index = True, right_index = True)

        return df_out

    def alphavantage_get_technical_indicators(self,
                                 indicator_dict = {},# [i for i in dir(TechIndicators) if i.startswith('get')], # all available alpha_vantage technical indicators
                                 join_method = 'inner',
                                 num_cores = mp.cpu_count(),
                                 **kwargs):

        assert str(type(self.api_client)) == "<class 'alpha_vantage.techindicators.TechIndicators'>", 'Wrong api_client submitted'

        if num_cores != 1:
            return 'multi-processing to be completed later since the alpha_vantage free api does not allow more than 5 api calls per minute'

        else:

            # There are the following indicator definitions
            # indicator_dict - input dictionary of indicators from alpha_vantage
            # df_indicator_dict_i - the dictionary alpha_vantage returns
            # df_inidicator_i - the df within the above df_indicator_dict
            # df_indicator_dict - the final output dict that has all of the api df calls. The dfs within the values of this dict are eventually merged

            df_indicator_dict = {}
            num_api_calls = 0

            for s in self.symbols:

                # default return is json --- there is an option 'datatype' to return csvs but it is broken
                try:
                    df_indicator_dict_i = getattr(self.api_client, list(indicator_dict.keys())[0])(**indicator_dict[list(indicator_dict.keys())[0]],
                                                                                                   symbol = s) # call the api
                    df_indicator_i = df_indicator_dict_i[0]
                    df_indicator_i = df_indicator_i.add_prefix(df_indicator_dict_i[1]['1: Symbol'] + '_')
                except ValueError:
                    print('No data returned for the first indicator in INDICATOR DICT --- ' + list(indicator_dict.keys())[0])
                    sys.exit()

                num_api_calls += 1

                for i in range(len(list(indicator_dict.keys())))[1:]:

                    # inputs: kwarg examples: {'get_macd': {'interval': 'daily', 'time_period': 999999, 'series_type': 'close'}}

                    try:

                        df_indicator_dict_i2 = getattr(self.api_client, list(indicator_dict.keys())[i])(**indicator_dict[list(indicator_dict.keys())[i]],
                                                                                                        symbol = s) # call the api
                        num_api_calls += 1

                        df_indicator_i2 = df_indicator_dict_i2[0]
                        df_indicator_i2 = df_indicator_i2.add_prefix(df_indicator_dict_i2[1]['1: Symbol'] + '_')

                        df_indicator_i = pd.merge(df_indicator_i,
                                                  df_indicator_i2,
                                                  how = join_method,
                                                  left_index = True,
                                                  right_index = True)

                    except ValueError:
                        print('No data returned for indicator ' + list(indicator_dict.keys())[0])
                        num_api_calls += 1

                    if num_api_calls % 4 == 0: # mod 4 because alpha_advantage errors out at the 5th API call
                        time.sleep(60) # free version of alpha_vantage only allows 5 api calls per min, so sleep for 60 seconds every 5 calls

                df_indicator_dict[s] = df_indicator_i

        # merge all the dataframes in the dict values
        df_indicators = reduce(lambda x, y: pd.merge(x, y, how = join_method, left_index = True, right_index = True), list(df_indicator_dict.values()))
        return df_indicators

class CreateFeatures():

    def __init__(self, df_symbols, copy = True):

        self.df_symbols = df_symbols
        self.copy = copy

    def create_naive_features_single_symbol(self,\
                                            symbol='',\
                                            symbol_sep='_',\
                                            open_col='open',\
                                            high_col='high',\
                                            low_col='low',\
                                            close_col='close',\
                                            volume_col='volume'):

        if self.copy: self.df_symbols = self.df_symbols.copy()

        orig_cols = self.df_symbols.columns

        self.df_symbols[symbol + symbol_sep + 'move'] = self.df_symbols[symbol + symbol_sep + close_col] - self.df_symbols[symbol + symbol_sep + open_col]
        self.df_symbols[symbol + symbol_sep + 'move_pct'] = self.df_symbols[symbol + symbol_sep + 'move'] / self.df_symbols[symbol + symbol_sep + open_col]
        self.df_symbols[symbol + symbol_sep + 'move_pct_change'] = self.df_symbols[symbol + symbol_sep + 'move'].pct_change()

        self.df_symbols[symbol + symbol_sep + 'pct_chg'] = self.df_symbols[symbol + symbol_sep + 'move'] / self.df_symbols[symbol + symbol_sep + close_col].shift()

        self.df_symbols[symbol + symbol_sep + 'range'] = self.df_symbols[symbol + symbol_sep + high_col] - self.df_symbols[symbol + symbol_sep + low_col]
        self.df_symbols[symbol + symbol_sep + 'range_pct_change'] = self.df_symbols[symbol + symbol_sep + 'range'].pct_change()

        self.df_symbols[symbol + symbol_sep + 'high_move'] = self.df_symbols[symbol + symbol_sep + high_col] - self.df_symbols[symbol + symbol_sep + open_col]
        self.df_symbols[symbol + symbol_sep + 'high_move_pct'] = self.df_symbols[symbol + symbol_sep + 'high_move'] / self.df_symbols[symbol + symbol_sep + open_col]
        self.df_symbols[symbol + symbol_sep + 'high_move_pct_change'] = self.df_symbols[symbol + symbol_sep + 'high_move'].pct_change()

        self.df_symbols[symbol + symbol_sep + 'low_move'] = self.df_symbols[symbol + symbol_sep + low_col] - self.df_symbols[symbol + symbol_sep + open_col]
        self.df_symbols[symbol + symbol_sep + 'low_move_pct'] = self.df_symbols[symbol + symbol_sep + 'low_move'] / self.df_symbols[symbol + symbol_sep + open_col]
        self.df_symbols[symbol + symbol_sep + 'low_move_pct_change'] = self.df_symbols[symbol + symbol_sep + 'low_move'].pct_change()

        self.df_symbols[symbol + symbol_sep + 'volume_pct_change'] = self.df_symbols[symbol + symbol_sep + volume_col].pct_change()

        self.df_symbols[symbol + symbol_sep + 'low_minus_close'] = self.df_symbols[symbol + symbol_sep + low_col] - self.df_symbols[symbol + symbol_sep + close_col]
        self.df_symbols[symbol + symbol_sep + 'high_minus_close'] = self.df_symbols[symbol + symbol_sep + high_col] - self.df_symbols[symbol + symbol_sep + close_col]

        self.df_symbols[symbol + symbol_sep + 'low_minus_prev_close'] = self.df_symbols[symbol + symbol_sep + low_col] - self.df_symbols[symbol + symbol_sep + close_col].shift()
        self.df_symbols[symbol + symbol_sep + 'high_minus_prev_close'] = self.df_symbols[symbol + symbol_sep + high_col] - self.df_symbols[symbol + symbol_sep + close_col].shift()

        return self.df_symbols[[i for i in self.df_symbols.columns if not i in orig_cols]] # don't return original columns since the dfs are merged later


    def compute_naive_features(self, symbols, join_method = 'inner', num_workers = mp.cpu_count()):

        if num_workers != 1:

            df_init = self.df_symbols.copy() # create a copy to avoid duplicate columns

            delayed_list = [delayed(self.create_naive_features_single_symbol(symbol = s)) for s in symbols] # parallelize by symbol
            df_symbols_tuple = dask.compute(*delayed_list, num_workers = num_workers)
            output_list = [df_symbols_tuple[i] for i in range(len(df_symbols_tuple))]
            # df_symbols_out = pd.concat(output_list, axis=1, sort=False)

            df_features = reduce(lambda x, y: pd.merge(x, y, how = join_method, left_index = True, right_index = True), output_list)

            df_symbols_out = pd.merge(df_init,
                                      df_features,
                                      how = join_method,
                                      left_index = True,
                                      right_index = True)

        else:

            df_init = self.df_symbols.copy() # create a copy to avoid duplicate columns
            df_symbols_out = self.create_naive_features_single_symbol(symbol = symbols[0])
            df_symbols_out = pd.merge(df_init, df_symbols_out, how = join_method, left_index = True, right_index = True)

            for symbol in symbols[1:]:

                df_symbols_i = self.create_naive_features_single_symbol(symbol = symbol)

                # df_symbols_out = pd.concat([df_symbols_out, self.create_naive_features_single_symbol(symbol = symbol)], axis = 1)
                df_symbols_out = pd.merge(df_symbols_out,
                                          df_symbols_i,
                                          how = join_method,
                                          left_index = True,
                                          right_index = True)
        return df_symbols_out


    def move_iar(self, feature, copy = True):

        """
        Calculate move in a row
        This function is slow for a single feature since it using a for loop, but the function can be parallelized across different features
        """

        lst=[]
        prev_move_iar = 0

        for move in self.df_symbols[feature]:
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
        return pd.DataFrame(lst, index=self.df_symbols.index, columns=[feature]).rename(columns={feature: feature + '_iar'})

    def pos_neg_move(self, feature, min_move_up, copy = True):
        if self.copy: self.df_symbols = self.df_symbols.copy()
        return self.df_symbols[feature] > min_move_up


def RSI(prices, interval=14):

    '''
    Computes Relative Strength Index given a price series and lookback interval
    Modified from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
    See more here https://www.investopedia.com/terms/r/rsi.asp
    '''

    dUp, dDown = prices.diff().copy(), prices.diff().copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(interval).mean()
    RolDown = dDown.rolling(interval).mean().abs()

    RS = RolUp / RolDown
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI




class CreateTargets():

    def __init__(self,
                 df_symbols,
                 symbols,
                 copy = True):

        """
        Parameters
        __________

        df_symbols: pandas df where features are like AAPL_open, TSLA_close, etc...


        Note: to compute the target based on pct, pass the pct column names into the individual functions

        """

        self.df_symbols = df_symbols
        self.symbols = symbols
        self.copy = copy

        if self.copy: self.df_symbols = self.df_symbols.copy()

    def create_targets_HL5(self,
                           strong_buy,
                           med_buy,
                           med_sell,
                           strong_sell,
                           threshold,
                           stop,
                           move_col = '_move_pct',
                           lm_col = '_low_move_pct',
                           hm_col = '_high_move_pct',
                           target_suffix = '_target_HL5'):


        for s in self.symbols:
            # hm stands for high move, lm stands for low move
            # Strong Buy
            self.df_symbols.loc[(self.df_symbols[s + hm_col] >= strong_buy) &
                                (self.df_symbols[s + lm_col] >= (-1)*stop),
                                s + target_suffix] = 4

            # Strong Sell
            self.df_symbols.loc[(self.df_symbols[s + lm_col] <= (-1)*strong_sell) &
                        (self.df_symbols[s + hm_col] <= stop) &
                        (self.df_symbols[s + target_suffix] != 4),
                        s + target_suffix] = 0

            # Medium Buy
            self.df_symbols.loc[(self.df_symbols[s + hm_col] >= med_buy) &
                                (self.df_symbols[s + lm_col] >= (-1)*stop) &
                                (self.df_symbols[s + target_suffix] != 4) &
                                (self.df_symbols[s + target_suffix] != 0),
                                s + target_suffix] = 3

            # Medium Sell
            self.df_symbols.loc[(self.df_symbols[s + lm_col] <= (-1)*med_sell) &
                                (self.df_symbols[s + hm_col] <= stop) &
                                (self.df_symbols[s + target_suffix] != 4) &
                                (self.df_symbols[s + target_suffix] != 0) &
                                (self.df_symbols[s + target_suffix] != 3),
                                s + target_suffix] = 1

            # No Trade
            self.df_symbols.loc[(self.df_symbols[s + target_suffix] != 0) &
                                (self.df_symbols[s + target_suffix] != 1) &
                                (self.df_symbols[s + target_suffix] != 3) &
                                (self.df_symbols[s + target_suffix] != 4),
                                s + target_suffix] = 2


        return self.df_symbols


    def create_targets_HL3(self,
                           buy,
                           sell,
                           threshold,
                           stop,
                           move_col = '_move_pct',
                           lm_col = '_low_move_pct',
                           hm_col = '_high_move_pct',
                           target_suffix = '_target_HL3'):


        for s in self.symbols:
            # hm stands for high move, lm stands for low move
            # Buy
            self.df_symbols.loc[(self.df_symbols[s + hm_col] >= buy) &
                                (self.df_symbols[s + lm_col] >= (-1)*stop),
                                s + target_suffix] = 2

            # Sell
            self.df_symbols.loc[(self.df_symbols[s + lm_col] <= (-1)*sell) &
                                (self.df_symbols[s + hm_col] <= stop) &
                                (self.df_symbols[s + target_suffix] != 2),
                                s + target_suffix] = 0

            # No Trade
            self.df_symbols.loc[(self.df_symbols[s + target_suffix] != 0) &
                                (self.df_symbols[s + target_suffix] != 2),
                                s + target_suffix] = 1

        return self.df_symbols
