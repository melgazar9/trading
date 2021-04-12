# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:51:17 2020

@author: Matt
"""

import pandas as pd
import requests
import os
import dask
from dask import delayed
import multiprocessing as mp
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.cryptocurrencies import CryptoCurrencies
from alpha_vantage.foreignexchange import ForeignExchange
from alpha_vantage.sectorperformance import SectorPerformances


##############################
###### Global Variables ######
##############################


API_KEY = 'D46872A04A9M1143L'
API_HEADERS = {'key' : API_KEY, 'output_format' : 'pandas'}

# timeseries_api = TimeSeries(key = API_KEY, output_format = 'pandas')
# techindicator_api = TechIndicators(key = API_KEY)
# crypto_api = CryptoCurrencies(key = API_KEY)
# forex_api = ForeignExchange(key = API_KEY)
# sector_api = SectorPerformances(key = API_KEY)










class PullAVData():
    
    """
    
    PARAMETERS
    __________
    
    api_headers : dict input api parameters from the Alpha Vantage API 
                  Every api_headers input should have the API key
    
    symbols : A list of symbols to pull - this can also be a combination of stocks, futures, or forex (options might be addressed later)
    
    """
    
    def __init__(self, api_headers, symbols, copy = True):
        
        self.api_headers = api_headers
        self.symbols = symbols
        self.copy = copy
    

    def create_historical_timeseries_df(self, interval = '1min', outputsize = 'full'):

        timeseries_api = TimeSeries(**self.api_headers)     
        
        for symbol in self.symbols:
            
            timeseries_dict = timeseries_api.get_intraday(symbol, interval = interval, outputsize = outputsize)
            df_timeseries = timeseries_dict[0]
            df_timeseries.columns = [i[3: ] for i in df_timeseries.columns]
            renamed_dict = {k[3: ] : v for k, v in timeseries_dict[1].items()}
            df_timeseries = df_timeseries.add_prefix(renamed_dict['Symbol'] + '_')
            df_timeseries.sort_index(inplace = True)
        
        return df_timeseries

    # def __init__(self, api, symbols_to_pull, dropna_rows, copy):
        
    #     # api example: tradeapi.REST(**self.api_headers)
        
    #     self.api = api
    #     self.symbols_to_pull = symbols_to_pull
    #     self.dropna_rows = dropna_rows
    #     self.copy = copy
    
    # def pull_alpaca_symbol_data(self, 
    #                     api_call = 'get_barset',
    #                     time_interval = 'day',
    #                     limit = None):
        
        # if api_call == 'get_barset':
        #     df = self.api.get_barset(self.symbols_to_pull, time_interval, limit = limit).df
        #     df.columns = ['_'.join(col).strip() for col in df.columns.values]
        
        # elif api_call == 'historic_agg_v2':
        #     try:
        #         df = self.api.polygon.historic_agg_v2(self.symbols_to_pull, multiplier = 1, timespan = time_interval, _from='1900', to='2099-12-31').df
        #         if df.shape[0] < 100:
        #             print('****** Warning - cannot reliably pass a list of symbols to historic_agg_v2. Only selecting the first symbol in the list! ******')
        #             df = self.create_alpaca_historic_agg_v2_df()
        #     except:
        #         df = self.create_alpaca_historic_agg_v2_df()
        
        
        # if self.dropna_rows:  
        #     df.dropna(inplace = True)
        
        # return df    
    
    # def create_alpaca_historic_agg_v2_df_single_thread(self):
    
    #     if self.dropna_rows:
    #         join_method = 'inner'
    #     else:
    #         join_method = 'outer'
    
    #     symbol_df = self.api.polygon.historic_agg_v2(self.symbols_to_pull[0], multiplier = 1, timespan = 'day', _from='1900-01-01', to='2099-02-01').df
    #     symbol_df.columns = self.symbols_to_pull[0] + '_' + symbol_df.columns
    
    #     for symbol in self.symbols_to_pull[1:]:
    #         tmp_df = self.api.polygon.historic_agg_v2(symbol, 1, 'day', _from='1900-01-01', to='2099-02-01').df
    #         tmp_df.columns = symbol + '_' + tmp_df.columns
    #         symbol_df = pd.merge(symbol_df, tmp_df, how = join_method, left_index = True, right_index = True)
            
    #     return symbol_df
    
    
    # def create_alpaca_historic_agg_v2_df_parallel(self, join_method = 'inner'):
        
    #     delayed_list = [delayed(self.api.polygon.historic_agg_v2)(symbol, 1, 'day', _from='1900-01-01', to='2099-02-01').df for symbol in self.symbols_to_pull]
    #     df_tuple = dask.compute(*delayed_list)
    #     output_list = [df_tuple[i] for i in range(len(df_tuple))]            
    #     df_out = pd.concat(output_list, axis=1, sort=False)
        
    #     return df_out
        






















class CreateFeatures():
    
    def __init__(self, df):
        
        self.df = df
    
    def create_naive_features_single_symbol(self, symbol):
        
        if self.copy: self.df = self.df.copy()
                    
        self.df[symbol + '_move'] = self.df[symbol + '_close'] - self.df[symbol + '_open']
        self.df[symbol + '_pct_chg'] = self.df[symbol + '_move'] / self.df[symbol + '_close']
        self.df[symbol + '_range'] = self.df[symbol + '_high'] - self.df[symbol + '_low']
        self.df[symbol + '_range_pct'] = self.df[symbol + '_range'] / self.df[symbol + '_open']
        self.df[symbol + '_high_move'] = self.df[symbol + '_high'] - self.df[symbol + '_high'].shift()
        self.df[symbol + '_high_move_pct'] = self.df[symbol + '_high_move'] / self.df[symbol + '_open']
        self.df[symbol + '_low_move'] = self.df[symbol + '_low'] - self.df[symbol + '_low'].shift()
        self.df[symbol + '_low_move_pct'] = self.df[symbol + '_low_move'] / self.df[symbol + '_open']
        self.df[symbol + '_volume_chg'] = self.df[symbol + '_volume'] - self.df[symbol + '_volume'].shift()
        self.df[symbol + '_low_minus_close'] = self.df[symbol + '_low'] - self.df[symbol + '_close']
        self.df[symbol + '_high_minus_close'] = self.df[symbol + '_high'] - self.df[symbol + '_close']
        self.df[symbol + '_low_minus_prev_close'] = self.df[symbol + '_low'] - self.df[symbol + '_close'].shift()
        self.df[symbol + '_high_minus_prev_close'] = self.df[symbol + '_high'] - self.df[symbol + '_close'].shift()
        
        return self.df
        
    def compute_naive_features(self, symbols, run_parallel = True, num_cores = mp.cpu_count()):
        
        if run_parallel:
                
            delayed_list = [delayed(self.create_naive_features_single_symbol)(df = self.df, symbol = symbol) for symbol in symbols]
            df_tuple = dask.compute(*delayed_list)
            output_list = [df_tuple[i] for i in range(len(df_tuple))]            
            df_out = pd.concat(output_list, axis=1, sort=False)
        
        else:
            for symbol in symbols:
                df_out = pd.concat([df_out, self.create_naive_features_single_symbol(df = self.df, symbol = symbol)], axis = 1)
        
        return df_out

        


class CreateTarget():
    
    def __init__(self):
        
        self.df = df
        
    def create_target_HL(self,
                         strong_buy_pct = .07,
                         med_buy_pct = .03,
                         med_sell_pct = .03,
                         strong_sell_pct = .07,
                         stop_pct = .02):
        
        # Strong Buy
        self.df.loc[(self.df['high_move_pct'] >= strong_buy_pct) & 
                   (self.df['low_move_pct'] >= (-1)*stop_pct), 
                   'target_HL'] = 4
        
        # Strong Sell
        self.df.loc[(self.df['low_move_pct'] <= (-1)*strong_sell_pct) & 
                   (self.df['high_move_pct'] <= stop_pct) & 
                   (self.df['target_HL'] != 4), 'target_HL'] = 0
    
        # Medium Buy
        self.df.loc[(self.df['high_move'] >= med_buy_pct) & 
                   (self.df['low_move_pct'] >= (-1)*stop_pct) & 
                   (self.df['target_HL'] != 4) & 
                   (self.df['target_HL'] != 0), 
                   'target_HL'] = 3
    
        # Medium Sell
        self.df.loc[(self.df['low_move_pct'] <= (-1)*med_sell_pct) & 
                   (self.df['high_move_pct'] <= stop_pct) & 
                   (self.df['target_HL'] != 4) & (self.df['target_HL'] != 0) & 
                   (self.df['target_HL'] != 3), 
                   'target_HL'] = 1
    
        # No Trade
        self.df.loc[(self.df['target_HL'] != 0) & 
                   (self.df['target_HL'] != 1) & 
                   (self.df['target_HL'] != 3) & 
                   (self.df['target_HL'] != 4), 
                   'target_HL'] = 2
        return








symbol = timeseries_api.get_daily_adjusted('AAPL', outputsize='full')

base_url = 'https://www.alphavantage.co/query?'
params = {'function': 'TIME_SERIES_DAILY_ADJUSTED',
		 'symbol': 'IBM',
		 'apikey': api_key}

response = requests.get(base_url, params=params)
print(response.json())


forex_app = ForeignExchange(api_key, output_format='pandas')

eurusd = app.get_currency_exchange_intraday('EUR', 'USD')
print(eurusd[0])




base_url = 'https://www.alphavantage.co/query?'
params = {'function': 'FX_INTRADAY',
  'from_symbol': 'EUR',
  'to_symbol': 'USD', 
  'interval': '15min',
  'apikey': api_key}

response = requests.get(base_url, params=params)
response_dict = response.json()
_, header = response.json()

#Convert to pandas dataframe
df = pd.DataFrame.from_dict(response_dict[header], orient='index')

#Clean up column names
df_cols = [i.split(' ')[1] for i in df.columns]
df.columns = df_cols
#df.set_index('timestamp', inplace=True) #Time-series index 


indicator_api = TechIndicators(api_key, output_format='pandas')
#help(app.get_macd)
aapl_macd = indicator_api.get_macd('aapl', fastperiod=12, slowperiod=26, signalperiod=9)

