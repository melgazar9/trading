# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:57:00 2020

@author: Matt
"""

#####################
###### Imports ######
#####################

import os
import datetime

import json
from twelvedata import TDClient
import pandas as pd

##############################
###### Global Variables ######
##############################

WORKING_DIRECTORY = 'C:/Users/Matt/trading/dev/'
CONFIG_DIRECTORY = WORKING_DIRECTORY + 'configs/'

os.chdir(WORKING_DIRECTORY + 'scripts/')
from dev.scripts.trading_utils import *

os.chdir(CONFIG_DIRECTORY)
from dev.configs.pull_data_config import *

pd.set_option('display.max_columns', 10)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', 3)



#######################
###### Pull Data ######
#######################

    
# raw alpaca examples
# df_symbols = alpaca_api.get_barset(['AAPL', 'TSLA'], '1Min', limit = None).df
# df_symbols.columns = ['_'.join(col).strip() for col in df_symbols.columns.values]
# df_minute = api.polygon.historic_agg_v2('AAPL', 1, timespan = 'minute', _from='2019-01-01', to='2019-02-01').df

# df_symbols = PullHistoricalData(api_client = alpaca_api.REST(**ALPACA_HEADERS),
#                       symbols = ['AAPL', 'TSLA', 'FB', 'SPY', 'QQQ']).get_barset_data()

if type(SYMBOLS_TO_PULL) != list or len(STOCKS) == 0:

    # ASSETS = alpaca_api.REST(**ALPACA_HEADERS).list_assets(status = 'active')

    # STOCKS = [i.symbol for i in ASSETS if i.tradable] # only take the tradeable assets # ~9371 stocks

    stock_req = requests.get(TWELVEDATA_BASE_URL + '/stocks', TWELVEDATA_HEADERS['apikey'])
    if json.loads(stock_req.content)['status'] == 'ok':
        df_stock_symbols = pd.DataFrame.from_dict(json.loads(stock_req.content)['data'])

    etf_req = requests.get(TWELVEDATA_BASE_URL + '/etf', TWELVEDATA_HEADERS['apikey'])
    if json.loads(etf_req.content)['status'] == 'ok':
        df_etf_symbols = pd.DataFrame.from_dict(json.loads(etf_req.content)['data'])

    if US_ONLY:
        df_stock_symbols = df_stock_symbols.loc[df_stock_symbols['country'] == 'United States']
        df_etf_symbols = df_etf_symbols.loc[df_etf_symbols['currency'] == 'USD']

    if IGNORE_OTC:
        df_stock_symbols = df_stock_symbols.loc[~df_stock_symbols['exchange'].isin(['OTC'])]
        df_etf_symbols = df_etf_symbols.loc[df_etf_symbols['currency'] == 'USD']
    SYMBOLS_TO_PULL = list(set(df_stock_symbols['symbol']))
    #SYMBOLS_TO_PULL = list(set(list(df_stock_symbols['symbol']) + list(df_etf_symbols['symbol']))) # could be ~109 duplicate etf / stock symbols when removing the "set" operation

if TIMESERIES_API_SOURCE.lower() == 'twelvedata':

    # Get raw stock time series data from Twelvedata
    df_symbols = PullHistoricalData(api_client=TDClient(**TWELVEDATA_HEADERS),
                                    symbols=SYMBOLS_TO_PULL).twelvedata_pull_timeseries(interval='1day',
                                                                                        max_requests_per_min=8,
                                                                                        max_requests_per_day=800)

elif TIMESERIES_API_SOURCE.lower() == 'alpaca':
    # Get raw stock time series data from alpaca
    df_symbols = PullHistoricalData(api_client=alpaca_api.REST(**ALPACA_HEADERS),
                                    symbols=STOCKS).get_historic_agg_v2_data(timespan='day',
                                                                             join_method='left',
                                                                             left_table='AAPL')

# Get technical indicators
# if TECHINDICATOR_API_SOURCE.lower() == 'alpha_vantage':
    # TechIndicators(**AV_HEADERS).get_macd(symbol = 'AAPL')
    # getattr(TechIndicators(**AV_HEADERS), "get_macd")(symbol = 'AAPL', interval = 'daily', series_type = 'close')

    # indicator api fails when trying to pull too many indicators - can only call alpha_vantage API 500 times per day, or 4 api calls per minute
    # df_indicators = PullHistoricalData(api_client = TechIndicators(**AV_HEADERS),
    #                                    symbols = ['AAPL', 'SPY', 'QQQ']).get_technical_indicators(indicator_dict=AV_INDICATOR_DICT,
    #                                                                                               join_method = 'inner',
    #                                                                                               num_cores = 1)

if save_initial_pull:
    if save_df_init_filename.endswith('feather'):
        df_symbols.to_feather(WORKING_DIRECTORY + save_df_init_filename)
    elif save_df_init_filename.endswith('csv'):
        df_symbols.to_csv(WORKING_DIRECTORY + save_df_init_filename, compression = compression)



# Compute naive features
df_symbols = CreateFeatures(df_symbols = df_symbols).compute_naive_features(STOCKS)





############################
###### Create Targets ######
############################

if create_HL5_target:

    df_symbols = CreateTargets(df_symbols, symbols = STOCKS).create_targets_HL5(**target_HL5_params)

if create_HL3_target:
    
    df_symbols = CreateTargets(df_symbols, symbols = STOCKS).create_targets_HL3(**target_HL3_params)

if save_df_with_targets_and_features:
    if df_naive_features_and_targets_filename.endswith('feather'):
        df_symbols.reset_index().to_feather(WORKING_DIRECTORY + df_naive_features_and_targets_filename)
    elif df_naive_features_and_targets_filename.endswith('pq') or df_naive_features_and_targets_filename.endswith('parquet'):
        df_symbols.to_parquet(WORKING_DIRECTORY + df_naive_features_and_targets_filename)
    elif df_naive_features_and_targets_filename.endswith('csv'):
        df_symbols.to_csv(WORKING_DIRECTORY + df_naive_features_and_targets_filename, compression = compression)













