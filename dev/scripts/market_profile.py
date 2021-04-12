# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:38:51 2021

@author: Matt
"""

from market_profile import MarketProfile
import pandas as pd
import numpy as np
import alpaca_trade_api as alpaca_api
import os


WORKING_DIRECTORY = 'C:/Users/Matt/trading/dev/'
CONFIG_DIRECTORY = WORKING_DIRECTORY + 'configs/'

os.chdir(WORKING_DIRECTORY + 'configs')

from alpaca_config import *

api = alpaca_api.REST(**ALPACA_HEADERS)
df_aapl = api.polygon.historic_agg_v2('AAPL', multiplier = 1, timespan = 'day', _from = '2015-01-01', to = '2099-01-01').df

mp = MarketProfile(df_aapl)
mp[df_aapl.index.min() : df_aapl.index.max()]
df_aapl
