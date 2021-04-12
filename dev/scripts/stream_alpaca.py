# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 14:13:08 2020

@author: Matt
"""

##############################
###### GLOBAL VARIABLES ######
##############################

WORKING_DIRECTORY = 'C:/Users/Matt/trading/'
CONFIG_DIRECTORY = WORKING_DIRECTORY + 'configs/'

import alpaca_trade_api as tradeapi
import os
os.chdir(WORKING_DIRECTORY + 'scripts/')
from trading_utils import *
os.chdir(CONFIG_DIRECTORY)
from stock_config import *

conn = tradeapi.StreamConn(**HEADERS)


@conn.on(r'^trade_updates$')
async def on_account_updates(conn, channel, account):
    print('account', account)

@conn.on(r'^status$')
async def on_status(conn, channel, data):
    print('polygon status update', data)

@conn.on(r'^AM$')
async def on_minute_bars(conn, channel, bar):
    print('bars', bar)

@conn.on(r'^A$')
async def on_second_bars(conn, channel, bar):
    print('bars', bar)

# blocks forever
conn.run(['trade_updates', 'AM.*'])