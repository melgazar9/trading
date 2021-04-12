# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 17:09:21 2020

@author: Matt
"""

import pandas as pd


API_TOKEN = 'pk_795b51c5697942c79e6c0bd0be5a5a65' # production key
SANDBOX_TOKEN = 'Tpk_6180bf08ab1b4ed9babd9d5f43b87c94' # sandbox (testing) key


# =============================================================================
# pyEX
# =============================================================================

import pyEX as p

client = p.Client(api_token = SANDBOX_TOKEN, version='sandbox')

client.quote(symbol = 'AAPL')

# Historical data
df = client.chartDF(symbol = 'AAPL', date = 'max')




# =============================================================================
# iexfinance
# =============================================================================

# Need to pay for this

from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data

df_stocks = Stock(['AAPL', 'TSLA'], token = SANDBOX_TOKEN) # live feed
df_stocks = get_historical_data(['AAPL', 'TSLA'], start = '2010-01-01', token = API_TOKEN)







































# =============================================================================
# requests
# =============================================================================




import requests
import json

base_url = 'https://cloud.iexapis.com/stable/'
sandbox_url = 'https://sandbox.iexapis.com/stable/'
#endpoint = 'stock/AAPL/intraday-prices'


# GET /stock/{symbol}/intraday-prices
intraday_data = requests.get(base_url + 'stock/AAPL/intraday-prices?token=' + API_TOKEN)
intraday_data = requests.get('https://cloud.iexapis.com/stock/AAPL/chart/range=15/date=2020-01-01?token=' + API_TOKEN)

json_req = json.loads(intraday_data.content)
df_req = pd.DataFrame.from_dict(json_req)
df_req.columns


# GET /stock/{symbol}/batch
#batch_data = requests.get(base_url + 'stock/market/batch?types=chart,splits,news&symbols=aapl,goog,fb&range=5y&token=' + API_TOKEN)
#batch_data = requests.get(sandbox_url + 'stock/market/batch?types=chart,splits,news&symbols=aapl,goog,fb&range=5y&token=' + SANDBOX_TOKEN)
batch_data = requests.get(sandbox_url + 'stock/market/batch?types=chart&symbols=aapl,goog,fb&range=5y&token=' + SANDBOX_TOKEN)
batch_json = json.loads(batch_data.content)
df_batch = pd.DataFrame.from_dict(batch_json)

#df_fb = pd.DataFrame.from_dict(df_batch['FB']['chart'])

df_merged = pd.DataFrame.from_dict(df_batch[df_batch.columns[0]]['chart']).set_index('date')
df_merged = df_merged.add_prefix(df_batch.columns[0] + '_')
for col in df_batch.columns[1:]:
    df_i = pd.DataFrame.from_dict(df_batch[col]['chart']).set_index('date')
    df_i = df_i.add_prefix(col + '_')
    df_merged = pd.merge(df_merged,
                         df_i,
                         how = 'inner',
                         left_index = True, 
                         right_index = True)
