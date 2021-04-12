########## Imports ##########

# General Python
import os
import sys
import multiprocessing as mp
import gc
from functools import reduce
import requests as re
import datetime
from dateutil.relativedelta import relativedelta, FR


### API
import numerapi
import yfinance
import simplejson


### Data Manipulation
import pandas as pd
# import vaex

### Visualization
import matplotlib.pyplot as plt

import numpy as np
# import modin
# if __name__ == "__main__": import modin.pandas as mpd

### EDA

### ML
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import *
from xgboost import XGBRegressor, XGBRFRegressor, XGBClassifier, XGBRFClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import *


########## Classes & Functions ##########


def download_yfinance_data(tickers,
                           intervals_to_download=['1d', '1h'],
                           num_workers=1,
                           join_method='outer',
                           max_intraday_lookback_days=363,
                           n_chunks=600,
                           yfinance_threads=False,
                           **yfinance_params):
    """
    Parameters
    __________

    See yfinance.download docs for a detailed description of yfinance parameters

    tickers : list of tickers to pass to yfinance.download - it will be parsed to be in the format "AAPL MSFT FB"
    intervals_to_download : list of intervals to download OHLCV data for each stock (e.g. ['1w', '1d', '1h'])
    num_workers : number of threads used to download the data
        so far only 1 thread is implemented
    join_method : can be 'inner', 'left', 'right' or 'outer'
        if 'outer' then all dates will be present
        if 'left' then all dates from the left most table will be present
        if 'right' then all dates from the left most table will be present
        if 'inner' then all dates must match for each ticker
    **yfinance_params : dict - passed to yfinance.dowload(yfinance_params)
        set threads = True for faster performance, but tickers will fail, scipt may hang
        set threads = False for slower performance, but more tickers will succeed

    NOTE: passing some intervals return unreliable stock data (e.g. '3mo' returns many NA data points when they should not be NA)

    """

    yfinance_params['threads'] = yfinance_threads
    yfinance_params2 = yfinance_params.copy() # create a copy for min / hour pulls because the start date can only go back 60 days

    if num_workers == 1:

        list_of_dfs = []

        for i in intervals_to_download:

            yfinance_params['interval'] = i

            if i.endswith('m') or i.endswith('h'): # min or hr

                yfinance_params2['interval'] = i
                yfinance_params2['start'] = str(datetime.datetime.today().date() - datetime.timedelta(days=max_intraday_lookback_days))

                if yfinance_params['threads'] == True:
                    df_i = yfinance.download(tickers, **yfinance_params2).\
                            stack().\
                            add_suffix('_' + str(i)).\
                            reset_index(level=1).\
                            rename(columns={'level_1' : 'ticker'})
                else:

                    ticker_chunks = [' '.join(tickers[i:i+n_chunks]) for i in range(0, len(tickers), n_chunks)]

                    chunk_dfs_lst = []
                    for chunk in ticker_chunks:
                        try:
                            temp_df = yfinance.download(chunk, **yfinance_params2).\
                                        stack().\
                                        add_suffix('_' + str(i)).\
                                        reset_index(level=1).\
                                        rename(columns={'level_1' : 'ticker'})
                            chunk_dfs_lst.append(temp_df)
                        except simplejson.errors.JSONDecodeError:
                            pass

                    df_i = pd.concat(chunk_dfs_lst)

                df_i = df_i.pivot_table(index=df_i.index.date, columns = ['ticker', df_i.index.hour]).stack(level=1)
                df_i.columns = list(pd.Index([str(e[0]).lower() + '_' + str(e[1]).lower() for e in df_i.columns.tolist()]).str.replace(' ', '_'))

            else:
                if yfinance_params['threads'] == True:
                    df_i = yfinance.download(tickers, **yfinance_params).\
                            stack().\
                            add_suffix('_' + str(i))
                else:
                    ticker_chunks = [' '.join(tickers[i:i+n_chunks]) for i in range(0, len(tickers), n_chunks)]

                    chunk_dfs_lst = []
                    for chunk in ticker_chunks:
                        try:
                            temp_df = yfinance.download(chunk, **yfinance_params).\
                                        stack().\
                                        add_suffix('_' + str(i))
                            chunk_dfs_lst.append(temp_df)
                        except simplejson.errors.JSONDecodeError:
                            pass
                    df_i = pd.concat(chunk_dfs_lst)

                df_i.columns = [col.replace(' ', '_').lower() for col in df_i.columns]

            df_i.index.names = ['date', 'ticker']

            list_of_dfs.append(df_i)


        df_yahoo = reduce(lambda x, y: pd.merge(x, y, how=join_method, left_index=True, right_index=True), list_of_dfs)
#         df_yahoo.reset_index(level=1, inplace=True)

    else:
        return 'multi-threading not implemented yet. Set num_workers to 1.'

    return df_yahoo
