# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 15:57:13 2020

@author: Matt
"""
import pandas as pd
import numpy as np
from fredapi import Fred
import multiprocessing as mp
from dask import delayed
import dask
import itertools
import datetime
import os




##############################
###### GLOBAL VARIABLES ######
##############################

WORKING_DIRECTORY = 'C:/Users/Matt/trading/'
os.chdir(WORKING_DIRECTORY)

fred = Fred(api_key='704531c00173911aa199c0f2d418f9ce')
FRED_REQUESTS = ['GDP','GDPC1', 'A191RL1Q225SBEA', 'GFDEGDQ188S', 'CPIAUCSL', 'UNRATE', 'PAYEMS', 'IC4WSA', 'DGS10', 'FEDFUNDS', 'USD3MTD156N']


#fred_data = fred.get_series_all_releases(**FRED_REQUESTS)
#fred.search('non-farm').T


################################
###### Build Fred Dataset ######
################################

class BuildFredData():
    
    def __init__(self, 
                 limit = 10, 
                 order_by = 'popularity', 
                 sort_order = 'desc', 
                 num_cores = mp.cpu_count()):
        
        self.limit = limit 
        self.order_by = order_by
        self.sort_order = sort_order
        self.num_cores = num_cores


    def pull_fred_category_id(self, cat_id):
        
        try:
            top_fred_id = fred.search_by_category(cat_id, 
                                                  limit = self.limit, 
                                                  order_by = self.order_by, 
                                                  sort_order = self.sort_order).index.tolist()
            return top_fred_id
        except:
            return


    def build_fred_data(self, category_ids):
        
        delayed_dict = {i : delayed(self.pull_fred_category_id)(cat_id = i) for i in category_ids}
        top_fred_ids = dask.compute(*list(delayed_dict.values()))
        top_fred_ids = np.unique(list(itertools.chain.from_iterable([i for i in top_fred_ids if not i is None])))
        
        return top_fred_ids
    
top_fred_ids = BuildFredData().build_fred_data(category_ids =  [i for i in range(1, 1001)])

np.savetxt('outputs/top_fred_ids.txt', top_fred_ids, fmt = '%s', delimiter =',')
# to load: np.loadtxt('outputs/top_fred_ids.txt', dtype = str)
# all(top_fred_ids == np.loadtxt('outputs/top_fred_ids.txt', dtype = str))
