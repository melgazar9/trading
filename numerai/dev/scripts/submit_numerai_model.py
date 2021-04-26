import datetime
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta, FR
import gc
from configparser import ConfigParser

config = ConfigParser()
config.read('numerai/numerai_keys.ini')
napi = numerapi.SignalsAPI(config['KEYS']['NUMERAI_PUBLIC_KEY'], config['KEYS']['NUMERAI_SECRET_KEY'])


### load live data ###







last_friday = (datetime.datetime.now() + relativedelta(weekday=FR(-1))).strftime('%Y-%m-%d')

live_data = full_data.loc[last_friday].copy()
live_data.dropna(subset=feature_names, inplace=True)
