import configparser
config = configparser.ConfigParser()
config.read('api_keys.ini')

TIMESERIES_API_SOURCE = 'twelvedata' # possible values 'twelvedata' or 'alpaca'
TECHINDICATOR_API_SOURCE = 'twelvedata' # possible values 'twelvedata' 'alpaca', or 'alphavantage'

############################################
################# API keys #################
############################################


####################
###### ALPACA ######
####################

ALPACA_BASE_URL = 'https://api.alpaca.markets'
ALPACA_ACCOUNT_URL = '{}/v2/account'.format(ALPACA_BASE_URL)
ALPACA_ORDERS_URL = '{}/v2/orders'.format(ALPACA_BASE_URL)


ALPACA_HEADERS = {'key_id': config['KEYS']['alpaca_api_key'], 'secret_key': config['KEYS']['alpaca_secret_key'], 'base_url' : ALPACA_BASE_URL}


########################
###### TWELVEDATA ######
########################

TWELVEDATA_BASE_URL = 'https://api.twelvedata.com'

TWELVEDATA_HEADERS = {'apikey': config['KEYS']['twelvedata_api_key']}

################################
###### PULL / SAVE PARAMS ######
################################

US_ONLY = True
IGNORE_OTC = True

SYMBOLS_TO_PULL = ['AAPL', 'TSLA', 'AMZN', 'FB', 'NFLX', 'GOOG', 'QQQ', 'SPY', 'IWM', 'NVDA', 'MSFT'] # 'all'

compression = None

save_initial_pull = True
save_df_init_filename = 'outputs/data/df_twelvedata_daily_left.feather'


save_df_with_targets_and_features = True
df_naive_features_and_targets_filename = 'outputs/data/df_twelvedata_daily_features_and_target.feather'

############################
###### Create Targets ######
############################

create_HL3_target = True
create_HL5_target = True

target_HL3_params = {'buy': .03,
                     'sell': .03,
                     'stop': .02,
                     'threshold': 10,
                     'move_col': '_move_pct',
                     'lm_col': '_low_move_pct',
                     'hm_col': '_high_move_pct'
                     }

target_HL5_params = {'strong_buy': .05,
                     'strong_sell': .05,
                     'med_buy': .02,
                     'med_sell': .02,
                     'stop': .02,
                     'threshold': 10,
                     'move_col': '_move_pct',
                     'lm_col': '_low_move_pct',
                     'hm_col': '_high_move_pct',
                     }






























###########################
###### ALPHA VANTAGE ######
###########################

AV_HEADERS = {'key': config['KEYS']['av_api_key'], 'output_format': 'pandas'}



# pass None to pull all indicators --- [i for i in dir(TechIndicators) if i.startswith('get')]
# otherwise pass a list of indicators inside alpha_vantage.techindicators.TechIndicators
AV_INDICATOR_DICT = {
                  'get_macd': {'interval': 'daily',
                               'fastperiod': 12,
                               'slowperiod': 26,
                               'signalperiod': 9},
                  
                  'get_rsi': {'interval': 'daily',
                               'time_period': 30,
                               'series_type': 'close'},
                  
                  'get_vwap': {'interval': 'daily'},
                  
                  'get_stoch': {'interval': 'daily',
                                 'fastkperiod': 5,
                                 'slowkperiod': 3,
                                 'slowdperiod': 3},
                  
                  'get_ht_trendline': {'interval': 'daily',
                                        'series_type': 'close'},
                  
                  'get_ht_trendmode': {'interval': 'daily',
                                        'series_type': 'close'},
                  
                  'get_ht_phasor': {'interval': 'daily',
                                     'series_type': 'close'},
                  
                  'get_adx': {'interval': 'daily',
                               'time_period': 30},
                  
                  'get_ema': {'interval': 'daily',
                               'series_type': 'close'},
                  
                  'get_mom': {'interval': 'daily',
                               'time_period': 30,
                               'series_type': 'close'},
                  
                  'get_cci': {'interval': 'daily',
                               'series_type': 'close'},
                  
                  'get_roc': {'interval': 'daily',
                               'time_period': 30},
                  
                  'get_aroon': {'interval': 'daily',
                               'time_period': 30},
                  
                  'get_bbands': {'interval': 'daily',
                                  'time_period': 30,
                                  'series_type': 'close',
                                  'nbdevup': 2,
                                  'nbdevdn': 2,
                                  'matype': 2}, # matypes: {1: SMA, 2: EMA, 3: WMA, 4: TEMA, 5: TRIMA, 6: T3, 7: KAMA, 8: MAMA}
                  
                  'get_atr': {'interval': 'daily',
                              'time_period': 30},
                  
                  'get_stochrsi': {'interval': 'daily',
                                    'time_period': 30,
                                    'series_type': 'close'}
                  }