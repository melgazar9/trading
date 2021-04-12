# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:55:36 2021

@author: Matt
"""

# output_name = WORKING_DIRECTORY + 'outputs/data/df_symbols_daily_' + str(datetime.datetime.now().replace(second=0, microsecond=0)).replace(' ', '_') + '.csv' # fails on windows
# output_name = WORKING_DIRECTORY + 'outputs/data/df_symbols_daily_features_and_target.csv'
# df_symbols.to_csv(output_name)

# tmp = PullHistoricalData(api_client = alpaca_api.REST(**ALPACA_HEADERS),
#                                symbols = ['NIO']).get_historic_agg_v2_data(timespan = 'day',
#                                                                           join_method = 'left',
#                                                                           left_table = 'AAPL')


# tmp2 = alpaca_api.REST(**ALPACA_HEADERS).polygon.historic_agg_v2('NIO', 1, 'day', _from='1900-01-01', to='2099-12-31').df
# #tmp3 = alpaca_api.REST(**ALPACA_HEADERS).get_barset('NIO', 'day', start='1900-01-01', end='2099-12-31', limit = 999999999).df
# #tmp3.index.min()



# EDA

# for col in df_symbols[[i for i in df_symbols.columns if i.endswith('HL5') or i.endswith('HL3')]].columns:
#     print(df_symbols[col].value_counts())


# df_symbols['TSLA_target_HL5'].value_counts()
