import os
import time
import json
import pandas as pd
import numpy as np
import gc
import datetime
from pandas.io.json import json_normalize
import itertools
import configparser
import warnings
import re
import sys

config = configparser.ConfigParser()
config.read('/home/melgazar9/Trading/TD/Scripts/Trading-Scripts/ES/scripts/ES_1min_concat-new-data.ini')

warnings.filterwarnings('ignore')


######################################
#       READ THE CORRECT FILE        #
######################################

def read_data(json_file):
    # read the entire file into a python array
    with open(json_file, 'rb') as f:
        data = f.readlines()

    # remove the trailing "\n" from each line
    data = map(lambda x: x.rstrip(), data)

    data_list = []
    try:
        #data = [json.loads(x) for x in data]
        for x in data:
            data_list.append(json.loads(x))
    except ValueError:
        print('THERE IS A NAN VALUE')

    return data_list


######################################
#        VERIFY DATA IS CORRECT      #
######################################


def verify_data_integrity(raw_data):
    json_response = raw_data[0]
    for k1 in json_response['response']:
        if (k1['command'] == 'LOGIN') and ((list(k1['content'].keys()) == ['msg', 'code']) or list(k1['content'].keys()) == ['code', 'msg']) and (k1['service'] == 'ADMIN'):
            print(True)
        else:
            print('REASON: RESPONSE ******Disconnecting****** REASON: RESPONSE')
            sys.exit()

    json_heartbeat = raw_data[1]
    for k2 in json_heartbeat['snapshot']:
        for k2a in k2.keys():
            if k2a == 'content':
                print(True)
            elif k2a == 'timestamp':
                print(True)
            elif k2a == 'command':
                print(True)
            elif k2a == 'service':
                print(True)
            else:
                print('REASON: NOTIFY ******Disconnecting****** REASON: NOTIFY')
                sys.exit()

    return


def get_1min_df(raw_data):

    cols = {'Key':'/' + config['PARAMS']['product'], '0':'Datetime','1':'1minOpen','2':'1minHigh','3':'1minLow','4':'1minClose','5':'1minVolume'}

    data = pd.io.json.json_normalize(raw_data)['snapshot'].dropna()
    for lst1 in data:
        for lst2 in lst1:
            for lst3 in lst2['content']:
                df = pd.DataFrame(lst3['3'])

    df = df.rename(columns=cols)
    df['1minRange'] = df['1minHigh'] - df['1minLow']
    df['1minMove'] = df['1minClose'] - df['1minOpen']
    df['1minLowMove'] = df['1minLow'] - df['1minOpen']
    df['1minHighMove'] = df['1minHigh'] - df['1minOpen']
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index, unit='ms')
    df = df.sort_index()

    return df


if config['PARAMS']['old_data_json_ON'] == 'TRUE':
    old_data = config['PATH']['old_data_json']
    raw_data_old = read_data(old_data)
    verify_data_integrity(raw_data_old)
    df_1min_old = get_1min_df(raw_data_old)
    old_df = df_1min_old.sort_index()

if config['PARAMS']['old_data_json_ON'] == 'FALSE':
    if config['PARAMS']['read_CSV'] == 'TRUE':
        old_df = pd.read_csv(config['PATH']['old_df']).set_index('Datetime')
    elif config['PARAMS']['read_feather'] == 'TRUE':
        old_df = pd.read_feather(config['PATH']['old_df'])
        try:
            old_df.set_index('Datetime', inplace=True)
        except ValueError:
            print('Datetime index already set!')
    elif config['PARAMS']['read_parquet'] == 'TRUE':
        old_df = pd.read_parquet(config['PATH']['old_df'])
        try:
            old_df.set_index('Datetime', inplace=True)
        except ValueError:
            print('Datetime index already set!')
new_data_filename = config['PATH']['new_data_json']
new_data = read_data(new_data_filename)
verify_data_integrity(new_data)

new_df = get_1min_df(new_data)
df_1min_updated = pd.concat([old_df, new_df], axis=0)

df_1min_updated.index = pd.to_datetime(df_1min_updated.index)
df_1min_updated.sort_index(inplace=True)
df_1min_updated.drop_duplicates(inplace=True)

print(df_1min_updated)
if config['PARAMS']['save_df'] == 'TRUE':
    new_data_date = re.search(r'(\d+-\d+-\d+)', new_data_filename[-30:]).group(0)
    df_1min_updated.to_csv(config['PATH']['save_df_path'] + config['PARAMS']['product'] + '_1min_historical-data_' + new_data_date + '.csv')
