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
config.read('/home/melgazar9/Trading/TD/Scripts/Trading-Scripts/CL/scripts/CL_10min_concat_new_data.ini')

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


def get_5min_df(raw_data):

    cols = {'Key':'/' + config['PARAMS']['product'], '0':'Datetime','1':'5minOpen','2':'5minHigh','3':'5minLow','4':'5minClose','5':'5minVolume'}

    data = pd.io.json.json_normalize(raw_data)['snapshot'].dropna()
    for lst1 in data:
        for lst2 in lst1:
            for lst3 in lst2['content']:
                df = pd.DataFrame(lst3['3'])

    df = df.rename(columns=cols)
    df['5minRange'] = df['5minHigh'] - df['5minLow']
    df['5minMove'] = df['5minClose'] - df['5minOpen']
    df['5minLowMove'] = df['5minLow'] - df['5minOpen']
    df['5minHighMove'] = df['5minHigh'] - df['5minOpen']
    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index, unit='ms')
    df = df.sort_index()

    return df


if config['PARAMS']['old_data_json_ON'] == 'TRUE':
    old_data = config['PATH']['old_data_json']
    raw_data_old = read_data(old_data)
    verify_data_integrity(raw_data_old)
    df_5min_old = get_5min_df(raw_data_old)
    df_5min_old = df_5min_old.sort_index()

if config['PARAMS']['old_data_json_ON'] == 'FALSE':
    old_df = pd.read_csv(config['PATH']['old_df']).set_index('Datetime')

new_data_filename = config['PATH']['new_data_json']
new_data = read_data(new_data_filename)
verify_data_integrity(new_data)

new_df = get_5min_df(new_data)
df_5min_updated = pd.concat([old_df, new_df], axis=0)

df_5min_updated.index = pd.to_datetime(df_5min_updated.index)
df_5min_updated.sort_index(inplace=True)

print(df_5min_updated)
if config['PARAMS']['save_df'] == 'TRUE':
    new_data_date = re.search(r'(\d+-\d+-\d+)', new_data_filename).group(0)
    df_5min_updated.to_csv(config['PATH']['save_df_path'] + config['PARAMS']['product'] + '_5min_historical-data_' + new_data_date + '.csv')
