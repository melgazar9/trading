# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 22:14:31 2021

@author: Matt
"""

def calc_dummy_pnl(df, prediction_colname, actual_open_colname, actual_close_colname, pnl_colname = 'PnL', 
                   class_dict = {'sells':[0, 1], 'no_trade' : 2, 'buys' : [3, 4]}):
    
    df = df.copy()
    df[pnl_colname] = 0
    df.loc[df[prediction_colname].isin(class_dict['buys']), pnl_colname] = df[actual_close_colname] - df[actual_open_colname]
    df.loc[df[prediction_colname].isin(class_dict['sells']), pnl_colname] = df[actual_open_colname] - df[actual_close_colname]
    return df


def calc_sharpe(df, prediction_colname, no_trade_class = 2, pnl_colname = 'PnL'):
        sharpe = np.sum(df[df[prediction_colname] != no_trade_class][pnl_colname] / (np.sqrt(len(df[df[prediction_colname] != no_trade_class]) * df[df[prediction_colname] != no_trade_class][pnl_colname].std())))
        return sharpe