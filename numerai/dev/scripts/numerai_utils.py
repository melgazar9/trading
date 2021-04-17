from functools import reduce
import pandas as pd
import numpy as np
import simplejson
import yfinance
import datetime



def download_yfinance_data(tickers,
                           intervals_to_download=['1d', '1h'],
                           num_workers=1,
                           join_method='outer',
                           max_intraday_lookback_days=363,
                           n_chunks=600,
                           yfinance_params={}):
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
        if 'left' then all dates from the left table will be present
        if 'right' then all dates from the right table will be present
        if 'inner' then all dates must match for each ticker
    **yfinance_params : dict - passed to yfinance.dowload(yfinance_params)
        set threads = True for faster performance, but tickers will fail, scipt may hang
        set threads = False for slower performance, but more tickers will succeed

    NOTE: passing some intervals return unreliable stock data (e.g. '3mo' returns many NA data points when they should not be NA)
    """

    intraday_lookback_days = datetime.datetime.today().date() - datetime.timedelta(days=max_intraday_lookback_days)
    start_date = yfinance_params['start']

    if num_workers == 1:

        list_of_dfs = []

        for i in intervals_to_download:

            yfinance_params['interval'] = i

            if (i.endswith('m') or i.endswith('h')) and (pd.to_datetime(yfinance_params['start']) < intraday_lookback_days):
                yfinance_params['start'] = str(intraday_lookback_days)

            if yfinance_params['threads'] == True:

                df_i = yfinance.download(' '.join(tickers), **yfinance_params)\
                               .stack()\
                               .rename_axis(index=['date', 'ticker'])\
                               .add_suffix('_' + i)\
                               .reset_index()
            else:

                ticker_chunks = [' '.join(tickers[i:i+n_chunks]) for i in range(0, len(tickers), n_chunks)]
                chunk_dfs_lst = []

                for chunk in ticker_chunks:
                    try:
                        df_tmp = yfinance.download(chunk, **yfinance_params)\
                                         .stack()\
                                         .rename_axis(index=['date', 'ticker'])\
                                         .add_suffix('_' + i)\
                                         .reset_index()
                        chunk_dfs_lst.append(df_tmp)
                    except simplejson.errors.JSONDecodeError:
                        pass

                df_i = pd.concat(chunk_dfs_lst)
                del chunk_dfs_lst
                yfinance_params['start'] = start_date

            if i.endswith('m') or i.endswith('h'):
                # Go long-to-wide on the min/hour bars
                df_i = df_i.pivot_table(index=[df_i['date'].dt.date, 'ticker'], columns=[df_i['date'].dt.hour], aggfunc='first',
                                        values=[i for i in df_i.columns if not i in ['date', 'ticker']])
                df_i.columns = list(pd.Index([str(e[0]).lower() + '_' + str(e[1]).lower() for e in df_i.columns.tolist()]).str.replace(' ', '_'))
                df_i.reset_index(inplace=True)
                df_i['date'] = pd.to_datetime(df_i['date']) # pivot table sets the index, and reset_index changes 'date' to an object

            df_i.columns = [col.replace(' ', '_').lower() for col in df_i.columns]

            list_of_dfs.append(df_i)

        df_yahoo = reduce(lambda x, y: pd.merge(x, y, how=join_method, on=['date', 'ticker']), list_of_dfs)
        date_plus_ticker = df_yahoo['date'].astype(str) + df_yahoo['ticker'].astype(str) # one last quality check to ensure date + ticker is unique

        assert len(date_plus_ticker) == len(set(date_plus_ticker)), i + ' date + ticker is not unique in df_yahoo!'

    else:
        print('Multi-threading is not fully implemented yet!! Set num_workers=1.')
        print(' *** Pulling yfinance data using', num_workers, 'threads! ***')
        list_of_dfs = []
        chunk_len = len(tickers) // num_workers
        ticker_chunks = [' '.join(tickers[i:i+chunk_len]) for i in range(0, len(tickers), chunk_len)]

        for i in intervals_to_download:

            yfinance_params['interval'] = i

            if (i.endswith('m') or i.endswith('h')) and (pd.to_datetime(yfinance_params['start']) < intraday_lookback_days):
                yfinance_params['start'] = str(intraday_lookback_days)

            if yfinance_params['threads'] == True:

                print('Parallelizing using both dask and yfinance threads - some tickers may return a JSONDecodeError. If so, set threads to False in yfinance_params')

                delayed_list = [delayed(yfinance.download)(' '.join(chunk), **yfinance_params)\
                                                              .stack()\
                                                              .rename_axis(index=['date', 'ticker'])\
                                                              .add_suffix('_' + i)\
                                                              .reset_index()\
                                for chunk in ticker_chunks]
            else:

                print('Running safer-parallel')

                def safe_yfinance_pull(ticker_chunks, yfinance_params):

                    chunk_dfs_lst = []

                    for chunk in ticker_chunks:
                        try:
                            df_tmp = yfinance.download(chunk, **yfinance_params)\
                                             .stack()\
                                             .rename_axis(index=['date', 'ticker'])\
                                             .add_suffix('_' + i)\
                                             .reset_index()
                            chunk_dfs_lst.append(df_tmp)
                        except simplejson.errors.JSONDecodeError:
                            pass

                    df_out = pd.concat(chunk_dfs_lst)
                    return df_out

                delayed_list = [delayed(safe_yfinance_pull)(chunk, yfinance_params) for chunk in ticker_chunks]

            tuple_of_dfs = dask.compute(*delayed_list, num_workers=num_workers)

            df_i = reduce(lambda x, y: pd.merge(x, y, how=join_method, on=['date', 'ticker']), tuple_of_dfs)
            del tuple_of_dfs
            yfinance_params['start'] = start_date

            if i.endswith('m') or i.endswith('h'):
                # Go long-to-wide on the min/hour bars
                df_i = df_i.pivot_table(index=[df_i['date'].dt.date, 'ticker'], columns=[df_i['date'].dt.hour], aggfunc='first',
                                        values=[i for i in df_i.columns if not i in ['date', 'ticker']])
                df_i.columns = list(pd.Index([str(e[0]).lower() + '_' + str(e[1]).lower() for e in df_i.columns.tolist()]).str.replace(' ', '_'))
                df_i.reset_index(inplace=True)
                df_i['date'] = pd.to_datetime(df_i['date']) # pivot table sets the index, and reset_index changes 'date' to an object

            df_i.columns = [col.replace(' ', '_').lower() for col in df_i.columns]

            list_of_dfs.append(df_i)

        df_yahoo = reduce(lambda x, y: pd.merge(x, y, how=join_method, on=['date', 'ticker']), list_of_dfs)
        date_plus_ticker = df_yahoo['date'].astype(str) + df_yahoo['ticker'].astype(str) # one last quality check to ensure date + ticker is unique

    return df_yahoo

def convert_df_dtypes(df,
                     exclude_cols=[],
                     new_float_dtype='float32',
                     new_int_dtype='int8',
                     new_obj_dtype='category',
                     float_qualifiers='auto',
                     int_qualifiers='auto',
                     obj_qualifiers='auto'):

    """
    Description
    ___________
    This function is meant to be used to reduce the memory of a pandas df - e.g. changing the dtype from float64 to float32

    Parameters
    _________
    df: pandas df
    exclude_cols: list of cols to exclude
    new_float_type: the new dtype of float columns
    new_int_type: the new dtype of int columns
    new_obj_type: the new dtype of obj columns
    float_qualifiers: if 'auto', then detect the float columns using np.floating, else a list of float dtypes must be provided.
        Only the columns with the exact dtypes provided in float_qualifiers will be changes
    int_qualifiers: same as float_qualifiers but for ints
    obj_qualifiers: same as float_qualifiers but for objects

    """

    dtype_df = pd.DataFrame(df.dtypes.reset_index()).rename(columns={'index':'col', 0:'dtype'})
    unique_dtypes = dtype_df['dtype'].unique()

    if float_qualifiers == 'auto':
        float_cols = [i for i in df.select_dtypes([np.floating]).columns if not i in exclude_cols]
        df[float_cols] = df[float_cols].astype(new_float_dtype)
    if int_qualifiers == 'auto':
        int_cols = [i for i in df.select_dtypes([np.int_]).columns if not i in exclude_cols]
        df[int_cols] = df[int_cols].astype(new_int_dtype)
    if obj_qualifiers == 'auto':
        obj_cols = [i for i in df.select_dtypes([np.object_]).columns if not i in exclude_cols]
        df[obj_cols] = df[obj_cols].astype(new_obj_dtype)

    return df

def create_naive_features_single_symbol(df,
                                        symbol='',
                                        symbol_sep='',
                                        open_col='open_1d',
                                        high_col='high_1d',
                                        low_col='low_1d',
                                        close_col='adj_close_1d',
                                        volume_col='volume_1d',
                                        new_col_suffix='_1d',
                                        copy=True):
    """
    Parameters
    __________

    df: Pandas-like / dask dataframe
        For the stacked yfinance data used for numerai, the syntax is <groupby('bloomberg_ticker').apply(func)>
    """

    if copy: df = df.copy()

    ### custom features ###

    df.loc[:, 'move' + new_col_suffix] = df[close_col] - df[open_col]
    df.loc[:, 'move_pct' + new_col_suffix] = df['move' + new_col_suffix] / df[open_col]
    df.loc[:, 'move_pct_change' + new_col_suffix] = df['move' + new_col_suffix].pct_change()
    df.loc[:, 'open_minus_prev_close' + new_col_suffix] = df[open_col] - df[close_col].shift()
    df.loc[:, 'prev_close_pct_chg' + new_col_suffix] = df['move' + new_col_suffix] / df[close_col].shift()

    df.loc[:, 'high_move' + new_col_suffix] = df[high_col] - df[open_col]
    df.loc[:, 'high_move_pct' + new_col_suffix] = df['high_move' + new_col_suffix] / df[open_col]
    df.loc[:, 'high_move_pct_change' + new_col_suffix] = df['high_move' + new_col_suffix].pct_change()

    df.loc[:, 'low_move' + new_col_suffix] = df[low_col] - df[open_col]
    df.loc[:, 'low_move_pct' + new_col_suffix] = df['low_move' + new_col_suffix] / df[open_col]
    df.loc[:, 'low_move_pct_change' + new_col_suffix] = df['low_move' + new_col_suffix].pct_change()

    df.loc[:, 'close_minus_low' + new_col_suffix] = df[close_col] - df[low_col]
    df.loc[:, 'high_minus_close' + new_col_suffix] = df[high_col] - df[close_col]

    df.loc[:, 'prev_close_minus_low_minus' + new_col_suffix] = df[close_col].shift() - df[low_col]
    df.loc[:, 'high_minus_prev_close' + new_col_suffix] = df[high_col] - df[close_col].shift()

    ### diffs ###

    df.loc[:, 'open_diff' + new_col_suffix] = df[open_col].diff()
    df.loc[:, 'high_diff' + new_col_suffix] = df[high_col].diff()
    df.loc[:, 'low_diff' + new_col_suffix] = df[low_col].diff()
    df.loc[:, 'close_diff' + new_col_suffix] = df[close_col].diff()
    df.loc[:, 'volume_diff' + new_col_suffix] = df[volume_col].diff()

    ### pct_change ###

    df.loc[:, 'open_pct_change' + new_col_suffix] = df[open_col].pct_change()
    df.loc[:, 'high_pct_change' + new_col_suffix] = df[high_col].pct_change()
    df.loc[:, 'low_pct_change' + new_col_suffix] = df[low_col].pct_change()
    df.loc[:, 'close_pct_change' + new_col_suffix] = df[close_col].pct_change()
    df.loc[:, 'volume_pct_change' + new_col_suffix] = df[volume_col].pct_change()

    ### pct_change of diff (e.g. second derivative of the diff)

    df.loc[:, 'open_diff_pct_change' + new_col_suffix] = df['open_diff' + new_col_suffix].pct_change()
    df.loc[:, 'high_diff_pct_change' + new_col_suffix] = df['high_diff' + new_col_suffix].pct_change()
    df.loc[:, 'low_diff_pct_change' + new_col_suffix] = df['low_diff' + new_col_suffix].pct_change()
    df.loc[:, 'close_diff_pct_change' + new_col_suffix] = df['close_diff' + new_col_suffix].pct_change()
    df.loc[:, 'volume_diff_pct_change' + new_col_suffix] = df['volume_diff' + new_col_suffix].pct_change()

    ### range features ###
    df.loc[:, 'range' + new_col_suffix] = df[high_col] - df[low_col]
    df.loc[:, 'range_pct_change' + new_col_suffix] = df['range' + new_col_suffix].pct_change()

    return df



class CreateTargets():

    def __init__(self, df, copy=True):
        """
        Parameters
        __________

        df : pandas df
        copy : Boolean whether to make a copy of the df before applying transformations

        Note: to compute the target based on pct, pass the pct column names into the individual functions
        """

        self.df = df
        self.copy = copy

        if self.copy: self.df = self.df.copy()

    def create_targets_HL5(self,
                           strong_buy,
                           med_buy,
                           med_sell,
                           strong_sell,
                           threshold,
                           stop,
                           move_col='move_pct',
                           lm_col='low_move_pct',
                           hm_col='high_move_pct',
                           target_suffix='target_HL5'):

        # hm stands for high move, lm stands for low move
        # Strong Buy
        self.df.loc[(self.df[hm_col] >= strong_buy) &
                    (self.df[lm_col] >= (-1) * stop),
                    target_suffix] = 4

        # Strong Sell
        self.df.loc[(self.df[lm_col] <= (-1) * strong_sell) &
                    (self.df[hm_col] <= stop) &
                    (self.df[target_suffix] != 4),
                    target_suffix] = 0

        # Medium Buy
        self.df.loc[(self.df[hm_col] >= med_buy) &
                    (self.df[lm_col] >= (-1) * stop) &
                    (self.df[target_suffix] != 4) &
                    (self.df[target_suffix] != 0),
                    target_suffix] = 3

        # Medium Sell
        self.df.loc[(self.df[lm_col] <= (-1) * med_sell) &
                    (self.df[hm_col] <= stop) &
                    (self.df[target_suffix] != 4) &
                    (self.df[target_suffix] != 0) &
                    (self.df[target_suffix] != 3),
                    target_suffix] = 1

        # No Trade
        self.df.loc[(self.df[target_suffix] != 0) &
                    (self.df[target_suffix] != 1) &
                    (self.df[target_suffix] != 3) &
                    (self.df[target_suffix] != 4),
                    target_suffix] = 2

        return self.df

    def create_targets_HL3(self,
                           buy,
                           sell,
                           threshold,
                           stop,
                           move_col='move_pct',
                           lm_col='low_move_pct',
                           hm_col='high_move_pct',
                           target_suffix='target_HL3'):

        # hm stands for high move, lm stands for low move
        # Buy
        self.df.loc[(self.df[hm_col] >= buy) &
                    (self.df[lm_col] >= (-1) * stop),
                    target_suffix] = 2

        # Sell
        self.df.loc[(self.df[lm_col] <= (-1) * sell) &
                    (self.df[hm_col] <= stop) &
                    (self.df[target_suffix] != 2),
                    target_suffix] = 0

        # No Trade
        self.df.loc[(self.df[target_suffix] != 0) &
                    (self.df[target_suffix] != 2),
                    target_suffix] = 1

        return self.df



# def create_lagging_features(df, lagging_map, groupby_cols=None, new_col_prefix='prev', copy=True):
#     """
#
#     Parameters
#     __________
#
#     df : pandas df
#     groupby_cols : str or list of cols to groupby before creating lagging transformation cols
#     lagging_map : dict with keys as colnames and values as a list of periods for computing lagging features
#     periods : periods to look back
#
#     """
#
#     if copy: df = df.copy()
#
#     unique_lagging_values = list(sorted({k for v in lagging_map.values() for k in v}))
#
#     if groupby_cols is None or len(groupby_cols) == 0:
#         for period in unique_lagging_values:
#             new_col_prefix_tmp = new_col_prefix + str(period) + '_'
#             cols_to_lag = [k for k, v in lagging_map.items() if period in v]
#             df[[new_col_prefix_tmp + c for c in cols_to_lag]] = df[cols_to_lag].transform(lambda s: s.shift(periods=period))
#
#     else:
#         for period in unique_lagging_values:
#             new_col_prefix_tmp = new_col_prefix + str(period) + '_'
#             cols_to_lag = [k for k, v in lagging_map.items() if period in v]
#
#             df[[new_col_prefix_tmp + c for c in cols_to_lag]] = df.groupby(groupby_cols)[cols_to_lag]\
#                                                                   .transform(lambda s: s.shift(periods=period))
#     return df

def create_lagging_features(df, lagging_map, groupby_cols=None, new_col_prefix='prev', copy=True):
    """

    Parameters
    __________

    df : pandas df
    groupby_cols : str or list of cols to groupby before creating lagging transformation cols
    lagging_map : dict with keys as colnames and values as a list of periods for computing lagging features
    periods : periods to look back

    """

    if copy: df = df.copy()

    unique_lagging_values = list(sorted({k for v in lagging_map.values() for k in v}))

    for period in unique_lagging_values:
        new_col_prefix_tmp = new_col_prefix + str(period) + '_'
        cols_to_lag = [k for k, v in lagging_map.items() if period in v]
        # df[[new_col_prefix_tmp + c for c in cols_to_lag]] = df[cols_to_lag].transform(lambda df: df.shift(periods=period))
        df[[new_col_prefix_tmp + c for c in cols_to_lag]] = df[cols_to_lag].shift(periods=period)

    return df

# def create_rolling_features(df,
#                             rolling_fn='mean',
#                             ewm_fn='mean',
#                             rolling_params={},
#                             ewm_params={},
#                             rolling_cols='all_numeric',
#                             ewm_cols='all_numeric',
#                             join_method='outer',
#                             groupby_cols=None,
#                             create_diff_cols=True,
#                             copy=True):
#     """
#
#     Parameters
#     __________
#     df : pandas df
#
#     rolling_fn : str called from df.rolling().rolling_fn (e.g. df.rolling.mean() is called with getattr)
#     ewm_fn : str called from df.ewm().ewm_fn (e.g. df.ewm.mean() is called with getattr)
#
#     rolling_params : dict params passed to df.rolling()
#     ewm_params : dict params passed to df.ewm()
#
#     rolling_cols : cols to apply rolling_fn
#     ewm_cols : cols to apply ewm_fn
#
#     join_method : str 'inner', 'outer', 'left', or 'right' - how to join the dfs
#     groupby_cols : list or str cols to group by before applying rolling transformations
#         example: pass groupby_cols to the stacked ticker numerai dataset, but not a wide df
#
#     copy : bool whether or not to make a copy of the df
#
#     """
#
#     if copy: df = df.copy()
#
#     if isinstance(rolling_cols, str) and rolling_cols.lower() == 'all_numeric':
#         rolling_cols = list(df.select_dtypes(include=np.number).columns)
#
#     if isinstance(rolling_cols, str) and ewm_cols.lower() == 'all_numeric':
#         ewm_cols = list(df.select_dtypes(include=np.number).columns)
#
#     if groupby_cols is None or len(groupby_cols) == 0:
#
#         # rolling
#         if rolling_fn is not None and len(rolling_cols) > 0:
#             new_rolling_cols = [i + '_rolling_' + rolling_fn for i in rolling_cols]
#             df[new_rolling_cols] = getattr(df[rolling_cols].rolling(**rolling_params), rolling_fn)()
#
#         # ewm
#         if ewm_fn is not None and len(ewm_cols) > 0:
#             new_ewm_cols = [i + '_ewm_' + ewm_fn for i in ewm_cols]
#             df[new_ewm_cols] = getattr(df[ewm_cols].ewm(**ewm_params), ewm_fn)()
#
#     else:
#
#         if isinstance(groupby_cols, str):
#             groupby_cols = [groupby_cols]
#         else:
#             raise ('Input param groupby_cols is not a list, string, or None!')
#
#         assert len(groupby_cols) == len(set(groupby_cols)), 'There are duplicates in groupby_cols!'
#         rolling_cols_to_select = [i for i in list(set(groupby_cols + rolling_cols)) if i in df.columns] # check index name
#         ewm_cols_to_select = [i for i in list(set(groupby_cols + ewm_cols)) if i in df.columns]
#
#         # rolling
#         if rolling_fn is not None and len(rolling_cols) > 0:
#             new_rolling_cols = [i + '_rolling_' + rolling_fn for i in rolling_cols_to_select if not i in groupby_cols]
#             df[new_rolling_cols] = df[rolling_cols_to_select]\
#                                     .groupby(groupby_cols)\
#                                     .transform(lambda x: getattr(x.rolling(**rolling_params), rolling_fn)())
#
#         # ewm
#         if ewm_fn is not None and len(ewm_cols) > 0:
#             new_ewm_cols = [i + '_ewm_' + ewm_fn for i in ewm_cols_to_select if not i in groupby_cols]
#             df[new_ewm_cols] = df[ewm_cols_to_select]\
#                                 .groupby(groupby_cols)\
#                                 .transform(lambda x: getattr(x.ewm(**ewm_params), ewm_fn)())
#
#     if create_diff_cols:
#         diff_cols = [i for i in df.columns if 'ewm' in i or 'rolling' in i]
#         if groupby_cols is None or len(groupby_cols) == 0:
#             df = pd.concat([df, df[diff_cols].diff().add_suffix('_diff')], axis=1)
#         else:
#             df[[i + '_diff' for i in diff_cols]] = df.groupby(groupby_cols)[diff_cols].transform(lambda col: col.diff())
#
#     return df


def create_rolling_features(df,
                            rolling_fn='mean',
                            ewm_fn='mean',
                            rolling_params={},
                            ewm_params={},
                            rolling_cols='all_numeric',
                            ewm_cols='all_numeric',
                            join_method='outer',
                            groupby_cols=None,
                            create_diff_cols=True,
                            copy=True):
    """

    Parameters
    __________
    df : pandas df

    rolling_fn : str called from df.rolling().rolling_fn (e.g. df.rolling.mean() is called with getattr)
    ewm_fn : str called from df.ewm().ewm_fn (e.g. df.ewm.mean() is called with getattr)

    rolling_params : dict params passed to df.rolling()
    ewm_params : dict params passed to df.ewm()

    rolling_cols : cols to apply rolling_fn
    ewm_cols : cols to apply ewm_fn

    join_method : str 'inner', 'outer', 'left', or 'right' - how to join the dfs
    groupby_cols : list or str cols to group by before applying rolling transformations
        example: pass groupby_cols to the stacked ticker numerai dataset, but not a wide df

    copy : bool whether or not to make a copy of the df

    """

    if copy: df = df.copy()

    if isinstance(rolling_cols, str) and rolling_cols.lower() == 'all_numeric':
        rolling_cols = list(df.select_dtypes(include=np.number).columns)

    if isinstance(rolling_cols, str) and ewm_cols.lower() == 'all_numeric':
        ewm_cols = list(df.select_dtypes(include=np.number).columns)

    # rolling
    if rolling_fn is not None and len(rolling_cols) > 0:
        new_rolling_cols = [i + '_rolling_' + rolling_fn for i in rolling_cols]
        df[new_rolling_cols] = getattr(df[rolling_cols].rolling(**rolling_params), rolling_fn)()

    # ewm
    if ewm_fn is not None and len(ewm_cols) > 0:
        new_ewm_cols = [i + '_ewm_' + ewm_fn for i in ewm_cols]
        df[new_ewm_cols] = getattr(df[ewm_cols].ewm(**ewm_params), ewm_fn)()

    if create_diff_cols:
        diff_cols = [i for i in df.columns if 'ewm' in i or 'rolling' in i]
        df[[i + '_diff' for i in diff_cols]] = df[diff_cols].diff()

    return df



def drop_nas(df, col_contains, exception_cols=[], how=None, copy=True):

    """
    Description:
    ___________
    The goal of this function is to conditionally drop null rows

    Parameters
    __________

    df: pandas df
    col_contains: str or list of strings that select columns if the column contains this string - drop NAs based on these columns and exception_cols
    exception_cols: list or str that will not drop that row if this col is non-null
        example: All values are NA in col_contains, but 'target' is passed to exception_cols, and target is non-null,
                 In this case, since 'target' (e.g. exception_cols) is non-null, that row will not be dropped

    how: str passed to pd.dropna() - default is None - if None, the function will use 'all' if exception_cols are non-empty.
         If exception cols are empty it uses 'any'. This can be overwritten.
    copy: bool - if True make a copy of the df before applying transformations

    Returns
    _______

    pandas df with new subsetted rows
    """

    if copy: df = df.copy()

    if isinstance(col_contains, str):
        col_contains = [col_contains]
    if isinstance(exception_cols, str):
        exception_cols = [exception_cols]

    assert isinstance(col_contains, list), 'col_contains must be a list or str!'
    assert isinstance(exception_cols, list), 'exception_cols must be a list or str!'

    selected_cols = list(set([col for col in df.columns for j in col_contains for k in exception_cols if j in col] + exception_cols))

    if len(exception_cols) and how is None:
        how = 'all'
    elif len(exception_cols) == 0 and how is None:
        how='any'

    df.dropna(how=how, subset=selected_cols, inplace=True)
    return df


def calc_move_iar(df, iar_cols, iar_suffix='_iar', copy=True):

    if copy: df = df.copy()

    assert isinstance(iar_cols, str) or isinstance(iar_cols, list), 'iar_cols must be a str or list!'

    upmove_iar = df[iar_cols].transform(lambda x: x.cumsum().sub(x.cumsum().mask(x >= 0).ffill(), fill_value=0), axis=0).replace(0, np.nan)
    downmove_iar = df[iar_cols].transform(lambda x: x.cumsum().sub(x.cumsum().mask(x <= 0).ffill(), fill_value=0), axis=0).replace(0, np.nan)

    if isinstance(iar_cols, str):
        new_iar_cols = iar_cols + iar_suffix
    else:
        new_iar_cols = [i + iar_suffix for i in iar_cols]

    df[new_iar_cols] = upmove_iar.fillna(downmove_iar).ffill()

    return df



def calc_trend(df, iar_cols, iar_suffix='_iar', trend_suffix='_trend', flat_threshold=0.005, copy=True):

    if copy: df = df.copy()

    new_iar_colname = iar_cols + iar_suffix
    trend_colname = iar_cols + trend_suffix

    df.loc[df[iar_cols] > 0, trend_colname] = 'up'
    df.loc[df[iar_cols] < 0, trend_colname] = 'down'
    df.loc[np.abs(df[iar_cols]) <= flat_threshold, trend_colname] = 'flat'

    return df
