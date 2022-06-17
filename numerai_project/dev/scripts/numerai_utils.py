from functools import reduce
import pandas as pd
import numpy as np
import simplejson
import yfinance
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import warnings
import dask
from dask import delayed
from skimpy import clean_columns
import time


class DataValidator():

    """

    Objective: Validate all outputs from a pandas df
    This class does not specifically return an output.

    Parameters
    ----------

    df: A pandas dataframe

    """

    def __init__(self, df):
        self.df = df

    def _check_duplicate_colnames(self):
        duplicate_colnames = self.df.columns[self.df.columns.duplicated()]

        if len(duplicate_colnames):
            warnings.warn("The df has duplicate column names! \nDuplicate column names: {}" \
                          .format(duplicate_colnames))

        return duplicate_colnames

    def validate_data(self):
        self._check_duplicate_colnames()


def download_yfinance_data(tickers,
                           intervals_to_download=['1d', '1h'],
                           num_workers=1,
                           max_intraday_lookback_days=363,
                           n_chunks=600,
                           tz_localize_location='US/Central',
                           yfinance_params={},
                           yahoo_ticker_colname='yahoo_ticker',
                           verbose=True):
    """
    Parameters
    __________

    See yfinance.download docs for a detailed description of yfinance parameters

    tickers: list of tickers to pass to yfinance.download - it will be parsed to be in the format "AAPL MSFT FB"
    intervals_to_download : list of intervals to download OHLCV data for each stock (e.g. ['1w', '1d', '1h'])
    num_workers: number of threads used to download the data
        so far only 1 thread is implemented
    n_chunks: int number of chunks to pass to yfinance.download()
        1 is the slowest but most reliable because if two are passed and one fails, then both tickers are not returned
    tz_localize_location: timezone location to set the datetime
    **yfinance_params: dict - passed to yfinance.dowload(yfinance_params)
        set threads = True for faster performance, but tickers will fail, scipt may hang
        set threads = False for slower performance, but more tickers will succeed

    NOTE: passing some intervals return unreliable stock data (e.g. '3mo' returns many NA data points when they should not be NA)
    """

    failed_ticker_downloads = []
    failed_datetime_features = []

    if not 'start' in yfinance_params.keys():
        if verbose: print('*** yfinance params start set to 2005-01-01! ***')
        yfinance_params['start'] = '2005-01-01'
    if not 'threads' in yfinance_params.keys() or yfinance_params['threads'] != True:
        if verbose: print('*** yfinance params threads set to False! ***')
        yfinance_params['threads'] = False
    if not verbose:
        yfinance_params['progress'] = False

    intraday_lookback_days = datetime.datetime.today().date() - datetime.timedelta(days=max_intraday_lookback_days)
    start_date = yfinance_params['start']
    assert pd.Timestamp(start_date) <= datetime.datetime.today(), 'Start date cannot be after the current date!'

    if num_workers == 1:

        dict_of_dfs = {}
        for i in intervals_to_download:

            yfinance_params['interval'] = i

            if (i.endswith('m') or i.endswith('h')) and (pd.Timestamp(yfinance_params['start']) < pd.Timestamp(intraday_lookback_days)):
                yfinance_params['start'] = str(intraday_lookback_days)

            if yfinance_params['threads'] == True:

                df_i = yfinance.download(' '.join(tickers), **yfinance_params)\
                               .stack()\
                               .rename_axis(index=['date', yahoo_ticker_colname])\
                               .add_suffix('_' + i)\
                               .reset_index()
            else:
                ticker_chunks = [' '.join(tickers[i:i+n_chunks]) for i in range(0, len(tickers), n_chunks)]
                chunk_dfs_lst = []
                column_order = ['date', yahoo_ticker_colname, 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

                for chunk in ticker_chunks:
                    if verbose: print(f"Running chunk {chunk}")
                    try:
                        if n_chunks == 1 or len(chunk.split(' ')) == 1:
                            try:
                                df_tmp = yfinance.download(chunk, **yfinance_params)\
                                                 .rename_axis(index='date')\
                                                 .reset_index()
                                if len(df_tmp) == 0:
                                    continue
                            except:
                                if verbose: print(f"failed download for tickers: {chunk}")
                                failed_ticker_downloads.append(chunk)

                            df_tmp[yahoo_ticker_colname] = chunk
                            df_tmp = df_tmp[column_order]
                            df_tmp.columns = df_tmp.columns.map(lambda c: c + '_' + i if c != 'date' and c != yahoo_ticker_colname else c)

                        else:
                            # should be the order of column_order
                            df_tmp = yfinance.download(chunk, **yfinance_params) \
                                .stack() \
                                .add_suffix('_' + i) \
                                .rename_axis(index=['date', yahoo_ticker_colname]) \
                                .reset_index()

                            if len(df_tmp) == 0:
                                continue

                            new_column_order = ['date', yahoo_ticker_colname] + [x + '_' + i for x in column_order if not x in ['date', yahoo_ticker_colname]]
                            df_tmp = df_tmp[new_column_order]

                        chunk_dfs_lst.append(df_tmp)

                    except simplejson.errors.JSONDecodeError:
                        pass

                df_i = pd.concat(chunk_dfs_lst)
                del chunk_dfs_lst
                yfinance_params['start'] = start_date

            # set UTC to True because we're pulling data from all over the world, and pandas cannot convert Tz-aware datetimes unless UTC is true
            if i == '1d' and type(tz_localize_location) == str:
                df_i['date_localized'] = pd.to_datetime(df_i['date'], utc=True).dt.tz_convert(tz_localize_location)
                try:
                    create_datetime_features(df_i, 'date', include_hour=False, make_copy=False)
                except AttributeError:
                    failed_datetime_features.append(chunk)

            elif i == '1h':
                # date will be dtype object after concat because when pulling hour data from yfinance the date
                # is tz_localized to the location of that specific ticker
                df_i['date_localized'] = pd.to_datetime(df_i['date'], utc=True).dt.tz_convert(tz_localize_location)
                try:
                    df_i.loc[:, 'hour'] = df_i['date'].dt.hour
                except:
                    try:
                        df_i.loc[:, 'hour'] = pd.to_datetime(df_i['date'].dt.hour)
                    except AttributeError:
                        df_i.loc[:, 'hour'] = np.nan
                df_i.reset_index(inplace=True)

            df_i = clean_columns(df_i)
            dict_of_dfs[i] = df_i

            ### print errors ###

            if verbose:
                if len(failed_ticker_downloads) > 0:
                    if n_chunks > 1:
                        failed_ticker_downloads = list(itertools.chain(*failed_ticker_downloads))

                if len(failed_datetime_features) > 0:
                    if n_chunks > 1:
                        failed_datetime_features = list(itertools.chain(*failed_datetime_features))

                print(f"\nFailed ticker downloads:\n{failed_ticker_downloads}")
                print(f"\nFailed to create datetime features for:\n{failed_datetime_features}")

    else:
        raise ValueError("Multi-threading not supported yet.")

    return dict_of_dfs


def convert_df_dtypes(df,
                     exclude_cols=None,
                     new_float_dtype='float32',
                     new_int_dtype='Int64',
                     new_obj_dtype='category',
                     float_qualifiers='auto',
                     int_qualifiers='auto',
                     obj_qualifiers='auto',
                     verbose=True,
                     make_copy=False):

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

    if make_copy: df = df.copy()
    exclude_cols = [] if exclude_cols is None else exclude_cols

    dtype_df = pd.DataFrame(df.dtypes.reset_index()).rename(columns={'index': 'col', 0: 'dtype'})

    if verbose:
        unique_dtypes = dtype_df['dtype'].unique()
        print('\nunique dtypes: {}'.format(unique_dtypes), '\n')

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

class CalcMoves(DataValidator):

    def __init__(self, copy=False):
        self.copy = copy

    def calc_basic_moves(self,
                         df,
                         open_col='open',
                         high_col='high',
                         low_col='low',
                         close_col='close',
                         volume_col='volume',
                         suffix='',
                         index_cols=None,
                         reset_index=True):

        """
            Description
            ___________
            This function is meant to be used to reduce the memory of a pandas df - e.g. changing the dtype from float64 to float32

            Parameters
            _________
            df: pandas df
            open_col: str
            high_col: str
            low_col: str
            close_col: str
            volume_col: str
            index_cols: None, list, or tuple of cols that are the indices of the df
            reset_index: Bool: if true then reset the index after setting index before joining, else return the index_cols as the index in the df

            """

        if self.copy: df = df.copy()

        if index_cols is None:
            preserve_cols = [open_col, high_col, low_col, close_col, volume_col]
        else:
            preserve_cols = [open_col, high_col, low_col, close_col, volume_col]
            df.set_index(list(index_cols), inplace=True)

        tmp_frame = df[preserve_cols].copy()

        tmp_frame.loc[:, 'move' + suffix] = tmp_frame[close_col] - tmp_frame[open_col]
        tmp_frame.loc[:, 'move_pct' + suffix] = tmp_frame['move' + suffix] / tmp_frame[open_col]
        tmp_frame.loc[:, 'move_pct_change' + suffix] = tmp_frame['move' + suffix].pct_change()

        tmp_frame.loc[:, 'pct_chg' + suffix] = tmp_frame['move' + suffix] / tmp_frame[close_col].shift()

        tmp_frame.loc[:, 'range' + suffix] = tmp_frame[high_col] - tmp_frame[low_col]
        tmp_frame.loc[:, 'range_pct_change' + suffix] = tmp_frame['range' + suffix].pct_change()

        tmp_frame.loc[:, 'high_move' + suffix] = tmp_frame[high_col] - tmp_frame[open_col]
        tmp_frame.loc[:, 'high_move_pct' + suffix] = tmp_frame['high_move' + suffix] / tmp_frame[open_col]
        tmp_frame.loc[:, 'high_move_pct_change' + suffix] = tmp_frame['high_move' + suffix].pct_change()

        tmp_frame.loc[:, 'low_move' + suffix] = tmp_frame[low_col] - tmp_frame[open_col]
        tmp_frame.loc[:, 'low_move_pct' + suffix] = tmp_frame['low_move' + suffix] / tmp_frame[open_col]
        tmp_frame.loc[:, 'low_move_pct_change' + suffix] = tmp_frame['low_move' + suffix].pct_change()
        tmp_frame.loc[:, 'volume_pct_change' + suffix] = tmp_frame[volume_col].pct_change()

        tmp_frame.loc[:, 'low_minus_close' + suffix] = tmp_frame[low_col] - tmp_frame[close_col]
        tmp_frame.loc[:, 'high_minus_close' + suffix] = tmp_frame[high_col] - tmp_frame[close_col]
        tmp_frame.loc[:, 'low_minus_prev_close' + suffix] = tmp_frame[low_col] - tmp_frame[close_col].shift()
        tmp_frame.loc[:, 'high_minus_prev_close' + suffix] = tmp_frame[high_col] - tmp_frame[close_col].shift()

        if index_cols is None:
            df = pd.concat([df, tmp_frame[[i for i in tmp_frame.columns if i not in preserve_cols]]], axis=1)
        else:
            df = pd.merge(df, tmp_frame[[i for i in tmp_frame.columns if i not in preserve_cols]], left_index=True, right_index=True)

        del tmp_frame
        if reset_index:
            df.reset_index(inplace=True)
            if 'index' in df.columns:
                df.drop('index', axis=1, inplace=True)

        super().__init__(df) # initialize DataValidator with df as the input
        self.validate_data()
        return df


    def compute_multi_basic_moves(self, df, basic_move_params, num_workers=1, dask_join_cols=None, reset_index=True):

        """

        Parameters
        ----------
        df: pandas dataframe: dataframe that consists of all columns in the values of the basic_move_params dictionary
        num_workers: int: number of threads to use - backend is dask
        basic_move_params: dict: a specified dictionary with each value of the dictionary being a dictionary of parameters to pass to calc_basic_moves (called in a loop or in parallel)
            Example of basic_move_params passed:
            {
            'loop1': {'open_col': 'open_1d', 'high_col': 'high_1d', 'low_col': 'low_1d', 'close_col': 'close_1d', 'volume_col': 'volume_1d', 'suffix': '_1d'},
            'loop2': {'open_col': 'open_1h', 'high_col': 'high_1h', 'low_col': 'low_1h', 'close_col': 'close_1h', 'volume_col': 'volume_1h', 'suffix': '_1h'}
             }
        dask_join_cols: list: a list of cols to combine the parallelized computed dataframes

        Returns
        -------
        pandas dataframe with the new columns

        """

        if dask_join_cols is None and num_workers != 1:
            raise('You must specify dask_join_cols when num_workers != 1')

        if num_workers == 1:
            for p in basic_move_params.keys():
                df = self.calc_basic_moves(df, **basic_move_params[p])
        else:
            delayed_list = [delayed(self.calc_basic_moves(df, **basic_move_params[p])) for p in basic_move_params.keys()]
            tuple_of_dfs = dask.compute(*delayed_list, num_workers=num_workers)
            list_of_dfs = [tuple_of_dfs[i] for i in range(len(tuple_of_dfs))]
            df.set_index(dask_join_cols, inplace=True)

            for frame in list_of_dfs:
                frame.set_index(dask_join_cols, inplace=True)

            df = reduce(lambda x, y: pd.merge(x, y[[i for i in y.columns if i not in x.columns]], how='outer', left_index=True, right_index=True), list_of_dfs)

            if reset_index:
                df.reset_index(inplace=True)
                if 'index' in df.columns:
                    df.drop('index', axis=1, inplace=True)

        super().__init__(df)  # initialize DataValidator with df as the input
        self.validate_data()
        return df



def calc_diffs(df, diff_cols, diff_suffix='_diff', index_cols=None, reset_index=True, copy=False):

    """
    Description: Calc diffs from previous value of a pandas dataframe

    Parameters
    ----------
    df: pandas df
    diff_cols: list of cols to take the diffs of
    diff_suffix: str suffix to add the the new diff colname
    index_cols: if None then just append the dataframe with the new diff cols.
        If not none then it is efficient to pass index_cols when running in parallel so there are not copies of large dfs being made
    reset_index: if False then return the df with index_cols as the new index, else reset the index
    copy: make a copy of the df before applying logic

    Returns
    -------
    pandas df
    """

    if copy: df = df.copy()
    if index_cols is None:
        df[[i + diff_suffix for i in diff_cols]] = df[diff_cols].diff()
    else:
        assert len(np.intersect1d(list(index_cols), diff_cols)) == 0, 'index_cols cannot overlap with diff_cols'
        diff_df = df[list(index_cols) + list(diff_cols)]
        diff_df.set_index(list(index_cols), inplace=True)
        df.set_index(list(index_cols), inplace=True)
        diff_df = diff_df[diff_cols].diff().add_suffix(diff_suffix)
        df = pd.merge(df, diff_df, left_index=True, right_index=True)

        if reset_index:
            df.reset_index(inplace=True)
            if 'index' in df.columns:
                df.drop('index', axis=1, inplace=True)

    DataValidator(df).validate_data()
    return df


def calc_pct_changes(df, pct_change_cols, pct_change_suffix='_pct_change', epsilon=0.00000000001, index_cols=None, reset_index=True, copy=False):
    """
    Description: Calc diffs from previous value of a pandas dataframe

    Parameters
    ----------
    df: pandas df
    pct_change_cols: list of cols to take the pct changes of
    pct_change_suffix: str suffix to add the the new pct_change colname
    epsilon: small floating point number in case there is a zero division error
    index_cols: if None then just append the dataframe with the new diff cols.
        If not none then it is efficient to pass index_cols when running in parallel so there are not copies of large dfs being made
    reset_index: if False then return the df with index_cols as the new index, else reset the index
    copy: make a copy of the df before applying logic

    Returns
    -------
    pandas df

    """

    if copy: df = df.copy()

    if index_cols is None:
        try:
            df[[i + pct_change_suffix for i in pct_change_cols]] = df[pct_change_cols].pct_change()
        except ZeroDivisionError:
            df[pct_change_cols] = df[pct_change_cols] + epsilon
            df[[i + pct_change_suffix for i in pct_change_cols]] = df[pct_change_cols].pct_change()
    else:
        assert len(np.intersect1d(list(index_cols), pct_change_cols)) == 0, 'index_cols cannot overlap with diff_cols'
        pct_chg_df = df[list(index_cols) + list(pct_change_cols)]
        pct_chg_df.set_index(list(index_cols), inplace=True)
        df.set_index(list(index_cols), inplace=True)

        try:
            pct_chg_df = pct_chg_df[pct_change_cols].pct_change().add_suffix(pct_change_suffix)
        except ZeroDivisionError:
            pct_chg_df = pct_chg_df + epsilon
            pct_chg_df = pct_chg_df.pct_change().add_suffix(pct_change_suffix)

        df = pd.merge(df, pct_chg_df, left_index=True, right_index=True)

        if reset_index:
            df.reset_index(inplace=True)
            if 'index' in df.columns:
                df.drop('index', axis=1, inplace=True)

    DataValidator(df).validate_data()

    return df


class CreateTargets():

    def __init__(self, df, copy=False):
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
                           stop,
                           lm_col='low_move_pct',
                           hm_col='high_move_pct',
                           target_suffix='target_HL5'):


        """ note: hm stands for high move, lm stands for low move """

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
                    (~self.df[target_suffix].isin([0, 4])),
                    target_suffix] = 3


        # Medium Sell
        self.df.loc[(self.df[lm_col] <= (-1) * med_sell) &
                    (self.df[hm_col] <= stop) &
                    (~self.df[target_suffix].isin([0, 3, 4])),
                    target_suffix] = 1

        # No Trade
        self.df.loc[(~self.df[target_suffix].isin([0, 1, 3, 4])), target_suffix] = 2
        return self.df

    def create_targets_HL3(self,
                           buy,
                           sell,
                           stop,
                           lm_col='low_move_pct',
                           hm_col='high_move_pct',
                           target_suffix='target_HL3'):

        """ hm stands for high move, lm stands for low move """

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
        self.df.loc[~(self.df[target_suffix].isin([0, 2])), target_suffix] = 1

        return self.df


def create_lagging_features(df, lagging_map, new_col_prefix='prev', copy=False):

    """

    Parameters
    __________

    df : pandas df
    lagging_map : dict with keys as colnames and values as a list of periods for computing lagging features
    periods : periods to look back

    """

    if copy: df = df.copy()

    unique_lagging_values = list(sorted({k for v in lagging_map.values() for k in v}))

    for period in unique_lagging_values:
        new_col_prefix_tmp = new_col_prefix + str(period) + '_'
        cols_to_lag = [k for k, v in lagging_map.items() if period in v]
        df[[new_col_prefix_tmp + c for c in cols_to_lag]] = df[cols_to_lag].shift(periods=period)

    return df




def create_rolling_features(df,
                            rolling_fn='mean',
                            ewm_fn='mean',
                            rolling_params={},
                            ewm_params={},
                            rolling_cols='all_numeric',
                            ewm_cols='all_numeric',
                            join_method='outer',
                            index_cols=None,
                            reset_index=True,
                            copy=False):
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

    index_cols: None, list, or tuple of cols that are the indices of the df

    reset_index: Bool: if true then reset the index after setting index before joining, else return the index_cols as the index in the df
    copy : bool whether or not to make a copy of the df before applying logic

    """

    if copy: df = df.copy()

    if isinstance(rolling_cols, str) and rolling_cols.lower() == 'all_numeric':
        rolling_cols = list(df.select_dtypes(include=np.number).columns)

    if isinstance(rolling_cols, str) and ewm_cols.lower() == 'all_numeric':
        ewm_cols = list(df.select_dtypes(include=np.number).columns)

    if index_cols is None:
        # rolling
        if rolling_fn is not None and len(rolling_cols) > 0:
            new_rolling_cols = [i + '_rolling_' + rolling_fn for i in rolling_cols]
            df[new_rolling_cols] = getattr(df[rolling_cols].rolling(**rolling_params), rolling_fn)()

        # ewm
        if ewm_fn is not None and len(ewm_cols) > 0:
            new_ewm_cols = [i + '_ewm_' + ewm_fn for i in ewm_cols]
            df[new_ewm_cols] = getattr(df[ewm_cols].ewm(**ewm_params), ewm_fn)()

    else:
        df.set_index(list(index_cols), inplace=True)

        # rolling
        if rolling_fn is not None and len(rolling_cols) > 0:
            new_rolling_cols = [i + '_rolling_' + rolling_fn for i in rolling_cols]
            rolling_df = getattr(df[rolling_cols].rolling(**rolling_params), rolling_fn)()
            rolling_df.columns = new_rolling_cols
            df = pd.merge(df, rolling_df[[i for i in rolling_df.columns if i not in df.columns]], left_index=True, right_index=True)
        # ewm
        if ewm_fn is not None and len(ewm_cols) > 0:
            new_ewm_cols = [i + '_ewm_' + ewm_fn for i in ewm_cols]
            ewm_df = getattr(df[ewm_cols].ewm(**ewm_params), ewm_fn)()
            ewm_df.columns = new_ewm_cols
            df = pd.merge(df, ewm_df[[i for i in ewm_df.columns if i not in df.columns]], left_index=True, right_index=True)

    if reset_index:
        df.reset_index(inplace=True)
        if 'index' in df.columns:
            df.drop('index', axis=1, inplace=True)

    DataValidator(df).validate_data()

    return df



def drop_nas(df, col_contains, exception_cols=None, how=None, copy=False, **dropna_params):

    """

    Description: The goal of this function is to conditionally drop rows with a significant amount of NAs

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

    exception_cols = [] if exception_cols is None else exception_cols

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

    return df.dropna(how=how, subset=selected_cols, **dropna_params)


def calc_move_iar(df, iar_cols, iar_suffix='_iar', copy=False):

    if copy: df = df.copy()

    assert isinstance(iar_cols, str) or isinstance(iar_cols, list), 'iar_cols must be a str or list!'

    upmove_iar = df[iar_cols].transform(lambda x: x.cumsum().sub(x.cumsum().mask(x >= 0).ffill(), fill_value=0), axis=0).replace(0, np.nan)
    downmove_iar = df[iar_cols].transform(lambda x: x.cumsum().sub(x.cumsum().mask(x <= 0).ffill(), fill_value=0), axis=0).replace(0, np.nan)

    if isinstance(iar_cols, str):
        new_iar_cols = iar_cols + iar_suffix
    else:
        new_iar_cols = [i + iar_suffix for i in iar_cols]

    df[new_iar_cols] = upmove_iar.fillna(downmove_iar).ffill()

    DataValidator(df).validate_data()

    return df

def calc_trend(df, iar_cols, iar_suffix='_iar', trend_suffix='_trend', flat_threshold=0.005, copy=False):

    if copy: df = df.copy()

    new_iar_colname = iar_cols + iar_suffix
    trend_colname = iar_cols + trend_suffix
    df.loc[df[iar_cols] > 0, trend_colname] = 'up'
    df.loc[df[iar_cols] < 0, trend_colname] = 'down'
    df.loc[np.abs(df[iar_cols]) <= flat_threshold, trend_colname] = 'flat'

    return df

def calc_coef(df, target_colname, pred_colname):

    """Takes df as input and calculates spearman correlation between target and prediction"""

    # method="first" breaks ties based on order in array
    correlation = np.corrcoef(df[target_colname],
                              df[pred_colname].rank(pct=True, method="first")
                              )[0,1]
    return correlation

def split_list(lst, n):
    assert isinstance(lst, list) | isinstance(lst, np.ndarray), 'lst is not a list'
    yield [lst[i: i + n] for i in range(0, len(lst), n)]


def plot_coef_scores(era_scores,
                     x='date',
                     y='era_score',
                     groupby_cols=None,
                     rolling_period=10,
                     verbose=True,
                     copy=False,
                     **plotly_params):

    if copy: era_scores = era_scores.copy()

    ### not grouped ###

    if groupby_cols is None or not len(groupby_cols):

        if not era_scores.index.name == x:
            era_scores.set_index(x, inplace=True)

        ### rolling mean ###

        fig1 = px.line(era_scores.rolling(rolling_period).mean().reset_index(), x=x, y=y, **plotly_params)
        fig1.add_hline(y=0)
        fig1.show()

        ### cumsum ###

        fig2 = px.line(era_scores.cumsum(), **plotly_params)
        fig2.add_hline(y=0)
        fig2.show()

    ### grouped ###

    else:

        ### rolling mean ###

        era_scores_grouped1 = era_scores.groupby(groupby_cols).rolling(rolling_period).mean()

        fig1 = px.line(era_scores_grouped1.reset_index(),
                       x=x,
                       y=y,
                       line_group=groupby_cols,
                       color=groupby_cols,
                       **plotly_params)
        fig1.add_hline(y=0)
        fig1.show()


        ### cumsum ###

        era_scores_grouped2 = era_scores.reset_index()\
                                        .set_index([x, groupby_cols])\
                                        .groupby(groupby_cols)\
                                        .cumsum()\
                                        .reset_index(level=groupby_cols)

        fig2 = px.line(era_scores_grouped2,
                       line_group=groupby_cols,
                       color=groupby_cols,
                       **plotly_params)
        fig2.add_hline(y=0)
        fig2.show()

    if verbose:
        if groupby_cols: era_scores = era_scores[y]
        print(f"Mean Correlation: {era_scores.mean(): .3f}")
        print(f"Median Correlation: {era_scores.median(): .3f}")
        print(f"Standard Deviation: {era_scores.std(): .3f}\n")
        print(f"Mean Pseudo-Sharpe: {era_scores.mean()/era_scores.std(): .3f}")
        print(f"Median Pseudo-Sharpe: {era_scores.median()/era_scores.std(): .3f}\n")
        print(f'Hit Rate (% positive eras): {era_scores.apply(lambda x: np.sign(x)).value_counts()[1]/len(era_scores):.2%}')
    return


def calc_fnc(sub, targets, features):
    """
    Args:
        sub (pd.Series)
        targets (pd.Series)
        features (pd.DataFrame)
    """

    # Normalize submission
    sub = (sub.rank(method="first").values - 0.5) / len(sub)
    # Neutralize submission to features
    f = features.values
    sub -= f.dot(np.linalg.pinv(f).dot(sub))
    sub /= sub.std()

    sub = pd.Series(np.squeeze(sub)) # Convert np.ndarray to pd.Series

    # FNC: Spearman rank-order correlation of neutralized submission to target
    fnc = np.corrcoef(sub.rank(pct=True, method="first"), targets)[0, 1]
    return fnc





def download_ticker_map(napi,
                        numerai_ticker_link,
                        main_ticker_col='bloomberg_ticker',
                        yahoo_ticker_colname='yahoo',
                        verbose=True):

    eligible_tickers = pd.Series(napi.ticker_universe(), name=yahoo_ticker_colname)
    ticker_map = pd.read_csv(numerai_ticker_link)
    ticker_map = ticker_map[ticker_map[main_ticker_col].isin(eligible_tickers)]

    if verbose:
        print(f"Number of eligible tickers: {len(eligible_tickers)}")
        print(f"Number of eligible tickers in map: {len(ticker_map)}")

    # Remove null / empty tickers from the yahoo tickers
    valid_tickers = [i for i in ticker_map[yahoo_ticker_colname]
                     if not pd.isnull(i)
                     and not str(i).lower() == 'nan' \
                     and not str(i).lower() == 'null' \
                     and i is not None \
                     and not str(i).lower() == '' \
                     and len(i) > 0 \
                     ]

    if verbose: print('tickers before cleaning:', ticker_map.shape)  # before removing bad tickers
    ticker_map = ticker_map[ticker_map[yahoo_ticker_colname].isin(valid_tickers)]
    if verbose: print('tickers after cleaning:', ticker_map.shape)

    return ticker_map


def create_datetime_features(df, datetime_col, include_hour=True, make_copy=False):

    if make_copy: df = df.copy()

    if not is_datetime(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
    if include_hour: df.loc[:, 'hour'] = df[datetime_col].dt.hour

    df.loc[:, 'day'] = df[datetime_col].dt.isocalendar().day
    df.loc[:, 'week'] = df[datetime_col].dt.isocalendar().week
    df.loc[:, 'month'] = df[datetime_col].dt.month
    df.loc[:, 'dayofweek'] = df[datetime_col].dt.dayofweek
    df.loc[:, 'dayofyear'] = df[datetime_col].dt.dayofyear
    df.loc[:, 'quarter'] = df[datetime_col].dt.quarter
    df.loc[:, 'is_month_start'] = df[datetime_col].dt.is_month_start
    df.loc[:, 'is_month_end'] = df[datetime_col].dt.is_month_end
    df.loc[:, 'is_quarter_start'] = df[datetime_col].dt.is_quarter_start
    df.loc[:, 'is_quarter_end'] = df[datetime_col].dt.is_quarter_end
    
    holidays = calendar().holidays(start=df[datetime_col].min(), end=df[datetime_col].max())
    
    df.loc[:, 'is_holiday'] = df[datetime_col].isin(holidays)
    
    return df
