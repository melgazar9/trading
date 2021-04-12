from functools import reduce
import pandas as pd
import numpy as np

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

    df['move' + new_col_suffix] = df[close_col] - df[open_col]
    df['move_pct' + new_col_suffix] = df['move' + new_col_suffix] / df[open_col]
    df['move_pct_change' + new_col_suffix] = df['move' + new_col_suffix].pct_change()
    df['open_minus_prev_close' + new_col_suffix] = df[open_col] - df[close_col].shift()
    df['prev_close_pct_chg' + new_col_suffix] = df['move' + new_col_suffix] / df[close_col].shift()

    df['range' + new_col_suffix] = df[high_col] - df[low_col]
    df['range_pct_change' + new_col_suffix] = df['range' + new_col_suffix].pct_change()

    df['high_move' + new_col_suffix] = df[high_col] - df[open_col]
    df['high_move_pct' + new_col_suffix] = df['high_move' + new_col_suffix] / df[open_col]
    df['high_move_pct_change' + new_col_suffix] = df['high_move' + new_col_suffix].pct_change()

    df['low_move' + new_col_suffix] = df[low_col] - df[open_col]
    df['low_move_pct' + new_col_suffix] = df['low_move' + new_col_suffix] / df[open_col]
    df['low_move_pct_change' + new_col_suffix] = df['low_move' + new_col_suffix].pct_change()

    df['volume_diff' + new_col_suffix] = df[volume_col] - df[volume_col].shift()
    df['volume_pct_change' + new_col_suffix] = df[volume_col].pct_change()

    df['close_minus_low' + new_col_suffix] = df[close_col] - df[low_col]
    df['high_minus_close' + new_col_suffix] = df[high_col] - df[close_col]

    df['prev_close_minus_low_minus' + new_col_suffix] = df[close_col].shift() - df[low_col]
    df['high_minus_prev_close' + new_col_suffix] = df[high_col] - df[close_col].shift()

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

    if groupby_cols is None or len(groupby_cols) == 0:
        for period in unique_lagging_values:
            new_col_prefix_tmp = new_col_prefix + str(period) + '_'
            cols_to_lag = [k for k, v in lagging_map.items() if period in v]
            df[[new_col_prefix_tmp + c for c in cols_to_lag]] = df[cols_to_lag].transform(lambda s: s.shift(periods=period))

    else:
        for period in unique_lagging_values:
            new_col_prefix_tmp = new_col_prefix + str(period) + '_'
            cols_to_lag = [k for k, v in lagging_map.items() if period in v]

            df[[new_col_prefix_tmp + c for c in cols_to_lag]] = df.groupby(groupby_cols)[cols_to_lag]\
                                                                  .transform(lambda s: s.shift(periods=period))
    return df


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


    if groupby_cols is None or len(groupby_cols) == 0:

        # rolling
        if rolling_fn is not None and len(rolling_cols) > 0:
            new_rolling_cols = [i + '_rolling_' + rolling_fn for i in rolling_cols]
            df[new_rolling_cols] = getattr(df[rolling_cols].rolling(**rolling_params), rolling_fn)()

        # ewm
        if ewm_fn is not None and len(ewm_cols) > 0:
            new_ewm_cols = [i + '_ewm_' + ewm_fn for i in ewm_cols]
            df[new_ewm_cols] = getattr(df[ewm_cols].ewm(**ewm_params), ewm_fn)()

    else:

        if isinstance(groupby_cols, str):
            groupby_cols = [groupby_cols]
        else:
            raise ('Input param groupby_cols is not a list, string, or None!')

        assert len(groupby_cols) == len(set(groupby_cols)), 'There are duplicates in groupby_cols!'
        rolling_cols_to_select = [i for i in list(set(groupby_cols + rolling_cols)) if i in df.columns] # check index name
        ewm_cols_to_select = [i for i in list(set(groupby_cols + ewm_cols)) if i in df.columns]

        # rolling
        if rolling_fn is not None and len(rolling_cols) > 0:
            new_rolling_cols = [i + '_rolling_' + rolling_fn for i in rolling_cols_to_select if not i in groupby_cols]
            df[new_rolling_cols] = df[rolling_cols_to_select]\
                                    .groupby(groupby_cols)\
                                    .transform(lambda x: getattr(x.rolling(**rolling_params), rolling_fn)())

        # ewm
        if ewm_fn is not None and len(ewm_cols) > 0:
            new_ewm_cols = [i + '_ewm_' + ewm_fn for i in ewm_cols_to_select if not i in groupby_cols]
            df[new_ewm_cols] = df[ewm_cols_to_select]\
                                .groupby(groupby_cols)\
                                .transform(lambda x: getattr(x.ewm(**ewm_params), ewm_fn)())

    if create_diff_cols:
        diff_cols = [i for i in df.columns if 'ewm' in i or 'rolling' in i]
        if groupby_cols is None or len(groupby_cols) == 0:
            df = pd.concat([df, df[diff_cols].diff().add_suffix('_diff')], axis=1)
        else:
            df[[i + '_diff' for i in diff_cols]] = df.groupby(groupby_cols)[diff_cols].transform(lambda col: col.diff())

    return df



def drop_suffix_nas(df, col_suffix='1d', id_cols=['date', 'bloomberg_ticker']):
    df_ids = df[[col for col in df.columns \
                 if col.endswith(col_suffix) \
                 or col in id_cols] \
        ].dropna()[id_cols].isin(df[id_cols])

    df = df[df[id_cols].isin(df_ids[id_cols])]
    return df
