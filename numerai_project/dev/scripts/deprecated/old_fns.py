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
