def get_column_names_from_ColumnTransformer(column_transformer, clean_column_names=True, verbose=True):
    """
    Reference: Kyle Gilde: https://github.com/kylegilde/Kaggle-Notebooks/blob/master/Extracting-and-Plotting-Scikit-Feature-Names-and-Importances/feature_importance.py
    Description: Get the column names from the a ColumnTransformer containing transformers & pipelines
    Parameters
    ----------
    verbose: Bool indicating whether to print summaries. Default set to True.
    Returns
    -------
    a list of the correct feature names
    Note:
    If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely new columns,
    it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator & SimpleImputer(add_indicator=True) add columns
    to the dataset that didn't exist before, so there should come last in the Pipeline.
    Inspiration: https://github.com/scikit-learn/scikit-learn/issues/12525
    """

    assert isinstance(column_transformer, ColumnTransformer), "Input isn't a ColumnTransformer"

    check_is_fitted(column_transformer)

    new_feature_names, transformer_list = [], []

    for i, transformer_item in enumerate(column_transformer.transformers_):
        transformer_name, transformer, orig_feature_names = transformer_item
        orig_feature_names = list(orig_feature_names)

        if len(orig_feature_names) == 0:
            continue

        if verbose:
            print(f"\n\n{i}.Transformer/Pipeline: {transformer_name} {transformer.__class__.__name__}\n")
            print(f"\tn_orig_feature_names:{len(orig_feature_names)}")

        if transformer == 'drop':
            continue

        if isinstance(transformer, Pipeline):
            # if pipeline, get the last transformer in the Pipeline
            transformer = transformer.steps[-1][1]

        if hasattr(transformer, 'get_feature_names_out'):
            if 'input_features' in transformer.get_feature_names_out.__code__.co_varnames:
                names = list(transformer.get_feature_names_out(orig_feature_names))
            else:
                names = list(transformer.get_feature_names_out())
        elif hasattr(transformer, 'get_feature_names'):
            if 'input_features' in transformer.get_feature_names.__code__.co_varnames:
                names = list(transformer.get_feature_names(orig_feature_names))
            else:
                names = list(transformer.get_feature_names())

        elif hasattr(transformer, 'indicator_') and transformer.add_indicator:
            # is this transformer one of the imputers & did it call the MissingIndicator?

            missing_indicator_indices = transformer.indicator_.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag' \
                                  for idx in missing_indicator_indices]
            names = orig_feature_names + missing_indicators

        elif hasattr(transformer, 'features_'):
            # is this a MissingIndicator class?
            missing_indicator_indices = transformer.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag' \
                                  for idx in missing_indicator_indices]

        else:
            names = orig_feature_names

        if verbose:
            print(f"\tn_new_features:{len(names)}")
            print(f"\tnew_features: {names}\n")

        new_feature_names.extend(names)
        transformer_list.extend([transformer_name] * len(names))

    if clean_column_names:
        new_feature_names = list(clean_columns(pd.DataFrame(columns=new_feature_names)).columns)

    return new_feature_names


class PreprocessFeatures(TransformerMixin):
    """
        Parameters
        ----------
        preserve_vars : A list of variables that won't be fitted or transformed by any sort of feature engineering
        target : A string - the name of the target variable.
        remainder : A string that gets passed to the column transformer whether to
                    drop preserve_vars or keep them in the final dataset
                    options are 'drop' or 'passthrough'
        max_oh_cardinality : A natural number - one-hot encode all features with unique categories <= to this value
        FE_pipeline_dict : Set to None to use "standard" feature engineering pipeline. Otherwise, supply a dictionary of pipelines to hc_pipe, oh_pipe, numeric_pipe, and custom_pipe
        n_jobs : An int - the number of threads to use
        copy : boolean to copy X_train and X_test while preprocessing
        -------
        Attributes
        detect_feature_types attributes are dictionary attributes
        fit attributes are sklearn ColumnTransformer attributes
        -------
        Returns
        detect features returns a dictionary
        fit returns a ColumnTransformer object
        We can call fit_transform because we inherited the sklearn base TransformerMixin class
        -------
    """

    def __init__(self,
                 target=None,
                 preserve_vars=None,
                 FE_pipeline_dict=None,
                 remainder='drop',
                 max_oh_cardinality=11,
                 detect_dtypes=True,
                 numeric_features=None,
                 oh_features=None,
                 hc_features=None,
                 overwrite_detection=True,
                 n_jobs=-1,
                 copy=True,
                 verbose=True):

        self.preserve_vars = preserve_vars
        self.target = target
        self.FE_pipeline_dict = FE_pipeline_dict
        self.remainder = remainder
        self.max_oh_cardinality = max_oh_cardinality
        self.detect_dtypes = detect_dtypes
        self.numeric_features = [] if numeric_features is None else numeric_features
        self.oh_features = [] if oh_features is None else oh_features
        self.hc_features = [] if hc_features is None else hc_features
        self.overwrite_detection = overwrite_detection
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.copy = copy

        self.preserve_vars = [] if self.preserve_vars is None else self.preserve_vars
        self.target = '' if self.target is None else self.target

    def detect_feature_types(self, X):

        if self.copy: X = X.copy()

        if not self.detect_dtypes:
            ds_print('Not detecting dtypes.', verbose=self.verbose)
            feature_dict = {'numeric_features': self.numeric_features,
                            'oh_features': self.oh_features,
                            'hc_features': self.hc_features,
                            'custom_features': self.FE_pipeline_dict['custom_pipeline'].values()}
            return feature_dict

        if self.FE_pipeline_dict is not None and 'custom_pipeline' in self.FE_pipeline_dict.keys():
                custom_features = list(itertools.chain(*self.FE_pipeline_dict['custom_pipeline'].values()))
        else:
            custom_features = []

        assert len(
            np.intersect1d(list(set(self.numeric_features + self.oh_features + self.hc_features + custom_features)),
                           self.preserve_vars)) == 0, \
            'There are duplicate features in preserve_vars either the input numeric_features, oh_features, or hc_features'

        detected_numeric_vars = make_column_selector(dtype_include=np.number)(
            X[[i for i in X.columns if i not in self.preserve_vars + [self.target] + custom_features]])
        detected_oh_vars = [i for i in X.loc[:, (X.nunique() < self.max_oh_cardinality) & (X.nunique() > 1)].columns if
                            i not in self.preserve_vars + [self.target] + custom_features]
        detected_hc_vars = X[[i for i in X.columns if i not in self.preserve_vars + custom_features]] \
            .select_dtypes(['object', 'category']) \
            .apply(lambda col: col.nunique()) \
            .loc[lambda x: x > self.max_oh_cardinality] \
            .index.tolist()

        discarded_features = [i for i in X.isnull().sum()[X.isnull().sum() == X.shape[0]].index if i not in self.preserve_vars]

        numeric_features = list(set([i for i in self.numeric_features + [i for i in detected_numeric_vars if
                                                                         i not in list(self.oh_features) + list(
                                                                             self.hc_features) + list(
                                                                             discarded_features) + custom_features]]))

        oh_features = list(set([i for i in self.oh_features + [i for i in detected_oh_vars if
                                                               i not in list(self.numeric_features) + list(
                                                                   self.hc_features) + list(
                                                                   discarded_features) + custom_features]]))

        hc_features = list(set([i for i in self.hc_features + [i for i in detected_hc_vars if
                                                               i not in list(self.numeric_features) + list(
                                                                   self.oh_features) + list(
                                                                   discarded_features) + custom_features]]))

        ds_print('Overlap between numeric and oh_features: ' + str(list(set(np.intersect1d(numeric_features, oh_features)))), verbose=self.verbose)
        ds_print('Overlap between numeric and hc_features: ' + str(list(set(np.intersect1d(numeric_features, hc_features)))), verbose=self.verbose)
        ds_print('Overlap between numeric oh_features and hc_features: ' + str(list(set(np.intersect1d(oh_features, hc_features)))), verbose=self.verbose)
        ds_print('Overlap between oh_features and hc_features will be moved to oh_features', verbose=self.verbose)

        if self.overwrite_detection:
            numeric_features = [i for i in numeric_features if
                                i not in oh_features + hc_features + discarded_features + custom_features]
            oh_features = [i for i in oh_features if
                           i not in hc_features + numeric_features + discarded_features + custom_features]
            hc_features = [i for i in hc_features if
                           i not in oh_features + numeric_features + discarded_features + custom_features]
        else:
            numeric_overlap = [i for i in numeric_features if
                               i in oh_features or i in hc_features and i not in discarded_features + custom_features]
            oh_overlap = [i for i in oh_features if
                          i in hc_features or i in numeric_features and i not in discarded_features + custom_features]
            hc_overlap = [i for i in hc_features if
                          i in oh_features or i in numeric_features and i not in discarded_features + custom_features]

            if numeric_overlap or oh_overlap or hc_overlap:
                raise('Error - There is an overlap between numeric, oh, and hc features! To ignore this set overwrite_detection to True.')

        all_features = list(set(numeric_features + oh_features + hc_features + discarded_features + custom_features))
        all_features_debug = set(all_features) - set([i for i in X.columns if i not in self.preserve_vars + [self.target]])

        if len(all_features_debug) > 0:
            print('\n{}\n'.format(all_features_debug))
            raise('There was a problem detecting all features!! Check if there is an overlap between preserve_vars and other custom input features')

        ds_print('\nnumeric_features:' + str(numeric_features), verbose=self.verbose)
        ds_print('\noh_features:' + str(oh_features), verbose=self.verbose)
        ds_print('\nhc_features:' + str(hc_features), verbose=self.verbose)
        ds_print('\ndiscarded_features:' + str(discarded_features), verbose=self.verbose)
        ds_print('\ncustom_pipeline:' + str(custom_features), verbose=self.verbose)

        feature_dict = {'numeric_features': numeric_features,
                        'oh_features': oh_features,
                        'hc_features': hc_features,
                        'custom_features': custom_features,
                        'discarded_features': discarded_features}

        return feature_dict

    def fit(self, X, y=None, remainder='drop'):

        """ This breaks the sklearn standard of returning self, but I don't currently know a better way to do this """

        if self.target is None and y is not None:
            self.target = y.name

        assert y is not None and self.target is not None, '\n Both self.target and y cannot be None!'

        if self.copy:
            X = X.copy()
            if y is not None:
                y = y.copy()

        feature_types = self.detect_feature_types(X)

        if self.FE_pipeline_dict is None:

            # Default below
            numeric_pipe = make_pipeline(
                FunctionTransformer(lambda x: x.replace([np.inf, -np.inf], np.nan)),
                # Winsorizer(distribution='gaussian', tail='both', fold=3, missing_values = 'ignore'),
                MinMaxScaler(feature_range=(0, 1)),
                SimpleImputer(strategy='median', add_indicator=True)
            )

            hc_pipe = make_pipeline(
                FunctionTransformer(lambda x: x.replace([np.inf, -np.inf], np.nan).astype(str)),
                TargetEncoder(return_df=True,
                              handle_missing='value',
                              handle_unknown='value',
                              min_samples_leaf=10)
            )

            oh_pipe = make_pipeline(
                FunctionTransformer(lambda x: x.replace([np.inf, -np.inf], np.nan).astype(str)),
                OneHotEncoder(handle_unknown='ignore', sparse=False)
            )

            custom_pipe = None

        else:
            hc_pipe = self.FE_pipeline_dict['hc_pipe']
            numeric_pipe = self.FE_pipeline_dict['numeric_pipe']
            oh_pipe = self.FE_pipeline_dict['oh_pipe']
            custom_pipe = self.FE_pipeline_dict['custom_pipeline'] if 'custom_pipeline' in self.FE_pipeline_dict.keys() else {}

        transformers = [
            ('hc_pipeline', hc_pipe, feature_types['hc_features']),
            ('numeric_pipeline', numeric_pipe, feature_types['numeric_features']),
            ('oh_encoder', oh_pipe, feature_types['oh_features'])
        ]

        if custom_pipe:
            i = 0
            for cp in custom_pipe.keys():
                transformers.append(('custom_pipe{}'.format(str(i)), cp, custom_pipe[cp]))
                i += 1

        if y is None:
            feature_transformer = ColumnTransformer(
                transformers=transformers,
                remainder=remainder,
                n_jobs=self.n_jobs).fit(X)
        else:
            feature_transformer = ColumnTransformer(
                transformers=transformers,
                remainder=remainder,
                n_jobs=self.n_jobs).fit(X, y)

        setattr(feature_transformer, 'feature_types', feature_types)

        return feature_transformer
