# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 10:57:45 2021

@author: Matt
"""
import os
import time
import json
import pandas as pd
from dask import delayed
import dask
import numpy as np
import gc
import datetime
from pandas.io.json import json_normalize
import itertools
import dill
import warnings
import subprocess
import sys
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler
from itertools import cycle
from collections import Counter
import re
from multiprocessing import Pool, Process
import matplotlib.pyplot as plt
import configparser
from sklearn.pipeline import Pipeline, make_pipeline
from category_encoders import TargetEncoder
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from category_encoders import TargetEncoder
from sklearn.impute import MissingIndicator
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder, LabelEncoder
from sklearn.base import TransformerMixin
from feature_engine.outliers import Winsorizer, ArbitraryOutlierCapper#, OutlierTrimmer
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, KMeansSMOTE, SMOTENC, SVMSMOTE, BorderlineSMOTE
from imblearn.under_sampling import (
     RandomUnderSampler,
     EditedNearestNeighbours,
     TomekLinks,
     AllKNN,
     ClusterCentroids,
     CondensedNearestNeighbour,
     InstanceHardnessThreshold,
     NearMiss,
     OneSidedSelection)
from sklearn.metrics import (
     mean_squared_error,
     mean_absolute_error,
     mean_squared_log_error,
     accuracy_score,
     precision_score,
     recall_score,
     r2_score)
from sklearn.ensemble import (
     RandomForestRegressor,
     RandomForestClassifier,
     StackingRegressor,
     StackingClassifier,AdaBoostClassifier,
     AdaBoostRegressor,
     HistGradientBoostingClassifier,
     HistGradientBoostingRegressor)
from sklearn.decomposition import PCA, TruncatedSVD
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from xgboost import XGBRegressor, XGBClassifier
import pymysql
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.metrics import roc_auc_score
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoLars, BayesianRidge, LogisticRegression, TweedieRegressor
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.dummy import DummyRegressor, DummyClassifier
import plotly.express as px
import plotly
import multiprocessing as mp
from functools import partial
import inspect
from sklearn.utils.validation import check_is_fitted
import sklearn
import feather


def parallize_pandas_func(df, df_attribute, parallelize_by_col=True, num_workers=mp.cpu_count(), **kwargs):
    """ parallelize by row not implemented yet """
    start_pos = 0
    chunk_len = int(np.floor(len(df.columns) / num_workers))
    delayed_list = []
    end_pos = chunk_len

    if parallelize_by_col:
        for chunk in range(num_workers):
            if chunk != num_workers - 1:
                df_subset = df.iloc[:, start_pos:end_pos]
                delayed_list.append(delayed(getattr(df_subset, df_attribute)(**kwargs)))
                start_pos += chunk_len
                end_pos += chunk_len
            else:
                df_subset = df.iloc[:, start_pos:]
                delayed_list.append(delayed(getattr(df_subset, df_attribute)(**kwargs)))

        dask_tuple = dask.compute(*delayed_list)
        df_out = pd.concat([i for i in dask_tuple], axis = 1)
        return df_out


def get_column_names_from_ColumnTransformer(column_transformer, clean_column_names=False, verbose=True):

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

        if transformer == 'drop' or transformer == 'passthrough':
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
            missing_indicators = [orig_feature_names[idx] + '_missing_flag' for idx in missing_indicator_indices]
            names = orig_feature_names + missing_indicators

        elif hasattr(transformer, 'features_'):
            # is this a MissingIndicator class?
            missing_indicator_indices = transformer.features_
            missing_indicators = [orig_feature_names[idx] + '_missing_flag' for idx in missing_indicator_indices]

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

def ds_print(*args, verbose=True):
    if verbose: print(*args)


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
                 remainder='passthrough',
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

        self.feature_transformer = ColumnTransformer(transformers=[], remainder=self.remainder, n_jobs=self.n_jobs)

    def detect_feature_types(self, X):

        if self.copy: X = X.copy()

        if not self.detect_dtypes:
            if self.verbose: print('Not detecting dtypes.')
  
            feature_dict = {'numeric_features': self.numeric_features,
                            'oh_features': self.oh_features,
                            'hc_features': self.hc_features}
            if 'custom_pipeline' in self.FE_pipeline_dict.keys():
                feature_dict['custom_pipeline'] = self.FE_pipeline_dict['custom_pipeline'].values()
            return feature_dict

        if self.FE_pipeline_dict is not None and 'custom_pipeline' in self.FE_pipeline_dict.keys():
            custom_features = list(itertools.chain(*self.FE_pipeline_dict['custom_pipeline'].values()))
        else:
            custom_features = []

        assert len(np.intersect1d(list(set(self.numeric_features + \
                                           self.oh_features + \
                                           self.hc_features + \
                                           custom_features)), \
                                  self.preserve_vars)) == 0, \
            'There are duplicate features in preserve_vars either the input\
             numeric_features, oh_features, or hc_features'

        detected_numeric_vars = make_column_selector(dtype_include=np.number)(
            X[[i for i in X.columns \
               if i not in self.preserve_vars + \
               [self.target] + \
               custom_features]])
        
        detected_oh_vars = [i for i in X.loc[:, (X.nunique() <= self.max_oh_cardinality) & \
                                             (X.nunique() > 1)].columns \
                            if i not in self.preserve_vars + \
                            [self.target] + \
                            custom_features]
        
        detected_hc_vars = X[[i for i in X.columns \
                              if i not in self.preserve_vars + \
                              custom_features]] \
            .select_dtypes(['object', 'category']) \
            .apply(lambda col: col.nunique()) \
            .loc[lambda x: x > self.max_oh_cardinality] \
            .index.tolist()

        discarded_features = [i for i in X.isnull().sum()[X.isnull().sum() == X.shape[0]].index \
                              if i not in self.preserve_vars]

        numeric_features = list(set([i for i in self.numeric_features + \
                                     [i for i in detected_numeric_vars \
                                      if i not in list(self.oh_features) + \
                                      list(self.hc_features) + \
                                      list(discarded_features) + \
                                      custom_features]]))

        oh_features = list(set([i for i in self.oh_features + \
                                [i for i in detected_oh_vars \
                                 if i not in list(self.numeric_features) + \
                                 list(self.hc_features) + \
                                 list(discarded_features) + \
                                 custom_features]]))

        hc_features = list(set([i for i in self.hc_features + \
                                [i for i in detected_hc_vars \
                                 if i not in list(self.numeric_features) + \
                                 list(self.oh_features) + \
                                 list(discarded_features) + \
                                 custom_features]]))

        if self.verbose:
            
            print('Overlap between numeric and oh_features: ' + \
                  str(list(set(np.intersect1d(numeric_features, oh_features)))))
            print('Overlap between numeric and hc_features: ' + \
                  str(list(set(np.intersect1d(numeric_features, hc_features)))))
            print('Overlap between numeric oh_features and hc_features: ' + \
                  str(list(set(np.intersect1d(oh_features, hc_features)))))
            print('Overlap between oh_features and hc_features will be moved to oh_features')


        if self.overwrite_detection:
            numeric_features = [i for i in numeric_features \
                                if i not in oh_features + \
                                hc_features + \
                                discarded_features + \
                                custom_features]
            
            oh_features = [i for i in oh_features \
                           if i not in hc_features + \
                           numeric_features + \
                           discarded_features + \
                           custom_features]
            
            hc_features = [i for i in hc_features if i not in \
                           oh_features + \
                           numeric_features + \
                           discarded_features + \
                           custom_features]

        else:
            numeric_overlap = [i for i in numeric_features \
                               if i in oh_features \
                               or i in hc_features \
                               and i not in discarded_features + \
                               custom_features]
            
            oh_overlap = [i for i in oh_features \
                          if i in hc_features \
                          or i in numeric_features \
                          and i not in discarded_features + \
                          custom_features]
            
            hc_overlap = [i for i in hc_features \
                          if i in oh_features \
                          or i in numeric_features \
                          and i not in discarded_features + \
                          custom_features]

            if numeric_overlap or oh_overlap or hc_overlap:
                
                raise AssertionError('There is an overlap between numeric, \
                                     oh, and hc features! \
                                     To ignore this set overwrite_detection to True.')

        all_features = list(set(numeric_features + \
                                oh_features + \
                                hc_features + \
                                discarded_features + \
                                custom_features))
        
        all_features_debug = set(all_features) - \
            set([i for i in X.columns \
                 if i not in \
                 self.preserve_vars + [self.target]])

        if len(all_features_debug) > 0:
            print('\n{}\n'.format(all_features_debug))
            raise AssertionError('There was a problem detecting all features!! \
                Check if there is an overlap between preserve_vars and other custom input features')

        if self.verbose:
            print('\nnumeric_features:' + str(numeric_features))
            print('\noh_features:' + str(oh_features))
            print('\nhc_features:' + str(hc_features))
            print('\ndiscarded_features:' + str(discarded_features))
            print('\ncustom_pipeline:' + str(custom_features))

        feature_dict = {'numeric_features': numeric_features,
                        'oh_features': oh_features,
                        'hc_features': hc_features,
                        'custom_features': custom_features,
                        'discarded_features': discarded_features}

        return feature_dict


    def fit(self, X, y=None):

        if self.target is None and y is not None:
            self.target = y.name

        assert y is not None or self.target is not None, '\n Both self.target and y cannot be None!'

        if self.copy:
            X = X.copy()
            y = y.copy() if y is not None else y

        feature_types = self.detect_feature_types(X)

        # set a default transformation pipeline if FE_pipeline_dict is not specified
        if self.FE_pipeline_dict is None:

            # Default pipeline
            numeric_pipe = make_pipeline(
                FunctionTransformer(lambda x: x.replace([np.inf, -np.inf, None], np.nan)),
                # Winsorizer(distribution='gaussian', tail='both', fold=3, missing_values = 'ignore'),
                MinMaxScaler(feature_range=(0, 1)),
                SimpleImputer(strategy='median', add_indicator=True)
            )

            hc_pipe = make_pipeline(
                FunctionTransformer(lambda x: x.replace([np.inf, -np.inf, None], np.nan).astype(str)),
                TargetEncoder(return_df=True,
                              handle_missing='value',
                              handle_unknown='value',
                              min_samples_leaf=10)
            )

            oh_pipe = make_pipeline(
                FunctionTransformer(lambda x: x.replace([np.inf, -np.inf, None], np.nan).astype(str)),
                OneHotEncoder(handle_unknown='ignore', sparse=False)
            )

            custom_pipe = None

        else:
            hc_pipe = self.FE_pipeline_dict['hc_pipe']
            numeric_pipe = self.FE_pipeline_dict['numeric_pipe']
            oh_pipe = self.FE_pipeline_dict['oh_pipe']
            
            custom_pipe = self.FE_pipeline_dict['custom_pipeline'] \
                if 'custom_pipeline' in self.FE_pipeline_dict.keys() else {}

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

        self.feature_transformer.transformers = transformers

        if y is None:
            self.feature_transformer.fit(X)
        else:
            self.feature_transformer.fit(X, y)


        output_cols = get_column_names_from_ColumnTransformer(self.feature_transformer, verbose=self.verbose)

        if self.remainder == "passthrough":
            output_cols = output_cols + [X.columns[i] for i in self.feature_transformer._remainder[2]]

        setattr(self, 'output_cols', output_cols)
        setattr(self, 'output_features', [i for i in output_cols if i not in self.preserve_vars + [self.target]])
        setattr(self, 'feature_types', feature_types)

        return self

    def transform(self, X, return_df=True):

        X_out = self.feature_transformer.transform(X)

        if return_df:
            return pd.DataFrame(list(X_out), columns=self.output_cols)
        else:
            return X_out



class FeatureImportance:

    """

    Extract & Plot the Feature Names & Importance Values from a Scikit-Learn Pipeline.

    The input is a Pipeline that starts with a ColumnTransformer & ends with a regression or classification model.
    As intermediate steps, the Pipeline can have any number or no instances from sklearn.feature_selection.

    Note:
    If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely new columns,
    it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator & SimpleImputer(add_indicator=True) add columns
    to the dataset that didn't exist before, so there should come last in the Pipeline.

    Parameters
    ----------
    pipeline : a Scikit-learn Pipeline class where the a ColumnTransformer is the first element and model estimator is the last element
    verbose : a boolean. Whether to print all of the diagnostics. Default is False.

    Attributes
    __________
    column_transformer_features :  A list of the feature names created by the ColumnTransformer prior to any selectors being applied
    transformer_list : A list of the transformer names that correspond with the `column_transformer_features` attribute
    discarded_features : A list of the features names that were not selected by a sklearn.feature_selection instance.
    discarding_selectors : A list of the selector names corresponding with the `discarded_features` attribute
    feature_importance :  A Pandas Series containing the feature importance values and feature names as the index.
    plot_importances_df : A Pandas DataFrame containing the subset of features and values that are actually displaced in the plot.
    feature_info_df : A Pandas DataFrame that aggregates the other attributes. The index is column_transformer_features. The transformer column contains the transformer_list.
        value contains the feature_importance values. discarding_selector contains discarding_selectors & is_retained is a Boolean indicating whether the feature was retained.



    """
    def __init__(self, pipeline, verbose=False):

        self.pipeline = pipeline
        self.verbose = verbose


    def get_feature_names(self):

        """

        Get the column names from the a ColumnTransformer containing transformers & pipelines

        Returns
        -------
        a list of the correct feature names

        Note:
        If the ColumnTransformer contains Pipelines and if one of the transformers in the Pipeline is adding completely new columns,
        it must come last in the pipeline. For example, OneHotEncoder, MissingIndicator & SimpleImputer(add_indicator=True) add columns
        to the dataset that didn't exist before, so there should come last in the Pipeline.

        Inspirations
        https://github.com/scikit-learn/scikit-learn/issues/12525
        https://www.kaggle.com/kylegilde/extracting-scikit-feature-names-importances

        """

        if self.verbose: print('''\n\n---------\nRunning get_feature_names\n---------\n''')

        if type(self.pipeline) == sklearn.compose._column_transformer.ColumnTransformer:
            column_transformer = self.pipeline
        else:
            column_transformer = self.pipeline[0]


        assert isinstance(column_transformer, ColumnTransformer), "Input isn't a ColumnTransformer"
        check_is_fitted(column_transformer)

        new_feature_names, transformer_list = [], []

        for i, transformer_item in enumerate(column_transformer.transformers_):

            transformer_name, transformer, orig_feature_names = transformer_item
            orig_feature_names = list(orig_feature_names)

            if self.verbose:
                print('\n\n', i, '. Transformer/Pipeline: ', transformer_name, ',',
                      transformer.__class__.__name__, '\n')
                print('\tn_orig_feature_names:', len(orig_feature_names))

            if transformer == 'drop':
                continue

            if isinstance(transformer, Pipeline):
                # if pipeline, get the last transformer in the Pipeline
                transformer = transformer.steps[-1][1]

            if hasattr(transformer, 'get_feature_names_out'):
                if 'input_features' in transformer.get_feature_names_out.__code__.co_varnames:
                    names = list(transformer.get_feature_names_out(orig_feature_names))
                else:
                    try:
                        names = list(transformer.get_feature_names_out())
                    except:
                        if self.verbose: print('transformer ' + str(i) + ' is not fitted!')
                        continue

            elif hasattr(transformer,'indicator_') and transformer.add_indicator:
                # is this transformer one of the imputers & did it call the MissingIndicator?

                missing_indicator_indices = transformer.indicator_.features_
                missing_indicators = [orig_feature_names[idx] + '_missing_flag' for idx in missing_indicator_indices]
                names = orig_feature_names + missing_indicators

            elif hasattr(transformer,'features_'):
                # is this a MissingIndicator class?
                missing_indicator_indices = transformer.features_
                missing_indicators = [orig_feature_names[idx] + '_missing_flag' for idx in missing_indicator_indices]

            else:
                names = orig_feature_names

            if self.verbose:
                print('\tn_new_features:', len(names))
                print('\tnew_features:\n', names)

            new_feature_names.extend(names)
            transformer_list.extend([transformer_name] * len(names))

        self.transformer_list, self.column_transformer_features = transformer_list, new_feature_names

        return new_feature_names


    def get_selected_features(self):
        """

        Get the Feature Names that were retained after Feature Selection (sklearn.feature_selection)

        Returns
        -------
        a list of the selected feature names


        """

        assert isinstance(self.pipeline, Pipeline), "Input isn't a Pipeline"

        features = self.get_feature_names_out()

        if self.verbose: print('\n\n---------\nRunning get_selected_features\n---------\n')

        all_discarded_features, discarding_selectors = [], []

        for i, step_item in enumerate(self.pipeline.steps[:]):

            step_name, step = step_item

            if hasattr(step, 'get_support'):

                if self.verbose: print('\nStep ', i, ": ", step_name, ',', step.__class__.__name__, '\n')

                check_is_fitted(step)

                feature_mask_dict = dict(zip(features, step.get_support()))

                features = [feature for feature, is_retained in feature_mask_dict.items() if is_retained]

                discarded_features = [feature for feature, is_retained in feature_mask_dict.items() if not is_retained]

                all_discarded_features.extend(discarded_features)
                discarding_selectors.extend([step_name] * len(discarded_features))

                if self.verbose:
                    print(f'\t{len(features)} retained, {len(discarded_features)} discarded')
                    if len(discarded_features) > 0:
                        print('\n\tdiscarded_features:\n\n', discarded_features)

        self.discarded_features, self.discarding_selectors = all_discarded_features,\
                                                                discarding_selectors

        return features

    def get_feature_importance(self):

        """
        Creates a Pandas Series where values are the feature importance values from the model and feature names are set as the index.

        This Series is stored in the `feature_importance` attribute.

        Returns
        -------
        A pandas Series containing the feature importance values and feature names as the index.

        """

        assert isinstance(self.pipeline, Pipeline), "Input isn't a Pipeline"

        features = self.get_selected_features()

        assert hasattr(self.pipeline[-1], 'feature_importances_'),\
            "The last element in the pipeline isn't an estimator with a feature_importances_ attribute"

        importance_values = self.pipeline[-1].feature_importances_

        assert len(features) == len(importance_values),\
            "The number of feature names & importance values doesn't match"

        feature_importance = pd.Series(importance_values, index=features)
        self.feature_importance = feature_importance

        # create feature_info_df
        column_transformer_df =\
            pd.DataFrame(dict(transformer=self.transformer_list),
                         index=self.column_transformer_features)

        discarded_features_df =\
            pd.DataFrame(dict(discarding_selector=self.discarding_selectors),
                         index=self.discarded_features)

        importance_df = self.feature_importance.rename('value').to_frame()

        self.feature_info_df = \
            column_transformer_df\
            .join([importance_df, discarded_features_df])\
            .assign(is_retained = lambda df: ~df.value.isna())


        return feature_importance


    def plot(self, top_n_features=100, rank_features=True, max_scale=True,
             display_imp_values=True, display_imp_value_decimals=1,
             height_per_feature=25, orientation='h', width=750, height=None,
             str_pad_width=15, yaxes_tickfont_family='Courier New',
             yaxes_tickfont_size=15):
        """

        Plot the Feature Names & Importances


        Parameters
        ----------

        top_n_features : the number of features to plot, default is 100
        rank_features : whether to rank the features with integers, default is True
        max_scale : Should the importance values be scaled by the maximum value & mulitplied by 100?  Default is True.
        display_imp_values : Should the importance values be displayed? Default is True.
        display_imp_value_decimals : If display_imp_values is True, how many decimal places should be displayed. Default is 1.
        height_per_feature : if height is None, the plot height is calculated by top_n_features * height_per_feature.
        This allows all the features enough space to be displayed
        orientation : the plot orientation, 'h' (default) or 'v'
        width :  the width of the plot, default is 500
        height : the height of the plot, the default is top_n_features * height_per_feature
        str_pad_width : When rank_features=True, this number of spaces to add between the rank integer and feature name.
            This will enable the rank integers to line up with each other for easier reading.
            Default is 15. If you have long feature names, you can increase this number to make the integers line up more.
            It can also be set to 0.
        yaxes_tickfont_family : the font for the feature names. Default is Courier New.
        yaxes_tickfont_size : the font size for the feature names. Default is 15.

        Returns
        -------
        plot

        """
        if height is None:
            height = top_n_features * height_per_feature

        # prep the data

        all_importances = self.get_feature_importance()
        n_all_importances = len(all_importances)

        plot_importances_df =\
            all_importances\
            .nlargest(top_n_features)\
            .sort_values()\
            .to_frame('value')\
            .rename_axis('feature')\
            .reset_index()

        if max_scale:
            plot_importances_df['value'] = \
                                plot_importances_df.value.abs() /\
                                plot_importances_df.value.abs().max() * 100

        self.plot_importances_df = plot_importances_df.copy()

        if len(all_importances) < top_n_features:
            title_text = 'All Feature Importances'
        else:
            title_text = f'Top {top_n_features} (of {n_all_importances}) Feature Importances'

        if rank_features:
            padded_features = \
                plot_importances_df.feature\
                .str.pad(width=str_pad_width)\
                .values

            ranked_features =\
                plot_importances_df.index\
                .to_series()\
                .sort_values(ascending=False)\
                .add(1)\
                .astype(str)\
                .str.cat(padded_features, sep='. ')\
                .values

            plot_importances_df['feature'] = ranked_features

        if display_imp_values:
            text = plot_importances_df.value.round(display_imp_value_decimals)
        else:
            text = None

        # create the plot

        fig = px.bar(plot_importances_df,
                     x='value',
                     y='feature',
                     orientation=orientation,
                     width=width,
                     height=height,
                     text=text)
        fig.update_layout(title_text=title_text, title_x=0.5)
        fig.update(layout_showlegend=False)
        fig.update_yaxes(tickfont=dict(family=yaxes_tickfont_family,
                                       size=yaxes_tickfont_size),
                         title='')
        fig.show()


def ds_print(*args, verbose=True):
    if verbose: print(*args)

class RunModel():

    """

        Parameters
        ----------
        algorithm : A class object that must contain fit and transform attributes

        X_train : The training data - a pandas dataframe or array-like, sparse matrix of shape (n_samples, n_features)
                  The input samples datatypes will be converted to ``dtype=np.float32`` if convert_float32 == True
                  Note many algorithms already do this internally.

        y_train : a pandas series or numpy array containing the target variable data

        X_test : The testing data - a pandas dataframe or array-like, sparse matrix of shape (n_samples, n_features)
                 The input samples datatypes will be converted to ``dtype=np.float32``
                 In production we use this as df_full (all data consisting of train, val, and test)

        features : A list of training features used to train the model.
                   All feature datatypes in X_train must be numeric

        seed : integer random seed
        convert_float32 : boolean to convert X_train, X_val (if supplied), and X_test to datatype float32
        bypass_all_numeric : boolean - set to True if you want to bypass the error that should normally get thrown if all variables are not numeric
        df_full : pandas df - the original "full" dataframe that includes train, val, and test - NOT transformed
        map_predictions_to_df_full : boolean whether to map model predictions onto df_full

        NOTE: If map_predictions_to_df_full == True, then df_full needs to be the original full
        pandas df that consists of train, val, and test data, and X_test needs to be df_full_transformed

        ****** To use a validation set during training you need to pass fit_params ******

        -------

        Attributes
        To get train_model attributes see the documenation for which algorithm you choose
        predict_model are pandas dataframe attributes

        run_everything attributes are dictionary attributes
        -------
        Returns

        train_model returns a model object
        test_model returns a dataframe
        run_everything returns a dictionary

        -------

        """

    def __init__(self,
                 features,
                 X_test=None,
                 X_train=None,
                 y_train=None,
                 algorithm=None,
                 eval_set=None,
                 copy=True,
                 prediction_colname='prediction',
                 seed=100,
                 convert_float32=True,
                 bypass_all_numeric=False,
                 df_full=None,
                 map_predictions_to_df_full=True,
                 predict_proba=False,
                 use_eval_set_when_possible=True,
                 **kwargs):

        self.features = features
        self.X_test = X_test
        self.X_train = X_train
        self.y_train = y_train
        self.algorithm = algorithm
        self.eval_set = eval_set
        self.use_eval_set_when_possible = use_eval_set_when_possible
        self.prediction_colname = prediction_colname
        self.seed = seed
        self.convert_float32 = convert_float32
        self.copy = copy
        self.bypass_all_numeric = bypass_all_numeric
        self.df_full = df_full
        self.map_predictions_to_df_full = map_predictions_to_df_full
        self.predict_proba = predict_proba
        self.kwargs = kwargs


        if self.copy:
            self.X_train, self.X_test = self.X_train.copy(), self.X_test.copy()
            self.y_train = self.y_train.copy()

        if self.convert_float32:
            # self.X_train[self.features] = parallize_pandas_func(self.X_train[self.features], 'astype', dtype='float32', copy=self.copy)
            # self.X_test[self.features] = parallize_pandas_func(self.X_test[self.features], 'astype', dtype='float32', copy=self.copy)
            self.X_train[self.features] = self.X_train[self.features].astype('float32')
            self.X_test[self.features] = self.X_test[self.features].astype('float32')

        if (self.use_eval_set_when_possible) and (self.eval_set is not None):
            X_val_tmp, y_val_tmp = self.eval_set[0][0], self.eval_set[0][1]

            if self.copy:
                X_val_tmp, y_val_tmp = X_val_tmp.copy(), y_val_tmp.copy()

            if self.convert_float32:

                # self.X_val[self.features] = parallize_pandas_func(self.X_val[self.features], 'astype', dtype='float32', copy=self.copy)
                X_val_tmp[self.features] = X_val_tmp[self.features].astype('float32')

            self.eval_set = [(X_val_tmp, y_val_tmp)]
            del X_val_tmp, y_val_tmp

    def train_model(self):

        np.random.seed(self.seed)

        assert all(f in self.X_train.columns for f in self.features), 'Missing features in X_train!'

        if not self.bypass_all_numeric: # need to check all features are numeric
            assert len(list(set([i for i in self.X_train.select_dtypes(include = np.number).columns if i in self.features]))) == len(self.features), 'Not all features in X_train are numeric!'

            if (self.use_eval_set_when_possible) and (self.eval_set is not None):

                assert all(f in self.eval_set[0][0].columns for f in self.features), 'Missing features in X_val!'
                assert len(list(set([i for i in self.eval_set[0][0].select_dtypes(include = np.number).columns if i in self.features]))) == len(self.features), 'Not all features in X_val are numeric!'

        assert self.X_train.shape[0] == len(self.y_train),'X_train shape does not match y_train shape!'

        ds_print('\nRunning ' + type(self.algorithm).__name__ + '...\n')

        if ('fit_params' in self.kwargs.keys()) and \
            (set(list(self.kwargs['fit_params'].keys()) + ['eval_set'])).issubset(list(inspect.signature(self.algorithm.fit).parameters)) and \
            (len(self.kwargs['fit_params']) > 0):

            # All kwargs in the fit_params are available in the model.fit object
            # (e.g. list(inspect.getfullargspec(RandomForestClassifier.fit))[0] must have all params inside fit_params)

            ds_print('\nUsing a validation set for ' + type(self.algorithm).__name__ + '...\n')

            if self.use_eval_set_when_possible and self.eval_set is not None:
                model = self.algorithm.fit(self.X_train, self.y_train, eval_set = self.eval_set, **self.kwargs['fit_params'])
            else:
                model = self.algorithm.fit(self.X_train, self.y_train, **self.kwargs['fit_params'])

        elif self.use_eval_set_when_possible and self.eval_set is not None and \
                {'eval_set'}.issubset(list(inspect.signature(XGBRegressor.fit).parameters)):

            ds_print('\nUsing a validation set for ' + type(self.algorithm).__name__ + '...\n')
            model = self.algorithm.fit(self.X_train, self.y_train, eval_set=self.eval_set)

        else:
            ds_print('\nNot using an eval_set for ' + type(self.algorithm).__name__ + '...\n')
            model = self.algorithm.fit(self.X_train, self.y_train)
            ds_print("\n" + type(model).__name__ + " training done!\n")

        return model


    def predict_model(self, model):

        """

        Parameters
        ----------

        model : a trained model object

        X_test : a pandas dataframe or np.array-like object to perform predictions on

        """

        if ('predict_params' in self.kwargs) and (len(self.kwargs['predict_params']) > 0):

            if not self.predict_proba:
                predictions = model.predict(self.X_test[self.features], **self.kwargs['predict_params'])
            else:
                try:
                    predictions = model.predict_proba(self.X_test[self.features], **self.kwargs['predict_params'])[:, 1]
                except:
                    try:
                        predictions = model.decision_function(self.X_test[self.features], **self.kwargs['predict_params'])[:, 1]
                    except:
                        sys.exit('model does not have predict_proba or decision_function attribute')
        else:

            if not self.predict_proba:
                predictions = model.predict(self.X_test[self.features])
            else:
                try:
                    predictions = model.predict_proba(self.X_test[self.features])[:, 1]
                except:
                    try:
                        predictions = model.decision_function(self.X_test[self.features])[:, 1]
                    except:
                        ds_print('model does not have predict_proba or decision_function attribute')
                        sys.exit()


        if not self.map_predictions_to_df_full:
            ds_print('Returning only X_test to df_pred in model_dict')
            assert len(predictions) == self.X_test.shape[0]
            self.X_test[self.prediction_colname] = predictions
            return self.X_test
        else:
            if self.copy: self.df_full = self.df_full.copy()
            ds_print('Returning df_full to df_pred in model_dict')
            assert self.df_full is not None, 'df_full is None'
            assert len(predictions) == self.df_full.shape[0]
            self.df_full[self.prediction_colname] = predictions
            return self.df_full


    def train_and_predict(self):

        run_model_dict = dict()
        run_model_dict['model'] = self.train_model()
        run_model_dict['df_pred'] = self.predict_model(model=run_model_dict['model'])

        return run_model_dict

class Resampler():

    """
    Parameters
    ----------
    algorithm: The resampling algorithm to use

    In fit params:
    X: The training dataframe that does not include the target variable
    y: A pandas Series of the target variable
    -------
    Attributes
    See imblearn documentation for the attributes of whichever resampling algorithm you choose
    -------
    Returns a resampled pandas dataframe
    -------

    """

    def __init__(self, algorithm):
        self.algorithm = algorithm

    def fit_transform(self, X, y):
        ds_print('Running {}'.format(type(self.algorithm).__name__))
        df_resampled = self.algorithm.fit_resample(X, y)
        return df_resampled

def timeseries_split(df,
                     datetime_col='date',
                     split_by_datetime_col=True,
                     train_prop=0.7,
                     val_prop=0.15,
                     target_name='target',
                     return_single_df=True,
                     split_colname='dataset_split',
                     sort_df_params={},
                     reset_datetime_index=True,
                     copy=True):

    """

    Get the column names from the a ColumnTransformer containing transformers & pipelines

    Parameters
    ----------
    df: a pandas (compatible) dataframe
    datetime_col: str date column to run the timeseries split on
    train_prop: float - proportion of train orders assuming the data is sorted in ascending order
        example: 0.7 means 70% of the df will train orders
    val_prop: float - proportion of val orders
        e.g. 0.1 means 10% of the df will be val orders, where the val orders come after the train orders
    target_name: str - name of the target variable
    return_single_df: If true then a new column <split_colname> will be concatenated to the df with 'train', 'val', or 'test'
    split_colname: If return_single_df is True, then this is the name of the new split col
    sort_df_params: Set to False or empty dict to not sort the df by any column.
        To sort, specify as dict the input params to either df.sort_index or df.sort_values.
    reset_datetime_index: Bool to reset the datetime index if splitting by datetime_col

    Returns
    -------

    Either a tuple of dataframes: X_train, y_train, X_val, y_val, X_test, y_test
      Or the same df with a new <split_colname> having ['train', 'val', 'test'] or 'None' populated

    """

    if copy: df = df.copy()

    if len(df.index) != df.index[-1] - df.index[0]:
        df.reset_index(inplace=True)

    nrows = len(df)

    if sort_df_params:
        if list(sort_df_params.keys())[0].lower() == 'index' and 'index' not in df.columns:
            df.sort_index(**sort_df_params)
        else:
            df.sort_values(**sort_df_params)

    elif datetime_col in df.columns:
        df.set_index(datetime_col, inplace=True)
        if reset_datetime_index:
            df.reset_index(datetime_col, inplace=True)

    if 'index' in df.columns:
        df.drop('index', axis=1, inplace=True)

    if return_single_df:

        if split_by_datetime_col:
            # max_train_date = df.iloc[0:int(np.floor(nrows*train_prop))][datetime_col].max()
            # max_val_date = df.iloc[int(np.floor(nrows*train_prop)): int(np.floor(nrows*(1 - val_prop)))][datetime_col].max()

            dates = np.unique(df['date'].sort_values())
            max_train_date = dates[int(len(dates) * train_prop)]
            max_val_date = dates[int(len(dates) * (1 - val_prop))]

            df.loc[df[datetime_col] < max_train_date, split_colname] = 'train'
            df.loc[(df[datetime_col] >= max_train_date) & (df[datetime_col] < max_val_date), split_colname] = 'val'
            df.loc[df[datetime_col] >= max_val_date, split_colname] = 'test'

        else:
            df.loc[0:int(np.floor(nrows*train_prop)), split_colname] = 'train'
            df.loc[int(np.floor(nrows*train_prop)): int(np.floor(nrows*(1 - val_prop))), split_colname] = 'val'
            df.loc[int(np.floor(nrows*(1 - val_prop))):, split_colname] = 'test'

        return df

    else:

        X_train = df.iloc[0:int(np.floor(nrows*train_prop))][feature_list]
        y_train = df.iloc[0:int(np.floor(nrows*train_prop))][target_name]

        X_val = df.iloc[int(np.floor(nrows*train_prop)):int(np.floor(nrows*(1 - val_prop)))][feature_list]
        y_val = df.iloc[int(np.floor(nrows*train_prop)):int(np.floor(nrows*(1 - val_prop)))][target_name]

        X_test = df.iloc[int(np.floor(nrows*(1 - val_prop))):][feature_list]
        y_test = df.iloc[int(np.floor(nrows*(1 - val_prop))):][target_name]

        return X_train, y_train, X_val, y_val, X_test, y_test


def merge_dicts(*dict_args):
    """
    Given any number of dictionaries, shallow copy and merge into a new dict,
    precedence goes to key-value pairs in latter dictionaries.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result
