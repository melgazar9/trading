######################## Summary Functions ########################

get_na_summary <- function(DT) {
  
  #' @title Get a Summary of the NAs by Column
  #' 
  #' @param DT a data.table or data.frame
  #' 
  #' @return a data.table, it will be an empty data.table if no NAs are found at all
  
  if (!data.table::is.data.table(DT)) data.table::setDT(DT)
  
  n_train_rows <- DT[, .N]
  
  is_numeric_feature <- DT %>% sapply(., is.numeric) %>% unname
  na_counts <- sapply(DT, function(x) x %>% is.na %>% sum) # count the NAs
  
  if (length(na_counts) == 0) {
    ds_print("\nNo NAs found\n")
    return(data.table())
    } else {
      
    na_dt_train <- 
      na_counts %>% 
      stack() %>%                                     # stack vector into data.frame
      dplyr::transmute(                               # rename the columns, calculate the NA pct
        feature = as.character(ind),                  # & drop the original columns
        is_numeric_feature = is_numeric_feature,
        n_NAs = values,
        pct_NAs = round(values / n_train_rows * 100, 2)
      ) %>%
      dplyr::filter(n_NAs > 0) %>%                    # filter to the features containing NAs
      dplyr::arrange(-n_NAs) %>%                      # sort descendingly
      as.data.table()
    
    return(na_dt_train)
  }
}


get_categorical_feature_summary <- function(DT) {
  #' @title Get a Summary of the Unique Number of Categorical Values - 1
  #' 
  #' @param DT a data.table or data.frame
  #' 
  #' @return a data.table, it will be an empty data.table if no nominal features are found at all
  
  if (!data.table::is.data.table(DT)) data.table::setDT(DT)
  nominal_features <- sapply(DT, function(x) is.character(x) | is.factor(x)) %>% names
  
  if (length(nominal_features) == 0) {
    
    ds_print("\nNo nominal features found\n")
    return(data.table())
    
  } else {
    
    categorical_feature_summary <-
      DT[, ..nominal_features] %>% # filter to all_nominal() features
      sapply(., function(x) data.table::uniqueN(x) - 1) %>% # count the unique vals in each feature
      stack() %>%                                     # stack vector into data.frame
      dplyr::transmute(                               # rename the columns
        feature = as.character(ind),                  # drop the original columns
        n_unique_vals_minus1 = values
      ) %>%
      dplyr::arrange(-n_unique_vals_minus1) %>%       # sort descending
      janitor::adorn_totals() %>%
      data.table::as.data.table()
    
    return(categorical_feature_summary)
    
  }
}


######################## Selector Functions ########################

all_numeric_predictors <- function() {
  
  #' @title Numeric Predictor Wrapper Function for the Recipe Selectors
  #' @description It only works inside of a recipe step
  #' @return Integer vector of column positions
  
  intersect(recipes::all_numeric(), recipes::all_predictors())
  
}


all_nominal_predictors <- function() {  
  
  #' @title Nominal Predictor Wrapper Function for the Recipe Selectors
  #' @description It only works inside of a recipe step
  #' @return Integer vector of column positions

  intersect(recipes::all_nominal(), recipes::all_predictors())
  
}


######################## Feature Engineering ########################

remove_na_rows <- function(dt_full, input_features, split_colname = 'dataset_split'){
  
  #' @title Removes the Rows Containing an NA for the Model's Features
  #' 
  #' @description We don't need to impute these rows. 
  #' Our prod system actually uses these missing values in order to determine which model to call
  #' @called_by run_model
  
  #' @param dt_full a data.table
  #' @param input_features the character vector of features from config file
  #' 
  #' @return a data.frame or data.table w/o the NAs
  
  if (!data.table::is.data.table(dt_full)) data.table::setDT(dt_full)
  
  # NA statistics
  ds_print("\nFor diagnostic purposes, let's see which features still have NAs & how many by dataset split\n")
  
  na_dt_full <- dt_full[, get_na_summary(.SD), by = split_colname, .SDcols = input_features]
  
  
  if (na_dt_full[, .N]) {ds_print("\nThese features have NAs:\n"); ds_print(na_dt_full, '\n', cat = F)} else ds_print("No NAs")
  
  good_rows <- dt_full[, ..input_features] %>% is.na %>% rowSums %>% {. == 0} 
  
  ds_print("\nKeeping ", sum(good_rows), " rows out of ", dt_full[, .N], ", ", sum(good_rows) / dt_full[, .N] * 100, "%\n", sep = '')
  
  return(dt_full[good_rows])
  
}

make_dummy_names <- function(var, lvl, ordinal = FALSE, sep = "_") {
  
  #' @title This function is used within step_dummy to create the dummy-encoded column names.
  #' step_dummy default is to use recipes::dummy_names(), which reassigns variable names with base::make.names
  #' for example, var at level 1 gets renamed to to var_X1 with dummy_names(),
  #' but this function renames the step_dummy variables from var_X1 to var_1 (or whatever you specify as a sep)
  #' recipes::names0 creates a series of num names with a common prefix, e.g. var01-var10 instead of var1-var10
  #' @param var the variable name
  #' @param lvl the level within each variable
  #' @param ordinal A logical; was the original factor ordered?
  #' @param sep A single character value for the separator between the names and levels.
  #' 
  #' @return the dummified feature names
  
  dummy_nms <- 
    recipes::dummy_names(var, lvl, ordinal = ordinal, sep = sep) %>% 
    stringr::str_replace(., '_X', sep) %>%
    make.unique(.)
  
  return(dummy_nms)
}


prep_features <- function(dt_full,
                          dt_train,
                          dep_var,
                          input_features,
                          FE_helper_vars = c('placedat', 'firstAppeared'),
                          preserve_vars = NULL,
                          skip_custom_step = F,
                          custom_mutate_step = NULL,
                          
                          skip_scale = F,
                          scaler = 'min_max_scaler',
                          min_max_range = c(-1, 1),
                          skip_missing = F,
                          skip_median_impute = F,
                          
                          skip_novel = F,
                          skip_mode_impute = F,
                          skip_unknown = F,
                          step_unknown_value = "unknown_value",
                          skip_dummy = F,
                          one_hot = F,
                          dummy_naming_fn = make_dummy_names,
                          max_oh_size = 3,
                          skip_rename_at = F,
                          skip_ordinalscore = F,
                          retain_original_dummies = c(),
                          
                          skip_step_other = F,
                          step_other_threshold = .05,
                          
                          skip_zv = F,
                          skip_corr = F,
                          corr_threshold = 0.99,
                          skip_lincomb = T,
                          skip_pca = T,
                          n_pca_components = NULL,
                          
                          log_changes = T,
                          strings_as_factors = F,
                          make_copy = T) {
  
  #' @title Prep Features Using the Recipes Package: Imputation & Unsupervised Feature Filtering
  #' @description Perform customizeable feature engineering
  #' Below are some example steps. Each step is configurable
  #' create missing flag columns
  #' impute the median for numeric features
  #' impute the mode for categorical features
  #' remove features containing only a single value
  #' remove highly correlated features, etc...
  #'
  #' Recipes Reference:
  #' https://tidymodels.github.io/recipes/reference/#section-step-functions-imputation
  #' https://tidymodels.github.io/recipes/reference/#section-step-functions-filters  
  #'
  #' 
  #' @param dt_full a data.table of features
  #' @param input_features a character vector of features names
  #' @param dep_var the name of the dependent variable
  #' @param FE_helper_vars extra features needed to manually engineer our features with step_mutate
  #' 
  #' @param preserve_vars which variables to keep with the final features. If null, keep all columns
  #' @param skip_scale should we scale or normalize our data?
  #' @param scaler the scaler algorithm to use. Set it as a string. Some options are 'min_max_scaler' and 'standard_scaler'
  #' @param skip_missing should a missing flag be created for each features w/ NAs?
  #' @param skip_median_impute should missing values be imputed with the median for numeric features?
  #' 
  #' @param skip_novel should the recipe assign a previously unseen factor level to a new value?
  #' @param skip_mode_impute should missing values be imputed with the mode for ordinal features?
  #' @param skip_unknown for remaining non-imputed NA's, should we treat them as a separate category?
  #' @param step_unknown_value NA replacement value
  #' @param skip_dummy should features be oh-encoded?
  #' @param one_hot should one-hot encoding be used instead of dummy encoding (e.g keep N features instead of N-1)?
  #' @param dummy_naming_fn this fn controls the way the dummy column names are created
  #' @param max_oh_size only OH encode (or dummy encode) variables with uniqueN <= max_oh_size
  #' @param skip_rename_at If T, then replace the periods w/ underscores in the feature names?
  #'
  #' @param skip_ordinalscore convert ordinal factor variables into numeric scores
  #'        this is done as the last step if variables are not OH-encoded
  #' 
  #' @param skip_zv should features containing a single value be removed?
  #' @param skip_corr should highly correlated features be removed?
  #' @param corr_threshold the correlation threshold used to remove features
  #' @param skip_lincomb should the feature that are linear combinations by removed?
  #' @param skip_pca should we run pca on the dataset?
  #' @param n_pca_components number of pca components (i.e. features) returned when running pca. If null use the floor of 65% of the features
  #' 
  #' @param log_changes should the recipe prepping use verbose printing?
  #' @param strings_as_factors should the recipe cast all character columns to factors?
  #' @param make_copy create a copy of the data.table - if F then the input data.table will be overwritten
  #' 
  #'
  #' @future_functionality_to_consider
  #' Unsupervised feature filtering: step_nzv
  #' More imputations: step_meanimpute, step_knnimpute or step_bagimpute
  #' 
  #' @return list with data table, feature transformer, and new variable names
  #' list2env(prep_features_params, globalenv())
  
  if (make_copy) {
    dt_full <- copy(dt_full)
    dt_train <- copy(dt_train)
    }
  
  ###### validate inputs ######
  
  if (!data.table::is.data.table(dt_full)) data.table::setDT(dt_full)
  if (!data.table::is.data.table(dt_train)) data.table::setDT(dt_train)
  if (!skip_scale) stopifnot(scaler %in% c('normalizer', 'min_max_scaler'))
  if (!skip_mode_impute & ! skip_unknown) warning("skip_mode_impute & skip_unknown are both set to FALSE. However, this is redundant, and you should only run one of these steps")
  stopifnot(all(input_features %in% names(dt_full)))
  
  if (uniqueN(na.omit(dt_full[[dep_var]])) == 2) dt_full[, (dep_var) := as.factor(get(dep_var))] else dt_full[, (dep_var) := as.numeric(get(dep_var))]
  
  
  ###### Get NA feature statistics ######
  
  ds_print("\n\nGet NA feature statistics\n\n")
  na_dt_train <- get_na_summary(dt_train[, ..input_features])
  
  if (na_dt_train[, .N]) {ds_print("\nThese features have NAs:", '\n'); ds_print(na_dt_train, cat = F)}
  
  ###### Get categorical / nominal feature summary ######
  
  ds_print("\n\nGet categorical / nominal feature summary\n\n")
  
  categorical_feature_summary <- get_categorical_feature_summary(dt_train[, ..input_features])
  
  if (categorical_feature_summary[, .N]) {ds_print("\nCategorical feature cardinality:\n", cat = T, '\n'); ds_print(categorical_feature_summary, cat = F)} else ds_print("No categorical variables to encoded")
  
  #####  Create a "recipe" pipeline & "prep" it #####
  
  # a recipe is a set of instructions/steps & prepping the recipe fits it to the training data data
  needed_cols <- unique(c(dep_var, input_features, FE_helper_vars))
  gc()
  feature_transformer <-
    
    ##### 1. Initialize recipe & assign roles #####
  
    dt_train[, ..needed_cols] %>%                              # select all the needed columns
    recipes::recipe(as.formula(paste(dep_var, " ~ .")), data = .) %>%    # initialize a recipe, a set of instructions
    recipes::update_role(., tidyselect::all_of(FE_helper_vars), new_role = "helper") %>%
    recipes::update_role(., tidyselect::all_of(dep_var), new_role = "outcome") %>%
    
    ##### 2. Custom step #####
    
    {if (!skip_custom_step & is.function(custom_mutate_step)) custom_mutate_step(., features = input_features, helper_vars = FE_helper_vars) else .} %>%
    
    ##### 3. Numeric Transformations #####
    # 3a. scale the numeric predictors
    {if (!skip_scale) {
      {if (scaler == 'normalizer') {recipes::step_normalize(., all_numeric_predictors(), skip = skip_scale)} # standard deviation of 1 and mean of 0
      else if (scaler == 'min_max_scaler') {recipes::step_range(., all_numeric_predictors(),
                                                                min = min_max_range[1],
                                                                max = min_max_range[2],
                                                                skip = skip_scale)}} %>%
        step_rename_at(., all_numeric_predictors(), fn = ~paste0(., '_scaled'))
      }
      else .} %>%
    
    
    # 3b. Create _missing_flag columns
    # for the predictors with missing values, create "_missing_flag" columns
    # if only one variable is missing, "missing_flag" will be created
    
    {if (!skip_missing) recipes::step_mutate_at(., all_numeric_predictors() & where(anyNA),
                                                 fn = list(missing_flag = ~as.integer(is.na(.))),
                                                 skip = skip_missing,
                                                 id = rand_id("numeric_missing_flags"))
      else .} %>%
    
    # 3c. Impute the median for NAs
    
    {if (!skip_median_impute) {recipes::step_impute_median(., all_numeric_predictors(), skip = skip_median_impute)} else .} %>%
    
    ##### 4. Categorical / Nomimal Transformations #####
  
    {if (!skip_novel) recipes::step_novel(., all_nominal_predictors(), skip = skip_novel) else .} %>%  # adds "_new" columns to ensure that new values do not cause bake errors
    {if (!skip_mode_impute) recipes::step_impute_mode(., all_nominal_predictors(), skip = skip_mode_impute) else .} %>%       # impute the mode for categorical predictors
    {if (!skip_unknown) recipes::step_unknown(., all_nominal_predictors(), new_level = step_unknown_value, skip = skip_unknown) else .} %>%
    {if (!skip_dummy) {
      recipes::step_string2factor(., function(x) is.character(x) & length(unique(x)) <= 3) %>%
      recipes::step_dummy(.,
                          function(x) is.factor(x) & length(unique(x)) <= max_oh_size,
                          skip = skip_dummy,
                          one_hot = one_hot,
                          naming = dummy_naming_fn,
                          retain = tidyselect::all_of(retain_original_dummies)
                          )
      } else .} %>%
    
    {if (!skip_step_other) recipes::step_other(., all_nominal_predictors(), skip = skip_step_other, threshold = step_other_threshold)} %>%
    
    {if (!skip_ordinalscore) {
      # recipes refuses to make a label-encoder, so below is a workaround to label-encode everything that is not oh-encoded (as the last major step)
      {
        recipes::step_factor2string(., all_nominal_predictors()) %>% # the step_ordinalscore recipe will fail without this step
        recipes::step_string2factor(., all_nominal_predictors(), ordered=T) %>% # the step_ordinalscore recipe will fail without this step
        recipes::step_ordinalscore(., all_nominal_predictors())
        }
      } else .} %>%
    {if (!skip_rename_at) recipes::step_rename_at(., recipes::all_predictors(), fn = ~stringr::str_replace_all(., "\\.", "_")) else .} %>%
    
    ##### 5. Feature Filtering #####
    
    # all predictors are numeric by this point
    {if (!skip_zv) recipes::step_zv(., all_predictors(), skip = skip_zv) else .} %>%                   # remove single-value predictors
    {if (!skip_corr) recipes::step_corr(., all_predictors(), threshold = corr_threshold, skip = skip_corr) else .} %>% # remove correlated predictors
    {if (!skip_lincomb) recipes::step_lincomb(., all_predictors(), skip = skip_lincomb) else .} %>%    # default is skip=TRUE
    {if (!skip_pca) {
      if (skip_scale) {ds_print('Warning - you are running PCA without centering!')}
      if (is.null(n_pca_components)) {
        n_pca_components <- floor(ncol(dt_train) * 0.65)
        ds_print(paste0('Note: n_pca_components is NULL - setting n_pca components to ', n_pca_components, ' components, which is the floor of 65% of the features'))
      }
      recipes::step_pca(., all_numeric_predictors(), num_comp = n_pca_components, skip = skip_pca) # all_predictors should be all_numeric_predictors at this point
      } else .} %>%
    prep(log_changes = log_changes, strings_as_factors = strings_as_factors)
  
  gc()
  
  ##### transform / bake the train, test and validation features #####

  dt_full_transformed <- recipes::bake(feature_transformer, new_data = dt_full)
  transformed_feature_cols <- names(dt_full_transformed) # includes more than just the predictor columns
  data.table::setDT(dt_full_transformed)

  # store the final features in dt_full
  dt_full[, (transformed_feature_cols) := dt_full_transformed]
  
  # if preserve_vars is not NULL, remove all unused columns (e.g. keep only the final_features, preserve_vars, dep_var, and FE_helper_vars)
  if (!is.null(preserve_vars)) {
    keep_cols <- unique(c(transformed_feature_cols, preserve_vars, FE_helper_vars, dep_var))
    drop_cols <- setdiff(names(dt_full), keep_cols)
    dt_full <- dt_full[, keep_cols, with = F]
  }
  
  #####  create fit summary  #####
  
  tryCatch({
    
    ds_print("\n\nHere is the fit summary:\n"); ds_print(data.table::as.data.table(tidy(feature_transformer)), cat = F)
    
    removal_summary_dt <-
      feature_transformer$steps %>%
      lapply(., function(x) x %>% .[c("id", "removals")] %>% data.table::as.data.table()) %>%
      data.table::rbindlist(., fill = T) %>%
      na.omit
    ds_print("\n\nHere are the removed features:\n"); ds_print(removal_summary_dt, cat = F)
    
  }, error = function(e) warning("\n\nCouldn't print all or part of the fit summary\n\n"))
  
  ##### get only the predictors from the recipe object #####
  
  final_features <-
    feature_transformer$term_info %>% 
    dplyr::filter(role == "predictor") %>% 
    dplyr::pull(variable)
  
  
  ##### summary of features changes #####
  
  orig_features_removed <- setdiff(input_features, final_features)
  net_new_features <- setdiff(final_features, input_features)
  
  if (length(orig_features_removed)) ds_print("\n\nThese original features were removed OR replaced:\n", paste(orig_features_removed, collapse = "\n")) else ds_print("\nNo original features removed\n")
  if (length(net_new_features)) ds_print("\n\nThese are the net new features:\n", paste(net_new_features, collapse = "\n")) else ds_print("\nNo net new features\n")
  if (length(FE_helper_vars)) ds_print('\n\nThese are the helper vars: ', FE_helper_vars, '\n')
  
  
  ##### Return a list with each object #####
  
  outlist <- list(
    final_features = final_features,
    feature_transformer = feature_transformer,
    dt_full = dt_full,
    drop_cols = {if (!is.null(preserve_vars)) drop_cols else NULL})
  
  if (exists("removal_summary_dt")) outlist$removal_summary_dt <- removal_summary_dt
  
  return(outlist)
  
}


######################################  Standardize Zip Codes ###################################### 


standardize_zip_codes <- function(char_vec) {
  
  #' @title Standardize US & CA Postal Codes
  #' @description All US codes are changed to the 5-digit format.
  #' All CA codes are changed to a standard 6-character format.  
  #'  
  #' @param char_vec a character vector of billing or shipping zip codes
  #' 
  #' @return a standardized character vector
  
  stopifnot(is.character(char_vec) | is.factor(char_vec))
  
  # The regular expression for CA codes came from the book below. 
  # "Canadian postal codes simply use an alternating sequence of six alphanumeric characters with an optional space in the middle"
  # https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch04s15.html
  ca_code_regex <- "^(?!.*[DFIOQU])[A-VXY][0-9][A-Z] ?[0-9][A-Z][0-9]$"
  
  char_vec %>% 
    stringr::str_replace_all("\\W", "") %>%         # replace all non-alphanumeric characters with ""
    stringr::str_to_upper() %>%                     # CA codes should be uppercase
    {ifelse(stringr::str_detect(., ca_code_regex),  # is this a CA code?  
            .,                                      # if yes, pass it through                       
            stringr::str_sub(., 1, 5))}             # else truncate it to 5 characters
  
}



######################################  Resample function ###################################### 

resample <- function(dt_train, resample_algorithm, resample_col, training_features, resample_ratio, n_neighbors, dep_var, random_seed = 100, ...) {
  
  #' @title oversample or undersample a training dataset.
  #' @description Supply input features, the col to resample, and other hyperparameters within the recipe / themis call
  #' 
  #' @param resample_algorithm a string of the algorithm to set - some options are 'smote', 'upsample', 'undersample'
  #' @param resample_col the column you want to resample
  #' @param training_features the input features to your model (e.g. input_features)
  #' @param resample_ratio A numeric value for the ratio of the majority-to-minority frequencies. All other levels are sampled up to have the same frequency as the most occurring level. 
  #' A value of 0.5 would mean that the minority levels will have (at most / approximately) half as many rows than the majority level.
  #' @param random_seed set the random seed
  #' @param ... NEEDS WORK - for other hyperparameters of the step_function since the parameter names in each function are not always the same.
  #' Some common hyperparameters of the step_functions are n_neighbors, resample_ratio
  #' @calls call this function before before run_model to oversample the training data
  #' @return a resampled data.table
  
  # extra_params <-
  #   match.call() %>%
  #   as.list() %>%
  #   .[2:length(.)]
  
  setDT(dt_train)
  
  training_features <- unique(c(training_features, dep_var))
  
  stopifnot(all(training_features %in% names(dt_train)))
  
  if (!class(dt_train[[resample_col]]) == 'factor') {
    dt_train[[resample_col]] <- as.factor(dt_train[[resample_col]])
  }
  
  if (any(is.na(dt_train[[resample_col]]))) {
    dt_train[[resample_col]] <- ifelse(is.na(dt_train[[resample_col]]), 0, dt_train[[resample_col]])
  }
  
  resample_formula <- as.formula(paste0(resample_col, "~", paste(training_features[!training_features %in% resample_col], collapse= "+")))
  
  
  dt_resampled <-
    
    recipe(resample_formula, data = dt_train) %>%
    
    {if (tolower(resample_algorithm) == 'upsample') {themis::step_upsample(., tidyselect::all_of(resample_col), over_ratio = resample_ratio, skip = F, seed = random_seed)}
      else if (tolower(resample_algorithm) == 'downsample') {themis::step_downsample(., tidyselect::all_of(resample_col), under_ratio = resample_ratio, skip = F, seed = random_seed)}
      else if (tolower(resample_algorithm) == 'smote') {themis::step_smote(., tidyselect::all_of(resample_col), over_ratio = resample_ratio, skip = F, 
                                                                           seed = random_seed, neighbors = n_neighbors)}
      # else if (tolower(resample_algorithm) == 'rose') {themis::step_rose(., resample_col, over_ratio = resample_ratio, skip = F, 
      #                                                                    seed = random_seed)}
      else .} %>%
    prep() %>%
    juice() %>%
    as.data.table()
  
  ds_print('\ninput shape:', toString(dim(dt_train)))
  ds_print('\noutput shape:', toString(dim(dt_resampled)))
  ds_print('\nDone!\n')
  
  return(dt_resampled)
  
}