run_sql <- function(sqlQuery, user = Sys.getenv("USER")) {
  
  #' @description Runs provided SQL query in mysql on local server and returns results
  #' in a data.table
  #' 
  #' @author Kunal Bhandari
  #' 
  #' @import RMySQL
  #' @references None
  #' 
  #' @param sqlQuery Character string containing SQL query to be run in mysql
  #' Can also be DDLs for creating / deleting tables
  #' @param user Which user should be used for running the SQL (Should have permissions)
  #' 
  #' @return A data.table with the results of the SQL query passed
  
  suppressMessages({
    if(! "RMySQL" %in% installed.packages()) install.packages("RMySQL", depend = TRUE)
    library("RMySQL")
  })
  
  #Connect to database
  try({
    lapply(dbListConnections(MySQL()), dbDisconnect)
  }, silent = T)
  
  w <- dbConnect(MySQL(), db = user)
  
  bad_sql_type <- grepl("\\bSHOW\\b", sqlQuery, ignore.case = T) ||
    grepl("\\bDROP\\b", sqlQuery, ignore.case = T)
  
  ds_print("sqlDB: Running below sql")
  ds_print(sqlQuery)
  
  # Run SQL query
  start_time <- Sys.time()
  ds_print(paste0("sqlDB: Start time [",start_time,"]"))
  tryCatch( {
    dbExecute(w, "SET group_concat_max_len = 1000000;")
    suppressWarnings({
      sqlData <- dbGetQuery(w, sqlQuery)
    })
  },
  error = function(e) stop(e),
  
  finally = {
    
    try({
      dbDisconnect(w)
    }, silent = T)
    rm(w)
    
    #Close DB connections
    try({
      lapply(dbListConnections(MySQL()), dbDisconnect)
    }, silent = T)
    
    #Unload library that interferes with sqldf library
    try({
      detach("package:RMySQL", unload=TRUE)
    })
  }
  )
  
  
  ds_print(sprintf("sqlDB: %d Rows", nrow(sqlData)))
  time_to_run <- Sys.time() - start_time
  ds_print(paste0("sqlDB: Time to run query [",round(time_to_run,2), " ",attr(time_to_run,"units"),"]"))
  setDT(sqlData)
  return(sqlData)
}



train_test_split <- function(dt_full,
                             n_test_days = 30,
                             n_val_days = 30,
                             split_colname = 'dataset_split',
                             reference_date_col = "datetime",
                             matured_flag_to_use = "matured",
                             max_dt_col = 'placedat',
                             traintime = NULL) { 
  
  #' @title Partitions dt_full into train, test & validation datasets
  #'
  #' @description In June 2020, we changed the way this train_test_split functions.
  #' Previously, the input parameters where date ranges. Now, the main parameters are the number of recent matured days to use. 
  #' A more limited training date range can be specified with traintime
  #' @called_by run_fraud_modelling
  #'
  #' @param dt_full a data.frame or data.table of features & business metrics
  #' @param n_test_days the number of days to subtract from the maximum matured date
  #' @param n_val_days the number of days to subtract from the minimum test period date. To use no validation set, set to 0
  #' @param reference_date_col the name of the date column that should be used. Ideally, it should be the one that is used to create matured_flag_to_use
  #' @param matured_flag_to_use either dep_var_matured and profitability_matured. They are columns in the base_table. see DATA-204 for definition
  #' @param traintime a character/date vector of 2 dates. 
  #'
  #' @return data.table w/ train (0,1) & test (0,1) columns, validation too if specified
  #' list2env(train_test_split_params, globalenv())
  
  # default::default(ds_print) <- list(verbose = True)
  
  ds_print("\nSplitting the data using the ", reference_date_col)
  ds_print("\nn_test_days: ", n_test_days)
  ds_print("\nn_val_days: ", n_val_days)
  ds_print("\ntraintime: ", traintime, collapse = " & ")
  
  stopifnot(reference_date_col %in% names(dt_full))
  stopifnot(matured_flag_to_use %in% names(dt_full))
  
  if (!is.data.table(dt_full)) data.table::setDT(dt_full)
  
  ###### cast reference_date_col as "POSIXt" datetime class if necessary.   ######  
  
  if (!is.POSIXt(dt_full[[reference_date_col]])) {
    
    orig_n_NAs <- dt_full[[reference_date_col]] %>% is.na %>% sum
    dt_full[, c(reference_date_col) := lubridate::as_datetime(get(reference_date_col))]
    new_n_NAs <- dt_full[[reference_date_col]] %>% is.na %>% sum
    stopifnot(orig_n_NAs == new_n_NAs) # make sure it doesn't produce new NAs
    
  }
  
  
  ###### determine the test time period ###### 
  
  max_test_date <- dt_full[get(matured_flag_to_use) == 1 & get(reference_date_col) <= max(get(max_dt_col), na.rm = T), max(get(reference_date_col), na.rm = T)]
  min_test_date <- max_test_date - lubridate::days(n_test_days)
  orig_timezone <- attr(max_test_date, "tzone") # R changes the timezone when you combine datetime vectors
  
  # store the split dates for later reference
  data_splitting_dates <- c(min_test_date, max_test_date)
  
  attr(data_splitting_dates, "tzone") <- orig_timezone
  
  # create column
  dt_full[, (split_colname) := 
            ifelse(data.table::between(get(reference_date_col), min_test_date, max_test_date) &
                     get(matured_flag_to_use) == 1,
                   'test', 'None')]
  
  if (n_val_days > 0) {
    
    ######  determine the validation time period if provided  ###### 
    max_val_date <- min_test_date - lubridate::seconds(1)
    min_val_date <- max_val_date - lubridate::days(n_val_days)
    
    data_splitting_dates <- append(data_splitting_dates, c(min_val_date, max_val_date))
    attr(data_splitting_dates, "tzone") <- orig_timezone
    
    # create column
    dt_full[, (split_colname) := 
              ifelse(data.table::between(get(reference_date_col), min_val_date, max_val_date) &
                       get(matured_flag_to_use) == 1,
                     'val', get(split_colname))]
  }
  
  ###### determine the training time period   ###### 
  if (is.null(traintime)) {
    
    # if traintime is NULL, use all other matured orders  
    min_train_date <- dt_full[get(matured_flag_to_use) == 1, min(get(reference_date_col), na.rm = T)]
    max_train_date <- min(data_splitting_dates, na.rm = T) - lubridate::seconds(1)
    
  } else {
    
    stopifnot(length(traintime) == 2)
    
    traintime <- lubridate::as_datetime(traintime)
    min_train_date <- traintime[1]
    max_train_date <- traintime[2]
    
    
    if(max_train_date > min(data_splitting_dates, na.rm = T)) {
      
      warning("Your max training date is greater the min validation/test date.
                It's going to be capped at the min validation/test date")
      
      max_train_date <- min(data_splitting_dates, na.rm = T) - lubridate::seconds(1)
    }
  }
  
  
  # create column
  dt_full[, (split_colname) := 
            ifelse(data.table::between(get(reference_date_col), min_train_date, max_train_date) &
                     get(matured_flag_to_use) == 1,
                   'train', get(split_colname))]
  
  
  ###### summary of the data splitting   ######
  
  ds_print("\nData Split Summary\n")
  
  data_splitting_cols <- base::intersect(c(matured_flag_to_use, split_colname), names(dt_full)) # get the need columns
  
  if (length(data_splitting_cols)) {
    
    dt_full[, c(reference_date_col, data_splitting_cols), with = F                         # select the needed cols
            ][, 
              c(reference_date_col) := get(reference_date_col) %>% lubridate::as_datetime() # cast to date for easier reading
              ][,
                .(min_date = get(reference_date_col) %>% min(., na.rm = T),             # aggregation by group of columns
                  max_date = get(reference_date_col) %>% max(., na.rm = T),
                  n_distinct_dates = get(reference_date_col) %>% as.Date %>% uniqueN,
                  .N), 
                keyby = data_splitting_cols
                ][, (data_splitting_cols) := lapply(.SD, as.character),  # cast to characters so adorn_totals doesn't add them
                  .SDcols = data_splitting_cols 
                  ][, pct_N := N / sum(N) * 100                          # add pct_N
                    ][order(min_date)
                      ] %>% 
      janitor::adorn_totals() %>%                                            # total the numeric columns
      as.data.table %>%
      print(., digits=2, big.mark=",")
  }
  
  return(dt_full)
}


train_model <- function(algorithm, train_data, target, features, matrix_needed_algos = c('xgboost', 'glmnet', 'glm'), seed = 100) {
  
  #' @description train a model with the parsnip framework
  #' @param algorithm must be a parsnip object that can be called with fit or fit_xy
  #' @param train_data the training data to train the model on
  #' @param target the dependent variable
  #' @param features input features to train the model with
  #' @param matrix_needed_algos algorithms that require X_train to be a data.matrix data type
  #' @param seed set the random seed
  
  data.table::setDT(train_data)
  
  ##### validations #####
  
  stopifnot(!anyNA(train_data[, c(features, target), with = F])) # no NAs at this point
  stopifnot(length(intersect(features, target)) == 0) # the target should never be a feature
  stopifnot(all(c(target, features) %in% names(train_data)))
  stopifnot((algorithm$mode == "classification" && is.factor(train_data[[target]])) | (algorithm$mode == "regression" && is.numeric(train_data[[target]])))
  stopifnot(all(sapply(train_data[, ..features], is.numeric))) # needs to be all numeric

  ##### train #####
  
  set.seed(seed)
  model <- parsnip::fit_xy(algorithm,
                           x = {if (algorithm$engine %in% matrix_needed_algos) data.matrix(train_data[, ..features]) else train_data[, ..features]},
                           y = train_data[[target]])
  
  return(model)
}

ds_print <- function(..., cat = T, verbose = T) {
  
  #' @description a custom print function so we don't log print statements while running cronjobs in production
  #' @calls this function will be called throughout the UW, Fraud, and ds_machine_learning repositories
  #' @param verbose a boolean whether to print the output or not. If F, then do nothing. If T, print the output
  
  if (verbose) {
    if (cat) {
      cat(...)
    } else {
      print(...)
    }
    
  }
  
}



align_states <- function(wd = getwd(), input_lockfile_location = paste0(getwd(), '/renv.lock')) {
  
  #' @title this function will align the R version and libraries with the versions stored in the lock file
  #' 
  #' @description A lock file is a json-based text file generated by renv that stores the R version and library versions.
  #' We can pass the path to the lock file as a parameter (input_lock_file). This file is shared among team members so we're all working in the same R state.
  #' We call renv::hydrate() to install or update any misaligned packages.
  #' Then we restore the R state to its "lock" state by calling renv::restore()
  #' Once you run this function, you will need to restart R, so rstudio can unload the old packages and load the new ones
  #' To see R state history, you can run renv::history(), and revert it to a state by calling renv::revert(commit = '...')
  #' 
  #' @param wd set the working directory so renv can search for the correct lock file
  #' @param input_lockfile_location the location to find the initial input lock file that we want to match our R state to
  #' 
  #' @calls call this function before running another team members code. This function aligns the R version and R libraries used in the input_lockfile_location
  #' 
  #' To install a new package to the existing renv.lock file
  #' 
  #' git branch new_branch
  #' git checkout new_branch
  #' setwd("~/ds_machine_learning")
  #' library(renv)
  #' renv::hydrate()
  #' renv::restore()
  #' renv::install("new_package") # or renv::update("existing_package")
  #' renv::snapshot() 
  #' # check that the new package is in the renv.lock file
  #' # add, commit & push new_branch to origin
  
  if (!('renv' %in% installed.packages())) {install.packages('renv')}
  library('renv')
  setwd(wd)
  
  if(!file.exists(paste0(wd, '/renv.lock'))) {system(paste0('cp ', input_lockfile_location, ' ', wd, '/renv.lock'))}
  
  renv::restore()
  renv::hydrate()
  
  return('Done!')
}




test_model <- function(model, test_data, features, target, prediction_colname, type = NULL, matrix_needed_algos = c('xgboost', 'glmnet', 'glm')) {
  
  #' @description test or predict on a dataset with the trained parsnip model
  #' 
  #' @param model a trained parsnip modle object
  #' @param test_data the data to test the model on
  #' @param target the dependent variable
  #' @param features input features that were used in training the model
  #' @param type the type of model that was trained. 
  #' Valid inputs are:
  #' NULL - if NULL then parsnip will choose the appropriate value based on the models mode
  #' 'numeric' for regression
  #' 'class' to predict the class
  #' 'prob' for classification probability. If 'prob' then 1 is the positive class probability that is returned
  #' 'raw' - if type is set to raw, read more information about opts here: ?predict.model_fit
  #' 'conf_int'for confidence interval
  #' 'pred_int'
  #' 'quantile'
  #' 
  #' @param prediction_colname the column name of the prediction
  #' 
  #' @return a data.table with the prediction_colname
  
  setDT(test_data)
  
  if (is.null(type)) {type <- ifelse(model$spec$mode == 'classification', 'prob',
                                     ifelse(model$spec$mode == 'regression', 'numeric', NA))}
  
  X_test <- 
    test_data %>% 
    dplyr::select(tidyselect::all_of(features))
  
  if (model$spec$engine %in% matrix_needed_algos) {
    X_test <- X_test %>%
      data.matrix()
  }
  
  if (model$spec$engine == 'glmnet') {
    ds_print("***Predicting with type = 'raw' and taking rowMeans***")
    predictions <- rowMeans(parsnip::predict.model_fit(object = model, new_data = X_test, type = 'raw'))}
  else {
    predictions <- parsnip::predict.model_fit(object = model, new_data = X_test, type = type) # prediction column will be called .pred
  }
  
  stopifnot(nrow(predictions) == nrow(X_test))
  
  if (type == 'prob') {
    
    test_data[[prediction_colname]] <- predictions$.pred_1
    
  } else if (type %in% c('numeric', 'class')) {
    
    if (model$spec$engine %in% c('glmnet')) {
      
      test_data[[prediction_colname]] <- predictions
      
    } else {
      
      test_data[[prediction_colname]] <- predictions$.pred
      
    }
  }
  
  return(test_data)
  
}


run_model <- function(dt_full, dt_train, algorithm, dep_var, training_features, prediction_colname, config_collapsed_params, type = NULL) {
  
  #' @title Trains & Tests the Model
  #'
  #' @called_by run_fraud_modelling zZounds_run_report, and AMS_run_report
  #' @calls train_model and test_model
  #'
  #' @param dt_full all the data created by main data-processing pipeline
  #' @param dt_train only the training data with training features and target variable
  #' @param algorithm a parsnip model object that you specify in the config under train_model_params$algorithm. 
  #' Specify all hyperparameters in this parsnip object
  #' @param dep_var the name of your dependent variable
  #' @param training_features a character vector of training feature names. 
  #' It's the output of the prep_features function, which takes input training_features in the config
  #' @param prediction_colname the column name of the prediction
  #' @param config_collapsed_params the list of params passed to the main run fn
  #' @return a list with 3 elements:
  #'   1. model: model object
  #'   2. dt_full: data.table of train & test orders and their corresponding predictions and business metrics
  #'   3. config_collapsed_params: list of config params used in the creation of the prediction. This will allow us to better track which models led to what predictions
  
  ds_print('\nRunning ', algorithm$engine, '...\n', sep = '')
  
  run_model_list <- list(config_collapsed_params = config_collapsed_params)
  
  run_model_list$model <- train_model(algorithm = algorithm, train_data = dt_train, 
                                      target = dep_var, features = training_features)
  run_model_list$dt_full <- test_model(model = run_model_list$model, test_data = dt_full, target = dep_var,
                                      features = training_features, prediction_colname = prediction_colname, 
                                      type = type)
  
  ds_print("\nModel training done!\n")
  
  return(run_model_list)
}

detect_datetime_cols <- function(DT, ignore_cols=c()) {
    datetime_cols <-
      sapply(names(DT),
             function(x) {
               tryCatch({
                 datetime_val <- DT[!is.na(get(x)), as.POSIXct(get(x))]
                 return(x)
               }, error = function(e) NULL)
             })
    
    datetime_cols <- datetime_cols[sapply(datetime_cols, function(x) !is.null(x))] %>% names
    
    return(datetime_cols)
}

select_bw_cols <- function(DT,
                           blacklist_input_features,
                           blacklist_input_patterns,
                           whitelist_input_features,
                           whitelist_input_patterns,
                           rm_single_level_cols = T,
                           make_copy = F) {
  
  #' @description this function cleans / filters the blacklist / whitelist cols (e.g. bw cols)
  #' @param blacklist_input_features
  #' @param blacklist_input_patterns
  #' @param whitelist_input_features
  #' @param whitelist_input_patterns
  #' @param rm_single_level_cols
  #' @param make_copy = F
  
  # list2env(select_bw_params, globalenv())
  
  if (make_copy) DT <- copy(DT)
  
  single_level_cols <-
    DT %>%
    purrr::keep(., ~n_distinct(.) <= 1) %>%
    names() # this computation takes the most time in the function
  
  # if (rm_single_level_cols) DT[, (single_level_cols) := NULL]
  
  blacklist_input_features <-
    c(blacklist_input_features,
      names(DT)[grepl(paste(blacklist_input_patterns, collapse="|"), names(DT))]) %>%
    unique() %>%
    {if (rm_single_level_cols) .[!. %in% single_level_cols] else .}
  
  whitelist_input_features <-
    names(DT)[grepl(paste(whitelist_input_patterns, collapse="|"), names(DT))] %>%
    unique() %>%
    {if (rm_single_level_cols) .[!. %in% single_level_cols] else .}
  
  ds_print(paste('The below features will be moved to whitelist features:\n
                  Whitelist / blacklist overlap:\n',
                 toString(intersect(blacklist_input_features, whitelist_input_features))))
  
  
  blacklist_input_features <- blacklist_input_features[!blacklist_input_features %in% whitelist_input_features]
  
  input_features <- unique(c(whitelist_input_features,
                             names(DT)[!names(DT) %in% unique(c(blacklist_input_features))])) %>%
    {if (rm_single_level_cols) .[!. %in% single_level_cols] else .}
  
  input_feature_types <-
    sapply(DT[, ..input_features], class) %>%
    cbind(names(.), as.character(.)) %>%
    as.data.table() %>% # this will be appended to the output object later
    .[, -1] %>%
    setNames(., c('col', 'dtype'))
  
  outlist <-
    list(
      input_features = input_features,
      whitelist_input_features = whitelist_input_features,
      blacklist_input_features = blacklist_input_features,
      input_feature_types = input_feature_types,
      single_level_cols = single_level_cols
      )
  
  return(outlist)

  }


fast_na_fill <- function(DT, value, make_copy = F) {
  
  #' @param DT a data.table
  #' @param value a constant value
  #' @param make_copy whether to make an internal copy of the DT
  
  stopifnot(is.data.table(DT))
  if (make_copy) DT <- copy(DT)
  
  for (j in names(DT)) set(DT, which(is.na(DT[[j]])), j, value)
  
  return(DT)
}
