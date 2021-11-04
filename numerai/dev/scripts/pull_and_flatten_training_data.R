# Creates the data.table needed for training a model

user <- as.character(Sys.info()['user'])

# Globally set align_R_states on or off.
# It cannot be set in the config because the config calls %>%
# and we need to source the libraries after align_states() is called.

align_R_states <- T

lockfile_location <- paste0('/home/', user, '/ds_machine_learning/renv.lock')

setwd(paste0('/home/', user, '/ds_machine_learning/'))

###### Align R States ######

source('utils/ml_utils.R')

if (align_R_states) {align_states(wd = getwd(), input_lockfile_location = 'renv.lock')}
source('utils/required_libraries.R')
source('utils/ml_utils.R')

###### source config ######

option_args <- commandArgs(trailingOnly = TRUE) # we can pass config filename in the terminal to overwrite the default configs
config_file <- if (length(option_args)) option_args[1] else "configs/uw_config_mvp.R"

setwd(paste0('/home/', user, '/ds_ml_data_wrangling/'))

source('utils/training_functions.R')
source('utils/flatten_dt_function.R')
source('utils/run_and_monitor.R')
source(config_file)

options(mc.cores = num_dt_threads)
data.table::setDTthreads(num_dt_threads)

if (run_memory_profiler) Rprof(tmp <- tempfile(), memory.profiling = T) # initializing memory-profiler

total_run_time <- proc.time()

if (run_etl) {
  
  ##### Create linkage mapping tables #####
  
  if (length(sql_linkage_mapping_to_run) > 0) {
    
    for (link_file in sql_linkage_mapping_to_run) {
      split_queries <-
        getSQL(link_file) %>%
        insert_min_and_max_orderid(.,
                                   min_orderid = min_orderid,
                                   max_orderid = max_orderid) %>%
        strsplit(., split = ';') %>%
        .[[1]]
      
      if (create_mapping_tables) {
        for (single_query in split_queries) {
          ds_print(single_query)
          sqlDB(single_query, user)
          }
        }
      }
    }
  
  
  ###### Parse Queries ######
  
  sql_file_schemas <- sapply(sql_files, function(x) if(length(get_schema(x)) == 0) user else get_schema(x)) # this process assumes that each SQL file has only one set schema
  if (run_whitelist_ETL) {tmp_names <- names(sql_files); sql_files <- lapply(names(sql_files), function(x) str_replace(sql_files[[x]], x, paste0('whitelist_', x))) %>% setNames(., tmp_names); rm(tmp_names);}
  
  # sql_file_queries is a named list, where each name is the "type" of query that comes from the ETL/ folder (e.g. general_realtime, uw_realtime, ctype, etc...).
  # The names in the vector are the FROM-clause table names, and are deduped if necessary.
  
  sql_file_queries <-
    sql_files %>%
    lapply(.,
           function(sql_file)
             sql_file %>% 
             parse_queries(.,
                           name_trigger_clause = name_trigger_clause[[names(which(sql_files == .))]], 
                           query_name_prefix = names(which(sql_files == .))) %>% 
             insert_min_and_max_orderid(., min_orderid = min_orderid, max_orderid = max_orderid))
  
  stopifnot(sapply(sql_file_queries, function(x) startsWith(toupper(x), 'SELECT')) %>% sapply(., all) %>% all())
  
  ###### run queries and pull data ######

  raw_table_names <- c()
  raw_tables <- list()
  
  all_table_names <- unique(c(names(sql_file_queries$general_realtime), names(sql_file_queries$uw_realtime), names(sql_file_queries$ctype)))
  table_names_to_run <- intersect(names(flatten_dt_params), all_table_names)
  
  num_tables <- lapply(table_names_to_run, length) %>% unlist() %>% sum()
  date_stamp <- toString(Sys.Date())
  
  if (length(table_names_to_run) < length(all_table_names)) {ds_print(paste0(" \n\n\n *** WARNING --- The following tables are not in flatten_dt_params, but are in all_table_names: *** \n\n", str_replace(toString(setdiff(all_table_names, table_names_to_run)), ',', ',\n')))}
  if (length(setdiff(table_names_to_run, all_table_names)) > 0) {ds_print(paste0(" \n\n\n *** WARNING --- The following tables are not in all_table_names, but are in flatten_dt_params: *** \n\n", str_replace(toString(setdiff(table_names_to_run, all_table_names)), ',', ',\n')))}
  
  etl_run_time <- proc.time()
  
  i <- 0
  
  for (sql_file_type in names(sql_file_queries)) {
    
    sql_file <- sql_files[[sql_file_type]]
    schema <- sql_file_schemas[[sql_file_type]]
    queries <- sql_file_queries[[sql_file_type]]
    query_names <- names(queries)
    output_table_names <- query_names %>% setNames(., query_names)
    raw_table_names <- c(raw_table_names, output_table_names) %>% unique()
    ds_print("\n\n\nRunning the queries in ", sql_file, "in the schema ", schema, "\n\n\n")
    
    for (query_name in query_names[query_names %in% table_names_to_run]) {
      
      i <- i + 1
      
      ds_print("\n\n------------------------------\nquery_name:", query_name, " (", i, "/", num_tables, " --- ", round( (i / num_tables)*100, 3), "%)", "\n------------------------------\n\n")
      
      query <- queries[[query_name]]
      
      output_table_name <- output_table_names[[query_name]]
      
      dt_i <- sqlDB(query, schema)
      
      if ('cur_orderid' %in% names(dt_i)) dt_i[, orderid := cur_orderid] # need to set orderid to be cur_orderid before flattening
      
      if (write_raw_tables_to_mysql) {if (nrow(dt_i) > 0) {write_mysql_table(dt_i, output_table_name, user, primary_key = "orderid")} else {raw_table_names <- raw_table_names[raw_table_names != output_table_name]}}
      
      if (load_custom_feature_tables_with_feather & calc_custom_features) {if (nrow(dt_i) > 0 & query_name %in% tables_needed_for_custom_features) write_feather(dt_i, paste0(write_each_raw_table_path, query_name, '_', date_stamp, '.feather'))} # if this fails it is likely because there is no space left on disk!
      
      if (flatten_after_each_raw_query) { # if we want to flatten each table immediately after the raw table is queried
        
        if (nrow(dt_i) > 0) {
          
          dt_flat <- do.call(flatten_dt, purrr::list_modify(flatten_dt_params[[output_table_names[[query_name]]]], DT = dt_i))
         
          if (rm_raw_tables_after_flattening) rm(dt_i) # remove each raw table after flattening
          stopifnot(nrow(dt_flat) > 0)
          if (is.character(write_each_flat_table_path)) write_feather(dt_flat, paste0(write_each_flat_table_path, output_table_name, '_', date_stamp, '.feather'))
          
          rm(dt_flat)
        
        }
      }
      if (keep_raw_tables_in_mem) {
        if (nrow(dt_i) > 0) {raw_tables[[output_table_name]] <- dt_i} else {raw_table_names <- raw_table_names[raw_table_names != output_table_name]}
      }
    }
  }
  
  ##### calculate custom features #####
  
  if (calc_custom_features) {
    
    if (load_custom_feature_tables_with_feather && !keep_raw_tables_in_mem) {
      
      ds_print('\n\n*** calculating custom features ***\n\n')
      
      stopifnot(length(setdiff(tables_needed_for_custom_features, names(flatten_dt_params))) == 0)
      
      # re-load the raw tables used to compute custom features by calling the reading each raw feather file
      raw_filenames <- list.files(write_each_raw_table_path)[endsWith(list.files(write_each_raw_table_path) %>% str_replace(., '.feather', ''), date_stamp)] %>%
        .[str_replace(., paste0('_', date_stamp, '.feather'), '') %in% tables_needed_for_custom_features]
      
      raw_tables_to_read <- paste0(write_each_raw_table_path, raw_filenames)
      
      tables_missing <- setdiff(str_replace(raw_filenames, paste0('_', date_stamp, '.feather'), ''), tables_needed_for_custom_features)
      if (length(tables_missing)) ds_print('\n*** WARNING THE FOLLOWING TABLES ARE NOT IN raw_filenames BUT ARE IN tables_needed_for_custom_features ***\n', str_replace(toString(tables_missing), ',', '\n'))
      
      raw_tables <- mclapply(raw_tables_to_read, function(x) {dt_out <- read_feather(x) %>% as.data.table(); gc(); return(dt_out)}, mc.cores = num_dt_threads)
      names(raw_tables) <- raw_filenames %>% str_replace(., paste0('_', date_stamp, '.feather'), '')
      
      gc()
      
    } else if (!load_custom_feature_tables_with_feather && !keep_raw_tables_in_mem) {
      
      if (length(setdiff(tables_needed_for_custom_features, names(flatten_dt_params))) > 0) {ds_print('\n*** WARNING THE FOLLOWING TABLES ARE IN tables_needed_for_custom_features BUT ARE NOT IN names(flatten_dt_params) ***\n', str_replace(toString(setdiff(tables_needed_for_custom_features, names(flatten_dt_params))), ',', '\n'))}
      
      i <- 1
      # re-load the raw tables used to compute custom features by calling the SQL queries
      for (table_name in tables_needed_for_custom_features) {
        for (query_source in names(sql_file_queries)) {
          if (startsWith(table_name, query_source)) {
            ds_print(paste0('\n\n *** running query ', i, ' out of ', length(tables_needed_for_custom_features), ' (', round(i/length(tables_needed_for_custom_features)*100, 3), '%) ***\n\n'))
            query <- sql_file_queries[[query_source]][[table_name]]
            # query <- paste0(str_replace(sql_file_queries[[query_source]][[table_name]], ';', ''), ' limit 500000;') # for debugging
            raw_tables[[table_name]] <- sqlDB(query, sql_file_schemas[[query_source]])
            i <- i + 1
            gc()
          }
        }
      }
    }
    
    raw_tables <-
      raw_tables %>%
      purrr::list_modify(.,
                         
                         # financing_amount
                         financing_amount = calc_financing_amount(table_list = .),
                         
                         # num_prev_good_single_pays
                         num_prev_good_single_pays = calc_num_good_single_pays(table_list = .),
                         
                         # num_prev_good_installments
                         num_prev_good_installments = calc_num_previous_good_installments(table_list = .)
                         )
    gc()
  }
  
  # if (save_raw_tables_as_rds) saveRDS(raw_tables, save_raw_tables_file_path)
  
  } else {
    
    raw_tables <-
      readRDS(read_raw_tables_file_path) %>%
      {if (is.numeric(limit_raw_tables)) lapply(., function(x) tail(x, floor(limit_raw_tables))) else .}
  }


flatten_run_time <- proc.time()

if (run_flattening) {
  
  if (keep_raw_tables_in_mem) {
    
    ##### Flattening process #####
    
    # sort raw_tables by object_size increasingly, so we can remove the small tables
    sorted_raw_tables_by_size <-
      raw_tables %>%
      lapply(., function(x) tail(x, 100000)) %>% # limit size so we don't run out of memory
      mclapply(., object.size, mc.cores=num_dt_threads) %>%
      as.data.table() %>%
      melt %>%
      .[, value := str_replace(value, 'bytes', '') %>%
          as.numeric()] %>%
      .[order(value),]
   
    gc()
    
    diff1 <- setdiff(names(flatten_dt_params), sorted_raw_tables_by_size$variable)
    if (length(diff1)) ds_print(paste0(' \n\n\n *** WARNING --- ', diff1, ' IN names(flatten_dt_params) and NOT in sorted_raw_tables_by_size *** \n ONLY SELECTING names(flatten_dt_params) \n\n\n'))
    
    diff2 <- setdiff(sorted_raw_tables_by_size$variable, names(flatten_dt_params))
    if (length(diff2)) ds_print(paste0(' \n\n\n *** WARNING --- ', diff2, ' IN names(flatten_dt_params) and NOT in sorted_raw_tables_by_size *** \n ONLY SELECTING names(flatten_dt_params) \n\n\n'))
    
    
    i <- 0
    flattened_dts <-
      mclapply(names(flatten_dt_params)[sorted_raw_tables_by_size$variable] %>% .[!is.na(.)],
               
               function(table_name) {
                 
                 i <<- i + 1
                 
                 ds_print(paste0('\n\n\n****** ', table_name, ' --- ', i, '/', length(flatten_dt_params), ' (', round((i / length(flatten_dt_params)*100), 3), '%) ******\n\n\n'))
                 
                 if (is.null(raw_tables[[table_name]])) ds_print(paste0('\n\n*** WARNING ', table_name, ' IS NOT AVAILABLE IN raw_tables ***\n\n'))
                 
                 if (!is.null(raw_tables[[table_name]])) {
                   
                   dt_flat <- do.call(flatten_dt, purrr::list_modify(flatten_dt_params[[table_name]], DT = raw_tables[[table_name]]))
                   # dt_flat <- dt_flat[, names(dt_flat)[1:floor(ncol(dt_flat) * .20)], with = F] # for debugging RAM
                   
                   if (rm_raw_tables_after_flattening) raw_tables[[table_name]] <<- NULL
                   gc()
                   
                   return(dt_flat)
                   
                 } else return(NULL)
               }, mc.cores = flatten_mc_cores) %>%
      setNames(., names(flatten_dt_params))
    
    if (!is.null(save_flattened_rds_filepath)) saveRDS(flattened_dts, save_flattened_rds_filepath)
    # flattened_dts <- readRDS('~/flattened_dts.rds')
  } else {
    
    if (calc_custom_features) raw_tables <- raw_tables[c('financing_amount', 'num_prev_good_single_pays', 'num_prev_good_installments')]
    flattened_filenames <- list.files(write_each_flat_table_path)[endsWith(list.files(write_each_flat_table_path) %>% str_replace(., '.feather', ''), date_stamp)]
    flattened_tables_to_read <- paste0(write_each_flat_table_path, flattened_filenames)
    flattened_dts <- mclapply(flattened_tables_to_read, function(x) {dt_out <- read_feather(x) %>% as.data.table(); gc(); return(dt_out)}, mc.cores = num_dt_threads)
    names(flattened_dts) <- flattened_filenames %>% str_replace(., paste0('_', date_stamp, '.feather'), '')
    }

  if (calc_custom_features) {
    flattened_dts <-
      flattened_dts %>%
      purrr::list_modify(.,
                         financing_amount = raw_tables$financing_amount,
                         num_prev_good_single_pays = raw_tables$num_prev_good_single_pays,
                         num_prev_good_installments = raw_tables$num_prev_good_installments
                         )
    }
  
  rm(raw_tables)
  
  gc()
  
  dt_flat <-
    flattened_dts %>%
    purrr::discard(is.null) %>%
    full_outer_join_dts(., key_col = "orderid") # outer join the flattened tables
    
  rm(flattened_dts)
  
  
  ### fill_by_group ###
  
  if (!is.null(fill_params) & (length(setdiff(names(fill_params), names(dt_flat))) == 0)) {
        mclapply(names(fill_params),
                 function(x) {
                   needed_cols <- formals(fill_by_group_uw) %>% .[c('groupby_cols', 'fill_col', 'non_na_col', 'preapproved_col', 'active_cr_col', 'date_col')] %>% as.character()
                   if (all(needed_cols %in% names(dt_flat))) {
                     do.call(fill_by_group_uw, list(DT = dt_flat) %>% append(., fill_params[[x]])) # run inplace instead of making a copy
                      } else {
                       ds_print(paste0("\n\n *** WARNING --- NOT ALL needed_cols ARE IN names(dt_flat) BELOW ***\n", toString(setdiff(needed_cols, names(dt_flat))), "\n\n"))
                     }
                   },
                 mc.cores = num_dt_threads)
    } else {
      ds_print(paste0('\n\n *** WARNING --- SKIPPING cols_to_fill STEP BECAUSE THE FOLLOWING COLUMNS ARE NOT IN dt_flat *** \n\n',
                      toString(setdiff(names(fill_params), names(dt_flat))) %>% str_replace(., ',', ',\n')))
      }
  
  gc()
  
  if (!is.null(save_dt_flat_file_path)) {write_feather(dt_flat, save_dt_flat_file_path)}
  }

if (run_memory_profiler) {Rprof(); mem_summary <- summaryRprof(tmp, memory = "stats");} # close the memory profiler session

ds_print('\nDone!\n')
ds_print("\nETL time taken:", data.table::timetaken(etl_run_time),
         "\nFlatten time taken:", data.table::timetaken(flatten_run_time),
         "\nTotal time taken:", data.table::timetaken(total_run_time))
