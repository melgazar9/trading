# Creates the Data.Table needed for training a model

user <- Sys.info()['user']

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
source('utils/ml_utils.R') # uses the sqlDB function

###### source config ######

option_args <- commandArgs(trailingOnly = TRUE) # we can pass config filename in the terminal to overwrite the default configs
config_file <- if (length(option_args)) option_args[1] else "configs/config_slim.R"

setwd(paste0('/home/', user, '/ds_ml_data_wrangling/'))
source(config_file)

options(mc.cores = num_threads)
data.table::setDTthreads(num_threads)

###### Parse Queries ######

source('utils/training_functions.R')
source('utils/flatten_dt_function.R')

sql_files <- sql_files %>% setNames(., .)

# this process assumes that each SQL file has ONE AND ONLY ONE schema to be set
# I was trying to avoid hardcoding the schema in each and every FROM and JOIN clause
sql_file_schemas <- sapply(sql_files, get_schema)


# Creates a list with the names of the sql_files
# each element in the list is a named character vector of the SQL queries
# the names in the vector are the FROM-clause table names. They are de-duped if necessary
sql_file_queries <- 
  sql_files %>% 
  setNames(., .) %>% 
  lapply(., 
         function(sql_file)
           sql_file %>% 
           parse_queries %>% 
           insert_min_and_max_orderid(., min_orderid = min_orderid, max_orderid = max_orderid)
           ) 


###### Run Queries ######

# this double lapply iterates over each file in sql_files and each SELECT statement in that file

print(system.time({

  lapply(names(sql_file_queries), function(sql_file) {
    # sql_file <- names(sql_file_queries)[[1]]
    
    # get the schema that is for that sql_file
    schema <- sql_file_schemas[[sql_file]]
    queries <- sql_file_queries[[sql_file]]
    queries_names <- names(queries)
    output_table_names <- paste0("flattened_", queries_names) %>% setNames(., queries_names)
    
    cat("\n\n\nRunning the queries in ", sql_file, "in the schema ", schema, "\n\n\n")

    lapply(queries_names, function(query_name) 
           {
             cat("\n\n------------------------------\nquery_name:", query_name, "\n------------------------------\n\n")
            # try({
              
              stopifnot(query_name %in% names(flatten_dt_param_list))
              
              # get parameters
              query <- queries[[query_name]]
              output_table_name <- output_table_names[[query_name]]
              flatten_dt_params <- flatten_dt_param_list[[query_name]]
              
              sqlDB(query, schema) %>% 
                # an empty DT will be missing the orderid column
                # {if (nrow(.) == 0) .[, orderid := integer()] else .} %>%  
                {do.call(flatten_dt, purrr::list_modify(flatten_dt_params, DT = .))} %>% 
                {if (nrow(.) > 0) write_mysql_table(., output_table_name, user, primary_key = "orderid")}
              
              gc()

            # })
           }) 
  })
}))