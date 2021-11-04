setwd(paste0('/Users/', as.character(Sys.info()['user']), '/scripts/github/trading'))

source('numerai/dev/scripts/required_libraries.R')

source('numerai/dev/scripts/pull_data_utils.R')
source('numerai/dev/scripts/flatten_dt_fn.R')
source('numerai/dev/scripts/getSymbols_tidyquant.R')

###### Global Variables #####

NUM_THREADS = 1

TICKERS = download_ticker_map()$yahoo # c('aapl', 'msft') %>% toupper()

##### Pull the data from yahoo finance #####

dt_stocks_list <-
  download_yfinance_data(TICKERS) %>%
  setNames(TICKERS) %>%
  {mclapply(names(.), function(x) calc_indicators(.[[x]]), mc.cores = NUM_THREADS)} %>% # calculate indicators for each stock
  setNames(TICKERS) %>%
  {mclapply(names(.), function(x) mc.cores = NUM_THREADS)}
  {full_outer_join_dts(.)}

gc()
