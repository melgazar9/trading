setwd(paste0('/Users/', as.character(Sys.info()['user']), '/scripts/github/trading'))

source('numerai_project/dev/scripts/required_libraries.R')

library('tidyquant')


download_yfinance_data <- function(tickers,
                                   intervals_to_download = c('1d', '1h'),
                                   tq_get_params = list(from = '2021-01-01', get = "stock.prices"),
                                   n_threads = 1,
                                   merge_params = list(by = 'date', all = T),
                                   max_intraday_lookback_days = 363,
                                   yahoo_ticker_colname = 'yahoo_ticker',
                                   verbose = T,
                                   sleep_seconds = 2) {
  
  #' @description pull yahoo data in it's raw form
  #' 
  #' @Note See yfinance.download docs for a detailed description of yfinance parameters
  #' @Note passing some intervals return unreliable stock data (e.g. '3mo' returns many NA data points when they should not be NA)
  #' 
  #' @param tickers list of tickers to pass to yfinance.download - it will be parsed to be in the format "AAPL MSFT FB"
  #' @param intervals_to_download : list of intervals to download OHLCV data for each stock --- e.g. c('1w', '1d', '1h')
  #' @param tq_get_params params passed to tidyquant::tq_get
  #' @param n_threads number of threads used to download the data. So far only 1 thread is implemented
  #' @param merge_params merge each data.table on the `date` column - see merge inputs for inner / left/right outer joins
  #' @return a data.table if there is only one table pulled. Return a list of tables if the number of tables is greater than 1 
  
  start_date <- ifelse('from' %in% names(tq_get_params), tq_get_params$from, '2005-01-01')
  end_date <- ifelse('to' %in% names(tq_get_params), tq_get_params$to, toString(Sys.Date()))

  intraday_lookback_days <- ifelse(Sys.Date() - max_intraday_lookback_days < start_date, Sys.Date() - as.Date(start_date), Sys.Date() - (Sys.Date() - max_intraday_lookback_days))
  
  tickers <- toupper(tickers)
  
  getSymbols(tickers,
             from = start_date,
             to = end_date,
             auto.assign = T)
  
  # prices <-
  #   map(tickers, function(x) Ad(get(x))) %>%
  #   reduce(., merge) %>%
  #   as.data.table() %>%
  #   setNames(c('datetime', tickers))
  
  dt_stocks_list <-
    mclapply(tickers,
             function(x) {
               do.call(tq_get, list(x = x) %>% append(., tq_get_params)) %>%
                         setDT(.)
             },
             mc.cores = n_threads)
  
  ### Below is how to merge the output and create a suffix for each ticker, but that will go in its own fn or process ###
  
  # dt_stocks_wide <-
  #   mclapply(tickers,
  #            function(x) {
  #              tq_get(x,
  #                     from = start_date,
  #                     to = end_date,
  #                     get = "stock.prices") %>%
  #                setDT(.) %>%
  #                dplyr::rename_at(vars(-all_of(c(merge_cols, 'symbol') %>% unique())), ~ paste0(x, '_', .)) %>%
  #                .[, symbol := NULL]
  #            }, mc.cores = n_threads) %>%
  #   Reduce(function(dt1, dt2) do.call(merge, list(dt1, dt2) %>% append(., merge_params)), .)
  
  return(dt_stocks_list)
  
}


calc_indicators <- function(dt_stocks,
                            
                              
                            tq_fns_to_apply = list(
                              
                              EVWMA = list(
                                x = 'close',
                                y = 'volume',
                                mutate_fun = 'EVWMA',
                                col_rename = 'EVWMA'
                                ),
                              
                              
                              MACD = list(
                                x = 'adjusted',
                                mutate_fun = 'MACD',
                                col_rename = 'macd'
                                ),
                              
                              SMA = list(
                                x = 'adjusted',
                                mutate_fun = 'SMA',
                                col_rename = 'SMA'
                                )
                              
                              ),
                            
                            merge_cols = c('symbol', 'date'),
                            num_threads = 1
                            ) {
  
  
  dt_tq_mutated <-
    mclapply(names(tq_fns_to_apply),
             function(x) {
               do.call(tq_mutate_xy_, list(data = dt_stocks) %>% append(., tq_fns_to_apply[[x]])) %>% setDT(.)
             },
             mc.cores = num_threads
             ) %>%
    {if (is.list(.)) full_outer_join_dts(., merge_cols = c('date', 'symbol')) else .}
  
  return(dt_tq_mutated)
    
}

download_ticker_map <- function(numerai_ticker_link='https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv',
                                main_ticker_col='bloomberg_ticker',
                                verbose=T) {
  # eligible_tickers = pd.Series(napi.ticker_universe(), name='yahoo_ticker')
  ticker_map = fread(numerai_ticker_link)
  
  # Remove null / empty tickers from the yahoo tickers
  valid_tickers <- ticker_map[!is.na(yahoo) &
                                !is.null(yahoo) &
                                !tolower(yahoo) == 'nan' &
                                !tolower(yahoo) == 'null' &
                                !tolower(yahoo) == 'none' &
                                !str_replace(tolower(yahoo), ' ', '') == '' &
                                length(yahoo) > 0,
                              ]
  
  if (verbose) print(paste0('tickers before cleaning: ', toString(dim(ticker_map))))  # before removing bad tickers
  ticker_map <- ticker_map[yahoo %in% valid_tickers$yahoo,]
  if (verbose) print(paste0('tickers after cleaning: ', toString(dim(ticker_map))))
  
  return(ticker_map)
}
  
 