calc_standard_metrics <- function(DT,
                                  fits,
                                  reg_or_clf=NULL,
                                  numeric_clf_fits = c(),
                                  dep_var = 'actual_loss_ratio',
                                  evaluation_colname = "dataset_split",
                                  evaluation_datasets = c('train', 'val', 'test'),
                                  reg_metrics = yardstick::metric_set(yardstick::rmse,
                                                                      yardstick::rsq,
                                                                      yardstick::mae),
                                  clf_class_metrics = yardstick::metric_set(yardstick::precision, 
                                                                      yardstick::recall, 
                                                                      yardstick::f_meas),
                                  
                                  clf_numeric_metrics = yardstick::metric_set(yardstick::pr_auc),
                                  event_level = 'second') {
  
  #' @title Calculates Train & Test Classification Metrics for One or More Fits
  #' 
  #' @param DT data.table
  #' @param reg_or_clf can be NULL, 'reg' or 'clf' specifying whether it is a regression or classification problem. If null, then if dep_var is numeric, set to reg, else set to clf
  #' @param fits vector of column names that consist of the predicted classes
  #' @param dep_var column name of the target variable
  #' @param evaluation_colname the name of the column that contains the evaluation_datasets (train, val, test, etc.)
  #' @param evaluation_datasets a character vector of datasets that should be plotted from the evaluation_colname
  #' @param reg_metrics an instance of yardstick::metric_set
  #' @param clf_class_metrics an instance of yardstick::metric_set to be performed on pred_class cols
  #' @param clf_numeric_metrics an instance of yardstick::metric_set to be performed on continuous fit cols (e.g. probability fits)
  #' @param event_level only matters when reg_or_clf = 'clf' - By default, yardstick assumes that the first level is the positive class, but our code assumes the second level is
  #' 
  #' @return a data.table containing the columns: fit, dataset, variable & value
  
  ##### validations  #####
  
  required_cols <- c(evaluation_colname, dep_var, fits, numeric_clf_fits)
  stopifnot(all(required_cols %in% names(DT)))
  
  if (!data.table::is.data.table(DT)) data.table::setDT(DT)
  
  if (is.null(reg_or_clf)) {reg_or_clf <- ifelse(uniqueN(na.omit(DT[[dep_var]])) == 2, 'clf', 'reg')}
  
  
  ##### create & prepare melted_dt  #####
  
  suppressWarnings({
    melted_dt <-
      DT[get(evaluation_colname) %in% evaluation_datasets, ..required_cols] %>% # select only the columns & datasets needed
      melt(., measure.vars = c(fits, numeric_clf_fits),                                   # melt all the fit columns to a single class_prob column
           variable.name = "fit",
           value.name = "prediction",
           value.factor = T)
  })
  
  
  ##### check for missing values #####
  
  grouping_cols <- c("fit", evaluation_colname)
  
  na_counts_dt <-
    melted_dt[, .(.N, n_NAs = sum(is.na(prediction))), 
              by = grouping_cols
              ][,
                pct_NAs := n_NAs / N
                ][
                  n_NAs > 0
                  ]
  
  
  if (na_counts_dt[, .N]) {
    ds_print("\nWe have some missing values & they will be removed\n")
    print(na_counts_dt)
  }
  
  
  ##### reg_metrics #####
  
  if (reg_or_clf == 'reg') {
    
    stopifnot(is.numeric(DT[[dep_var]]))
    
    reg_metrics_dt <-
      melted_dt %>%
      na.omit %>%
      .[, reg_metrics(.SD,
                      truth = get(dep_var),
                      prediction,    # this class probability column is only used if pr_auc is in the metric set
                      estimate = prediction),
        keyby = grouping_cols
        ]
    
    data.table::setnames(reg_metrics_dt, c(".metric", ".estimate"), c("variable", "value"))
    reg_metrics_dt[, variable := forcats::fct_relevel(variable, c('rmse', 'rsq', 'mae'))]
    reg_metrics_dt[, (evaluation_colname) := forcats::fct_relevel(get(evaluation_colname), c("train", "val", "test"))]
    
    return(reg_metrics_dt)
  } else {
    
    stopifnot(is.factor(DT[[dep_var]]))
    
    clf_class_metrics_dt <-
      melted_dt[fit %in% fits,] %>%
      na.omit %>%
      .[, clf_class_metrics(.SD,
                            truth = factor(get(dep_var), levels = c('0', '1')),
                            estimate = factor(prediction, levels = c('0', '1')),
                            event_level = event_level),
        keyby = grouping_cols
        ]
    
    clf_numeric_metrics_dt <-
      melted_dt[fit %in% numeric_clf_fits,] %>%
      na.omit %>%
      .[, clf_numeric_metrics(.SD,
                            truth = factor(get(dep_var), levels = c('0', '1')),
                            prediction,
                            estimate = factor(prediction, levels = c('0', '1')),
                            event_level = event_level),
        keyby = grouping_cols
        ]
    
    clf_metrics_dt <- rbindlist(list(clf_class_metrics_dt, clf_numeric_metrics_dt))
    
    data.table::setnames(clf_metrics_dt, c(".metric", ".estimate"), c("variable", "value"))
    clf_metrics_dt[, variable := forcats::fct_relevel(variable, c("precision", "recall", "f_meas"))]
    clf_metrics_dt[, (evaluation_colname) := forcats::fct_relevel(get(evaluation_colname), levels(evaluation_datasets))]
    setorderv(clf_metrics_dt, c('fit', evaluation_colname, 'variable'))
    
    return(clf_metrics_dt)
  }
}


plot_cross_tabulated_metrics <- function(melted_dt,
                                         plot_aes_x,
                                         value_col = "value",
                                         fill = 'variable',
                                         position = 'stack',
                                         facet_grid_range = c(0,1),
                                         facet_cols = "variable",
                                         facet_rows = NULL,
                                         facet_labeller = "label_value",
                                         text_size = 16,
                                         label_accuracy = NULL,
                                         label_prefix = "",
                                         title_size = 20,
                                         geom_text_size = 5,
                                         show.legend = T,
                                         plot_title = "Metrics",
                                         scale = 'free',
                                         facet_type = NA,
                                         geom_text_position = position_stack(vjust = .5)) {
  
  #' @title Plots a Cross-Tabulated & Labeled Column Plot 
  #' 
  #' @param melted_dt data.table output of train_test_classification_metrics
  #' 
  #' @param plot_aes_x the x and fill parameters to geom_col aesthetic
  #' @param value_col the name of the column containing the y values
  #' @param facet_cols the name of the column containing the facet_grid column values
  #' @param facet_rows the name of the column containing the facet_grid row values
  #' 
  #' @param facet_labeller see https://ggplot2.tidyverse.org/reference/facet_grid.html
  #' @param text_size plot's text size, except for geom_text size
  #' @param label_accuracy number of percentage decimal places, e.g. 1, .1, .01, etc.
  #' @param title_size wrap the text after this many characters
  #' @param geom_text_size the font size of the metric values
  #' @param show.legend should the "fill" legend be shown?
  #' @param plot_title the plot's title
  #' 
  #' @return plot
  
  required_cols <- c(facet_rows, plot_aes_x, facet_cols, value_col)
  stopifnot(required_cols %in% names(melted_dt))
  
  if (is.na(fill)) {fill <- plot_aes_x}
  
  ##### add labels #####
  if (!'label' %in% names(melted_dt)) {
    melted_dt[, label := scales::label_number_si(accuracy = label_accuracy, prefix = label_prefix)(get(value_col))]
  }

  ##### create facet_grid formula #####
  # facet_cols cannot be NULL without facet_rows also being NULL
  
  stopifnot(is.null(facet_cols) & is.null(facet_rows) | !is.null(facet_cols))
  
  
  facet_formula <- 
    if (is.null(facet_rows) & is.null(facet_cols)){
      NULL
    } else {
      as.formula(paste0(facet_rows, "~", facet_cols))
    }
  
  if (is.na(facet_type)) {
    if (all(range(melted_dt[[value_col]]) %between% facet_grid_range, na.rm = T)) {
      facet_type <- facet_grid
    } else {
      facet_type <- facet_wrap
    }
  }
  
  ##### plot the data #####
  
  melted_dt %>%
    ggplot(aes_string(x = plot_aes_x, 
                      y = value_col,
                      label = "label",
                      fill = fill)
    ) +
    geom_col(position = position, show.legend = show.legend) +
    geom_text(position = geom_text_position, size = geom_text_size) + 
    facet_type(facet_formula, labeller = facet_labeller, scales = scale) +
    theme(text = element_text(size = text_size),
          plot.title = element_text(hjust = 0.5, size = title_size)) +
    ggtitle(plot_title)
}


tabulate_confusion_matrix <- function(DT, pred_class, target_class,
                                      evaluation_colname = 'dataset_split',
                                      evaluation_datasets = c('train', 'val', 'test')) {
  #' @title Calculate the Confusion Matrix Metrics
  #'
  #' @param DT data.table created by pull_data, which contains fits, dep_var & evaluation_colname columns
  #' @param pred_class column name for the column containing predicted class (e.g. 'pred_cancel')
  #' @param target_class column name for the actual class (e.g. 'big_writeoff')
  #' 
  #' @return a data.table with the columns N, prop, correct, actual & pred_class with properly renamed columns
  
  
  metrics_dt <-
    DT[get(evaluation_colname) %in% evaluation_datasets, .N,
       by = .(evaluation_colname = get(evaluation_colname),
              target_class = get(target_class),
              pred_class = get(pred_class))
       ][, `:=` (prop = N / sum(N),
                 correct = as.factor(target_class) == pred_class), 
         by = evaluation_colname
         ]
  
  names(metrics_dt) <- c(evaluation_colname, target_class, 'pred_class', 'N', 'prop', 'correct')
  
  return(metrics_dt)
}




plot_confusion_matrices <- function(conf_matrix_dt,
                                    required_cols,
                                    x_actual,
                                    y_pred,
                                    facet_cols = NULL,
                                    facet_rows = NULL,
                                    text_size = 16,
                                    geom_text_size = 5,
                                    pct_accuracy = 1,
                                    show.legend = T) {
  
  #' @title Plot one or more confusion matrices
  #' 
  #' Inspired by https://stackoverflow.com/a/59119854/4463701
  #'
  #' @param facet_cols the name of the column containing the facet_grid column values
  #' @param facet_rows the name of the column containing the facet_grid column values
  #' @param text_size plot's text size, except for geom_text size
  #' @param geom_text_size the font size of the metric values in geom_text
  #' @param pct_accuracy how many decimals should be displayed in the percentages?
  #' @param show.legend should the legend be shown?
  #' 
  #' @return
  
  stopifnot(all(required_cols %in% names(conf_matrix_dt)))
  
  conf_matrix_dt[, label := paste0(scales::label_number(big.mark = ",")(N), " (",
                                   scales::label_percent(pct_accuracy)(prop), ")")]
  
  # stopifnot(is.null(facet_cols) & is.null(facet_rows) | !is.null(facet_cols))
  
  facet_formula <-
    if (is.null(facet_rows) & is.null(facet_cols)) {
      NULL
    } else if (is.null(facet_cols)) {
      as.formula(paste0(facet_rows, '~ .'))
    } else if (is.null(facet_rows)) {
      as.formula(paste0('~ ', facet_cols))
    } else {
      as.formula(paste0(facet_rows, "~", facet_cols))
    }
  
  conf_matrix_dt[[x_actual]] <- as.factor(conf_matrix_dt[[x_actual]])
  conf_matrix_dt[[y_pred]] <- as.factor(conf_matrix_dt[[y_pred]])
  
  ggplot(conf_matrix_dt, aes_string(x = x_actual, y = y_pred, fill = 'correct', alpha = 'prop')) +
    geom_tile(show.legend = show.legend) +
    geom_text(aes(label = label), size = geom_text_size, vjust = .5, fontface  = "bold", alpha = 1) +
    scale_fill_manual(values = c(`TRUE` = "green", `FALSE` = "red")) +
    theme_bw() +
    ylim(rev(levels(factor(conf_matrix_dt[[y_pred]])))) +
    theme(text = element_text(size = text_size)) +
    facet_grid(facet_formula)
}


plot_importance <- function(imp_dt, 
                            top_x_only = NULL, 
                            label_size = 16,
                            geom_text_size = 3,
                            vjust = 0.75, 
                            text_label_accuracy = 0.001,
                            plot_title = "Feature Importance") {
  
  #' @title Create bar plot of the feature importances 
  #' 
  #' @param imp_dt the tibble created by vip::vi()
  #' @param top_x_only if not null, the plot only shows this many features
  #' @param label_size controls the label's font size
  #' @param vjust adjusts the text's height placement
  #' @param text_label_accuracy number of digits to display in geom_text
  #' 
  #' @return plot
  
  if (!is.data.table(imp_dt)) data.table::setDT(imp_dt)
  
  # create the enumerated x-axis labels
  imp_dt[, x_axis_label := reorder(paste0(1:.N, ". ", Variable), Importance)]
  
  if (!is.null(top_x_only)) {
    
    imp_dt <- imp_dt[1:top_x_only]
    plot_title <- paste0(plot_title, ": Top ", top_x_only)
    
  }
  
  # plot
  imp_dt %>% 
    ggplot(aes(x_axis_label, 
               Importance, 
               label = scales::label_number(accuracy = text_label_accuracy, big.mark = " ")(Importance))
    ) +
    geom_col(fill="lightblue") +
    geom_text(position = position_stack(vjust = vjust), size = geom_text_size) +
    theme(text = element_text(size=label_size), plot.title = element_text(hjust = 0.5)) +
    coord_flip() +
    ggtitle(plot_title)
  
}



plot_multifit_ttest <- function(multifit_output_object,
                                text_label_size = 6,
                                pVal_x = 1.5,
                                pVal_y = 0,
                                pVal_size = 6,
                                nudge_y = .1,
                                vjust = 0.75,
                                x_label_angle = 0,
                                si_accuracy = 0.1,
                                caption_size = 15) {
  
  #' @title Plots the T-test Winners & Losers w/ P-Values
  #' 
  #' @param multifit_object the list variable produced by Bootstrap_AB_multifit_Installments
  #' @param text_label_size text size of the dollar values
  #' @param pVal_x/pVal_y/pVal_size controls the size and placement of the p-value text
  #' @param nudge_y prevents the text from overlapping
  #' @param text_vjust adjusts the text's height placement
  #' @param x_label_angle controls the angle of the Winner & Loser labels 
  #' @param si_accuracy controls the number of decimals displayed the dollar amounts on the bars
  #' @param caption_size caption's font size
  #' 
  #' 
  #' @return a plot that is faceted by each T-test
  #' multifit_output_object=output_list$ab_multifit_list
  
  stopifnot(is.list(multifit_output_object))
  stopifnot("ttest_df" %in% names(multifit_output_object))
  
  ttest_dt <- as.data.table(multifit_output_object$ttest_df)
  setorder(ttest_dt, WinnerNetLoss, LoserNetLoss)
  
  # concateante the winner & loser columns
  ttest_dt[, `:=`(round = 1:.N,
                  Winner = str_c(Winner, WinnerNetLoss, sep = ";"),
                  Loser = str_c(Loser, LoserNetLoss, sep = ";"))]
  
  # melt into a longer data.table and separate the previous concatenation
  plot_dt <- 
    melt(ttest_dt,
         id.vars=c("round", "pVal"), 
         measure.vars=c("Winner", "Loser"),
         variable.name="Outcome",
         value.name="NetLoss") %>%
    tidyr::separate(NetLoss, c("Fit", "NetLoss"), sep=";")
  
  # update 2 columns
  plot_dt[, NetLoss := as.numeric(NetLoss)]
  plot_dt[, Outcome := Outcome %>% str_replace("NetLoss", "") %>% reorder(., NetLoss)] 
  
  round_1_winning_margin <- plot_dt[round == 1, max(NetLoss) - min(NetLoss)]
  
  # plot
  ggplot(plot_dt, 
         aes(Outcome,
             NetLoss, 
             fill = Fit)) + 
    geom_col() +
    facet_wrap(~ round) +
    geom_text(aes(x = pVal_x, 
                  y = pVal_y, 
                  label = paste0("pVal = ", scales::label_scientific()(pVal))),
              nudge_y = nudge_y,
              size = pVal_size) +
    geom_text(aes(label = scales::label_number_si(prefix ="$", accuracy = si_accuracy)(NetLoss)), 
              size = text_label_size, 
              position = position_stack(vjust = vjust)) +
    ggtitle("T-Test Winners & Losers") +
    theme(axis.text.x = element_text(angle = x_label_angle),
          plot.title = element_text(hjust = 0.5),
          plot.caption = element_text(hjust = 0, size = caption_size)) +
    labs(caption = sprintf("The winner's NetLoss margin is %s", 
                           scales::label_number(prefix ="$", big.mark = ",")(round_1_winning_margin)))
}



estimate_annual_savings_AB_multifit <- function(ab_multifit_obj,
                                                num_orders_used_in_ab_multifit,
                                                est_num_annual_orders,
                                                sample_prop_used = .25,
                                                comparison_fit = 'expected_loss_ratio_prod',
                                                current_siteid = 2) {
  
  
  #' @description calculate the estimated annualized savings from AB multifit based on the average number of orders per year
  #' Our models combine AMS and zZounds now so we should combine avg number of orders to consider both sites
  #' @param ab_multifit_obj the ab_multifit output object from Bootstrap_AB_multifit
  #' @param avg_annual_orders_in_ab_multifit the average number of annual orders for the specific group AB multifit was ran
  #' @param sample_prop_used the sample proportion used in AB multifit - by default it uses .25
  
  #' @Note: Breakdown of average number of annual orders by CC vs NCC and ZZ vs AMS below
  #' For zZounds CC - avg num of annual orders is ~75k
  #' For AMS CC - avg num of annual orders is ~98k
  #' For zZounds NCC - avg num of annual orders is ~170k
  #' For AMS NCC - avg num of annual orders is ~80k
  
  pct_orders_used <- num_orders_used_in_ab_multifit / est_num_annual_orders # estimate the percentage of total annual orders used in AB multifit
  
  if (class(ab_multifit_obj) == 'list') {
    
    setDT(ab_multifit_obj$ttest_df)
    
    # estimate annualized savings
    savings_against_best <- round( (( (min(ab_multifit_obj$ttest_df$LoserNetLoss) - min(ab_multifit_obj$ttest_df$WinnerNetLoss)) * (1/sample_prop_used) ) / pct_orders_used) , 2)
    
    savings_against_worst <- round( (( (max(ab_multifit_obj$ttest_df$LoserNetLoss) - min(ab_multifit_obj$ttest_df$WinnerNetLoss)) * (1/sample_prop_used) ) / pct_orders_used ), 2)
    
    if (comparison_fit %in% unique(c(ab_multifit_obj$ttest_df$Winner, ab_multifit_obj$ttest_df$Loser))) {
      
      best_prod_loss <- min(ab_multifit_obj$ttest_df[Winner == comparison_fit,]$WinnerNetLoss, ab_multifit_obj$ttest_df[Loser == comparison_fit,]$LoserNetLoss)
      
      savings_against_prod <- round( (((best_prod_loss - min(ab_multifit_obj$ttest_df$WinnerNetLoss)) * (1/sample_prop_used)) / pct_orders_used ), 2)
      
      
      return (list(savings_against_worst = savings_against_worst,
                   savings_against_best = savings_against_best,
                   savings_against_prod = savings_against_prod))
    }
    
    else {
      return (list(savings_against_worst = savings_against_worst,
                   savings_against_best = savings_against_best))
    }
  } else if (class(ab_multifit_obj) %in% c('data.table', 'data.frame')) {
    
    if (current_siteid == 2) {
      new_model_loss <- ab_multifit_obj[[names(ab_multifit_obj)[names(ab_multifit_obj) %like% paste0('zz_', prediction_colname)]]]
      prod_model_loss <- ab_multifit_obj[[names(ab_multifit_obj)[names(ab_multifit_obj) %like% paste0('zz_', comparison_fit)]]]
      dummy_model_loss <- ab_multifit_obj[[names(ab_multifit_obj)[names(ab_multifit_obj) == 'zz_dummy_loss']]]
      best_not_new_loss <- ab_multifit_obj %>% select(names(ab_multifit_obj)[!names(ab_multifit_obj) %like% paste0('zz_', prediction_colname)] & !names(ab_multifit_obj)[startsWith(names(ab_multifit_obj), 'ams')]) %>% min
    } else {
      new_model_loss <- ab_multifit_obj[[names(ab_multifit_obj)[names(ab_multifit_obj) %like% paste0('ams_', prediction_colname)]]]
      prod_model_loss <- ab_multifit_obj[[names(ab_multifit_obj)[names(ab_multifit_obj) %like% paste0('ams_', comparison_fit)]]]
      dummy_model_loss <- ab_multifit_obj[[names(ab_multifit_obj)[names(ab_multifit_obj) == 'ams_dummy_loss']]]
      best_not_new_loss <- ab_multifit_obj %>% select(names(ab_multifit_obj)[!names(ab_multifit_obj) %like% paste0('ams_', prediction_colname)] & !names(ab_multifit_obj)[startsWith(names(ab_multifit_obj), 'zz')]) %>% min
      
    }
    
    savings_against_best <- round( (( (best_not_new_loss - new_model_loss) * (1/sample_prop_used) ) / pct_orders_used) , 2)
    savings_against_worst <- round( (( (max(prod_model_loss, dummy_model_loss, best_not_new_loss) - new_model_loss) * (1/sample_prop_used) ) / pct_orders_used ), 2)
    savings_against_prod <- round( (( (prod_model_loss - new_model_loss) * (1/sample_prop_used) ) / pct_orders_used) , 2)
    
    return (list(savings_against_worst = savings_against_worst,
                 savings_against_best = savings_against_best,
                 savings_against_prod = savings_against_prod))
  }
}


calc_pred_cancel <- function(ab_obj,
                             dt_in,
                             by_installment = 1,
                             prediction_colname = 'ELR',
                             prod_fit = 'expected_loss_ratio_prod',
                             pred_cancel_thresholds_by_installment = 'ab_multifit',
                             pred_class_suffix = '_pred_cancel'
) {
  
  
  #' @description creates pred_cancel and prod_model_canceled columns based on the ab_multifit results
  #' @param ab_obj ab_multifit obj from the uw workflow - must have both zZ and AMS ab_multifit object inside named ab_multifit_list_zZ and ab_multifit_list_AMS
  #' @param dt_in data.table of at least orders and predictions 
  #' @param by_installment can be 0 or 1 - use whatever is passed to ab_dualfit_wrapper
  #' @param prediction_colname new model fit column name (e.g. prediction_colname)
  #' @param prod_fit rare use case - only used if by_installment != 0 or != 1...it's the main head to head fit column name that will be compared to the new ML models fit
  #' @param pred_cancel_thresholds_by_installment a list or string
  #'    if string use optimal ab_multifit thresholds
  #'    if list then use those thresholds for prediction cutoff to cancel an order
  #' @param pred_class_suffix suffix to append to the predicted class based on the fits in the ab_multifit fit_mapping df
  #' @Note: Must have column num_installments because sometimes num_installments can be scaled from the prep_features function
  #' @returns the original R object with datain having the pred_cancel column input
  
  ab_obj <- copy(ab_obj)
  
  setDT(dt_in)
  
  stopifnot(nrow(ab_obj$fitmapping) >= 2)
  
  setDT(ab_obj$fitmapping)
  setDT(ab_obj$result_df)
  setDT(ab_obj$ttest_df)
  
  if (by_installment == 0) {
    
    ds_print('pred_cancel will be calculated by avg - not by installment\n')
    
    for (fit in ab_obj$fitmapping$Name) {
      ab_thresholds_i <- ab_obj$result_df[,
                                          .(
                                            tmp_fit = mean(get(paste0('atThreshold', '_', ab_obj$fitmapping[Name == fit, ]$Id)))
                                          )
                                          ]
      names(ab_thresholds_i) <- fit
      dt_in[[(paste0(fit, pred_class_suffix))]] <- 0
      dt_in[as.numeric(as.character(get(fit))) >= as.numeric(as.character(ab_thresholds_i[[fit]])), (paste0(fit, pred_class_suffix)) := 1]
      
    }
    
  } else if (by_installment == 1) {
    
    ds_print('pred_cancel will be predicted by installment\n')
    
    ab_thresholds <- list()
    
    for (fit in as.character(ab_obj$fitmapping$Name)) {
      
      dt_in[,(paste0(fit, pred_class_suffix)) := 0]
      
      for (npay in unique(as.numeric(as.character(dt_in$num_installments)))) {
        paythres <- paste0('atThreshold_', npay, 'P_')
        
        if (paste0(paythres, '1') %in% names(ab_obj$result_df)) {
          
          stopifnot(paste0(paythres, '2') %in% names(ab_obj$result_df))
          
          ab_thresholds_i <- ab_obj$result_df[,
                                              .(
                                                tmp_fit = mean(get(paste0(paythres, ab_obj$fitmapping[Name == fit, ]$Id)))
                                              )
                                              ]
          names(ab_thresholds_i) <- fit
          
          ab_thresholds[[toString(npay)]] <- ab_thresholds_i
          
          dt_in[as.numeric(as.character(get(fit))) >= ab_thresholds_i[[fit]] & as.numeric(as.character(num_installments)) == npay, (paste0(fit, pred_class_suffix)) := 1]
        }
      }
      
    }
    
  } else {stop('Input params are not correct!')}
  
  outlist <- list(dt_out = dt_in,
                  ab_thresholds = ab_thresholds)
  
  return (outlist)
}


onerow_barplot <- function(dt_in,
                           xlabel = 'x',
                           ylabel = 'y',
                           scales = 'free',
                           facet_type = facet_wrap,
                           angle = 0,
                           geom_text_size = 5,
                           text_size = 10,
                           title_size = 5,
                           show.legend = F,
                           sort_x = 'desc',
                           plot_title = 'onerow_barplot') {
  #' @description plot a barplot of a data.frame or data.table with a single row
  #' @param dt_in data.table with one row
  #' @param xlabel x label
  #' @param ylabel y label
  #' @param facet_type pass the facet_type (e.g. facet_grid or facet_wrap)
  #' @param angle angle of the x label
  #' @param geom_text_size text size of geom_text in ggplot
  #' @param text_size text size in ggplot
  #' @param title_size title size in ggplot
  #' @param show.legend T or F to show the legend in ggplot
  #' @param sort_x either 'asc' or 'desc' to sort the x values after stacking the DT
  #' @param plot_title the ggplot title
  
  dt_in <- dt_in %>% stack() %>% as.data.table()
  
  if (sort_x == 'desc') {
    dt_in <- dt_in %>% arrange(values)
    dt_in$ind <- factor(dt_in$ind,levels = dt_in$ind)
  } else if (sort_x == 'asc') {
    dt_in <- dt_in %>% arrange(desc(values))
    dt_in$ind <- factor(dt_in$ind,levels = dt_in$ind)
  }
  
  stopifnot(nrow(dt_in) > 0)
  
  my_plot <- 
    dt_in %>% 
    ggplot(aes(x = ind, 
               y = round(values, 2),
               label = as.character(round(values, 2)),
               fill = ind)) +
    geom_col(position = 'dodge', show.legend = show.legend) +
    geom_text(position = position_stack(vjust = .5), size = 5) + 
    theme(text = element_text(size = text_size),
          plot.title = element_text(hjust = 0.5)) +
    xlab(xlabel) +
    ylab(ylabel) +
    ggtitle(plot_title)
  
  return(my_plot)
}










stack_dt <- function(named_vec,
                     sort_values = F,
                     col_names = c("variable", "value"),
                     order = -1L) {
  
  #' @title Casts a named vector into a data.table
  #' @description this converts the utils::stack function to data.table conventions
  #' @param sort_values should the value column be sorted descendingly?
  #' @param col_names the 2 names of the columns in the data.table, the 1st one is the vector names
  #' @param order determines the order of the sort_values
  #' 
  #' @return a 2-column data.table
  # named_vec <- setNames(letters, letters); sort_values = T
  
  new_DT <- stack(named_vec) 
  
  setDT(new_DT)
  setcolorder(new_DT, sort(names(new_DT)))  # make ind the 1st column
  new_DT[, ind := as.character(ind)]        # make ind a character vector
  if (sort_values) setorderv(new_DT, "values", order = -1L)
  
  setnames(new_DT, col_names) 
  
  return(new_DT)
}







###### primarily from fraud ######

count_nas_by_column <- function(DT, sort_values = T, drop_zero_na_cols = T, col_names = c("column", "n_NAs", "pct_NAs")){
  
  #'@title Counts the NAs by Column
  #' 
  #' @param DT a data.table
  #' @param sort_values should the NA counts be sorted?
  #' @param drop_zero_na_cols if a column has no NAs, should it be dropped from the summary?
  #' 
  #' @return a data.table, containing the columns column, n_NAs, & pct_NAs
  
  n_rows <- nrow(DT)
  
  na_counts_dt <- 
    parallel::mclapply(DT, function(col) sum(is.na(col))) %>% 
    stack_dt(., sort_values = sort_values) %>% 
    .[, pct := value / n_rows * 100] %>% 
    {if (drop_zero_na_cols) .[value > 0] else .}
  
  data.table::setnames(na_counts_dt, col_names)
  
  return(na_counts_dt)
}


min_max_scaler <- function(vec, min_value = 0, max_value = 1){
  
  #' @title Rescale the Numeric Vector between the Min and Max Values if Needed
  #' 
  #' @description This function enables us to use features like the MaxMind RiskScore as a fit.
  #' If all values are already between the min and max values, 
  #' the vector is returned without changes. Any NAs in the input will also be returned.
  #' 
  #' Also, this function tries to coerce non-numeric vectors to numeric
  #' 
  #' @param vec a numeric vector
  #' @param min_value a number
  #' @param max_value a number
  #'
  #' @return a scaled vector
  
  
  # try to coerce to numeric
  if (is.factor(vec) || is.character(vec)) vec <- vec %>% as.character %>% as.numeric
  if (is.logical(vec)) vec <- vec %>% as.integer
  
  # stop if all NAs or NULL
  stopifnot(length(na.omit(vec)) > 0)
  
  # check if scaling is needed 
  already_scaled <- all(data.table::between(na.omit(vec), min_value, max_value))
  
  if (already_scaled){
    
    vec
    
  } else {
    
    scales::rescale(vec, to = c(min_value, max_value))
    
  }
}



tabulate_confusion_matrix_by_fit <- function(DT,
                                             target_class,
                                             fits, 
                                             evaluation_colname = 'dataset_split',
                                             evaluation_datasets = c('train', 'val', 'test')) {
  
  #' @title Run tabulate_confusion_matrix across multiple fits
  #' @note can only be used with a classification dependent variable
  #' @calls tabulate_confusion_matrix
  #' @param target_class column name for the column containing observed / actual classes
  #' @param fits name of the fit columns
  #' @param evaluation_colname the name of the column that contains the evaluation_datasets (train, val, test, etc.)
  #' @param evaluation_datasets a character vector of datasets that should be plotted from the evaluation_colname
  #' @return a tabulate_confusion_matrix data.table with the fit and evaluation_colname columns
  #' DT <- output_list$dt_full; fits = evaluation_fits
  
  required_cols <- c(evaluation_colname, target_class, fits)
  
  stopifnot(all(required_cols %in% names(DT)))
  
  # cast target_class as factor
  if (!is.factor(DT[[target_class]])) DT[, (target_class) := as.factor(get(target_class))]
  
  
  ##### create, melt and group-by calculate #####
  
  suppressWarnings({
    
    grouped_metrics_dt <-
      DT[get(evaluation_colname) %in% evaluation_datasets, ..required_cols] %>% # select only the columns & datasets needed
      melt(., measure.vars = fits,                                            # melt all the fit columns to a single class_prob column
           variable.name = "fit",
           value.name = "pred_class",
           value.factor = T) %>%
      .[, correct := as.factor(get(target_class) == pred_class)]
    
    grouped_metrics_dt <- grouped_metrics_dt[, .N,
                                             by = c('fit', evaluation_colname, 'correct',
                                                    target_class, 'pred_class')][,
                                                                            prop := N / sum(N),
                                                                            by = c('fit', evaluation_colname)]
  })
  return(grouped_metrics_dt)
}


pairwise_correlation <- function(DT,
                                 method = "pearson",
                                 use = "pairwise.complete.obs") {
  
  #' @title Calculate the Unique Pairwise Correlations
  #' 
  #' @param DT a data.table or data.frame
  #' @param method the same argument found in cor
  #' @param use the same argument found in cor
  #' 
  #' @return a data.table with the columns V1, V2, r & r_squared, 
  #'   sorted descendingly by r_squared
  #'   V1 & V2 are row sorted for easier interpretation
  
  cor_dt <-
    DT[, .SD, .SDcols = function(x) uniqueN(x) > 1] %>%
      cor(., use = use, method = method) %>%
      replace(., lower.tri(., diag = T), NA) %>%
      data.table(keep.rownames = T) %>%
      melt() %>%
      na.omit %>%
      .[, 1:2 := .SD %>% apply(., 1, sort) %>% t() %>% data.table::as.data.table(), .SDcols=1:2]
  
  setnames(cor_dt, c("col1", "col2", "corr"))
  cor_dt[, r_squared := corr^2]
  setorder(cor_dt, -r_squared, -corr, col1, col2)
  
  cor_dt

}


summarize_empty_cols <- function(DT,
                              empty_values = c(NA, NULL, '<NA>', 'NA', '', ' ', 'NULL'),
                              nthreads = parallel::detectCores()) {
  
  stopifnot(class(DT) %in% c('data.frame', 'data.table'))
  if (!is.data.table(DT)) setDT(DT)
  
  agg_fns <-
    function(x)
      list(
        n_unique = uniqueN(x),
        n_missing = sum(ifelse(x %in% empty_values, 1, 0))
        )
  
  missing_summary <-
    DT[, unlist(.(N.N = .N, unlist(mclapply(.SD, agg_fns, mc.cores = nthreads), recursive = F)), recursive = F)] %>%
    melt.data.table() %>%
    as.data.table(., keep.rownames = T) %>%
    tidyr::separate(., col='variable', into=c('variable', 'agg_fn'), sep='\\.') %>%
    data.table::dcast(., variable ~ agg_fn, fun.aggregate = function(x) head(x, 1), fill=NA) %>%
    .[, N := lapply(.SD, zoo::na.locf), .SDcols = 'N'] %>%
    .[variable != 'N',] %>%
    .[, pct_missing := (n_missing / N)] %>%
    .[order(pct_missing, decreasing = T), ] %>%
    .[, pct_missing := paste0(round(pct_missing * 100, 3), '%')]
  
  return(missing_summary)
}
