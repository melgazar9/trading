library('compiler')

Bootstrap_AB_multifit_Installments <- function(testData_df, fits, sampleProp = .25, reps = 25,
                                               simple_return = 1, thresholdCutOffs=c(), currentThresholds=1,
                                               ByInstallment=1,
                                               prodThresholds=list("3" = 1, "4" = 1, "5" = 0.15, "6" = 0.15, "8" = 0.20, "12" = 0.20, "18" = 0.25, "default" = 0.25)){
  #' @description Returns evaluation of net-loss from false positives and false negatives
  #' for different fits and tests statistical significance of one fit v/s other in minimizing
  #' net losses
  #' 
  #' This function takes as input a test data frame (testData_df) which has at least 2 predicted model fit columns (fit_A) and (fit_B), 
  #' as well as a column indicating fraud and a column indicating total orderdollars.
  #' Additional inputs include
  #' sampleProp: number from .01 - 1. This is the proportion of the test df you would like to sample each iteration, 
  #' reps: number from 2-inf.  This is the number of times the model will iteratively resample the test df.  
  #' The more reps, the more acurate your final assessment of whether or not there is a true difference between fit_A and fit_B.  
  #' Takes about 20-40 seconds to complete for each rep, depending on test df and sampleProp size.
  #' thresholds: 0 or 1.  put 0 if you only want to compare AUC, in which case dollars arent calculated, and the t test isnt completed.  
  #' pulldollars: 0 or 1.  1 if you already have a column called orderdollars in your df, and 1 if you do not and want toe AssessThresholds function to pull the dollars for you.
  #' 
  #' @author Kunal Bhandari
  #' 
  #' @import None
  #' @references None
  #' 
  #' @param testData_df Input data.frame with order level data and required columns
  #' @param fits Character vector with names of fit (or predicted score) columns
  #' @param sampleProp % of data to be sampled for bootstrapping
  #' @param reps Number of times to sample data for bootstrapping
  #' @param simple_return Should extended reporting summary be returned or just a simple one
  #' @thresholdCutOffs List of additional score cut-offs that must be tested for loss evaluation
  #' @currentThresholds Boolean indicating whether to search for optimal thresholds or use provided
  #'                    input thresholds
  #'                    currentThresholds = 0 => Just test exaustive sequence of threshold values;
  #'                                             Don't test production thresholds if not included in
  #'                                             the exaustive list of thresholds
  #'                    currentThresholds = 1 => Test production thresholds along with exaustive list of
  #'                                             thresholds
  #'                    currentThresholds = 2 => Only test production thresholds
  #' @ByInstallment Boolean indicating whether to search / use different thresholds for different installment plans
  #'                ByInstallment = 0 (currentThresholds must be set to 0 or 2) => Pull at aggregated level
  #'                ByInstallment = 1 => Pull at installment level
  #' @prodThresholds List of thresholds with element names represting number of installments and values
  #'                 representing cut-off values to be treated as production cut-offs
  #' 
  #' @return A list representing evaluation of different fits with winners that minimize net-loss
  #' 
  
  #DEBUG
  #testData_df<<-testData_df
  #fits<<-fits
  #sampleProp<<-sampleProp
  #reps<<-reps
  #simple_return<<-simple_return
  #thresholdCutOffs<<-thresholdCutOffs
  #currentThresholds<<-currentThresholds
  #ByInstallment<<-ByInstallment
  #prodThresholds<<-prodThresholds
  #stop("Hello")
  #DEBUG
  #initializing the structure of the output df
  
  getCol<-function(colN, fitN, overall_cols = c("atThreshold","num_installments","totaldollars","model_net_all_profit",
                                                "production_net_all_profit","actual_production_netLoss","actual_production_profit")) {
    if (colN %in% overall_cols) return(colN)
    return(sprintf("%s_%i",colN,fitN))
  }
  
  getThresholds<-function(npays, currentThrs, fitvars=fits) {
    fit_level <- is.list(currentThrs) && length(currentThrs) > 0 && is.list(currentThrs[[1]])
    
    if(!fit_level) {
      currentThrs <- lapply(fitvars, function(x) currentThrs)
      names(currentThrs) <- fitvars
      print("getThresholds: Using same thresholds for all fitvars")
    } else if(length(currentThrs) == 1) {
      currentThrs <- lapply(fitvars, function(x) currentThrs[[1]])
      names(currentThrs) <- fitvars
      print("getThresholds: Using same thresholds for all fitvars")
    } else if(!any(fitvars %in% names(currentThrs)) && length(currentThrs) == length(fitvars)) {
      names(currentThrs) <- fitvars
      print("getThresholds: Using custom thresholds for each fitvar")
    }
    
    if(!all(fitvars %in% names(currentThrs)) &&
       length(fitvars) > 1 && length(fitvars) != length(currentThrs)) 
      stop("getThresholds: Either only one set of thresholds should be supplied or thresholds should be supplied for all fit variables")
    
    fit_lens <- c(sapply(currentThrs, function(x) {
      sapply(x, length)
    }))
    if(min(fit_lens) != max(fit_lens))
      stop("getThresholds: Threshold vector lengths don't match for different payment plans within fits")
    
    fit_lens <- sapply(currentThrs, length)
    if(min(fit_lens) != max(fit_lens))
      stop("getThresholds: Number of payment plans differ across fits in threshold vector")
    
    npays<-as.character(sort(as.numeric(as.character(npays))))
    for(i in fitvars) {
      missingThr<-setdiff(npays, names(currentThrs[[i]]))
      if (length(missingThr) > 0) {
        print(paste0("getThresholds: Missing values for prod thresholds in fit [",i,"] for pay [",
                     paste(missingThr, collapse=", "), "]"))
        if(!"default" %in% names(currentThrs[[i]])) {
          stop(paste0("getThresholds: Default threshold value not specified",
                      " to fill missing threshold values for plans for fit [",i,"]"))
          defaultValue <- 0.25
        } else defaultValue <- currentThrs[[i]][["default"]]
        missingThrValues<-rep(defaultValue, length(missingThr))
        names(missingThrValues)<-missingThr
        currentThrs[[i]]<-append(currentThrs[[i]], missingThrValues)
      }
      currentThrs[[i]]<-currentThrs[[i]][intersect(names(currentThrs[[i]]),npays)]
    }
    return(currentThrs)
  }
  
  nfits <-length(fits)
  
  if(nfits <= 1)
    stop("Bootstrap_AB_multifit_Installments: Must provide at least two fits for comparison")
  
  selected_cols <- c("atThreshold", "falsepositiveLoss", "netLoss", "loss_percent", "totaldollars", "netProfit", "model_net_all_profit",
                     "production_net_all_profit","actual_production_netLoss","actual_production_profit")
  
  if (simple_return == 1) {
    selected_cols <- selected_cols[1:5]
  }
  excludedf <- data.frame(Exclude = c("loss_percent", "adjusted_bad_debt_rate", "unadjusted_bad_debt_rate"),
                          Nume = c("netLoss", "falsenegativeLoss","falsenegative_unadjusted"),
                          Denom = c("totaldollars","totaldollars","totaldollars"), stringsAsFactors=F)
  exclude_cols <- excludedf$Exclude
  exclude_cols <- intersect(selected_cols, exclude_cols)
  selected_cols <- setdiff(selected_cols, exclude_cols)
  base_cols <- c("netLoss", "falsenegativeLoss","falsenegative_unadjusted")
  
  if (currentThresholds >= 1 && ByInstallment == 0) {
    print("ByInstallment must be set to 1 with currentThresholds IN 1,2: Overriding and setting ByInstallment = 1")
    ByInstallment <- 1
  }
  if (ByInstallment == 1) {
    npays <- sort(as.numeric(as.character(unique(testData_df$num_installments)))) #Assumes representation of every installment plan
    Nnpays <- length(npays)
    prodThresholds <- getThresholds(npays, prodThresholds, fits)
  }
  
  ATC_im <- cmpfun(AssessThresholds_Installment_Multifit)
  optimalthresholdbtsummary<-data.frame()
  prodthresholdbtsummary<-data.frame()
  r <- 1
  while( r <= reps){
    print(paste("Starting rep ", r, sep = ""))
    rowadd <- c()
    #subset the test df to a random subset of proportion sampleProp
    subdf <- testData_df[sample(1:nrow(testData_df), round((nrow(testData_df)*sampleProp)), replace=FALSE),]
    
    thresholdsSummary <- ATC_im(subdf, fits, dollarvar = 'netdollars', cutOffs=thresholdCutOffs,
                                currentThresholds=currentThresholds, prodThresholds=prodThresholds,
                                ByInstallment=ByInstallment)
    prods <- list()
    mins<-list()
    repeatIteration <- 0
    tryCatch( {
      if (ByInstallment == 1) {
        for (npay in npays) {
          mins[[as.character(npay)]] <- lapply(1:nfits, function(X) 
            return(thresholdsSummary[thresholdsSummary[[paste0("netLoss_",X)]] == min(thresholdsSummary[thresholdsSummary$num_installments == npay, paste0("netLoss_",X)]) & thresholdsSummary$num_installments == npay,]))
          if (currentThresholds > 0) {
            prods[[as.character(npay)]] <- lapply(1:nfits, function(X)
              return(thresholdsSummary[thresholdsSummary$atThreshold == as.numeric(prodThresholds[[fits[X]]][[as.character(npay)]]) & thresholdsSummary$num_installments == npay,]))
          }
        }
      } else {
        mins <- lapply(1:nfits, function(X) return(thresholdsSummary[thresholdsSummary[[paste0("netLoss_",X)]] == min(thresholdsSummary[[paste0("netLoss_",X)]]),]))
        if (currentThresholds > 0)
          prods <- lapply(1:nfits, function(X) return(thresholdsSummary[thresholdsSummary$atThreshold == -1,]))
      }
    }, error = function(e) stop("Error while getting mins for threshold summaries"),
    warning = function(w) {
      repeatIteration <<- 1
    })
    
    if (repeatIteration == 1) {
      print("Restarting rep due to biased sample pull")
      rm(repeatIteration)
      next
    }
    
    newrow<-nrow(optimalthresholdbtsummary)+1
    for (i in 1:nfits) {
      for (colN in selected_cols) {
        if (ByInstallment == 1) {
          optimalthresholdbtsummary[newrow,sprintf("%s_%d",colN,i)] <- 0
          if (currentThresholds > 0) prodthresholdbtsummary[newrow,sprintf("%s_%d",colN,i)] <- 0
          for (npay in npays) {
            if (colN == "atThreshold") {
              optimalthresholdbtsummary[newrow,sprintf("%s_%dP_%d",colN,npay,i)] <- mins[[as.character(npay)]][[i]][1, getCol(colN,i)]
              if (currentThresholds > 0) prodthresholdbtsummary[newrow,sprintf("%s_%dP_%d",colN,npay,i)] <- prods[[as.character(npay)]][[i]][1, getCol(colN,i)]
            } else {
              optimalthresholdbtsummary[newrow,sprintf("%s_%d",colN,i)] <- optimalthresholdbtsummary[newrow,sprintf("%s_%d",colN,i)] +
                mins[[as.character(npay)]][[i]][1, getCol(colN,i)]
              if (currentThresholds > 0) {
                prodthresholdbtsummary[newrow,sprintf("%s_%d",colN,i)] <- prodthresholdbtsummary[newrow,sprintf("%s_%d",colN,i)] +
                  prods[[as.character(npay)]][[i]][1, getCol(colN,i)]
              }
            }
          }
        } else {
          optimalthresholdbtsummary[newrow,sprintf("%s_%d",colN,i)] <-  mins[[i]][1, getCol(colN,i)]
          if (currentThresholds > 0) prodthresholdbtsummary[newrow,sprintf("%s_%d",colN,i)] <- prods[[i]][1, getCol(colN,i)]
        }
      }
      for (colN in exclude_cols) {
        optimalthresholdbtsummary[newrow,sprintf("%s_%d",colN,i)] <- optimalthresholdbtsummary[newrow,sprintf("%s_%d", excludedf[excludedf$Exclude==colN,"Nume"], i)] /
          optimalthresholdbtsummary[newrow,sprintf("%s_%d", excludedf[excludedf$Exclude==colN,"Denom"], i)]
        if (currentThresholds == 1) {
          prodthresholdbtsummary[newrow,sprintf("%s_%d",colN,i)] <- prodthresholdbtsummary[newrow,sprintf("%s_%d", excludedf[excludedf$Exclude==colN,"Nume"], i)] /
            prodthresholdbtsummary[newrow,sprintf("%s_%d", excludedf[excludedf$Exclude==colN,"Denom"], i)]
        }
      }
    }
    
    r <- r + 1
  }
  
  testGrid <- expand.grid(fitA=1:nfits, fitB=1:nfits)
  testGrid <- testGrid[testGrid$fitA < testGrid$fitB,]
  
  comparisons<-data.frame()
  totalcomparisonscompleted <- 0
  totalComparisons <-nrow(testGrid)
  
  while(nrow(testGrid) > 0) {
    totalcomparisonscompleted <- totalcomparisonscompleted + 1
    fita <- testGrid[1, "fitA"]
    fitb <- testGrid[1, "fitB"]
    netlossaCol <- paste0("netLoss_",fita)
    netlossbCol <- paste0("netLoss_",fitb)
    losspaCol <- paste0("loss_percent_",fita)
    losspbCol <- paste0("loss_percent_",fitb)
    
    myTtest <- t.test(optimalthresholdbtsummary[,netlossaCol], optimalthresholdbtsummary[,netlossbCol], paired = TRUE)
    
    netlossa <- mean(optimalthresholdbtsummary[,netlossaCol])
    netlossb <- mean(optimalthresholdbtsummary[,netlossbCol])
    losspa<- mean(optimalthresholdbtsummary[,losspaCol])
    losspb<- mean(optimalthresholdbtsummary[,losspbCol])
    
    if (netlossa > netlossb) {
      winner<-testGrid[1, 2]
      loser<-testGrid[1, 1]
      nl_winner<-netlossb
      lp_winner<-losspb
      nl_loser<-netlossa
      lp_loser<-losspa
    } else {
      winner<-testGrid[1, 1]
      loser<-testGrid[1, 2]
      nl_winner<-netlossa
      lp_winner<-losspa
      nl_loser<-netlossb
      lp_loser<-losspb
    }
    testGrid <- testGrid[-c(1),]
    comparisons[totalcomparisonscompleted, "Winner"] <- fits[winner]
    comparisons[totalcomparisonscompleted, "Loser"] <- fits[loser]
    comparisons[totalcomparisonscompleted, "pVal"] <- myTtest$p.value
    comparisons[totalcomparisonscompleted, "WinnerNetLoss"] <- nl_winner
    comparisons[totalcomparisonscompleted, "LoserNetLoss"] <- nl_loser
    comparisons[totalcomparisonscompleted, "WinnerLossP"] <- lp_winner
    comparisons[totalcomparisonscompleted, "LoserLossP"] <- lp_loser
    comparisons[totalcomparisonscompleted, "GainDiff"] <- lp_loser - lp_winner
    comparisons[totalcomparisonscompleted, "GainP"] <- (lp_loser - lp_winner) / lp_winner
    if (ByInstallment == 1) {
      for (npay in npays) {
        meanThreshold <- mean(optimalthresholdbtsummary[,sprintf("atThreshold_%dP_%d",npay,winner)])
        sdThreshold <- sd(optimalthresholdbtsummary[,sprintf("atThreshold_%dP_%d",npay,winner)])
        comparisons[totalcomparisonscompleted, sprintf("atThreshold5_%d",npay)] <- meanThreshold - 1.96 * sdThreshold / sqrt(reps)
        comparisons[totalcomparisonscompleted, sprintf("atThresholdAvg_%d",npay)] <- meanThreshold
        comparisons[totalcomparisonscompleted, sprintf("atThreshold95_%d",npay)] <- meanThreshold + 1.96 * sdThreshold / sqrt(reps)
        comparisons[totalcomparisonscompleted, sprintf("atThresholdMin_%d",npay)] <- min(optimalthresholdbtsummary[,sprintf("atThreshold_%dP_%d",npay,winner)])
        comparisons[totalcomparisonscompleted, sprintf("atThresholdMax_%d",npay)] <- max(optimalthresholdbtsummary[,sprintf("atThreshold_%dP_%d",npay,winner)])
      }
    } else {
      meanThreshold <- mean(optimalthresholdbtsummary[,sprintf("atThreshold_%d",winner)])
      sdThreshold <- sd(optimalthresholdbtsummary[,sprintf("atThreshold_%d",winner)])
      comparisons[totalcomparisonscompleted, "atThreshold5"] <- meanThreshold - 1.96 * sdThreshold / sqrt(reps)
      comparisons[totalcomparisonscompleted, "atThresholdAvg"] <- meanThreshold
      comparisons[totalcomparisonscompleted, "atThreshold95"] <- meanThreshold + 1.96 * sdThreshold / sqrt(reps)
      comparisons[totalcomparisonscompleted, "atThresholdMin"] <- min(optimalthresholdbtsummary[,sprintf("atThreshold_%d",winner)])
      comparisons[totalcomparisonscompleted, "atThresholdMax"] <- max(optimalthresholdbtsummary[,sprintf("atThreshold_%d",winner)])
    }
    
    if (!is.na(myTtest$p.value) && myTtest$p.value <= 0.10) {
      testGrid<-testGrid[testGrid$fitA!=loser, ]
      testGrid<-testGrid[testGrid$fitB!=loser, ]
    }
    print(sprintf("%d/%d (%d%%) more fit comparisons to go", nrow(testGrid), totalComparisons, round(nrow(testGrid)*100/totalComparisons)))
  }
  
  if (currentThresholds == 1) {
    testGrid <- expand.grid(fitA=1:nfits, fitB=1:nfits)
    testGrid <- testGrid[testGrid$fitA < testGrid$fitB,]
    
    comparisonsProd<-data.frame()
    totalcomparisonscompleted <- 0
    totalComparisons <-nrow(testGrid)
    
    while(nrow(testGrid) > 0) {
      totalcomparisonscompleted <- totalcomparisonscompleted + 1
      fita <- testGrid[1, "fitA"]
      fitb <- testGrid[1, "fitB"]
      netlossaCol <- paste0("netLoss_",fita)
      netlossbCol <- paste0("netLoss_",fitb)
      losspaCol <- paste0("loss_percent_",fita)
      losspbCol <- paste0("loss_percent_",fitb)
      
      myTtest <- t.test(prodthresholdbtsummary[,netlossaCol], prodthresholdbtsummary[,netlossbCol], paired = TRUE)
      
      netlossa <- mean(prodthresholdbtsummary[,netlossaCol])
      netlossb <- mean(prodthresholdbtsummary[,netlossbCol])
      losspa<- mean(prodthresholdbtsummary[,losspaCol])
      losspb<- mean(prodthresholdbtsummary[,losspbCol])
      
      if (netlossa > netlossb) {
        winner<-testGrid[1, 2]
        loser<-testGrid[1, 1]
        nl_winner<-netlossb
        lp_winner<-losspb
        nl_loser<-netlossa
        lp_loser<-losspa
      } else {
        winner<-testGrid[1, 1]
        loser<-testGrid[1, 2]
        nl_winner<-netlossa
        lp_winner<-losspa
        nl_loser<-netlossb
        lp_loser<-losspb
      }
      testGrid <- testGrid[-c(1),]
      comparisonsProd[totalcomparisonscompleted, "Winner"] <- fits[winner]
      comparisonsProd[totalcomparisonscompleted, "Loser"] <- fits[loser]
      comparisonsProd[totalcomparisonscompleted, "pVal"] <- myTtest$p.value
      comparisonsProd[totalcomparisonscompleted, "WinnerNetLoss"] <- nl_winner
      comparisonsProd[totalcomparisonscompleted, "LoserNetLoss"] <- nl_loser
      comparisonsProd[totalcomparisonscompleted, "WinnerLossP"] <- lp_winner
      comparisonsProd[totalcomparisonscompleted, "LoserLossP"] <- lp_loser
      comparisonsProd[totalcomparisonscompleted, "GainDiff"] <- lp_loser - lp_winner
      comparisonsProd[totalcomparisonscompleted, "GainP"] <- (lp_loser - lp_winner) / lp_winner
      
      if (ByInstallment == 1) {
        for (npay in npays) {
          meanThreshold <- prodThresholds[[fits[winner]]][[as.character(npay)]]
          comparisonsProd[totalcomparisonscompleted, sprintf("atThreshold5_%d",npay)] <- meanThreshold
          comparisonsProd[totalcomparisonscompleted, sprintf("atThresholdAvg_%d",npay)] <- meanThreshold
          comparisonsProd[totalcomparisonscompleted, sprintf("atThreshold95_%d",npay)] <- meanThreshold
          comparisonsProd[totalcomparisonscompleted, sprintf("atThresholdMin_%d",npay)] <- meanThreshold
          comparisonsProd[totalcomparisonscompleted, sprintf("atThresholdMax_%d",npay)] <- meanThreshold
        }
      } else {
        meanThreshold <- -1
        comparisonsProd[totalcomparisonscompleted, "atThreshold5"] <- -1
        comparisonsProd[totalcomparisonscompleted, "atThresholdAvg"] <- -1
        comparisonsProd[totalcomparisonscompleted, "atThreshold95"] <- -1
        comparisonsProd[totalcomparisonscompleted, "atThresholdMin"] <- -1
        comparisonsProd[totalcomparisonscompleted, "atThresholdMax"] <- -1
      }
      
      if (!is.na(myTtest$p.value) && myTtest$p.value <= 0.10) {
        testGrid<-testGrid[testGrid$fitA!=loser, ]
        testGrid<-testGrid[testGrid$fitB!=loser, ]
      }
      print(sprintf("Prod cutoffs %d/%d (%d%%) more fit comparisons to go", nrow(testGrid), totalComparisons, round(nrow(testGrid)*100/totalComparisons)))
    }
  }
  
  finresults <- list()
  finresults$result_df <- optimalthresholdbtsummary
  finresults$ttest_df <- comparisons
  if (currentThresholds == 1) {
    finresults$bt_results_prod_thresholds <- prodthresholdbtsummary
    finresults$ttest_df_prod_thresholds <- comparisonsProd
  }
  finresults$fitmapping <- data.frame(Id = 1:length(fits), Name=fits)
  
  return(finresults)
}


AssessThresholds_Installment_Multifit <- function(internaldata, fitvars, dollarvar="netdollars",
                                                  writeoffvar="actual_loss_dollars", profitvar="profitdollars", 
                                                  overhead_cost = .15, dependentVar=NA, cutOffs=list(), currentThresholds=0,
                                                  ByInstallment = 0,
                                                  prodThresholds=list("3" = 1, "4" = 1, "5" = 0.15, "6" = 0.15, "8" = 0.20, "12" = 0.20, "18" = 0.25, "default" = 0.25)){
  
  #' @description Returns evaluation of net-loss from false positives and false negatives
  #' at different cut-off thresholds to determine optimal threshold value
  #' 
  #' @author Kunal Bhandari
  #' 
  #' @import None
  #' @references None
  #' 
  #' @param internaldata Input data.frame with order level data and required columns
  #' @param fitvars Character vector with names of fit (or predicted score) columns
  #' @param dollarvar Character specifying name of the column that stores net dollars
  #' @param writeoffvar Character specifying name of the column that stores write-off dollars
  #' @param profitvar Character specifying name of the column that stores profit dollars
  #' @param overhead_cost Decimal value < 1 representing overhead cost as a percentage of net dollars
  #' @param dependentVar Character specifying dependent variable column name
  #'                     (if one would like to analyze false positives and negatives)
  #' @cutOffs List of additional score cut-offs that must be tested for loss evaluation
  #' @currentThresholds Boolean indicating whether to search for optimal thresholds or use provided
  #'                    input thresholds
  #'                    currentThresholds = 0 => Just test exaustive sequence of threshold values;
  #'                                             Don't test production thresholds if not included in
  #'                                             the exaustive list of thresholds
  #'                    currentThresholds = 1 => Test production thresholds along with exaustive list of
  #'                                             thresholds
  #'                    currentThresholds = 2 => Only test production thresholds
  #' @ByInstallment Deprecated - Boolean indicating whether to search / use different thresholds for different installment plans
  #'                ByInstallment = 0 (currentThresholds must be set to 0 or 2) => Pull at aggregated level
  #'                ByInstallment = 1 => Pull at installment level
  #' @prodThresholds List of thresholds with element names represting number of installments and values
  #'                 representing cut-off values to be treated as production cut-offs
  #' 
  #' @return A list representing evaluation of different fits with winners that minimize net-loss
  #' 
  #'  
  
  
  # if currentThresholds = 2 and ByInstallment = 0 only then aggregate installment level
  thresholdSummarySQL<-function(fitvars, writeoffvar, dollarvar, dependentvar, currentThresholds,
                                ByInstallment, add_suffix=0, thresholdvector="thresholdvector", orderdata="internaldata",
                                output = "summaryVals") {
    if (ByInstallment == 1) {
      groupbySQL <- "t.num_installments, t.atThreshold"
      joinSQL <- "ON t.num_installments = i.num_installments"
    } else {
      groupbySQL <- "t.atThreshold"
      joinSQL <- "ON 1=1"
    }
    
    commonSQL<-sprintf("count(1) as num_rows,
                       abs(sum(actual_netLoss)) as actual_production_netLoss,
                       abs(sum(actual_profit)) as actual_production_profit,
                       (abs(sum(actual_profit)) - abs(sum(actual_netLoss))) as production_net_all_profit,
                       sum(%s) as unadjusted_production_net_Loss,
                       sum(%s) as totaldollars", writeoffvar, dollarvar)
    
    suffix <- ""
    repeatSQL<-""
    nvars <- length(fitvars)
    for (i in 1:nvars) {
      fitvar <- fitvars[i]
      if (nvars > 1 || add_suffix == 1) suffix <- paste0("_",i)
      
      if (is.na(dependentvar)) {
        tmpSQL<-sprintf(",NULL as tp%s, NULL as p%s, NULL as truepositiverate%s, NULL as falsepositiverate%s",
                        suffix, suffix, suffix, suffix)
      } else {
        tmpSQL<-sprintf(",sum(Case When %s > atThreshold AND %s = 1 Then 1 Else 0 End) as tp%s,
                        sum(Case When %s > 1 Then 1 Else 0 End) as p%s,
                        sum(Case When %s > atThreshold AND %s = 1 Then 1 Else 0 End) /
                        sum(Case When %s > 1 Then 1 Else 0 End) as truepositiverate%s,
                        (sum(Case When %s > atThreshold Then 1 Else 0 End) - sum(Case When %s > atThreshold AND %s = 1 Then 1 Else 0 End)) /
                        (count(1) - sum(Case When %s > 1 Then 1 Else 0 End)) as falsepositiverate%s",
                        fitvar, dependentvar, suffix, dependentvar, suffix,
                        fitvar, dependentvar, dependentvar, suffix, fitvar, fitvar, dependentvar, dependentvar, suffix)
      }
      
      repeatSQL<-sprintf("%s,
                         abs(sum(Case When %s > atThreshold Then falsepositiveLoss Else 0 End)) as falsepositiveLoss%s,
                         abs(sum(Case When %s <= atThreshold Then falsenegativeLoss Else 0 End)) as falsenegativeLoss%s,
                         sum(Case When %s <= atThreshold Then %s Else 0 End) as falsenegative_unadjusted%s,
                         abs(sum(Case When %s <= atThreshold Then netProfit Else 0 End)) as netProfit%s,
                         sum(Case When %s > atThreshold Then 1 Else 0 End) as cancel_count%s,
                         sum(Case When %s > atThreshold AND %s = 0 Then 1 Else 0 End) as payed_in_full_canceled%s,
                         sum(Case When %s <= atThreshold Then 1 Else 0 End) as pass_count%s,
                         abs(sum(Case When %s > atThreshold Then falsepositiveLoss Else 0 End)) + 
                         abs(sum(Case When %s <= atThreshold Then falsenegativeLoss Else 0 End)) as netLoss%s,
                         abs(sum(Case When %s <= atThreshold Then netProfit Else 0 End)) - 
                         abs(sum(Case When %s <= atThreshold Then falsenegativeLoss Else 0 End)) as model_net_all_profit%s,
                         (abs(sum(Case When %s > atThreshold Then falsepositiveLoss Else 0 End)) + 
                         abs(sum(Case When %s <= atThreshold Then falsenegativeLoss Else 0 End))) /
                         sum(%s) * 100 as loss_percent%s,
                         abs(sum(Case When %s <= atThreshold Then falsenegativeLoss Else 0 End)) / sum(%s) * 100 as adjusted_bad_debt_rate%s,
                         sum(Case When %s <= atThreshold Then %s Else 0 End) / sum(%s) * 100 as unadjusted_bad_debt_rate%s
                         %s", repeatSQL, fitvar, suffix, fitvar, suffix, fitvar, writeoffvar, suffix, fitvar, suffix,
                         fitvar, suffix, fitvar, writeoffvar, suffix, fitvar, suffix, fitvar, fitvar, suffix,
                         fitvar, fitvar, suffix, fitvar, fitvar, dollarvar, suffix, fitvar, dollarvar, suffix, 
                         fitvar, writeoffvar, dollarvar, suffix, tmpSQL)
    }
    
    installmentSQL<-sprintf("Select %s, %s %s from  %s as t inner join %s as i
                            %s group by %s", groupbySQL, commonSQL, repeatSQL, thresholdvector, orderdata, joinSQL, groupbySQL)
    
    aggSQL <- ""
    if(currentThresholds == 2 && ByInstallment == 0) {
      commonSQL<-"sum(num_rows) as num_rows,
                 sum(actual_production_netLoss) as actual_production_netLoss,
                 sum(actual_production_profit) as actual_production_profit,
                 sum(production_net_all_profit) as production_net_all_profit,
                 sum(unadjusted_production_net_Loss) as unadjusted_production_net_Loss,
                 sum(totaldollars) as totaldollars"
      
      suffix <- ""
      repeatSQL<-""
      for (i in 1:nvars) {
        fitvar <- fitvars[i]
        if (nvars > 1 || add_suffix == 1) suffix <- paste0("_",i)
        
        if (is.na(dependentvar)) {
          tmpSQL<-sprintf(",NULL as tp%s, NULL as p%s, NULL as truepositiverate%s, NULL as falsepositiverate%s",
                          suffix, suffix, suffix, suffix)
        } else {
          tmpSQL<-sprintf(",sum(tp%s) as tp%s, sum(p%s) as p%s,
                          sum(tp%s) / sum(p%s) as truepositiverate%s,
                          (sum(cancel_count%s) - sum(tp%s)) /
                          (sum(num_rows) - sum(p%s) as falsepositiverate%s",
                          suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix)
        }
        
        repeatSQL<-sprintf("%s,
                           sum(falsepositiveLoss%s) as falsepositiveLoss%s,
                           sum(falsenegativeLoss%s) as falsenegativeLoss%s,
                           sum(falsenegative_unadjusted%s) as falsenegative_unadjusted%s,
                           sum(netProfit%s) as netProfit%s,
                           sum(cancel_count%s) as cancel_count%s,
                           sum(payed_in_full_canceled%s) as payed_in_full_canceled%s,
                           sum(pass_count%s) as pass_count%s,
                           sum(netLoss%s) as netLoss%s,
                           sum(model_net_all_profit%s) as model_net_all_profit%s,
                           sum(netLoss%s) / sum(totaldollars) * 100 as loss_percent%s,
                           sum(falsenegativeLoss%s) / sum(totaldollars) * 100 as adjusted_bad_debt_rate%s,
                           sum(falsenegative_unadjusted%s) / sum(totaldollars) * 100 as unadjusted_bad_debt_rate%s
                           %s", repeatSQL, suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix,
                           suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix, suffix,
                           suffix, suffix, suffix, tmpSQL)
      }
      
      aggSQL<-sprintf("Select -1 as atThreshold, %s %s from %s", commonSQL, repeatSQL, output)
    }
    return(list(installmentSQL, aggSQL))
  }
  
  missingVars <- setdiff(c(fitvars, dollarvar, writeoffvar, profitvar, dependentVar, "num_installments"), names(internaldata))
  missingVars <- missingVars[!is.na(missingVars)]
  if(length(missingVars) > 0) {
    stop(paste0("Assess Thresholds Error: Essential variables missing in the input dataset - ", paste(missingVars, sep=",")))
  }
  
  #if(min(internaldata[[fitvar]], na.rm=T) < 0 || max(internaldata[[fitvar]], na.rm=T) > 1) {
  #  print("Assess Thresholds Error: Invalid range for fitvar; Expected range 0-1")
  #  return(data.frame(Error="Assess Thresholds Error: Invalid range for fitvar; Expected range 0-1"))
  #}
  
  if (overhead_cost > 0){
    internaldata$overhead_cost <- internaldata[[dollarvar]] * overhead_cost
  }else{
    internaldata$overhead_cost <- 0
  }
  
  internaldata$actual_paid_dollars <- internaldata[[dollarvar]] - internaldata[[writeoffvar]]
  
  internaldata$falsepositiveLoss <- pmin(0, internaldata[[writeoffvar]] -
                                           internaldata[[profitvar]] +
                                           internaldata[["overhead_cost"]])
  
  internaldata$falsenegativeLoss <- pmax(0, internaldata[[dollarvar]] -
                                           internaldata[[profitvar]] -
                                           internaldata[["actual_paid_dollars"]] +
                                           internaldata[["overhead_cost"]])
  
  internaldata$netProfit <- pmin(0, internaldata[[dollarvar]] -
                                   internaldata[[profitvar]] -
                                   internaldata[["actual_paid_dollars"]] +
                                   internaldata[["overhead_cost"]])
  
  internaldata$actual_netLoss <- pmax(0, internaldata[[dollarvar]] -
                                        internaldata[[profitvar]] -
                                        internaldata[["actual_paid_dollars"]] +
                                        internaldata[["overhead_cost"]])
  
  internaldata$actual_profit <- pmin(0, internaldata[[dollarvar]] -
                                       internaldata[[profitvar]] -
                                       internaldata[["actual_paid_dollars"]] +
                                       internaldata[["overhead_cost"]])
  
  if (currentThresholds == 1 && ByInstallment == 0) print("AssessThresholds: ByInstallment must be set to 1 with currentThreshold = 1; Returning thresholds by installment")
  
  threshold_values <- c()
  if(currentThresholds <= 1) threshold_values <- seq(from = .01, to = .98, by = .01)
  if(currentThresholds >= 1) threshold_values <- sort(unique(c(threshold_values, unlist(prodThresholds))))
  if(length(cutOffs) > 0)
    threshold_values <- sort(unique(c(threshold_values, unlist(cutOffs))))
  
  if (ByInstallment == 1 || currentThresholds > 0) {
    nPays <- sort(unique(as.numeric(as.character(internaldata$num_installments))))
    thresholdvector <- expand_grid(num_installments=nPays, atThreshold=threshold_values)
  } else {
    thresholdvector <- data.frame(atThreshold = threshold_values)
  }
  
  sqlQueries <- thresholdSummarySQL(fitvars = fitvars, writeoffvar = writeoffvar, dollarvar = dollarvar,
                                    dependentvar = dependentVar, currentThresholds = currentThresholds,
                                    ByInstallment = ByInstallment, add_suffix = 0, output = "summaryVals")
  
  summaryVals <- sqldf(sqlQueries[[1]])
  
  if (currentThresholds == 2 && ByInstallment == 0) summaryVals <- sqldf(sqlQueries[[2]])
  
  return(summaryVals)
}



ATC_im <- cmpfun(AssessThresholds_Installment_Multifit)

ab_dualfit_wrapper <- function(dataset_A, dataset_B, fits, currentThresholds = 0, by_installment = 1) {
  
  #' @description This function is designed to use the optimal thresholds found in one dataset (e.g. train) and pass those thresholds to another (e.g. val set)
  #' @param dataset_A a data.frame (e.g. df_train) that will be used to find the optimal thresholds. These optimal thresholds will be passed to dataset_B.
  #' @param dataset_B a data.frame (e.g. df_test) that uses the optimal thresholds found in dataset_A
  #' An example of dataset_A would be the validation set and dataset_B would be the test set. 
  #' This wrapper uses the val set to find the optimal thresholds and passes those optimal thresholds to dataset_B
  #' @param by_installment If 1 calculate optimal thresholds for each installment plan.
  #'                       If 0 calculate the overall optimal threshold across all installments
  #'                       This is the same param as currentThresholds in Bootstrap_AB_multifit_Installments
  #' @calls Bootstrap_AB_multifit_Installments
  
  ab_out_A <- Bootstrap_AB_multifit_Installments(dataset_A,
                                                 fits = fits,
                                                 currentThresholds = currentThresholds,
                                                 ByInstallment = by_installment)
  
  setDT(ab_out_A$fitmapping)
  setDT(ab_out_A$result_df)
  setDT(ab_out_A$ttest_df)
  
  avg_optimal_thresholds_by_fit <- sapply(ab_out_A$result_df, mean) %>%
    t %>%
    as.data.table() %>%
    select(starts_with('atThres'))
  
  #if (by_installment) {avg_optimal_thresholds_by_fit <- avg_optimal_thresholds_by_fit %>% select(matches('P_'))}
  if (by_installment) {avg_optimal_thresholds_by_fit <- avg_optimal_thresholds_by_fit %>% select(names(avg_optimal_thresholds_by_fit)[!is.na(str_extract(names(avg_optimal_thresholds_by_fit), '.*[:digit:]P_'))])}
  fit_list <- list()
  
  for (f in ab_out_A$fitmapping$Name) {
    curr_id <- ab_out_A$fitmapping[Name == f,]$Id
    curr_fitnames <- names(avg_optimal_thresholds_by_fit)[endsWith(names(avg_optimal_thresholds_by_fit), toString(curr_id))]
    curr_thresholds <- avg_optimal_thresholds_by_fit[, ..curr_fitnames]
    names(curr_thresholds) <- str_extract(names(curr_thresholds), "[[:digit:]]+")
    fit_list[[f]] <- lapply(curr_thresholds, first)
  }
  
  ab_out_B <- Bootstrap_AB_multifit_Installments(dataset_B,
                                                 fits = fits,
                                                 currentThresholds = 2,
                                                 ByInstallment = by_installment,
                                                 prodThresholds = fit_list)
  
  
  outlist <- list(ab_out_A = ab_out_A,
                  ab_out_B = ab_out_B)
  
  return(outlist)   
}
