# Install packages
# Note: this process could take a couple of minutes

if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(Matrix)) install.packages("Matrix", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dslabs)) install.packages("dslabs", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(reshape2)) install.packages("reshape2", repos = "http://cran.us.r-project.org")
if(!require(Hmisc)) install.packages("Hmisc", repos = "http://cran.us.r-project.org")
if(!require(rstudioapi)) install.packages("rstudioapi", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(Brobdingnag)) install.packages("Brobdingnag", repos = "http://cran.us.r-project.org")
if(!require(parallel)) install.packages("parallel", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(skimr)) install.packages("skimr", repos = "http://cran.us.r-project.org")
if(!require(tictoc)) install.packages("tictoc", repos = "http://cran.us.r-project.org")

# Load libraries
library(rstudioapi)
library(data.table)
library(Matrix)
library(caret)
library(xgboost)
library(dslabs)
library(dplyr)
library(tidyverse)
library(ggplot2)
library(reshape2)
library(Hmisc)
library(corrplot)
library(Brobdingnag)
library(parallel)
library(doParallel)
library(skimr)
library(tictoc)

###############################################
# Loading training data dynamically:
# Download data
dl <- tempfile()
download.file("https://storagemalteenterprise.blob.core.windows.net/processdata/process_data.zip", dl)

# Load train data from csv file into dataframe
data_train <- fread(unzip(dl, "process_data_train.csv"), header = TRUE, showProgress = F)
################################################
# Check basic data structure
dim(data_train)
str(data_train[, c(1, 2:4, 970)]) # structure of the data, due to huge number of columns only showing some (Id, 3 features, response)
summary(data_train[, c(1, 2:4, 970)]) # summary of the data, due to huge number of columns only showing some (Id, 3 features, response)

# plot of the quality observations
data.frame(as.factor(data_train$Response)) %>%
  group_by(as.factor.data_train.Response.) %>%
  summarise(cnt = n()) %>%
  ggplot(aes(x= (as.factor.data_train.Response.), y=cnt)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label=cnt),position=position_dodge(width=0.9), vjust=-0.25) +
  labs(x="Quality Outcome", y="Number Of Observations", caption = "source data: data_train set") +
  ggtitle("Number Of Quality Observations")

# Ratio of "Fail" observations
sum(data_train$Response)/nrow(data_train)

##########################################
# check occurence of missing values (NA's):
missingCount <- sapply(data_train, function(x) sum(is.na(x)))                                 # count NA's per feature column
missingCount <- data.frame(missingCount)                                                      # convert to dataframe
missingCount$name <- rownames(missingCount)                                                   # add feature names
missingCount <- missingCount[order(missingCount$missingCount),]                               # order descending (most NA's on top)
subset(missingCount, missingCount == 0)                                                       # check which have no missing values
# the columns "id" and "response" are the only ones having 0 missing values, so these columns I filter out
missingCount <- subset(missingCount, missingCount > 0)                                        # remove ID + Response
missingCount["missingRatio"] <- missingCount$missingCount / nrow(data_train)                     # add ratio column

par(mfrow=c(1,2))    # set the plotting area into a 1*2 array
ggplot(missingCount, aes(x=missingCount)) +
  geom_histogram(bins=50) +
  scale_x_continuous(labels = function(x) format(x, big.mark = ",", scientific = FALSE)) +
  ggtitle("Histogram: Missing Values of Features") +
  labs(x="Number Of Missing Values", y="Variables", caption = "source data: data_train set")

ggplot(missingCount, aes(x=missingRatio)) +
  geom_histogram(bins=50) +
  ggtitle("Histogram: Missing Ratio of Features") +
  labs(x="Missing Ratio", y="Variables", caption = "source data: data_train set")

rm(missingCount)
gc()

#########################################
######## Manual Feature Analysis ########
#########################################
# As example, we do the analysis only with the first 20 features. Otherwise too much computation is needed.
dt_selectedFeatures <- data_train[, 2:21] # not select the 1st column because thats "Id"
skim(dt_selectedFeatures)[, c(1:6, 12)] # get quick overview of features

# Is there a correlation between a feature and the Reponse?
# How strong the correlation?
# Is it a positive or negative correlation?

# Compute correlation between features and Response
# Since the function does not work with NA's,  they must be replaced.
# We replace them with the columns mean, so that the new values we impute do not influence the correlation.
func_NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE)) # function to replace NA's with column mean value
dt_selectedFeatures <- replace(dt_selectedFeatures, TRUE, lapply(dt_selectedFeatures, func_NA2mean)) # replace NA's with column mean in dataset
anyNA(dt_selectedFeatures) # check if any NA is left

# Now let us calculate at the correlation coefficient of each of the choosen features
dt_corr <- data.frame(names(dt_selectedFeatures), sapply(dt_selectedFeatures, function(x) { cor(x, data_train$Response) }))
names(dt_corr) <- c("variableName", "correlationValue")
rownames(dt_corr) <- seq(1,20)
# Let's plot the results
ggplot(data = dt_corr, aes(x = reorder(variableName, correlationValue), y = correlationValue)) +
  geom_bar(stat = "identity", width = 0.5) +
  coord_flip() +
  ggtitle("Barplot: Correlation of selected features with Quality Outcome") +
  labs(x="Featurename", y="Correlation Value", caption = "source data: dt_corr")

# Also look at the inter-correlation of features and plot the results
correlationMatrix <- rcorr(as.matrix(dt_selectedFeatures), type= c("pearson"))
corrplot(correlationMatrix$r, type="upper", order="hclust",
         p.mat = correlationMatrix$P, sig.level = 0.01, insig = "blank", # Insignificant correlations are left blank
         tl.cex = 0.7, tl.col = "black")

rm(func_NA2mean, dt_corr, correlationMatrix)
gc()
Sys.sleep(5)

##########################################
######## Manual Feature Selection ########
# Desciptive Statistics with Caret package

# For the following we preprocess the data a little more:
# - We do some tranformation of the data so that the values get centered (substract mean) & scaled (devided by standard deviation)
# - Finally removing features with nearZeroVariance, since we dont need features that provide (nearly) no prediction potential
#      and removing features that have high inter-correlation to other features, since taking multiple features which are highly corrrelated also dont provide any further prediction power.
dt_selectedFeatures_preprocessed <- dt_selectedFeatures # make copy of dataset to work with
preProcess_transformValues_model <- preProcess(dt_selectedFeatures_preprocessed, method = c("center", "scale"), na.remove = T)      # create tranformation model for centering & scaling
dt_selectedFeatures_preprocessed <- predict(preProcess_transformValues_model, newdata = dt_selectedFeatures_preprocessed)  # center values
# Remove features with nearZeroVariance
nzv <- nearZeroVar(dt_selectedFeatures_preprocessed)
dt_selectedFeatures_preprocessed_cleaned <- dt_selectedFeatures_preprocessed[, -..nzv]
# Remove features with high inter-correlation
corr_mat <- cor(dt_selectedFeatures_preprocessed_cleaned)
too_high <- findCorrelation(corr_mat, cutoff = .9)
dt_selectedFeatures_preprocessed_cleaned <- dt_selectedFeatures_preprocessed_cleaned[, -..too_high]
skim(dt_selectedFeatures_preprocessed_cleaned)[, c(1:6, 12)] # check remaining features

# Plotting the remaining features. !!! Creating these plots takes some time !!!
# Those with distinctive curves are the first to take into further investigation.
# Though this step should not be about ruling features out because their curves are not interesting!
# Feature importance analysis by mean
featurePlot_mean <- featurePlot(x = dt_selectedFeatures_preprocessed_cleaned,
                                y = as.factor(data_train$Response),
                                plot = "box",
                                strip=strip.custom(par.strip.text=list(cex=.7)),
                                scales = list(x = list(relation="free"),
                                              y = list(relation="free")),
                                pch='.')
# Feature importance analysis by density
featurePlot_density <- featurePlot(x = dt_selectedFeatures_preprocessed_cleaned,
                                   y = as.factor(data_train$Response),
                                   plot = "density",
                                   strip=strip.custom(par.strip.text=list(cex=.7)),
                                   scales = list(x = list(relation="free"),
                                                 y = list(relation="free")),
                                   pch='.')
rm(dt_selectedFeatures, dt_selectedFeatures_preprocessed_cleaned, too_high, corr_mat, nzv, dt_selectedFeatures_preprocessed, preProcess_transformValues_model)
gc()
Sys.sleep(5)

###################################
### Automatic Feature Selection ###
# Due to the big amount of data and limited computation power we cannot train a model on the complete training dataset.
# But it would be very time consuming and error-prone to go through all 900+ features and select them by hand, since we have 900+ features.
# Luckily, we can also use ML to make this selection for us and give us a list of features to use.
# Therefore we train models on a smaller portion of the training dataset (of all features). This subset should be kind of representative of the complete dataset.
# Then we just check which features the model uses.

# We use a subset of 200.000 rows (observations) from the data_train set to train the feature-selection models
X_train_selection <- data_train[1:200000,-c("Id", "Response")]
X_train_selection[is.na(X_train_selection)] <- 0
# For classification, Response must be provided as factors
Y_train_selection <- as.factor(data_train$Response[1:200000])
levels(Y_train_selection)[levels(Y_train_selection)==0] <- "good"
levels(Y_train_selection)[levels(Y_train_selection)==1] <- "fail"

#models <- c("glm", "svmLinear", "naive_bayes", "gamLoess", "multinom", "rf", "adaboost", "xgbTree")
models <- c("xgbTree")

fitControl <- trainControl(
  allowParallel = F,            # make use of multiple cpu cores
  verboseIter = F)              # give verbose info

# set up hyper-parameter search, in this case: reduce all parameter to one value, so no hyper-parameter search is done -> takes too long (but works!)
xgb_grid = expand.grid(
  nrounds = 20,
  eta = c(0.01),               # default 0.3
  max_depth = c(7),            # default 6
  gamma = 0,                   # default 0
  subsample = c(0.5), 
  colsample_bytree = c(0.5),
  min_child_weight = seq(1)
)

tic("training time for feature-selection model")    # set timer for training process of feature-selection model
featureSelection_models <- lapply(models, function(model){
  print(model)
  ## prepare parallel computing
  #cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
  #registerDoParallel(cluster)
  model <- train(
    x = X_train_selection,
    y = Y_train_selection,
    method = model,
    trControl = fitControl,
    tuneGrid = xgb_grid
  )
  ## end parallel computing
  #stopCluster(cluster)
  #rm(cluster)
  #gc()
  return (model)
})
toc() # end timer for training process of feature-selection model
names(featureSelection_models) <- models

# Getting features from feature-selection model
imp <- xgb.importance(model = featureSelection_models$xgbTree$finalModel, feature_names = colnames(X_train_selection))
head(imp, 10)
featureList <- imp$Feature

rm(corr_mat, correlationMatrix, dt_corr, dt_selectedFeatures, dt_selectedFeatures_preprocessed, dt_selectedFeatures_preprocessed_cleaned, imp, missingCount, X_train_selection, Y_train_selection, func_NA2mean)
gc()
Sys.sleep(5)
##########################################
### Training Model to predict Response ###

#################################
# Training final classification prediction model
# Again, as in the feature selection step, the XGB algorithm is used
# Prepare training data
X_train_final <- data_train %>% select(one_of(featureList))
Y_train_final_numeric <- data_train$Response
rm(data_train)
gc()
Sys.sleep(5)

# Preprocess data
X_train_final[is.na(X_train_final)] <- 0
Y_train_final_factor <- as.factor(Y_train_final_numeric)
levels(Y_train_final_factor)[levels(Y_train_final_factor)==0] <- "good"
levels(Y_train_final_factor)[levels(Y_train_final_factor)==1] <- "fail"

tic("training time for  final classification model")    # set timer for training process of  final classification model
prediction_model_classification <- train(
  x = X_train_final,
  y = Y_train_final_factor,
  method = "xgbTree",
  trControl = fitControl,
  tuneGrid = xgb_grid
)
toc() # end timer for training process of  final classification model

#################################
# Training final regression prediction model
# Training model for regression: This time the model will predict numeric values. (~ range 0,1)
# We will determine a cutoff and treat all values above that cutoff as 1=fail, all other values as 0=good
tic("training time for final regression model")    # set timer for training process of final regression model
prediction_model_regression <- train(
  x = X_train_final,
  y = Y_train_final_numeric,
  method = "xgbTree",
  trControl = fitControl,
  tuneGrid = xgb_grid
)
toc() # end timer for training process of  final regression model

##################################################
# Assessment of the performance of all build models via the MCC

### Target Function
# Define target function, to assess algorithm performance: Matthews correlation coefficient (MCC)
# (see https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
# The coefficient takes into account true and false positives and negatives and return a value between -1 and 1.
# mcc base function
mcc <- function(TP, FP, FN, TN)
{
  num <- (TP*TN) - (FP*FN)
  # using brob neccessary to avoid error "NAs produced by integer overflow"
  den <- as.brob(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
  
  if (den == 0)
  {
    return(as.numeric(0)) # return 0 per defintion if devided by 0
  }else
  {
    return(as.numeric(num / sqrt(den)))
  }
}
# mcc wrapper function to call with outcomes
calc_mcc <- function(truth, prediction){
  mcc_factors_table <- table(Truth = truth, Prediction = prediction)
  # catch error if factor-level was never predicted/appeard and return 0
  tp <- tryCatch({mcc_factors_table[1,1]}, error = function(e){0})
  fp <- tryCatch({mcc_factors_table[2,1]}, error = function(e){0})
  fn <- tryCatch({mcc_factors_table[1,2]}, error = function(e){0})
  tn <- tryCatch({mcc_factors_table[2,2]}, error = function(e){0})
  mcc(tp, fp, fn, tn)
}

#########################################
# Load validation data from csv file into dataframe
data_val <- fread(unzip(dl, "process_data_val.csv"), header = TRUE, showProgress = F)
# Preparing validation data
X_val <- data_val[,-c("Id", "Response")]
Y_val_numeric <- data_val$Response
rm(data_val)
gc()
Sys.sleep(5)

## Guessing MCC
# What is the MCC by guessing the Response? - repeating the process 100 times and taking the mean as result
set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
guessing_results <- function(){
  y_hat <- sample(c(0, 1), nrow(data_val), replace = TRUE)
  y_hat
  calc_mcc(data_val$Response, y_hat)
}
mcc_guessing <- mean(replicate(n = 100, expr = guessing_results()))
names(mcc_guessing) <- "MCC Guessing"
mcc_guessing
rm(guessing_results)

# For comparison, also check MCC of intermediate model for featureSelection

# Preprocess validation data
X_val[is.na(X_val)] <- 0
Y_val_factor <- as.factor(Y_val_numeric)
levels(Y_val_factor)[levels(Y_val_factor)==0] <- "good"
levels(Y_val_factor)[levels(Y_val_factor)==1] <- "fail"

mcc_featureSelection_model <- sapply(featureSelection_models, function(model){
  pred <- predict(model, newdata = X_val)
  return(calc_mcc(truth = Y_val, prediction = pred))
})
names(mcc_featureSelection_model) <- "MCC featureSelection classification"
# MCC with featureSelection Model
mcc_featureSelection_model

rm(featureSelection_model, imp)
gc()
Sys.sleep(5)

## final classificaton model
X_val <- X_val %>% select(one_of(featureList))       # Prepare validation data
pred_final_classificaton <- predict(prediction_model_classification, newdata = X_val)      # predict outcome
table(Truth = Y_val, Prediction = pred_final_classificaton)                                      # print table
mcc_predictionModel_classification <- calc_mcc(truth = Y_val, prediction = pred_final_classificaton)    # calculate MCC
names(mcc_predictionModel_classification) <- "MCC finalModel classification"
# MCC for final model trained for classification
mcc_predictionModel_classification

rm(prediction_model_classification, pred_final_classificaton)
gc()
Sys.sleep(5)

## finalregression model
pred_final_regression <- predict(prediction_model_regression, newdata = X_val)

# determine cutoff
matt <- data.table(quant = seq(0.7, 1, by = 0.1))
matt$mcc <- sapply(matt$quant, FUN =
                     function(x) {
                       calc_mcc(Y_val_numeric, (pred_final_regression > quantile(pred_final_regression, x)) * 1)})
print(matt)
best <- matt[which(matt$mcc == max(matt$mcc))][1]
best$cutoff <- quantile(pred_final_regression, best$quant)
best
# MCC for final model trained for classification
mcc_predictionModel_regression <- best$scores
names(mcc_predictionModel_regression) <- "MCC finalModel regression"

## Overall results summary
results <- list(mcc_guessing, mcc_featureSelection_model, mcc_predictionModel_classification, mcc_predictionModel_regression)
results
# knitr::kable(results, caption = "Summary of all predictions (all models used XGB algorithm") 
