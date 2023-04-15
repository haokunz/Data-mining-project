# Data mining project

###
# Want to determine which type of restatement has what kind of impact on net income and stakeholder's equity 
###


# set working directory
setwd('C:/Users/Haokun Zhang/Desktop/github/Data-mining-project/data')

# import mods
library(tidyverse)
library(dplyr)
library(usmap)
library(lubridate)
library(randomForest)
library(splines)
library(pdp)
library(string)
library(ggplot2)
library(rsample)
library(modelr)
library(caret)
library(gbm)
library(ggmap)
library(glmnet)
library(rpart) # a powerful ML library that is used for building classification and regression trees
library(gamlr)
library(rpart.plot)
library(data.table)
library(DMwR2)
library(knitr)

# functions

clean_string1 <- function(string) {
  cleaned <- gsub("\\|", "", string)
  return(cleaned)
} # clean "|" in string

clean_string2 <- function(string) {
  cleaned <-gsub("\\,", "", string)
  number <- as.numeric(cleaned)
  return(number)
} # remove "," in strings and convert them to numeric

# read data
restate <- read.csv('restatement_data.csv', header=TRUE, sep=",", row.names = NULL)

head(restate)

# shift all columns one step to the left
names(restate)[1:(ncol(restate)-1)] <- names(restate)[2:ncol(restate)]
restate[, ncol(restate)] <- NULL

# remove rows with empty value on net income or stakeholder equity
restate2 <- restate %>% drop_na(Cumulative.Change.in.Net.Income)
restate2 <- restate2 %>% drop_na(Cumulative.Change.in.Stockholder.Equity)
restate2<- restate2[-which(restate2$Auditor...Opinion.Period.End.During.Restated.Period == ""),]

# remove columns unrelated to research objects
restate2 = subset(restate2, select = -c(X, X.1, # removed unrelated empty columns 
                                        2:20, # removed company information such as physical address or phone number
                                        Restatement.Key, # removed unique restatement ID for each case
                                        Securities.Class.Action.Litigation.Legal.Case.Key)) # removed security class action case key 
# clean text data, remove any symbols
restate2$Auditor...During.Restated.Period <- lapply(restate2$Auditor...During.Restated.Period, clean_string1)
restate2$Auditor...During.Restated.Period.Keys <- lapply(restate2$Auditor...During.Restated.Period.Keys, clean_string1)
restate2$Auditor...Opinion.Period.End.During.Restated.Period <- lapply(restate2$Auditor...Opinion.Period.End.During.Restated.Period, clean_string1)
restate2$Auditor...Opinion.Period.End.During.Restated.Period.Keys <- lapply(restate2$Auditor...Opinion.Period.End.During.Restated.Period.Keys, clean_string1)

# define a dummy variable that is 1 if current auditor is one of Big 4 or 0 if not Big 4
big4_firms <- c("Ernst & Young", "Deloitte", "KPMG", "PricewaterhouseCoopers")


###
# determine whether auditor is big 4
###
restate2$Current_big4 <- grepl(paste(big4_firms, collapse = "|"), restate2$Auditor...Current, 
                               ignore.case = TRUE) # determine auditor...current
restate2$Disclosure_big4 <- grepl(paste(big4_firms, collapse = "|"), restate2$Auditor...At.Disclosure.Date, 
                                  ignore.case = TRUE) # determine auditor...at disclosure date
restate2$resated_big4 <- grepl(paste(big4_firms, collapse = "|"), restate2$Auditor...During.Restated.Period, 
                               ignore.case = TRUE) # determine auditor during restated period
restate2$opinion_big4 <- grepl(paste(big4_firms, collapse = "|"), restate2$Auditor...Opinion.Period.End.During.Restated.Period, 
                               ignore.case = TRUE) # determine auditor during opinion period

# Convert logical values to 1s and 0s
restate2$Current_big4 <- as.integer(restate2$Current_big4)
restate2$Disclosure_big4 <- as.integer(restate2$Disclosure_big4)
restate2$resated_big4 <- as.integer(restate2$resated_big4)
restate2$opinion_big4 <- as.integer(restate2$opinion_big4)

# Convert date from character type to date type
restate2$MR...Stock.Price.Date <- as.Date(restate2$MR...Stock.Price.Date, format = "%Y-%m-%d")
restate2$MR...Financials.Date <- as.Date(restate2$MR...Financials.Date, format = "%Y-%m-%d")
restate2$H...Financials.Date <- as.Date(restate2$H...Financials.Date, format = "%Y-%m-%d")
restate2$H...Stock.Price.Date <- as.Date(restate2$H...Stock.Price.Date, format ="%Y-%m-%d")

###
# calculate duration of restate period
###
restate2$Restated.Period.Begin <- as.Date(restate2$Restated.Period.Begin, format = "%Y-%m-%d") # convert restate begin date to the correct format
restate2$Restated.Period.Ended <- as.Date(restate2$Restated.Period.Ended, format = "%Y-%m-%d") # convert restate end date to the correct format

# measure the duration
restate2$duration_in_days <- as.integer(difftime(restate2$Restated.Period.Ended, restate2$Restated.Period.Begin, units = "days"))

# switch SEC investigation, Board involvement and Auditor Letter...Discussion to categorical variable
restate2$Board.Involvement <- gsub('ND', 'N', restate2$Board.Involvement) # replace "ND" with "N" in board involvement
restate2$Auditor.Letter...Discussion <- gsub("ND", "N", restate2$Auditor.Letter...Discussion)

# change type of these three variables to categorical from character
restate2$SEC.Investigation <- as.factor(restate2$SEC.Investigation)
restate2$Auditor.Letter...Discussion <- as.factor(restate2$Auditor.Letter...Discussion) # change auditor letter to 3 level categorical variables
restate2$Board.Involvement <- as.factor(restate2$Board.Involvement) 

# Use clean_string2 to remove "," in col 29, 31-34, 37, 39-42 and convert them to numeric
restate2[c(29, 31:34, 37, 39:42)]<- sapply(restate2[c(29, 31:34, 37, 39:42)], clean_string2)

# assign Key number to GAAP Failure types
restate2$GAAP_Failure_ID <- with(restate2, match(Accounting.Rule..GAAP.FASB..Application.Failures, 
                                                 unique(Accounting.Rule..GAAP.FASB..Application.Failures)))

# remove rows with na value
restate3 <- na.omit(restate2)

