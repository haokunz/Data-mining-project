# Data mining project

###
# Want to determine which type of restatement has what kind of impact on audit fee 
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
library(caret)
library(car)
library(factoextra)
library(rfUtilities)

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


################################################################################
# Import data for doing the analysis on the internal control database
################################################################################

df <- read.csv("internal-controls-data-1680556746.csv",
               header = TRUE)

# delete copyright and lines of notes
df <- df[-c(nrow(df), nrow(df)-1), ]

# remove records with restated internal control report
duplicated_indexes <- which(df$Restated.Internal.Control.Report == "Yes (1)")
duplicated_companies <- unique(df$Company[duplicated_indexes])
restate_indexes <- which(df$Company %in% duplicated_companies)
remove_index <- setdiff(restate_indexes, duplicated_indexes)
df1 <- df[-remove_index, ]

# remove duplicated records from different auditors working at the same time
multi_auditors <- table(df1$Company)[table(df1$Company) >= 2]
remove_index_2 <- setdiff(which(df1$Company %in% names(multi_auditors)), match(names(multi_auditors), df1$Company))
df2 <- df1[-remove_index_2, ]

# remove rows with missing revenue data
df2 <- df2[df2$Revenue.... != "", ]

# select target columns
df3 <- df2[ ,c("Company", "City", "State.Code", "State.Name", "State.Region", 
               "Auditor", "Auditor.Key", "Auditor.State.Name", 
               "Effective.Internal.Controls", "Audit.Fees....", "Non.Audit.Fees....",
               "Total.Fees....", "Share.Price", "Market.Cap....", "Revenue....",
               "Earnings....", "Book.Value....", "Assets....")]

# change column names to mark the targets
colnames(df3) <- c("company", "city", "state_code", "state_name", "state_region",
                   "auditor", "auditor_key", "auditor_state_name", 
                   "effective_internal_controls", "audit_fees", "non_audit_fees",
                   "total_fees", "share_price", "market_cap","revenue",
                   "earnings", "book_value", "assets")

# convert money amount character into numeric
df3$audit_fees = as.numeric(gsub(",", "", df3$audit_fees))
df3$non_audit_fees = as.numeric(gsub(",", "", df3$non_audit_fees))
df3$total_fees = as.numeric(gsub(",", "", df3$total_fees))
df3$market_cap = as.numeric(gsub(",", "", df3$market_cap))
df3$revenue = as.numeric(gsub(",", "", df3$revenue))
df3$earnings = as.numeric(gsub(",", "", df3$earnings))
df3$book_value = as.numeric(gsub(",", "", df3$book_value))
df3$assets = as.numeric(gsub(",", "", df3$assets))

# add indicator for analysis
df3$big_four_indicator <- ifelse(df3$auditor_key <= 4, 1, 0)
df3$five_category <- ifelse(df3$auditor_key < 5, df3$auditor_key, 5)
df3$audit_percent <- df3$audit_fees / df3$total_fees

# add transformation variables to the data
df3$audit_fees_bc <- predict(BoxCoxTrans(df3$audit_fees), df3$audit_fees)
non_audit_bc <- predict(BoxCoxTrans(df3$non_audit_fees[df3$non_audit_fees!=0]),
                        df3$non_audit_fees[df3$non_audit_fees!=0])
df3$total_fees_bc <- predict(BoxCoxTrans(df3$total_fees), df3$total_fees)
df3$market_cap_bc <- predict(BoxCoxTrans(df3$market_cap), df3$market_cap)
df3$market_fee_ratio <- log(df3$market_cap/ df3$total_fees)
df3$assets_log <- log(df3$assets)

revenue_0 = jitter(df3$revenue)
df3$revenue_trans <- (revenue_0/abs(revenue_0)) * log(abs(df3$revenue) + 1)

earnings_0 = jitter(df3$earnings)
df3$earnings_trans <- (earnings_0/abs(earnings_0)) * log(abs(df3$earnings) + 1)

# preliminary test on big_four_indicator,chang coloums to factor
df3$big_4_factor <- as.factor(df3$big_four_indicator)
df3$five_category_factor <- as.factor(df3$five_category)
df3$state_region <- as.factor(df3$state_region)


################################################################################
# Import data for restatement analysis
################################################################################

# read data
restate <- read.csv('restatement_data_1681675957.csv', header=TRUE, sep=",", row.names = NULL)

# shift all columns one step to the left
names(restate)[1:(ncol(restate)-1)] <- names(restate)[2:ncol(restate)]
restate[, ncol(restate)] <- NULL

# rename Company to company
colnames(restate)[1] ="company"

# remove rows with empty value on net income or stakeholder equity and add small
# number to rows that are 0 for future transformation
restate2 <- restate %>% drop_na(Cumulative.Change.in.Net.Income)
restate2$Cumulative.Change.in.Net.Income = ifelse(restate2$Cumulative.Change.in.Net.Income == 0, 1, 
                                                  restate2$Cumulative.Change.in.Net.Income) # replace 0 with 1
restate2 <- restate2 %>% drop_na(Cumulative.Change.in.Stockholder.Equity)
restate2$Cumulative.Change.in.Stockholder.Equity =ifelse(restate2$Cumulative.Change.in.Stockholder.Equity == 0,1,
                                                         restate2$Cumulative.Change.in.Stockholder.Equity) # replace 0 with 1
restate2 <- restate2[-which(restate2$Auditor...Opinion.Period.End.During.Restated.Period == ""),]

# remove columns unrelated to research objects
restate2 = subset(restate2, select = -c(X, X.1, # removed unrelated empty columns 
                                        3:20, # removed company information such as physical address or phone number
                                        Restatement.Key, # removed unique restatement ID for each case
                                        Securities.Class.Action.Litigation.Legal.Case.Key)) # removed security class action case key 
# clean text data, remove any symbols
restate2$Auditor...During.Restated.Period <- lapply(restate2$Auditor...During.Restated.Period, 
                                                    clean_string1)
restate2$Auditor...During.Restated.Period.Keys <- lapply(restate2$Auditor...During.Restated.Period.Keys, 
                                                         clean_string1)
restate2$Auditor...Opinion.Period.End.During.Restated.Period <- lapply(restate2$Auditor...Opinion.Period.End.During.Restated.Period, 
                                                                       clean_string1)
restate2$Auditor...Opinion.Period.End.During.Restated.Period.Keys <- lapply(restate2$Auditor...Opinion.Period.End.During.Restated.Period.Keys, 
                                                                            clean_string1)

# define a dummy variable that is 1 if current auditor is one of Big 4 or 0 if not Big 4
big4_firms <- c("Ernst & Young", "Deloitte", "KPMG", "PricewaterhouseCoopers")

###
# determine whether auditor is big 4
###
restate2$Current_big4 <- grepl(paste(big4_firms, collapse = "|"), 
                               restate2$Auditor...Current, 
                               ignore.case = TRUE) # determine auditor...current
restate2$Disclosure_big4 <- grepl(paste(big4_firms, collapse = "|"), 
                                  restate2$Auditor...At.Disclosure.Date, 
                                  ignore.case = TRUE) # determine auditor...at disclosure date
restate2$resated_big4 <- grepl(paste(big4_firms, collapse = "|"), 
                               restate2$Auditor...During.Restated.Period, 
                               ignore.case = TRUE) # determine auditor during restated period
restate2$opinion_big4 <- grepl(paste(big4_firms, collapse = "|"), 
                               restate2$Auditor...Opinion.Period.End.During.Restated.Period, 
                               ignore.case = TRUE) # determine auditor during opinion period

# Convert logical values to 1s and 0s
restate2$Current_big4 <- as.integer(restate2$Current_big4)
restate2$Disclosure_big4 <- as.integer(restate2$Disclosure_big4)
restate2$resated_big4 <- as.integer(restate2$resated_big4)
restate2$opinion_big4 <- as.integer(restate2$opinion_big4)

# Convert date from character type to date type
restate2$MR...Stock.Price.Date <- as.Date(restate2$MR...Stock.Price.Date, 
                                          format = "%Y-%m-%d")
restate2$MR...Financials.Date <- as.Date(restate2$MR...Financials.Date, 
                                         format = "%Y-%m-%d")
restate2$H...Financials.Date <- as.Date(restate2$H...Financials.Date, 
                                        format = "%Y-%m-%d")
restate2$H...Stock.Price.Date <- as.Date(restate2$H...Stock.Price.Date, 
                                         format ="%Y-%m-%d")

###
# calculate duration of restate period
###
restate2$Restated.Period.Begin <- as.Date(restate2$Restated.Period.Begin, 
                                          format = "%Y-%m-%d") # convert restate begin date to the correct format
restate2$Restated.Period.Ended <- as.Date(restate2$Restated.Period.Ended, 
                                          format = "%Y-%m-%d") # convert restate end date to the correct format
# measure the duration
restate2$duration_in_days <- as.integer(difftime(restate2$Restated.Period.Ended, 
                                                 restate2$Restated.Period.Begin, 
                                                 units = "days"))

# switch SEC investigation, Board involvement and Auditor Letter...Discussion to categorical variable
restate2$Board.Involvement_ID <- unclass(as.factor(restate2$Board.Involvement))
                                 # 3 for Y, 2 for ND, 1 for N
restate2$Auditor.Letter_ID <- unclass(as.factor(restate2$Auditor.Letter...Discussion))
                                # 3 for Y, 2 for ND, 1 for N
restate2$SEC.Investigation_ID <- unclass(as.factor(restate2$SEC.Investigation))
                                # 1 for Y and 2 for ""

# change type of these three variables to categorical from character
restate2$SEC.Investigation <- as.factor(restate2$SEC.Investigation)
restate2$Auditor.Letter_ID <- as.factor(restate2$Auditor.Letter_ID) # change auditor letter to 3 level categorical variables
restate2$Board.Involvement_ID <- as.factor(restate2$Board.Involvement_ID) 

# Use clean_string2 to remove "," in col 29, 31-34, 37, 39-42 and convert them to numeric
restate2[c(28, 30, 32:35, 38, 40:43)]<- sapply(restate2[c(28, 30, 32:35, 38, 40:43)], 
                                           clean_string2)

# assign Key number to GAAP Failure types
restate2$GAAP_Failure_ID <- with(restate2, match(Accounting.Rule..GAAP.FASB..Application.Failures, 
                                                 unique(Accounting.Rule..GAAP.FASB..Application.Failures)))

restate2 <- na.omit(restate2)

# Combine df3 and restate2 for audit fee analysis
total <- merge(df3, restate2, by="company")

################################################################################
# Build model to predict restatement's effect on audit fee
################################################################################

# split data to training and testing dataset
total_split = initial_split(total, prop=0.8)
total_train = training(total_split)
total_test = testing(total_split)

# use randomforest to 

########################################
# basic plots, preliminary exploration #
########################################

# plot the number distribution of companies in different regions
company_numbers <- sort(table(df3$state_region[df3$state_region != ""]), decreasing = FALSE, na.last = NA)

par(mar = c(5.1, 6.5, 4.1, 2.1))
barplot(height=company_numbers,
        names.arg=c("Canada", "US_NewEng", "US_Southwest", "US_Southeast",
                    "US_Midwest", "Foreign", "US_MAtlan", "US_West"),
        col="#69b3a2", horiz=TRUE, las = 1, main = "Num. of Companies", xlab = "numbers")
par(mar = c(5.1, 4.1, 4.1, 2.1))

# use eight plots to display the effect of transformation on fee related variables
par(mfrow = c(2, 4))
hist(df3$audit_fees, breaks="Scott", main="audit fees", xlab="Audit fees")
hist(df3$audit_fees_bc, main="audit fees (transformed)", xlab="Audit fees")
hist(df3$non_audit_fees, breaks="Scott", main="non audit fees", xlab="Non-audit fees")
hist(non_audit_bc, main="non audit fees (transformed)", xlab="Non-audit fees")
hist(df3$total_fees, breaks="Scott", main="total fees", xlab="Total fees")
hist(df3$total_fees_bc, main="total fees (transformed)", xlab="Total fees")
hist(df3$market_cap, breaks="Scott", main="Market cap", xlab="Market cap")
hist(df3$market_cap_bc, main="Market cap (transformed)", xlab="Market cap")
par(mfrow = c(1, 1))

# use three plots to display the categorical data
par(mfrow = c(1, 3))
barplot(table(df3$five_category_factor), ylab = "Frequency", main="Auditing company distribution")
barplot(table(df3$big_4_factor), yaxt='n', ylab="Frequency", main="Num. big4 vs. other")
axis(side=2, at=seq(0, nrow(df3), 200))
barplot(table(df3$effective_internal_controls), yaxt='n', ylab="Frequency", main="Num. effective internal controls")
axis(side=2, at=seq(0, nrow(df3), 200))
par(mfrow = c(1, 1))

# plot the transformed company market cap, total auditing fees, and effective internal control
sp = ggplot(df3, aes(x=market_cap_bc, y=five_category_factor,
                     group=effective_internal_controls)) +
  geom_point(aes(color=effective_internal_controls), size=0.9,
             position=position_dodge2(0.3))

labels = as.vector(outer(rep("Num. of 'No'="), table(df3$effective_internal_controls,
                                                     df3$five_category_factor)[1,],
                         paste, sep=""))
sp + annotate(geom="text", x=rep(27.5, 5), y=seq(0.7, 4.7, 1), label= labels) 

# plot the transformed company market cap vs. total auditing fees
ggplot(df3, aes(x=market_cap_bc, y=total_fees_bc, group=five_category_factor)) +
  geom_point(aes(color=five_category_factor), size=0.9)

cor(df3$market_cap_bc, df3$total_fees_bc)

# plot the auditing fees
ggplot(df3, aes(x=five_category_factor, y=total_fees_bc)) + 
  geom_violin(trim=FALSE, fill="gray")+
  labs(title="Auditing fees",x="category", y = "total fees")+
  geom_boxplot(width=0.3)+
  theme_classic()
# Change color by groups
dp <- ggplot(df3, aes(x=five_category_factor, y=total_fees_bc, fill=five_category_factor)) + 
  geom_violin(trim=FALSE)+
  geom_boxplot(width=0.3, fill="white")+
  labs(title="Plot of auditing fees",x="category", y = "total fees")
dp + theme_classic()

# perform analysis of variance for 1-5 auditing company levels(146-153 for future thoughts)
#m5_ancova = lm(total_fees_bc ~ five_category_factor + market_cap_bc +
#five_category_factor*market_cap_bc, df3)
#Anova(m5_ancova, type=3)

#library(multcomp)
#postHocs <- glht(m5_ancova, linfct=mcp(five_category_factor="Tukey"))
#summary(postHocs)

# perform analysis to compare big 4
# between big4, considering the market cap as covariate, the fee charges from
# big4 do not have a significant difference __need to add nonBIg4 and BIg4 fee difference considering balance
m4_ancova = lm(total_fees_bc ~ five_category_factor + market_cap_bc +
                 five_category_factor*market_cap_bc, df3[df3$big_4_factor==1,])
Anova(m4_ancova, type=3)

# perform a statistical test here to compare big4 vs non big4 when considering 
# market cap as covariate 

#?whether hierarchical is better? how to produce a better K? how to deal with data to make cluster better?
set.seed(143)
pca_data = na.omit(df3[ ,c("audit_fees_bc", "total_fees_bc", "market_cap_bc",
                           "market_fee_ratio", "assets_log", "revenue_trans",
                           "earnings_trans")])
km.res <- kmeans(pca_data, 4, nstart=25)

aggregate(pca_data, by=list(cluster=km.res$cluster), mean)

dd <- cbind(pca_data, cluster=km.res$cluster)
head(dd)
fviz_cluster(km.res, data=pca_data)


# predict the total auditing fee based on other variables
set.seed(2501)
data_split = initial_split(df3, prop=0.8)
data_train = training(data_split)
data_test = testing(data_split)

rf1 <- randomForest(total_fees_bc ~ five_category_factor + state_region + 
                      market_cap_bc + assets_log + revenue_trans + earnings_trans,
                    data=data_train, importance=TRUE)

rf1
plot(rf1)
modelr::rmse(rf1, data_test)
varImpPlot(rf1, type=1)
rf_predict <- predict(rf1, data_test)

RSQUARE = function(y_actual,y_predict){
  cor(y_actual,y_predict)^2
}

R2 <- RSQUARE(data_test$audit_fees_bc, rf_predict)


# plot predicted vs. actual values
plot(x=rf_predict, y= data_test$audit_fees_bc, xlab="Predicted Values",
     ylab="Actural Values", main="Predicted vs. Actual Values")

# add diagonal line for estimated regression line
abline(a=0, b=1)
mylabel =  bquote(italic(R)^2 == .(format(R2, digits = 3)))
legend("topleft", legend=mylabel)


# question1: is there a difference between big 4 and others in term of auditing fees(?)
# question2: is there a significant difference among big 4
# question3: big4 and non big4, do they have similar auditing fee structure (?)
# question4: can we predict the approximate auditing fees based on other information(keyi)

# book_value has more than 600 missings, if use that column, may need imputation 