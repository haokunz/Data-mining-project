### Workflow
### 1. Define the question:  How much should we pay for Audit fee?
### 2. Data wrangling(Completed)
### 3. Data Visualization(EDA - Exploratory Data Analysis) 
### 4. Unsupervised learning: Use PCAmix and K-means++ clustering to do the company segmentation? Like 繼鵬他們做的)
### 5. Choose the model(Supervised learning): 
###     Linear regression, knn, Regression tree: random forest, CART, Boosting & Lasso regression
### 學繼鵬：用lasso 和Random Forest兩個model，找出會影響audit fee的因素有哪些
### 6. Model training: Split to Training set & Testing set,  do the K-fold Cross-Validation
### 7. Assess the model performance: k-fold cv, MSE, R-square, F-1 score(the chart of actual and  predicted)
### ( Choose the best performance model，use it to do the prediction，並且將結果圖表化？ Partial dependence plots etc)
### 8. Conclusion: What did we learn- 根據不同的input value, 可以預測audit fee為多少


### Workflow if learn from other ppl
### 1~4 same
### 5. Use lasso regression and do th cv, find the important coefficients(more robust and understandable coefficients)
### 6. Use Random Forest to measure the variable importance(better fit but less interpretability), and use partial plots 
### End




## Problem zone
### 1. How to assess the model performance? BC Lasso model 好像不能用MSE去評價？ If not, what can we use to assess the model?
### 2. Model是要用來找出variable的重要性的嗎？ 繼鵬好像是用lasso 找出重要的 independent variable
### 繼鵬他們沒有用模型進行預測，只是針對數據分析結果做出解釋，提出哪些因素對acquire more kisses on dating app有影響
### 如果要學繼鵬他們，就是提出哪些因素對acquire more kisses有影響





# Step 2. Data wrangling & load the needed packages
## Loaded the needed packages
library(tidyverse)
library(ggplot2)
library(rsample)
library(modelr)
library(randomForest)
library(caret)
library(gbm)
library(ggmap)
library(glmnet)
library(rpart) # a powerful ML library that is used for building classification and regression trees
library(gamlr)
library(rpart.plot)
library(data.table)
library(DMwR2)
library(car)
library(factoextra)
library(rfUtilities)
library(LICORS)
library(PCAmixdata)

## read the dataset
df <- read.csv("https://raw.githubusercontent.com/haokunz/Data_mining_project/main/data/internal_controls_data_1680556746.csv",
               header = TRUE)
## delete copyright and lines of notes
df <- df[-c(nrow(df), nrow(df)-1), ]

## remove records with restated internal control report
duplicated_indexes <- which(df$Restated.Internal.Control.Report == "Yes (1)")
duplicated_companies <- unique(df$Company[duplicated_indexes])
restate_indexes <- which(df$Company %in% duplicated_companies)
remove_index <- setdiff(restate_indexes, duplicated_indexes)
df1 <- df[-remove_index, ]

## remove duplicated records from different auditors working at the same time
multi_auditors <- table(df1$Company)[table(df1$Company) >= 2]
remove_index_2 <- setdiff(which(df1$Company %in% names(multi_auditors)), match(names(multi_auditors), df1$Company))
df2 <- df1[-remove_index_2, ]

## remove rows with missing revenue data
df2 <- df2[df2$Revenue.... != "", ]

## select target columns
df3 <- df2[ ,c("Company", "City", "State.Code", "State.Name", "State.Region", 
               "Auditor", "Auditor.Key", "Auditor.State.Name", 
               "Effective.Internal.Controls", "Audit.Fees....", "Non.Audit.Fees....",
               "Total.Fees....", "Share.Price", "Market.Cap....", "Revenue....",
               "Earnings....", "Book.Value....", "Assets....")]

## change column names to mark the targets
colnames(df3) <- c("company", "city", "state_code", "state_name", "state_region",
                   "auditor", "auditor_key", "auditor_state_name", 
                   "effective_internal_controls", "audit_fees", "non_audit_fees",
                   "total_fees", "share_price", "market_cap","revenue",
                   "earnings", "book_value", "assets")

## convert money amount character into numeric
df3$audit_fees = as.numeric(gsub(",", "", df3$audit_fees))
df3$non_audit_fees = as.numeric(gsub(",", "", df3$non_audit_fees))
df3$total_fees = as.numeric(gsub(",", "", df3$total_fees))
df3$market_cap = as.numeric(gsub(",", "", df3$market_cap))
df3$revenue = as.numeric(gsub(",", "", df3$revenue))
df3$earnings = as.numeric(gsub(",", "", df3$earnings))
df3$book_value = as.numeric(gsub(",", "", df3$book_value))
df3$assets = as.numeric(gsub(",", "", df3$assets))

## add indicator for analysis
df3$big_four_indicator <- ifelse(df3$auditor_key <= 4, 1, 0)
df3$five_category <- ifelse(df3$auditor_key < 5, df3$auditor_key, 5)
df3$audit_percent <- df3$audit_fees / df3$total_fees

## add transformation variables to the data
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

## preliminary test on big_four_indicator,change coloumns to factor
# add the big 4 indicator, five category and state region into the dataframe
df3$big_4_factor <- as.factor(df3$big_four_indicator)
df3$five_category_factor <- as.factor(df3$five_category)
df3$state_region <- as.factor(df3$state_region)


# Step 3. Data visualization(EDA - Exploratory Data Analysis) 
### Add the explanation for the chart
# basic plots, preliminary exploration #


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



# Step 4. Unsupervised learning: Use PCAmix and K-means++ clustering to do the company segmentation
df3$effective_internal_controls_factor = as.factor(ifelse(df3$effective_internal_controls == "No", 0, 1))

X.quanti <- splitmix(df3)$X.quanti %>% scale()
X.quali <- splitmix(df3)$X.quali
df3_pca <- PCAmix(X.quanti, X.quali, ndim=4, rename.level = TRUE, graph=FALSE)

df3_pca_scores = df3_pca$ind$coord %>% as.data.frame()

# append 4 pc to df3
df3$PC1 <- df3_pca_scores$`dim 1`
df3$PC2 <- df3_pca_scores$`dim 2`
df3$PC3 <- df3_pca_scores$`dim 3`
df3$PC4 <- df3_pca_scores$`dim 4`


# KClustering
## Choose optimal K - CH index
k_grid = seq(2, 6, by=1)
set.seed(8964)
df_CH_grid = foreach(k=k_grid, .combine='rbind') %do% {
  cluster_k = kmeanspp(df3_pca_scores, k, nstart = 50)
  W = cluster_k$tot.withinss
  B = cluster_k$betweenss
  CH = (B/W)*((nrow(df3_pca_scores)-k)/(k-1))
  c(k=k, stat = CH)
} %>% as.data.frame()

df_kmpp = kmeanspp(df3_pca_scores, k=3, nstart=25)
df3$cluster = df_kmpp$cluster 

## Analysis
### Find out significant differences among groups
### Ans: although CH index suggests that optimal k = 4, but we can only
### find meaningful clustering with k = 3 instead of 4
# group 1+4 vs. 3
clus1 = ggplot(df3) +
  geom_point(aes(x=counts_profileVisits, y=conversion, color=factor(cluster))) +
  labs(color='Cluster')
# group 2: low conversion, high count details
clus2 = ggplot(lovoo_data) +
  geom_point(aes(x=countDetails, y=conversion, color=factor(cluster))) +
  labs(color='Cluster')
ggarrange(clus1, clus2, common.legend = TRUE,
          legend = "bottom")
# Step 5. Modeling: Linear regression, knn, Regression tree: random forest, CART, Boosting

### Base model: Linear regression(Bid model)
## Syntax: lm3 = lm(price ~ (. - pctCollege - sewer - waterfront - landValue - newConstruction)^2, data=saratoga_train)


### Model 2: KNN
## Syntax: KNN with K = 70
## knn100 = knnreg(COAST ~ KHOU, data=loadhou_train, k=100)
### modelr::rmse(knn100, loadhou_test)
## predict(knn100, loadhou_test)


### Model 3 to 5 are belong to regression trees
### Syntax in Exercise3 Q2
### Model 3: Random Forest model
## Syntax: rforest_dengue = randomForest(total_cases ~ .,data = dengue_training, importance=TRUE)
## Performance: Use rmse. Syntax: rmse(gbm_dengue, dengue_testing)

### Model 4: CART model 
## Syntax: cart_dengue = rpart(total_cases ~ . , data = dengue_training, control = rpart.control(cp = 0.002, minsplit=20))
## Split only if we have at least 20 obs in a node,
## and the split improves the fit by a factor of 0.002 aka 0.2%
## Performance: Use rmse. Syntax: rmse(gbm_dengue, dengue_testing)

### Model 5: Gradient-boosted model
## in the "capmetro.R"
## Syntax: gbm_dengue = gbm(total_cases ~ ., data = dengue_training, interaction.depth=4, n.trees=500, shrinkage=.05)
## Performance: Use rmse. Syntax: rmse(gbm_dengue, dengue_testing)


### Model 6: Lasso regression
### Use lasso to find the important variables?
## 好像不能用MSE out of sample去評價，要用AIC？
## Syntax in homework exercise 2 
## Syntax for Lasso: 
### lasso_selected = glm(children ~ (.-arrival_date-deposit_type) + hotel:reserved_room_type+ meal:is_repeated_guest+ adults:previous_bookings_not_canceled+ meal:previous_bookings_not_canceled+ market_segment:customer_type+is_repeated_guest:assigned_room_type+ assigned_room_type:required_car_parking_spaces, data = hotels_dev_train, family = "binomial")








# Step 6. 訓練模型: Split to Training set & Testing set, 做K-fold Cross-Validation
## Split to Training set & Testing set
### Syntax:
### hotels_dev_split = initial_split(hotels_dev, prop = 0.7)
### hotels_dev_train = training(hotels_dev_split)
### hotels_dev_test = testing(hotels_dev_split)


### Step 7. 評價模型: K-fold Cross-validation
## K-fold Cross-validation: (k-1) is training set, 1 is testing set
## 從沒當過testing set的 dataset中挑一個來做testing set, 剛剛做過testing set 的那份則加回去做training set
## repeat the step until every set 都當過testing set ## 會執行k次, 得到k個 validation error
## Average the k validation error, then we can know which model is better

### Syntax:
#### set.seed(123)
#### k <- 5
#### cart_cv <- rpart.control(cp = 0.01)
#### rf_cv <- list(mtry = sqrt(ncol(dengue_training)), replace = TRUE)
#### gb_cv <- list(n.trees = 1000, interaction.depth = 4, shrinkage = 0.01, cv.folds = k)
#### cart_cv_results <- rpart(total_cases ~ ., data = dengue_training, control = cart_cv)
#### rf_cv_results <- randomForest(total_cases ~ ., data = dengue_training, mtry = rf_cv$mtry, replace = rf_cv$replace)
#### gb_cv_results <- gbm(total_cases ~ ., data = dengue_training, n.trees = gb_cv$n.trees, interaction.depth = gb_cv$interaction.depth, shrinkage = gb_cv$shrinkage, cv.folds = gb_cv$cv.folds, verbose = FALSE)
#### Evaluate the performance of each model
#### cart_performance <- predict(cart_cv_results, newdata = dengue_testing)
#### rf_performance <- predict(rf_cv_results, newdata = dengue_testing)
#### gb_performance <- predict(gb_cv_results, newdata = dengue_testing, n.trees = gb_cv$n.trees)
#### Compare the performance of each model by measuring MSE
#### cart_accuracy <- mean((cart_performance - dengue_testing$total_cases)^2)
#### rf_accuracy <- mean((rf_performance - dengue_testing$total_cases)^2)
#### gb_accuracy <- mean((gb_performance - dengue_testing$total_cases)^2)
####  print out the MSE of each model to console, the lower the better the model is 
#### cat("CART accuracy:", cart_accuracy, "\n")
#### cat("Random Forest accuracy:", rf_accuracy, "\n")
#### cat("Gradient Boosting accuracy:", gb_accuracy, "\n")
### Syntax for k-fold cv ends









