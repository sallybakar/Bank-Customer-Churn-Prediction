#Import Libraries 


{library(tidyr)
library(ggplot2)
library(purrr)
library(printr)
library(pROC) 
library(ROCR) 
library(caret)
library(car)
library(rpart)
library(rpart.plot)
}

# Loading Dataset

df <- read.csv("C:/Users/840 G3/Downloads/Bank Customer Churn Prediction.csv", stringsAsFactors = TRUE)
df

# Descriptive Statistic by taking a glimpse on our dataset, we have total of 10,000 and 12 columns. One non-useful variables is identified:  customer_id. Two categorical variables: country and gender need to be encoded into numbers because machine learning models can only work with numerical input
str(df)
View(df)


# detect missing value
knitr::kable(sapply(df, function(x) sum(is.na(x))), col.names = c("Missing Value Count"))


# show summary statistics of the useful variables
summary(df[, !names(stanbic) %in% c('customer_id')])

# plot box plot
df[, names(stanbic) %in% c( 'age', 'balance', 'credit_score', 'estimated_salary')] %>%
  gather() %>%
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_boxplot() +
  theme(axis.text.x = element_text(size = 7, angle=90), axis.text.y = element_text(size = 7))

 ### {Box plot is plotted to show the data distribution of continuous variables and check if there is any outlier.
#Outliers are detected in age and credit_score, but they are not erroneous outliers and this outlier situation occurs 
#because of the small sample number between these outliers range. age, credit_score, balance variables are skewed
#toward the majority values while estimated_salary seems to be normally distributed.
#Log transformation can be applied to these three variables to solve the skewed and outliers data issues.}



# data encoding
df$country = factor(df$country, labels=c(0, 1, 2))
df$gender = factor(df$gender, labels=c(0, 1))

# data transformation
df$age = log(df$age)
df$credit_score = log(df$credit_score)
df$balance = log(df$balance)
df[stanbic$balance == -Inf, 'balance'] <- 0


# scaling
fun_scale_0to1 <- function(x) {                           
  (x - min(x)) / (max(x) - min(x))
}

df$age = fun_scale_0to1(df$age)
df$credit_score = fun_scale_0to1(df$credit_score)
df$balance = fun_scale_0to1(df$balance)
df$estimated_salary= fun_scale_0to1(df$estimated_salary)


head(df, 5)

#Splitting Data into training set and testing set

set.seed(1000)
trainIndex <- createDataPartition(df$churn, p = 0.8, list = FALSE, times = 1)
training_df <- df[ trainIndex,]
testing_df  <- df[-trainIndex,]

# Check if the splitting process is correct
prop.table(table(training_df$churn))

prop.table(table(testing_df$churn))

# Model Training (Logistic Regression)


LR_model = glm(churn ~ ., data = training_df, family = "binomial")
summary(LR_model)

#Call:
  glm(formula = churn ~ ., family = "binomial", data = training_df)
  

# (From the summary above, we can drop feature of "credit_card", "estimated_salary" from the training model,
#thus they’re not statistical significance to the target column (p-value > 0.05)

LR_model = glm(churn ~ credit_score + country + gender + age + tenure + balance + products_number + active_member, data = training_df, family = "binomial")
summary(LR_model)

#Call:
glm(formula = churn ~ credit_score + country + gender + age + tenure + balance + products_number + active_member, family = "binomial", data = training_df)


#After dropping those features, we can notice that the statistical significance of credit_score, country, gender,balance has significantly increase. Apart from that, the deviance residuals has also move closer to 0 and AIC reduces as well.

#Apart from checking the p-value, we can also check on the VIF of features. Variance inflation factor (VIF) provides a measure of multicollinearity among the independent variables in a multiple regression model. Multicollinearity exist when two/ more predictor are highly relative to each other and it will become difficult to understand the impact of an independent variable.

#One of the assumptions from logistic regression is the feature should be independent. A predictor having a VIF of 2 or less is generally considered safe and it can be assumed that it is not correlated with other predictor variables. Higher the VIF, greater is the correlation of the predictor variable with other predictor variables.


##From the result below, all the feature selected is good to use for training the model.

vif(LR_model)


# Performance of model on testing data set
pred2 <- predict(LR_model,testing_df,type="response")
cutoff_churn <- ifelse(pred2>=0.50, 1,0)
cm <- confusionMatrix(as.factor(testing_df$churn),as.factor(cutoff_churn),positive ='1')
cm
#From above, Logistic Regression Result. The model has achieved 81% of accuracy, 70% of sensitivity and 82.77% of specificity. The Area Under Curve for this model achieves 80% which is considered a good result.


# Plot ROC Curve
ROCpred = prediction(pred2, testing_df$churn)
ROCperf <- performance(ROCpred, "tpr", "fpr")
plot(ROCperf, colorize=TRUE)
abline(a=0, b=1)
auc_train <- round(as.numeric(performance(ROCpred, "auc")@y.values),2)
legend(.8, .2, auc_train, title = "AUC", cex=1)

#Decision Tree
#A supervised machine learning model that works as flow chat that used to visualize the decision-making process by mapping out different courses of action, as well as their potential outcomes.

#We first build the decision tree with all the feature. However, fitting all the features into the model is always not the best choice. From the summary of the model, we obtain the result of CP, which stands for Complexity Parameter. It refers to the trade-off between the size of a tree and the error rate that help to prevent overfitting. So we want the cp value of the smallest tree that is having the smallest cross validation error.

Dtree = rpart(churn ~., data = training_df, method = "class")
printcp(Dtree)


# Plot Full Tree
prp(Dtree, type = 1, extra = 1, under = TRUE, split.font = 2, varlen = 0) 


# Find the best pruned Decision Tree by selecting the tree that is having least cross validation error.

set.seed(12345)
cv.ct <- rpart(churn ~., data = training_df, method = "class", 
               cp = 0.00001, minsplit = 5, xval = 5)
printcp(cv.ct)


# Prune by lowest cp
prune_dt <- prune(cv.ct,cp=cv.ct$cptable[which.min(cv.ct$cptable[,"xerror"]),"CP"])
predict_dt <- predict(prune_dt, testing_df,type="class") 
length(prune_dt$frame$var[prune_dt$frame$var == "<leaf>"])

prp(prune_dt, type = 1, extra = 1, split.font = 1, varlen = -10)

# Confusion Matrix and Statistics

cm_dt <- confusionMatrix(as.factor(testing_df$churn),as.factor(predict_dt),positive='1')
cm_dt

# Decision Tree Result,the model has achieved 86.4% of accuracy, 84% of sensitivity and 86% of specificity. The Area Under Curve for this model achieves 79% slightly lower compared to logistic regression.


pred_dt <- predict(prune_dt, newdata= testing_df,type = "prob")[, 2]
Pred_val = prediction(pred_dt, testing_df$churn) 
plot(performance(Pred_val, "tpr", "fpr"),colorize=TRUE)
abline(0, 1, lty = 2)
auc_train <- round(as.numeric(performance(Pred_val, "auc")@y.values),2)
legend(.8, .2, auc_train, title = "AUC", cex=1)



## 1. Machine Learning - Regression
# The Tenure variable will be used as the target variable to predict how long (year) the bank customer will stay with the bank.

set.seed(1000)
df[, names(stanbic)] = apply(df[, names(stanbic)], 2, function(x) as.numeric(as.character(x)))
trainIndex <- createDataPartition(df$tenure, p=0.8, list=FALSE, times=1)
data_train <- df[trainIndex,]
data_test <- df[-trainIndex,]




# Model Training (Linear Regression)

# set method as "lm" to train a linear regression model using the training data.

linRegModel <- train(tenure ~., data = data_train, method = "lm")
summary(linRegModel)

# From the summary above, only the "active_member variable is statistically significant in predicting the Tenure target outcome. The adjusted R-square achieved by the model is 0.0009634 , which is considered extremely low as it is far from the perfect score of 1.


# set method as "lm" to train a linear regression model and use 5-fold cross validation on the whole data. 
#In the following code, 5-fold cross validation is used for the linear regression model to see if the model performance can be improved.

linRegModelcv <- train(tenure ~., data = df, method = "lm", trControl = trainControl(method="cv", number=5))
summary(linRegModelcv$finalModel)

# From the summary above, we have now two variables that are statistically significant which are "active_member and credit_card. The adjusted R-square (0.001304) is slightly improved but the score achieved is still considered very low.

# root mean square error function
rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}

# performance of model on testing data set 
pred_tenure = as.integer(predict(linRegModel, data_test))
rmse(data_test$tenure, pred_tenure)
# The linear regression model has achieved RMSE score of 2.944713.

# 2. Regression Tree

# set method as anova for regression tree
modelTree <- rpart(tenure ~., data= data_train, method="anova")
summary(modelTree)

rpart.plot(modelTree)

## From the model summary and tree plot above, we notice that the regression tree is not able to grow. This happens because the independent variables are not useful in predicting the Tenure target outcome, hence the information provided by the independent variables are insufficient to grow the tree.


#Accuracy of model on testing data set

pred_tenure2 <- as.integer(predict(modelTree,  data_test, type="vector"))

rmse(data_test$tenure, pred_tenure2)



#The regression tree model has achieved RMSE score of 2.894164. The RMSE score of this model is slightly better than the RMSE of linear regression model. However, neither of the two models achieved a good model performance that is ready for deployment because the independent variables in this data set are not contributing to the prediction of the Tenure target variable.


{
  
 # Conclusion
  
  #Some interesting findings in the dataset:
  
  #Older customers are churning more than younger ones alluding to a difference in service preference in the age categories. The bank may need to review their target market or review the strategy for retention between the different age groups.
  
  #Having a credit card is not a good predictor for churn status mainly due to the high credit card ownership in Germany, France and Spain.
  
  #Credit Score may be perceived as an important factor, but its significance is minimal among the other factors given that the distribution of credit score is similar for churned and retained customers.
  
  #Clients with the longest and shortest tenure are more likely to churn compared to those that are of average tenure.
  
  #Surprisingly, the churning customers are also those who have a greater bank balance with the bank. This should be concerning to the bank as they are losing customers that provide higher capitals.
  
  #In predicting if a customer will churn or not, we employed 3 types of models: Logistics Regression, Decision Tree and Support Vector Machine. The performances of the models are fairly good with accuracies ranging from 81% - 86%. Other performance metrics that we considered are sensitivity(recall), precision f1-scores and the Area Under Curve of ROC.
  
  #Overall, Decision Tree is the best model for predicting churn among the three models.
  
  #For objective 2, neither of the regression models can a good prediction on how long a bank customer will stay with the bank because the independent variables are not useful in contributing to the prediction of our Tenure target variable.
  
  #Recommendations:
    
  #Regression works better with more continuous data as features. Active Member status is quite arbitrary and it is better to replaced with other useful features such as Recency, Frequency and Lifetime Value of customers that captures customer behaviour in interacting with the services provided by the bank.
  
  #Through this project, we learn that a good quality data set is important as it directly influence the performance of machine learning models and the independent variables have to be relevant in order to contribute to the prediction of the target variable.
  
}

