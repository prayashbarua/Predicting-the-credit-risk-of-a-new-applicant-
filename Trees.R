library(dplyr)
library(magrittr)
library(rpart)
library(RWeka)
#library(caret)
library(e1071)
library(ROCR)
library(randomForest)
#Read the German Credit Dataset
credit_data <- as.data.frame(read.csv("GermanCredit_assgt.csv"))

str(credit_data)setwd

attach(credit_data)

#Proportion of 0's and 1's in the response variable
table(RESPONSE)

#-------------Data Transformations --------------------

#Find columns that have missing values
credit_missing <- colnames(credit_data)[apply(is.na(credit_data), 2, any)]

#Replace missing values with zero
credit_data <- mutate(credit_data,
                      NEW_CAR = ifelse(is.na(NEW_CAR), 0, NEW_CAR),
                      USED_CAR = ifelse(is.na(USED_CAR), 0, USED_CAR),
                      FURNITURE = ifelse(is.na(FURNITURE), 0, FURNITURE),
                      RADIO.TV = ifelse(is.na(RADIO.TV), 0, RADIO.TV),
                      EDUCATION = ifelse(is.na(EDUCATION), 0, EDUCATION),
                      RETRAINING = ifelse(is.na(RETRAINING), 0, RETRAINING),
                      AGE = ifelse(is.na(AGE),mean(AGE,na.rm = TRUE), AGE))

#Column names of categorical variables
cols_cat <- c("CHK_ACCT","HISTORY","NEW_CAR","USED_CAR",
           "FURNITURE","RADIO.TV","EDUCATION","RETRAINING","SAV_ACCT",
           "EMPLOYMENT","MALE_DIV","MALE_SINGLE","MALE_MAR_or_WID",
           "CO.APPLICANT","GUARANTOR","PRESENT_RESIDENT","REAL_ESTATE","PROP_UNKN_NONE","OTHER_INSTALL",
           "RENT","OWN_RES","NUM_CREDITS","JOB","TELEPHONE","FOREIGN","RESPONSE")

#Convert categorical variables to factors
credit_data %<>% mutate_each_(funs(factor(.)),cols_cat)

#Column names of continuous variables
cols_cont <- c("DURATION","AMOUNT","INSTALL_RATE","AGE","NUM_DEPENDENTS")

#Extract continuous columns
credit_cont <- credit_data[,cols_cont]

#Calculate mean, median and standard deviation of continuous variables
credit_mean <- apply(credit_cont,2, function(x) mean(x))
credit_median <- apply(credit_cont,2, function(x) median(x))
credit_sd <- apply(credit_cont,2, function(x) sd(x))

#Extract categorical columns
credit_cat <- credit_data[,cols_cat]

#Calculate frequencies of different category values
credit_freq <- apply(credit_cat,2, function(x) table(x))


##Use various algorithms to build the models

#Develop Decision Tree on the full data
credit_fit1 <- rpart(RESPONSE ~ .,data=credit_data,method="class",
                    control = rpart.control(minsplit = 8,
                                            minbucket = 6,
                                            maxdepth = 10))

#Make prediction
predict_full <- predict(credit_fit1,credit_data,type="class")

#Confusion matrix for training set
confusionMatrix(data = predict_full, reference = credit_data$RESPONSE, positive = "1")

#Plot Gain Chart
predict_full1 <- predict(credit_fit1,credit_data,type="prob")

#Plot ROC curve
pred1 <- prediction(predict_full1[,2],credit_data$RESPONSE)
gain <- performance(pred,"tpr","rpp")
plot(gain)


printcp(credit_fit1)
plotcp(credit_fit1)
summary(credit_fit1)

# plot tree 
plot(credit_fit1, uniform=TRUE, 
     main="Classification Tree for German Credit Dataset")
text(credit_fit1, use.n=TRUE, all=TRUE, cex=.8)

#Plot Lift chart
lift.chart(credit_fit1, data=credit_data, targLevel="1",
           trueResp=0.01, type="cumulative", sub="Validation")


#Divide data into train and test data set

# Try out various sample sizes
smp_size <- floor(0.5 * nrow(credit_data))

# set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(credit_data)), size = smp_size)

train <- credit_data[train_ind, ]
test <- credit_data[-train_ind, ]


#Develop Decision Tree on training data
credit_fit2 <- rpart(RESPONSE ~ .,data=train,method="class",
                     control = rpart.control(minsplit = 30))

predict_train <- predict(credit_fit2,train,type="class")
predict_test <- predict(credit_fit2,test,type="class")
printcp(credit_fit2)
plotcp(credit_fit2)
summary(credit_fit2)

#Prune the tree
pfit<- prune(credit_fit2, cp= credit_fit2$cptable[which.min(credit_fit2$cptable[,"xerror"]),"CP"])

#Plot tree 
plot(pfit, uniform=TRUE, 
     main="Pruned Classification Tree for German Credit Dataset")
text(pfit, use.n=TRUE, all=TRUE, cex=.8)

#Develop a Random Forest Model
credit_fit2 <- randomForest(as.factor(RESPONSE) ~ .,data=train,
                            importance=TRUE,
                            ntree=200)

varImpPlot(credit_fit2)

predict_train <- predict(credit_fit2,train,type="class")
predict_test <- predict(credit_fit2,test,type="class")


##Validation Metrics

#Confusion matrix for training set
confusionMatrix(data = predict_train, reference = train$RESPONSE, positive = "1")

#Confusion matrix for test set
confusionMatrix(data = predict_test, reference = test$RESPONSE, positive = "1")

#Predict probability of response variable
predict_train <- predict(credit_fit2,train,type="prob")
predict_test <- predict(credit_fit2,test,type="prob")

#Plot ROC curve
pred <- prediction(predict_test[,2],test$RESPONSE)
rocs <- performance(pred,"tpr","fpr")
plot(rocs)


