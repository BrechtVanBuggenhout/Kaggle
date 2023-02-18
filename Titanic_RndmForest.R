# clear the environment
rm(list = ls())

# read in csv file and aprropriate libraries
library(Hmisc)
library(pROC)
library(ROCR)
library(rpart)
dat_train <- read.table("train.csv", sep=",", header=TRUE)
dat_test <- read.csv("test.csv")
gender <- read.csv("gender_submission.csv")
dim(dat_train)
describe(dat_train)[,1:12]
describe(dat_test)[,1:11]
head(dat_train)
dat_test <- merge(dat_test,gender, by = "PassengerId")

# get rid of columns that are not useful to predictions
# name, passengerid, ticket, and cabin
dat_train_fin <- dat_train[,-c(1,4,9,11)]
dat_test_fin <- dat_test[,-c(1,3,8,10)]

# get rid of null values
df1_train <- na.omit(dat_train_fin)
describe(df1_train)
describe(dat_test_fin)

# convert sex to binary values
df1_train$Sex <- ifelse(df1_train$Sex=="male",1,0)
dat_test_fin$Sex <- ifelse(dat_test_fin$Sex=="male",1,0)

# removing rows without any embarked port
df1_train <- df1_train[!df1_train$Embarked == "",]

# converting embarked to factor
df1_train$Embarked <- as.factor(df1_train$Embarked)
library(dplyr)

# Setting up the random forest
library(xgboost)
library(randomForest)
X <- as.matrix(df1_train[,-c(1,8)])
Y <- df1_train$Survived

mtry <- round(ncol(X)^0.5); mtry

# Fitting random forest to the dataset
ntree <- 1000
set.seed(652)
rf1 <- randomForest(x=X, y=Y, ntree=ntree, mtry=mtry, importance=TRUE)
rf1

# summary of the random forest
summary(rf1)
names(rf1)


# look at "importance' since it gives us the influence of each feature
importance(rf1)

# look at the rf1 in a plot setting
varImpPlot(rf1)

# 
a <- 30.27
describe(dtest)
dtest <- dat_test_fin[,-c(7)]
dtest$Age <- ifelse(is.na(dtest$Age) == TRUE, a, dtest$Age)
dtest$Fare <- ifelse(is.na(dtest$Fare) == TRUE, 35.63, dtest$Fare)

# evaluating predictions
pred.rf1 <- predict(rf1, dtest)
table(pred.rf1, dtest$Survived)
yhat.rf1 <- predict(rf1, dtest)
yhat.rf1

# let's look at the ROC curve and AUC
library(pROC)
rf1.roc <- roc(dtest$Survived, yhat.rf1, direction="<")
rf1.roc
plot(rf1.roc, lwd=3)

# create dataset to show results of first predictions
round(yhat.rf1)
df <- data.frame(round(yhat.rf1))
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived_rndmforest.csv", row.names=FALSE)
df
