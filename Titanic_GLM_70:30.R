# clear the environment
rm(list = ls())

# read in csv file and aprropriate libraries
library(dplyr)
library(Hmisc)
library(pROC)
library(ROCR)
library(rpart)
library(MASS)
library(mvtnorm)
library(ggplot2)
library(class)
dat_train <- read.table("train.csv", sep=",", header=TRUE)
dat_test2 <- read.csv("test.csv")
dim(dat_train)
describe(dat_train)[,1:12]
head(dat_train)

# splitting the dataset --> 70% goes into training set
set.seed(42)
trn <- runif(nrow(dat_train)) < .7
dat_train <- dat_train[trn==TRUE,]
dat_test <- dat_train[trn==FALSE,]

# get rid of null values
dat_train <- na.omit(dat_train)
dat_test <- na.omit(dat_test)
describe(dat_train)
describe(dat_test)

# converting embarked to factor
dat_train$Embarked <- as.factor(dat_train$Embarked)
dat_test$Embarked <- as.factor(dat_test$Embarked)
dat_test2$Embarked <- as.factor(dat_test2$Embarked)

# removing rows without any embarked port
dat_train <- dat_train[!dat_train$Embarked == "",]
dat_test <- dat_test[!dat_test$Embarked == "",]
dat_test2 <- dat_test2[!dat_test2$Embarked == "",]

# convert sex to binary values
dat_train$Sex <- ifelse(dat_train$Sex=="male",1,0)
dat_test$Sex <- ifelse(dat_test$Sex=="male",1,0)
dat_test2$Sex <- ifelse(dat_test2$Sex=="male",1,0)
dat_test$Age <- ifelse(is.na(dat_test$Age) == TRUE, 30.27, dat_test$Age)
dat_test2$Age <- ifelse(is.na(dat_test2$Age) == TRUE, 30.27, dat_test2$Age)
dat_test$Fare <- ifelse(is.na(dat_test$Fare) == TRUE, 35.63, dat_test$Fare)
dat_test2$Fare <- ifelse(is.na(dat_test2$Fare) == TRUE, 35.63, dat_test2$Fare)

Y.tst <- dat_test$Survived

# get rid of columns that are not useful to predictions
# name, passengerid, ticket, and cabin
dat_train <- dat_train[,-c(1,4,9,11)]
dat_test <- dat_test[,-c(1,4,9,11)]
dat_test2 <- dat_test2[,-c(1,3,8,10)]

# remove embarked
dat_train <- dat_train[,-c(8)]

# run glm
glm <- glm(Survived ~., family = "binomial", data = dat_train)
summary(glm)

# predictions of this glm
yhat1 <- predict(glm, dat_test, type = "response")
test.glm <- rep(0,length(dat_test[,1]))
test.glm[yhat1 > 0.5] <- 1
table(test.glm, dat_test$Survived)


# AUC and ROC
g1.roc <- roc(dat_test$Survived, yhat1, direction="<")
g1.roc
plot(g1.roc, lwd=3)

# predicting for new data
yhat2 <- predict(glm, dat_test2, type = "response")
yhat2

# create dataset to show results of first predictions
round(yhat2)
df <- data.frame(round(yhat2))
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived_glm.csv", row.names=FALSE)
df

#########################
#########################
#########################


# clear the environment
rm(list = ls())

# read in csv file and aprropriate libraries
library(dplyr)
library(Hmisc)
library(pROC)
library(ROCR)
library(rpart)
library(MASS)
library(mvtnorm)
library(ggplot2)
library(class)
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

# remove 0 value for fare in train data
df1_train <- df1_train[df1_train$Fare > 0,] #7 observations removed

# convert sex to binary values
df1_train$Sex <- ifelse(df1_train$Sex=="male",1,0)
dat_test_fin$Sex <- ifelse(dat_test_fin$Sex=="male",1,0)
dat_test_fin$Age <- ifelse(is.na(dat_test_fin$Age) == TRUE, 30.27, dat_test_fin$Age)
dat_test_fin$Fare <- ifelse(is.na(dat_test_fin$Fare) == TRUE, 35.63, dat_test_fin$Fare)
df1_train$Fare <- log(df1_train$Fare)
dat_test_fin$Fare <- log(dat_test_fin$Fare)

# removing rows without any embarked port
df1_train <- df1_train[!df1_train$Embarked == "",]

# converting embarked to factor
df1_train$Embarked <- as.factor(df1_train$Embarked)

# remove embarked
df1_train <- df1_train[,-c(8)]

# run glm
glm <- glm(Survived ~., family = "binomial", data = df1_train)
summary(glm)

# predictions of this glm
yhat1 <- predict(glm, dat_test_fin, type = "response")
test.glm <- rep(0,length(dat_test_fin[,1]))
test.glm[yhat1 > 0.5] <- 1
table(test.glm, dat_test_fin$Survived)


# AUC and ROC
g1.roc <- roc(dat_test_fin$Survived, yhat1, direction="<")
g1.roc
plot(g1.roc, lwd=3)

# create dataset to show results of first predictions
round(yhat1)
df <- data.frame(round(yhat1))
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived_glm.csv", row.names=FALSE)
df