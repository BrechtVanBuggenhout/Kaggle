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