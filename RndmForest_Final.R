# clear the environment
rm(list = ls())
# Loading the library
library(class)
library(ggplot2)
library(mvtnorm)
library(pROC)
library(MASS)
library(tree)
library(psych)
library(rpart)
library(randomForest)
library(gbm)
library(xgboost)
library(caret)
library(plyr)
library(dplyr)
library(Hmisc)
library(corrplot)
library(ROCR)
library(rpart)

# read in csv file and aprropriate libraries
library(Hmisc)
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
dat_test_fin$Age <- ifelse(is.na(dat_test_fin$Age) == TRUE, 30.27, dat_test_fin$Age)
dat_test_fin$Fare <- ifelse(is.na(dat_test_fin$Fare) == TRUE, 35.63, dat_test_fin$Fare)

# removing rows without any embarked port
df1_train <- df1_train[!df1_train$Embarked == "",]

# converting embarked to factor
df1_train$Embarked <- as.factor(df1_train$Embarked)

# how many people died versus how many people survived
table(df1_train$Survived)
library(ggplot2)

# histogram to show distribution of age
ggplot(df1_train, aes(x = Age)) +
  geom_histogram(breaks = seq(0,80, by = 5)) 
labs(x = "Age") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# histogram of fare --> using log
ggplot(df1_train, aes(x = log(Fare)) +
         geom_histogram(breaks = seq(0,10, by = 1)) 
       
df1_train %>%
         group_by(Embarked) %>%
         summarise(
           zeros = sum(Survived == 0),
           ones = sum(Survived == 1),
           n = n(),
           proportion = mean(Survived)
         )

# Split the training data into train_train and train_validation
set.seed(42)
trn <- runif(nrow(df1_train)) < .7
df1_train_train <- df1_train[trn,]
df1_train_validation <- df1_train[trn==FALSE,]

# Setting up the random forest
library(xgboost)
library(randomForest)


X <- as.matrix(df1_train_train[,-c(1,8)])
Y <- df1_train_train$Survived
dim(X)

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

# evaluating predictions
yhat.rf1 <- predict(rf1, df1_train_validation)
table(yhat.rf1>0.625,df1_train_validation$Survived)


# let's look at the ROC curve and AUC
library(pROC)
rf1.roc <- roc(df1_train_validation$Survived, yhat.rf1, direction="<")
rf1.roc
plot(rf1.roc, lwd=3)

# create dataset to show results of first predictions
yhat.rf1.test <- predict(rf1, dat_test_fin)
yhat.rf1.test <- ifelse(yhat.rf1.test >0.625,1,0)
df <- data.frame(yhat.rf1.test)
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived_rndmforest.csv", row.names=FALSE)
df


       
       
