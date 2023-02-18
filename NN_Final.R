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
labs(x = "Fare") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

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

# create formula
form1 <- formula(df1_train_train$Survived~.)

# We commence with a good combination from prior studies
# maximal iterations of 100, and a decay= of 0.0001.
library(nnet)
n1 <- nnet(form1, data=df1_train_train, size=3, maxit=500, decay=0.001)

# lets check predictions
yhat.n1 <- predict(n1, df1_train_validation)

#create confusion matrix
table(yhat.n1[,1]>0.625, df1_train_validation$Survived)

# check the ROC curve and AUC
library("pROC")
n1.roc <- roc(df1_train_validation$Survived, yhat.n1[,1], direction="<")
n1.roc
plot(n1.roc, lwd=3)

# create dataset to show results of first predictions
yhat.n1.test <- predict(n1, dat_test_fin)
yhat.n1.test <- ifelse(yhat.n1.test[,1] >0.625,1,0)
df <- data.frame(yhat.n1.test)
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived.csv", row.names=FALSE)

# neural net 2
n2 <- nnet(form1, data=df1_train_train, size=10, maxit=200, decay=0.001)
yhat.n2 <- predict(n2, df1_train_validation)
table(yhat.n2[,1]>0.625, df1_train_validation$Survived)
n2.roc <- roc(df1_train_validation$Survived, yhat.n2[,1], direction="<")
n2.roc
plot(n2.roc, add = TRUE, col="blue")

# create dataset fro 2nd neural net
yhat.n2.test <- predict(n2, dat_test_fin)
yhat.n2.test <- ifelse(yhat.n2.test[,1] >0.625,1,0)
df2 <- data.frame(yhat.n2.test)
names(df2) <- c('Survived')
df2$PassengerId <- c(892:1309)
write.csv(df2, "Survived2.csv", row.names=FALSE)

# neural net 3
n3 <- nnet(form1, data=df1_train_train, size=10, maxit=500, decay=0.001)
yhat.n3 <- predict(n3, df1_train_validation)
table(yhat.n3[,1]>0.625, df1_train_validation$Survived)
n3.roc <- roc(df1_train_validation$Survived, yhat.n3[,1], direction="<")
n3.roc
plot(n3.roc, add = TRUE, col="red")

# create dataset fro 3rd neural net
yhat.n3.test <- predict(n3, dat_test_fin)
yhat.n3.test <- ifelse(yhat.n3.test[,1] >0.625,1,0)
df3 <- data.frame(yhat.n3.test)
names(df3) <- c('Survived')
df3$PassengerId <- c(892:1309)
write.csv(df3, "Survived3.csv", row.names=FALSE)

# lets use more iterations for neural net 4
n4 <- nnet(form1, data=df1_train_train, size=5, maxit=1000, decay=0.0001)
yhat.n4 <- predict(n4, df1_train_validation)
table(yhat.n4[,1]>0.625, df1_train_validation$Survived)
n4.roc <- roc(df1_train_validation$Survived, yhat.n4[,1], direction="<")
n4.roc
plot(n4.roc, add = TRUE, col="green")

# create dataset fro 4th neural net
yhat.n4.test <- predict(n4, dat_test_fin)
yhat.n4.test <- ifelse(yhat.n4.test[,1] >0.625,1,0)
df4 <- data.frame(yhat.n4.test)
names(df4) <- c('Survived')
df4$PassengerId <- c(892:1309)
write.csv(df4, "Survived4.csv", row.names=FALSE)

# compare nets
temp <- data.frame(yhat.n1[,1],yhat.n2[,1],yhat.n3[,1],yhat.n4[,1],df1_train_validation$Survived)
rho <- cor(temp)
corrplot(rho, method="number")
