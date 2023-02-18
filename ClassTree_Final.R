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

form1 <- formula(Survived~.)
t1 <- rpart(form1, data=df1_train_train, cp=.001, method="class")
plot(t1,uniform=T,compress=T,margin=.05,branch=0.3)
text(t1, cex=.7, col="navy",use.n=TRUE)

# make predictions on first tree
yhat.t1 <- predict(t1, df1_train_validation, type="prob")[,2]
table(yhat.t1>0.625,df1_train_validation$Survived)

# plotting the t1
plotcp(t1)

# 
CP <- printcp(t1)

# the left variable is a good choice for pruning if below the line in classification trees
cp <- CP[,1][CP[,2]==2]
cp

# prune according to cp calculate above
t2 <- prune(t1,cp=cp[1])
plot(t2,uniform=T,compress=T,margin=.05,branch=0.3)
text(t2, cex=.7, col="navy",use.n=TRUE)

# make predictions based on tree 2
yhat.t2 <- predict(t2, df1_train_validation, type="prob")[,2]
table(yhat.t2>0.625,df1_train_validation$Survived)

# make predictions based on tree 1
yhat.t1 <- predict(t1, df1_train_validation, type="prob")[,2]
table(yhat.t1>0.625,df1_train_validation$Survived)

# plotting the ROC curve and calculating the AUC
t1.roc <- roc(df1_train_validation$Survived, yhat.t1, direction="<")
t1.roc
plot(t1.roc, lwd=3)

# plotting the ROC curve and calculating the AUC
t2.roc <- roc(df1_train_validation$Survived, yhat.t2, direction="<")
t2.roc
plot(t2.roc, lwd=3)

# create dataset to show results of first predictions
yhat.t1.test <- predict(t1, dat_test_fin)
yhat.t1.test <- ifelse(yhat.t1.test[,1] >0.625,1,0)
df <- data.frame(yhat.t1.test)
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived_classtree.csv", row.names=FALSE)
df

# create dataset to show results of first predictions
yhat.t2.test <- predict(t2, dat_test_fin)
yhat.t2.test <- ifelse(yhat.t2.test[,1] >0.625,1,0)
df <- data.frame(yhat.t2.test)
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived_classtreeprune.csv", row.names=FALSE)
df
