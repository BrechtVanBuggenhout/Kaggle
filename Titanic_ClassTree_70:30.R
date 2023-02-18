
# clear the environment
rm(list = ls())

# read in csv file and aprropriate libraries
library(Hmisc)
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


# removing rows without any embarked port
dat_train <- dat_train[!dat_train$Embarked == "",]
dat_test <- dat_test[!dat_test$Embarked == "",]

# create classification tree
library(corrplot)
library(pROC)
library(ROCR)
library(rpart)
form1 <- formula(Survived~.)
t1 <- rpart(form1, data=dat_train, cp=.001, method="class")
plot(t1,uniform=T,compress=T,margin=.05,branch=0.3)
text(t1, cex=.7, col="navy",use.n=TRUE)

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
yhat.t2 <- predict(t2, dat_test, type="prob")[,2]
table(yhat.t2>0.5,dat_test$Survived)

# plotting the ROC curve and calculating the AUC
t2.roc <- roc(dat_test$Survived, yhat.t2, direction="<")
t2.roc
plot(t2.roc, lwd=3)

# new predictions
yhat.t3 <- predict(t2, dat_test2, type="prob")[,2]
yhat.t3

# create dataset to show results of first predictions
round(yhat.t3)
df <- data.frame(round(yhat.t3))
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived_classtree.csv", row.names=FALSE)
df

##########################
##########################
##########################

# clear the environment
rm(list = ls())

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

# convert sex to binary values
df1_train$Sex <- ifelse(df1_train$Sex=="male",1,0)
dat_test_fin$Sex <- ifelse(dat_test_fin$Sex=="male",1,0)
dat_test_fin$Age <- ifelse(is.na(dat_test_fin$Age) == TRUE, 30.27, dat_test_fin$Age)
dat_test_fin$Fare <- ifelse(is.na(dat_test_fin$Fare) == TRUE, 35.63, dat_test_fin$Fare)

# removing rows without any embarked port
df1_train <- df1_train[!df1_train$Embarked == "",]

# converting embarked to factor
df1_train$Embarked <- as.factor(df1_train$Embarked)

# create classification tree
library(corrplot)
library(pROC)
library(ROCR)
library(rpart)
form1 <- formula(Survived~.)
t1 <- rpart(form1, data=df1_train, cp=.001, method="class")
plot(t1,uniform=T,compress=T,margin=.05,branch=0.3)
text(t1, cex=.7, col="navy",use.n=TRUE)

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
yhat.t2 <- predict(t2, dat_test_fin, type="prob")[,2]
table(yhat.t2>0.5,dat_test_fin$Survived)

# plotting the ROC curve and calculating the AUC
t2.roc <- roc(dat_test_fin$Survived, yhat.t2, direction="<")
t2.roc
plot(t2.roc, lwd=3)

# create dataset to show results of first predictions
round(yhat.t2)
df <- data.frame(round(yhat.t2))
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived_classtree.csv", row.names=FALSE)
df
