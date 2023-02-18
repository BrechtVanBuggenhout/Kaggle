
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

# create formula
form1 <- formula(dat_train$Survived~.)

# We commence with a good combination from prior studies
# maximal iterations of 100, and a decay= of 0.0001.
library(nnet)
set.seed(1)
n1 <- nnet(form1, data=dat_train, size=3, maxit=500, decay=0.001)

# lets check predictions
yhat.n1 <- predict(n1, dat_test)

#create confusion matrix
table(yhat.n1[,1]>0.5, dat_test$Survived)

# check the ROC curve and AUC
library("pROC")
n1.roc <- roc(dat_test$Survived, yhat.n1[,1], direction="<")
n1.roc
plot(n1.roc, lwd=3)

# predict for test set
yhat.n1.2 <- predict(n1, dat_test2)
yhat.n1.2

# create dataset to show results of first predictions
round(yhat.n1.2)
df <- data.frame(round(yhat.n1.2))
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived.csv", row.names=FALSE)

# neural net 2
set.seed(4)
n2 <- nnet(form1, data=dat_train, size=10, maxit=200, decay=0.001)
yhat.n2 <- predict(n2, dat_test)
table(yhat.n2[,1]>0.5, dat_test$Survived)
n2.roc <- roc(dat_test$Survived, yhat.n2[,1], direction="<")
n2.roc
plot(n2.roc, add = TRUE, col="blue")

# predict for test set
yhat.n2.2 <- predict(n2, dat_test2)
yhat.n2.2

# create dataset fro 2nd neural net
round(yhat.n2.2)
df2 <- data.frame(round(yhat.n2.2))
names(df2) <- c('Survived')
df2$PassengerId <- c(892:1309)
write.csv(df2, "Survived2.csv", row.names=FALSE)


# neural net 3
set.seed(4)
n3 <- nnet(form1, data=dat_train, size=10, maxit=500, decay=0.001)
yhat.n3 <- predict(n3, dat_test)
table(yhat.n3[,1]>0.5, dat_test$Survived)
n3.roc <- roc(dat_test$Survived, yhat.n3[,1], direction="<")
n3.roc
plot(n3.roc, add = TRUE, col="red")

# predict for test set
yhat.n3.2 <- predict(n3, dat_test2)
yhat.n3.2

# create dataset fro 2nd neural net
round(yhat.n3.2)
df3 <- data.frame(round(yhat.n3.2))
names(df3) <- c('Survived')
df3$PassengerId <- c(892:1309)
write.csv(df3, "Survived3.csv", row.names=FALSE)

# lets use more iterations for neural net 4
set.seed(4)
n4 <- nnet(form1, data=dat_train, size=5, maxit=1000, decay=0.0001)
yhat.n4 <- predict(n4, dat_test)
table(yhat.n4[,1]>0.5, dat_test$Survived)
n4.roc <- roc(dat_test$Survived, yhat.n4[,1], direction="<")
n4.roc
plot(n4.roc, add = TRUE, col="green")

# predict for test set
yhat.n4.2 <- predict(n4, dat_test2)
yhat.n4.2

# create dataset fro 2nd neural net
round(yhat.n4.2)
df4 <- data.frame(round(yhat.n4.2))
names(df4) <- c('Survived')
df4$PassengerId <- c(892:1309)
write.csv(df4, "Survived4.csv", row.names=FALSE)

# compare nets
temp <- data.frame(yhat.n1[,1],yhat.n2[,1],yhat.n3[,1],yhat.n4[,1],dat_test$Survived)
rho <- cor(temp)
corrplot(rho, method="number")

#########################
#########################
#########################

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

# create formula
form1 <- formula(df1_train$Survived~.)

# We commence with a good combination from prior studies
# maximal iterations of 100, and a decay= of 0.0001.
library(nnet)
set.seed(1)
n1 <- nnet(form1, data=df1_train, size=3, maxit=500, decay=0.001)

# lets check predictions
yhat.n1 <- predict(n1, dat_test_fin)

#create confusion matrix
table(yhat.n1[,1]>0.5, dat_test_fin$Survived)

# check the ROC curve and AUC
library("pROC")
n1.roc <- roc(dat_test_fin$Survived, yhat.n1[,1], direction="<")
n1.roc
plot(n1.roc, lwd=3)

# create dataset to show results of first predictions
round(yhat.n1)
df <- data.frame(round(yhat.n1))
names(df) <- c('Survived')
df$PassengerId <- c(892:1309)
write.csv(df, "Survived.csv", row.names=FALSE)

# neural net 2
set.seed(4)
n2 <- nnet(form1, data=df1_train, size=10, maxit=200, decay=0.001)
yhat.n2 <- predict(n2, dat_test_fin)
table(yhat.n2[,1]>0.5, dat_test_fin$Survived)
n2.roc <- roc(dat_test_fin$Survived, yhat.n2[,1], direction="<")
n2.roc
plot(n2.roc, add = TRUE, col="blue")

# create dataset fro 2nd neural net
round(yhat.n2)
df2 <- data.frame(round(yhat.n2))
names(df2) <- c('Survived')
df2$PassengerId <- c(892:1309)
write.csv(df2, "Survived2.csv", row.names=FALSE)

# neural net 3
set.seed(4)
n3 <- nnet(form1, data=df1_train, size=10, maxit=500, decay=0.001)
yhat.n3 <- predict(n3, dat_test_fin)
table(yhat.n3[,1]>0.5, dat_test_fin$Survived)
n3.roc <- roc(dat_test_fin$Survived, yhat.n3[,1], direction="<")
n3.roc
plot(n3.roc, add = TRUE, col="red")

# create dataset fro 2nd neural net
round(yhat.n3)
df3 <- data.frame(round(yhat.n3))
names(df3) <- c('Survived')
df3$PassengerId <- c(892:1309)
write.csv(df3, "Survived3.csv", row.names=FALSE)

# lets use more iterations for neural net 4
set.seed(4)
n4 <- nnet(form1, data=df1_train, size=5, maxit=1000, decay=0.0001)
yhat.n4 <- predict(n4, dat_test_fin)
table(yhat.n4[,1]>0.5, dat_test_fin$Survived)
n4.roc <- roc(dat_test_fin$Survived, yhat.n4[,1], direction="<")
n4.roc
plot(n4.roc, add = TRUE, col="green")

# create dataset fro 2nd neural net
round(yhat.n4)
df4 <- data.frame(round(yhat.n4))
names(df4) <- c('Survived')
df4$PassengerId <- c(892:1309)
write.csv(df4, "Survived4.csv", row.names=FALSE)

# compare nets
temp <- data.frame(yhat.n1[,1],yhat.n2[,1],yhat.n3[,1],yhat.n4[,1],dat_test_fin$Survived)
rho <- cor(temp)
corrplot(rho, method="number")

