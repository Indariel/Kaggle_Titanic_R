# Titanic ML

# load packages

library(dplyr)
library(ggplot2)
library(scales)
library(car)
library(reshape2)
library(mice)
library(VIM)
library(caret) # ML packages

# read data

train <- read.csv("train.csv")
test <- read.csv("test.csv")

complete.df <- bind_rows(test,train)

# Missing data

pMiss <- function(x){sum(is.na(x))/length(x)*100}
apply(train,2,pMiss)
apply(train,1,pMiss)

aggr_plot <- aggr(train, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(data), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))
aggr_plot

aggr_plot_test <- aggr(test, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(data), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))
aggr_plot_test

# Impute Data (Age)

# Get predictor matrix

predMat <- quickpred(train)
predMat["Age",c("Name","Cabin", "Embarked")] <- 0
predMat["Age",c("Sex","Survived")] <- 1

tempTrain <- mice(train,m=5,maxit=50,meth='pmm', predictorMatrix = predMat, seed=1234)
summary(tempTrain)

completedTrain <- complete(tempTrain,1)

predMat <- quickpred(test)
predMat["Age",c("Name","Cabin", "Embarked", "Ticket")] <- 0
predMat["Age",c("Sex","sibAp","Parch")] <- 1
predMat["Fare",c("Cabin", "Embarked")] <- 0

tempTest <- mice(test,m=5,maxit=50,meth='pmm', predictorMatrix = predMat, seed=1234)
summary(tempTest)

completedTest <- complete(tempTest,1)

# Plot imputed data agianst other covariates

xyplot(tempTrain,Age ~ Survived + Sex + Pclass + SibSp + Parch + Fare,
       pch=18,cex=1)

densityplot(tempTrain)

stripplot(tempTrain, pch = 20, cex = 1.2)

# Graphical OVerview

train$Pclass <- as.numeric(train$Pclass)

ggplot(train, aes(x = Pclass, fill = Survived)) +
  geom_bar(aes(fill=as.factor(Survived)), position="fill") +
  xlab("Pclass") +
  ylab("Percentage") +
  labs(fill = "Survived") 

# Correlations

cormat <- round(cor(train[c("Pclass","Sex","Age", "SibSp", "Parch", "Fare", "Survived")], 
                    use = "complete.obs"),2)
cormat

# Logistic Regression

fit <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare,
           family=binomial(link='logit'),
           data = train)

summary(fit)

# Pool Data for Log Reg

modelFit1 <- with(tempTrain,glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare,
                                family=binomial(link='logit')))

format(summary(pool(modelFit1)), scientific = FALSE)

# use pooled estimates for prediction

pooled <- pool(modelFit1)

# Copy one of the fitted lm models fit to
#   one of the imputed datasets
pooled_lm = modelFit1$analyses[[1]]

# Replace the fitted coefficients with the pooled
#   estimates (need to check they are replaced in
#   the correct order)
pooled_lm$coefficients = summary(pooled)$estimate

# Predict - predictions seem to match the
#   pooled coefficients rather than the original
#   lm that was copied

predict.glm(modelFit1$analyses[[1]], newdata = test,type = "response")
predictTest <- predict.glm(pooled_lm, newdata = completedTest, type = "response")

Survived <- as.numeric(predictTest >= 0.5)
table(is.na(Survived))

#

submit <- data.frame(PassengerId = completedTest$PassengerId, Survived = Survived)

write.csv(submit, file = "logreg.csv", row.names = FALSE)
