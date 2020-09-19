---
title: "Machine Learning Course Project"
author: "RJP"
date: "9/19/2020"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE, message=FALSE, warning=FALSE)
```

## Course Project Instructions

### Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

### Data
The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

*The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.*

### Goals

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

### Loading data and packages used in the analysis

```{r}
# Packages
 library(dplyr)
 library(caret)

# Train and test data sets
 TrainData <- read.csv(file="pml-training.csv")
 TestData <- read.csv(file="pml-testing.csv")
```

### Looking to the structure of the training data set and excluding some variables

```{r}
 str(TrainData)
```

As we can see above, some variables (columns) have a lot of NA (or blank) values. Lets do some calculations to quantify relative NA/blank values:

```{r}
# Counting NA values for each variable 
 naValues <- data.frame(sapply(TrainData, function(y) sum(length(which(is.na(y))))))
 naValues <- data.frame(rownames(naValues),naValues)
 rownames(naValues)<-NULL
 colnames(naValues)<- c("variable","naValues")

# Counting NA values stored as blank for each variable
 blankValues <- data.frame(sapply(TrainData, function(y) sum(length(which(y == "")))))
 blankValues <- data.frame(rownames(blankValues),blankValues)
 rownames(blankValues)<-NULL
 colnames(blankValues)<- c("variable","blankValues")
 
# Merge na and blank values
 na.blankValues <- merge(naValues, blankValues, by = "variable")
 na.blankValues <- mutate(na.blankValues, naBlank = naValues + blankValues,
                                          naBlankPerc = (naValues + blankValues) / nrow(TrainData)*100)
 
# Discovering how many variables have a lot of NA values
 table(round(na.blankValues$naBlankPerc,1))
```
As we can see above, 100 variables have almost 98% of NA values. Lets discard these variables because they could bring trouble to the analysis.   

```{r}
# Keeping not-NA variables
 notNA <- subset(na.blankValues, naBlank == 0)
 notNA <- notNA$variable
 TrainData <- select(TrainData,all_of(notNA))
 str(TrainData)
```
As we can see, there are some other variables to discard. Lets do that:

```{r}
 TrainData <- select(TrainData,-c("cvtd_timestamp","new_window","num_window","raw_timestamp_part_1","raw_timestamp_part_2","X"))
```

Now we can start the machine learning analyses.

### Creating train and test data sets

We can split data in two slices, 75% for training the model and 25% for test.

```{r}
# Split data into train and test
 DataPartition <- createDataPartition(TrainData$classe, p = 0.75, list = FALSE)
 MyTrainData = TrainData[DataPartition,]
 MyTestData = TrainData[-DataPartition,]
```

### Models tested

We will use two methods, Classification Tree and Random Forest, and see which of them performs better.

### Classification Tree

Lets see how the Classification Tree performs:

```{r}
# Classification tree
 set.seed(12345)
 ClassTreeModel <- train(classe ~ ., data = select(MyTrainData,-user_name), method="rpart")
 ClassTreeModel
```

Accuracy for Classification Tree Model was 52%. Lets see the performance for the test set:

```{r}
# Performance on test set
 PredClassTree <- predict(ClassTreeModel, newdata = MyTestData)
 CorrectPredClassTree <- data.frame(classe = MyTestData$classe, correct = PredClassTree == MyTestData$classe)
# Tables with absolute frequency of correct and incorrect predictions
 AbsFreqClassTree <- table(CorrectPredClassTree$classe,CorrectPredClassTree$correct)
 AbsFreqClassTree
 prop.table(AbsFreqClassTree,1)
```

The Classification Tree model only performs relatively well for the classe "A".

### Random Forest

Now, lets see how the Random Forest performs:

```{r}
# Random Forest
 set.seed(12345)
 RandForestModel <- train(classe ~ ., data = select(MyTrainData,-user_name), method="rf")
 RandForestModel
```

Accuracy for Random Forest Model was 99%. Lets see the performance for the test set:

```{r}
# Performance on test set
 PredRandForest <- predict(RandForestModel, newdata = MyTestData)
 CorrectPredRandForest <- data.frame(classe = MyTestData$classe, correct = PredRandForest == MyTestData$classe)
# Tables with absolute frequency of correct and incorrect predictions
 AbsFreqRandForest <- table(CorrectPredRandForest$classe,CorrectPredRandForest$correct)
 AbsFreqRandForest
 prop.table(AbsFreqRandForest,1)
```

As we can see above, the Random Forest model performs much better than Classification Tree model.

### Prediction of the 20 test cases

Lets use the Random Forest model to predict the new 20 test cases:

```{r}
 predict(RandForestModel, newdata = TestData)
# Expected error
 colSums(AbsFreqRandForest)/sum(AbsFreqRandForest)
```

Thus, we expect about 0.6% of error in new predictions using the Random Forest model.