---
title: "Detecting Errors in Weight Lifting"
author: "Bruce H"
date: "2020-03-30"
output: 
  html_document: 
    keep_md: yes
---



# Synopsis

In this study, I will examine the weight Lifting Exercise dataset from Velloso, et al.  In this dataset there are a number of readings from wearable accelerometers and the outcome is the manner in which the weight lifting is being performed.  I will determine a set of predictors and a model that will allow me to try to predict whether the subject is performing the weight lifting correctly (outcome "A") or making one of four common mistakes (outcomes "B" to "E").  I will do the predictor selection with subsets of the data, and then I will use the full training data to create a model using a random forest.


# Data Loading and Cleaning




## Reading the data


```r
training <- read.csv("pml-training.csv", stringsAsFactors = FALSE)
```

## Exploring the data

Aside from the first few columns and the last column, it was clear from the column names and the example data that the columns should be numeric.  I found that many columns were coded as character because the reading algorithm had found no good data in the number of rows that it used for checking the column type.  All of these columns and several other columns had many NA's and few rows with data.

I also determined that most of the first few columns were used to organize the data.  The column X is just row numbers, and the window numbers and timestamps are not relevant to future predictions.  In fact, because of the way that the outcomes are clumped together, these could end up being very good predictors within the training set and absolutely useless for handling any new data.  Cross validation and bootstrapping would not catch this error in the model building.

## Removing columns

I will change all the columns to numeric as needed, and then remove all columns that have values less than 50% of the time; this will not leave any NA's in the data.  I will also remove the columns that provide the row numbers, the timestamps and the window information, as none of these are relevant to my predictions.


```r
# Change columns that should be numeric

x <- lapply(training, class)
x1 <- x[x == "character"]
tr1 <- training
for (col1 in names(x1)[4:36]) {
    tr1[[col1]] <- as.numeric(tr1[[col1]])
}

# Get list of all with majority of NA's and remove

x2 <- numeric()
for (i in 1:dim(tr1)[2]) { if(mean(is.na(tr1[[i]])) > 0.5) x2 <- c(x2, i)}
tr2 <- select(tr1, -all_of(x2))

# Remove other unwanted columns

tr2 <- select(tr2, -all_of(c(1, 3:7)))
```

## Converting some columns to factors

Finally, I will change the user name and output classes to factor variables


```r
# Take care of classe
tr2$user_name <- as.factor(tr2$user_name)
tr2$classe <- as.factor(tr2$classe)
```


# Modeling

## Feature selection

Because feature selection algorithms can be very processor intensive, I will run it against two small subsets of the data rather than the full data set; each of these subsets will contain about 10% of the rows.  Then I will select the top features based on these two sets.

The improvement in predictability is less consistent after about 15 factors, though the best results are seen with between 25 and 30 factors.  To keep down the processing time, I will select the top 15 factors based on comparing the factor ordering from these two tests.  Lower numbers are better, so I will choose the ones with the lowest total scores from the two runs.


```r
set.seed(1001)
minitrain <- createDataPartition(tr2$user_name, p=0.1, list=FALSE)
tr2m = tr2[minitrain,]
```


```r
# Using RFE to check which factors to use
# see https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/

set.seed(1002)
control <- rfeControl(functions = rfFuncs, method="cv", number=10)
feature.time.1 <- system.time(results <- rfe(tr2m[,-c(1,54)],tr2m[,54], sizes = c(8:30), rfeControl = control))
```


```r
# Recheck with another minitrain set

set.seed(1003)
minitrain3 <- createDataPartition(tr2$classe, p=0.1, list=FALSE)
tr3m = tr2[minitrain3,]
set.seed(1004)
feature.time.2 <- system.time(res3 <- rfe(tr3m[,-c(1,54)],tr3m[,54], sizes = c(8:30), rfeControl = control))
```


```r
r2t <- data.frame(pred = predictors(results), rank = 1:length(predictors(results)))
r3t <- data.frame(pred = predictors(res3), rank = 1:length(predictors(res3)))
rtmatch <- inner_join(r2t, r3t, by = "pred")
rtmatch$rank <- rtmatch$rank.x + rtmatch$rank.y
rtmatch <- arrange(rtmatch, rank)
rtmatch$pred[1:15]
```

```
##  [1] "roll_belt"         "magnet_dumbbell_z" "magnet_dumbbell_y"
##  [4] "pitch_forearm"     "yaw_belt"          "pitch_belt"       
##  [7] "roll_dumbbell"     "roll_forearm"      "accel_dumbbell_y" 
## [10] "magnet_belt_y"     "magnet_belt_z"     "magnet_dumbbell_x"
## [13] "accel_forearm_x"   "gyros_belt_z"      "accel_dumbbell_z"
```

```r
trnarrow <- select(training, all_of(rtmatch$pred[1:15]), classe)
trnarrow$classe <- as.factor(trnarrow$classe)
```

Here is the plot of number of features vs. accuracy in one of the plots.  Note that all results have above 90% accuracy in small set that they come from.


```r
# predictors(results)
plot(results, type = c("g", "o"))
```

![](index_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

NOTE: Although I did not remove the variable "user_name" during data cleanup, I did not use it in my analysis.  Running a model similar to my final model with user_name added did not improve the accuracy, and the dummy variables created for this purpose had much lower importance ratings than the other variables selected.

## Fitting

Using this smaller set of predictors, I ran the caret train function to find the best random forest parameters and final model.  I used 10-fold cross validation in this process in order to get the cross validation accuracy predictions in the same process.


```r
set.seed(1020)
trControl <- trainControl(method = "cv", number = 10)
timertr <- system.time(mrfn.t <- train(classe ~ ., trnarrow, method = "rf", trControl = trControl))
r1 <- mrfn.t$results
mtry.sel <- mrfn.t$bestTune[1]
acc1 <- round(r1[r1$mtry == as.character(mtry.sel),2], 4)
```

The training process selected 8 variables at each branch as the best parameter, and has estimated an accuracy of 0.9908 for new data.

Here is the confusion matrix for the data in the training set.


```r
fin1 <- mrfn.t$finalModel
cm1 <- fin1$confusion[,1:5]
tr.acc <- sum(diag(cm1)) / sum(cm1)
print(fin1$confusion)
```

```
##      A    B    C    D    E class.error
## A 5563   11    3    1    2 0.003046595
## B   25 3734   33    5    0 0.016592046
## C    0   15 3394   13    0 0.008182350
## D    0    4   26 3181    5 0.010883085
## E    0    2    7    8 3590 0.004713058
```

This shows a training accuracy of 0.9918, which is a little higher than the predicted testing accuracy, as we would expect.

# Conclusions

Based on these tests, I should expect to predict the way the subject is doing their weight lifting based with high accuracy, even using only a subset of the measures provided by the original study.  On the other hand, because we had a limited number of subjects, and they were guided in how to make the errors, we would need further study to determine how well we can generalized these predictors to the general population working out without immediate guidance.

# Credits

The data comes from this study.

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>
