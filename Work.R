## Work.R - Saved commands

library(dplyr)
library(caret)
library(ggplot2)


training <- read.csv("pml-training.csv", stringsAsFactors = FALSE)


# Shows that column X is just row number

x <- training$X != 1:nrow(training)
sum(x)

# Remove columns that are majority NA's
# First change char to num as needed

x <- lapply(training, class)
x1 <- x[x == "character"]
t1 <- select(training, all_of(names(x1)))
lapply(t1, max)

# All have values, and all between new_window and classe should be numeric.

tr1 <- training
for (col1 in names(x1)[4:36]) {
    tr1[[col1]] <- as.numeric(tr1[[col1]])
}

# Get list of all with majority of NA's and remove

xl <- numeric()
for (i in 1:dim(tr1)[2]) { if(mean(is.na(tr1[[i]])) > 0.5) xl <- c(xl, i)}
tr2 <- select(tr1, -all_of(xl))


# None have an NA average above 0 and less than or equal to 0.5

xl2 <- numeric()
for (i in 1:dim(tr2)[2]) { if(mean(is.na(tr2[[i]])) > 0) xl2 <- c(xl2, i)}
xl2

# Person Names are same in train and test, so we will keep those.

# Removing other columns

head(names(tr2),10)
tr2 <- select(tr2, -all_of(c(1, 3:7)))
head(names(tr2),10)

# We should have 54 left
dim(tr2)

tr2[,54] <- as.factor(tr2[,54])
minitrain <- createDataPartition(tr2$user_name, p=0.1, list=FALSE)
tr2m = tr2[minitrain,]
mtr <- train(classe ~ ., tr2m[,45:54], method = "rpart")
mtr <- train(classe ~ ., tr2m, method = "rpart")

mtr$finalModel$tuneValue
mtr$finalModel$splits
mtr$finalModel$variable.importance

# These take some time, even with limited variables on a 10% subset
mrf1 <- train(classe ~ ., tr2m[,45:54], method = "rf")
mrf2 <- train(classe ~ ., tr2m[,c(35:44,54)], method = "rf")
plot(mrf2)
mrf2$finalModel
mrf2$finalModel$importance
mrf3 <- train(classe ~ ., tr2m[,c(25:34,54)], method = "rf")
mrf4 <- train(classe ~ ., tr2m[,c(15:24,54)], method = "rf")
mrf5 <- train(classe ~ ., tr2m[,c(1:14,54)], method = "rf")

# The first three did better with 2 vars, but 4 liked 6 vars and 5 liked 10 vars (#Randmly selected predictors)

# Using RFE to check which factors to use
# see https://machinelearningmastery.com/feature-selection-with-the-caret-r-package/

control <- rfeControl(functions = rfFuncs, method="cv", number=10)

results <- rfe(tr2m[,2:14],tr2m[,54], sizes = c(6:14), rfeControl = control)
predictors(results)
plot(results, type = c("g", "o"))


results <- rfe(tr2m[,-c(1,54)],tr2m[,54], sizes = c(20:30), rfeControl = control)
# checked 1:20 separately, but those values were lower than a full set

# Recheck with another minitrain set

minitrain3 <- createDataPartition(tr2$classe, p=0.1, list=FALSE)
tr3m = tr2[minitrain3,]
res3 <- rfe(tr3m[,-c(1,54)],tr2m[,54], sizes = c(8:30), rfeControl = control)

# results were very similar

# Reduce number of variables and create model for full set.

r2t <- data.frame(pred = predictors(results), rank = 1:length(predictors(results)))
r3t <- data.frame(pred = predictors(res3), rank = 1:length(predictors(res3)))
rtmatch <- inner_join(r2t, r3t, by = "pred")
rtmatch$rank <- rtmatch$rank.x + rtmatch$rank.y
rtmatch <- arrange(rtmatch, rank)
rtmatch$pred[1:15]
trnarrow <- select(training, all_of(rtmatch$pred[1:15]), classe)
trnarrow$classe <- as.factor(trnarrow$classe)
system.time(mrfn <- randomForest(classe ~ ., data = trnarrow))
# this took about 22 sec, not like train(method = "rf"), but it doesn't look like it tried many possibilities of settings.

# With full training (will test mult values for mtry, number of variables select at each split)
system.time(mrfn.t <- train(classe ~ ., trnarrow, method = "rf"))
#user  system elapsed 
#1791.44   39.47 1836.55
# = 30 min

mrfn.t
mrfn.t$finalModel
varImpPlot(mrfn.t$finalModel)
mrfn.t$results
par.old <- par()
trellis.par.set(caretTheme())
plot(mrfn.t)

# Accuracy using CV
trControl <- trainControl(method = "cv", number = 10)
system.time(mrfn.t <- train(classe ~ ., trnarrow, method = "rf", trControl = trControl))
r1 <- mrfn.t$results
mtry.sel <- mrfn.t$bestTune[1]
acc1 <- round(r1[r1$mtry == as.character(mtry.sel),2], 4)

# Confusion Matrix

fin1 <- mrfn.t$finalModel
fin1$confusion

# Predictions
nuse <- names(trnarrow)[1:15]
tenarrow <- select(testing, all_of(nuse))
predict(fin1, newdata = tenarrow)

# Can't use testing as is, because the system complains of the NA's,
# even though they don't occur in the columns that are used.

