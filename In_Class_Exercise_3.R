library(mlr3)
library(data.table)
library(mlr3learners)

# Loading Task
task = tsk("german_credit")

# Splitting data into training and test set
set.seed(123)
splits = partition(task, ratio = 0.7)

# Logistic Regression
logreg = mlr3::lrn("classif.log_reg")
logreg$predict_type = "prob"

# Train the model
logreg$train(task, splits$train)

# Define measures 
measures = msrs(c("classif.acc"))

# Predict
logreg$predict(task, splits$test)$score(measures)
logreg$predict(task, splits$test)
