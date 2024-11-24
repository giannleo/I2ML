library(mlr3)
library(data.table)
library(mlr3learners)

# Loading Task
task = tsk("german_credit")

# Splitting data into training and test set
# We want a 0.7 to 0.3 ratio
set.seed(123)
splits = partition(task, ratio = 0.7)

# Logistic Regression and retrieve target probabilities
logreg = mlr3::lrn("classif.log_reg")
logreg$predict_type = "prob"

# Train the model
logreg$train(task, splits$train)

# Define performance measures
measures = msrs(c("classif.acc"))

# Predict
logreg$predict(task, splits$test)$score(measures)
logreg$predict(task, splits$test)
