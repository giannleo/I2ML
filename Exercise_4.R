# Exercise Sheet 4
# Exercise 2

# Loading libraries:
library(mlr3)
library(mlr3learners)
library(mlbench)
library(ml3viz)
library(ggplot2)
library(e1071)

sample_class_1 <- rnorm(500, mean = 1, sd = 1)
sample_class_2 <- rnorm(500, mean = 7, sd = 1)
sample_class_3 <- rnorm(500, mean = 4, sd = 2)

df <- cbind(sample_class_1, sample_class_2, sample_class_3)
df <- as.data.frame(df)                   

plot(density(df$sample_class_1), col = "red", 
     lwd = 2, xlim = c(-5, 10), ylim = c(0, 0.5), 
     main = "Kernel Density Estimation", xlab = "x", ylab = "Density")
lines(density(df$sample_class_2), col = "blue", lwd = 2)
lines(density(df$sample_class_3), col = "green", lwd = 2)

# New observations:
x_1 = -10
x_2 = 7

# What is the probability we observe x in each class?
dnorm(x_2, mean = 1, sd = 1)
dnorm(x_2, mean = 7, sd = 1)
dnorm(x_2, mean = 4, sd = 2)

# Exercise 3
data <- mlbench::mlbench.cassini(n = 1000)
data_df <- as.data.frame(data)

# Copy data frame:
data_manipulated <- data_df
# Add noise:
data_manipulated$x.2 <- data_manipulated$x.2 + rnorm(1000, mean = 0, sd = 0.5)

# Plot data using ggplot2:
ggplot(data_manipulated, aes(x = x.1, y = x.2, color = classes)) 
+ geom_point()

# Create a task and learn all three learners:
task <- mlr3::TaskClassif$new(
  id = "spirals_task",
  backend = data_manipulated,
  target = "classes")

# Define learners
learners <- list(
  mlr3::lrn("classif.lda"),
  mlr3::lrn("classif.qda"),
  mlr3::lrn("classif.naive_bayes"))

# Train and plot decision boundaries
plots <- lapply(learners, function(i) mlr3viz::plot_learner_prediction(i, task))
for (i in plots) print(i)
