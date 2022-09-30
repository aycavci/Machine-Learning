setwd("D:/Downloads/engr421_dasc521_fall2019_hw02")

images <- data.matrix(read.csv("D:/Downloads/engr421_dasc521_fall2019_hw02/hw02_images.csv", header=FALSE))
labels <- data.matrix(read.csv("D:/Downloads/engr421_dasc521_fall2019_hw02/hw02_labels.csv", header = FALSE))

w0 <- data.matrix(read.csv("D:/Downloads/engr421_dasc521_fall2019_hw02/initial_w0.csv", header=FALSE))
w0 <- matrix(w0, 500, 5, byrow =TRUE)
W <- data.matrix(read.csv("D:/Downloads/engr421_dasc521_fall2019_hw02/initial_W.csv", header=FALSE))

train_data <- images[1:500,]
test_data <- images[501:1000,]
labels_train <- labels[1:500,]
labels_test <- labels[501:1000,]

y_truth <- matrix(0, 500, 5)
y_truth[cbind(1:500, labels_train)] <- 1

y_test <- matrix(0, 500, 5)
y_test[cbind(1:500, labels_test)] <- 1

sigmoid <- function(train_data, W, w0) {
  return (1 / (1 + exp(-(train_data %*% W + w0))))
}

gradient_W <- function(train_data, y_truth, y_predicted) {
  return (-sapply(X = 1:ncol(y_truth), function(c) colSums(matrix(((y_truth[,c] - y_predicted[,c]) * y_predicted[,c] * (1-y_predicted[,c])), nrow = nrow(train_data), ncol = ncol(train_data), byrow = FALSE) * train_data)))
}

gradient_w0 <- function(y_truth, y_predicted) {
  return (-colSums((y_truth - y_predicted) * y_predicted * (1-y_predicted)))
}

eta <- 0.0001
epsilon <- 1e-3
max_iteration <- 500

iteration <- 1
objective_values <- c()

while (1) {
  
  y_predicted <- sigmoid(train_data, W, w0)
  
  objective_values <- c(objective_values, 0.5*sum((y_truth-y_predicted)^2))
  
  W_old <- W
  w0_old <- w0
  
  W <- W - eta * gradient_W(train_data, y_truth, y_predicted)
  w0 <- w0 - eta * gradient_w0(y_truth, y_predicted)
 
  if (sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}

plot(1:max_iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

y_hat <- apply(y_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(y_hat, labels_train)
print(confusion_matrix)

y_test <- sigmoid(test_data, W, w0)

y_hat <- apply(y_test, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(y_hat, labels_test)
print(confusion_matrix)

