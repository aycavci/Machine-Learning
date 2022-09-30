setwd("D:/Downloads/engr421_dasc521_fall2019_hw03")

images <- data.matrix(read.csv("D:/Downloads/engr421_dasc521_fall2019_hw03/hw03_images.csv", header=FALSE))
labels <- data.matrix(read.csv("D:/Downloads/engr421_dasc521_fall2019_hw03/hw03_labels.csv", header = FALSE))

v <- data.matrix(read.csv("D:/Downloads/engr421_dasc521_fall2019_hw03/initial_V.csv", header=FALSE))
w <- data.matrix(read.csv("D:/Downloads/engr421_dasc521_fall2019_hw03/initial_W.csv", header=FALSE))

train_data <- images[1:500,]
test_data <- images[501:1000,]
labels_train <- labels[1:500,]
labels_test <- labels[501:1000,]

y_truth <- matrix(0, 500, 5)
y_truth[cbind(1:500, labels_train)] <- 1

y_test <- matrix(0, 500, 5)
y_test[cbind(1:500, labels_test)] <- 1

safelog <- function(x) {
  return (log(x + 1e-100))
}

sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

softmax <- function(i1, i2) {
  scores <- cbind(1, i1) %*% i2
  scores <- exp(scores - matrix(apply(scores, MARGIN = 1, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

H <- 20
eta <- 0.0005
epsilon <- 1e-3
max_iteration <- 500

z <- sigmoid(cbind(1, train_data) %*% w)
y_predicted <- softmax(z, v)
objective_values <- -sum(y_truth * safelog(y_predicted))

iteration <- 1

while (1) {
  
  z <- sigmoid(cbind(1, train_data) %*% w)
  
  y_predicted <- softmax(z, v)
  
  delta_v <- eta * t(cbind(1, z)) %*% (y_truth - y_predicted)
  delta_w <- eta * t(cbind(1, train_data)) %*% (((y_truth - y_predicted) %*% t(v[2:(H + 1),])) * z * (1 - z))
  
  v <- v + delta_v
  w <- w + delta_w
  
  z <- sigmoid(cbind(1, train_data) %*% w)
  y_predicted <- softmax(z, v)
  objective_values <- c(objective_values, -sum(y_truth * safelog(y_predicted)))
  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}

plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

y_hat <- apply(y_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(y_hat, labels_train)
print(confusion_matrix)

z <- sigmoid(cbind(1, test_data) %*% w)
y_predicted <- softmax(z, v)

y_hat <- apply(y_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(y_hat, labels_test)
print(confusion_matrix)


