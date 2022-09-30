set.seed(421)
setwd("D:/Downloads/engr421_dasc521_fall2019_hw01")

labels <- read.csv("D:/Downloads/engr421_dasc521_fall2019_hw01/hw01_labels.csv", header = FALSE)
images <- read.csv("D:/Downloads/engr421_dasc521_fall2019_hw01/hw01_images.csv", header=FALSE)

train_data <- images[1:200,]
test_data <- images[201:400,]
labels_train <- labels[1:200,]
labels_test <- labels[201:400,]

female <- images[which(labels_train %in% c(1)),]
male <- images[which(labels_train %in% c(2)),]

means <- sapply(X = 1:2, FUN = function(c) {colMeans(train_data[labels_train==c,])})

female_row_mean_distance <- apply(female, 1, function(x) (x-means[,1])^2)
male_row_mean_distance <- apply(male, 1, function(x) (x-means[,2])^2)

female_deviations = sqrt(rowMeans(female_row_mean_distance, na.rm = FALSE, dims = 1))
male_deviations = sqrt(rowMeans(male_row_mean_distance, na.rm = FALSE, dims = 1))

deviations <- cbind(female_deviations, male_deviations)

priors <-sapply(X=1:2,FUN=function(c) {mean(labels==c)})

score_values_train <- sapply(X = 1:2, FUN = function(c) {rowSums(- 0.5 * log(2 * pi * matrix(deviations[,c], nrow(train_data), ncol(train_data), byrow = TRUE)^2) - 0.5 * (train_data - matrix(means[,c], nrow(train_data), ncol(train_data), byrow = TRUE))^2 / matrix(deviations[,c], nrow(train_data), ncol(train_data), byrow = TRUE)^2) + log(priors[c])})

score_values_test <- sapply(X = 1:2, FUN = function(c) {rowSums(- 0.5 * log(2 * pi * matrix(deviations[,c], nrow(test_data), ncol(test_data), byrow = TRUE)^2) - 0.5 * (test_data - matrix(means[,c], nrow(test_data), ncol(test_data), byrow = TRUE))^2 / matrix(deviations[,c], nrow(test_data), ncol(test_data), byrow = TRUE)^2) + log(priors[c])})

y_hat <- apply(score_values_train, 1, which.max)
confusion_matrix <- table(labels_train, y_hat)
print(confusion_matrix)

y_hat <- apply(score_values_test, 1, which.max)
confusion_matrix <- table(labels_test, y_hat)
print(confusion_matrix)
