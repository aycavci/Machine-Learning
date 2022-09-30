data_set <- read.csv("D:/Downloads/engr421_dasc521_fall2019_hw05/hw05_data_set.csv")

eruptions <- data_set$eruptions
waiting <- data_set$waiting

x_train <- eruptions[1:150]
y_train <- waiting[1:150]

x_test <- eruptions[151:272]
y_test <- waiting[151:272]

minimum_value <- 1.5 
maximum_value <- 5.2

N_train <- length(x_train)
N_test <- length(x_test)

DecisionTreeRegression <- function(P) {
  node_splits <- c()
  node_means <- c()
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  while (1) {
    split_nodes <- which(need_split)
    
    if (length(split_nodes) == 0) {
      break
    }
    for (split_node in split_nodes) {
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      node_mean <- mean(y_train[data_indices])
      if (length(x_train[data_indices]) <= P) {
        is_terminal[split_node] <- TRUE
        node_means[split_node] <- node_mean
      } else {
        is_terminal[split_node] <- FALSE
        unique_values <- sort(unique(x_train[data_indices]))
        split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
        split_scores <- rep(0, length(split_positions))
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(x_train[data_indices] <= split_positions[s])]
          right_indices <- data_indices[which(x_train[data_indices] > split_positions[s])]
          total_error <- 0
          if (length(left_indices) > 0) {
            mean <- mean(y_train[left_indices])
            total_error <- total_error + sum((y_train[left_indices] - mean) ^ 2)
          }
          if (length(right_indices) > 0) {
            mean <- mean(y_train[right_indices])
            total_error <- total_error + sum((y_train[right_indices] - mean) ^ 2)
          }
          split_scores[s] <- total_error / (length(left_indices) + length(right_indices))
        }
        if (length(unique_values) == 1) {
          is_terminal[split_node] <- TRUE
          node_means[split_node] <- node_mean
          next 
        }
        best_split <- split_positions[which.min(split_scores)]
        node_splits[split_node] <- best_split
        
        left_indices <- data_indices[which(x_train[data_indices] < best_split)]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
        
        right_indices <- data_indices[which(x_train[data_indices] >= best_split)]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
  result <- list("splits"= node_splits, "means"= node_means, "is_terminal"= is_terminal)
  return(result)
}

P <- 25
result <- DecisionTreeRegression(P)
node_splits <- result$splits
node_means <- result$means
is_terminal <- result$is_terminal

get_prediction <- function(dp, is_terminal, node_splits, node_means){
  index <- 1
  while (1) {
    if (is_terminal[index] == TRUE) {
      return(node_means[index])
    } else {
      if (dp <= node_splits[index]) {
        index <- index * 2
      } else {
        index <- index * 2 + 1
      }
    }
  }
}

plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylim = c(floor(min(waiting)), ceiling(max(waiting))), xlim = c(minimum_value, maximum_value),
     ylab = "y", xlab = "x", las = 1, main = sprintf("P = %g", P))
points(x_test, y_test, type = "p", pch = 19, col= "red")

grid_interval <- 0.01
data_interval <- seq(from = minimum_value, to = maximum_value, by = grid_interval)

for (b in 1:length(data_interval)) {
  x_left <- data_interval[b]
  x_right <- data_interval[b+1]
  lines(c(x_left, x_right), c(get_prediction(x_left, is_terminal, node_splits, node_means), get_prediction(x_left, is_terminal, node_splits, node_means)), lwd = 2, col = "black")
  if (b < length(data_interval)) {
    lines(c(x_right, x_right), c(get_prediction(x_left, is_terminal, node_splits, node_means), get_prediction(x_right, is_terminal, node_splits, node_means)), lwd = 2, col = "black") 
  }
}

y_test_predicted <- sapply(X=1:N_test, FUN = function(i) get_prediction(x_test[i], is_terminal, node_splits, node_means))
RMSE <- sqrt(sum((y_test - y_test_predicted) ^ 2) / length(y_test))
sprintf("RMSE is %s when P is %s", RMSE, P)

RMSEs <- sapply(X=seq(5,50,5), FUN = function(p) {
  sprintf("Calculating RMSE for %d", p)
  result <- DecisionTreeRegression(p)
  node_splits <- result$splits
  node_means <- result$means
  is_terminal <- result$is_terminal
  y_test_predicted <- sapply(X=1:N_test, FUN = function(i) get_prediction(x_test[i], is_terminal, node_splits, node_means))
  RMSE <- sqrt(sum((y_test - y_test_predicted) ^ 2) / length(y_test))
})


plot(seq(5,50,5), RMSEs, pch = 19, col = "black", xlab = "Pre-pruning size (P)", ylab = "RMSE")
lines(seq(5,50,5), RMSEs)
