data_set <- read.csv("D:/Downloads/engr421_dasc521_fall2019_hw04/hw04_data_set.csv")

eruptions <- data_set$eruptions
waiting <- data_set$waiting

x_train <- eruptions[1:150]
y_train <- waiting[1:150]

x_test <- eruptions[151:272]
y_test <- waiting[151:272]

minimum_value <- 1.5
maximum_value <- 5.2

bin_width <- 0.37

data_interval <- seq(from = minimum_value, to = maximum_value, by = 0.01)

left_borders <- seq(from = minimum_value, to = maximum_value - bin_width, by = bin_width)
right_borders <- seq(from = minimum_value + bin_width, to = maximum_value, by = bin_width)

#Regressogram

g_head <- sapply(1:length(left_borders), function(i) {
  bin <- y_train[left_borders[i] < x_train & x_train <= right_borders[i]]
  return(mean(bin))
})

plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylim = c(floor(min(waiting)), ceiling(max(waiting))), xlim = c(minimum_value, maximum_value),
     ylab = "Waiting time to next eruption (min)", xlab = "Eruption time (min)", las = 1, main = sprintf("h = %g", bin_width))
points(x_test, y_test,type = "p", pch = 19, col= "red")
for (b in 1:length(left_borders)) {
  lines(c(left_borders[b], right_borders[b]), c(g_head[b], g_head[b]), lwd = 2, col = "black")
  if (b < length(left_borders)) {
    lines(c(right_borders[b], right_borders[b]), c(g_head[b], g_head[b + 1]), lwd = 2, col = "black") 
  }
}

get_bin_no <- function(v) {
  return(ceiling((v-minimum_value) / bin_width))
}

distances <- sapply(1:length(y_test), function(i) {
  y_predicted <- g_head[get_bin_no(x_test[i])]
  diff <- y_test[i] - y_predicted
  return(diff^2)
})

RMSE <- sqrt(sum(distances) / length(distances))
sprintf("Regressogram => RMSE is %s when h is %s", RMSE, bin_width)

#Running Mean Smoother

g_head <- sapply(data_interval, function(x) {
  y_train_bin <- y_train[(x - 0.5 * bin_width) < x_train & x_train <= (x + 0.5 * bin_width)]
  return(mean(y_train_bin))
})


plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylim = c(floor(min(waiting)), ceiling(max(waiting))), xlim = c(minimum_value, maximum_value),
     ylab = "Waiting time to next eruption (min)", xlab = "Eruption time (min)", las = 1, main = sprintf("h = %g", bin_width))
points(x_test, y_test,type = "p", pch = 19, col= "red")
lines(data_interval, g_head, type = "l", lwd = 2, col = "black")

y_head <- sapply(x_test, function(x) {
  y_train_bin <- y_train[(x - 0.5 * bin_width) < x_train & x_train <= (x + 0.5 * bin_width)]
  return(mean(y_train_bin))
})

distances <- sapply(1:length(y_test), function(i) {
  diff <- y_test[i] - y_head[i]
  return(diff^2)
})

RMSE <- sqrt(sum(distances) / length(distances))
sprintf("Running Mean Smoother => RMSE is %s when h is %s", RMSE, bin_width)

#Kernel Smoother

gaussian_kernel = function(u) {
  (1 / sqrt((2 * pi))) * exp(-u^2 / 2)
}

g_head <- sapply(data_interval, function(x) {
  nominator <- sapply(1:length(x_train), function(i) {
    u <- (x - x_train[i]) / bin_width
    kernel <- gaussian_kernel(u)
    return(kernel*y_train[i])
  })
  denominator <- sapply(1:length(x_train), function(i) {
    u <- (x - x_train[i]) / bin_width
    kernel <- gaussian_kernel(u)
    return(kernel)
  })
  return(sum(nominator) / sum(denominator))
})

plot(x_train, y_train, type = "p", pch = 19, col = "blue",
     ylim = c(floor(min(waiting)), ceiling(max(waiting))), xlim = c(minimum_value, maximum_value),
     ylab = "Waiting time to next eruption (min)", xlab = "Eruption time (min)", las = 1, main = sprintf("h = %g", bin_width))
points(x_test, y_test,type = "p", pch = 19, col= "red")
lines(data_interval, g_head, type = "l", lwd = 2, col = "black")

y_head <- sapply(x_test, function(x) {
  nominator <- sapply(1:length(x_train), function(i) {
    u <- (x - x_train[i]) / bin_width
    kernel <- gaussian_kernel(u)
    return(kernel*y_train[i])
  })
  denominator <- sapply(1:length(x_train), function(i) {
    u <- (x - x_train[i]) / bin_width
    kernel <- gaussian_kernel(u)
    return(kernel)
  })
  return(sum(nominator) / sum(denominator))
})

distances <- sapply(1:length(y_test), function(i) {
  diff <- y_test[i] - y_head[i]
  return(diff^2)
})

RMSE <- sqrt(sum(distances) / length(distances))
sprintf("Kernel Smoother => RMSE is %s when h is %s", RMSE, bin_width)
