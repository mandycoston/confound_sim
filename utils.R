
sigmoid <- function(x) {
  return(1 / (1 + exp(-x)))
}

logit <- function(x) {
  return(log(x / (1 - x)))
}

compute_mse <- function(preds, label) {
  mse <- (preds - label)^2
  low <- mean(mse) - 1.96 * sqrt(var(mse) / length(mse))
  high <- mean(mse) + 1.96 * sqrt(var(mse) / length(mse))
  return(list(
    "mse" = mean(mse),
    "low" = low,
    "high" = high
  ))
}

compute_misclass <- function(t, preds, label) {
  pred_label <- ifelse(preds >= t, 1, 0)
  return(mean(label != pred_label))
}
