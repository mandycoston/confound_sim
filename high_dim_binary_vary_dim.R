# This is the file to run on AWS to vary dim

library(tidyverse)
library(glmnet)
library(doParallel)
source("utils.R")

results_folder <- "results/highdim/binaryk/"
start_time <- Sys.time()
set.seed(100)
for (k in c(10, 12, 15)) { # try 10 or 20
  results <- tibble()
  n <- round(4 * 500 * k)
  n_sim <- 75
  d <- round(100 * k^2)
  q <- round(d / 2) # dimension of hidden confounder z
  p <- d - q # dimension of v
  beta <- min(round(2 / 5 * n / 4 * log(p) / log(d)^2), p)
  zeta <- round(3 * beta / 5) # number of non-zero predictors in z
  gamma <- beta - zeta # number of non-zero predictors in v
  alpha <- round(beta / 2) # sparsity in propensity
  alpha_z <- round(alpha / 2)
  alpha_v <- alpha - alpha_z
  s <- sort(rep(1:4, n / 4))
  
  # parallelize
  registerDoParallel(cores = 35)
  #registerDoParallel(cores = 14)
  
  
  results <- foreach(sim_num = 1:n_sim) %dopar% {
    # for(sim_num in c(1:n_sim)) {
    x <- matrix(rnorm(n * d), n, d)
    v <- x[, (q + 1):d]
    mu0 <- sigmoid(as.numeric(x %*% rep(c(1, 0, 0.5, 0), c(zeta, q - zeta, gamma, p - gamma))))
    nu <- sigmoid(as.numeric(x %*% rep(c(0, 0.5, 0), c(q, gamma, p - gamma))))
    prop <- sigmoid(as.numeric(x %*% rep(c(1, 0, 1, 0), c(alpha_z, q - alpha_z, alpha_v, p - alpha_v))) / sqrt(2 * alpha))
    a <- rbinom(n, 1, prop)
    y0 <- rbinom(n, 1, mu0)
    
    # qplot(mu0[((s == 2) & (a == 0))])
    
    # stage 1
    mu_lasso <- cv.glmnet(x[((s == 2) & (a == 0)), ], y0[((s == 2) & (a == 0))], family = "binomial")
    muhat <- as.numeric(predict(mu_lasso, newx = x, type = "response", s = "lambda.min"))
    
    prop_lasso <- cv.glmnet(x[s == 1, ], a[s == 1], family = "binomial")
    prophat <- as.numeric(predict(prop_lasso, newx = x, type = "response", s = "lambda.min"))
    
    bchat <- (1 - a) * (y0 - muhat) / (1 - prophat) + muhat
    bc_true <- (1 - a) * (y0 - mu0) / (1 - prop) + mu0
    bc_true_prop <- (1 - a) * (y0 - muhat) / (1 - prop) + muhat
    bc_true_mu <- (1 - a) * (y0 - mu0) / (1 - prophat) + mu0
    
    # stage 2
    conf_lasso <- cv.glmnet(v[((s == 3) & (a == 0)), ], y0[((s == 3) & (a == 0))], family = "binomial")
    conf <- predict(conf_lasso, newx = v, s = "lambda.min", type = "response")
    conf1se <- predict(conf_lasso, newx = v, type = "response")
    
    pl_lasso <- cv.glmnet(v[s == 3, ], muhat[s == 3])
    pl <- predict(pl_lasso, newx = v, s = "lambda.min")
    pl1se <- predict(pl_lasso, newx = v)
    
    bc_lasso <- cv.glmnet(v[s == 3, ], bchat[s == 3])
    bc <- predict(bc_lasso, newx = v, s = "lambda.min")
    
    bct_lasso <- cv.glmnet(v[s == 3, ], bc_true[s == 3])
    bct <- predict(bct_lasso, newx = v, s = "lambda.min")
    
    bctp_lasso <- cv.glmnet(v[s == 3, ], bc_true_prop[s == 3])
    bct_prop <- predict(bctp_lasso, newx = v, s = "lambda.min")
    
    bctm_lasso <- cv.glmnet(v[s == 3, ], bc_true_mu[s == 3])
    bct_mu <- predict(bctm_lasso, newx = v, s = "lambda.min")
    
    results <- rbind(results, tibble(
      "mse" = c(
        mean((conf - nu)[s == 4]^2),
        mean((pl - nu)[s == 4]^2),
        mean((bc - nu)[s == 4]^2),
        mean((bct - nu)[s == 4]^2),
        mean((bct_prop - nu)[s == 4]^2),
        mean((bct_mu - nu)[s == 4]^2),
        mean((conf1se - nu)[s == 4]^2),
        mean((pl1se - nu)[s == 4]^2),
        mean((mu0 - nu)[s == 4]^2)
      ),
      "method" = c("conf", "pl", "bc", "bct", "bc_true_prop", "bc_true_mu", "conf1se", "pl1se", "regression_diff"),
      "sim" = sim_num,
      "prop_nnzero" = nnzero(coef(prop_lasso, s = prop_lasso$lambda.1se)),
      "mu_nnzero" = nnzero(coef(mu_lasso, s = mu_lasso$lambda.1se)),
      "k" = k
    ))
  }
  saveRDS(tibble(
    "dim" = d,
    "k" = k,
    "n_in_each_fold" = n / 4,
    "q" = q,
    "dim_z" = q,
    "p" = p,
    "zeta" = zeta,
    "gamma" = gamma,
    "beta" = beta,
    "alpha_v" = alpha_v,
    "alpha_z" = alpha_z,
    "alpha" = alpha
  ), glue::glue(results_folder, "k{k}", "parameters.Rds"))
  
  saveRDS(bind_rows(results), glue::glue(results_folder, "k{k}", "results.Rds"))
}
task_time <- difftime(Sys.time(), start_time)
print(task_time)
