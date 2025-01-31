library(tidyverse)
library(glmnet)
library(doParallel)
source("utils.R")

# This is a script baesd on high_dim.Rmd and runs a simulation based on Edward's setup.
results_folder <- "results/highdim/regression/prop_widen/varyparams/varyalphaz/"
start_time <- Sys.time()
#set.seed(3)
set.seed(100)

#registerDoParallel(cores = 48)
registerDoParallel(cores = 4)


for (alpha_z in seq(0, 45, 5)) {
  results <- tibble()
  n <- 4 * 1000
  n_sim <- 500
  d <- 500
  p <- 400
  q <- d - p
  beta <- 45
  zeta <- 20
  gamma <- beta - zeta # number of non-zero predictors in v
  alpha <- 45
  alpha_v <- alpha - alpha_z
  s <- sort(rep(1:4, n / 4))
  
  
  
  results <- foreach (sim_num = 1:n_sim) %dopar% {
    x <- matrix(rnorm(n * d), n, d)
    v <- x[, (q + 1):d]
    mu0 <- as.numeric(x %*% rep(c(1, 0, 1, 0), c(zeta, q - zeta, gamma, p - gamma)))
    nu <- as.numeric(x %*% rep(c(0, 1, 0), c(q, gamma, p - gamma)))
    prop <- sigmoid(as.numeric(x %*% rep(c(1, 0, 1, 0), c(alpha_z, q - alpha_z, alpha_v, p - alpha_v))) / sqrt(alpha))
    
    a <- rbinom(n, 1, prop)
    y0 <- mu0 + rnorm(n, sd = sqrt(sum(mu0^2) / (n * 2)))
    
    # stage 1
    prop_lasso <- cv.glmnet(x[s == 1, ], a[s == 1], family = "binomial")
    prophat <- as.numeric(predict(prop_lasso, newx = x, type = "response"))
    
    mu_lasso <- cv.glmnet(x[((s == 2) & (a == 0)), ], y0[((s == 2) & (a == 0))])
    muhat <- as.numeric(predict(mu_lasso, newx = x))
    
    bchat <- (1 - a) * (y0 - muhat) / (1 - prophat) + muhat
    
    # stage 2
    conf_lasso <- cv.glmnet(v[((s == 3) & (a == 0)), ], y0[((s == 3) & (a == 0))])
    conf <- predict(conf_lasso, newx = v, s = "lambda.min")
    
    pl_lasso <- cv.glmnet(v[s == 3, ], muhat[s == 3])
    pl <- predict(pl_lasso, newx = v, s = "lambda.min")
    
    bc_lasso <- cv.glmnet(v[s == 3, ], bchat[s == 3])
    bc <- predict(bc_lasso, newx = v, s = "lambda.min")
    
    
    tibble(
      "mse" = c(
        mean((conf - nu)[s == 4]^2),
        mean((pl - nu)[s == 4]^2),
        mean((bc - nu)[s == 4]^2)
      ),
      "method" = c("conf", "pl", "bc"),
      "sim" = sim_num,
      "prop_nnzero" = nnzero(coef(prop_lasso, s=prop_lasso$lambda.1se)),
      "mu_nnzero" = nnzero(coef(mu_lasso, s=mu_lasso$lambda.1se)),
      "p" = p,
      "zeta" = zeta,
      "gamma" = gamma,
      "beta" = beta,
      "alpha_v" = alpha_v,
      "alpha_z" = alpha_z,
      "alpha" = alpha
    )
  }
  saveRDS(tibble(    
    "dim" = d,
    "n_in_each_fold" = n/4,
    "q" = q,
    "p" = p,
    "zeta" = zeta,
    "gamma" = gamma,
    "beta" = beta,
    "alpha_v" = alpha_v,
    "alpha_z" = alpha_z,
    "alpha" = alpha), glue::glue(results_folder, "alpha_z_{alpha_z}", "parameters.Rds"))
  
  saveRDS(bind_rows(results), glue::glue(results_folder,  "alpha_z_{alpha_z}","results.Rds"))
}

task_time <- difftime(Sys.time(), start_time)
print(task_time)

