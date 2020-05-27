library(tidyverse)
library(glmnet)
library(doParallel)
source("utils.R")


# This script performs high dimensional sparse regression when varying p = 16*gamma
# and q = d - p and q = 16*zeta
results_folder <- "results/paper/vary_p_q_gamma_zeta/"
start_time <- Sys.time()
set.seed(100)

registerDoParallel(cores = 48)

for (m in c(0, 0.25)) {
  for (p in seq(50, 450, 50)) {
    # for (p in seq(100, 400, 100)) {
    results <- tibble()
    n <- 4 * 1000
    n_sim <- 500
    d <- 500
    q <- d - p
    zeta <- round(q/16) #20 # number of non-zero predictors in z
    gamma <- round(p/16)#25 # number of non-zero predictors in v
    beta <- gamma + zeta
    alpha_z <- zeta 
    alpha_v <- gamma
    alpha <- alpha_z + alpha_v
    s <- sort(rep(1:4, n / 4))
    cf <- gamma/(m*zeta + gamma)# coefficient on sparse predictors, formerly 1
    
    
    
    results <- foreach(sim_num = 1:n_sim) %dopar% {
      v <- matrix(rnorm(n * p), n, p)
      means <- as.vector(v[, 1:q])
      z <- matrix(rnorm(n = n * q, mean = m * means, sd = sqrt(1-m^2)), n, q) #note the updated variance 
      # cor(v[,99], z[,99])
      x <- cbind(z, v)
      mu0 <- as.numeric(x %*% rep(c(cf, 0, cf, 0), c(zeta, q - zeta, gamma, p - gamma)))
      nu <- as.numeric(x %*% rep(c(0, cf, 0), c(q, gamma, p - gamma))) + m * as.numeric(x %*% rep(c(0, cf, 0), c(q, zeta, p - zeta)))
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
      
      if(var(muhat[s==3]) > 0) {
        pl_lasso <- cv.glmnet(v[s == 3, ], muhat[s == 3])
        pl <- predict(pl_lasso, newx = v, s = "lambda.min")
      }
      
      if(var(muhat[s==3]) == 0) {
        saveRDS(tibble(m = m, 
                       zeta = zeta,
                       sim_num = sim_num), glue::glue(results_folder, "m{m}_sim{sim_num}constant_mu.Rds"))
        pl <- muhat
      }
      
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
        "prop_nnzero" = nnzero(coef(prop_lasso, s = prop_lasso$lambda.1se)),
        "mu_nnzero" = nnzero(coef(mu_lasso, s = mu_lasso$lambda.1se)),
        "p" = p,
        "m" = m,
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
      "n_in_each_fold" = n / 4,
      "q" = q,
      "p" = p,
      "m" = m,
      "zeta" = zeta,
      "gamma" = gamma,
      "beta" = beta,
      "alpha_v" = alpha_v,
      "alpha_z" = alpha_z,
      "alpha" = alpha
    ), glue::glue(results_folder, "p{p}_mhash{(m+2)*100}", "parameters.Rds"))
    
    saveRDS(bind_rows(results), glue::glue(results_folder, "p{p}_mhash{(m+2)*100}", "results.Rds"))
  }
}
task_time <- difftime(Sys.time(), start_time)
print(task_time)
