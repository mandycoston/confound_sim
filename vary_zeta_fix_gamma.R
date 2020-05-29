library(tidyverse)
library(glmnet)
library(doParallel)
library(ranger)
source("utils.R")

# This script varies zeta and beta, holding gamma, alpha, p, d, q fixed.
#results_folder <- "results/highdim/regression/prop_widen/cor/varyparams/vary_zeta_fix_gamma_rho25/"
results_folder <- "results/paper/vary_zeta_fix_gamma/"
start_time <- Sys.time()
set.seed(100)

registerDoParallel(cores = 48)

for (m in  c(0)) {
for (zeta in seq(0, 50, 5)) {
    results <- tibble()
    n <- 4 * 1000
    n_sim <- 500
    d <- 500
    p <- 400
    q <- d - p
    gamma <- 25 # number of non-zero predictors in v
    beta <- gamma + zeta
    alpha_z <- zeta#20 # updated alphaz = zeta
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
      # create DF for random forests ranger
      colnames(x) <- c(sapply(c(1:q), function(x) {
        glue::glue("Z", x)
      }), sapply(c(1:p), function(x) {
        glue::glue("V", x)
      }))
      colnames(v) <- sapply(c(1:p), function(x) {
        glue::glue("V", x)
      })
      colnames(z) <- sapply(c(1:q), function(x) {
        glue::glue("Z", x)
      })
      
      df <- as_tibble(x) %>% mutate(
        prop = prop,
        a = factor(a, levels = c(0, 1)),
        y0 = y0,
        nu = nu,
        mu0 = mu0,
        s = s
      )
      
      prop_rf <- ranger("a ~.", data = select(filter(df, s == 1), colnames(x), a), probability = TRUE, num.trees = 1000)
      prop_rf_hat <- as.numeric(predict(prop_rf, data = x, type = "response")$predictions[, 2])
      
      mu_rf <- ranger("y0 ~ .", data = select(filter(df, s == 2, a == 0), y0, colnames(x)), num.trees = 1000)
      mu_rf_hat <- as.numeric(predict(mu_rf, data = x, type = "response")$predictions)

      # stage 1
      prop_lasso <- cv.glmnet(x[s == 1, ], a[s == 1], family = "binomial")
      prophat <- as.numeric(predict(prop_lasso, newx = x, type = "response"))

      mu_lasso <- cv.glmnet(x[((s == 2) & (a == 0)), ], y0[((s == 2) & (a == 0))])
      muhat <- as.numeric(predict(mu_lasso, newx = x))

      bchat <- (1 - a) * (y0 - muhat) / (1 - prophat) + muhat
      bc_rf_pseudo <- (1 - a) * (y0 - mu_rf_hat) / (1 - prop_rf_hat) + mu0
      
      df %>%
        dplyr::mutate(
          bchat_rf_1st = bchat_rf_1st,
          mu_rf_hat = mu_rf_hat
        ) -> df
      
      
      conf_rf <- ranger(y0 ~ ., data = select(filter(df, s == 3, a == 0), y0, colnames(v)), num.trees = 1000)
      conf_rf_hat <- as.numeric(predict(conf_rf, data = v, type = "response")$predictions)
      pl_rf <- ranger(mu_rf_hat ~ ., data = select(filter(df, s == 3), mu_rf_hat, colnames(v)), num.trees = 1000)
      pl_rf_hat <- as.numeric(predict(pl_rf, data = v, type = "response")$predictions)
      bc_rf <- ranger(bc_rf_pseudo ~ ., data = select(filter(df, s == 3), bchat_rf_1st, colnames(v)), num.trees = 1000)
      bc_rf_hat <- as.numeric(predict(bc_rf, data = v, type = "response")$predictions)
      

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
          mean((bc - nu)[s == 4]^2),
          mean((conf_rf_hat - nu)[s == 4]^2),
          mean((pl_rf_hat - nu)[s == 4]^2),
          mean((bc_rf_hat - nu)[s == 4]^2)
        ),
        "method" = c("conf", "pl", "bc", "conf", "pl", "bc"),
        "algorithm" = rep(c("LASSO", "RF"), c(3, 3)),
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
    ), glue::glue(results_folder, "zeta{zeta}_mhash{(m+2)*100}", "parameters.Rds"))

    saveRDS(bind_rows(results), glue::glue(results_folder, "zeta{zeta}_mhash{(m+2)*100}", "results.Rds"))
}
}
task_time <- difftime(Sys.time(), start_time)
print(task_time)
