library(tidyverse)
library(glmnet)
library(doParallel)
source("utils.R")

results_folder <- "results/highdim/regression/eval/p300/"
start_time <- Sys.time()
# set.seed(3)
set.seed(100)

registerDoParallel(cores = 48)

results <- tibble()
n <- 4 * 1000
n_sim <- 500
d <- 500
p <- 400
q <- d - p
zeta <- 20 # number of non-zero predictors in z
gamma <- 25 # number of non-zero predictors in v
beta <- gamma + zeta
alpha_z <- 20
alpha_v <- 25
alpha <- alpha_z + alpha_v
s <- sort(rep(1:4, n / 4))



results <- foreach(sim_num = 1:n_sim) %dopar% {
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
  bc_true <- (1 - a) * (y0 - mu0) / (1 - prop) + mu0
  
  # stage 2
  conf_lasso <- cv.glmnet(v[((s == 3) & (a == 0)), ], y0[((s == 3) & (a == 0))])
  conf <- predict(conf_lasso, newx = v, s = "lambda.min")
  
  pl_lasso <- cv.glmnet(v[s == 3, ], muhat[s == 3])
  pl <- predict(pl_lasso, newx = v, s = "lambda.min")
  
  bc_lasso <- cv.glmnet(v[s == 3, ], bchat[s == 3])
  bc <- predict(bc_lasso, newx = v, s = "lambda.min")
  
  # estimate MSE's
  eta_conf_lasso <- cv.glmnet(v[((s == 1) & (a == 0)), ], (y0[((s == 1) & (a == 0))] - conf[((s == 1) & (a == 0))])^2)
  eta_conf <- as.numeric(predict(eta_conf_lasso, newx = v))
  
  eta_pl_lasso <- cv.glmnet(v[((s == 1) & (a == 0)), ], (y0[((s == 1) & (a == 0))] - pl[((s == 1) & (a == 0))])^2)
  eta_pl <- as.numeric(predict(eta_pl_lasso, newx = v))
  
  eta_bc_lasso <- cv.glmnet(v[((s == 1) & (a == 0)), ], (y0[((s == 1) & (a == 0))] - bc[((s == 1) & (a == 0))])^2)
  eta_bc <- as.numeric(predict(eta_bc_lasso, newx = v))
  
  
  phi_conf <- (1 - a) / (1 - prophat) * ((y0 - conf)^2 - eta_conf) + eta_conf
  phi_pl <- (1 - a) / (1 - prophat) * ((y0 - pl)^2 - eta_pl) + eta_pl
  phi_bc <- (1 - a) / (1 - prophat) * ((y0 - bc)^2 - eta_bc) + eta_bc
  
  tibble(
    "mse" = c(
      mean((conf - nu)[s == 4]^2),
      mean((pl - nu)[s == 4]^2),
      mean((bc - nu)[s == 4]^2),
      mean((conf - y0)[s == 4]^2),
      mean((pl - y0)[s == 4]^2),
      mean((bc - y0)[s == 4]^2),
      mean(phi_conf[s == 4]),
      mean(phi_pl[s == 4]),
      mean(phi_bc[s == 4]),
      mean(phi_conf[s == 4]) - 1.96 * sqrt(var(phi_conf[s == 4]) / length(phi_conf[s == 4])),
      mean(phi_pl[s == 4]) - 1.96 * sqrt(var(phi_pl[s == 4]) / length(phi_pl[s == 4])),
      mean(phi_bc[s == 4]) - 1.96 * sqrt(var(phi_bc[s == 4]) / length(phi_bc[s == 4])),
      mean(phi_conf[s == 4]) + 1.96 * sqrt(var(phi_conf[s == 4]) / length(phi_conf[s == 4])),
      mean(phi_pl[s == 4]) + 1.96 * sqrt(var(phi_pl[s == 4]) / length(phi_pl[s == 4])),
      mean(phi_bc[s == 4]) + 1.96 * sqrt(var(phi_bc[s == 4]) / length(phi_bc[s == 4]))
    ),
    "method" = rep(c("conf", "pl", "bc"), 5),
    "eval" = c(
      rep("true_reg_mse", 3),
      rep("true_pred_mse", 3),
      rep("dr_observational", 3),
      rep("dr_observational_low", 3),
      rep("dr_observational_high", 3)
    ),
    "sim" = sim_num,
    "prop_nnzero" = nnzero(coef(prop_lasso, s = prop_lasso$lambda.1se)),
    "mu_nnzero" = nnzero(coef(mu_lasso, s = mu_lasso$lambda.1se)),
    "p" = p,
    "zeta" = zeta,
    "gamma" = gamma,
    "beta" = beta,
    "alpha_v" = alpha_v,
    "alpha_z" = alpha_z,
    "alpha" = alpha
  )
  
  # tibble(
  #     prop = prop,
  #     mu0 = mu0,
  #     nu = nu,
  #     bc = bc,
  #     bct = bct,
  #     pl = pl,
  #     conf = conf,
  #     sim = sim_num,
  #     s = s
  #   ) %>%
  #     filter(s == 4) %>%
  #     select(-s)
}
saveRDS(tibble(
  "dim" = d,
  "n_in_each_fold" = n / 4,
  "q" = q,
  "p" = p,
  "zeta" = zeta,
  "gamma" = gamma,
  "beta" = beta,
  "alpha_v" = alpha_v,
  "alpha_z" = alpha_z,
  "alpha" = alpha
), glue::glue(results_folder, "parameters.Rds"))

saveRDS(bind_rows(results), glue::glue(results_folder, "results.Rds"))


task_time <- difftime(Sys.time(), start_time)
print(task_time)