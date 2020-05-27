library(tidyverse)
library(glmnet)
library(doParallel)
source("utils.R")

results_folder <- "results/highdim/regression/prop_widen/misspec/"
start_time <- Sys.time()
#set.seed(3)
set.seed(100)
results <- tibble()
n <- 4 * 1000
n_sim <- 500
d <- 500
q <- 100 #20 # dimension of hidden confounder z
p <- d - q # dimension of v
zeta <- 20 # number of non-zero predictors in z
gamma <- 24 # number of non-zero predictors in v
beta <- gamma + zeta
alpha_z <- 20
alpha_v <- gamma#25 #updated but not run
alpha <- alpha_z + alpha_v
s <- sort(rep(1:4, n / 4))

# parallelize
registerDoParallel(cores = 48)

results <- foreach (sim_num = 1:n_sim) %dopar% {
  v_first_order <- matrix(rnorm(n * p/2), n, p/2)
  v_second_order <- v_first_order^2
  z <- matrix(rnorm(n * q), n, q)
  x <- cbind(z, v_first_order, v_second_order)
  
  # mu is linear in z and quadratic/linear in v
  mu0 <- as.numeric(z %*% rep(c(1, 0), c(zeta, q - zeta)) +
                      v_second_order %*% c(rep(c(1, -1), gamma/4), rep(0,p/2 - gamma/2))+
                      v_first_order %*% rep(c(1, 0), c(gamma/2, p/2 - gamma/2)))
  # nu is quadratic/linear in v
  nu <- as.numeric(    v_second_order %*% c(rep(c(1, -1), gamma/4), rep(0,p/2 - gamma/2))+
                         v_first_order %*% rep(c(1, 0), c(gamma/2, p/2 - gamma/2)))
  # prop is linear in v and z
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
  conf_lasso <- cv.glmnet(v_first_order[((s == 3) & (a == 0)), ], y0[((s == 3) & (a == 0))])
  conf <- predict(conf_lasso, newx = v_first_order, s = "lambda.min")
  
  pl_lasso <- cv.glmnet(v_first_order[s == 3, ], muhat[s == 3])
  pl <- predict(pl_lasso, newx = v_first_order, s = "lambda.min")
  
  bc_lasso <- cv.glmnet(v_first_order[s == 3, ], bchat[s == 3])
  bc <- predict(bc_lasso, newx = v_first_order, s = "lambda.min")
  
  bct_lasso <- cv.glmnet(v_first_order[s == 3, ], bc_true[s == 3])
  bct <- predict(bct_lasso, newx = v_first_order, s = "lambda.min")
  
  tibble(
    "mse" = c(
      mean((conf - nu)[s == 4]^2),
      mean((pl - nu)[s == 4]^2),
      mean((bc - nu)[s == 4]^2),
      mean((bct - nu)[s == 4]^2)
    ),
    "method" = c("conf", "pl", "bc", "bct"),
    "sim" = sim_num
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
  "alpha" = alpha), glue::glue(results_folder, "parameters.Rds"))

saveRDS(bind_rows(results), glue::glue(results_folder, "results.Rds"))

task_time <- difftime(Sys.time(), start_time)
print(task_time)