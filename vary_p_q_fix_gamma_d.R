library(tidyverse)
library(glmnet)
library(doParallel)
library(ranger)
source("utils.R")
source("learning_functions.R")

results_folder <- "results/paper/rf/vary_p_q/"
start_time <- Sys.time()
set.seed(100)

registerDoParallel(cores = 48)

for (m in  c(0, 0.25)) {
  for (p in seq(50, 400, 50)) {
  #for (p in seq(100, 400, 100)) {
    results <- tibble()
    n <- 4 * 1000
    n_sim <- 500
    d <- 500
    q <- d - p
    zeta <- 20 # number of non-zero predictors in z
    gamma <- 25 # number of non-zero predictors in v
    alpha_z <- zeta 
    alpha_v <- gamma
    alpha <- alpha_z + alpha_v
    s <- sort(rep(1:4, n / 4))
    cf <- gamma/(m*zeta + gamma)# coefficient on sparse predictors, formerly 
    results <- learn_rf(n_sim = n_sim, n =  n, d = d, p = p, q = q, zeta = zeta, gamma = gamma, s= s) 
    saveRDS(tibble(
      "dim" = d,
      "n_in_each_fold" = n / 4,
      "q" = q,
      "p" = p,
      "m" = m,
      "zeta" = zeta,
      "gamma" = gamma,
      "alpha_v" = alpha_v,
      "alpha_z" = alpha_z,
      "alpha" = alpha
    ), glue::glue(results_folder, "p{p}_mhash{(m+2)*100}", "parameters.Rds"))
    
    saveRDS(bind_rows(results), glue::glue(results_folder, "p{p}_mhash{(m+2)*100}", "results.Rds"))
  }
}
task_time <- difftime(Sys.time(), start_time)
print(task_time)
