library(tidyverse)
library(glmnet)
library(doParallel)
library(ranger)
source("utils.R")
source("learning_functions.R")
# This script varies zeta and beta, holding gamma, alpha, p, d, q fixed.
results_folder <- "results/paper/no_split/"
start_time <- Sys.time()
set.seed(100)

registerDoParallel(cores = 48)

for (zeta in seq(0, 50, 5)) {
  n <- 4 * 1000
  n_sim <- 500
  d <- 500
  p <- 400
  q <- d - p
  gamma <- 25 # number of non-zero predictors in v
  s <- rep(c(1,2), c(3000, 1000))
  
  results <- learn_no_split(n_sim = n_sim, n = n, d = d, p = p, q = q, zeta = zeta, gamma = gamma, s = s) 
  saveRDS(tibble(    
    "dim" = d,
    "n_in_each_fold" = 3000,
    "q" = q,
    "p" = p,
    "zeta" = zeta,
    "gamma" = gamma), glue::glue(results_folder, "zeta{zeta}", "parameters.Rds"))
  
  saveRDS(bind_rows(results), glue::glue(results_folder,  "zeta{zeta}","results.Rds"))
}

task_time <- difftime(Sys.time(), start_time)
print(task_time)

