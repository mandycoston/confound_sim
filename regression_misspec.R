library(tidyverse)
library(glmnet)
library(doParallel)
source("utils.R")
source("learning_functions.R")

results_folder <- "results/paper/misspec/"
start_time <- Sys.time()
#set.seed(3)
set.seed(100)
results <- tibble()
n <- 2 * 1000
n_sim <- 300
d <- 500
q <- 100 #20 # dimension of hidden confounder z
p <- d - q # dimension of v
zeta <- 20 # number of non-zero predictors in z
gamma <- 24 # number of non-zero predictors in v
beta <- gamma + zeta
alpha_z <- zeta
alpha_v <- gamma #25 #updated but not run
alpha <- alpha_z + alpha_v
s <- rep(c(3,4), c(1000, 1000))

# parallelize
registerDoParallel(cores = 48)

results <- second_order_no_split(n_sim = n_sim, n =  n, d = d, p = p, q = q, zeta = zeta, gamma = gamma, s= s) 
saveRDS(tibble(    
  "dim" = d,
  "n_in_each_fold" = n/2,
  "q" = q,
  "p" = p,
  "zeta" = zeta,
  "gamma" = gamma,
  "beta" = beta,
  "alpha" = alpha), glue::glue(results_folder, "parameters.Rds"))

saveRDS(bind_rows(results), glue::glue(results_folder, "results.Rds"))

task_time <- difftime(Sys.time(), start_time)
print(task_time)