
# random forests where tcr is trained on all three folds
learn_rf_tcr_more_train <- function(n_sim = 500, n = 4*1000, d = 500, p, q, zeta, gamma, s) {
  alpha_z <- zeta 
  alpha_v <- gamma
  alpha <- alpha_z + alpha_v
  results <- foreach(sim_num = 1:n_sim) %dopar% {
    v <- matrix(rnorm(n * p), n, p)
    if(p >=q) {
      means <- as.vector(v[, 1:q])
    }
    if(p < q) {
      means <- c(as.vector(v), rep(0, q-p))
    }
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
    
    # stage 1
    prop_rf <- ranger("a ~.", data = select(filter(df, s == 1), colnames(x), a), probability = TRUE, num.trees = 1000)
    prop_rf_hat <- as.numeric(predict(prop_rf, data = x, type = "response")$predictions[, 2])
    
    mu_rf <- ranger("y0 ~ .", data = select(filter(df, s == 2, a == 0), y0, colnames(x)), num.trees = 1000)
    mu_rf_hat <- as.numeric(predict(mu_rf, data = x, type = "response")$predictions)
    
    bc_rf_pseudo <- (1 - a) * (y0 - mu_rf_hat) / (1 - prop_rf_hat) + mu_rf_hat
    
    df %>%
      dplyr::mutate(
        bc_rf_pseudo = bc_rf_pseudo,
        mu_rf_hat = mu_rf_hat
      ) -> df
    
    
    # stage 2
    conf_rf <- ranger(y0 ~ ., data = select(filter(df, s <= 3, a == 0), y0, colnames(v)), num.trees = 1000)
    conf_rf_hat <- as.numeric(predict(conf_rf, data = v, type = "response")$predictions)
    
    pl_rf <- ranger(mu_rf_hat ~ ., data = select(filter(df, s == 3), mu_rf_hat, colnames(v)), num.trees = 1000)
    pl_rf_hat <- as.numeric(predict(pl_rf, data = v, type = "response")$predictions)
    
    bc_rf <- ranger(bc_rf_pseudo ~ ., data = select(filter(df, s == 3), bc_rf_pseudo, colnames(v)), num.trees = 1000)
    bc_rf_hat <- as.numeric(predict(bc_rf, data = v, type = "response")$predictions)
    
    tibble(
      "mse" = c(
        mean((conf_rf_hat - nu)[s == 4]^2),
        mean((pl_rf_hat - nu)[s == 4]^2),
        mean((bc_rf_hat - nu)[s == 4]^2)
      ),
      "method" = c("conf", "pl", "bc"),
      "algorithm" = "RF",
      "sim" = sim_num,
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
  return(results)
}

# random forests only
learn_rf <- function(n_sim = 500, n = 4*1000, d = 500, p, q, zeta, gamma, s) {
  alpha_z <- zeta 
  alpha_v <- gamma
  alpha <- alpha_z + alpha_v
  results <- foreach(sim_num = 1:n_sim) %dopar% {
    v <- matrix(rnorm(n * p), n, p)
    if(p >=q) {
      means <- as.vector(v[, 1:q])
    }
    if(p < q) {
      means <- c(as.vector(v), rep(0, q-p))
    }
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
    
    # stage 1
    prop_rf <- ranger("a ~.", data = select(filter(df, s == 1), colnames(x), a), probability = TRUE, num.trees = 1000)
    prop_rf_hat <- as.numeric(predict(prop_rf, data = x, type = "response")$predictions[, 2])
    
    mu_rf <- ranger("y0 ~ .", data = select(filter(df, s == 2, a == 0), y0, colnames(x)), num.trees = 1000)
    mu_rf_hat <- as.numeric(predict(mu_rf, data = x, type = "response")$predictions)
    
    bc_rf_pseudo <- (1 - a) * (y0 - mu_rf_hat) / (1 - prop_rf_hat) + mu_rf_hat
    
    df %>%
      dplyr::mutate(
        bc_rf_pseudo = bc_rf_pseudo,
        mu_rf_hat = mu_rf_hat
      ) -> df
    
    
    # stage 2
    conf_rf <- ranger(y0 ~ ., data = select(filter(df, s == 3, a == 0), y0, colnames(v)), num.trees = 1000)
    conf_rf_hat <- as.numeric(predict(conf_rf, data = v, type = "response")$predictions)
    
    pl_rf <- ranger(mu_rf_hat ~ ., data = select(filter(df, s == 3), mu_rf_hat, colnames(v)), num.trees = 1000)
    pl_rf_hat <- as.numeric(predict(pl_rf, data = v, type = "response")$predictions)
    
    bc_rf <- ranger(bc_rf_pseudo ~ ., data = select(filter(df, s == 3), bc_rf_pseudo, colnames(v)), num.trees = 1000)
    bc_rf_hat <- as.numeric(predict(bc_rf, data = v, type = "response")$predictions)
    
    tibble(
      "mse" = c(
        mean((conf_rf_hat - nu)[s == 4]^2),
        mean((pl_rf_hat - nu)[s == 4]^2),
        mean((bc_rf_hat - nu)[s == 4]^2)
      ),
      "method" = c("conf", "pl", "bc"),
      "algorithm" = "RF",
      "sim" = sim_num,
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
  return(results)
}


# no sample splitting
learn_no_split <- function(n_sim = 500, n = 4*1000, d = 500, p, q, zeta, gamma, s) {
  results <- foreach(sim_num = 1:n_sim) %dopar% {
    x <- matrix(rnorm(n * d), n, d)
    v <- x[, (q + 1):d]
    z <- x[, 1:q]
    mu0 <- as.numeric(x %*% rep(c(1, 0, 1, 0), c(zeta, q - zeta, gamma, p - gamma)))
    nu <- as.numeric(x %*% rep(c(0, 1, 0), c(q, gamma, p - gamma)))
    alpha_z <- zeta
    alpha_v <- gamma
    alpha <- alpha_z + alpha_v
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
    

    # stage 1
    prop_lasso <- cv.glmnet(x[s == 1, ], a[s == 1], family = "binomial")
    prophat <- as.numeric(predict(prop_lasso, newx = x, type = "response"))
    
    prop_rf <- ranger("a ~.", data = select(filter(df, s == 1), colnames(x), a), probability = TRUE, num.trees = 1000)
    prop_rf_hat <- as.numeric(predict(prop_rf, data = x, type = "response")$predictions[, 2])

    mu_lasso <- cv.glmnet(x[((s == 1) & (a == 0)), ], y0[((s == 1) & (a == 0))])
    muhat <- as.numeric(predict(mu_lasso, newx = x))
    
    mu_rf <- ranger("y0 ~ .", data = select(filter(df, s == 1, a == 0), y0, colnames(x)), num.trees = 1000)
    mu_rf_hat <- as.numeric(predict(mu_rf, data = x, type = "response")$predictions)


    bchat <- (1 - a) * (y0 - muhat) / (1 - prophat) + muhat
    bc_rf_pseudo <- (1 - a) * (y0 - mu_rf_hat) / (1 - prop_rf_hat) + mu0
    
    df %>%
      dplyr::mutate(
        bc_rf_pseudo = bc_rf_pseudo,
        mu_rf_hat = mu_rf_hat
      ) -> df
    

    # stage 2
    conf_lasso <- cv.glmnet(v[((s == 1) & (a == 0)), ], y0[((s == 1) & (a == 0))])
    conf <- predict(conf_lasso, newx = v, s = "lambda.min")
    conf_rf <- ranger(y0 ~ ., data = select(filter(df, s == 1, a == 0), y0, colnames(v)), num.trees = 1000)
    conf_rf_hat <- as.numeric(predict(conf_rf, data = v, type = "response")$predictions)
    
    if(var(muhat[s==1]) > 0) {
      pl_lasso <- cv.glmnet(v[s == 1, ], muhat[s == 1])
      pl <- predict(pl_lasso, newx = v, s = "lambda.min")
    }
    
    pl_rf <- ranger(mu_rf_hat ~ ., data = select(filter(df, s == 1), mu_rf_hat, colnames(v)), num.trees = 1000)
    pl_rf_hat <- as.numeric(predict(pl_rf, data = v, type = "response")$predictions)
    
    if(var(muhat[s==1]) == 0) {
      saveRDS(tibble(m = m, 
                     zeta = zeta,
                     sim_num = sim_num), glue::glue(results_folder, "m{m}_sim{sim_num}constant_mu.Rds"))
      pl <- muhat
    }
    
    bc_lasso <- cv.glmnet(v[s == 1, ], bchat[s == 1])
    bc <- predict(bc_lasso, newx = v, s = "lambda.min")
    
    bc_rf <- ranger(bc_rf_pseudo ~ ., data = select(filter(df, s == 1), bc_rf_pseudo, colnames(v)), num.trees = 1000)
    bc_rf_hat <- as.numeric(predict(bc_rf, data = v, type = "response")$predictions)
    
    tibble(
      "mse" = c(
        mean((conf - nu)[s == 2]^2),
        mean((pl - nu)[s == 2]^2),
        mean((bc - nu)[s == 2]^2),
        mean((conf_rf_hat - nu)[s == 2]^2),
        mean((pl_rf_hat - nu)[s == 2]^2),
        mean((bc_rf_hat - nu)[s == 2]^2)
      ),
      "method" = c("conf", "pl", "bc", "conf", "pl", "bc"),
      "algorithm" = rep(c("LASSO", "RF"), c(3, 3)),
      "sim" = sim_num,
      "prop_nnzero" = nnzero(coef(prop_lasso, s = prop_lasso$lambda.1se)),
      "mu_nnzero" = nnzero(coef(mu_lasso, s = mu_lasso$lambda.1se)),
      "p" = p,
      "zeta" = zeta,
      "gamma" = gamma
    )}
}
