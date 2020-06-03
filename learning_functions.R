
# random forests where tcr is trained on all three folds
learn_rf_tcr_more_train <- function(n_sim = 500, n = 4 * 1000, d = 500, p, q, zeta, gamma, s, m) {
  alpha_z <- zeta
  alpha_v <- gamma
  alpha <- alpha_z + alpha_v
  results <- foreach(sim_num = 1:n_sim) %dopar% {
    v <- matrix(rnorm(n * p), n, p)
    if (p >= q) {
      means <- as.vector(v[, 1:q])
    }
    if (p < q) {
      means <- c(as.vector(v), rep(0, q - p))
    }
    z <- matrix(rnorm(n = n * q, mean = m * means, sd = sqrt(1 - m^2)), n, q) # note the updated variance
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
learn_rf <- function(n_sim = 500, n = 4 * 1000, d = 500, p, q, zeta, gamma, s, m) {
  alpha_z <- zeta
  alpha_v <- gamma
  alpha <- alpha_z + alpha_v
  results <- foreach(sim_num = 1:n_sim) %dopar% {
    v <- matrix(rnorm(n * p), n, p)
    if (p >= q) {
      means <- as.vector(v[, 1:q])
    }
    if (p < q) {
      means <- c(as.vector(v), rep(0, q - p))
    }
    z <- matrix(rnorm(n = n * q, mean = m * means, sd = sqrt(1 - m^2)), n, q) # note the updated variance
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
learn_no_split <- function(n_sim = 500, n = 2 * 1000, d = 500, p, q, zeta, gamma, s, m) {
  results <- foreach(sim_num = 1:n_sim) %dopar% {
    v <- matrix(rnorm(n * p), n, p)
    if (p >= q) {
      means <- as.vector(v[, 1:q])
    }
    if (p < q) {
      means <- c(as.vector(v), rep(0, q - p))
    }
    z <- matrix(rnorm(n = n * q, mean = m * means, sd = sqrt(1 - m^2)), n, q) # note the updated variance
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
    prop_lasso <- cv.glmnet(x[s <= 3, ], a[s <= 3], family = "binomial")
    prophat <- as.numeric(predict(prop_lasso, newx = x, type = "response"))

    prop_rf <- ranger("a ~.", data = select(filter(df, s <= 3), colnames(x), a), probability = TRUE, num.trees = 1000)
    prop_rf_hat <- as.numeric(predict(prop_rf, data = x, type = "response")$predictions[, 2])

    mu_lasso <- cv.glmnet(x[((s <= 3) & (a == 0)), ], y0[((s <= 3) & (a == 0))])
    muhat <- as.numeric(predict(mu_lasso, newx = x))

    mu_rf <- ranger("y0 ~ .", data = select(filter(df, s <= 3, a == 0), y0, colnames(x)), num.trees = 1000)
    mu_rf_hat <- as.numeric(predict(mu_rf, data = x, type = "response")$predictions)


    bchat <- (1 - a) * (y0 - muhat) / (1 - prophat) + muhat
    bc_rf_pseudo <- (1 - a) * (y0 - mu_rf_hat) / (1 - prop_rf_hat) + mu_rf_hat

    df %>%
      dplyr::mutate(
        bc_rf_pseudo = bc_rf_pseudo,
        mu_rf_hat = mu_rf_hat
      ) -> df


    # stage 2
    conf_lasso <- cv.glmnet(v[((s <= 3) & (a == 0)), ], y0[((s <= 3) & (a == 0))])
    conf <- predict(conf_lasso, newx = v, s = "lambda.min")
    conf_rf <- ranger(y0 ~ ., data = select(filter(df, s <= 3, a == 0), y0, colnames(v)), num.trees = 1000)
    conf_rf_hat <- as.numeric(predict(conf_rf, data = v, type = "response")$predictions)

    if (var(muhat[s <= 3]) > 0) {
      pl_lasso <- cv.glmnet(v[s <= 3, ], muhat[s <= 3])
      pl <- predict(pl_lasso, newx = v, s = "lambda.min")
    }

    pl_rf <- ranger(mu_rf_hat ~ ., data = select(filter(df, s <= 3), mu_rf_hat, colnames(v)), num.trees = 1000)
    pl_rf_hat <- as.numeric(predict(pl_rf, data = v, type = "response")$predictions)

    if (var(muhat[s <= 3]) == 0) {
      saveRDS(tibble(
        m = m,
        zeta = zeta,
        sim_num = sim_num
      ), glue::glue(results_folder, "m{m}_sim{sim_num}constant_mu.Rds"))
      pl <- muhat
    }

    bc_lasso <- cv.glmnet(v[s <= 3, ], bchat[s <= 3])
    bc <- predict(bc_lasso, newx = v, s = "lambda.min")

    bc_rf <- ranger(bc_rf_pseudo ~ ., data = select(filter(df, s <= 3), bc_rf_pseudo, colnames(v)), num.trees = 1000)
    bc_rf_hat <- as.numeric(predict(bc_rf, data = v, type = "response")$predictions)

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
      "gamma" = gamma
    )
  }
}

# standard setup that you ran most your experiments on
# sample splitting
learn_rf_lasso_split <- function(n_sim = 500, n = 4 * 1000, d = 500, p, q, zeta, gamma, s, m) {
  results <- foreach(sim_num = 1:n_sim) %dopar% {
    v <- matrix(rnorm(n * p), n, p)
    if (p >= q) {
      means <- as.vector(v[, 1:q])
    }
    if (p < q) {
      means <- c(as.vector(v), rep(0, q - p))
    }
    z <- matrix(rnorm(n = n * q, mean = m * means, sd = sqrt(1 - m^2)), n, q) # note the updated variance
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
    prop_lasso <- cv.glmnet(x[s == 1, ], a[s == 1], family = "binomial")
    prophat <- as.numeric(predict(prop_lasso, newx = x, type = "response"))

    prop_rf <- ranger("a ~.", data = select(filter(df, s == 1), colnames(x), a), probability = TRUE, num.trees = 1000)
    prop_rf_hat <- as.numeric(predict(prop_rf, data = x, type = "response")$predictions[, 2])

    mu_lasso <- cv.glmnet(x[((s == 2) & (a == 0)), ], y0[((s == 2) & (a == 0))])
    muhat <- as.numeric(predict(mu_lasso, newx = x))

    mu_rf <- ranger("y0 ~ .", data = select(filter(df, s == 2, a == 0), y0, colnames(x)), num.trees = 1000)
    mu_rf_hat <- as.numeric(predict(mu_rf, data = x, type = "response")$predictions)

    bchat <- (1 - a) * (y0 - muhat) / (1 - prophat) + muhat
    bc_rf_pseudo <- (1 - a) * (y0 - mu_rf_hat) / (1 - prop_rf_hat) + mu_rf_hat

    df %>%
      dplyr::mutate(
        bc_rf_pseudo = bc_rf_pseudo,
        mu_rf_hat = mu_rf_hat
      ) -> df


    # stage 2
    conf_lasso <- cv.glmnet(v[((s == 3) & (a == 0)), ], y0[((s == 3) & (a == 0))])
    conf <- predict(conf_lasso, newx = v, s = "lambda.min")

    conf_rf <- ranger(y0 ~ ., data = select(filter(df, s == 3, a == 0), y0, colnames(v)), num.trees = 1000)
    conf_rf_hat <- as.numeric(predict(conf_rf, data = v, type = "response")$predictions)


    if (var(muhat[s == 3]) > 0) {
      pl_lasso <- cv.glmnet(v[s == 3, ], muhat[s == 3])
      pl <- predict(pl_lasso, newx = v, s = "lambda.min")
    }

    pl_rf <- ranger(mu_rf_hat ~ ., data = select(filter(df, s == 3), mu_rf_hat, colnames(v)), num.trees = 1000)
    pl_rf_hat <- as.numeric(predict(pl_rf, data = v, type = "response")$predictions)

    if (var(muhat[s == 3]) == 0) {
      saveRDS(tibble(
        m = m,
        zeta = zeta,
        sim_num = sim_num
      ), glue::glue(results_folder, "m{m}_sim{sim_num}constant_mu.Rds"))
      pl <- muhat
    }

    bc_lasso <- cv.glmnet(v[s == 3, ], bchat[s == 3])
    bc <- predict(bc_lasso, newx = v, s = "lambda.min")

    bc_rf <- ranger(bc_rf_pseudo ~ ., data = select(filter(df, s == 3), bc_rf_pseudo, colnames(v)), num.trees = 1000)
    bc_rf_hat <- as.numeric(predict(bc_rf, data = v, type = "response")$predictions)

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
}


# cross-fitting /using full data efficiently

learn_cross_fit <- function(n_sim = 500, n = 4 * 1000, d = 500, p, q, zeta, gamma, s, m) {
  results <- foreach(sim_num = 1:n_sim) %dopar% {
    v <- matrix(rnorm(n * p), n, p)
    if (p >= q) {
      means <- as.vector(v[, 1:q])
    }
    if (p < q) {
      means <- c(as.vector(v), rep(0, q - p))
    }
    z <- matrix(rnorm(n = n * q, mean = m * means, sd = sqrt(1 - m^2)), n, q) # note the updated variance
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

    # conf/TCR trained on full training
    conf_lasso <- cv.glmnet(v[((s <= 3) & (a == 0)), ], y0[((s <= 3) & (a == 0))])
    conf <- predict(conf_lasso, newx = v, s = "lambda.min")

    conf_rf <- ranger(y0 ~ ., data = select(filter(df, s <= 3, a == 0), y0, colnames(v)), num.trees = 1000)
    conf_rf_hat <- as.numeric(predict(conf_rf, data = v, type = "response")$predictions)

    # for loop over 2 for pl that outputs two predictions
    pl_split <- c(rep(c(1, 2), 3 / 8 * n), rep(0, 1000))
    df$pl_split <- pl_split
    pl_lasso_preds <- list()
    pl_rf_preds <- list()
    for (stage1_ix in c(1, 2)) {
      if (stage1_ix == 1) {
        stage2_ix <- 2
      }
      if (stage1_ix == 2) {
        stage2_ix <- 1
      }
      mu_lasso <- cv.glmnet(x[((pl_split == stage1_ix) & (a == 0)), ], y0[((pl_split == stage1_ix) & (a == 0))])
      muhat <- as.numeric(predict(mu_lasso, newx = x))

      mu_rf <- ranger("y0 ~ .", data = select(filter(df, pl_split == stage1_ix, a == 0), y0, colnames(x)), num.trees = 1000)
      df$mu_rf_hat <- as.numeric(predict(mu_rf, data = x, type = "response")$predictions)

      if (var(muhat[(pl_split == stage2_ix)]) > 0) {
        pl_lasso <- cv.glmnet(v[(pl_split == stage2_ix), ], muhat[(pl_split == stage2_ix)])
        pl <- predict(pl_lasso, newx = v, s = "lambda.min")
      }

      pl_rf <- ranger(mu_rf_hat ~ ., data = select(filter(df, pl_split == stage2_ix), mu_rf_hat, colnames(v)), num.trees = 1000)
      pl_rf_hat <- as.numeric(predict(pl_rf, data = v, type = "response")$predictions)

      if (var(muhat[(pl_split == stage2_ix)]) == 0) {
        pl <- muhat
      }
      pl_lasso_preds[[stage1_ix]] <- pl
      pl_rf_preds[[stage1_ix]] <- pl_rf_hat
    }

    pl <- 1 / 2 * (pl_lasso_preds[[1]] + pl_lasso_preds[[2]])
    pl_rf_hat <- 1 / 2 * (pl_rf_preds[[1]] + pl_rf_preds[[2]])


    ## do the loop for BC
    bc_split <- list(c(1, 2, 3), c(2, 3, 1), c(3, 1, 2))
    bc_lasso_preds <- list()
    bc_rf_preds <- list()
    
    for (ix in c(1, 2, 3)) {
      prop_split <- bc_split[[ix]][1]
      mu_split <- bc_split[[ix]][2]
      pseudo_split <- bc_split[[ix]][3]
      # stage 1
      prop_lasso <- cv.glmnet(x[s == prop_split, ], a[s == prop_split], family = "binomial")
      prophat <- as.numeric(predict(prop_lasso, newx = x, type = "response"))

      prop_rf <- ranger("a ~.", data = select(filter(df, s == prop_split), colnames(x), a), probability = TRUE, num.trees = 1000)
      prop_rf_hat <- as.numeric(predict(prop_rf, data = x, type = "response")$predictions[, 2])

      mu_lasso <- cv.glmnet(x[((s == mu_split) & (a == 0)), ], y0[((s == mu_split) & (a == 0))])
      muhat <- as.numeric(predict(mu_lasso, newx = x))

      mu_rf <- ranger("y0 ~ .", data = select(filter(df, s == mu_split, a == 0), y0, colnames(x)), num.trees = 1000)
      mu_rf_hat <- as.numeric(predict(mu_rf, data = x, type = "response")$predictions)

      bchat <- (1 - a) * (y0 - muhat) / (1 - prophat) + muhat
      bc_rf_pseudo <- (1 - a) * (y0 - mu_rf_hat) / (1 - prop_rf_hat) + mu_rf_hat

      df %>%
        dplyr::mutate(
          bc_rf_pseudo = bc_rf_pseudo
        ) -> df

      # stage 2
      bc_lasso <- cv.glmnet(v[s == pseudo_split, ], bchat[s == pseudo_split])
      bc_lasso_preds[[ix]] <- predict(bc_lasso, newx = v, s = "lambda.min")

      bc_rf <- ranger(bc_rf_pseudo ~ ., data = select(filter(df, s == pseudo_split), bc_rf_pseudo, colnames(v)), num.trees = 1000)
      bc_rf_preds[[ix]] <- as.numeric(predict(bc_rf, data = v, type = "response")$predictions)
    }
    
    bc <- 1/3*(bc_lasso_preds[[1]] + bc_lasso_preds[[2]] + bc_lasso_preds[[3]] )
    bc_rf_hat <- 1/3*(bc_rf_preds[[1]] + bc_rf_preds[[2]] + bc_rf_preds[[3]] )


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
      "p" = p,
      "m" = m,
      "zeta" = zeta,
      "gamma" = gamma,
      "alpha_v" = alpha_v,
      "alpha_z" = alpha_z,
      "alpha" = alpha
    )
  }
}

second_order_no_split <- function(n_sim = 500, n = 4 * 1000, d = 500, p, q, zeta, gamma, s) {
    results <- foreach(sim_num = 1:n_sim) %dopar% {
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
      prop_lasso <- cv.glmnet(x[s <= 3, ], a[s <= 3], family = "binomial")
      prophat <- as.numeric(predict(prop_lasso, newx = x, type = "response"))
      
      mu_lasso <- cv.glmnet(x[((s <= 3) & (a == 0)), ], y0[((s <= 3) & (a == 0))])
      muhat <- as.numeric(predict(mu_lasso, newx = x))
    
      bchat <- (1 - a) * (y0 - muhat) / (1 - prophat) + muhat
         
      
      # stage 2
      conf_lasso <- cv.glmnet(v_first_order[((s <= 3) & (a == 0)), ], y0[((s <= 3) & (a == 0))])
      conf <- predict(conf_lasso, newx = v_first_order, s = "lambda.min")
        
      if (var(muhat[s <= 3]) > 0) {
        pl_lasso <- cv.glmnet(v_first_order[s <= 3, ], muhat[s <= 3])
        pl <- predict(pl_lasso, newx = v_first_order, s = "lambda.min")
      }
      
      if (var(muhat[s <= 3]) == 0) {
        saveRDS(tibble(
          m = m,
          zeta = zeta,
          sim_num = sim_num
        ), glue::glue(results_folder, "m{m}_sim{sim_num}constant_mu.Rds"))
        pl <- muhat
      }
      
      bc_lasso <- cv.glmnet(v_first_order[s <= 3, ], bchat[s <= 3])
      bc <- predict(bc_lasso, newx = v_first_order, s = "lambda.min")
      
      tibble(
        "mse" = c(
          mean((conf - nu)[s == 4]^2),
          mean((pl - nu)[s == 4]^2),
          mean((bc - nu)[s == 4]^2)  ),
        "method" = c( "conf", "pl", "bc"),
        "algorithm" = "LASSO",
        "sim" = sim_num,
        "p" = p,
        "zeta" = zeta,
        "gamma" = gamma
      )
    }
  }

