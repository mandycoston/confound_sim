


compute_mu <- function(v, z) {
  return(sigmoid(v*.3 + 5* z -1))
}


# inputs: predictor v
# returns a number in [0,1]
compute_nu <- function(v, c) {
  p1 <- compute_mu(v, 1) * (c * sigmoid(v) + (1 - c) * .5)
  p0 <- compute_mu(v, 0) * (1 - (c * sigmoid(v) + (1 - c) * .5))
  return(p1 + p0)
}


a1 <- c(-1, .01, 1.75) # prop_constr : this one achieved expected behavior with bct using eps_sd =5 but not for bc
a2 <- c(1, .1, -2.75) # prop_expanded: this one expanded propensity range but didn't seem to work
a3 <- c(-1.75, .1, 1.75) # propa3 : new one

a <- a3