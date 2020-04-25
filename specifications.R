
# Catalogs possible functions for mu, pi 

a1 <- c(-1, .01, 1.75) # prop_constr : this one achieved expected behavior with bct using eps_sd =5 but not for bc
a2 <- c(1, .1, -2.75) # prop_expanded: this one expanded propensity range but didn't seem to work
a3 <- c(-1.75, .1, 1.75) # propa3 : new one

# smooth but spans whole range
compute_mu <- function(v, z) {
  return(sigmoid(v*.3 + 5* z -1))
}

# smooth and constrained range
# compute_mu <- function(v, z) {
#   return(.005 * v + .5 * z + .2)
# }


# discontinuous
compute_mu <- function(v, z) {
  return(dplyr::if_else(v < -10, 0.8*z + 0.6*(1-z),
                        dplyr::if_else(v <= -5, v/10 + 1 + z/3 + (1-z)/12, 
                                       dplyr::if_else(v < 0, v/10 + z*25/26 + (1-z) * 4/7, 
                                                      dplyr::if_else(v < 5, v*z/5 + v*(1-z)/18,
                                                                     z*0.65 +(1-z)*0.9)))))
}
