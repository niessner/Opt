UNKNOWN_COUNT = 2
require("problem_setup")

-- y = b1*b2*x*((1+b2*x)**(-1))  +  e
Energy(y - (b1*b2*x*(1.0/(1.0+b2*x))))