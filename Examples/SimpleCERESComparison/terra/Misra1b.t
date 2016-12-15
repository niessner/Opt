UNKNOWN_COUNT = 2
require("problem_setup")

-- y = b1 * (1-(1+b2*x/2)**(-2))  +  e
Energy(y - (b1 * (1.0-(1.0/ad.pow2(1.0+b2*x/2.0))))