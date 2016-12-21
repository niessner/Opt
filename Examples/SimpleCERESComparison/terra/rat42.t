UNKNOWN_COUNT = 3
require("problem_setup")

-- y = b1 / (1+exp[b2-b3*x])  +  e
Energy(y - (b1 / (1.0+ad.exp(b2-b3*x))))