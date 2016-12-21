UNKNOWN_COUNT = 5
require("problem_setup")

-- y = b1 + b2*exp[-x*b4] + b3*exp[-x*b5]  +  e
Energy(y - (b1 + b2*ad.exp(-x*b4) + b3*ad.exp(-x*b5)))