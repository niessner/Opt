UNKNOWN_COUNT = 3
require("problem_setup")

-- y = exp(-b1*x)/(b2+b3*x)  +  e
Energy(y - (ad.exp(-b1*x)/(b2+b3*x)))