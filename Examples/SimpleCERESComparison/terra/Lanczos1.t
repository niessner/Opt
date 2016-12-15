UNKNOWN_COUNT = 6
require("problem_setup")

--y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)  +  e
Energy(y - (b1*ad.exp(-b2*x) + b3*ad.exp(-b4*x) + b5*ad.exp(-b6*x)))