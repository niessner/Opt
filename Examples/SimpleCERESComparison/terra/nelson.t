UNKNOWN_COUNT = 3, DATA_DIMENSION = 3
require("problem_setup")

-- log[y] = b1 - b2*x1 * exp[-b3*x2]  +  e
Energy(log(y) - (b1 - b2*x1 * ad.exp(-b3*x2)))