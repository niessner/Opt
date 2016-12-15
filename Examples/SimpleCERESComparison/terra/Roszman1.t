UNKNOWN_COUNT = 4
require("problem_setup")

-- y =  b1 - b2*x - arctan[b3/(x-b4)]/pi  +  e
Energy(y - (b1 - b2*x - ad.atan(b3/(x-b4))/pi))