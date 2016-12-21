UNKNOWN_COUNT = 2
require("problem_setup")

-- y = b1 * (1-(1+2*b2*x)**(-.5))  +  e
Energy(y - (b1 * (1.0-ad.pow((1.0+2.0*b2*x),-.5))))
