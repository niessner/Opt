UNKNOWN_COUNT = 4
require("problem_setup")

-- y = b1 / ((1+exp[b2-b3*x])**(1/b4))  +  e
Energy(y - (b1 / (pow((1.0+ad.exp(b2-b3*x)),1.0/b4))))