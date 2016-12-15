UNKNOWN_COUNT = 4
require("problem_setup")

-- y = (b1/b2) * exp[-0.5*((x-b3)/b2)**2]  +  e
Energy(y - ((b1/b2) * ad.exp(-0.5*((x-b3)/b2)**2))