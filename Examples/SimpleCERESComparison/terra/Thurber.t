UNKNOWN_COUNT = 7
require("problem_setup")

local x2 = x*x
local x3 = x*x*x
-- y = (b1 + b2*x + b3*x**2 + b4*x**3) / (1 + b5*x + b6*x**2 + b7*x**3)  +  e
Energy(y - ((b1 + b2*x + b3*x2 + b4*x3) / (1.0 + b5*x + b6*x2 + b7*x3)))