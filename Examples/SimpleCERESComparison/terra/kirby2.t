require("problem_setup")
problemsetup(5,2)
local x2 = x*x
-- y = (b1 + b2*x + b3*x**2) / (1 + b4*x + b5*x**2)  +  e
terms:insert(y - ((b1 + b2*x + b3*x2) / (1.0 + b4*x + b5*x2)))
return S:Cost(unpack(terms))