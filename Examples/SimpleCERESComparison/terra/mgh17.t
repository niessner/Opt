require("problem_setup")
problemsetup(5,2)
-- y = b1 + b2*exp[-x*b4] + b3*exp[-x*b5]  +  e
terms:insert(y - (b1 + b2*ad.exp(-x*b4) + b3*ad.exp(-x*b5)))
return S:Cost(unpack(terms))