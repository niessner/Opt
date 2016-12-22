require("problem_setup")
problemsetup(3,2)
-- y = b1 / (1+exp[b2-b3*x])  +  e
terms:insert(y - (b1 / (1.0+ad.exp(b2-b3*x))))
return S:Cost(unpack(terms))
