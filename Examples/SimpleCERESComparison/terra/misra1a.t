require("problem_setup")
problemsetup(2,2)
-- y = b1*(1-exp[-b2*x])  +  e
terms:insert(y - (b1*(1.0 - ad.exp(-b2*x))))
return S:Cost(unpack(terms))