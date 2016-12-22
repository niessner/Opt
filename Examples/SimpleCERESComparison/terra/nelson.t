require("problem_setup")
problemsetup(3,3)
-- log[y] = b1 - b2*x1 * exp[-b3*x2]  +  e
terms:insert(log(y) - (b1 - b2*x1 * ad.exp(-b3*x2)))
return S:Cost(unpack(terms))