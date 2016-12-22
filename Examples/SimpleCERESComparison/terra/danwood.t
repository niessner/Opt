require("problem_setup")
problemsetup(2,2)
-- y  = b1*x**b2  +  e
terms:insert(y - (b1*ad.pow(x,b2)))
return S:Cost(unpack(terms))