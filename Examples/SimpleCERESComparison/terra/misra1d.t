require("problem_setup")
problemsetup(2,2)
-- y = b1*b2*x*((1+b2*x)**(-1))  +  e
terms:insert(y - (b1*b2*x*(1.0/(1.0+b2*x))))
return S:Cost(unpack(terms))