require("problem_setup")
problemsetup(2,2)
-- y = b1 * (1-(1+b2*x/2)**(-2))  +  e
terms:insert(y - (b1 * (1.0-(1.0/ad.pow2(1.0+b2*x/2.0))))
return S:Cost(unpack(terms))