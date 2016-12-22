require("problem_setup")
problemsetup(2,2)
-- y = b1 * (1-(1+2*b2*x)**(-.5))  +  e
terms:insert(y - (b1 * (1.0-ad.pow((1.0+2.0*b2*x),-.5))))
return S:Cost(unpack(terms))