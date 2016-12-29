require("problem_setup")
problemsetup(4,2)
-- y = b1 / ((1+exp[b2-b3*x])**(1/b4))  +  e
terms:insert(y - (b1 / (pow((1.0+ad.exp(b2-b3*x)),1.0/b4))))
return S:Cost(unpack(terms))