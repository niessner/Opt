require("problem_setup")
problemsetup(6,2)
--y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)  +  e
terms:insert(y - (b1*ad.exp(-b2*x) + b3*ad.exp(-b4*x) + b5*ad.exp(-b6*x)))
return S:Cost(unpack(terms))