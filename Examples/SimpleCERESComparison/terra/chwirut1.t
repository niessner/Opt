require("problem_setup")
problemsetup(3,2)
--y = exp[-b1*x]/(b2+b3*x)  +  e
terms:insert(y - ad.exp(-b1 * x) / (b2 + b3 * x))
return S:Cost(unpack(terms))