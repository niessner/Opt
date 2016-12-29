require("problem_setup")
problemsetup(4,2)
-- y = b1*(x**2+x*b2) / (x**2+x*b3+b4)  +  e
terms:insert(y - (b1*(x*x+x*b2) / (x*x+x*b3+b4)))
return S:Cost(unpack(terms))