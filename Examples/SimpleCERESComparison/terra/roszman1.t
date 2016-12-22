require("problem_setup")
problemsetup(4,2)
-- y =  b1 - b2*x - arctan[b3/(x-b4)]/pi  +  e
terms:insert(y - (b1 - b2*x - ad.atan(b3/(x-b4))/pi))
return S:Cost(unpack(terms))