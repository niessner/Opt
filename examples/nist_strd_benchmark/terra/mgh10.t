require("problem_setup")
problemsetup(3,2)
--  y = b1 * exp[b2/(x+b3)]  +  e
terms:insert(y - (b1 * ad.exp(b2/(x+b3))))
return S:Cost(unpack(terms))