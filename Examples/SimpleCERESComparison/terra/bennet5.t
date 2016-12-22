require("problem_setup")
problemsetup(3,2)
-- y = b1 * (b2+x)**(-1/b3)  +  e
terms:insert(y - (b1 * ad.pow((b2+x),(-1.0/b3))))
return S:Cost(unpack(terms))
