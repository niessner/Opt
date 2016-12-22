require("problem_setup")
problemsetup(4,2)
-- y = (b1/b2) * exp[-0.5*((x-b3)/b2)**2]  +  e
terms:insert(y - ((b1/b2) * ad.exp(-0.5*((x-b3)/b2)**2))
return S:Cost(unpack(terms))