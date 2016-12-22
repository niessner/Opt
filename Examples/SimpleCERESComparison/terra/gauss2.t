require("problem_setup")
problemsetup(8,2)
-- y = b1*exp( -b2*x ) + b3*exp( -(x-b4)**2 / b5**2 ) + b6*exp( -(x-b7)**2 / b8**2 ) + e
terms:insert(y - (b1*ad.exp(-b2*x) + b3*ad.exp(-sqr(x-b4)/sqr(b5)) + sqr(b6)*ad.exp(-sqr(x-b7)/sqr(b8))))
return S:Cost(unpack(terms))