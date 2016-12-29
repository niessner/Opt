require("problem_setup")
problemsetup(9,2)
--y = b1 + b2*cos( 2*pi*x/12 ) + b3*sin( 2*pi*x/12 ) 
--                      + b5*cos( 2*pi*x/b4 ) + b6*sin( 2*pi*x/b4 )
--                      + b8*cos( 2*pi*x/b7 ) + b9*sin( 2*pi*x/b7 )  + e
terms:insert(y - (b1 + b2*ad.cos( 2.0*pi*x/12.0 ) + b3*ad.sin( 2.0*pi*x/12.0 ) 
                      + b5*ad.cos( 2.0*pi*x/b4 ) + b6*ad.sin( 2.0*pi*x/b4 )
                      + b8*ad.cos( 2.0*pi*x/b7 ) + b9*ad.sin( 2.0*pi*x/b7 )))
return S:Cost(unpack(terms))