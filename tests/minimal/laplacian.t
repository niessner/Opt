W,H = Dim("W",0), Dim("H",1)
X = Unknown("X",float,{W,H},0)
A = Array("A",float,{W,H},1)
w_fit = .2
nbr_step = 3
Energy(w_fit*(X(0,0) - A(0,0)) --fitting
,(X(0,0) - X(nbr_step,0)) --regularization
,(X(0,0) - X(0,nbr_step))
,(X(0,0) - X(-nbr_step,0))
,(X(0,0) - X(0,-nbr_step))
)