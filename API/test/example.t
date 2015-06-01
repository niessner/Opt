local IO = terralib.includec("stdio.h")

R = opt.Dim("R")
C = opt.Dim("C")

A = opt.Image(double,R,C)
X = opt.Image(double,C,1)
B = opt.Image(double,R,1)

terra cost(i : uint32, j : uint32, x : X, a : A, b : B)
    IO.printf("the cost function, %d x %d\n",int(i),int(j))
    --IO.printf("A(%d,%d) X(%d,%d) B(%d,%d)\n",int(a.R),int(a.C),int(x.R),int(x.C),int(b.R),int(b.C))
    return 1.0
end

terra gradient(i : uint32, j : uint32, x : X, a : A, b : B)
    IO.printf("the grad function, %d x %d\n",int(i),int(j))
    --IO.printf("A(%d,%d) X(%d,%d) B(%d,%d)\n",int(a.R),int(a.C),int(x.R),int(x.C),int(b.R),int(b.C))
    return 1.0
end

return { dims = {R,C},
         cost = { dim = {R,1}, fn = cost },
         gradient = gradient }
