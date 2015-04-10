local IO = terralib.includec("stdio.h")

R = opt.Dim("R")
C = opt.Dim("C")

A = opt.Image(float,R,C)
X = opt.Image(float,C,1)
B = opt.Image(float,R,1)

terra cost(i : uint64, j : uint64, x : X, a : A, b : B)
    IO.printf("the cost function, %d x %d\n",int(i),int(j))
    --IO.printf("A(%d,%d) X(%d,%d) B(%d,%d)\n",int(a.R),int(a.C),int(x.R),int(x.C),int(b.R),int(b.C))
    return 1.0
end

terra gradient(i : uint64, j : uint64, x : X, a : A, b : B)
    IO.printf("the grad function, %d x %d\n",int(i),int(j))
    --IO.printf("A(%d,%d) X(%d,%d) B(%d,%d)\n",int(a.R),int(a.C),int(x.R),int(x.C),int(b.R),int(b.C))
    return 1.0
end

return { dims = {R,C},
         cost = { dim = {R,1}, fn = cost },
         gradient = gradient }
