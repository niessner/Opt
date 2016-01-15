require("helper")
local W,H = Dim("W",0), Dim("H",1)
local X = Array2D("X", opt.float4,W,H,0)
local T = Array2D("T", opt.float4,W,H,1)
local M = Array2D("M", float, W,H,2)
UsePreconditioner(false)

Exclude(Not(eq(M(0,0),0)))

for x,y in Stencil { {1,0},{-1,0},{0,1},{0,-1} } do
    local e = (X(0,0) - X(x,y)) - (T(0,0) - T(x,y))
    Energy(Select(InBounds(1,0),e,0))
end
return Result()