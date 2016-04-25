---
layout: default
---

![Teaser](teaser.jpg)

Many graphics and vision problems are naturally expressed as optimizations with either linear or non-linear least squares objective functions over visual data, such as images and meshes.
The mathematical descriptions of these functions are extremely concise, but their implementation in real code is tedious, especially when optimized for real-time performance in interactive applications.

**Opt** is a new language in which a user simply writes energy functions over image- or graph-structured unknowns, and a compiler automatically generates state-of-the-art GPU optimization kernels.
The end result is a system in which real-world energy functions in graphics and vision applications are expressible in tens of lines of code. They compile directly into highly optimized GPU solver implementations with performance competitive with the best published hand-tuned, application-specific GPU solvers, and 1--2 orders of magnitude beyond a general-purpose auto-generated solver.

[arxiv.org/abs/1604.06525](http://arxiv.org/abs/1604.06525)

For more information, including early access to the code, contact:

* zdevito at cs dot stanford dot edu
* niessner at cs dot stanford dot edu

<!--<form action="https://mailman.stanford.edu/mailman/subscribe/optlang-list" method="POST"> For up-to-date information, join our mailing list: <input name="e-mail" placeholder="email"/><input type="submit" value="Sign Me Up!" />
<br/>
<br/>
</form>-->

---

## Example Problems ##

### ARAP Image Warping ###

<img src="i1.jpg" width="50%"> <img src="c1.jpg" width="49%">
{% highlight lua %}
local W,H = Dim("W",0), Dim("H",1)
local X = Array2D("X", opt.float3,W,H,0) -- u,v,angle
local UrShape = Array2D("UrShape", opt.float2,W,H,1) --original mesh position
local Constraints = Array2D("Constraints", opt.float2,W,H,2)
local Mask = Array2D("Mask", float, W,H,3) -- validity mask for mesh
local w_fitSqrt = Param("w_fitSqrt", float, 0)
local w_regSqrt = Param("w_regSqrt", float, 1)
local Offsets,Angle = Slice(X,0,2), Slice(X,2,3) -- split x into <uv>,<angle>

--regularization
for x,y in Stencil { {1,0}, {-1,0}, {0,1}, {0, -1} } do
    local e_reg = w_regSqrt*((Offsets(0,0) - Offsets(x,y)) 
                  - Rotate(Angle(0,0),(UrShape(0,0) - UrShape(x,y))))
    local valid = InBounds(x,y) * eq(Mask(x,y),0)
    Energy(Select(valid,e_reg,0))
end
--fitting
local e_fit = w_fitSqrt*(Offsets(0,0)- Constraints(0,0))
local valid = All(greatereq(Constraints(0,0),0))
Energy(Select(valid, e_fit , 0.0))
{% endhighlight lua %}

---

### ARAP Mesh Deformation ###

<img src="i2.jpg" width="50%"> <img src="c2.jpg" width="49%">
{% highlight lua %}
local N = opt.Dim("N",0)
local X = Array1D("X", opt.float6,N,0) --vertex.xyz, rotation.xyz <- unknown
local UrShape = Array1D("UrShape", opt.float3,N,1) --original pos: vertex.xyz
local Constraints = Array1D("Constraints", opt.float3,N,2) --user constraints
local G = Graph("G", 0, 
                "v0", N, 0, 
                "v1", N, 1)

UsePreconditioner(true)
local w_fitSqrt = Param("w_fitSqrt", float, 0)
local w_regSqrt = Param("w_regSqrt", float, 1)

local Offset = Slice(X,0,3) --split x into <position>,<rotation>
local Angle = Slice(X,3,6)

--fitting
local e_fit = Offset(0,0) - Constraints(0,0)
local valid = greatereq(Constraints(0,0,0), -999999.9)
Energy(Select(valid,w_fitSqrt*e_fit,0))

--regularization
local ARAPCost = (Offset(G.v0) - Offset(G.v1)) 
                  - Rotate(Angle(G.v0),UrShape(G.v0) - UrShape(G.v1))
Energy(w_regSqrt*ARAPCost)

{% endhighlight lua %}

---

### Shape From Shading ###

<img src="i3.jpg" width="50%"> <img src="c3.jpg" width="49%">
{% highlight lua %}
local DEPTH_DISCONTINUITY_THRE = 0.01
local W,H     = Dim("W",0), Dim("H",1)

local X     = Array2D("X",float, W,H,0) -- Refined Depth
local D_i     = Array2D("D_i",float, W,H,1) -- Depth input

local Im     = Array2D("Im",float, W,H,2) -- Target Intensity

local edgeMaskR = Array2D("edgeMaskR",uint8, W,H,4) -- Edge mask. 
local edgeMaskC = Array2D("edgeMaskC",uint8, W,H,5) -- Edge mask. 

local w_p                        = sqrt(Param("w_p",float,0))-- Fitting weight
local w_s                         = sqrt(Param("w_s",float,1))-- Regularization weight
local w_g                        = sqrt(Param("w_g",float,3))-- Shading weight

local f_x                        = Param("f_x",float,7)
local f_y                        = Param("f_y",float,8)
local u_x                         = Param("u_x",float,9)
local u_y                         = Param("u_y",float,10)
    
local offset = 26;
local L = {}
for i=1,9 do -- lighting model parameters
    L[i] = Param("L_" .. i .. "",float,offset+i)
end

local posX,posY = W:index(),H:index()

-- equation 8
function p(offX,offY) 
    local d = X(offX,offY)
    local i = offX + posX
    local j = offY + posY
    return Vector(((i-u_x)/f_x)*d, ((j-u_y)/f_y)*d, d)
end

-- equation 10
function normalAt(offX, offY)
    local i = offX + posX -- good
    local j = offY + posY -- good
    
    local n_x = X(offX, offY - 1) * (X(offX, offY) - X(offX - 1, offY)) / f_y
    local n_y = X(offX - 1, offY) * (X(offX, offY) - X(offX, offY - 1)) / f_x
    local n_z = (n_x * (u_x - i) / f_x) + (n_y * (u_y - j) / f_y) - (X(offX-1, offY)*X(offX, offY-1) / (f_x*f_y))
    local sqLength = n_x*n_x + n_y*n_y + n_z*n_z
    local inverseMagnitude = Select(greater(sqLength, 0.0), 1.0/sqrt(sqLength), 1.0)
    return inverseMagnitude * Vector(n_x, n_y, n_z)
end

function B(offX, offY)
    local normal = normalAt(offX, offY)
    local n_x = normal[0]
    local n_y = normal[1]
    local n_z = normal[2]

    return           L[1] +
                     L[2]*n_y + L[3]*n_z + L[4]*n_x  +
                     L[5]*n_x*n_y + L[6]*n_y*n_z + L[7]*(-n_x*n_x - n_y*n_y + 2*n_z*n_z) + L[8]*n_z*n_x + L[9]*(n_x*n_x-n_y*n_y)
end

function I(offX, offY)
    return Im(offX,offY)*0.5 + 0.25*(Im(offX-1,offY)+Im(offX,offY-1))
end

local function DepthValid(x,y) return greater(D_i(x,y),0) end
 
local function B_I(x,y)
    local bi = B(x,y) - I(x,y)
    local valid = And(DepthValid(x-1,y),DepthValid(x,y),DepthValid(x,y-1))
    return Select(InBounds(0,0,1,1)*valid,bi,0)
end
B_I = ComputedArray(W,H, B_I(0,0))

-- fitting term
local E_p = X(0,0) - D_i(0,0)
Energy(Select(DepthValid(0,0),w_p*E_p,0))

-- shading term
local E_g_h = (B_I(0,0) - B_I(1,0))*edgeMaskR(0,0)
local E_g_v = (B_I(0,0) - B_I(0,1))*edgeMaskC(0,0)
Energy(Select(InBounds(0,0,1,1),w_g*E_g_h,0))
Energy(Select(InBounds(0,0,1,1),w_g*E_g_v,0))

-- regularization term
local function Continuous(x,y) return less(abs(X(0,0) - X(x,y)),DEPTH_DISCONTINUITY_THRE) end
local valid = And(DepthValid(0,0),DepthValid(0,-1),DepthValid(0,1),DepthValid(-1,0),DepthValid(1,0),
                  Continuous(0,-1),Continuous(0,1),Continuous(-1,0),Continuous(1,0),InBounds(0,0,1,1))
valid = eq(ComputedArray(W,H,valid)(0,0),1)
local E_s = 4.0*p(0,0) - (p(-1,0) + p(0,-1) + p(1,0) + p(0,1)) 
Energy(Select(valid,w_s*E_s,0))

-- do not include unknowns for where the depth is invalid
Exclude(Not(DepthValid(0,0)))

{% endhighlight lua %}

---

### Poisson Image Editing ###

<img src="i4.jpg" width="50%"> <img src="c4.jpg" width="49%">
{% highlight lua %}
local W,H = Dim("W",0), Dim("H",1)
local X = Array2D("X", opt.float4,W,H,0) -- unknown, initialized to base image
local T = Array2D("T", opt.float4,W,H,1) -- inserted image
local M = Array2D("M", float, W,H,2) -- mask, excludes parts of base image
UsePreconditioner(false)

-- do not include unmasked pixels in the solve
Exclude(Not(eq(M(0,0),0)))

for x,y in Stencil { {1,0},{-1,0},{0,1},{0,-1} } do
    local e = (X(0,0) - X(x,y)) - (T(0,0) - T(x,y))
    Energy(Select(InBounds(1,0),e,0))
end
{% endhighlight lua %}

---

### Cotangent Mesh Smoothing ###

<img src="i5.jpg" width="50%"> 
{% highlight lua %}
local N = Dim("N",0)

local X =             Array1D("X", opt.float12,N,0) -- 3d offset + 3x3 mat
local UrShape =     Array1D("UrShape", opt.float3,N,1) -- original position
local Constraints = Array1D("Constraints", opt.float3,N,2) -- user constraints
local G = Graph("G", 0, 
                 "v0", N, 0, 
                 "v1", N, 1)
UsePreconditioner(true)

local w_fitSqrt = Param("w_fitSqrt", float, 0)
local w_regSqrt = Param("w_regSqrt", float, 1)
local w_rotSqrt = Param("w_rotSqrt", float, 2)

local Offset = Slice(X,0,3) -- select part of unknown for position

--fitting
local e_fit = Offset(0,0) - Constraints(0,0)
local valid = greatereq(Constraints(0,0,0), -999999.9)
Energy(Select(valid, w_fitSqrt*e_fit, 0))

--rot
local RotMatrix = Slice(X,3,12) -- extract rotation matrix
local R = RotMatrix(0,0)
local c0 = Vector(R(0), R(3), R(6))
local c1 = Vector(R(1), R(4), R(7))
local c2 = Vector(R(2), R(5), R(8))
Energy(w_rotSqrt*Dot(c0,c1))
Energy(w_rotSqrt*Dot(c0,c2))
Energy(w_rotSqrt*Dot(c1,c2))
Energy(w_rotSqrt*(Dot(c0,c0)-1))
Energy(w_rotSqrt*(Dot(c1,c1)-1))
Energy(w_rotSqrt*(Dot(c2,c2)-1))

local regCost = (Offset(G.v1) - Offset(G.v0)) - 
                Matrix3x3Mul(RotMatrix(G.v0), (UrShape(G.v1) - UrShape(G.v0)))

Energy(w_regSqrt*regCost)
{% endhighlight lua %}
---

### Embedded Deformation ###

<img src="i6.jpg" width="50%"> 
{% highlight lua %}
local N = Dim("N",0)
local X = Array1D("X", opt.float3,N,0) -- position
local A = Array1D("A", opt.float3,N,1) -- orig position
local G = Graph("G", 0, "v0", N, 0, --current vertex
                       "v1", N, 1, --neighboring vertex
                       "v2", N, 2, --prev neighboring vertex
                       "v3", N, 3) --next neighboring vertex

UsePreconditioner(true)

local w_fitSqrt = Param("w_fit", float, 0)
local w_regSqrt = Param("w_reg", float, 1)

function cot(v0, v1) 
    local adotb = Dot(v0, v1)
    local disc = Dot(v0, v0)*Dot(v1, v1) - adotb*adotb
    disc = Select(greater(disc, 0.0), disc,  0.0001)
    return Dot(v0, v1) / sqrt(disc)
end

function normalize(v)
    return v / sqrt(Dot(v, v))
end

function length(v0, v1) 
    local diff = v0 - v1
    return sqrt(Dot(diff, diff))
end


-- fit energy
Energy(w_fitSqrt*(X(0,0) - A(0,0)))

local a = normalize(X(G.v0) - X(G.v2)) --float3
local b = normalize(X(G.v1) - X(G.v2))    --float3
local c = normalize(X(G.v0) - X(G.v3))    --float3
local d = normalize(X(G.v1) - X(G.v3))    --float3

--cotangent laplacian; Meyer et al. 03
local w = 0.5*(cot(a,b) + cot(c,d))
w = sqrt(Select(greater(w, 0.0), w, 0.0001)) 
Energy(w_regSqrt*w*(X(G.v1) - X(G.v0)))

{% endhighlight lua %}
---

### Optical Flow ###

<img src="i7.jpg" width="50%"> 
{% highlight lua %}
local W,H = Dim("W",0), Dim("H",1)
local X = Array2D("X", opt.float2,W,H,0) -- flow vectors
local I = Array2D("I",float,W,H,1) -- frame1
local I_hat_im = Array2D("I_hat",float,W,H,2) -- frame2, sampled
local I_hat_dx = Array2D("I_hat_dx",float,W,H,3) -- partials for frame
local I_hat_dy = Array2D("I_hat_dy",float,W,H,4)
 -- create a new math operator that samples from the image
local I_hat = SampledImage(I_hat_im,I_hat_dx,I_hat_dy)

local i,j = W:index(),H:index()
UsePreconditioner(false)
local w_fitSqrt = Param("w_fit", float, 0)
local w_regSqrt = Param("w_reg", float, 1)
-- fitting
local e_fit = w_fitSqrt*(I(0,0) - I_hat(i + X(0,0,0),j + X(0,0,1)))
Energy(e_fit)
-- regularization
for nx,ny in Stencil { {1,0}, {-1,0}, {0,1}, {0,-1} } do
    local e_reg = w_regSqrt*(X(0,0) - X(nx,ny))
    Energy(Select(InBounds(nx,ny),e_reg,0))
end
{% endhighlight lua %}