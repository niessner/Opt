local USE_MASK_REFINE = true
local FLOAT_EPSILON = 0.000001
local DEPTH_DISCONTINUITY_THRE = 0.01

local IO = terralib.includec("stdio.h")

local W,H 	= opt.Dim("W",0), opt.Dim("H",1)
local P 	= opt.ProblemSpec()
local D 	= P:Image("X",float, W,H,0) -- Refined Depth
local D_i 	= P:Image("D_i",float, W,H,1) -- Depth input
local I 	= P:Image("I",float, W,H,2) -- Target Intensity
local D_p 	= P:Image("D_p",float, W,H,3) -- Previous Depth
local edgeMaskR 	= P:Image("edgeMaskR",uint8, W,H,4) -- Edge mask. 
local edgeMaskC 	= P:Image("edgeMaskC",uint8, W,H,5) -- Edge mask. 

-- See TerraSolverParameters
local w_p						= P:Param("w_p",float,0)-- Is initialized by the solver!
local w_s		 				= P:Param("w_s",float,1)-- Regularization weight
local w_r						= P:Param("w_r",float,2)-- Prior weight
local w_g						= P:Param("w_g",float,3)-- Shading weight

local weightShadingStart		= P:Param("weightShadingStart",float,4)-- Starting value for incremental relaxation
local weightShadingIncrement	= P:Param("weightShadingIncrement",float,5)-- Update factor

local weightBoundary			= P:Param("weightBoundary",float,6)-- Boundary weight

local f_x						= P:Param("f_x",float,7)
local f_y						= P:Param("f_y",float,8)
local u_x 						= P:Param("u_x",float,9)
local u_y 						= P:Param("u_y",float,10)
    
local offset = 10;
local deltaTransform = {}
for i=1,16 do
	deltaTransform[i] = P:Param("deltaTransform" .. i .. "",float,offset+i)
end
offset = offset + 16

local L = {}
for i=1,9 do
	L[i] = P:Param("L" .. i .. "",float,offset+i)
end
offset = offset + 9
local nNonLinearIterations 	= P:Param("nNonLinearIterations",uint,offset+1) -- Steps of the non-linear solver	
local nLinIterations 		= P:Param("nLinIterations",uint,offset+2) -- Steps of the linear solver
local nPatchIterations 		= P:Param("nPatchIterations",uint,offset+3) -- Steps on linear step on block level

P:Stencil(2)

local C = terralib.includecstring [[
#include <math.h>
]]

local float3 = vector(float, 3)
local float4 = vector(float, 4)
local mat4   = vector(float, 16)


local terra make_float3(x : float, y : float, z : float)
	return vector(x, y, z)
end

local terra make_float4(x : float, y : float, z : float, w : float)
	return vector(x, y, z, w)
end

--[[ May not be needed. Check code gen with vector() first.
local float3 = terralib.types.newstruct("float3")
local float4 = terralib.types.newstruct("float4")

table.insert(float4.entries,{"x",float})
table.insert(float4.entries,{"y",float})
table.insert(float4.entries,{"z",float})
table.insert(float4.entries,{"w",float})

table.insert(float3.entries,{"x",float})
table.insert(float3.entries,{"y",float})
table.insert(float3.entries,{"z",float})

local terra make_float4(x : float, y : float, z : float, w : float)
	var result : float4
	result.x = x
	result.y = y
	result.z = z
	result.w = w
	return result
end

local terra make_float3(x : float, y : float, z : float)
	var result : float3
	result.x = x
	result.y = y
	result.z = z
	return result
end
]]


local terra inLaplacianBounds(i : int64, j : int64, xImage : P:UnknownType()) : bool
	return i > 0 and i < xImage:W()-1 and j > 0 and j < xImage:H()-1
end

local terra laplacian(i : int64, j : int64, gi : int64, gj : int64, xImage : P:UnknownType()) : float
	if not inLaplacianBounds(gi, gj, xImage) then
		return 0
	end

	var x = xImage(i, j)
	var n0 = xImage(i - 1, j)
    var n1 = xImage(i + 1, j)
    var n2 = xImage(i, j - 1)
    var n3 = xImage(i, j + 1)

	var v = 4*x - (n0 + n1 + n2 + n3)
	
	return v
end


local terra cost(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType())
	
	var x = self.X(i, j)

	return x
	
end

local terra IsValidPoint(d : float)
	return d > 0
end

local terra isInsideImage(i : int, j : int, width : int, height : int)
	return (i >= 0 and i < height and j >= 0 and j < width)
end

local terra mat4_times_float4(M : float[16], v : float4)
	var result : float[4]
	escape
		for i=0,3 do
			emit quote 
				result[i] = M[i*4+0]*v[0] + M[i*4+1]*v[1] + M[i*4+2]*v[2] + M[i*4+3]*v[3]
			end
		end
    end
	return make_float4(result[0], result[1], result[2], result[3])
end

local terra estimate_normal_from_depth2(inPriorDepth : P:UnknownType(), gidx : int, gidy : int, W : int, H : int, ax : float, ay : float, f_x : float, f_y : float)
	var retval : float3 = vector(0.0f, 0.0f, 0.0f);

	var d0 : float = inPriorDepth.data[gidy*W+gidx-1];
	var d1 : float = inPriorDepth.data[gidy*W+gidx];
	var d2 : float = inPriorDepth.data[(gidy-1)*W+gidx];

	if IsValidPoint(d0) and IsValidPoint(d1) and IsValidPoint(d2) then
		retval[0] = - d2*(d0-d1)/f_y;
		retval[1] = - d0*(d2-d1)/f_x;
		retval[2] = -ay*retval[1] - ax*retval[0] - d2*d0/f_x/f_y;			
		var an : float = math.sqrt( retval[0] * retval[0] + retval[1] * retval[1] + retval[2] * retval[2] );
		if(an ~= 0) then
			retval[0] = retval[0] / an; 
			retval[1] = retval[1] / an; 
			retval[2] = retval[2] / an;
		end
	end

	return retval;
end

local terra prior_normal_from_previous_depth(d : float, gidx : int64, gidy : int64, self : P:ParameterType(), normal0 : &float3, normal1 : &float3, normal2 : &float3)

	var f_x : float = self.f_x
	var f_y : float = self.f_y
	var u_x : float = self.u_x
	var u_y : float = self.u_y
	var uf_x : float = 1.0f / f_x
	var uf_y : float = 1.0f / f_y
	var W = self.X.W()
	var H = self.X.H()
	
	
	var position_prev : float4 = mat4_times_float4(self.deltaTransform, make_float4((gidx-u_x)*d/f_x, (gidy-u_y)*d/f_y,d,1.0f))

	if((not IsValidPoint(d)) or position_prev[3] ==0.0f) then
		@normal0 = make_float3(0.0f,0.0f,0.0f);
		@normal1 = make_float3(0.0f,0.0f,0.0f);
		@normal2 = make_float3(0.0f,0.0f,0.0f);
		return 	
	end

	var posx = [int](f_x*position_prev[0]/position_prev[2] + u_x +0.5f)
	var posy = [int](f_y*position_prev[1]/position_prev[2] + u_y +0.5f)

	if(posx<2 or posx>(W-3) or posy<2 or posy>(H-3)) then
		@normal0 = make_float3(0.0f,0.0f,0.0f)
		@normal1 = make_float3(0.0f,0.0f,0.0f)
		@normal2 = make_float3(0.0f,0.0f,0.0f)
		return 	
	end

	var ax : float = (posx-u_x)/f_x;
	var ay : float = (posy-u_y)/f_y;

	@normal0 = estimate_normal_from_depth2(self.D_p, posx, posy, W,H, ax, ay, f_x, f_y);
	@normal1 = estimate_normal_from_depth2(self.D_p, posx+1, posy, W, H, ax+uf_x, ay, f_x, f_y);
	@normal2 = estimate_normal_from_depth2(self.D_p, posx, posy+1, W, H,ax, ay+uf_y, f_x, f_y);

	return 
end


local terra calShading2depthGrad(i : int, j : int, posx : int, posy: int, self : P:ParameterType()) : float4
	var f_x : float = self.f_x
	var f_y : float = self.f_y
	var u_x : float = self.u_x
	var u_y : float = self.u_y

	var d0 : float = self.X(i, j-1);
	var d1 : float = self.X(i, j);
	var d2 : float = self.X(i-1, j);
	
	if (IsValidPoint(d0)) and (IsValidPoint(d1)) and (IsValidPoint(d2)) then
		-- TODO: Do we do this in the AD version?
		var greyval : float = (self.I(i, j)*0.5f + self.I(i-1, j)*0.25f + self.I(i, j-1)*0.25f)

		var ax : float = (posx-u_x)/f_x
		var ay : float = (posy-u_y)/f_y
		var an  : float,an2  : float

		var px  : float, py  : float, pz  : float
		px = d2*(d1-d0)/f_y;
		py = d0*(d1-d2)/f_x;
		pz =  - ax*px -ay*py - d2*d0/(f_x*f_y);			
		an2 = px*px+py*py+pz*pz;
		an = math.sqrt(an2);
		if an==0 then
			return vector(0.0f, 0.0f, 0.0f, 0.0f);
		end

		px = px / an;
		py = py / an;
		pz = pz / an;


		var sh_callist0 : float = L[0];
		var sh_callist1 : float = py*L[1];
		var sh_callist2 : float = pz*L[2];
		var sh_callist3 : float = px*L[3];				
		var sh_callist4 : float = px * py * L[4];
		var sh_callist5 : float = py * pz * L[5];
		var sh_callist6 : float = ((-px*px-py*py+2*pz*pz))*L[6];
		var sh_callist7 : float = pz * px * L[7];
		var sh_callist8 : float = ( px * px - py * py )*L[8];


		-- normal changes wrt depth
		var gradx = 0.0f
		var grady = 0.0f
		var gradz = 0.0f

				
		gradx = gradx - sh_callist1*px
		gradx = gradx - sh_callist2*px
		gradx = gradx + (L[3]-sh_callist3*px)
		gradx = gradx + py*L[4]-sh_callist4*2*px
		gradx = gradx - sh_callist5*2*px
		gradx = gradx + (-2*px)*L[6]-sh_callist6*2*px
		gradx = gradx + pz*L[7]-sh_callist7*2*px
		gradx = gradx + 2*px*L[8]-sh_callist8*2*px
		gradx = gradx / an
		

		grady = grady + (L[1]-sh_callist1*py)
		grady = grady - sh_callist2*py
		grady = grady - sh_callist3*py
		grady = grady + px*L[4]-sh_callist4*2*py
		grady = grady + pz*L[5]-sh_callist5*2*py
		grady = grady + (-2*py)*L[6]-sh_callist6*2*py
		grady = grady - sh_callist7*2*py
		grady = grady + (-2*py)*L[8]-sh_callist8*2*py
		grady = grady / an
		
		gradz = gradz - sh_callist1*pz
		gradz = gradz + (L[2]-sh_callist2*pz)
		gradz = gradz - sh_callist3*pz
		gradz = gradz - sh_callist4*2*pz
		gradz = gradz + py*L[5]-sh_callist5*2*pz
		gradz = gradz + 4*pz*L[6]-sh_callist6*2*pz
		gradz = gradz + px*L[7]-sh_callist7*2*pz
		gradz = gradz - sh_callist8*2*pz
		gradz = gradz / an

		-- shading value stored in sh_callist0
		sh_callist0 = sh_callist0 + sh_callist1
		sh_callist0 = sh_callist0 + sh_callist2
		sh_callist0 = sh_callist0 + sh_callist3
		sh_callist0 = sh_callist0 + sh_callist4
		sh_callist0 = sh_callist0 + sh_callist5
		sh_callist0 = sh_callist0 + sh_callist6
		sh_callist0 = sh_callist0 + sh_callist7
		sh_callist0 = sh_callist0 + sh_callist8
		sh_callist0 = sh_callist0 - greyval;



		---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- /
		---- 
		----                /|  2
		----              /  |
		----            /    |  
		----          0 -----|  1
		---- 
		---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- /

		var grnds : float3

		grnds[0] = -d2/f_y;
		grnds[1] = (d1-d2)/f_x;
		grnds[2] = -ax*grnds[0] - ay*grnds[1]-d2/(f_x*f_y);
		sh_callist1 = (gradx*grnds[0]+grady*grnds[1]+gradz*grnds[2]);

		grnds[0] = d2/f_y;
		grnds[1] = d0/f_x;
		grnds[2] = -ax*grnds[0] - ay*grnds[1];
		sh_callist2 = (gradx*grnds[0]+grady*grnds[1]+gradz*grnds[2]);

		grnds[0] = (d1-d0)/f_y;
		grnds[1] = -d0/f_x;
		grnds[2] = -ax*grnds[0] - ay*grnds[1] - d0/(f_x*f_y);
		sh_callist3 = (gradx*grnds[0]+grady*grnds[1]+gradz*grnds[2]);


		return vector(sh_callist1, sh_callist2, sh_callist3, sh_callist0);
	else
		return vector(0.0f, 0.0f, 0.0f, 0.0f);
	end
end


local terra est_lap_init_3d_imp(i : int,  j : int, self : P:ParameterType(), w0 : float, w1 : float, uf_x : float, uf_y : float, b_valid : &bool) : float3

	var retval = vector(0.0f, 0.0f, 0.0f)	
	var d  : float = self.X(i,j)
	var d0 : float = self.X(i-1,j)
	var d1 : float = self.X(i+1,j)
	var d2 : float = self.X(i,j-1)
	var d3 : float = self.X(i,j+1)

	if IsValidPoint(d) and IsValidPoint(d0) and IsValidPoint(d1) and IsValidPoint(d2) and IsValidPoint(d3)
		and math.abs(d-d0)< [float](DEPTH_DISCONTINUITY_THRE) 
		and math.abs(d-d1)< [float](DEPTH_DISCONTINUITY_THRE) 
		and math.abs(d-d2)< [float](DEPTH_DISCONTINUITY_THRE) 
		and math.abs(d-d3)< [float](DEPTH_DISCONTINUITY_THRE) then	
		retval[0] = d * w0 * 4;
		retval[1] = d * w1 * 4;
		retval[2] = d *4;

		retval[0] = retval[0] - d0*(w0 - uf_x);
		retval[1] = retval[1] - d0*w1;
		retval[2] = retval[2] - d0;

		retval[0] = retval[0] - d1*(w0+uf_x);
		retval[1] = retval[1] - d1*w1;
		retval[2] = retval[2] - d1;

		retval[0] = retval[0] - d2*w0;
		retval[1] = retval[1] - d2*(w1-uf_y);
		retval[2] = retval[2] - d2;

		retval[0] = retval[0] - d3*w0;
		retval[1] = retval[1] - d3*(w1+uf_y);
		retval[2] = retval[2] - d3;	
		
	else
		b_valid = false
	end

	return retval;
end



---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
----  evalMinusJTF
---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

local terra evalMinusJTFDeviceLS_SFS_Shared_Mask_Prior(i : int, j : int, posy : int, posx : int, W : uint, H : int, self : P:ParameterType(),
														normal0 : float3, normal1 : float3, normal2 : float3, outPre : &float)
	var f_x : float = self.f_x
	var f_y : float = self.f_y
	var u_x : float = self.u_x
	var u_y : float = self.u_y
	var uf_x : float = 1.0f / f_x
	var uf_y : float = 1.0f / f_y


	var b = 0.0f 
	var p = 0.0f;

	-- Common stuff

	var targetDepth : float = self.D_i(i, j); 
	var validTarget : bool  = IsValidPoint(targetDepth);
	var XC : float 			= self.X(i,j);


		
	if validTarget then
		if posx>1 and posx<(W-5) and posy>1 and posy<(H-5) then
			
			var sum : float, tmpval : float
			var val0 : float, val1 : float, val2 : float
			var maskval : uint8 = 1			

			-- TODO: How do we break this out to amortize for AD
			val0 = calShading2depthGrad(i, j, posx, posy, self)[1] --readValueFromCache2DLS_SFS(inGrady,tidy  ,tidx  );
			val1 = calShading2depthGrad(i + 1, j, posx + 1, posy, self)[0] --readValueFromCache2DLS_SFS(inGradx,tidy  ,tidx+1);
			val2 = calShading2depthGrad(i, j + 1, posx, posy + 1, self)[2] --readValueFromCache2DLS_SFS(inGradz,tidy+1,tidx  );
					
			var shadingDiff_0_m1 = calShading2depthGrad(i+0, j-1, posx+0, posy-1, self)[3]					
			var shadingDiff_1_m1 = calShading2depthGrad(i+1, j-1, posx+1, posy-1, self)[3]
			var shadingDiff_m1_0 = calShading2depthGrad(i-1, j-0, posx-1, posy-0, self)[3]
			var shadingDiff_0_0	 = calShading2depthGrad(i  , j  , posx,   posy,   self)[3]
			var shadingDiff_1_0	 = calShading2depthGrad(i+1, j  , posx+1, posy,   self)[3]
			var shadingDiff_2_0	 = calShading2depthGrad(i+2, j  , posx+2, posy,   self)[3]
			var shadingDiff_m1_1 = calShading2depthGrad(i-1, j+1, posx-1, posy+1, self)[3]
			var shadingDiff_0_1	 = calShading2depthGrad(i  , j+1, posx,   posy+1, self)[3]
			var shadingDiff_1_1	 = calShading2depthGrad(i+1, j+1, posx+1, posy+1, self)[3]
			var shadingDiff_0_2	 = calShading2depthGrad(i  , j+2, posx,   posy+2, self)[3]
			escape
				
				if USE_MASK_REFINE then
					emit quote
						--calculating residue error
						--shading term
						--row edge constraint
						sum = 0.0f;	
						tmpval = 0.0f;			
						tmpval = -(shadingDiff_m1_0 - shadingDiff_0_0) -- edge 0				
						maskval = self.edgeMaskR(i-1, j)
						sum = sum + tmpval*(-val0) * maskval --(posy, posx-1)*val0,(posy,posx)*(-val0)			
						tmpval = tmpval + val0*val0*maskval 

						tmpval = -(shadingDiff_0_0 - shadingDiff_1_0) -- edge 2				
						maskval = self.edgeMaskR(i, j)
						sum = sum + tmpval*(val0-val1) * maskval -- (posy,posx)*(val1-val0), (posy,posx+1)*(val0-val1)			
						tmpval	= tmpval + (val0-val1)*(val0-val1)* maskval

						tmpval = -(shadingDiff_1_0 - shadingDiff_2_0) -- edge 4				
						maskval = self.edgeMaskR(i+1, j)
						sum = sum + tmpval*(val1) * maskval -- (posy,posx+1)*(-val1), (posy,posx+2)*(val1)			
						tmpval	= tmpval + val1*val1* maskval

						tmpval = -(shadingDiff_m1_1 - shadingDiff_0_1) -- edge 5				
						maskval = self.edgeMaskR(i-1, j+1)
						sum = sum + tmpval*(-val2) * maskval -- (posy+1,posx-1)*(val2),(posy+1,pox)*(-val2)			
						tmpval	= tmpval + val2*val2* maskval

						tmpval  = -(shadingDiff_0_1 - shadingDiff_1_1) -- edge 7				
						maskval = self.edgeMaskR(i, j+1)
						sum 	= sum + tmpval*val2 * maskval -- (posy+1,posx)*(-val2),(posy+1,posx+1)*(val2)
						tmpval	= tmpval + val2*val2 * maskval
									
						--column edge constraint			
						tmpval 	= -(shadingDiff_0_m1 - shadingDiff_0_0) -- edge 1
						maskval = self.edgeMaskC(i, j-1)
						sum 	= sum + tmpval*(-val0) * maskval -- (posy-1,posx)*(val0),(posy,posx)*(-val0)			
						tmpval	= tmpval + val0*val0* maskval

						tmpval 	= -(shadingDiff_1_m1 - shadingDiff_1_0) --edge 3
						maskval = self.edgeMaskC(i+1, j-1)
						sum 	= sum + tmpval*(-val1) * maskval --(posy-1,posx+1)*(val1),(posy,posx+1)*(-val1)			
						tmpval	= tmpval +  val1*val1* maskval

						tmpval 	= -(shadingDiff_0_0 - shadingDiff_0_1) --edge 6
						maskval = self.edgeMaskC(i, j)
						sum 	= sum + tmpval*(val0-val2) * maskval -- (posy,posx)*(val2-val0),(posy+1,posx)*(val0-val2)			
						tmpval	= tmpval +  (val0-val2)*(val0-val2)* maskval

						tmpval = -(shadingDiff_1_0 - shadingDiff_1_1) --edge 8
						maskval = self.edgeMaskC(i+1, j)
						sum = sum + tmpval*val1 * maskval -- (posy,posx+1)*(-val1),(posy+1,posx+1)*(val1)
						tmpval	= tmpval +  val1*val1* maskval

						tmpval 	= -(shadingDiff_0_1 - shadingDiff_0_2) --edge 9
						maskval = self.edgeMaskC(i, j+1)
						sum 	= sum + tmpval*val2 * maskval --(posy+1,posx)*(-val2),(posy+2,posx)*(val2)
						tmpval	= tmpval + val2*val2* maskval

						b = b + sum * self.w_g
						p = p + tmpval * self.w_g -- shading constraint
					end
				else
					emit quote
						tmpval = 0.0f
						tmpval = val0 * val0 * 2.0f
						tmpval = tmpval + (val0 - val1) * (val0 - val1)
						tmpval = tmpval + (val0 - val2) * (val0 - val2)
						tmpval = tmpval + val1 * val1 * 3.0f
						tmpval = tmpval + val2 * val2 * 3.0f
						p = p + tmpval * self.w_g -- shading constraint


						sum = 0.0f
						sum = sum + val0 * shadingDiff_0_m1					
						sum = sum + val1 * shadingDiff_1_m1						
						sum = sum + val0 * shadingDiff_m1_0				
						sum = sum + (-val0+val1-val0-val0+val2-val0) * shadingDiff_0_0				
						sum = sum + (val0-val1-val1-val1-val1) * shadingDiff_1_0							
						sum = sum + val1 * shadingDiff_2_0			
						sum = sum + val2 * shadingDiff_m1_1				
						sum = sum +  (-val2-val2+val0-val2-val2) * shadingDiff_0_1				
						sum = sum + (val2+val1) * shadingDiff_1_1			
						sum = sum + val2  * shadingDiff_0_2	
						
						b = b + sum * self.w_g
					end
				end					
			end
			---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
			----                    smoothness term
			---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- /
			var d : float
			var b_valid = true
		
			val0 = (posx - u_x)/f_x	
			val1 = (posy - u_y)/f_y				

			-- smoothness term							
			var  lapval : float3 = est_lap_init_3d_imp(i,j,self,val0,val1,uf_x,uf_y,b_valid);
			sum =  0.0f;
			sum = sum + lapval[0]*val0*(-4.0f);
			sum = sum + lapval[1]*val1*(-4.0f);
			sum = sum + lapval[2]*(-4.0f);
									
			lapval = est_lap_init_3d_imp(i-1,j,self,val0-uf_x,val1,uf_x,uf_y,b_valid);
			sum = sum + lapval[0]*val0;
			sum = sum + lapval[1]*val1;
			sum = sum + lapval[2];
									
			lapval = est_lap_init_3d_imp(i+1,j,self,val0+uf_x,val1,uf_x,uf_y,b_valid);
			sum = sum + lapval[0]*val0;
			sum = sum + lapval[1]*val1;
			sum = sum + lapval[2];
									
			lapval = est_lap_init_3d_imp(i,j-1,self,val0,val1-uf_y,uf_x,uf_y,b_valid);
			sum = sum + lapval[0]*val0
			sum = sum + lapval[1]*val1
			sum = sum + lapval[2]
									
			lapval = est_lap_init_3d_imp(i,j+1,self,val0,val1+uf_y,uf_x,uf_y,b_valid);
			sum = sum + lapval[0]*val0;
			sum = sum + lapval[1]*val1;
			sum = sum + lapval[2]
				
			if b_valid then
				b = b + sum* self.w_s					
				tmpval = (val0 * val0 + val1 * val1 + 1)*(16+4);						
				p = p + tmpval * self.w_s --smoothness
			end
			

						
			--position term 			
			p = p + self.w_r --position constraint			
			b = b -(XC - targetDepth) * self.w_r;



			---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
			----                    prior term
			---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- /
			---- first: calculate the normal for PriorDepth

			sum = 0.0f;
			var ax = (posx-u_x)/f_x;
			var ay = (posy-u_y)/f_y;		
			
			tmpval = normal0[0] * ax + normal0[1] * ay + normal0[2] -- derative of prior energy wrt depth			
			p = p + tmpval * tmpval * 2  * self.w_r;

			d = self.X(i-1,j)
			if(IsValidPoint(d)) then
				sum = sum - tmpval * ( tmpval * self.X(i,j) + ( -tmpval + normal0[0]/f_x) * d );
			end

			d = self.X(i,j-1)
			if(IsValidPoint(d)) then
				sum = sum - tmpval * ( tmpval * self.X(i,j) + ( -tmpval + normal0[1]/f_y) * d );
			end

			tmpval = normal1[0] * ax + normal1[1] * ay + normal1[2] -- derative of prior energy wrt depth			
			p = p + tmpval * tmpval * self.w_r;
			d = self.X(i+1,j)
			if(IsValidPoint(d)) then
				sum = sum + tmpval * ( ( tmpval + normal1[0]/f_x) * d - tmpval * self.X(i,j))
			end

			tmpval = normal2[0] * ax + normal2[1] * ay + normal2[2] -- derative of prior energy wrt depth
			p = p + tmpval * tmpval * self.w_r;
			d = self.X(i,j+1)
			if(IsValidPoint(d)) then
				sum = sum + tmpval * ( ( tmpval + normal2[1]/f_y) * d - tmpval * self.X(i,j))
			end

			b = b + sum  * self.w_r;

			
		end
	end


	if p > [float](FLOAT_EPSILON) then
		outPre = 1.0f/p
	else 
		outPre = 1.0f
	end

	return b;
end


-- eval 2*JtF == \nabla(F); eval diag(2*(Jt)^2) == pre-conditioner
local terra gradient(i : int64, j : int64, gId_i : int64, gId_j : int64, self : P:ParameterType(), outPre : &float)
		
	var Pre     : float
	var normal0 : float3
	var normal1 : float3
	var normal2 : float3

	prior_normal_from_previous_depth(self.X(i, j), gId_j, gId_i, normal0, normal1, normal2);
	
	__syncthreads()

	---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
	----  Initialize linear patch systems
	---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 

	-- Negative gradient
	var negGradient : float = evalMinusJTFDeviceLS_SFS_Shared_Mask_Prior(i, j, gId_i, gId_j, W, H, normal0,normal1,normal2, self, &Pre); 
	@outPre = Pre
	return -2.0f*negGradient
end

local terra add_mul_inp_grad_ls_bsp(self : P:ParameterType(), pImage : P:UnknownType(), i : int, j : int, posx : int, posy : int)
	var gradient = calShading2depthGrad(i, j, posx, posy, self)
	return pImage(i-1,j)	* gradient[0]
		  + pImage(i, j)	* gradient[1]
	   	  + pImage(i,-j)	* gradient[2]
end

local terra est_lap_3d_bsp_imp(pImage : P:UnknownType(), i :int, j : int, w0 : float, w1 : float, uf_x : float, uf_y : float)
	var d  :float = P(i,   j)
	var d0 :float = P(i-1, j)
	var d1 :float = P(i+1, j)
	var d2 :float = P(i,   j-1)
	var d3 :float = P(i,   j+1)
	
	var x : float = ( d * 4 * w0 - d0 * (w0 - uf_x) - d1 * (w0 + uf_x)	- d2 * w0 - d3 * w0);
	var y : float = ( d * 4 * w1 - d0 * w1 - d1 * w1 - d2 * (w1 - uf_y) - d3 * (w1 + uf_y));
	var z : float = ( d * 4 - d0 - d1 - d2 - d3);
	return vector(x,y,z);
end



local terra applyJTJDeviceLS_SFS_Shared_BSP_Mask_Prior(i : int, j : int, posy : int, posx : int, W : uint, H : int, self : P:ParameterType(), pImage : P:UnknownType(),
														normal0 : float3, normal1 : float3, normal2 : float3)

	var f_x : float = self.f_x
	var f_y : float = self.f_y
	var u_x : float = self.u_x
	var u_y : float = self.u_y
	var uf_x : float = 1.0f / f_x
	var uf_y : float = 1.0f / f_y

	var b = 0.0f

	var targetDepth : float = self.D_i(i, j) 
	var validTarget : bool = IsValidPoint(targetDepth);
	var PC : float  		= pImage(i,j)
		
	if validTarget then
		if (posx>1) and (posx<(W-5)) and (posy>1) and (posy<(H-5)) then
			var sum : float = 0.0f;
			var tmpval : float = 0.0f;			

			-- TODO: How do we break this out to amortize for AD
			var val0 : float = calShading2depthGrad(i, j, posx, posy, self)[1] --readV
			var val1 : float = calShading2depthGrad(i + 1, j, posx + 1, posy, self)[0]
			var val2 : float = calShading2depthGrad(i, j + 1, posx, posy + 1, self)[2]

			var grad_0_m1 = calShading2depthGrad(i+0, j-1, posx+0, posy-1, self)					
			var grad_1_m1 = calShading2depthGrad(i+1, j-1, posx+1, posy-1, self)
			var grad_m1_0 = calShading2depthGrad(i-1, j-0, posx-1, posy-0, self)
			var grad_0_0  = calShading2depthGrad(i  , j  , posx,   posy,   self)
			var grad_1_0  = calShading2depthGrad(i+1, j  , posx+1, posy,   self)
			var grad_2_0  = calShading2depthGrad(i+2, j  , posx+2, posy,   self)
			var grad_m1_1 = calShading2depthGrad(i-1, j+1, posx-1, posy+1, self)
			var grad_0_1  = calShading2depthGrad(i  , j+1, posx, posy+1,   self)
			var grad_1_1  = calShading2depthGrad(i+1, j+1, posx+1, posy+1, self)
			var grad_0_2  = calShading2depthGrad(i  , j+2, posx, posy+2,   self)

			escape
				
				if USE_MASK_REFINE then
					emit quote
						--pImage(i  ,j) * grad_0_0[0] // Doesn't do anything?!

						-- the following is the adding of the relative edge constraints to the sum
						-- -val0, edge 0			
						tmpval  = pImage(i-2,j  ) *  grad_m1_0[0];
						tmpval = tmpval + pImage(i-1,j  ) * (grad_m1_0[1] - grad_0_0[0]);
						tmpval = tmpval + pImage(i-1,j-1) *  grad_m1_0[2];
						tmpval = tmpval - pImage(i  ,j  ) *  grad_0_0[1];
						tmpval = tmpval - pImage(i  ,j-1) *  grad_0_0[2];			
						sum = sum + (-val0) * tmpval  * self.edgeMaskR(i-1, j)
						
						-- -val0, edge 1
						tmpval  = pImage(i-1,j-1) *  grad_0_m1[0];
						tmpval = tmpval + pImage(i  ,j-1) * (grad_0_m1[1] - grad_0_0[2]);
						tmpval = tmpval + pImage(i  ,j-2) *  grad_0_m1[2];
						tmpval = tmpval - pImage(i-1,j  ) *  grad_0_0[0];
						tmpval = tmpval - pImage(i  ,j  ) *  grad_0_0[1];		
						sum = sum + (-val0) * tmpval  * self.edgeMaskC(i, j-1)

						-- val0-val1, edge 2
						tmpval  = pImage(i-1,j  ) *  grad_0_0[0];
						tmpval = tmpval + pImage(i  ,j  ) * (grad_0_0[1] - grad_1_0[0]);
						tmpval = tmpval + pImage(i  ,j-1) *  grad_0_0[2];
						tmpval = tmpval - pImage(i+1,j  ) *  grad_1_0[1];
						tmpval = tmpval - pImage(i+1,j-1) *  grad_1_0[2];		
						sum = sum + (val0-val1) * tmpval * self.edgeMaskR(i, j)

						-- -val1, edge 3			
						tmpval  = pImage(i  ,j-1) *  grad_1_m1[0];
						tmpval = tmpval + pImage(i+1,j-1) * (grad_1_m1[1] - grad_1_0[2]);
						tmpval = tmpval + pImage(i+1,j-2) *  grad_1_m1[2];
						tmpval = tmpval - pImage(i  ,j  ) *  grad_1_0[0];
						tmpval = tmpval - pImage(i+1,j  ) *  grad_1_0[1];		
						sum = sum + (-val1) * tmpval	* self.edgeMaskC(i+1, j-1)

						-- val1, edge 4
						tmpval  = pImage(i  ,j  ) *  grad_1_0[0];
						tmpval = tmpval + pImage(i+1,j  ) * (grad_1_0[1] - grad_2_0[0]);
						tmpval = tmpval + pImage(i+1,j-1) *  grad_1_0[2];
						tmpval = tmpval - pImage(i+2,j  ) *  grad_2_0[1];
						tmpval = tmpval - pImage(i+2,j-1) *  grad_2_0[2];		
						sum = sum + (val1) * tmpval * self.edgeMaskR(i+1, j)

						-- -val2, edge 5			
						tmpval  = pImage(i-2,j+1) *  grad_m1_1[0];
						tmpval = tmpval + pImage(i-1,j+1) * (grad_m1_1[1] - grad_0_1[0]);
						tmpval = tmpval + pImage(i-1,j  ) *  grad_m1_1[2];
						tmpval = tmpval - pImage(i  ,j+1) *  grad_0_1[1];
						tmpval = tmpval - pImage(i  ,j  ) *  grad_0_1[2];		
						sum = sum + (-val2) * tmpval * self.edgeMaskR(i-1, j+1)
						
						-- val0-val2, edge 6
						tmpval  = pImage(i-1,j  ) *  grad_0_0[0];
						tmpval = tmpval + pImage(i  ,j  ) * (grad_0_0[1] - grad_0_1[2]);
						tmpval = tmpval + pImage(i  ,j-1) *  grad_0_0[2];
						tmpval = tmpval - pImage(i-1,j+1) *  grad_0_1[0];
						tmpval = tmpval - pImage(i  ,j+1) *  grad_0_1[1];		
						sum = sum + (val0-val2) * tmpval * self.edgeMaskC(i, j)

						-- val2, edge 7
						tmpval  = pImage(i-1,j+1) *  grad_0_1[0];
						tmpval = tmpval + pImage(i  ,j+1) * (grad_0_1[1] - grad_1_1[0]);
						tmpval = tmpval + pImage(i  ,j  ) *  grad_0_1[2];
						tmpval = tmpval - pImage(i+1,j+1) *  grad_1_1[1];
						tmpval = tmpval - pImage(i+1,j  ) *  grad_1_1[2];		
						sum = sum + val2 * tmpval * self.edgeMaskR(i, j+1)

						-- val1, edge 8
						tmpval  = pImage(i  ,j  ) *  grad_1_0[0];
						tmpval = tmpval + pImage(i+1,j  ) * (grad_1_0[1] - grad_1_1[2]);
						tmpval = tmpval + pImage(i+1,j-1) *  grad_1_0[2];
						tmpval = tmpval - pImage(i  ,j+1) *  grad_1_1[0];
						tmpval = tmpval - pImage(i+1,j+1) *  grad_1_1[1];		
						sum = sum + val1 * tmpval * self.edgeMaskC(i+1, j)

						-- val2, edge 9
						tmpval  = pImage(i-1,j+1) *  grad_0_1[0];
						tmpval = tmpval + pImage(i  ,j+1) * (grad_0_1[1] - grad_0_2[2]);
						tmpval = tmpval + pImage(i  ,j  ) *  grad_0_1[2];
						tmpval = tmpval - pImage(i-1,j+2) *  grad_0_2[0];
						tmpval = tmpval - pImage(i  ,j+2) *  grad_0_2[1];		
						sum = sum + val2 * tmpval * self.edgeMaskC(i, j+1)

						b = b + sum * self.w_g
					end

				else
					emit quote										
						sum = sum + (val1*4.0f-val0) * 		add_mul_inp_grad_ls_bsp(self, pImage, i+1, j   , posx+1, posy)-- mulitplication of grad with inP needs to consid			
						sum = sum + (val2*4.0f-val0) * 		add_mul_inp_grad_ls_bsp(self, pImage, i,   j+1 , posx,   posy+1)							
						sum = sum + (val0*4.0f-val1-val2) * add_mul_inp_grad_ls_bsp(self, pImage, i,   j   , posx,   posy)				
						sum = sum + (-val2-val1) * 			add_mul_inp_grad_ls_bsp(self, pImage, i+1, j+1 , posx+1, posy+1) 					
						sum = sum + (-val0) *  				add_mul_inp_grad_ls_bsp(self, pImage, i-1, j   , posx-1, posy)							
						sum = sum + (-val1) *  				add_mul_inp_grad_ls_bsp(self, pImage, i+2, j   , posx+2, posy)						
						sum = sum + (-val0) *  				add_mul_inp_grad_ls_bsp(self, pImage, i,   j-1 , posx,   posy-1)		
						sum = sum + (-val1) *  				add_mul_inp_grad_ls_bsp(self, pImage, i+1, j-1 , posx+1, posy-1)			
						sum = sum + (-val2) *  				add_mul_inp_grad_ls_bsp(self, pImage, i-1, j+1 , posx-1, posy+1)				
						sum = sum + (-val2) *  				add_mul_inp_grad_ls_bsp(self, pImage, i,   j+2 , posx,   posy+2)
						b = b + sum * self.w_g
					end
				end
			end
			
						
			-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
			--                   Smoothness Term
			-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- /
				
			sum = 0
			val0 = (posx - u_x)/f_x
			val1 = (posy - u_y)/f_y
			
			var lapval : float3 = est_lap_3d_bsp_imp(pImage,i,j,val0,val1,uf_x,uf_y)			
			sum = sum + lapval[0]*val0*(4.0f)
			sum = sum + lapval[1]*val1*(4.0f)
			sum = sum + lapval[2]*(4.0f)
						
			lapval = est_lap_3d_bsp_imp(pImage,i-1,j,val0-uf_x,val1,uf_x,uf_y)
			sum = sum - lapval[0]*val0
			sum = sum - lapval[1]*val1
			sum = sum - lapval[2]
						
			lapval = est_lap_3d_bsp_imp(pImage,i+1,j,val0+uf_x,val1,uf_x,uf_y)
			sum = sum - lapval[0]*val0
			sum = sum - lapval[1]*val1
			sum = sum - lapval[2]
						
			lapval = est_lap_3d_bsp_imp(pImage,i,j-1,val0,val1-uf_y,uf_x,uf_y)
			sum = sum - lapval[0]*val0
			sum = sum - lapval[1]*val1
			sum = sum - lapval[2]
						
			lapval = est_lap_3d_bsp_imp(pImage,i,j+1,val0,val1+uf_y,uf_x,uf_y)
			sum = sum - lapval[0]*val0
			sum = sum - lapval[1]*val1
			sum = sum - lapval[2]

			b = b + sum*self.w_s
			

			-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
			--                   Position Term
			-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- /		
			
			b = b +  PC*self.w_p


			-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
			--                    prior term
			-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- /			

			sum = 0.0f
			var ax : float = (posx-u_x)/f_x
			var ay : float = (posy-u_y)/f_y;		
			tmpval = normal0[0] * ax + normal0[1] * ay + normal0[2] --  derative of prior energy wrt depth			
			sum = sum + tmpval * ( tmpval * pImage(i,j) + ( -tmpval + normal0[0]/f_x) * pImage(i,j-1) )
			sum = sum + tmpval * ( tmpval * pImage(i,j) + ( -tmpval + normal0[1]/f_y) * pImage(i-1,j) )
						
			tmpval = normal1[0] * ax + normal1[1] * ay + normal1[2] ;--  derative of prior energy wrt depth			
			sum = sum + -tmpval * ( ( tmpval + normal1[0]/f_x) * pImage(i,j+1) - tmpval * pImage(i,j))
						
			tmpval = normal2[0] * ax + normal2[1] * ay + normal2[2] ;--  derative of prior energy wrt depth			
			sum = sum + -tmpval * ( ( tmpval + normal2[1]/f_y) * pImage(i+1,j) - tmpval * pImage(i,j))

			b = b + sum * self.w_r
		
		end
	end		
	
		
	
	return b;
end


-- eval 2*JtJ (note that we keep the '2' to make it consistent with the gradient
local terra applyJTJ(i : int64, j : int64, gi : int64, gj : int64, self : P:ParameterType(), pImage : P:UnknownType())
	var W : int = self.X:W()
	var H : int = self.X:H()

	var normal0 : float3
	var normal1 : float3
	var normal2 : float3

	prior_normal_from_previous_depth(self.X(i, j), gj, gi, normal0, normal1, normal2);
	
	__syncthreads()

 	var JTJ: float = applyJTJDeviceLS_SFS_Shared_BSP_Mask_Prior(i, j, gj, gi, W, H, self, pImage, normal0, normal1, normal2)
	return 2.0*JTJ
end

P:Function("cost", {W,H}, cost)
P:Function("gradient", {W,H}, gradient)
P:Function("applyJTJ", {W,H}, applyJTJ)


return P