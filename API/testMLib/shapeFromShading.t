local USE_MASK_REFINE = true
local FLOAT_EPSILON = 0.000001f
local DEPTH_DISCONTINUITY_THRE 0.01f

local IO = terralib.includec("stdio.h")

local W,H 	= opt.Dim("W",0), opt.Dim("H",1)
local P 	= opt.ProblemSpec()
local D 	= P:Image("X",W,H,0) -- Refined Depth
local D_i 	= P:Image("D_i",W,H,1) -- Depth input
local I 	= P:Image("I",W,H,2) -- Target Intensity
local D_p 	= P:Image("D_p",W,H,3) -- Previous Depth
local edgeMaskR 	= P:Image("edgeMaskR",W,H,4) -- Edge mask. Currently unused
local edgeMaskC 	= P:Image("edgeMaskC",W,H,5) -- Edge mask. Currently unused

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


local make_float3(x : float, y : float, z : float)
	return vector(x, y, z)
end

local make_float4(x : float, y : float, z : float, w : float)
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
	var a = self.A(i, j)

	var v2 = x - a
	var e_fit = v2 * v2
	
	var v = laplacian(i, j, gi, gj, self.X)
	var e_reg = v * v

	var res = (float)(w_fit*e_fit + w_reg*e_reg)
	return res
	
end

local terra IsValidPoint(d : float)
	return d > 0
end

local isInsideImage(i : int, j : int, width : int, height : int)
	return (i >= 0 and i < height and j >= 0 and j < width)
end

local terra mat4_times_float4(M : float[16], v : float4)
	var result : float[4]
	escape
		for i=0,3 do
			emit `result[i] = M[i*4+0]*v[0] + M[i*4+1]*v[1] + M[i*4+2]*v[2] + M[i*4+3]*v[3]
		end
    end
	return make_float4(result[0], result[1], result[2], result[3])
end

local estimate_normal_from_depth2(inPriorDepth : P:UnknownType(), gidx : int, gidy : int, W : int, H : int, ax : float, ay : float, fx : float, fy : float)
{
	var retval : float3 = vector(0.0f, 0.0f, 0.0f);

	var d0 : float = inPriorDepth.data[gidy*W+gidx-1];
	var d1 : float = inPriorDepth.data[gidy*W+gidx];
	var d2 : float = inPriorDepth.data[(gidy-1)*W+gidx];

	if(IsValidPoint(d0) && IsValidPoint(d1) && IsValidPoint(d2) ){
		retval[0] = - d2*(d0-d1)/fy;
		retval[1] = - d0*(d2-d1)/fx;
		retval[2] = -ay*retval[1] - ax*retval[0] - d2*d0/fx/fy;			
		var an : float = sqrt( retval[0] * retval[0] + retval[1] * retval[1] + retval[2] * retval[2] );
		if(an ~= 0)
		{
			retval[0] = retval[0] / an; 
			retval[1] = retval[1] / an; 
			retval[2] = retval[2] / an;
		}
	}

	return retval;
}

local terra prior_normal_from_previous_depth(d : float, gidx : int64, gidy : int64, self : P:ParameterType(), normal0 : &float3, normal1 : &float3, normal2 : &float3)

	var fx : float = self.fx
	var fy : float = self.fy
	var ux : float = self.ux
	var uy : float = self.uy
	var ufx : float = 1.0f / fx
	var ufy : float = 1.0f / fy
	var W = self.X.W()
	var H = self.X.H()
	
	
	var position_prev : float4 = mat4_times_float4(self.deltaTransform, make_float4((gidx-ux)*d/fx, (gidy-uy)*d/fy,d,1.0f))

	if(!IsValidPoint(d) || position_prev[3] ==0.0f)
	{
		@normal0 = make_float3(0.0f,0.0f,0.0f);
		@normal1 = make_float3(0.0f,0.0f,0.0f);
		@normal2 = make_float3(0.0f,0.0f,0.0f);
		return ;	
	}

	var posx = [int](fx*position_prev[0]/position_prev[2] + ux +0.5f)
	var posy = [int](fy*position_prev[1]/position_prev[2] + uy +0.5f)

	if(posx<2 || posx>(W-3) || posy<2 || posy>(H-3))
	{
		@normal0 = make_float3(0.0f,0.0f,0.0f);
		@normal1 = make_float3(0.0f,0.0f,0.0f);
		@normal2 = make_float3(0.0f,0.0f,0.0f);
		return ;	
	}

	var ax : float = (posx-ux)/fx;
	var ay : float = (posy-uy)/fy;

	@normal0 = estimate_normal_from_depth2(P.D_p, posx, posy, W,H, ax, ay, fx, fy);
	@normal1 = estimate_normal_from_depth2(P.D_p, posx+1, posy, W, H, ax+ufx, ay, fx, fy);
	@normal2 = estimate_normal_from_depth2(P.D_p, posx, posy+1, W, H,ax, ay+ufy, fx, fy);

	return ;	
end


local terra calShading2depthGrad(i : int, j : int, posx : int, posy: int, self : P:ParameterType()) : float4
{
	var fx : float = self.fx
	var fy : float = self.fy
	var ux : float = self.ux
	var uy : float = self.uy

	var d0 : float = self.X(i, j-1);
	var d1 : float = self.X(i, j);
	var d2 : float = self.X(i-1, j);
	
	if (IsValidPoint(d0)) and (IsValidPoint(d1)) and (IsValidPoint(d2)) then
		-- TODO: Do we do this in the AD version?
		var greyval : float = (self.I(i, j)*0.5f + self.I(i-1, j)*0.25f + self.I(i, j-1)*0.25f)

		var ax : float = (posx-ux)/fx
		var ay : float = (posy-uy)/fy
		var an  : float,an2  : float

		var px  : float, py  : float, pz  : float
		px = d2*(d1-d0)/fy;
		py = d0*(d1-d2)/fx;
		pz =  - ax*px -ay*py - d2*d0/(fx*fy);			
		an2 = px*px+py*py+pz*pz;
		an = sqrt(an2);
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

		grnds[0] = -d2/fy;
		grnds[1] = (d1-d2)/fx;
		grnds[2] = -ax*grnds[0] - ay*grnds[1]-d2/(fx*fy);
		sh_callist1 = (gradx*grnds[0]+grady*grnds[1]+gradz*grnds[2]);

		grnds[0] = d2/fy;
		grnds[1] = d0/fx;
		grnds[2] = -ax*grnds[0] - ay*grnds[1];
		sh_callist2 = (gradx*grnds[0]+grady*grnds[1]+gradz*grnds[2]);

		grnds[0] = (d1-d0)/fy;
		grnds[1] = -d0/fx;
		grnds[2] = -ax*grnds[0] - ay*grnds[1] - d0/(fx*fy);
		sh_callist3 = (gradx*grnds[0]+grady*grnds[1]+gradz*grnds[2]);


		return vector(sh_callist1, sh_callist2, sh_callist3, sh_callist0);
	else
		return vector(0.0f, 0.0f, 0.0f, 0.0f);
	end
}


local terra est_lap_init_3d_imp(i : int,  j : int, self : P:ParameterType(), w0 : float, w1 : float, ufx : float, ufy : float, b_valid : &bool) : float3

	var retval = vector(0.0f, 0.0f, 0.0f)	
	var d  : float = self.X(i,j)
	var d0 : float = self.X(i-1,j)
	var d1 : float = self.X(i+1,j)
	var d2 : float = self.X(i,j-1)
	var d3 : float = self.X(i,j+1)

	if IsValidPoint(d) and IsValidPoint(d0) and IsValidPoint(d1) and IsValidPoint(d2) and IsValidPoint(d3)
		and abs(d-d0)<DEPTH_DISCONTINUITY_THRE 
		and abs(d-d1)<DEPTH_DISCONTINUITY_THRE 
		and abs(d-d2)<DEPTH_DISCONTINUITY_THRE 
		and abs(d-d3)<DEPTH_DISCONTINUITY_THRE ) then	
		retval[0] = d * w0 * 4;
		retval[1] = d * w1 * 4;
		retval[2] = d *4;

		retval[0] = retval[0] - d0*(w0 - ufx);
		retval[1] = retval[1] - d0*w1;
		retval[2] = retval[2] - d0;

		retval[0] = retval[0] - d1*(w0+ufx);
		retval[1] = retval[1] - d1*w1;
		retval[2] = retval[2] - d1;

		retval[0] = retval[0] - d2*w0;
		retval[1] = retval[1] - d2*(w1-ufy);
		retval[2] = retval[2] - d2;

		retval[0] = retval[0] - d3*w0;
		retval[1] = retval[1] - d3*(w1+ufy);
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
	var fx : float = self.fx
	var fy : float = self.fy
	var ux : float = self.ux
	var uy : float = self.uy
	var ufx : float = 1.0f / fx
	var ufy : float = 1.0f / fy


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
			var maskval : uchar = 1			

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

			---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
			----                    smoothness term
			---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- /
			var d : float
			var b_valid = true
		
			val0 = (posx - ux)/fx	
			val1 = (posy - uy)/fy				

			-- smoothness term							
			var  lapval : float3 = est_lap_init_3d_imp(inX, tidx,tidy,val0,val1,ufx,ufy,b_valid);
			sum =  0.0f;
			sum = sum + lapval[0]*val0*(-4.0f);
			sum = sum + lapval[1]*val1*(-4.0f);
			sum = sum + lapval[2]*(-4.0f);
									
			lapval = est_lap_init_3d_imp(inX, tidx-1,tidy,val0-ufx,val1,ufx,ufy,b_valid);
			sum = sum + lapval[0]*val0;
			sum = sum + lapval[1]*val1;
			sum = sum + lapval[2];
									
			lapval = est_lap_init_3d_imp(inX, tidx+1,tidy,val0+ufx,val1,ufx,ufy,b_valid);
			sum = sum + lapval[0]*val0;
			sum = sum + lapval[1]*val1;
			sum = sum + lapval[2];
									
			lapval = est_lap_init_3d_imp(inX, tidx,tidy-1,val0,val1-ufy,ufx,ufy,b_valid);
			sum = sum + lapval[0]*val0
			sum = sum + lapval[1]*val1
			sum = sum + lapval[2]
									
			lapval = est_lap_init_3d_imp(inX, tidx,tidy+1,val0,val1+ufy,ufx,ufy,b_valid);
			sum = sum + lapval[0]*val0;
			sum = sum + lapval[1]*val1;
			sum = sum + lapval[2]
				
			if(b_valid) then
				b = b + sum* self.w_s					
				tmpval = (val0 * val0 + val1 * val1 + 1)*(16+4);						
				p = p + tmpval * self.w_s --smoothness
			end
			

						
			--position term 			
			p = p + self.w_r --position constraint			
			b = b -(XC - targetDepth*DEPTH_RESCALE) * self.w_r;



			---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
			----                    prior term
			---- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- /
			---- first: calculate the normal for PriorDepth

			sum = 0.0f;
			var ax = (posx-ux)/fx;
			var ay = (posy-uy)/fy;		
			
			tmpval = normal0[0] * ax + normal0[1] * ay + normal0[2] -- derative of prior energy wrt depth			
			p = p + tmpval * tmpval * 2  * self.w_r;

			d = self.X(i-1,j)
			if(IsValidPoint(d)) then
				sum = sum - tmpval * ( tmpval * self.X(i,j) + ( -tmpval + normal0[0]/fx) * d );
			end

			d = self.X(i,j-1)
			if(IsValidPoint(d)) then
				sum = sum - tmpval * ( tmpval * self.X(i,j) + ( -tmpval + normal0[1]/fy) * d );
			end

			tmpval = normal1[0] * ax + normal1[1] * ay + normal1[2] -- derative of prior energy wrt depth			
			p = p + tmpval * tmpval * self.w_r;
			d = self.X(i+1,j)
			if(IsValidPoint(d)) then
				sum = sum + tmpval * ( ( tmpval + normal1[0]/fx) * d - tmpval * self.X(i,j))
			end

			tmpval = normal2[0] * ax + normal2[1] * ay + normal2[2] -- derative of prior energy wrt depth
			p = p + tmpval * tmpval * self.w_r;
			d = self.X(i,j+1)
			if(IsValidPoint(d)) then
				sum = sum + tmpval * ( ( tmpval + normal2[1]/fy) * d - tmpval * self.X(i,j))
			end

			b = b + sum  * self.w_r;

			
		end
	end


	if p > FLOAT_EPSILON then
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

local terra add_mul_inp_grad_ls_bsp(self : P:ParameterType(), pImage : P:UnknownType(), i : int, j : int)
	var gradient = calShading2depthGrad(i, j, posx, posy, self)
	return pImage(i-1,j)	* gradient[0]
		  + pImage(i, j)	* gradient[1]
	   	  + pImage(i,-j)	* gradient[2]
end

local terra est_lap_3d_bsp_imp(pImage : P:UnknownType(), i :int, j : int, w0 : float, w1 : float, ufx : float, ufx : float)
	var d  :float = P(i,   j)
	var d0 :float = P(i-1, j)
	var d1 :float = P(i+1, j)
	var d2 :float = P(i,   j-1)
	var d3 :float = P(i,   j+1)
	
	var x : float = ( d * 4 * w0 - d0 * (w0 - ufx) - d1 * (w0 + ufx)	- d2 * w0 - d3 * w0);
	var y : float = ( d * 4 * w1 - d0 * w1 - d1 * w1 - d2 * (w1 - ufy) - d3 * (w1 + ufy));
	var z : float = ( d * 4 - d0 - d1 - d2 - d3);
	return vector(x,y,z);
end



local terra applyJTJDeviceLS_SFS_Shared_BSP_Mask_Prior(i : int, j : int, posy : int, posx : int, W : uint, H : int, self : P:ParameterType(), pImage : P:UnknownType(),
														normal0 : float3, normal1 : float3, normal2 : float3)

	var fx : float = self.fx
	var fy : float = self.fy
	var ux : float = self.ux
	var uy : float = self.uy
	var ufx : float = 1.0f / fx
	var ufy : float = 1.0f / fy

	var b = 0.0f

	var targetDepth : float = self.D_i(i, j) 
	validTarget : bool = IsValidPoint(targetDepth);
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
						pImage(i  ,j)	* grad_0_0[0] ;

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
						sum = sum + (val1*4.0f-val0) * 		add_mul_inp_grad_ls_bsp(self, pImage, i+1, j)-- mulitplication of grad with inP needs to consid			
						sum = sum + (val2*4.0f-val0) * 		add_mul_inp_grad_ls_bsp(self, pImage, i,   j+1)							
						sum = sum + (val0*4.0f-val1-val2) * add_mul_inp_grad_ls_bsp(self, pImage, i,   j)				
						sum = sum + (-val2-val1) * 			add_mul_inp_grad_ls_bsp(self, pImage, i+1, j+1)					
						sum = sum + (-val0) *  				add_mul_inp_grad_ls_bsp(self, pImage, i-1, j)							
						sum = sum + (-val1) *  				add_mul_inp_grad_ls_bsp(self, pImage, i+2, j)						
						sum = sum + (-val0) *  				add_mul_inp_grad_ls_bsp(self, pImage, i,   j-1)		
						sum = sum + (-val1) *  				add_mul_inp_grad_ls_bsp(self, pImage, i+1, j-1)			
						sum = sum + (-val2) *  				add_mul_inp_grad_ls_bsp(self, pImage, i-1, j+1)				
						sum = sum + (-val2) *  				add_mul_inp_grad_ls_bsp(self, pImage, i,   j+2)
						b = b + sum * self.w_g
					end
				end
			end
			
						
			-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
			--                   Smoothness Term
			-- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- /
				
			sum = 0
			val0 = (posx - ux)/fx
			val1 = (posy - uy)/fy
			
			var lapval : float3 = est_lap_3d_bsp_imp(pImage,i,j,val0,val1,ufx,ufy)			
			sum = sum + lapval[0]*val0*(4.0f)
			sum = sum + lapval[1]*val1*(4.0f)
			sum = sum + lapval[2]*(4.0f)
						
			lapval = est_lap_3d_bsp_imp(pImage,i-1,j,val0-ufx,val1,ufx,ufy)
			sum = sum - lapval[0]*val0
			sum = sum - lapval[1]*val1
			sum = sum - lapval[2]
						
			lapval = est_lap_3d_bsp_imp(pImage,i+1,j,val0+ufx,val1,ufx,ufy)
			sum = sum - lapval[0]*val0
			sum = sum - lapval[1]*val1
			sum = sum - lapval[2]
						
			lapval = est_lap_3d_bsp_imp(pImage,i,j-1,val0,val1-ufy,ufx,ufy)
			sum = sum - lapval[0]*val0
			sum = sum - lapval[1]*val1
			sum = sum - lapval[2]
						
			lapval = est_lap_3d_bsp_imp(pImage,i,j+1,val0,val1+ufy,ufx,ufy)
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

			sum = 0.0f;
			float ax = (posx-ux)/fx;
			float ay = (posy-uy)/fy;			
			tmpval = normal0[0] * ax + normal0[1] * ay + normal0[2] ;--  derative of prior energy wrt depth			
			sum = sum + tmpval * ( tmpval * readValueFromCache2D_SFS(inP, tidy, tidx) + ( -tmpval + normal0[0]/fx) * readValueFromCache2D_SFS(inP, tidy, tidx-1) );
			sum = sum + tmpval * ( tmpval * readValueFromCache2D_SFS(inP, tidy, tidx) + ( -tmpval + normal0[1]/fy) * readValueFromCache2D_SFS(inP, tidy-1, tidx) );
						
			tmpval = normal1[0] * ax + normal1[1] * ay + normal1[2] ;--  derative of prior energy wrt depth			
			sum = sum + -tmpval * ( ( tmpval + normal1[0]/fx) * readValueFromCache2D_SFS(inP, tidy, tidx+1) - tmpval * readValueFromCache2D_SFS(inP, tidy, tidx));
						
			tmpval = normal2[0] * ax + normal2[1] * ay + normal2[2] ;--  derative of prior energy wrt depth			
			sum = sum + -tmpval * ( ( tmpval + normal2[1]/fy) * readValueFromCache2D_SFS(inP, tidy+1, tidx) - tmpval * readValueFromCache2D_SFS(inP, tidy, tidx));

			b = b + sum * self.w_r;				
		
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

	prior_normal_from_previous_depth(self.X(i, j), gId_j, gId_i, normal0, normal1, normal2);
	
	__syncthreads()

 	var JTJ: float = applyJTJDeviceLS_SFS_Shared_BSP_Mask_Prior(i, j, gj, gi, W, H, self, pImage, normal0, normal1, normal2)
	return 2.0*JTJ
end

P:Function("cost", {W,H}, cost)
P:Function("gradient", {W,H}, gradient)
P:Function("applyJTJ", {W,H}, applyJTJ)


return P