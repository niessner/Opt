#pragma once

#ifndef _PATCH_SOLVER_SFS_EQUATIONS_
#define _PATCH_SOLVER_SFS_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "PatchSolverSFSUtil.h"
#include "PatchSolverSFSState.h"
#include "PatchSolverSFSParameters.h"

#include "SolverTerms.h"

#define DEPTH_DISCONTINUITY_THRE 0.01

//////////////////////////////////////////
// evaluate the gradient of shading energy
//////////////////////////////////////////

__device__ inline float est_lap_init(volatile float *data,int tidx, int tidy, bool &b_valid)
{
	const float d  = readValueFromCache2D_SFS(data, tidy, tidx);
	float retval = 0.0f;
	
	const float d0 = readValueFromCache2D_SFS(data, tidy, tidx-1);
	const float d1 = readValueFromCache2D_SFS(data, tidy, tidx+1);
	const float d2 = readValueFromCache2D_SFS(data, tidy-1, tidx);
	const float d3 = readValueFromCache2D_SFS(data, tidy+1, tidx);

	if(IsValidPoint(d) && IsValidPoint(d0) && IsValidPoint(d1) && IsValidPoint(d2) && IsValidPoint(d3)
		&& abs(d-d0)<DEPTH_DISCONTINUITY_THRE 
		&& abs(d-d1)<DEPTH_DISCONTINUITY_THRE 
		&& abs(d-d2)<DEPTH_DISCONTINUITY_THRE 
		&& abs(d-d3)<DEPTH_DISCONTINUITY_THRE )
		retval = d*4.0f - d0 - d1 - d2 - d3;
	else
		b_valid = false;

	return retval;
}


__device__ inline float3 est_lap_init_3d_imp(volatile float *data, int tidx, int tidy, float w0, float w1, const float &ufx, const float &ufy, bool &b_valid)
{
	float3 retval;	
	retval.x = 0.0f;
	retval.y = 0.0f;
	retval.z = 0.0f;	
	float d  = readValueFromCache2D_SFS(data, tidy, tidx);
	float d0 = readValueFromCache2D_SFS(data, tidy, tidx-1);
	float d1 = readValueFromCache2D_SFS(data, tidy, tidx+1);
	float d2 = readValueFromCache2D_SFS(data, tidy-1, tidx);
	float d3 = readValueFromCache2D_SFS(data, tidy+1, tidx);

	if(IsValidPoint(d) && IsValidPoint(d0) && IsValidPoint(d1) && IsValidPoint(d2) && IsValidPoint(d3)
		&& abs(d-d0)<DEPTH_DISCONTINUITY_THRE 
		&& abs(d-d1)<DEPTH_DISCONTINUITY_THRE 
		&& abs(d-d2)<DEPTH_DISCONTINUITY_THRE 
		&& abs(d-d3)<DEPTH_DISCONTINUITY_THRE )

	{		
		retval.x = d * w0 * 4;
		retval.y = d * w1 * 4;
		retval.z = d *4;

		retval.x -= d0*(w0 - ufx);
		retval.y -= d0*w1;
		retval.z -= d0;

		retval.x -= d1*(w0+ufx);
		retval.y -= d1*w1;
		retval.z -= d1;

		retval.x -= d2*w0;
		retval.y -= d2*(w1-ufy);
		retval.z -= d2;

		retval.x -= d3*w0;
		retval.y -= d3*(w1+ufy);
		retval.z -= d3;	
		
	}else
		b_valid = false;

	return retval;
}


__device__ inline float3 est_lap_3d_bsp_imp(volatile float *data,int tidx, int tidy,float w0, float w1, const float &ufx, const float &ufy)
{
	float3 retval;

	const float d	= data[getLinearShareMemLocate_SFS(tidy, tidx)];
	const float d0	= data[getLinearShareMemLocate_SFS(tidy, tidx-1)];
	const float d1	= data[getLinearShareMemLocate_SFS(tidy, tidx+1)];
	const float d2	= data[getLinearShareMemLocate_SFS(tidy-1, tidx)];
	const float d3	= data[getLinearShareMemLocate_SFS(tidy+1, tidx)];
	
	retval.x = ( d * 4 * w0 - d0 * (w0 - ufx) - d1 * (w0 + ufx)	- d2 * w0 - d3 * w0);
	retval.y = ( d * 4 * w1 - d0 * w1 - d1 * w1 - d2 * (w1 - ufy) - d3 * (w1 + ufy));
	retval.z = ( d * 4 - d0 - d1 - d2 - d3);
	return retval;

}

__device__ inline float est_lap_bsp(volatile float *data,int tidx, int tidy)
{
	return (data[getLinearShareMemLocate_SFS(tidy, tidx)]*4.0f 
		- data[getLinearShareMemLocate_SFS(tidy, tidx-1)] 
		- data[getLinearShareMemLocate_SFS(tidy, tidx+1)] 
		- data[getLinearShareMemLocate_SFS(tidy-1, tidx)] 
		- data[getLinearShareMemLocate_SFS(tidy+1, tidx)]);
}





__inline__ __device__ float4 calShading2depthGrad(volatile float* inX, int tidx, int tidy, int posx,int posy, PatchSolverInput &input)
{
	const int imgind = posy * input.width + posx;

	const float fx = input.calibparams.fx;
	const float fy = input.calibparams.fy;
	const float ux = input.calibparams.ux;
	const float uy = input.calibparams.uy;

	const float d0 = readValueFromCache2D_SFS(inX, tidy, tidx-1);
	const float d1 = readValueFromCache2D_SFS(inX, tidy, tidx);
	const float d2 = readValueFromCache2D_SFS(inX, tidy-1, tidx);
	
	if( (IsValidPoint(d0)) && (IsValidPoint(d1)) && (IsValidPoint(d2)))	
	{
		const float greyval = (input.d_targetIntensity[imgind]*0.5f + input.d_targetIntensity[imgind-1]*0.25f + input.d_targetIntensity[imgind-input.width]*0.25f)*RGB_RANGE_SCALE;

		float ax = (posx-ux)/fx;
		float ay = (posy-uy)/fy;
		float an,an2;

		float px,py,pz;
		px = d2*(d1-d0)/fy;
		py = d0*(d1-d2)/fx;
		pz =  - ax*px -ay*py - d2*d0/(fx*fy);			
		an2 = px*px+py*py+pz*pz;
		an = sqrt(an2);
		if(an==0)
		{
			float4 retval;
			retval.x = 0.0f;retval.y = 0.0f;retval.z = 0.0f;retval.w = 0.0f;
			return retval;
		}

		px /= an;
		py /= an;
		pz /= an;


		float sh_callist0 = input.d_litcoeff[0];
		float sh_callist1 = py*input.d_litcoeff[1];
		float sh_callist2 = pz*input.d_litcoeff[2];
		float sh_callist3 = px*input.d_litcoeff[3];				
		float sh_callist4 = px * py * input.d_litcoeff[4];
		float sh_callist5 = py * pz * input.d_litcoeff[5];
		float sh_callist6 = ((-px*px-py*py+2*pz*pz))*input.d_litcoeff[6];
		float sh_callist7 = pz * px * input.d_litcoeff[7];
		float sh_callist8 = ( px * px - py * py )*input.d_litcoeff[8];


		//normal changes wrt depth
		register float gradx =0, grady = 0, gradz = 0;

				
		gradx += -sh_callist1*px;
		gradx += -sh_callist2*px;
		gradx += (input.d_litcoeff[3]-sh_callist3*px);
		gradx += py*input.d_litcoeff[4]-sh_callist4*2*px;
		gradx += -sh_callist5*2*px;
		gradx += (-2*px)*input.d_litcoeff[6]-sh_callist6*2*px;
		gradx += pz*input.d_litcoeff[7]-sh_callist7*2*px;
		gradx += 2*px*input.d_litcoeff[8]-sh_callist8*2*px;
		gradx /= an;
		

		grady += (input.d_litcoeff[1]-sh_callist1*py);
		grady += -sh_callist2*py;
		grady += -sh_callist3*py;
		grady += px*input.d_litcoeff[4]-sh_callist4*2*py;
		grady += pz*input.d_litcoeff[5]-sh_callist5*2*py;
		grady += (-2*py)*input.d_litcoeff[6]-sh_callist6*2*py;
		grady += -sh_callist7*2*py;
		grady += (-2*py)*input.d_litcoeff[8]-sh_callist8*2*py;
		grady /= an;
		
		gradz += -sh_callist1*pz;
		gradz += (input.d_litcoeff[2]-sh_callist2*pz);
		gradz += -sh_callist3*pz;
		gradz += -sh_callist4*2*pz;
		gradz += py*input.d_litcoeff[5]-sh_callist5*2*pz;
		gradz += 4*pz*input.d_litcoeff[6]-sh_callist6*2*pz;
		gradz += px*input.d_litcoeff[7]-sh_callist7*2*pz;
		gradz += -sh_callist8*2*pz;
		gradz /= an;

		//shading value stored in sh_callist0
		sh_callist0 += sh_callist1;
		sh_callist0 += sh_callist2;
		sh_callist0 += sh_callist3;
		sh_callist0 += sh_callist4;
		sh_callist0 += sh_callist5;
		sh_callist0 += sh_callist6;
		sh_callist0 += sh_callist7;
		sh_callist0 += sh_callist8;
		sh_callist0 -= greyval;



		///////////////////////////////////////////////////////
		//
		//               /|  2
		//             /  |
		//           /    |  
		//         0 -----|  1
		//
		///////////////////////////////////////////////////////

		float3 grnds;

		grnds.x = -d2/fy;
		grnds.y = (d1-d2)/fx;
		grnds.z = -ax*grnds.x - ay*grnds.y-d2/(fx*fy);
		sh_callist1 = (gradx*grnds.x+grady*grnds.y+gradz*grnds.z);

		grnds.x = d2/fy;
		grnds.y = d0/fx;
		grnds.z = -ax*grnds.x - ay*grnds.y;
		sh_callist2 = (gradx*grnds.x+grady*grnds.y+gradz*grnds.z);

		grnds.x = (d1-d0)/fy;
		grnds.y = -d0/fx;
		grnds.z = -ax*grnds.x - ay*grnds.y - d0/(fx*fy);
		sh_callist3 = (gradx*grnds.x+grady*grnds.y+gradz*grnds.z);

		float4 retval;			
		retval.w = sh_callist0;
		retval.x = sh_callist1;
		retval.y = sh_callist2;
		retval.z = sh_callist3;

		return retval;
	}else
	{
		float4 retval;
		retval.x = 0.0f;retval.y = 0.0f;retval.z = 0.0f;retval.w = 0.0f;
		return retval;
	}
}



__inline__ __device__ void PreComputeGrad_LS(int tidx,int tidy,int goffx, int goffy, volatile float* inX, PatchSolverInput &input, 
										  volatile float* outGradx,volatile float* outGrady,volatile float* outGradz, volatile float* outShadingdif)
{
	goffx -= 1;
	goffy -= 1;

	int idxtmp	= tidy*PATCH_SIZE+tidx;
	int mp_x	= idxtmp%(PATCH_SIZE+3);
	int mp_y	= idxtmp/(PATCH_SIZE+3);	
	if((mp_y+goffy)>=0 && (mp_y+goffy)<input.height && (goffx+mp_x)>=0 && (goffx+mp_x)<input.width)
	{
		float4 value = calShading2depthGrad(inX, mp_x-1, mp_y-1, mp_x+goffx,mp_y+goffy, input);		
		outGradx[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= value.x;
		outGrady[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= value.y;
		outGradz[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= value.z;
		outShadingdif[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]	= value.w;	
	}else
	{
		outGradx[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= 0;
		outGrady[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= 0;
		outGradz[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= 0;
		outShadingdif[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]	= 0;	
	}


	idxtmp	      += PATCH_SIZE*PATCH_SIZE;
	mp_x	      =  idxtmp%(PATCH_SIZE+3);
	mp_y	      =  idxtmp/(PATCH_SIZE+3);	
	if(idxtmp<((PATCH_SIZE+3)*(PATCH_SIZE+3)) ){
		if( (mp_y+goffy)>=0 && (mp_y+goffy)<input.height && (goffx+mp_x)>=0 && (goffx+mp_x)<input.width )
		{
		float4 value = calShading2depthGrad(inX, mp_x-1, mp_y-1, mp_x+goffx,mp_y+goffy, input);		
		outGradx[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= value.x;
		outGrady[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= value.y;
		outGradz[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= value.z;
		outShadingdif[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]	= value.w;		
		}else
		{
		outGradx[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= 0;
		outGrady[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= 0;
		outGradz[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]		= 0;
		outShadingdif[getLinearShareMemLocateLS_SFS(mp_y-1,mp_x-1)]	= 0;	
		}
	}
}


__device__ inline float add_mul_inp_grad_ls(volatile float *inP, volatile float *inGradx,volatile float *inGrady, volatile float *inGradz, int tidx,int tidy)
{
	float retval = 0.0f;	
	if(!IsOutsidePatch(tidy,tidx-1))	
		retval += inP[getLinearThreadId(tidy, tidx-1)] * readValueFromCache2DLS_SFS(inGradx, tidy, tidx);	
	if(!IsOutsidePatch(tidy,tidx))	
		retval += inP[getLinearThreadId(tidy, tidx)] * readValueFromCache2DLS_SFS(inGrady, tidy, tidx);		
	if(!IsOutsidePatch(tidy-1,tidx))	
		retval += inP[getLinearThreadId(tidy-1, tidx)] * readValueFromCache2DLS_SFS(inGradz, tidy, tidx);

	return retval;

}

__device__ inline float add_mul_inp_grad_ls_bsp(volatile float *inP, volatile float *inGradx,volatile float *inGrady, volatile float *inGradz, int tidx,int tidy)
{	
	return (readValueFromCache2D_SFS(inP,tidy  , tidx-1)	* readValueFromCache2DLS_SFS(inGradx, tidy, tidx) 
		  + readValueFromCache2D_SFS(inP,tidy  , tidx  )	* readValueFromCache2DLS_SFS(inGrady, tidy, tidx) 
	   	  + readValueFromCache2D_SFS(inP,tidy-1, tidx  )	* readValueFromCache2DLS_SFS(inGradz, tidy, tidx));	
}


__device__ inline float3 estimate_normal_from_depth2(float *inPriorDepth, int gidx, int gidy, int W, int H, float ax, float ay, const float &fx, const float &fy)
{
	float3 retval;
	retval.x = 0.0f;
	retval.y = 0.0f;
	retval.z = 0.0f;

	const float d0 = inPriorDepth[gidy*W+gidx-1];
	const float d1 = inPriorDepth[gidy*W+gidx];
	const float d2 = inPriorDepth[(gidy-1)*W+gidx];

	if(IsValidPoint(d0) && IsValidPoint(d1) && IsValidPoint(d2) ){
		retval.x = - d2*(d0-d1)/fy;
		retval.y = - d0*(d2-d1)/fx;
		retval.z = -ay*retval.y - ax*retval.x - d2*d0/fx/fy;			
		float an = sqrt( retval.x * retval.x + retval.y * retval.y + retval.z * retval.z );
		if(an!=0)
		{
			retval.x /= an; 
			retval.y /= an; 
			retval.z /= an;
		}
	}

	return retval;
}

__device__ inline void prior_normal_from_previous_depth(float d,int gidx, int gidy, PatchSolverInput &input,float3 &normal0,float3 &normal1, float3 &normal2)
{
	const float fx = input.calibparams.fx;
	const float fy = input.calibparams.fy;
	const float ux = input.calibparams.ux;
	const float uy = input.calibparams.uy;
	const float ufx = 1.0f / input.calibparams.fx;
	const float ufy = 1.0f / input.calibparams.fy;
	
	int W = input.width;
	int H = input.height;
	
	float4 position_prev = input.deltaTransform*make_float4((gidx-ux)*d/fx, (gidy-uy)*d/fy,d,1.0f);

	if(!IsValidPoint(d) || position_prev.z ==0.0f)
	{
		normal0 = make_float3(0.0f,0.0f,0.0f);
		normal1 = make_float3(0.0f,0.0f,0.0f);
		normal2 = make_float3(0.0f,0.0f,0.0f);
		return ;	
	}

	int posx = (int)(fx*position_prev.x/position_prev.z + ux +0.5f);
	int posy = (int)(fy*position_prev.y/position_prev.z + uy +0.5f);

	if(posx<2 || posx>(W-3) || posy<2 || posy>(H-3))
	{
		normal0 = make_float3(0.0f,0.0f,0.0f);
		normal1 = make_float3(0.0f,0.0f,0.0f);
		normal2 = make_float3(0.0f,0.0f,0.0f);
		return ;	
	}

	float ax = (posx-ux)/fx;
	float ay = (posy-uy)/fy;

	normal0 = estimate_normal_from_depth2(input.d_depthMapRefinedLastFrameFloat, posx, posy, W,H, ax, ay, fx, fy);
	normal1 = estimate_normal_from_depth2(input.d_depthMapRefinedLastFrameFloat, posx+1, posy, W, H, ax+ufx, ay, fx, fy);
	normal2 = estimate_normal_from_depth2(input.d_depthMapRefinedLastFrameFloat, posx, posy+1, W, H,ax, ay+ufy, fx, fy);

	return ;	
}

////////////////////////////////////////
// evalCost
////////////////////////////////////////
__inline__ __device__ float evalCost(int tidy, int tidx, int posy, int posx, unsigned int W, unsigned int H, volatile float* inGradx, volatile float* inGrady, volatile float* inGradz,
    volatile float* inShadingdif, volatile float* inX, volatile unsigned char* inMaskRow, volatile unsigned char* inMaskCol, float3 &normal0, float3 &normal1, float3 &normal2,//volatile float* inPriorDepth,
    PatchSolverInput& input, PatchSolverParameters& parameters) {
    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;
    const float ufx = 1.0f / input.calibparams.fx;
    const float ufy = 1.0f / input.calibparams.fy;


    // Common stuff

    const float targetDepth = input.d_targetDepth[posy*W + posx]; const bool validTarget = IsValidPoint(targetDepth);
    const float XC = readValueFromCache2D_SFS(inX, tidy, tidx);


    float cost = 0.0f;

    float E_s   = 0.0f;
    float E_p   = 0.0f;
    float E_r_h = 0.0f;
    float E_r_v = 0.0f;
    float E_r_d = 0.0f;
    float E_g_v = 0.0f;
    float E_g_h = 0.0f;

    if (validTarget)
    {
        if (posx>1 && posx<(W - 5) && posy>1 && posy<(H - 5)){

            float sum, tmpval;
            float val0, val1, val2;
            unsigned char maskval = 1;
#           if USE_SHADING_CONSTRAINT
            val0 = readValueFromCache2DLS_SFS(inGrady, tidy, tidx);
            val1 = readValueFromCache2DLS_SFS(inGradx, tidy, tidx + 1);
            val2 = readValueFromCache2DLS_SFS(inGradz, tidy + 1, tidx);

#               ifdef USE_MASK_REFINE
                    E_g_h = (readValueFromCache2DLS_SFS(inShadingdif, tidy, tidx) - readValueFromCache2DLS_SFS(inShadingdif, tidy, tidx+1)) * readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy, tidx);
                    E_g_v = (readValueFromCache2DLS_SFS(inShadingdif, tidy, tidx) - readValueFromCache2DLS_SFS(inShadingdif, tidy+1, tidx)) * readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy, tidx);
#               else
                    E_g_h = readValueFromCache2DLS_SFS(inShadingdif, tidy, tidx) - readValueFromCache2DLS_SFS(inShadingdif, tidy, tidx+1);
                    E_g_v = readValueFromCache2DLS_SFS(inShadingdif, tidy, tidx) - readValueFromCache2DLS_SFS(inShadingdif, tidy+1, tidx);
#               endif	
#           endif
            //////////////////////////////////////////////////////////////////
            //                   smoothness term
            /////////////////////////////////////////////////////////////////
#           if USE_REGULARIZATION

                float d = readValueFromCache2D_SFS(inX, tidy, tidx);
                float d0 = readValueFromCache2D_SFS(inX, tidy, tidx - 1);
                float d1 = readValueFromCache2D_SFS(inX, tidy, tidx + 1);
                float d2 = readValueFromCache2D_SFS(inX, tidy - 1, tidx);
                float d3 = readValueFromCache2D_SFS(inX, tidy + 1, tidx);

                if (IsValidPoint(d) && IsValidPoint(d0) && IsValidPoint(d1) && IsValidPoint(d2) && IsValidPoint(d3)
                    && abs(d - d0)<DEPTH_DISCONTINUITY_THRE
                    && abs(d - d1)<DEPTH_DISCONTINUITY_THRE
                    && abs(d - d2)<DEPTH_DISCONTINUITY_THRE
                    && abs(d - d3)<DEPTH_DISCONTINUITY_THRE)
                {
                    E_s = sqMagnitude(4.0f*point(d, posx, posy, input) - (point(d1, posx + 1, posy, input) + point(d0, posx - 1, posy, input) + point(d3, posx, posy + 1, input) + point(d2, posx, posy - 1, input)));
                }
#           endif


#           if USE_DEPTH_CONSTRAINT						
                //position term 			
                E_p = XC - targetDepth;
#           endif



            //////////////////////////////////////////////////////////////////
            //                   piror term
            /////////////////////////////////////////////////////////////////
            //first: calculate the normal for PriorDepth
#           if USE_TEMPORAL_CONSTRAINT
            // TODO: implement
#           endif
            cost =  (parameters.weightRegularizer   * E_s*E_s) + // This is usually on the order of 1/10,000,000
                    (parameters.weightFitting       * E_p*E_p) + // 
                    (parameters.weightShading       * (E_g_h*E_g_h + E_g_v*E_g_v)) + 
                    (parameters.weightPrior * (E_r_h*E_r_h + E_r_v*E_r_v + E_r_d*E_r_d));
            //cost = readValueFromCache2DLS_SFS(inShadingdif, tidy, tidx);
        }
    }


    return cost;
}


////////////////////////////////////////
// evalMinusJTF
////////////////////////////////////////

__inline__ __device__ float evalMinusJTFDeviceLS_SFS_Shared_Mask_Prior(int tidy, int tidx, int posy, int posx, unsigned int W, unsigned int H, volatile float* inGradx, volatile float* inGrady,volatile float* inGradz,
														  volatile float* inShadingdif, volatile float* inX, volatile unsigned char* inMaskRow, volatile unsigned char* inMaskCol, float3 &normal0, float3 &normal1, float3 &normal2,//volatile float* inPriorDepth,
														  PatchSolverInput& input, PatchSolverParameters& parameters, float& outPre)
{
	const float fx = input.calibparams.fx;
	const float fy = input.calibparams.fy;
	const float ux = input.calibparams.ux;
	const float uy = input.calibparams.uy;
	const float ufx = 1.0f / input.calibparams.fx;
	const float ufy = 1.0f / input.calibparams.fy;


	float b = 0.0f; float p = 0.0f;

	// Common stuff

	const float targetDepth = input.d_targetDepth[posy*W+posx]; const bool validTarget = IsValidPoint(targetDepth);
	const float XC = readValueFromCache2D_SFS(inX, tidy, tidx);

		
	if(validTarget) 
	{
		if(posx>1 && posx<(W-5) && posy>1 && posy<(H-5)){
			
			float sum, tmpval;
			float val0,val1,val2;
			unsigned char maskval = 1;			
#           if USE_SHADING_CONSTRAINT
			    val0 = readValueFromCache2DLS_SFS(inGrady,tidy  ,tidx  );
			    val1 = readValueFromCache2DLS_SFS(inGradx,tidy  ,tidx+1);
			    val2 = readValueFromCache2DLS_SFS(inGradz,tidy+1,tidx  );
					

#               ifdef USE_MASK_REFINE
			        //calculating residue error
			        //shading term
			        //row edge constraint
			        sum = 0.0f;	
			        tmpval = 0.0f;			
			        tmpval = -(readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx-1) - readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx));// edge 0				
			        maskval = readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy, tidx-1);
			        sum += tmpval*(-val0) * maskval;//(posy, posx-1)*val0,(posy,posx)*(-val0)			
			        tmpval	+= val0*val0*maskval;

			        tmpval = -(readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx) - readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx+1));//edge 2				
			        maskval = readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy, tidx);
			        sum += tmpval*(val0-val1) * maskval;// (posy,posx)*(val1-val0), (posy,posx+1)*(val0-val1)			
			        tmpval	+= (val0-val1)*(val0-val1)* maskval;

			        tmpval = -(readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx+1) - readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx+2));//edge 4				
			        maskval = readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy, tidx+1);
			        sum += tmpval*(val1) * maskval;//(posy,posx+1)*(-val1), (posy,posx+2)*(val1)			
			        tmpval	+= val1*val1* maskval;

			        tmpval = -(readValueFromCache2DLS_SFS(inShadingdif,tidy+1,tidx-1) - readValueFromCache2DLS_SFS(inShadingdif,tidy+1,tidx));//edge 5				
			        maskval = readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy+1, tidx-1);
			        sum += tmpval*(-val2) * maskval;//(posy+1,posx-1)*(val2),(posy+1,pox)*(-val2)			
			        tmpval	+= val2*val2* maskval;

			        tmpval = -(readValueFromCache2DLS_SFS(inShadingdif,tidy+1,tidx) - readValueFromCache2DLS_SFS(inShadingdif,tidy+1,tidx+1));//edge 7				
			        maskval = readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy+1, tidx);
			        sum += tmpval*val2 * maskval;//(posy+1,posx)*(-val2),(posy+1,posx+1)*(val2)
			        tmpval	+= val2*val2 * maskval;
						
			        //column edge constraint			
			        tmpval = -(readValueFromCache2DLS_SFS(inShadingdif,tidy-1,tidx) - readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx));//edge 1
			        maskval = readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy-1, tidx);
			        sum += tmpval*(-val0) * maskval;//(posy-1,posx)*(val0),(posy,posx)*(-val0)			
			        tmpval	+= val0*val0* maskval;

			        tmpval = -(readValueFromCache2DLS_SFS(inShadingdif,tidy-1,tidx+1) - readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx+1));//edge 3
			        maskval = readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy-1, tidx+1);
			        sum += tmpval*(-val1) * maskval;//(posy-1,posx+1)*(val1),(posy,posx+1)*(-val1)			
			        tmpval	+= val1*val1* maskval;

			        tmpval = -(readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx) - readValueFromCache2DLS_SFS(inShadingdif,tidy+1,tidx));//edge 6
			        maskval = readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy, tidx);
			        sum += tmpval*(val0-val2) * maskval;//(posy,posx)*(val2-val0),(posy+1,posx)*(val0-val2)			
			        tmpval	+= (val0-val2)*(val0-val2)* maskval;

			        tmpval = -(readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx+1) - readValueFromCache2DLS_SFS(inShadingdif,tidy+1,tidx+1));//edge 8
			        maskval = readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy, tidx+1);
			        sum += tmpval*val1 * maskval;//(posy,posx+1)*(-val1),(posy+1,posx+1)*(val1)
			        tmpval	+= val1*val1* maskval;

			        tmpval = -(readValueFromCache2DLS_SFS(inShadingdif,tidy+1,tidx) - readValueFromCache2DLS_SFS(inShadingdif,tidy+2,tidx));//edge 9
			        maskval = readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy+1, tidx);
			        sum += tmpval*val2 * maskval;//(posy+1,posx)*(-val2),(posy+2,posx)*(val2)
			        tmpval	+= val2*val2* maskval;

			        b += sum * parameters.weightShading;
			        p += tmpval * parameters.weightShading;//shading constraint
#               else
			        tmpval = 0.0f;
			        tmpval = val0 * val0 * 2;
			        tmpval += (val0 - val1) * (val0 - val1);
			        tmpval += (val0 - val2) * (val0 - val2);
			        tmpval += val1 * val1 * 3;
			        tmpval += val2 * val2 * 3;
			        p += tmpval * parameters.weightShading;//shading constraint


			        sum = 0.0f;			
			        sum += val0*readValueFromCache2DLS_SFS(inShadingdif,tidy-1,tidx);								
			        sum += val1 * readValueFromCache2DLS_SFS(inShadingdif,tidy-1,tidx+1);								
			        sum += val0 * readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx-1);					
			        sum += (-val0+val1-val0-val0+val2-val0) * readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx);					
			        sum += (val0-val1-val1-val1-val1) * readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx+1);					
			        sum += val1 * readValueFromCache2DLS_SFS(inShadingdif,tidy,tidx+2);					
			        sum += val2 * readValueFromCache2DLS_SFS(inShadingdif,tidy+1,tidx-1);						
			        sum +=  (-val2-val2+val0-val2-val2) * readValueFromCache2DLS_SFS(inShadingdif,tidy+1,tidx);						
			        sum += (val2+val1) * readValueFromCache2DLS_SFS(inShadingdif,tidy+1,tidx+1);					
			        sum += val2  * readValueFromCache2DLS_SFS(inShadingdif,tidy+2,tidx);
			
                    b += sum * parameters.weightShading;
#               endif	
#           endif
					

			//////////////////////////////////////////////////////////////////
			//                   smoothness term
			/////////////////////////////////////////////////////////////////
#           if USE_REGULARIZATION
			    bool b_valid = true;
		
			    val0 = (posx - ux)/fx;			
			    val1 = (posy - uy)/fy;					

			    //smoothness term							
			    float3 lapval = est_lap_init_3d_imp(inX, tidx,tidy,val0,val1,ufx,ufy,b_valid);
			    sum =  0.0f;
			    sum += lapval.x*val0*(-4.0f);
			    sum += lapval.y*val1*(-4.0f);
			    sum += lapval.z*(-4.0f);
									
			    lapval = est_lap_init_3d_imp(inX, tidx-1,tidy,val0-ufx,val1,ufx,ufy,b_valid);
			    sum += lapval.x*val0;
			    sum += lapval.y*val1;
			    sum += lapval.z;
									
			    lapval = est_lap_init_3d_imp(inX, tidx+1,tidy,val0+ufx,val1,ufx,ufy,b_valid);
			    sum += lapval.x*val0;
			    sum += lapval.y*val1;
			    sum += lapval.z;
									
			    lapval = est_lap_init_3d_imp(inX, tidx,tidy-1,val0,val1-ufy,ufx,ufy,b_valid);
			    sum += lapval.x*val0;
			    sum += lapval.y*val1;
			    sum += lapval.z;
									
			    lapval = est_lap_init_3d_imp(inX, tidx,tidy+1,val0,val1+ufy,ufx,ufy,b_valid);
			    sum += lapval.x*val0;
			    sum += lapval.y*val1;
			    sum += lapval.z;
				
			    if(b_valid)
			    {
				    b += sum*parameters.weightRegularizer;					
				    tmpval = (val0 * val0 + val1 * val1 + 1)*(16+4);						
				    p += tmpval *parameters.weightRegularizer;//smoothness
			    }
#           endif
			

#           if USE_DEPTH_CONSTRAINT						
			    //position term 			
			    p += parameters.weightFitting;//position constraint			
			    b += -(XC - targetDepth*DEPTH_RESCALE) * parameters.weightFitting;
#           endif



			//////////////////////////////////////////////////////////////////
			//                   piror term
			/////////////////////////////////////////////////////////////////
			//first: calculate the normal for PriorDepth
#           if USE_TEMPORAL_CONSTRAINT
                float d;
			    sum = 0.0f;
			    float ax = (posx-ux)/fx;
			    float ay = (posy-uy)/fy;		
			
			    tmpval = normal0.x * ax + normal0.y * ay + normal0.z ;// derative of prior energy wrt depth			
			    p += tmpval * tmpval * 2  * parameters.weightPrior;
			    d =  readValueFromCache2D_SFS(inX, tidy, tidx-1);
			    if(IsValidPoint(d))
				    sum -= tmpval * ( tmpval * readValueFromCache2D_SFS(inX, tidy, tidx) + ( -tmpval + normal0.x/fx) * d );
			    d = readValueFromCache2D_SFS(inX, tidy-1, tidx);
			    if(IsValidPoint(d))
				    sum -= tmpval * ( tmpval * readValueFromCache2D_SFS(inX, tidy, tidx) + ( -tmpval + normal0.y/fy) * d );
					
			    tmpval = normal1.x * ax + normal1.y * ay + normal1.z ;// derative of prior energy wrt depth			
			    p += tmpval * tmpval * parameters.weightPrior;
			    d = readValueFromCache2D_SFS(inX, tidy, tidx+1);
			    if(IsValidPoint(d))
				    sum -= -tmpval * ( ( tmpval + normal1.x/fx) * d - tmpval * readValueFromCache2D_SFS(inX, tidy, tidx));
						
			    tmpval = normal2.x * ax + normal2.y * ay + normal2.z ;// derative of prior energy wrt depth
			    p += tmpval * tmpval * parameters.weightPrior;
			    d = readValueFromCache2D_SFS(inX, tidy+1, tidx);
			    if(IsValidPoint(d))
				    sum -= -tmpval * ( ( tmpval + normal2.y/fy) * d - tmpval * readValueFromCache2D_SFS(inX, tidy, tidx));
			
			    b += sum  * parameters.weightPrior;
#           endif
			
		}
	}


	if(p > FLOAT_EPSILON) outPre = 1.0f/p;
	else			      outPre = 1.0f;
#if USE_PRECONDITIONER == 0
    outPre = 1.0f;
#endif
	return b;
}







////////////////////////////////////////
// applyJTJ
////////////////////////////////////////



__inline__ __device__ float applyJTJDeviceLS_SFS_Shared_BSP_Mask_Prior(int tidy, int tidx, int posy, int posx, unsigned int W, unsigned int H, volatile float* inGradx,volatile float* inGrady, volatile float* inGradz,
													  volatile float* inP,  volatile unsigned char* inMaskRow, volatile unsigned char* inMaskCol, float3 &normal0, float3 &normal1, float3 &normal2, //volatile float* inPriorDepth,
													  PatchSolverInput& input, PatchSolverParameters& parameters)
{
	const float fx = input.calibparams.fx;
	const float fy = input.calibparams.fy;
	const float ux = input.calibparams.ux;
	const float uy = input.calibparams.uy;
	const float ufx = 1.0f / input.calibparams.fx;
	const float ufy = 1.0f / input.calibparams.fy;

	float b = 0.0f;	

	const float targetDepth = input.d_targetDepth[posy*W+posx]; const bool validTarget = IsValidPoint(targetDepth);
	const float PC = inP[getLinearShareMemLocate_SFS(tidy, tidx)];
		
	if(validTarget)
	{
		if((posx>1) && (posx<(W-5)) && (posy>1) && (posy<(H-5)))
		{				

			float sum = 0.0f;
			float tmpval = 0.0f;			
			
            float val0, val1, val2;
#           if USE_SHADING_CONSTRAINT
			    val0 = readValueFromCache2DLS_SFS(inGrady,tidy  ,tidx  );
			    val1 = readValueFromCache2DLS_SFS(inGradx,tidy  ,tidx+1);
			    val2 = readValueFromCache2DLS_SFS(inGradz,tidy+1,tidx  );		
						
#               ifdef USE_MASK_REFINE
			        readValueFromCache2D_SFS(inP,tidy  , tidx)	* readValueFromCache2DLS_SFS(inGradx, tidy, tidx) ; // Doesn't do anything?!

			        //the following is the adding of the relative edge constraints to the sum
			        //-val0, edge 0			
			        tmpval  = readValueFromCache2D_SFS(inP,tidy  , tidx-2) *  readValueFromCache2DLS_SFS(inGradx, tidy  , tidx-1);
			        tmpval += readValueFromCache2D_SFS(inP,tidy  , tidx-1) * (readValueFromCache2DLS_SFS(inGrady, tidy  , tidx-1) - readValueFromCache2DLS_SFS(inGradx, tidy  , tidx));
			        tmpval += readValueFromCache2D_SFS(inP,tidy-1, tidx-1) *  readValueFromCache2DLS_SFS(inGradz, tidy  , tidx-1);
			        tmpval -= readValueFromCache2D_SFS(inP,tidy  , tidx  ) *  readValueFromCache2DLS_SFS(inGrady, tidy  , tidx  );
			        tmpval -= readValueFromCache2D_SFS(inP,tidy-1, tidx  ) *  readValueFromCache2DLS_SFS(inGradz, tidy  , tidx  );			
			        sum += (-val0) * tmpval  * readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy, tidx-1);
			
			        //-val0, edge 1
			        tmpval  = readValueFromCache2D_SFS(inP,tidy-1, tidx-1) *  readValueFromCache2DLS_SFS(inGradx, tidy-1, tidx  );
			        tmpval += readValueFromCache2D_SFS(inP,tidy-1, tidx  ) * (readValueFromCache2DLS_SFS(inGrady, tidy-1, tidx  ) - readValueFromCache2DLS_SFS(inGradz, tidy  , tidx));
			        tmpval += readValueFromCache2D_SFS(inP,tidy-2, tidx  ) *  readValueFromCache2DLS_SFS(inGradz, tidy-1, tidx  );
			        tmpval -= readValueFromCache2D_SFS(inP,tidy  , tidx-1) *  readValueFromCache2DLS_SFS(inGradx, tidy  , tidx  );
			        tmpval -= readValueFromCache2D_SFS(inP,tidy  , tidx  ) *  readValueFromCache2DLS_SFS(inGrady, tidy  , tidx  );		
			        sum += (-val0) * tmpval  * readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy-1, tidx);

			        //val0-val1, edge 2
			        tmpval  = readValueFromCache2D_SFS(inP,tidy  , tidx-1) *  readValueFromCache2DLS_SFS(inGradx, tidy  , tidx  );
			        tmpval += readValueFromCache2D_SFS(inP,tidy  , tidx  ) * (readValueFromCache2DLS_SFS(inGrady, tidy  , tidx  ) - readValueFromCache2DLS_SFS(inGradx, tidy  , tidx+1));
			        tmpval += readValueFromCache2D_SFS(inP,tidy-1, tidx  ) *  readValueFromCache2DLS_SFS(inGradz, tidy  , tidx  );
			        tmpval -= readValueFromCache2D_SFS(inP,tidy  , tidx+1) *  readValueFromCache2DLS_SFS(inGrady, tidy  , tidx+1);
			        tmpval -= readValueFromCache2D_SFS(inP,tidy-1, tidx+1) *  readValueFromCache2DLS_SFS(inGradz, tidy  , tidx+1);		
			        sum += (val0-val1) * tmpval * readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy, tidx);

			        //-val1, edge 3			
			        tmpval  = readValueFromCache2D_SFS(inP,tidy-1, tidx  ) *  readValueFromCache2DLS_SFS(inGradx, tidy-1, tidx+1);
			        tmpval += readValueFromCache2D_SFS(inP,tidy-1, tidx+1) * (readValueFromCache2DLS_SFS(inGrady, tidy-1, tidx+1) - readValueFromCache2DLS_SFS(inGradz, tidy  , tidx+1));
			        tmpval += readValueFromCache2D_SFS(inP,tidy-2, tidx+1) *  readValueFromCache2DLS_SFS(inGradz, tidy-1, tidx+1);
			        tmpval -= readValueFromCache2D_SFS(inP,tidy  , tidx  ) *  readValueFromCache2DLS_SFS(inGradx, tidy  , tidx+1);
			        tmpval -= readValueFromCache2D_SFS(inP,tidy  , tidx+1) *  readValueFromCache2DLS_SFS(inGrady, tidy  , tidx+1);		
			        sum += (-val1) * tmpval	* readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy-1, tidx+1);

			        //val1, edge 4
			        tmpval  = readValueFromCache2D_SFS(inP,tidy  , tidx  ) *  readValueFromCache2DLS_SFS(inGradx, tidy  , tidx+1);
			        tmpval += readValueFromCache2D_SFS(inP,tidy  , tidx+1) * (readValueFromCache2DLS_SFS(inGrady, tidy  , tidx+1) - readValueFromCache2DLS_SFS(inGradx, tidy  , tidx+2));
			        tmpval += readValueFromCache2D_SFS(inP,tidy-1, tidx+1) *  readValueFromCache2DLS_SFS(inGradz, tidy  , tidx+1);
			        tmpval -= readValueFromCache2D_SFS(inP,tidy  , tidx+2) *  readValueFromCache2DLS_SFS(inGrady, tidy  , tidx+2);
			        tmpval -= readValueFromCache2D_SFS(inP,tidy-1, tidx+2) *  readValueFromCache2DLS_SFS(inGradz, tidy  , tidx+2);		
			        sum += (val1) * tmpval * readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy, tidx+1);

			        //-val2, edge 5			
			        tmpval  = readValueFromCache2D_SFS(inP,tidy+1, tidx-2) *  readValueFromCache2DLS_SFS(inGradx, tidy+1, tidx-1);
			        tmpval += readValueFromCache2D_SFS(inP,tidy+1, tidx-1) * (readValueFromCache2DLS_SFS(inGrady, tidy+1, tidx-1) - readValueFromCache2DLS_SFS(inGradx, tidy+1, tidx));
			        tmpval += readValueFromCache2D_SFS(inP,tidy  , tidx-1) *  readValueFromCache2DLS_SFS(inGradz, tidy+1, tidx-1);
			        tmpval -= readValueFromCache2D_SFS(inP,tidy+1, tidx  ) *  readValueFromCache2DLS_SFS(inGrady, tidy+1, tidx  );
			        tmpval -= readValueFromCache2D_SFS(inP,tidy  , tidx  ) *  readValueFromCache2DLS_SFS(inGradz, tidy+1, tidx  );		
			        sum += (-val2) * tmpval * readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy+1, tidx-1);
			
			        //val0-val2, edge 6
			        tmpval  = readValueFromCache2D_SFS(inP,tidy  , tidx-1) *  readValueFromCache2DLS_SFS(inGradx, tidy  , tidx  );
			        tmpval += readValueFromCache2D_SFS(inP,tidy  , tidx  ) * (readValueFromCache2DLS_SFS(inGrady, tidy  , tidx  ) - readValueFromCache2DLS_SFS(inGradz, tidy+1, tidx));
			        tmpval += readValueFromCache2D_SFS(inP,tidy-1, tidx  ) *  readValueFromCache2DLS_SFS(inGradz, tidy  , tidx  );
			        tmpval -= readValueFromCache2D_SFS(inP,tidy+1, tidx-1) *  readValueFromCache2DLS_SFS(inGradx, tidy+1, tidx  );
			        tmpval -= readValueFromCache2D_SFS(inP,tidy+1, tidx  ) *  readValueFromCache2DLS_SFS(inGrady, tidy+1, tidx  );		
			        sum += (val0-val2) * tmpval * readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy, tidx);

			        //val2, edge 7
			        tmpval  = readValueFromCache2D_SFS(inP,tidy+1, tidx-1) *  readValueFromCache2DLS_SFS(inGradx, tidy+1, tidx  );
			        tmpval += readValueFromCache2D_SFS(inP,tidy+1, tidx  ) * (readValueFromCache2DLS_SFS(inGrady, tidy+1, tidx  ) - readValueFromCache2DLS_SFS(inGradx, tidy+1, tidx+1));
			        tmpval += readValueFromCache2D_SFS(inP,tidy  , tidx  ) *  readValueFromCache2DLS_SFS(inGradz, tidy+1, tidx  );
			        tmpval -= readValueFromCache2D_SFS(inP,tidy+1, tidx+1) *  readValueFromCache2DLS_SFS(inGrady, tidy+1, tidx+1);
			        tmpval -= readValueFromCache2D_SFS(inP,tidy  , tidx+1) *  readValueFromCache2DLS_SFS(inGradz, tidy+1, tidx+1);		
			        sum += val2 * tmpval * readValueFromCache2DLS_SFS_MASK(inMaskRow, tidy+1, tidx);

			        //val1, edge 8
			        tmpval  = readValueFromCache2D_SFS(inP,tidy  , tidx  ) *  readValueFromCache2DLS_SFS(inGradx, tidy  , tidx+1);
			        tmpval += readValueFromCache2D_SFS(inP,tidy  , tidx+1) * (readValueFromCache2DLS_SFS(inGrady, tidy  , tidx+1) - readValueFromCache2DLS_SFS(inGradz, tidy+1, tidx+1));
			        tmpval += readValueFromCache2D_SFS(inP,tidy-1, tidx+1) *  readValueFromCache2DLS_SFS(inGradz, tidy  , tidx+1);
			        tmpval -= readValueFromCache2D_SFS(inP,tidy+1, tidx  ) *  readValueFromCache2DLS_SFS(inGradx, tidy+1, tidx+1);
			        tmpval -= readValueFromCache2D_SFS(inP,tidy+1, tidx+1) *  readValueFromCache2DLS_SFS(inGrady, tidy+1, tidx+1);		
			        sum += val1 * tmpval * readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy, tidx+1);

			        //val2, edge 9
			        tmpval  = readValueFromCache2D_SFS(inP,tidy+1, tidx-1) *  readValueFromCache2DLS_SFS(inGradx, tidy+1, tidx  );
			        tmpval += readValueFromCache2D_SFS(inP,tidy+1, tidx  ) * (readValueFromCache2DLS_SFS(inGrady, tidy+1, tidx  ) - readValueFromCache2DLS_SFS(inGradz, tidy+2, tidx));
			        tmpval += readValueFromCache2D_SFS(inP,tidy  , tidx  ) *  readValueFromCache2DLS_SFS(inGradz, tidy+1, tidx  );
			        tmpval -= readValueFromCache2D_SFS(inP,tidy+2, tidx-1) *  readValueFromCache2DLS_SFS(inGradx, tidy+2, tidx  );
			        tmpval -= readValueFromCache2D_SFS(inP,tidy+2, tidx  ) *  readValueFromCache2DLS_SFS(inGrady, tidy+2, tidx  );		
			        sum += val2 * tmpval * readValueFromCache2DLS_SFS_MASK(inMaskCol, tidy+1, tidx);

			        b += sum * parameters.weightShading;

#               else											
			        sum += (val1*4.0f-val0) * add_mul_inp_grad_ls_bsp(inP,inGradx,inGrady,inGradz,tidx+1,tidy);//mulitplication of grad with inP needs to consid			
			        sum += (val2*4.0f-val0) * add_mul_inp_grad_ls_bsp(inP,inGradx,inGrady,inGradz,tidx,tidy+1);							
			        sum += (val0*4.0f-val1-val2) * add_mul_inp_grad_ls_bsp(inP,inGradx,inGrady,inGradz,tidx,tidy);					
			        sum += (-val2-val1) * add_mul_inp_grad_ls_bsp(inP,inGradx,inGrady,inGradz,tidx+1,tidy+1);					
			        sum += (-val0) * add_mul_inp_grad_ls_bsp(inP,inGradx,inGrady,inGradz,tidx-1,tidy);								
			        sum += (-val1) * add_mul_inp_grad_ls_bsp(inP,inGradx,inGrady,inGradz,tidx+2,tidy);							
			        sum += (-val0) * add_mul_inp_grad_ls_bsp(inP,inGradx,inGrady,inGradz,tidx,tidy-1);			
			        sum += (-val1) *  add_mul_inp_grad_ls_bsp(inP,inGradx,inGrady,inGradz,tidx+1,tidy-1);				
			        sum += (-val2) *  add_mul_inp_grad_ls_bsp(inP,inGradx,inGrady,inGradz,tidx-1,tidy+1);				
			        sum += (-val2) *  add_mul_inp_grad_ls_bsp(inP,inGradx,inGrady,inGradz,tidx,tidy+2);	
			        b += sum * parameters.weightShading;
#               endif
#           endif

						
			//////////////////////////////////////////////////////////////////
			//                  Smoothness Term
			/////////////////////////////////////////////////////////////////
#           if USE_REGULARIZATION				
			    sum = 0;	
			    val0 = (posx - ux)/fx;
			    val1 = (posy - uy)/fy;
			
			    float3 lapval = est_lap_3d_bsp_imp(inP,tidx,tidy,val0,val1,ufx,ufy);			
			    sum += lapval.x*val0*(4.0f);
			    sum += lapval.y*val1*(4.0f);
			    sum += lapval.z*(4.0f);
						
			    lapval = est_lap_3d_bsp_imp(inP,tidx-1,tidy,val0-ufx,val1,ufx,ufy);
			    sum -= lapval.x*val0;
			    sum -= lapval.y*val1;
			    sum -= lapval.z;
						
			    lapval = est_lap_3d_bsp_imp(inP,tidx+1,tidy,val0+ufx,val1,ufx,ufy);
			    sum -= lapval.x*val0;
			    sum -= lapval.y*val1;
			    sum -= lapval.z;
						
			    lapval = est_lap_3d_bsp_imp(inP,tidx,tidy-1,val0,val1-ufy,ufx,ufy);
			    sum -= lapval.x*val0;
			    sum -= lapval.y*val1;
			    sum -= lapval.z;
						
			    lapval = est_lap_3d_bsp_imp(inP,tidx,tidy+1,val0,val1+ufy,ufx,ufy);
			    sum -= lapval.x*val0;
			    sum -= lapval.y*val1;
			    sum -= lapval.z;


                b += sum*parameters.weightRegularizer;
#           endif

			//////////////////////////////////////////////////////////////////
			//                  Position Term
			/////////////////////////////////////////////////////////////////		

#           if USE_DEPTH_CONSTRAINT
			    b += PC*parameters.weightFitting;
#           endif


			//////////////////////////////////////////////////////////////////
			//                   piror term
			/////////////////////////////////////////////////////////////////			
#           if USE_TEMPORAL_CONSTRAINT
			    sum = 0.0f;
			    float ax = (posx-ux)/fx;
			    float ay = (posy-uy)/fy;			
			    tmpval = normal0.x * ax + normal0.y * ay + normal0.z ;// derative of prior energy wrt depth			
			    sum += tmpval * ( tmpval * readValueFromCache2D_SFS(inP, tidy, tidx) + ( -tmpval + normal0.x/fx) * readValueFromCache2D_SFS(inP, tidy, tidx-1) );
			    sum += tmpval * ( tmpval * readValueFromCache2D_SFS(inP, tidy, tidx) + ( -tmpval + normal0.y/fy) * readValueFromCache2D_SFS(inP, tidy-1, tidx) );
						
			    tmpval = normal1.x * ax + normal1.y * ay + normal1.z ;// derative of prior energy wrt depth			
			    sum += -tmpval * ( ( tmpval + normal1.x/fx) * readValueFromCache2D_SFS(inP, tidy, tidx+1) - tmpval * readValueFromCache2D_SFS(inP, tidy, tidx));
						
			    tmpval = normal2.x * ax + normal2.y * ay + normal2.z ;// derative of prior energy wrt depth			
			    sum += -tmpval * ( ( tmpval + normal2.y/fy) * readValueFromCache2D_SFS(inP, tidy+1, tidx) - tmpval * readValueFromCache2D_SFS(inP, tidy, tidx));

			    b += sum  * parameters.weightPrior;				
#           endif
		}				
	}
		
	
	return b;
}

#endif
