#pragma once

#ifndef _PATCH_SOLVER_SFS_UTIL_
#define _PATCH_SOLVER_SFS_UTIL_

#include "../SolverUtil.h"

#define PATCH_SIZE				  (16) // keep consistent with CPU
#define SHARED_MEM_SIZE_PATCH	  ((PATCH_SIZE+2)*(PATCH_SIZE+2))
#define SHARED_MEM_SIZE_VARIABLES ((PATCH_SIZE)*(PATCH_SIZE))
#define SHARED_MEM_SIZE_RESIDUUMS ((SHARED_MEM_SIZE_VARIABLES)+4*(SHARED_MEM_SIZE_VARIABLES))

#define EXTRA_BD 2
#define SHARED_MEM_SIZE_PATCH_SFS	  ((PATCH_SIZE+4)*(PATCH_SIZE+4))
#define SHARED_MEM_SIZE_PATCH_SFS_GE	((PATCH_SIZE+EXTRA_BD*2-1)*(PATCH_SIZE+EXTRA_BD*2-1))

/////////////////////////////////////////////////////////////////////////
// Helper
/////////////////////////////////////////////////////////////////////////

__inline__ __device__ bool IsValidPoint(float d)
{
	return ((d != MINF)&&(d>0));
}

__inline__ __device__ unsigned int getLinearThreadId(int tId_i, int tId_j)
{
	return tId_i*PATCH_SIZE+tId_j;
}

__inline__ __device__ bool IsOutsidePatch(int tId_i, int tId_j)
{
	return ((tId_i<0) || (tId_i>=PATCH_SIZE) || (tId_j<0) || (tId_j>=PATCH_SIZE));
}

__inline__ __device__ unsigned int getLinearShareMemLocate_SFS(int tId_i, int tId_j)
{
	return (tId_i+2)*(PATCH_SIZE+4)+(tId_j+2);
}

__inline__ __device__ unsigned int getLinearShareMemLocateLS_SFS(int tId_i, int tId_j)
{
	return (tId_i+1)*(PATCH_SIZE+3)+(tId_j+1);
}

__inline__ __device__ float readValueFromCache2D(volatile float* cache, int tId_i, int tId_j)
{
	return cache[(tId_i+1)*(PATCH_SIZE+2)+(tId_j+1)];
}

__inline__ __device__ float readValueFromCache2D_SFS(volatile float* cache, int tId_i, int tId_j)
{
	return cache[(tId_i+2)*(PATCH_SIZE+4)+(tId_j+2)];
}

__inline__ __device__ float readValueFromCache2DLS_SFS(volatile float* cache, int tId_i, int tId_j)
{
	return cache[(tId_i+1)*(PATCH_SIZE+3)+(tId_j+1)];
}

__inline__ __device__ unsigned char readValueFromCache2DLS_SFS_MASK(volatile unsigned char* cache, int tId_i, int tId_j)
{
	return cache[(tId_i+1)*(PATCH_SIZE+2)+(tId_j+1)];
}

__inline__ __device__ void loadVariableToCache(volatile float* cache, float* data, int tId_i, int tId_j, int gId_i, int gId_j, unsigned int W, unsigned int H)
{
	cache[(tId_i+1)*(PATCH_SIZE+2)+(tId_j+1)] = data[gId_i*W+gId_j];
}

__inline__ __device__ void loadPatchToCache2(volatile float* cache, float* data, int tidy, int tidx, int goffy, int goffx, unsigned int W, unsigned int H)
{
	goffy -= 1;
	goffx -= 1;

	unsigned int idxtmp	= tidy*PATCH_SIZE+tidx;
	unsigned int mp_x	= idxtmp%(PATCH_SIZE+2);
	unsigned int mp_y	= idxtmp/(PATCH_SIZE+2);
	unsigned int corpos = (mp_y+goffy)*W + goffx+mp_x;
	//if((mp_y+goffy)<H && (goffx+mp_x)<W)
	if((mp_y+goffy)>=0 && (goffx+mp_x)>=0 & (mp_y+goffy)<H && (goffx+mp_x)<W)
		cache[idxtmp]       = data[corpos]*DEPTH_RESCALE;
	else
		cache[idxtmp] = MINF;


	idxtmp	      += PATCH_SIZE*PATCH_SIZE;
	mp_x	      =  idxtmp%(PATCH_SIZE+2);
	mp_y	      =  idxtmp/(PATCH_SIZE+2);
	corpos		  =  (mp_y+goffy)*W + goffx+mp_x;
	if(idxtmp<((PATCH_SIZE+2)*(PATCH_SIZE+2)) && (mp_y+goffy)<H && (goffx+mp_x)<W && (mp_y+goffy)>=0 && (goffx+mp_x)>=0)
		cache[idxtmp] =  data[corpos]*DEPTH_RESCALE;
	else
		cache[idxtmp] = MINF;
}

__inline__ __device__ void loadPatchToCache_SFS(volatile float* cache, float* data, int tidy, int tidx, int goffy, int goffx, unsigned int W, unsigned int H)
{
	goffx -= 2;
	goffy -= 2;

	int idxtmp	= tidy*PATCH_SIZE+tidx;
	int mp_x	= idxtmp%(PATCH_SIZE+4);
	int mp_y	= idxtmp/(PATCH_SIZE+4);
	int corpos = (mp_y+goffy)*W + goffx+mp_x;
	if((mp_y+goffy)>=0 && (mp_y+goffy)<H && (goffx+mp_x)>=0 && (goffx+mp_x)<W)
		cache[idxtmp]       = data[corpos] * DEPTH_RESCALE;
	else
		cache[idxtmp] = MINF;

	idxtmp	      += PATCH_SIZE*PATCH_SIZE;
	mp_x	      =  idxtmp%(PATCH_SIZE+4);
	mp_y	      =  idxtmp/(PATCH_SIZE+4);
	corpos		  =  (mp_y+goffy)*W + goffx+mp_x;
	if(idxtmp<((PATCH_SIZE+4)*(PATCH_SIZE+4)) ){
		if((mp_y+goffy)>=0 && (mp_y+goffy)<H && (goffx+mp_x)>=0 && (goffx+mp_x)<W)
			cache[idxtmp] =  data[corpos] * DEPTH_RESCALE;
		else
			cache[idxtmp] = MINF;
	}
}

__inline__ __device__ void loadMaskToCache_SFS(volatile unsigned char* cachrow, volatile unsigned char* cachcol, unsigned char* data, int tidy, int tidx, int goffy, int goffx, unsigned int W, unsigned int H)
{
	goffx -= 1;
	goffy -= 1;

	int idxtmp	= tidy*PATCH_SIZE+tidx;
	int mp_x	= idxtmp%(PATCH_SIZE+2);
	int mp_y	= idxtmp/(PATCH_SIZE+2);	
	int corpos = (mp_y+goffy)*W + goffx+mp_x;
	if((mp_y+goffy)>=0 && (mp_y+goffy)<H && (goffx+mp_x)>=0 && (goffx+mp_x)<W)
	{
		cachrow[idxtmp]     = data[corpos];
		cachcol[idxtmp]		= data[W*H+corpos];	
	}else
	{
		cachrow[idxtmp] = 0;
		cachcol[idxtmp] = 0;
	}


	idxtmp	      += PATCH_SIZE*PATCH_SIZE;
	mp_x	      =  idxtmp%(PATCH_SIZE+2);
	mp_y	      =  idxtmp/(PATCH_SIZE+2);	
	corpos = (mp_y+goffy)*W + goffx+mp_x;
	if(idxtmp<((PATCH_SIZE+2)*(PATCH_SIZE+2)) ){
		if( (mp_y+goffy)>=0 && (mp_y+goffy)<H && (goffx+mp_x)>=0 && (goffx+mp_x)<W )
		{
			cachrow[idxtmp]     = data[corpos];
			cachcol[idxtmp]		= data[W*H+corpos];	
		}else
		{
			cachrow[idxtmp] = 0;
			cachcol[idxtmp] = 0;
		}
	}
}

__inline__ __device__ int accessJPReg(int tId_i, int tId_j, unsigned int offset)
{
	return SHARED_MEM_SIZE_VARIABLES+4*getLinearThreadId(tId_i, tId_j)+offset;
}

__inline__ __device__ bool isOnBoundary(int tId_i, int tId_j)
{
	return (tId_i<0 || tId_i>=PATCH_SIZE || tId_j<0 || tId_j>=PATCH_SIZE);
}


__inline__ __device__ void SetPatchToZero_SFS(volatile float* cache, float value, int tidy, int tidx, int goffy, int goffx)
{
	goffx -= 2;
	goffy -= 2;

	unsigned int idxtmp	= tidy*PATCH_SIZE+tidx;
	cache[idxtmp]       = value;

	idxtmp	      += PATCH_SIZE*PATCH_SIZE;	
	if(idxtmp<((PATCH_SIZE+4)*(PATCH_SIZE+4)))
		cache[idxtmp] =  value;
}


__inline__ __device__ float3 point(float d, int posx, int posy, PatchSolverInput& input) {
    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;
    return make_float3((((float)posx - ux) / fx)*d, (((float)posy - uy) / fy)*d, d);

}

__inline__ __device__ float sqMagnitude(float3 in) {
    return in.x*in.x + in.y*in.y + in.z*in.z;
}


__inline__ __device__ float4 calShading2depthGradHelper(const float d0, const float d1, const float d2, int posx, int posy, PatchSolverInput &input)
{


    const int imgind = posy * input.width + posx;

    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;


    if ((IsValidPoint(d0)) && (IsValidPoint(d1)) && (IsValidPoint(d2)))
    {
        const float greyval = (input.d_targetIntensity[imgind] * 0.5f + input.d_targetIntensity[imgind - 1] * 0.25f + input.d_targetIntensity[imgind - input.width] * 0.25f)*RGB_RANGE_SCALE;

        float ax = (posx - ux) / fx;
        float ay = (posy - uy) / fy;
        float an, an2;

        float px, py, pz;
        px = d2*(d1 - d0) / fy;
        py = d0*(d1 - d2) / fx;
        pz = -ax*px - ay*py - d2*d0 / (fx*fy);
        an2 = px*px + py*py + pz*pz;
        an = sqrt(an2);
        if (an == 0)
        {
            float4 retval;
            retval.x = 0.0f; retval.y = 0.0f; retval.z = 0.0f; retval.w = 0.0f;
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
        float sh_callist6 = ((-px*px - py*py + 2 * pz*pz))*input.d_litcoeff[6];
        float sh_callist7 = pz * px * input.d_litcoeff[7];
        float sh_callist8 = (px * px - py * py)*input.d_litcoeff[8];


        //normal changes wrt depth
        register float gradx = 0, grady = 0, gradz = 0;


        gradx += -sh_callist1*px;
        gradx += -sh_callist2*px;
        gradx += (input.d_litcoeff[3] - sh_callist3*px);
        gradx += py*input.d_litcoeff[4] - sh_callist4 * 2 * px;
        gradx += -sh_callist5 * 2 * px;
        gradx += (-2 * px)*input.d_litcoeff[6] - sh_callist6 * 2 * px;
        gradx += pz*input.d_litcoeff[7] - sh_callist7 * 2 * px;
        gradx += 2 * px*input.d_litcoeff[8] - sh_callist8 * 2 * px;
        gradx /= an;


        grady += (input.d_litcoeff[1] - sh_callist1*py);
        grady += -sh_callist2*py;
        grady += -sh_callist3*py;
        grady += px*input.d_litcoeff[4] - sh_callist4 * 2 * py;
        grady += pz*input.d_litcoeff[5] - sh_callist5 * 2 * py;
        grady += (-2 * py)*input.d_litcoeff[6] - sh_callist6 * 2 * py;
        grady += -sh_callist7 * 2 * py;
        grady += (-2 * py)*input.d_litcoeff[8] - sh_callist8 * 2 * py;
        grady /= an;

        gradz += -sh_callist1*pz;
        gradz += (input.d_litcoeff[2] - sh_callist2*pz);
        gradz += -sh_callist3*pz;
        gradz += -sh_callist4 * 2 * pz;
        gradz += py*input.d_litcoeff[5] - sh_callist5 * 2 * pz;
        gradz += 4 * pz*input.d_litcoeff[6] - sh_callist6 * 2 * pz;
        gradz += px*input.d_litcoeff[7] - sh_callist7 * 2 * pz;
        gradz += -sh_callist8 * 2 * pz;
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

        grnds.x = -d2 / fy;
        grnds.y = (d1 - d2) / fx;
        grnds.z = -ax*grnds.x - ay*grnds.y - d2 / (fx*fy);
        sh_callist1 = (gradx*grnds.x + grady*grnds.y + gradz*grnds.z);

        grnds.x = d2 / fy;
        grnds.y = d0 / fx;
        grnds.z = -ax*grnds.x - ay*grnds.y;
        sh_callist2 = (gradx*grnds.x + grady*grnds.y + gradz*grnds.z);

        grnds.x = (d1 - d0) / fy;
        grnds.y = -d0 / fx;
        grnds.z = -ax*grnds.x - ay*grnds.y - d0 / (fx*fy);
        sh_callist3 = (gradx*grnds.x + grady*grnds.y + gradz*grnds.z);

        float4 retval;
        retval.w = sh_callist0;
        retval.x = sh_callist1;
        retval.y = sh_callist2;
        retval.z = sh_callist3;

        return retval;
    }
    else
    {
        float4 retval;
        retval.x = 0.0f; retval.y = 0.0f; retval.z = 0.0f; retval.w = 0.0f;
        return retval;
    }
}

#endif
