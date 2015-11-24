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


#endif
