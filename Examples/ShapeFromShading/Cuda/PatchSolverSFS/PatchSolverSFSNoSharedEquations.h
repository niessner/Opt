#ifndef PatchSolverSFSNoSharedEquations_h
#define PatchSolverSFSNoSharedEquations_h


#include <cutil_inline.h>
#include <cutil_math.h>

#include "PatchSolverSFSUtil.h"
#include "PatchSolverSFSState.h"
#include "PatchSolverSFSParameters.h"

#include "SolverTerms.h"

#define DEPTH_DISCONTINUITY_THRE 0.01

__inline__ __device__ float readX(PatchSolverState& state, int posy, int posx, int W) {
    return state.d_x[posy*W + posx];
}

__inline__ __device__ float rowMask(int posy, int posx, PatchSolverInput &input) {
    return input.d_maskEdgeMap[posy*input.width + posx];
}

__inline__ __device__ float colMask(int posy, int posx, PatchSolverInput &input) {
    return input.d_maskEdgeMap[posy*input.width + posx + (input.width*input.height)];
}


__inline__ __device__ float4 calShading2depthGradNoShared(PatchSolverState& state, int posx, int posy, PatchSolverInput &input)
{
    const int W = input.width;
    const float d0 = readX(state, posy, posx - 1, W);
    const float d1 = readX(state, posy, posx, W);
    const float d2 = readX(state, posy - 1, posx, W);

    return calShading2depthGradHelper(d0, d1, d2, posx, posy, input);
}

__device__ inline float3 est_lap_init_3d_impNoShared(PatchSolverState& state, int posx, int posy, const float w0, const float w1, const float &ufx, const float &ufy, const int W, bool &b_valid)
{
    float3 retval;
    retval.x = 0.0f;
    retval.y = 0.0f;
    retval.z = 0.0f;
    float d = readX(state, posy, posx, W);
    float d0 = readX(state, posy, posx - 1, W);
    float d1 = readX(state, posy, posx + 1, W);
    float d2 = readX(state, posy - 1, posx, W);
    float d3 = readX(state, posy + 1, posx, W);

    if (IsValidPoint(d) && IsValidPoint(d0) && IsValidPoint(d1) && IsValidPoint(d2) && IsValidPoint(d3)
        && abs(d - d0)<DEPTH_DISCONTINUITY_THRE
        && abs(d - d1)<DEPTH_DISCONTINUITY_THRE
        && abs(d - d2)<DEPTH_DISCONTINUITY_THRE
        && abs(d - d3)<DEPTH_DISCONTINUITY_THRE)

    {
        retval.x = d * w0 * 4;
        retval.y = d * w1 * 4;
        retval.z = d * 4;

        retval.x -= d0*(w0 - ufx);
        retval.y -= d0*w1;
        retval.z -= d0;

        retval.x -= d1*(w0 + ufx);
        retval.y -= d1*w1;
        retval.z -= d1;

        retval.x -= d2*w0;
        retval.y -= d2*(w1 - ufy);
        retval.z -= d2;

        retval.x -= d3*w0;
        retval.y -= d3*(w1 + ufy);
        retval.z -= d3;

    }
    else
        b_valid = false;

    return retval;
}

__inline__ __device__ float evalMinusJTFDeviceNoShared(int posy, int posx, unsigned int W, unsigned int H, PatchSolverState& state, PatchSolverInput& input, PatchSolverParameters& parameters, float& outPre) {
    int variableIdx = posy*W + posx;

    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;
    const float ufx = 1.0f / input.calibparams.fx;
    const float ufy = 1.0f / input.calibparams.fy;


    float b = 0.0f; float p = 0.0f;

    // Common stuff
    const float targetDepth = input.d_targetDepth[variableIdx];
    const bool validTarget = IsValidPoint(targetDepth);
    const float XC = state.d_x[variableIdx];
    if (validTarget)
    {
        if (posx>1 && posx<(W - 5) && posy>1 && posy<(H - 5)){

            float sum, tmpval;
            float val0, val1, val2;
            unsigned char maskval = 1;
#           if USE_SHADING_CONSTRAINT
            val0 = calShading2depthGradNoShared(state, posx, posy, input).y;
            val1 = calShading2depthGradNoShared(state, posx + 1, posy, input).x;
            val2 = calShading2depthGradNoShared(state, posx, posy + 1, input).z;


#               ifdef USE_MASK_REFINE
            //calculating residue error
            //shading term
            //row edge constraint
            sum = 0.0f;
            tmpval = 0.0f;
            tmpval = -(calShading2depthGradNoShared(state, posx - 1, posy, input).w - calShading2depthGradNoShared(state, posx, posy, input).w);// edge 0				
            maskval = rowMask(posy, posx - 1, input);
            sum += tmpval*(-val0) * maskval;//(posy, posx-1)*val0,(posy,posx)*(-val0)			
            tmpval += val0*val0*maskval;

            tmpval = -(calShading2depthGradNoShared(state, posx, posy, input).w - calShading2depthGradNoShared(state, posx + 1, posy, input).w);//edge 2				
            maskval = rowMask(posy, posx, input);
            sum += tmpval*(val0 - val1) * maskval;// (posy,posx)*(val1-val0), (posy,posx+1)*(val0-val1)			
            tmpval += (val0 - val1)*(val0 - val1)* maskval;

            tmpval = -(calShading2depthGradNoShared(state, posx + 1, posy, input).w - calShading2depthGradNoShared(state, posx + 2, posy, input).w);//edge 4				
            maskval = rowMask(posy, posx + 1, input);
            sum += tmpval*(val1)* maskval;//(posy,posx+1)*(-val1), (posy,posx+2)*(val1)			
            tmpval += val1*val1* maskval;

            tmpval = -(calShading2depthGradNoShared(state, posx - 1, posy + 1, input).w - calShading2depthGradNoShared(state, posx, posy + 1, input).w);//edge 5				
            maskval = rowMask(posy + 1, posx - 1, input);
            sum += tmpval*(-val2) * maskval;//(posy+1,posx-1)*(val2),(posy+1,pox)*(-val2)			
            tmpval += val2*val2* maskval;

            tmpval = -(calShading2depthGradNoShared(state, posx, posy + 1, input).w - calShading2depthGradNoShared(state, posx + 1, posy + 1, input).w);//edge 7				
            maskval = rowMask(posy + 1, posx, input);
            sum += tmpval*val2 * maskval;//(posy+1,posx)*(-val2),(posy+1,posx+1)*(val2)
            tmpval += val2*val2 * maskval;

            //column edge constraint			
            tmpval = -(calShading2depthGradNoShared(state, posx, posy - 1, input).w - calShading2depthGradNoShared(state, posx, posy, input).w);//edge 1
            maskval = colMask(posy - 1, posx, input);
            sum += tmpval*(-val0) * maskval;//(posy-1,posx)*(val0),(posy,posx)*(-val0)			
            tmpval += val0*val0* maskval;

            tmpval = -(calShading2depthGradNoShared(state, posx + 1, posy - 1, input).w - calShading2depthGradNoShared(state, posx + 1, posy, input).w);//edge 3
            maskval = colMask(posy - 1, posx + 1, input);
            sum += tmpval*(-val1) * maskval;//(posy-1,posx+1)*(val1),(posy,posx+1)*(-val1)			
            tmpval += val1*val1* maskval;

            tmpval = -(calShading2depthGradNoShared(state, posx, posy, input).w - calShading2depthGradNoShared(state, posx, posy + 1, input).w);//edge 6
            maskval = colMask(posy, posx, input);
            sum += tmpval*(val0 - val2) * maskval;//(posy,posx)*(val2-val0),(posy+1,posx)*(val0-val2)			
            tmpval += (val0 - val2)*(val0 - val2)* maskval;

            tmpval = -(calShading2depthGradNoShared(state, posx + 1, posy, input).w - calShading2depthGradNoShared(state, posx + 1, posy + 1, input).w);//edge 8
            maskval = colMask(posy, posx + 1, input);
            sum += tmpval*val1 * maskval;//(posy,posx+1)*(-val1),(posy+1,posx+1)*(val1)
            tmpval += val1*val1* maskval;

            tmpval = -(calShading2depthGradNoShared(state, posx, posy + 1, input).w - calShading2depthGradNoShared(state, posx, posy + 2, input).w);//edge 9
            maskval = colMask(posy + 1, posx, input);
            sum += tmpval*val2 * maskval;//(posy+1,posx)*(-val2),(posy+2,posx)*(val2)
            tmpval += val2*val2* maskval;

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
            sum += val0*calShading2depthGradNoShared(state, posx, posy - 1, input).w;
            sum += val1 * calShading2depthGradNoShared(state, posx + 1, posy - 1, input).w;
            sum += val0 * calShading2depthGradNoShared(state, posx - 1, posy, input).w;
            sum += (-val0 + val1 - val0 - val0 + val2 - val0) * calShading2depthGradNoShared(state, posx, posy, input).w;
            sum += (val0 - val1 - val1 - val1 - val1) * calShading2depthGradNoShared(state, posx + 1, posy, input).w;
            sum += val1 * calShading2depthGradNoShared(state, posx + 2, posy, input).w;
            sum += val2 * calShading2depthGradNoShared(state, posx - 1, posy + 1, input).w;
            sum += (-val2 - val2 + val0 - val2 - val2) * calShading2depthGradNoShared(state, posx, posy + 1, input).w;
            sum += (val2 + val1) * calShading2depthGradNoShared(state, posx + 1, posy + 1, input).w;
            sum += val2  * calShading2depthGradNoShared(state, posx, posy + 2, input).w;

            b += sum * parameters.weightShading;
#               endif	
#           endif


            //////////////////////////////////////////////////////////////////
            //                   smoothness term
            /////////////////////////////////////////////////////////////////
#           if USE_REGULARIZATION
            bool b_valid = true;

            val0 = (posx - ux) / fx;
            val1 = (posy - uy) / fy;

            //smoothness term							
            float3 lapval = est_lap_init_3d_impNoShared(state, posx, posy, val0, val1, ufx, ufy, W, b_valid);
            sum = 0.0f;
            sum += lapval.x*val0*(-4.0f);
            sum += lapval.y*val1*(-4.0f);
            sum += lapval.z*(-4.0f);

            lapval = est_lap_init_3d_impNoShared(state, posx - 1, posy, val0 - ufx, val1, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;

            lapval = est_lap_init_3d_impNoShared(state, posx + 1, posy, val0 + ufx, val1, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;

            lapval = est_lap_init_3d_impNoShared(state, posx, posy - 1, val0, val1 - ufy, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;

            lapval = est_lap_init_3d_impNoShared(state, posx, posy + 1, val0, val1 + ufy, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;

            if (b_valid)
            {
                b += sum*parameters.weightRegularizer;
                tmpval = (val0 * val0 + val1 * val1 + 1)*(16 + 4);
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
            float ax = (posx - ux) / fx;
            float ay = (posy - uy) / fy;

            tmpval = normal0.x * ax + normal0.y * ay + normal0.z;// derative of prior energy wrt depth			
            p += tmpval * tmpval * 2 * parameters.weightPrior;
            d = readValueFromCache2D_SFS(inX, tidy, tidx - 1);
            if (IsValidPoint(d))
                sum -= tmpval * (tmpval * readValueFromCache2D_SFS(inX, tidy, tidx) + (-tmpval + normal0.x / fx) * d);
            d = readValueFromCache2D_SFS(inX, tidy - 1, tidx);
            if (IsValidPoint(d))
                sum -= tmpval * (tmpval * readValueFromCache2D_SFS(inX, tidy, tidx) + (-tmpval + normal0.y / fy) * d);

            tmpval = normal1.x * ax + normal1.y * ay + normal1.z;// derative of prior energy wrt depth			
            p += tmpval * tmpval * parameters.weightPrior;
            d = readValueFromCache2D_SFS(inX, tidy, tidx + 1);
            if (IsValidPoint(d))
                sum -= -tmpval * ((tmpval + normal1.x / fx) * d - tmpval * readValueFromCache2D_SFS(inX, tidy, tidx));

            tmpval = normal2.x * ax + normal2.y * ay + normal2.z;// derative of prior energy wrt depth
            p += tmpval * tmpval * parameters.weightPrior;
            d = readValueFromCache2D_SFS(inX, tidy + 1, tidx);
            if (IsValidPoint(d))
                sum -= -tmpval * ((tmpval + normal2.y / fy) * d - tmpval * readValueFromCache2D_SFS(inX, tidy, tidx));

            b += sum  * parameters.weightPrior;
#           endif

        }
    }


    if (p > FLOAT_EPSILON) outPre = 1.0f / p;
    else			      outPre = 1.0f;
#if USE_PRECONDITIONER == 0
    outPre = 1.0f;
#endif
    return b;
}
#endif