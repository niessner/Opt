#include <cutil_inline.h>
#define DEPTH_DISCONTINUITY_THRE 0.01
__inline__ __device__ float readX(SolverState& state, int posy, int posx, int W) {return state.d_x[posy*W + posx];}
__inline__ __device__ float readP(int posy, int posx, SolverState& state, int W) {return state.d_p[posy*W + posx];}
__inline__ __device__ float rowMask(int posy, int posx, SolverInput &input) {return float(input.d_maskEdgeMap[posy*input.width + posx]);}
__inline__ __device__ float colMask(int posy, int posx, SolverInput &input) {return float(input.d_maskEdgeMap[posy*input.width + posx + (input.width*input.height)]);}
__inline__ __device__ float4 calShading2depthGradHelper(const float d0, const float d1, const float d2, int posx, int posy, SolverInput &input) {
    const int imgind = posy * input.width + posx;
    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;
    if ((IsValidPoint(d0)) && (IsValidPoint(d1)) && (IsValidPoint(d2))) {
        const float greyval = (input.d_targetIntensity[imgind] * 0.5f + input.d_targetIntensity[imgind - 1] * 0.25f + input.d_targetIntensity[imgind - input.width] * 0.25f);
        float ax = (posx - ux) / fx;
        float ay = (posy - uy) / fy;
        float an, an2;
        float px, py, pz;
        px = d2*(d1 - d0) / fy;
        py = d0*(d1 - d2) / fx;
        pz = -ax*px - ay*py - d2*d0 / (fx*fy);
        an2 = px*px + py*py + pz*pz;
        an = sqrt(an2);
        if (an == 0) {
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
        sh_callist0 += sh_callist1;
        sh_callist0 += sh_callist2;
        sh_callist0 += sh_callist3;
        sh_callist0 += sh_callist4;
        sh_callist0 += sh_callist5;
        sh_callist0 += sh_callist6;
        sh_callist0 += sh_callist7;
        sh_callist0 += sh_callist8;
        sh_callist0 -= greyval;
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
    } else {
        float4 retval;
        retval.x = 0.0f; retval.y = 0.0f; retval.z = 0.0f; retval.w = 0.0f;
        return retval;
    }
}
__inline__ __device__ float4 calShading2depthGradCompute(SolverState& state, int posx, int posy, SolverInput &input) {
    const int W = input.width;
    const float d0 = readX(state, posy, posx - 1, W);
    const float d1 = readX(state, posy, posx, W);
    const float d2 = readX(state, posy - 1, posx, W);
    return calShading2depthGradHelper(d0, d1, d2, posx, posy, input);
}
__global__ void Precompute_Kernel(SolverInput input, SolverState state, SolverParameters parameters) {
    const unsigned int N = input.N; // Number of block variables
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < N) {
        int W = input.width;
        int posy; int posx; get2DIdx(x, input.width, input.height, posy, posx);
        float4 temp = calShading2depthGradCompute(state, posx, posy, input);
        state.B_I_dx0[x] = temp.x;
        state.B_I_dx1[x] = temp.y;
        state.B_I_dx2[x] = temp.z;
        state.B_I[x]     = temp.w;    
        float d  = readX(state, posy, posx, W);
        float d0 = readX(state, posy, posx - 1, W);
        float d1 = readX(state, posy, posx + 1, W);
        float d2 = readX(state, posy - 1, posx, W);
        float d3 = readX(state, posy + 1, posx, W);
        state.pguard[x] = 
           IsValidPoint(d) && IsValidPoint(d0) && IsValidPoint(d1) && IsValidPoint(d2) && IsValidPoint(d3)
            && abs(d - d0)<DEPTH_DISCONTINUITY_THRE
            && abs(d - d1)<DEPTH_DISCONTINUITY_THRE
            && abs(d - d2)<DEPTH_DISCONTINUITY_THRE
            && abs(d - d3)<DEPTH_DISCONTINUITY_THRE;
    }
}
__inline__ __device__ float4 calShading2depthGrad(SolverState& state, int x, int y, SolverInput &inp){
    return make_float4(state.B_I_dx0[y*inp.width+x],state.B_I_dx1[y*inp.width+x],state.B_I_dx2[y*inp.width+x],state.B_I[y*inp.width+x]);
}
__inline__ __device__ float evalFDevice(int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
    int posy; int posx; get2DIdx(variableIdx, input.width, input.height, posy, posx);
    const int W = input.width;
    const int H = input.height;
    const float targetDepth = input.d_targetDepth[variableIdx]; const bool validTarget = IsValidPoint(targetDepth);
    const float XC = state.d_x[variableIdx];
    float cost = 0.0f;
    float3 E_s = make_float3(0.0f, 0.0f, 0.0f);
    float E_p = 0.0f;
    float E_g_v = 0.0f;
    float E_g_h = 0.0f;
    if (validTarget) {
        if (posx>1 && posx<(W - 5) && posy>1 && posy<(H - 5)) {
            float4 temp00 = calShading2depthGrad(state, posx, posy, input);
            float4 temp10 = calShading2depthGrad(state, posx+1, posy, input);
            float4 temp01 = calShading2depthGrad(state, posx, posy+1, input);
            E_g_h = (temp00.w - temp10.w) * input.d_maskEdgeMap[variableIdx];
            E_g_v = (temp00.w - temp01.w) * input.d_maskEdgeMap[variableIdx + W*H];
            float d  = readX(state, posy, posx, W);
            float d0 = readX(state, posy, posx - 1, W);
            float d1 = readX(state, posy, posx + 1, W);
            float d2 = readX(state, posy - 1, posx, W);
            float d3 = readX(state, posy + 1, posx, W);
            if (state.pguard[posy*W+posx]) 
                E_s = 4.0f*point(d, posx, posy, input) - (point(d1, posx + 1, posy, input) + point(d0, posx - 1, posy, input) + point(d3, posx, posy + 1, input) + point(d2, posx, posy - 1, input));	
            E_p = XC - targetDepth;
            cost = (parameters.weightRegularizer   * sqMagnitude(E_s)) + (parameters.weightFitting       * E_p*E_p) + (parameters.weightShading       * (E_g_h*E_g_h + E_g_v*E_g_v)) + (parameters.weightPrior * (E_r_h*E_r_h + E_r_v*E_r_v + E_r_d*E_r_d));
        }
    }
    return cost;
}
__device__ inline float3 est_lap_init_3d_imp(SolverState& state, int posx, int posy, const float w0, const float w1, const float &ufx, const float &ufy, const int W, bool &b_valid) {
    float3 retval;
    retval.x = 0.0f;
    retval.y = 0.0f;
    retval.z = 0.0f;
    if (state.pguard[posy*W+posx]) {
        float d = readX(state, posy, posx, W);
        float d0 = readX(state, posy, posx - 1, W);
        float d1 = readX(state, posy, posx + 1, W);
        float d2 = readX(state, posy - 1, posx, W);
        float d3 = readX(state, posy + 1, posx, W);
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
    } else
        b_valid = false;
    return retval;
}
__device__ inline float3 est_lap_3d_bsp_imp_with_guard(SolverState& state, int posx, int posy, float w0, float w1, const float &ufx, const float &ufy, const int W) {
    float3 retval = make_float3(0.0f, 0.0f, 0.0f); 
    if (state.pguard[posy*W+posx]) {
        const float p = readP(posy, posx, state, W);
        const float p0 = readP(posy, posx - 1, state, W);
        const float p1 = readP(posy, posx + 1, state, W);
        const float p2 = readP(posy - 1, posx, state, W);
        const float p3 = readP(posy + 1, posx, state, W);   
        retval.x = ( p * 4 * w0 - p0 * (w0 - ufx) - p1 * (w0 + ufx)   - p2 * w0 - p3 * w0);
        retval.y = ( p * 4 * w1 - p0 * w1 - p1 * w1 - p2 * (w1 - ufy) - p3 * (w1 + ufy));
        retval.z = ( p * 4 - p0 - p1 - p2 - p3);
    } 
    return retval;
}
__device__ inline float3 est_lap_3d_bsp_imp(SolverState& state, int posx, int posy, float w0, float w1, const float &ufx, const float &ufy, const int W) {
    float3 retval;
    const float d = readP(posy, posx, state, W);
    const float d0 = readP(posy, posx - 1, state, W);
    const float d1 = readP(posy, posx + 1, state, W);
    const float d2 = readP(posy - 1, posx, state, W);
    const float d3 = readP(posy + 1, posx, state, W);
    retval.x = (d * 4 * w0 - d0 * (w0 - ufx) - d1 * (w0 + ufx) - d2 * w0 - d3 * w0);
    retval.y = (d * 4 * w1 - d0 * w1 - d1 * w1 - d2 * (w1 - ufy) - d3 * (w1 + ufy));
    retval.z = (d * 4 - d0 - d1 - d2 - d3);
    return retval;
}
__inline__ __device__ float evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float& pre) {
    int posy; int posx; get2DIdx(variableIdx, input.width, input.height, posy, posx);
    const int W = input.width;
    const int H = input.height;
    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;
    const float ufx = 1.0f / input.calibparams.fx;
    const float ufy = 1.0f / input.calibparams.fy;
    float b = 0.0f; float p = 0.0f;
    const float targetDepth = input.d_targetDepth[variableIdx]; 
    const bool validTarget  = IsValidPoint(targetDepth);
    const float XC = state.d_x[variableIdx];
    if (validTarget) {
        if (posx>1 && posx<(W - 5) && posy>1 && posy<(H - 5)){
            float sum, tmpval;
            float val0, val1, val2;
            unsigned char maskval = 1;
            val0 = calShading2depthGrad(state, posx, posy, input).y;
            val1 = calShading2depthGrad(state, posx + 1, posy, input).x;
            val2 = calShading2depthGrad(state, posx, posy + 1, input).z;
            sum = 0.0f;
            tmpval = 0.0f;
            tmpval = -(calShading2depthGrad(state, posx - 1, posy, input).w - calShading2depthGrad(state, posx, posy, input).w);// edge 0				
            maskval = rowMask(posy, posx - 1, input);
            sum += tmpval*(-val0) * maskval;//(posy, posx-1)*val0,(posy,posx)*(-val0)			
            tmpval += val0*val0*maskval;
            tmpval = -(calShading2depthGrad(state, posx, posy, input).w - calShading2depthGrad(state, posx + 1, posy, input).w);//edge 2				
            maskval = rowMask(posy, posx, input);
            sum += tmpval*(val0 - val1) * maskval;// (posy,posx)*(val1-val0), (posy,posx+1)*(val0-val1)			
            tmpval += (val0 - val1)*(val0 - val1)* maskval;
            tmpval = -(calShading2depthGrad(state, posx + 1, posy, input).w - calShading2depthGrad(state, posx + 2, posy, input).w);//edge 4				
            maskval = rowMask(posy, posx + 1, input);
            sum += tmpval*(val1)* maskval;//(posy,posx+1)*(-val1), (posy,posx+2)*(val1)			
            tmpval += val1*val1* maskval;
            tmpval = -(calShading2depthGrad(state, posx - 1, posy + 1, input).w - calShading2depthGrad(state, posx, posy + 1, input).w);//edge 5				
            maskval = rowMask(posy + 1, posx - 1, input);
            sum += tmpval*(-val2) * maskval;//(posy+1,posx-1)*(val2),(posy+1,pox)*(-val2)			
            tmpval += val2*val2* maskval;
            tmpval = -(calShading2depthGrad(state, posx, posy + 1, input).w - calShading2depthGrad(state, posx + 1, posy + 1, input).w);//edge 7				
            maskval = rowMask(posy + 1, posx, input);
            sum += tmpval*val2 * maskval;//(posy+1,posx)*(-val2),(posy+1,posx+1)*(val2)
            tmpval += val2*val2 * maskval;		
            tmpval = -(calShading2depthGrad(state, posx, posy - 1, input).w - calShading2depthGrad(state, posx, posy, input).w);//edge 1
            maskval = colMask(posy - 1, posx, input);
            sum += tmpval*(-val0) * maskval;//(posy-1,posx)*(val0),(posy,posx)*(-val0)			
            tmpval += val0*val0* maskval;
            tmpval = -(calShading2depthGrad(state, posx + 1, posy - 1, input).w - calShading2depthGrad(state, posx + 1, posy, input).w);//edge 3
            maskval = colMask(posy - 1, posx + 1, input);
            sum += tmpval*(-val1) * maskval;//(posy-1,posx+1)*(val1),(posy,posx+1)*(-val1)			
            tmpval += val1*val1* maskval;
            tmpval = -(calShading2depthGrad(state, posx, posy, input).w - calShading2depthGrad(state, posx, posy + 1, input).w);//edge 6
            maskval = colMask(posy, posx, input);
            sum += tmpval*(val0 - val2) * maskval;//(posy,posx)*(val2-val0),(posy+1,posx)*(val0-val2)			
            tmpval += (val0 - val2)*(val0 - val2)* maskval;
            tmpval = -(calShading2depthGrad(state, posx + 1, posy, input).w - calShading2depthGrad(state, posx + 1, posy + 1, input).w);//edge 8
            maskval = colMask(posy, posx + 1, input);
            sum += tmpval*val1 * maskval;//(posy,posx+1)*(-val1),(posy+1,posx+1)*(val1)
            tmpval += val1*val1* maskval;
            tmpval = -(calShading2depthGrad(state, posx, posy + 1, input).w - calShading2depthGrad(state, posx, posy + 2, input).w);//edge 9
            maskval = colMask(posy + 1, posx, input);
            sum += tmpval*val2 * maskval;//(posy+1,posx)*(-val2),(posy+2,posx)*(val2)
            tmpval += val2*val2* maskval;
            b += sum * parameters.weightShading;
            p += tmpval * parameters.weightShading;//shading constraint
            val0 = (posx - ux) / fx;
            val1 = (posy - uy) / fy;		
            float3 lapval = est_lap_init_3d_imp(state, posx, posy, val0, val1, ufx, ufy, W, b_valid);
            sum = 0.0f;
            sum += lapval.x*val0*(-4.0f);
            sum += lapval.y*val1*(-4.0f);
            sum += lapval.z*(-4.0f);
            lapval = est_lap_init_3d_imp(state, posx - 1, posy, val0 - ufx, val1, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;
            lapval = est_lap_init_3d_imp(state, posx + 1, posy, val0 + ufx, val1, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;
            lapval = est_lap_init_3d_imp(state, posx, posy - 1, val0, val1 - ufy, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;
            lapval = est_lap_init_3d_imp(state, posx, posy + 1, val0, val1 + ufy, ufx, ufy, W, b_valid);
            sum += lapval.x*val0;
            sum += lapval.y*val1;
            sum += lapval.z;
                b += sum*parameters.weightRegularizer;
                tmpval = (val0 * val0 + val1 * val1 + 1)*(16 + 4);
                p += tmpval *parameters.weightRegularizer;//smoothness		
            p += parameters.weightFitting;//position constraint			
            b += -(XC - targetDepth) * parameters.weightFitting;
        }
    }
    if (p > FLOAT_EPSILON) pre = 1.0f / p;
    else			      pre = 1.0f;
    return b;
}
__inline__ __device__ float applyJTJDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters){
    int posy; int posx; get2DIdx(variableIdx, input.width, input.height, posy, posx);
    const int W = input.width;
    const int H = input.height;
    const float fx = input.calibparams.fx;
    const float fy = input.calibparams.fy;
    const float ux = input.calibparams.ux;
    const float uy = input.calibparams.uy;
    const float ufx = 1.0f / input.calibparams.fx;
    const float ufy = 1.0f / input.calibparams.fy;
    float b = 0.0f;
    const float targetDepth = input.d_targetDepth[posy*W + posx]; const bool validTarget = IsValidPoint(targetDepth);
    const float PC = state.d_p[variableIdx];
    if (validTarget){
        if ((posx>1) && (posx<(W - 5)) && (posy>1) && (posy<(H - 5))){
            float sum = 0.0f;
            float tmpval = 0.0f;
            float val0, val1, val2;
            val0 = calShading2depthGrad(state, posx, posy, input).y;
            val1 = calShading2depthGrad(state, posx + 1, posy, input).x;
            val2 = calShading2depthGrad(state, posx, posy + 1, input).z;
            float4 grad_0_m1 = calShading2depthGrad(state,posx+0, posy-1, input);				
            float4 grad_1_m1 = calShading2depthGrad(state,posx+1, posy-1, input);
            float4 grad_m1_0 = calShading2depthGrad(state,posx-1, posy-0, input);
            float4 grad_0_0  = calShading2depthGrad(state,posx  , posy  , input);
            float4 grad_1_0  = calShading2depthGrad(state,posx+1, posy  , input);
            float4 grad_2_0  = calShading2depthGrad(state,posx+2, posy  , input);
            float4 grad_m1_1 = calShading2depthGrad(state,posx-1, posy+1, input);
            float4 grad_0_1  = calShading2depthGrad(state,posx  , posy+1, input);
            float4 grad_1_1  = calShading2depthGrad(state,posx+1, posy+1, input);
            float4 grad_0_2  = calShading2depthGrad(state,posx  , posy+2, input);
            tmpval = readP(posy, posx - 2, state, W) *  grad_m1_0.x;
            tmpval += readP(posy, posx - 1, state, W) * (grad_m1_0.y - grad_0_0.x);
            tmpval += readP(posy - 1, posx - 1, state, W) *  grad_m1_0.z;
            tmpval -= readP(posy, posx, state, W) *  grad_0_0.y;
            tmpval -= readP(posy - 1, posx, state, W) *  grad_0_0.z;
            sum += (-val0) * tmpval  * rowMask(posy, posx - 1, input);
            tmpval = readP(posy - 1, posx - 1, state, W) *  grad_0_m1.x;
            tmpval += readP(posy - 1, posx, state, W) * (grad_0_m1.y - grad_0_0.z);
            tmpval += readP(posy - 2, posx, state, W) *  grad_0_m1.z;
            tmpval -= readP(posy, posx - 1, state, W) *  grad_0_0.x;
            tmpval -= readP(posy, posx, state, W) *  grad_0_0.y;
            sum += (-val0) * tmpval  * colMask(posy - 1, posx, input);
            tmpval = readP(posy, posx - 1, state, W) *  grad_0_0.x;
            tmpval += readP(posy, posx, state, W) * (grad_0_0.y - grad_1_0.x);
            tmpval += readP(posy - 1, posx, state, W) *  grad_0_0.z;
            tmpval -= readP(posy, posx + 1, state, W) *  grad_1_0.y;
            tmpval -= readP(posy - 1, posx + 1, state, W) *  grad_1_0.z;
            sum += (val0 - val1) * tmpval * rowMask(posy, posx, input);	
            tmpval = readP(posy - 1, posx, state, W) *  grad_1_m1.x;
            tmpval += readP(posy - 1, posx + 1, state, W) * (grad_1_m1.y - grad_1_0.z);
            tmpval += readP(posy - 2, posx + 1, state, W) *  grad_1_m1.z;
            tmpval -= readP(posy, posx, state, W) *  grad_1_0.x;
            tmpval -= readP(posy, posx + 1, state, W) *  grad_1_0.y;
            sum += (-val1) * tmpval	* colMask(posy - 1, posx + 1, input);
            tmpval = readP(posy, posx, state, W) *  grad_1_0.x;
            tmpval += readP(posy, posx + 1, state, W) * (grad_1_0.y - grad_2_0.x);
            tmpval += readP(posy - 1, posx + 1, state, W) *  grad_1_0.z;
            tmpval -= readP(posy, posx + 2, state, W) *  grad_2_0.y;
            tmpval -= readP(posy - 1, posx + 2, state, W) *  grad_2_0.z;
            sum += (val1)* tmpval * rowMask(posy, posx + 1, input);
            tmpval = readP(posy + 1, posx - 2, state, W) *  grad_m1_1.x;
            tmpval += readP(posy + 1, posx - 1, state, W) * (grad_m1_1.y - grad_0_1.x);
            tmpval += readP(posy, posx - 1, state, W) *  grad_m1_1.z;
            tmpval -= readP(posy + 1, posx, state, W) *  grad_0_1.y;
            tmpval -= readP(posy, posx, state, W) *  grad_0_1.z;
            sum += (-val2) * tmpval * rowMask(posy + 1, posx - 1, input);
            tmpval = readP(posy, posx - 1, state, W) *  grad_0_0.x;
            tmpval += readP(posy, posx, state, W) * (grad_0_0.y - grad_0_1.z);
            tmpval += readP(posy - 1, posx, state, W) *  grad_0_0.z;
            tmpval -= readP(posy + 1, posx - 1, state, W) *  grad_0_1.x;
            tmpval -= readP(posy + 1, posx, state, W) *  grad_0_1.y;
            sum += (val0 - val2) * tmpval * colMask(posy, posx, input);
            tmpval = readP(posy + 1, posx - 1, state, W) *  grad_0_1.x;
            tmpval += readP(posy + 1, posx, state, W) * (grad_0_1.y - grad_1_1.x);
            tmpval += readP(posy, posx, state, W) *  grad_0_1.z;
            tmpval -= readP(posy + 1, posx + 1, state, W) *  grad_1_1.y;
            tmpval -= readP(posy, posx + 1, state, W) *  grad_1_1.z;
            sum += val2 * tmpval * rowMask(posy + 1, posx, input);
            tmpval = readP(posy, posx, state, W) *  grad_1_0.x;
            tmpval += readP(posy, posx + 1, state, W) * (grad_1_0.y - grad_1_1.z);
            tmpval += readP(posy - 1, posx + 1, state, W) *  grad_1_0.z;
            tmpval -= readP(posy + 1, posx, state, W) *  grad_1_1.x;
            tmpval -= readP(posy + 1, posx + 1, state, W) *  grad_1_1.y;
            sum += val1 * tmpval * colMask(posy, posx + 1, input);
            tmpval = readP(posy + 1, posx - 1, state, W) *  grad_0_1.x;
            tmpval += readP(posy + 1, posx, state, W) * (grad_0_1.y - grad_0_2.z);
            tmpval += readP(posy, posx, state, W) *  grad_0_1.z;
            tmpval -= readP(posy + 2, posx - 1, state, W) *  grad_0_2.x;
            tmpval -= readP(posy + 2, posx, state, W) *  grad_0_2.y;
            sum += val2 * tmpval * colMask(posy + 1, posx, input);
            b += sum * parameters.weightShading;
            sum = 0;
            val0 = (posx - ux) / fx;
            val1 = (posy - uy) / fy;
            float3 lapval = est_lap_3d_bsp_imp_with_guard(state, posx, posy, val0, val1, ufx, ufy, W);
            sum += lapval.x*val0*(4.0f);
            sum += lapval.y*val1*(4.0f);
            sum += lapval.z*(4.0f);
            lapval = est_lap_3d_bsp_imp_with_guard(state, posx - 1, posy, val0 - ufx, val1, ufx, ufy, W);
            sum -= lapval.x*val0;
            sum -= lapval.y*val1;
            sum -= lapval.z;
            lapval = est_lap_3d_bsp_imp_with_guard(state, posx + 1, posy, val0 + ufx, val1, ufx, ufy, W);
            sum -= lapval.x*val0;
            sum -= lapval.y*val1;
            sum -= lapval.z;
            lapval = est_lap_3d_bsp_imp_with_guard(state, posx, posy - 1, val0, val1 - ufy, ufx, ufy, W);
            sum -= lapval.x*val0;
            sum -= lapval.y*val1;
            sum -= lapval.z;
            lapval = est_lap_3d_bsp_imp_with_guard(state, posx, posy + 1, val0, val1 + ufy, ufx, ufy, W);
            sum -= lapval.x*val0;
            sum -= lapval.y*val1;
            sum -= lapval.z;
            b += sum*parameters.weightRegularizer;
            b += PC*parameters.weightFitting;
        }
    }
    return b;
}
