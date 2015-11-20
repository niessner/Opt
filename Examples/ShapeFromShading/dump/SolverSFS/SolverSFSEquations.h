#pragma once

#ifndef _SOLVER_SFS_EQUATIONS_
#define _SOLVER_SFS_EQUATIONS_

#include <cutil_inline.h>
#include <cutil_math.h>

#include "SolverSFSUtil.h"
#include "SolverSFSState.h"
#include "SolverSFSParameters.h"

#define USE_UPLEFT_TRI
#define WEIGHT_SHADING_VALUE 1.0f

////////////////////////////////////////
// evalMinusJTF
////////////////////////////////////////

__inline__ __device__ float evalMinusJTFDevice(unsigned int variableIdx, SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int imgind = variableIdx;
	const unsigned int posx = imgind % input.width;
	const unsigned int posy = imgind / input.width;

	float b = 0.0f, p = 0.0f;

	// Reset linearized update vector
	state.d_delta[variableIdx] = 0.0f;

	if(posx>1 && posx<(input.width-2) && posy>1 && posy<(input.height-2)){
		bool validTarget = (input.d_targetDepth[imgind] != MINF);
		bool validTarget0 = (input.d_targetDepth[imgind-1] != MINF);
		bool validTarget1 = (input.d_targetDepth[imgind+1] != MINF);
		bool validTarget2 = (input.d_targetDepth[imgind-input.width] != MINF);
		bool validTarget3 = (input.d_targetDepth[imgind+input.width] != MINF);

		if(validTarget && validTarget0 && validTarget1 && validTarget2 && validTarget3) 
		//if(validTarget)
		{
			float val0 = state.d_grad[imgind*3+1];
			float val1 = state.d_grad[(imgind+1)*3];
			float val2 = state.d_grad[(imgind+input.width)*3+2];

			//shading term
			//row edge constraint
			float sum = 0.0f, tmpval = 0.0f;
			int tmpind = (posy)*(input.width)+posx-1;
			tmpval = -(state.d_shadingdif[tmpind] - state.d_shadingdif[tmpind+1]);				
			sum += tmpval*(-val0);
			tmpind += 1;
			tmpval = -(state.d_shadingdif[tmpind] - state.d_shadingdif[tmpind+1]);				
			sum += tmpval*(val0-val1);
			tmpind += 1;
			tmpval = -(state.d_shadingdif[tmpind] - state.d_shadingdif[tmpind+1]);				
			sum += tmpval*(val1);
			tmpind = (posy+1)*(input.width)+posx-1;
			tmpval = -(state.d_shadingdif[tmpind] - state.d_shadingdif[tmpind+1]);				
			sum += tmpval*(-val2);
			tmpind += 1;
			tmpval = -(state.d_shadingdif[tmpind] - state.d_shadingdif[tmpind+1]);				
			sum += tmpval*val2;
						
			//column edge constraint
			tmpind = (posy-1)*(input.width)+posx;
			tmpval = -(state.d_shadingdif[tmpind] - state.d_shadingdif[tmpind+input.width]);
			sum += tmpval*(-val0);
			tmpind += 1;
			tmpval = -(state.d_shadingdif[tmpind] - state.d_shadingdif[tmpind+input.width]);
			sum += tmpval*(-val1);
			tmpind = (posy)*(input.width)+posx;
			tmpval = -(state.d_shadingdif[tmpind] - state.d_shadingdif[tmpind+input.width]);
			sum += tmpval*(val0-val2);
			tmpind += 1;
			tmpval = -(state.d_shadingdif[tmpind] - state.d_shadingdif[tmpind+input.width]);
			sum += tmpval*val1;
			tmpind = (posy+1)*(input.width)+posx;
			tmpval = -(state.d_shadingdif[tmpind] - state.d_shadingdif[tmpind+input.width]);
			sum += tmpval*val2;

			b += sum * parameters.weightShading  * parameters.weightShading;


			//shading value constraint
			
#ifdef USE_UPLEFT_TRI
			p += (val0*val0+val1*val1+val2*val2) * WEIGHT_SHADING_VALUE;//shading constraint
			sum = 0.0f;//WEIGHT_SHADING_VALUE
			sum -= val0 * state.d_shadingdif[tmpind];//readValueFromCache2DLS_SFS(inShadingdif0,tidy  ,tidx  );
			sum -= val1 * state.d_shadingdif[tmpind+1];//readValueFromCache2DLS_SFS(inShadingdif0,tidy  ,tidx+1);
			sum -= val2 * state.d_shadingdif[tmpind+input.width];//readValueFromCache2DLS_SFS(inShadingdif0,tidy+1,tidx  );
			b += sum * WEIGHT_SHADING_VALUE * WEIGHT_SHADING_VALUE;
#else

			val0 = readValueFromCache2DLS_SFS(inGrady1,tidy  ,tidx  );
			val1 = readValueFromCache2DLS_SFS(inGradx1,tidy  ,tidx-1);
			val2 = readValueFromCache2DLS_SFS(inGradz1,tidy-1,tidx  );

			p += (val0*val0+val1*val1+val2*val2) * WEIGHT_SHADING_VALUE;//shading constraint
			sum = 0.0f;//WEIGHT_SHADING_VALUE
			sum -= val0 * readValueFromCache2DLS_SFS(inShadingdif1,tidy  ,tidx  );
			sum -= val1 * readValueFromCache2DLS_SFS(inShadingdif1,tidy  ,tidx-1);
			sum -= val2 * readValueFromCache2DLS_SFS(inShadingdif1,tidy-1,tidx  );
			b += sum * WEIGHT_SHADING_VALUE ;

#endif

			//smoothness term
			tmpind = (posy)*(input.width) + (posx);
			sum = -(state.d_x[tmpind]*4 - state.d_x[tmpind+1] - state.d_x[tmpind-1] - state.d_x[tmpind+input.width] - state.d_x[tmpind-input.width]) * 4;			
			tmpind -= 1;
			sum += (state.d_x[tmpind]*4 - state.d_x[tmpind+1] - state.d_x[tmpind-1] - state.d_x[tmpind+input.width] - state.d_x[tmpind-input.width]);
			tmpind += 2;
			sum += (state.d_x[tmpind]*4 - state.d_x[tmpind+1] - state.d_x[tmpind-1] - state.d_x[tmpind+input.width] - state.d_x[tmpind-input.width]);
			tmpind -= (input.width + 1);
			sum += (state.d_x[tmpind]*4 - state.d_x[tmpind+1] - state.d_x[tmpind-1] - state.d_x[tmpind+input.width] - state.d_x[tmpind-input.width]);
			tmpind += input.width * 2;
			sum += (state.d_x[tmpind]*4 - state.d_x[tmpind+1] - state.d_x[tmpind-1] - state.d_x[tmpind+input.width] - state.d_x[tmpind-input.width]);
			
			b += sum*parameters.weightRegularizer;	

			
			//position term 			
			sum = -(state.d_x[imgind]-input.d_targetDepth[imgind]);
			b += sum * parameters.weightFitting * parameters.weightFitting;


			//calculating  the preconditioner						
			tmpval = val0 * val0 * 2;
			tmpval += (val0 - val1) * (val0 - val1);
			tmpval += (val0 - val2) * (val0 - val2);
			tmpval += val1 * val1 * 3;
			tmpval += val2 * val2 * 3;
			p += tmpval * parameters.weightShading * parameters.weightShading;//shading constraint
			p += (16+4)*parameters.weightRegularizer*parameters.weightRegularizer;//smoothness
			p += parameters.weightFitting*parameters.weightFitting;//position constraint

		}
		else
			p += parameters.weightBoundary*parameters.weightBoundary;

	}
	{
		p += parameters.weightBoundary*parameters.weightBoundary;		
	}


	if(p > FLOAT_EPSILON) state.d_precondioner[variableIdx] = 1.0f/p;
	else				  state.d_precondioner[variableIdx] = 1.0f;

	return b;
}

////////////////////////////////////////
// applyJTJ
////////////////////////////////////////

__inline__ __device__ float applyJTJDevice(unsigned int x, SolverInput input, SolverState state, SolverParameters parameters)
{
	const int imgind = x;

	const int posx = imgind % input.width;
	const int posy = imgind / input.width;
	
	float b = 0.0f;
	
	if((posx>1) && (posx<(input.width-2)) && (posy>1) && (posy<(input.height-2))){
		bool validTarget  = (input.d_targetDepth[imgind] != MINF);
		bool validTarget0 = (input.d_targetDepth[imgind-1] != MINF);
		bool validTarget1 = (input.d_targetDepth[imgind+1] != MINF);
		bool validTarget2 = (input.d_targetDepth[imgind-input.width] != MINF);
		bool validTarget3 = (input.d_targetDepth[imgind+input.width] != MINF);

		if(validTarget && validTarget0 && validTarget1 && validTarget2 && validTarget3)
		//if(validTarget)
		{

			float sum = 0, sumsm = 0, tmpval = 0;;

			float val0 = state.d_grad[imgind*3+1];
			float val1 = state.d_grad[(imgind+1)*3];
			float val2 = state.d_grad[(imgind+input.width)*3+2];

			//the following is the adding the relative edge constraint to sum
			//-val0, edge 0
			tmpval =  state.d_p[imgind-2]       * state.d_grad[(imgind-1)*3];
			tmpval += state.d_p[imgind-1]       * (state.d_grad[(imgind-1)*3+1]-state.d_grad[(imgind)*3]);
			tmpval += state.d_p[imgind-1-input.width] * state.d_grad[(imgind-1)*3+2] ;			
			tmpval += state.d_p[imgind]         * (-val0);
			tmpval += state.d_p[imgind-input.width]   * (-state.d_grad[(imgind)*3+2]) ;
			sum += (-val0) * tmpval;

			//-val0, edge 1
			tmpval =  state.d_p[imgind-1-input.width] * state.d_grad[(imgind-input.width)*3];
			tmpval += state.d_p[imgind-input.width]   * (state.d_grad[(imgind-input.width)*3+1]-state.d_grad[(imgind)*3+2]);
			tmpval += state.d_p[imgind-input.width*2] * state.d_grad[(imgind-input.width)*3+2];			
			tmpval += state.d_p[imgind-1]       * (-state.d_grad[(imgind)*3]) ;
			tmpval += state.d_p[imgind]         * (-val0);
			sum += (-val0) * tmpval;

			//val0-val1, edge 2
			tmpval =  state.d_p[imgind-1]       * state.d_grad[(imgind)*3];
			tmpval += state.d_p[imgind]         * (state.d_grad[(imgind)*3+1]-state.d_grad[(imgind+1)*3]);
			tmpval += state.d_p[imgind-input.width]   * state.d_grad[(imgind)*3+2] ;
			tmpval += state.d_p[imgind+1]       * (-state.d_grad[(imgind+1)*3+1]);
			tmpval += state.d_p[imgind+1-input.width] * (-state.d_grad[(imgind+1)*3+2]) ;
			sum += (val0-val1) * tmpval;

			//-val1, edge 3
			tmpval =  state.d_p[imgind-input.width]     * state.d_grad[(imgind-input.width+1)*3];
			tmpval += state.d_p[imgind-input.width+1]   * (state.d_grad[(imgind-input.width+1)*3+1]-state.d_grad[(imgind+1)*3+2]);
			tmpval += state.d_p[imgind-input.width*2+1] * state.d_grad[(imgind-input.width+1)*3+2];			
			tmpval += state.d_p[imgind]           * (-state.d_grad[(imgind+1)*3]) ;
			tmpval += state.d_p[imgind+1]         * (-state.d_grad[(imgind+1)*3+1]) ;
			sum += (-val1) * tmpval	;

			//val1, edge 4
			tmpval =  state.d_p[imgind]         * state.d_grad[(imgind+1)*3];
			tmpval += state.d_p[imgind+1]       * (state.d_grad[(imgind+1)*3+1]-state.d_grad[(imgind+2)*3]);
			tmpval += state.d_p[imgind+1-input.width] * state.d_grad[(imgind+1)*3+2] ;
			tmpval += state.d_p[imgind+2]       * (-state.d_grad[(imgind+2)*3+1]);
			tmpval += state.d_p[imgind+2-input.width] * (-state.d_grad[(imgind+2)*3+2]) ;
			sum += (val1) * tmpval;

			//-val2, edge 5
			tmpval  = state.d_p[imgind-2+input.width] * state.d_grad[(imgind+input.width-1)*3];
			tmpval += state.d_p[imgind-1+input.width] * (state.d_grad[(imgind+input.width-1)*3+1]-state.d_grad[(imgind+input.width)*3]);
			tmpval += state.d_p[imgind-1]       * state.d_grad[(imgind+input.width-1)*3+2] ;
			tmpval += state.d_p[imgind+input.width]   * (-state.d_grad[(imgind+input.width)*3+1]);
			tmpval += state.d_p[imgind]         * (-state.d_grad[(imgind+input.width)*3+2]) ;
			sum += (-val2) * tmpval;


			//val0-val2, edge 6
			tmpval =  state.d_p[imgind-1]       * state.d_grad[(imgind)*3];
			tmpval += state.d_p[imgind]         * (state.d_grad[(imgind)*3+1]-state.d_grad[(imgind+input.width)*3+2]);
			tmpval += state.d_p[imgind-input.width]   * state.d_grad[(imgind)*3+2];			
			tmpval += state.d_p[imgind+input.width-1] * (-state.d_grad[(imgind+input.width)*3]) ;
			tmpval += state.d_p[imgind+input.width]   * (-state.d_grad[(imgind+input.width)*3+1]) ;
			sum += (val0-val2) * tmpval;

			//val2, edge 7
			tmpval =  state.d_p[imgind+input.width-1] * state.d_grad[(imgind+input.width)*3];
			tmpval += state.d_p[imgind+input.width]   * (state.d_grad[(imgind+input.width)*3+1]-state.d_grad[(imgind+input.width+1)*3]);
			tmpval += state.d_p[imgind]         * state.d_grad[(imgind+input.width)*3+2] ;
			tmpval += state.d_p[imgind+input.width+1] * (-state.d_grad[(imgind+input.width+1)*3+1]);
			tmpval += state.d_p[imgind+1]       * (-state.d_grad[(imgind+input.width+1)*3+2]) ;
			sum += val2 * tmpval;

			//val1, edge 8
			tmpval =  state.d_p[imgind]         * state.d_grad[(imgind+1)*3];
			tmpval += state.d_p[imgind+1]       * (state.d_grad[(imgind+1)*3+1]-state.d_grad[(imgind+input.width+1)*3+2]);
			tmpval += state.d_p[imgind+1-input.width] * state.d_grad[(imgind+1)*3+2];			
			tmpval += state.d_p[imgind+input.width]   * (-state.d_grad[(imgind+input.width+1)*3]) ;
			tmpval += state.d_p[imgind+input.width+1] * (-state.d_grad[(imgind+input.width+1)*3+1]) ;
			sum += val1 * tmpval;

			//val2, edge 9
			tmpval =  state.d_p[imgind+input.width-1]   * state.d_grad[(imgind+input.width)*3];
			tmpval += state.d_p[imgind+input.width]     * (state.d_grad[(imgind+input.width)*3+1]-state.d_grad[(imgind+input.width*2)*3+2]);
			tmpval += state.d_p[imgind]           * state.d_grad[(imgind+input.width)*3+2];			
			tmpval += state.d_p[imgind+input.width*2-1] * (-state.d_grad[(imgind+input.width*2)*3]) ;
			tmpval += state.d_p[imgind+input.width*2]   * (-state.d_grad[(imgind+input.width*2)*3+1]) ;
			sum += val2 * tmpval;

			//sum *= parameters.weightShading*parameters.weightShading;
			b += sum * parameters.weightShading*parameters.weightShading;


						
#ifdef USE_UPLEFT_TRI
			sum = 0;
			tmpval = state.d_p[imgind-1]				* state.d_grad[(imgind)*3];
			tmpval += state.d_p[imgind]					* state.d_grad[(imgind)*3+1];
			tmpval += state.d_p[imgind-input.width]		* state.d_grad[(imgind)*3+2];
			sum += val0 *  tmpval;//add_mul_inp_grad_ls_bsp(inP,inGradx0,inGrady0,inGradz0,tidx  ,tidy  );		

			tmpval = state.d_p[imgind]					* state.d_grad[(imgind+1)*3];
			tmpval += state.d_p[imgind+1]				* state.d_grad[(imgind+1)*3+1];
			tmpval += state.d_p[imgind+1-input.width]   * state.d_grad[(imgind+1)*3+2];
			sum += val1 *  tmpval;//add_mul_inp_grad_ls_bsp(inP,inGradx0,inGrady0,inGradz0,tidx+1,tidy  );

			tmpval =  state.d_p[imgind+input.width-1]   * state.d_grad[(imgind+input.width)*3];
			tmpval += state.d_p[imgind+input.width]		* state.d_grad[(imgind+input.width)*3+1];
			tmpval += state.d_p[imgind]					* state.d_grad[(imgind+input.width)*3+2];
			sum += val2 *  tmpval;//add_mul_inp_grad_ls_bsp(inP,inGradx0,inGrady0,inGradz0,tidx  ,tidy+1);
			b += sum * WEIGHT_SHADING_VALUE * WEIGHT_SHADING_VALUE ;
#else


			val0 = readValueFromCache2DLS_SFS(inGrady1,tidy  ,tidx  );
			val1 = readValueFromCache2DLS_SFS(inGradx1,tidy  ,tidx-1);
			val2 = readValueFromCache2DLS_SFS(inGradz1,tidy-1,tidx  );

			sum = 0;
			sum += val0 *  add_mul_inp_grad_ls_bsp(inP,inGradx1,inGrady1,inGradz1,tidx  ,tidy  );		
			sum += val1 *  add_mul_inp_grad_ls_bsp(inP,inGradx1,inGrady1,inGradz1,tidx-1,tidy  );
			sum += val2 *  add_mul_inp_grad_ls_bsp(inP,inGradx1,inGrady1,inGradz1,tidx  ,tidy-1);
			b += sum * WEIGHT_SHADING_VALUE ;
#endif


			//smoothness constraint
			//center vertex, 4
			sumsm += 16 * state.d_p[imgind];
			sumsm -= 4 * state.d_p[imgind-1];
			sumsm -= 4 * state.d_p[imgind+1];
			sumsm -= 4 * state.d_p[imgind+input.width];
			sumsm -= 4 * state.d_p[imgind-input.width];

			//neighboring vertex 0, -1
			sumsm -= 4 * state.d_p[imgind-1];
			sumsm += state.d_p[imgind];
			sumsm += state.d_p[imgind-2];
			sumsm += state.d_p[imgind+input.width-1];
			sumsm += state.d_p[imgind-input.width-1];

			//neighboring vertex 1, -1
			sumsm -= 4 * state.d_p[imgind+1];
			sumsm += state.d_p[imgind];
			sumsm += state.d_p[imgind+2];
			sumsm += state.d_p[imgind+input.width+1];
			sumsm += state.d_p[imgind-input.width+1];

			//neighboring vertex 2, -1
			sumsm -= 4 * state.d_p[imgind-input.width];
			sumsm += state.d_p[imgind];
			sumsm += state.d_p[imgind-input.width-1];
			sumsm += state.d_p[imgind-input.width+1];
			sumsm += state.d_p[imgind-2*input.width];

			//neighboring vertex 3, -1
			sumsm -= 4 * state.d_p[imgind+input.width];
			sumsm += state.d_p[imgind];
			sumsm += state.d_p[imgind+input.width-1];
			sumsm += state.d_p[imgind+input.width+1];
			sumsm += state.d_p[imgind+2*input.width];

			//sum += sumsm*parameters.weightRegularizer*parameters.weightRegularizer;
			b += sumsm*parameters.weightRegularizer*parameters.weightRegularizer;// smoothness 			

			b += state.d_p[imgind]*parameters.weightFitting*parameters.weightFitting;// position term
		
		}
		else
			b += state.d_p[imgind]*parameters.weightBoundary*parameters.weightBoundary;		
	}else
		b += state.d_p[imgind]*parameters.weightBoundary*parameters.weightBoundary;	
	
	return b;
}

//add by chenglei
#define USE_ORDER2
__global__ void Cal_Shading2depth_Grad_1d(SolverInput input, SolverState state)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;	
	if(input.m_useRemapping) x = input.d_remapArray[x];

	const unsigned int posx = x % input.width;
	const unsigned int posy = x / input.width;	

	const float fx = input.calibparams.fx;
	const float fy = input.calibparams.fy;
	const float ux = input.calibparams.ux;
	const float uy = input.calibparams.uy;
	const float ufx = 1.0f/input.calibparams.fx;
	const float ufy = 1.0f/input.calibparams.fy;

	if(posx>0 && posy>0 && posx<(input.width-1)&&posy<(input.height-1))
	{
		float d0 = state.d_x[x-1];
		float d1 = state.d_x[x];
		float d2 = state.d_x[x-input.width];

		//float m0 = input.d_targetDepth[imgind-1];
		//float m1 = input.d_targetDepth[imgind];
		//float m2 = input.d_targetDepth[imgind-input.width];

		float greyval = input.d_targetIntensity[x];

		//if( (m0!= MINF) && (m1!= MINF) && (m2!= MINF))
		if( (d0!= MINF) && (d1!= MINF) && (d2!= MINF))
		{
			float ax = (posx-ux)/fx;
			float ay = (posy-uy)/fy;
			float an;

			float px,py,pz;
			px = - d2*(d0-d1)/fy;
			py = - d0*(d2-d1)/fx;
			pz = -ay*py - ax*px - d2*d0/fx/fy;			
			an = sqrt(px*px+py*py+pz*pz);
			if(an==0)
				return;
			else
				px/=an;py/=an;pz/=an;
									
			float sh_callist0 = input.d_litcoeff[0];
			float sh_callist1 = py*input.d_litcoeff[1];
			float sh_callist2 = pz*input.d_litcoeff[2];
			float sh_callist3 = px*input.d_litcoeff[3];
#ifdef USE_ORDER2
			float sh_callist4 = px * py*input.d_litcoeff[4];
			float sh_callist5 = py *pz*input.d_litcoeff[5];
			float sh_callist6 = (3*pz*pz-1)*input.d_litcoeff[6];
			float sh_callist7 = pz * px *input.d_litcoeff[7];
			float sh_callist8 = (px*px-py*py)*input.d_litcoeff[8];
#endif

			//normal changes wrt depth
			float gradx =0, grady = 0, gradz = 0;

			gradx += -sh_callist1*px;
			gradx += -sh_callist2*px;
			gradx += input.d_litcoeff[3]-sh_callist3*px;
#ifdef USE_ORDER2
			gradx += py*input.d_litcoeff[4]-sh_callist4*2*px;
			gradx += -sh_callist5*2*px;
			gradx += (-2*px)*input.d_litcoeff[6]-sh_callist6*2*px;
			gradx += pz*input.d_litcoeff[7]-sh_callist7*2*px;
			gradx += 2*px*input.d_litcoeff[8]-sh_callist8*2*px;
#endif
			gradx /= an;

			grady += input.d_litcoeff[1]-sh_callist1*py;
			grady += -sh_callist2*py;
			grady += -sh_callist3*py;
#ifdef USE_ORDER2
			grady += px*input.d_litcoeff[4]-sh_callist4*2*py;
			grady += pz*input.d_litcoeff[5]-sh_callist5*2*py;
			grady += (-2*py)*input.d_litcoeff[6]-sh_callist6*2*py;
			grady += -sh_callist7*2*py;
			grady += (-2*py)*input.d_litcoeff[8]-sh_callist8*2*py;
#endif
			grady /= an;

			gradz += -sh_callist1*pz;
			gradz += input.d_litcoeff[2]-sh_callist2*pz;
			gradz += -sh_callist3*pz;
#ifdef USE_ORDER2
			gradz += -sh_callist4*2*pz;
			gradz += py*input.d_litcoeff[5]-sh_callist5*2*pz;
			gradz += 4*pz*input.d_litcoeff[6]-sh_callist6*2*pz;
			gradz += px*input.d_litcoeff[7]-sh_callist7*2*pz;
			gradz += -sh_callist8*2*pz;
#endif
			gradz /= an;

			//shading value in sh_callist[0]
			sh_callist0 += sh_callist1;
			sh_callist0 += sh_callist2;
			sh_callist0 += sh_callist3;
#ifdef USE_ORDER2
			sh_callist0 += sh_callist4;
			sh_callist0 += sh_callist5;
			sh_callist0 += sh_callist6;
			sh_callist0 += sh_callist7;
			sh_callist0 += sh_callist8;
#endif
			sh_callist0 -= greyval;


			///////////////////////////////////////////////////////
			//
			//               /|  2
			//             /  |
			//           /    |  
			//         0 -----|  1
			//
			///////////////////////////////////////////////////////

			float3 edge,grnd;
			//edge opposite to point 0
			edge.x = ax*(d2-d1);
			edge.y = ay*(d2-d1) - d2/fy;
			edge.z = d2 - d1;
			an = sqrt((ax-ufx)*(ax-ufx)+ay*ay+1);
			grnd.x = (-edge.z*ay + edge.y);
			grnd.y = (edge.z*(ax-ufx) - edge.x);
			grnd.z = (-edge.y*(ax-ufx)+edge.x*ay);
			sh_callist1 = (gradx*grnd.x+grady*grnd.y+gradz*grnd.z)/an;

			//edge opposite to point 1
			edge.x = ax*(d0-d2) - d0/fx;
			edge.y = ay*(d0-d2) + d2/fy;
			edge.z = d0 -d2;
			an = sqrt(ax*ax+ay*ay+1);
			grnd.x = (-edge.z*ay + edge.y);
			grnd.y = (edge.z*ax - edge.x);
			grnd.z = (-edge.y*ax+edge.x*ay);
			sh_callist2 = (gradx*grnd.x+grady*grnd.y+gradz*grnd.z)/an;

			//edge opposite to point 2
			edge.x = ax*(d1-d0)+d0/fx;
			edge.y = ay*(d1-d0);
			edge.z = d1-d0;
			an = sqrt(ax*ax+(ay-ufy)*(ay-ufy)+1);
			grnd.x = (-edge.z*(ay-ufy) + edge.y);
			grnd.y = (edge.z*ax - edge.x);
			grnd.z = (-edge.y*ax+edge.x*(ay-ufy));
			sh_callist3 = (gradx*grnd.x+grady*grnd.y+gradz*grnd.z)/an;

			state.d_shadingdif[x] = sh_callist0;
			state.d_grad[x*3] = sh_callist1;
			state.d_grad[x*3+1] = sh_callist2;
			state.d_grad[x*3+2] = sh_callist3;
		}
	}
}
#undef USE_ORDER2

//add by chenglei
#define USE_ORDER2
__global__ void Cal_Shading2depth_Grad_2d(SolverInput input, SolverState state)
{
	unsigned int posx = blockIdx.x * blockDim.x  + threadIdx.x +1;
	unsigned int posy = blockIdx.y* blockDim.y + threadIdx.y +1;	

	unsigned int imgind = posy *input.width + posx;

	const float fx = input.calibparams.fx;
	const float fy = input.calibparams.fy;
	const float ux = input.calibparams.ux;
	const float uy = input.calibparams.uy;
	const float ufx = 1.0f/input.calibparams.fx;
	const float ufy = 1.0f/input.calibparams.fy;

	if(posx<(input.width-1)&&posy<(input.height-1))
	{
		float d0 = state.d_x[imgind-1];
		float d1 = state.d_x[imgind];
		float d2 = state.d_x[imgind-input.width];

		//float m0 = input.d_targetDepth[imgind-1];
		//float m1 = input.d_targetDepth[imgind];
		//float m2 = input.d_targetDepth[imgind-input.width];

		float greyval = input.d_targetIntensity[imgind];

		//if( (m0!= MINF) && (m1!= MINF) && (m2!= MINF))
		if( (d0!= MINF) && (d1!= MINF) && (d2!= MINF))
		{
			float ax = (posx-ux)/fx;
			float ay = (posy-uy)/fy;
			float an;

			float px,py,pz;
			px = - d2*(d0-d1)/fy;
			py = - d0*(d2-d1)/fx;
			pz = -ay*py - ax*px - d2*d0/fx/fy;			
			an = sqrt(px*px+py*py+pz*pz);
			if(an==0)
				return;
			else
				px/=an;py/=an;pz/=an;
									
			float sh_callist0 = input.d_litcoeff[0];
			float sh_callist1 = py*input.d_litcoeff[1];
			float sh_callist2 = pz*input.d_litcoeff[2];
			float sh_callist3 = px*input.d_litcoeff[3];
#ifdef USE_ORDER2
			float sh_callist4 = px * py*input.d_litcoeff[4];
			float sh_callist5 = py *pz*input.d_litcoeff[5];
			float sh_callist6 = (3*pz*pz-1)*input.d_litcoeff[6];
			float sh_callist7 = pz * px *input.d_litcoeff[7];
			float sh_callist8 = (px*px-py*py)*input.d_litcoeff[8];
#endif

			//normal changes wrt depth
			float gradx =0, grady = 0, gradz = 0;

			gradx += -sh_callist1*px;
			gradx += -sh_callist2*px;
			gradx += input.d_litcoeff[3]-sh_callist3*px;
#ifdef USE_ORDER2
			gradx += py*input.d_litcoeff[4]-sh_callist4*2*px;
			gradx += -sh_callist5*2*px;
			gradx += (-2*px)*input.d_litcoeff[6]-sh_callist6*2*px;
			gradx += pz*input.d_litcoeff[7]-sh_callist7*2*px;
			gradx += 2*px*input.d_litcoeff[8]-sh_callist8*2*px;
#endif
			gradx /= an;

			grady += input.d_litcoeff[1]-sh_callist1*py;
			grady += -sh_callist2*py;
			grady += -sh_callist3*py;
#ifdef USE_ORDER2
			grady += px*input.d_litcoeff[4]-sh_callist4*2*py;
			grady += pz*input.d_litcoeff[5]-sh_callist5*2*py;
			grady += (-2*py)*input.d_litcoeff[6]-sh_callist6*2*py;
			grady += -sh_callist7*2*py;
			grady += (-2*py)*input.d_litcoeff[8]-sh_callist8*2*py;
#endif
			grady /= an;

			gradz += -sh_callist1*pz;
			gradz += input.d_litcoeff[2]-sh_callist2*pz;
			gradz += -sh_callist3*pz;
#ifdef USE_ORDER2
			gradz += -sh_callist4*2*pz;
			gradz += py*input.d_litcoeff[5]-sh_callist5*2*pz;
			gradz += 4*pz*input.d_litcoeff[6]-sh_callist6*2*pz;
			gradz += px*input.d_litcoeff[7]-sh_callist7*2*pz;
			gradz += -sh_callist8*2*pz;
#endif
			gradz /= an;

			//shading value in sh_callist[0]
			sh_callist0 += sh_callist1;
			sh_callist0 += sh_callist2;
			sh_callist0 += sh_callist3;
#ifdef USE_ORDER2
			sh_callist0 += sh_callist4;
			sh_callist0 += sh_callist5;
			sh_callist0 += sh_callist6;
			sh_callist0 += sh_callist7;
			sh_callist0 += sh_callist8;
#endif
			sh_callist0 -= greyval;


			///////////////////////////////////////////////////////
			//
			//               /|  2
			//             /  |
			//           /    |  
			//         0 -----|  1
			//
			///////////////////////////////////////////////////////

			float3 edge,grnd;
			//edge opposite to point 0
			edge.x = ax*(d2-d1);
			edge.y = ay*(d2-d1) - d2/fy;
			edge.z = d2 - d1;
			an = sqrt((ax-ufx)*(ax-ufx)+ay*ay+1);
			grnd.x = (-edge.z*ay + edge.y);
			grnd.y = (edge.z*(ax-ufx) - edge.x);
			grnd.z = (-edge.y*(ax-ufx)+edge.x*ay);
			sh_callist1 = (gradx*grnd.x+grady*grnd.y+gradz*grnd.z)/an;

			//edge opposite to point 1
			edge.x = ax*(d0-d2) - d0/fx;
			edge.y = ay*(d0-d2) + d2/fy;
			edge.z = d0 -d2;
			an = sqrt(ax*ax+ay*ay+1);
			grnd.x = (-edge.z*ay + edge.y);
			grnd.y = (edge.z*ax - edge.x);
			grnd.z = (-edge.y*ax+edge.x*ay);
			sh_callist2 = (gradx*grnd.x+grady*grnd.y+gradz*grnd.z)/an;

			//edge opposite to point 2
			edge.x = ax*(d1-d0)+d0/fx;
			edge.y = ay*(d1-d0);
			edge.z = d1-d0;
			an = sqrt(ax*ax+(ay-ufy)*(ay-ufy)+1);
			grnd.x = (-edge.z*(ay-ufy) + edge.y);
			grnd.y = (edge.z*ax - edge.x);
			grnd.z = (-edge.y*ax+edge.x*(ay-ufy));
			sh_callist3 = (gradx*grnd.x+grady*grnd.y+gradz*grnd.z)/an;

			state.d_shadingdif[imgind] = sh_callist0;
			state.d_grad[imgind*3] = sh_callist1;
			state.d_grad[imgind*3+1] = sh_callist2;
			state.d_grad[imgind*3+2] = sh_callist3;
		}
	}
}

#undef USE_ORDER2

#endif
