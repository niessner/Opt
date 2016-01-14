
#include "stdafx.h"

#include "TerraSolverParameters.h"

#ifdef USE_CERES

const bool performanceTest = false;
//const int linearIterationMin = 100;

#include <cuda_runtime.h>

#include "SolverSFSCeres.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

/*
local USE_MASK_REFINE 			= true

local USE_DEPTH_CONSTRAINT 		= true
local USE_REGULARIZATION 		= true
local USE_SHADING_CONSTRAINT 	= true

local USE_CRAPPY_SHADING_BOUNDARY = true

local DEPTH_DISCONTINUITY_THRE = 0.01

-- See TerraSolverParameters
local w_p						= P:Param("w_p",float,0)-- Is initialized by the solver!
local w_s		 				= P:Param("w_s",float,1)-- Regularization weight
local w_r						= P:Param("w_r",float,2)-- Prior weight
local w_g						= P:Param("w_g",float,3)-- Shading weight

w_p = ad.sqrt(w_p)
w_s = ad.sqrt(w_s)
w_r = ad.sqrt(w_r)
w_g = ad.sqrt(w_g)


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
deltaTransform[i] = P:Param("deltaTransform_" .. i .. "",float,offset+i)
end
offset = offset + 16

local L = {}
for i=1,9 do
L[i] = P:Param("L_" .. i .. "",float,offset+i)
end
offset = offset + 9
local nNonLinearIterations 	= P:Param("nNonLinearIterations",uint,offset+1) -- Steps of the non-linear solver
local nLinIterations 		= P:Param("nLinIterations",uint,offset+2) -- Steps of the linear solver
local nPatchIterations 		= P:Param("nPatchIterations",uint,offset+3) -- Steps on linear step on block level


local util = require("util")

local posX = W:index()
local posY = H:index()


function sqMagnitude(point)
return
point[1]*point[1] + point[2]*point[2] +
point[3]*point[3]
end

function times(number,point)
return {number*point[1], number*point[2], number*point[3]}
end

function plus(p0,p1)
return {p0[1]+p1[1], p0[2]+p1[2], p0[3]+p1[3]}
end

-- equation 8
function p(offX,offY)
local d = X(offX,offY)
local i = offX + posX
local j = offY + posY
local point = {((i-u_x)/f_x)*d, ((j-u_y)/f_y)*d, d}
return point
end

-- equation 10
function normalAt(offX, offY)
local i = offX + posX -- good
local j = offY + posY -- good
--f_x good, f_y good

local n_x = X(offX, offY - 1) * (X(offX, offY) - X(offX - 1, offY)) / f_y
local n_y = X(offX - 1, offY) * (X(offX, offY) - X(offX, offY - 1)) / f_x
local n_z = (n_x * (u_x - i) / f_x) + (n_y * (u_y - j) / f_y) - (X(offX-1, offY)*X(offX, offY-1) / (f_x*f_y))
local sqLength = n_x*n_x + n_y*n_y + n_z*n_z
local inverseMagnitude = ad.select(ad.greater(sqLength, 0.0), 1.0/ad.sqrt(sqLength), 1.0)
return times(inverseMagnitude, {n_x, n_y, n_z})
end

function B(offX, offY)
local normal = normalAt(offX, offY)
local n_x = normal[1]
local n_y = normal[2]
local n_z = normal[3]

local lighting = L[1] +
L[2]*n_y + L[3]*n_z + L[4]*n_x  +
L[5]*n_x*n_y + L[6]*n_y*n_z + L[7]*(-n_x*n_x - n_y*n_y + 2*n_z*n_z) + L[8]*n_z*n_x + L[9]*(n_x*n_x-n_y*n_y)

return 1.0*lighting -- replace 1.0 with estimated albedo in slower version
end

function I(offX, offY)
-- TODO: WHYYYYYYYYYY?
return Im(offX,offY)*0.5 + 0.25*(Im(offX-1,offY)+Im(offX,offY-1))
end

local E_s = 0.0
local E_p = 0.0
local E_r_h = 0.0
local E_r_v = 0.0
local E_r_d = 0.0
local E_g_v = 0.0
local E_g_h = 0.0
local pointValid = ad.greater(D_i(0,0), 0)

if USE_DEPTH_CONSTRAINT then
local E_p_noCheck = X(0,0) - D_i(0,0)
E_p = ad.select(opt.InBounds(0,0,0,0), ad.select(pointValid, E_p_noCheck, 0.0), 0.0)
end

if USE_SHADING_CONSTRAINT then
if USE_CRAPPY_SHADING_BOUNDARY then
local shading_center_valid = ad.greater(D_i(-1,0) + D_i(0,0) + D_i(0,-1), 0)
local center_tap_noCheck = B(0,0) - I(0,0)
local center_tap = ad.select(shading_center_valid, center_tap_noCheck, 0.0)
local shading_h_valid = ad.greater(D_i(1,-1) + D_i(0,0) + D_i(1,0), 0)
local shading_v_valid = ad.greater(D_i(-1,1) + D_i(0,0) + D_i(0,1), 0)
local E_g_h_noCheck = (B(1,0) - I(1,0))
local E_g_v_noCheck = (B(0,1) - I(0,1))

local E_g_h_someCheck = center_tap - ad.select(shading_h_valid, E_g_h_noCheck, 0.0)
local E_g_v_someCheck = center_tap - ad.select(shading_v_valid, E_g_v_noCheck, 0.0)
if USE_MASK_REFINE then
E_g_h_someCheck = E_g_h_someCheck * edgeMaskR(0,0)
E_g_v_someCheck = E_g_v_someCheck * edgeMaskC(0,0)
end
E_g_h = ad.select(opt.InBounds(0,0,1,1), E_g_h_someCheck, 0.0)
E_g_v = ad.select(opt.InBounds(0,0,1,1), E_g_v_someCheck, 0.0)

else
local shading_h_valid = ad.greater(D_i(-1,0) + D_i(0,0) + D_i(1,0) + D_i(0,-1) + D_i(1,-1), 0)

local E_g_h_noCheck = B(0,0) - B(1,0) - (I(0,0) - I(1,0))
if USE_MASK_REFINE then
E_g_h_noCheck = E_g_h_noCheck * edgeMaskR(0,0)
end
E_g_h = ad.select(opt.InBounds(0,0,1,1), ad.select(shading_h_valid, E_g_h_noCheck, 0.0), 0.0)

local shading_v_valid = ad.greater(D_i(0,-1) + D_i(0,0) + D_i(0,1) + D_i(-1,0) + D_i(-1,1), 0)

local E_g_v_noCheck = B(0,0) - B(0,1) - (I(0,0) - I(0,1))
if USE_MASK_REFINE then
E_g_v_noCheck = E_g_v_noCheck * edgeMaskC(0,0)
end
E_g_v = ad.select(opt.InBounds(0,0,1,1), ad.select(shading_v_valid, E_g_v_noCheck, 0.0), 0.0)
end
end


if USE_REGULARIZATION then
local cross_valid = ad.greater(D_i(0,0) + D_i(0,-1) + D_i(0,1) + D_i(-1,0) + D_i(1,0), 0)

local E_s_noCheck = sqMagnitude(plus(times(4.0,p(0,0)), times(-1.0, plus(p(-1,0), plus(p(0,-1), plus(p(1,0), p(0,1)))))))
--local E_s_noCheck = p(0,0)[1] --sqMagnitude(times(4.0,p(0,0)))
local d = X(0,0)

local E_s_guard =   ad.and_(ad.less(ad.abs(d - X(0,-1)), DEPTH_DISCONTINUITY_THRE),
ad.and_(ad.less(ad.abs(d - X(0,1)), DEPTH_DISCONTINUITY_THRE),
ad.and_(ad.less(ad.abs(d - X(-1,0)), DEPTH_DISCONTINUITY_THRE),
ad.and_(ad.less(ad.abs(d - X(1,0)), DEPTH_DISCONTINUITY_THRE),
ad.and_(opt.InBounds(0,0,1,1), cross_valid)
))))

E_s = ad.select(E_s_guard, E_s_noCheck, 0)

end

local cost = ad.sumsquared(w_g*E_g_h, w_g*E_g_v, w_s*E_s, w_p*E_p)
return P:Cost(cost)
*/
void CeresSolverSFS::solve(int _width, int _height, const std::vector<uint32_t>& elemsize, const std::vector<void*> &images, const TerraSolverParameterPointers &params)
{
    width = _width;
    height = _height;

    float *x = (float *)images[0];
    float *D_i = (float *)images[1];
    float *Im = (float *)images[2];
    float *D_p = (float *)images[3];
    BYTE *edgeMaskR = (BYTE *)images[4];
    BYTE *edgeMaskC = (BYTE *)images[5];

    auto getPixel = [=](int x, int y) {
        return y * width + x;
    };
    /*
    float w_p = sqrtf(params.floatPointers[0]);
    float w_s = sqrtf(params.floatPointers[1]);
    float w_r = sqrtf(params.floatPointers[2]);
    float w_g = sqrtf(params.floatPointers[3]);

    --See TerraSolverParameters
        local w_p = P:Param("w_p", float, 0)--Is initialized by the solver!
        local w_s = P : Param("w_s", float, 1)--Regularization weight
        local w_r = P : Param("w_r", float, 2)--Prior weight
        local w_g = P : Param("w_g", float, 3)--Shading weight

        w_p = ad.sqrt(w_p)
        w_s = ad.sqrt(w_s)
        w_r = ad.sqrt(w_r)
        w_g = ad.sqrt(w_g)


        local weightShadingStart = P:Param("weightShadingStart", float, 4)--Starting value for incremental relaxation
        local weightShadingIncrement = P : Param("weightShadingIncrement", float, 5)--Update factor

        local weightBoundary = P : Param("weightBoundary", float, 6)--Boundary weight

        local f_x = P : Param("f_x", float, 7)
        local f_y = P : Param("f_y", float, 8)
        local u_x = P : Param("u_x", float, 9)
        local u_y = P : Param("u_y", float, 10)

        local offset = 10;
    local deltaTransform = {}
        for i = 1, 16 do
            deltaTransform[i] = P:Param("deltaTransform_" ..i .. "", float, offset + i)
            end
            offset = offset + 16

            local L = {}
            for i = 1, 9 do
                L[i] = P:Param("L_" ..i .. "", float, offset + i)
                end*/
}

//
////--fitting
////local constraintUV = Constraints(0, 0)	--float2
////
////local e_fit = ad.select(ad.eq(m, 0.0), x - constraintUV, ad.Vector(0.0, 0.0))
////e_fit = ad.select(ad.greatereq(constraintUV(0), 0.0), e_fit, ad.Vector(0.0, 0.0))
////e_fit = ad.select(ad.greatereq(constraintUV(1), 0.0), e_fit, ad.Vector(0.0, 0.0))
////
////terms:insert(w_fitSqrt*e_fit)
//
///*
//Translation A:
//if(mask(i, j) == 0)
//fit = x - constraints(i, j)
//else
//fit = 0
//
//if(constaints(i, j).u < 0 || constaints(i, j).v < 0)
//fit = 0
//
//Translation B:
//if(mask(i, j) == 0 && constaints(i, j).u >= 0 && constaints(i, j).v >= 0)
//fit = (x(i, j) - constraints(i, j)) * w_fitSqrt
//*/
//
////--reg
////local xHat = UrShape(0,0)
////local i, j = unpack(o)
////local n = ad.Vector(X(i, j, 0), X(i, j, 1))
////local ARAPCost = (x - n) - mul(R, (xHat - UrShape(i, j)))
////local ARAPCostF = ad.select(opt.InBounds(0, 0, 0, 0), ad.select(opt.InBounds(i, j, 0, 0), ARAPCost, ad.Vector(0.0, 0.0)), ad.Vector(0.0, 0.0))
////local m = Mask(i, j)
////ARAPCostF = ad.select(ad.eq(m, 0.0), ARAPCostF, ad.Vector(0.0, 0.0))
////terms:insert(w_regSqrt*ARAPCostF)
//
///*
//Translation A:
//
//// ox,oy = offset x/y
//if(mask(i, j) == 0 && offset-in-bounds)
//cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
//*/
//
//vec2f toVec(const float2 &v)
//{
//    return vec2f(v.x, v.y);
//}
//
//template<class T> void evalR(const T angle, T matrix[4])
//{
//    T cosAlpha = cos(angle);
//    T sinAlpha = sin(angle);
//    matrix[0] = cosAlpha;
//    matrix[1] = -sinAlpha;
//    matrix[2] = sinAlpha;
//    matrix[3] = cosAlpha;
//}
//
//template<class T> void mul(const T matrix[4], const T vx, const T vy, T out[2])
//{
//    out[0] = matrix[0] * vx + matrix[1] * vy;
//    out[1] = matrix[2] * vx + matrix[3] * vy;
//}
//
//struct FitTerm
//{
//    FitTerm(const vec2f &_constraint, float _weight)
//        : constraint(_constraint), weight(_weight) {}
//
//    template <typename T>
//    bool operator()(const T* const x, T* residuals) const
//    {
//        residuals[0] = (x[0] - T(constraint.x)) * T(weight);
//        residuals[1] = (x[1] - T(constraint.y)) * T(weight);
//        return true;
//    }
//
//    static ceres::CostFunction* Create(const vec2f &constraint, float weight)
//    {
//        return (new ceres::AutoDiffCostFunction<FitTerm, 2, 2>(
//            new FitTerm(constraint, weight)));
//    }
//
//    vec2f constraint;
//    float weight;
//};
//
//struct RegTerm
//{
//    RegTerm(const vec2f &_deltaUr, float _weight)
//        : deltaUr(_deltaUr), weight(_weight) {}
//
//    template <typename T>
//    bool operator()(const T* const xA, const T* const xB, const T* const a, T* residuals) const
//    {
//        //cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
//        T R[4];
//        evalR(a[0], R);
//
//        T urOut[2];
//        mul(R, (T)deltaUr.x, (T)deltaUr.y, urOut);
//
//        T deltaX[2];
//        deltaX[0] = xA[0] - xB[0];
//        deltaX[1] = xA[1] - xB[1];
//
//        residuals[0] = (deltaX[0] - urOut[0]) * T(weight);
//        residuals[1] = (deltaX[1] - urOut[1]) * T(weight);
//        return true;
//    }
//
//    static ceres::CostFunction* Create(const vec2f &deltaUr, float weight)
//    {
//        return (new ceres::AutoDiffCostFunction<RegTerm, 2, 2, 2, 1>(
//            new RegTerm(deltaUr, weight)));
//    }
//
//    vec2f deltaUr;
//    float weight;
//};
//
//void CeresSolverWarping::solve(float2* h_x_float, float* h_a_float, float2* h_urshape, float2* h_constraints, float* h_mask, float weightFit, float weightReg)
//{
//    float weightFitSqrt = sqrt(weightFit);
//    float weightRegSqrt = sqrt(weightReg);
//
//    Problem problem;
//
//    auto getPixel = [=](int x, int y) {
//        return y * m_width + x;
//    };
//
//    const int pixelCount = m_width * m_height;
//    for (int i = 0; i < pixelCount; i++)
//    {
//        h_x_double[i].x = h_x_float[i].x;
//        h_x_double[i].y = h_x_float[i].y;
//        h_a_double[i] = h_a_float[i];
//    }
//
//    // add all fit constraints
//    //if (mask(i, j) == 0 && constaints(i, j).u >= 0 && constaints(i, j).v >= 0)
//    //    fit = (x(i, j) - constraints(i, j)) * w_fitSqrt
//    for (int y = 0; y < m_height; y++)
//    {
//        for (int x = 0; x < m_width; x++)
//        {
//            const float mask = h_mask[getPixel(x, y)];
//            const vec2f constraint = toVec(h_constraints[getPixel(x, y)]);
//            if (mask == 0.0f && constraint.x >= 0.0f && constraint.y >= 0.0f)
//            {
//                ceres::CostFunction* costFunction = FitTerm::Create(constraint, weightFitSqrt);
//                double2 *varStart = h_x_double + getPixel(x, y);
//                problem.AddResidualBlock(costFunction, NULL, (double*)varStart);
//            }
//        }
//    }
//
//    //add all reg constraints
//    //if(mask(i, j) == 0 && offset-in-bounds)
//    //  cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
//    for (int y = 0; y < m_height; y++)
//    {
//        for (int x = 0; x < m_width; x++)
//        {
//            const vec2i offsets[] = { vec2i(0, 1), vec2i(1, 0), vec2i(0, -1), vec2i(-1, 0) };
//            for (vec2i offset : offsets)
//            {
//                const float mask = h_mask[getPixel(x, y)];
//                const vec2i oPos = offset + vec2i(x, y);
//                if (mask == 0.0f && oPos.x >= 0 && oPos.x < m_width && oPos.y >= 0 && oPos.y < m_height)
//                {
//                    vec2f deltaUr = toVec(h_urshape[getPixel(x, y)]) - toVec(h_urshape[getPixel(oPos.x, oPos.y)]);
//                    ceres::CostFunction* costFunction = RegTerm::Create(deltaUr, weightRegSqrt);
//                    double2 *varStartA = h_x_double + getPixel(x, y);
//                    double2 *varStartB = h_x_double + getPixel(oPos.x, oPos.y);
//                    problem.AddResidualBlock(costFunction, NULL, (double*)varStartA, (double*)varStartB, h_a_double + getPixel(x, y));
//                }
//            }
//        }
//    }
//
//    cout << "Solving..." << endl;
//
//    Solver::Options options;
//    Solver::Summary summary;
//
//    options.minimizer_progress_to_stdout = !performanceTest;
//    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
//    options.linear_solver_type = ceres::LinearSolverType::CGNR;
//    //options.min_linear_solver_iterations = linearIterationMin;
//    options.max_num_iterations = 10000;
//    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
//    //options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR;
//
//    //options.min_lm_diagonal = 1.0f;
//    //options.min_lm_diagonal = options.max_lm_diagonal;
//    //options.max_lm_diagonal = 10000000.0;
//
//    //problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
//    //cout << "Cost*2 start: " << cost << endl;
//
//    double elapsedTime;
//    {
//        ml::Timer timer;
//        Solve(options, &problem, &summary);
//        elapsedTime = timer.getElapsedTimeMS();
//    }
//
//    cout << "Solver used: " << summary.linear_solver_type_used << endl;
//    cout << "Minimizer iters: " << summary.iterations.size() << endl;
//    cout << "Total time: " << elapsedTime << "ms" << endl;
//
//    double iterationTotalTime = 0.0;
//    int totalLinearItereations = 0;
//    for (auto &i : summary.iterations)
//    {
//        iterationTotalTime += i.iteration_time_in_seconds;
//        totalLinearItereations += i.linear_solver_iterations;
//        cout << "Iteration: " << i.linear_solver_iterations << " " << i.iteration_time_in_seconds * 1000.0 << "ms" << endl;
//    }
//
//    cout << "Total iteration time: " << iterationTotalTime << endl;
//    cout << "Cost per linear solver iteration: " << iterationTotalTime * 1000.0 / totalLinearItereations << "ms" << endl;
//
//    double cost = -1.0;
//    problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
//    cout << "Cost*2 end: " << cost * 2 << endl;
//
//    cout << summary.FullReport() << endl;
//
//    for (int i = 0; i < pixelCount; i++)
//    {
//        h_x_float[i].x = h_x_double[i].x;
//        h_x_float[i].y = h_x_double[i].y;
//        h_a_float[i] = h_a_double[i];
//    }
//}

#endif