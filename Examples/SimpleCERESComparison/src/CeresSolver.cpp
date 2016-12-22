#pragma once

#include <cuda_runtime.h>

#include "config.h"

#include "CeresSolver.h"

#include "ceres/ceres.h"
#include "glog/logging.h"

using ceres::DynamicAutoDiffCostFunction;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
using namespace std;

struct TermDefault
{
	TermDefault(double x, double y)
        : x(x), y(y) {}

    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        
        residuals[0] = y - (funcParams[0] * cos(funcParams[1] * x) + funcParams[1] * sin(funcParams[0]*x));
        return true;
    }

    static ceres::CostFunction* Create(double x, double y)
    {
		return (new ceres::AutoDiffCostFunction<TermDefault, 1, 2>(
			new TermDefault(x, y)));
    }

    double x;
    double y;
};
struct TermBennett5
{
    TermBennett5(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1 * (b2+x)**(-1/b3)  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        residuals[0] = y - b1 * pow(b2 + x, (T)-1.0 / b3);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermBennett5, 1, 3>(
            new TermBennett5(x, y)));
    }
    double x, y;
};

struct TermBoxBOD
{
    TermBoxBOD(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1*(1-exp[-b2*x])  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        residuals[0] = y - b1*((T)1.0 - exp(-b2*x));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermBoxBOD, 1, 2>(
            new TermBoxBOD(x, y)));
    }
    double x, y;
};

struct TermChwirut1
{
    TermChwirut1(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = exp[-b1*x]/(b2+b3*x)  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        residuals[0] = y - exp(-b1*x) / (b2 + b3*x);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermChwirut1, 1, 3>(
            new TermChwirut1(x, y)));
    }
    double x, y;
};

struct TermChwirut2
{
    TermChwirut2(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = exp(-b1*x)/(b2+b3*x)  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        residuals[0] = y - exp(-b1*x) / (b2 + b3*x);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermChwirut2, 1, 3>(
            new TermChwirut2(x, y)));
    }
    double x, y;
};

struct TermDanWood
{
    TermDanWood(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y  = b1*x**b2  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        residuals[0] = y - pow(b1*x, b2);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermDanWood, 1, 2>(
            new TermDanWood(x, y)));
    }
    double x, y;
};

struct TermEckerle4
{
    TermEckerle4(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = (b1/b2) * exp[-0.5*((x-b3)/b2)**2]  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        residuals[0] = y - (b1 / b2) * exp(pow(-(T)0.5*((x - b3) / b2), 2));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermEckerle4, 1, 3>(
            new TermEckerle4(x, y)));
    }
    double x, y;
};

struct TermENSO
{
    TermENSO(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1 + b2*cos( 2*pi*x/12 ) + b3*sin( 2*pi*x/12 )
        + b5*cos( 2*pi*x/b4 ) + b6*sin( 2*pi*x/b4 )
        + b8*cos( 2*pi*x/b7 ) + b9*sin( 2*pi*x/b7 )  + e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        T b6 = funcParams[5];
        T b7 = funcParams[6];
        T b8 = funcParams[7];
        T b9 = funcParams[8];
        const double pi = 3.141592653589793238462643383279;
        residuals[0] = y - b1 + b2*cos((T)2 * pi*x / (T)12) + b3*sin((T)2 * pi*x / (T)12)
            + b5*cos((T)2 * pi*x / b4) + b6*sin((T)2 * pi*x / b4)
            + b8*cos((T)2 * pi*x / b7) + b9*sin((T)2 * pi*x / b7);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermENSO, 1, 9>(
            new TermENSO(x, y)));
    }
    double x, y;
};

struct TermGauss1
{
    TermGauss1(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1*exp( -b2*x ) + b3*exp( -(x-b4)**2 / b5**2 )
        + b6*exp( -(x-b7)**2 / b8**2 ) + e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        T b6 = funcParams[5];
        T b7 = funcParams[6];
        T b8 = funcParams[7];
        residuals[0] = y - b1*exp(-b2*x) + b3*exp(-((x - b4)*(x - b4)) / (b5*b5))
            + b6*exp(-((x - b7)*(x - b7)) / (b8*b8));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermGauss1, 1, 8>(
            new TermGauss1(x, y)));
    }
    double x, y;
};

struct TermGauss2
{
    TermGauss2(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1*exp( -b2*x ) + b3*exp( -(x-b4)**2 / b5**2 )
        + b6*exp( -(x-b7)**2 / b8**2 ) + e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        T b6 = funcParams[5];
        T b7 = funcParams[6];
        T b8 = funcParams[7];
        residuals[0] = y - b1*exp(-b2*x) + b3*exp(-((x - b4)*(x - b4)) / (b5*b5))
            + b6*exp(-((x - b7)*(x - b7)) / (b8*b8));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermGauss2, 1, 8>(
            new TermGauss2(x, y)));
    }
    double x, y;
};

struct TermGauss3
{
    TermGauss3(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1*exp( -b2*x ) + b3*exp( -(x-b4)**2 / b5**2 )
        + b6*exp( -(x-b7)**2 / b8**2 ) + e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        T b6 = funcParams[5];
        T b7 = funcParams[6];
        T b8 = funcParams[7];
        residuals[0] = y - b1*exp(-b2*x) + b3*exp(-((x - b4)*(x - b4)) / (b5*b5))
            + b6*exp(-((x - b7)*(x - b7)) / (b8*b8));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermGauss3, 1, 8>(
            new TermGauss3(x, y)));
    }
    double x, y;
};

struct TermHahn1
{
    TermHahn1(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = (b1+b2*x+b3*x**2+b4*x**3) /
        (1+b5*x+b6*x**2+b7*x**3)  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        T b6 = funcParams[5];
        T b7 = funcParams[6];
        residuals[0] = y - (b1 + b2*x + b3*x*x + b4*x*x*x) /
            ((T)1 + b5*x + b6*x*x + b7*x*x*x);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermHahn1, 1, 7>(
            new TermHahn1(x, y)));
    }
    double x, y;
};

struct TermKirby2
{
    TermKirby2(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = (b1 + b2*x + b3*x**2) /
        (1 + b4*x + b5*x**2)  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        residuals[0] = y - (b1 + b2*x + b3*x*x) /
            ((T)1 + b4*x + b5*x*x);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermKirby2, 1, 5>(
            new TermKirby2(x, y)));
    }
    double x, y;
};

struct TermLanczos1
{
    TermLanczos1(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        T b6 = funcParams[5];
        residuals[0] = y - b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermLanczos1, 1, 6>(
            new TermLanczos1(x, y)));
    }
    double x, y;
};

struct TermLanczos2
{
    TermLanczos2(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        T b6 = funcParams[5];
        residuals[0] = y - b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermLanczos2, 1, 6>(
            new TermLanczos2(x, y)));
    }
    double x, y;
};

struct TermLanczos3
{
    TermLanczos3(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x)  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        T b6 = funcParams[5];
        residuals[0] = y - b1*exp(-b2*x) + b3*exp(-b4*x) + b5*exp(-b6*x);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermLanczos3, 1, 6>(
            new TermLanczos3(x, y)));
    }
    double x, y;
};

struct TermMGH09
{
    TermMGH09(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1*(x**2+x*b2) / (x**2+x*b3+b4)  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        residuals[0] = y - b1*(x*x + x*b2) / (x*x + x*b3 + b4);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermMGH09, 1, 4>(
            new TermMGH09(x, y)));
    }
    double x, y;
};

struct TermMGH10
{
    TermMGH10(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1 / (1+exp[b2-b3*x])  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        residuals[0] = y - b1 / ((T)1 + exp(b2 - b3*x));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermMGH10, 1, 3>(
            new TermMGH10(x, y)));
    }
    double x, y;
};

struct TermMGH17
{
    TermMGH17(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1 + b2*exp[-x*b4] + b3*exp[-x*b5]  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        residuals[0] = y - b1 + b2*exp(-x*b4) + b3*exp(-x*b5);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermMGH17, 1, 5>(
            new TermMGH17(x, y)));
    }
    double x, y;
};

struct TermMisra1a
{
    TermMisra1a(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1*(1-exp[-b2*x])  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        residuals[0] = y - b1*((T)1 - exp(-b2*x));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermMisra1a, 1, 2>(
            new TermMisra1a(x, y)));
    }
    double x, y;
};

struct TermMisra1b
{
    TermMisra1b(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1 * (1-(1+b2*x/2)**(-2))  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        residuals[0] = y - b1 * ((T)1 - (T)1.0 / (((T)1 + b2*x / (T)2)*((T)1 + b2*x / (T)2)));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermMisra1b, 1, 2>(
            new TermMisra1b(x, y)));
    }
    double x, y;
};

struct TermMisra1c
{
    TermMisra1c(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1 * (1-(1+2*b2*x)**(-.5))  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        residuals[0] = y - b1 * ((T)1 - (T)1.0 / sqrt((T)1 + (T)2 * b2*x));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermMisra1c, 1, 2>(
            new TermMisra1c(x, y)));
    }
    double x, y;
};

struct TermMisra1d
{
    TermMisra1d(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1*b2*x*((1+b2*x)**(-1))  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        residuals[0] = y - b1*b2*x/((T)1 + b2*x);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermMisra1d, 1, 2>(
            new TermMisra1d(x, y)));
    }
    double x, y;
};
/*
struct TermNelson
{
    TermNelson(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        // log[y] = b1 - b2*x1 * exp[-b3*x2]  +  e 
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        residuals[0] = log(y) - b1 - b2*x1 * exp(-b3*x2);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermNelson, 1, 3>(
            new TermNelson(x, y)));
    }
    double x, y;
};
*/
struct TermRat42
{
    TermRat42(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1 / (1+exp[b2-b3*x])  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        residuals[0] = y - b1 / ((T)1 + exp(b2 - b3*x));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermRat42, 1, 3>(
            new TermRat42(x, y)));
    }
    double x, y;
};

struct TermRat43
{
    TermRat43(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = b1 / ((1+exp[b2-b3*x])**(1/b4))  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        residuals[0] = y - b1 / pow(((T)1 + exp(b2 - b3*x)), ((T)1.0 / b4));
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermRat43, 1, 4>(
            new TermRat43(x, y)));
    }
    double x, y;
};

struct TermRoszman1
{
    TermRoszman1(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* pi = 3.141592653589793238462643383279E0
        y =  b1 - b2*x - arctan[b3/(x-b4)]/pi  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        const double pi = 3.141592653589793238462643383279;
        residuals[0] = 
            y - b1 - b2*x - atan(b3 / (x - b4)) / pi;
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermRoszman1, 1, 4>(
            new TermRoszman1(x, y)));
    }
    double x, y;
};

struct TermThurber
{
    TermThurber(double x, double y) : x(x), y(y) {}
    template <typename T>
    bool operator()(const T* const funcParams, T* residuals) const
    {
        /* y = (b1 + b2*x + b3*x**2 + b4*x**3) /
        (1 + b5*x + b6*x**2 + b7*x**3)  +  e */
        T b1 = funcParams[0];
        T b2 = funcParams[1];
        T b3 = funcParams[2];
        T b4 = funcParams[3];
        T b5 = funcParams[4];
        T b6 = funcParams[5];
        T b7 = funcParams[6];
        residuals[0] = y - (b1 + b2*x + b3*x*x + b4*x*x*x) /
            ((T)1 + b5*x + b6*x*x + b7*x*x*x);
        return true;
    }
    static ceres::CostFunction* Create(double x, double y)
    {
        return (new ceres::AutoDiffCostFunction<TermThurber, 1, 7>(
            new TermThurber(x, y)));
    }
    double x, y;
};


std::vector<SolverIteration> CeresSolver::solve(
	const NLLSProblem &problemInfo,
    UNKNOWNS* funcParameters,
    double2* funcData)
{

    std::vector<SolverIteration> result;
    for (int i = 0; i < functionData.size(); i++)
    {
        functionData[i].x = funcData[i].x;
        functionData[i].y = funcData[i].y;

    }

    Problem problem;
    for (int i = 0; i < functionData.size(); i++)
    {
		ceres::CostFunction* costFunction = nullptr;

		if (useProblemDefault) costFunction = TermDefault::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "bennett5") costFunction = TermBennett5::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "boxbod") costFunction = TermBoxBOD::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "chwirut1") costFunction = TermChwirut1::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "chwirut2") costFunction = TermChwirut2::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "danwood") costFunction = TermDanWood::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "eckerle4") costFunction = TermEckerle4::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "enso") costFunction = TermENSO::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "gauss1") costFunction = TermGauss1::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "gauss2") costFunction = TermGauss2::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "gauss3") costFunction = TermGauss3::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "hahn1") costFunction = TermHahn1::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "kirby2") costFunction = TermKirby2::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "lanczos1") costFunction = TermLanczos1::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "lanczos2") costFunction = TermLanczos2::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "lanczos3") costFunction = TermLanczos3::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "mgh09") costFunction = TermMGH09::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "mgh10") costFunction = TermMGH10::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "mgh17") costFunction = TermMGH17::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "misra1a") costFunction = TermMisra1a::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "misra1b") costFunction = TermMisra1b::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "misra1c") costFunction = TermMisra1c::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "misra1d") costFunction = TermMisra1d::Create(functionData[i].x, functionData[i].y);
        //if (problemInfo.baseName == "nelson") costFunction = TermNelson::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "rat42") costFunction = TermRat42::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "rat43") costFunction = TermRat43::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "roszman1") costFunction = TermRoszman1::Create(functionData[i].x, functionData[i].y);
        if (problemInfo.baseName == "thurber") costFunction = TermThurber::Create(functionData[i].x, functionData[i].y);

		if (costFunction == nullptr)
		{
			cout << "No problem specified!" << endl;
            return result;
		}
		problem.AddResidualBlock(costFunction, NULL, (double*)funcParameters);
    }
    
    //ceres::CostFunction* regCostFunction = HackRegularizerTerm::Create(1.0);
    //problem.AddResidualBlock(regCostFunction, NULL, (double*)funcParameters);
    
    cout << "Solving..." << endl;

    Solver::Options options;
    Solver::Summary summary;

    //shut off annoying output
    options.minimizer_progress_to_stdout = false;
    options.logging_type = ceres::SILENT;

    //faster methods
    options.num_threads = 8;
    options.num_linear_solver_threads = 8;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY; //7.2s
    //options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; //10.0s

    //slower methods
    //options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR; //40.6s
   // options.linear_solver_type = ceres::LinearSolverType::CGNR; //46.9s
	//options.linear_solver_type = ceres::LinearSolverType::DENSE_NORMAL_CHOLESKY;

    //options.min_linear_solver_iterations = linearIterationMin;
    options.max_num_iterations = 10000;
    options.function_tolerance = 1e-20;
    options.gradient_tolerance = 1e-10 * options.function_tolerance;

    // Default values, reproduced here for clarity
    //options.trust_region_strategy_type = ceres::TrustRegionStrategyType::LEVENBERG_MARQUARDT;
    options.initial_trust_region_radius = 1e4;
    options.max_trust_region_radius = 1e16;
    options.min_trust_region_radius = 1e-32;
    options.min_relative_decrease = 1e-3;
    // Disable to match Opt
    //options.min_lm_diagonal = 1e-32;
    //options.max_lm_diagonal = std::numeric_limits<double>::infinity();
    //options.min_trust_region_radius = 1e-256;

    //options.initial_trust_region_radius = 0.005;

    options.initial_trust_region_radius = 1e4;
    options.eta = 1e-4;

    options.jacobi_scaling = true;
    //options.preconditioner_type = ceres::PreconditionerType::IDENTITY;

    Solve(options, &problem, &summary);

	for (auto &i : summary.iterations)
	{
		SolverIteration iter;
		iter.cost = i.cost;
        iter.timeInMS = i.iteration_time_in_seconds * 1000.0;
		result.push_back(iter);
	}

    cout << "Solver used: " << summary.linear_solver_type_used << endl;
    cout << "Minimizer iters: " << summary.iterations.size() << endl;

    double iterationTotalTime = 0.0;
    int totalLinearItereations = 0;
    for (auto &i : summary.iterations)
    {
        iterationTotalTime += i.iteration_time_in_seconds;
        totalLinearItereations += i.linear_solver_iterations;
        //cout << "Iteration: " << i.linear_solver_iterations << " " << i.iteration_time_in_seconds * 1000.0 << "ms" << endl;
    }

    cout << "Total iteration time: " << iterationTotalTime << endl;
    cout << "Cost per linear solver iteration: " << iterationTotalTime * 1000.0 / totalLinearItereations << "ms" << endl;

    double cost = -1.0;
    problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
    cout << "Cost*2 end: " << cost * 2 << endl;

    cout << summary.FullReport() << endl;

	return result;
}

