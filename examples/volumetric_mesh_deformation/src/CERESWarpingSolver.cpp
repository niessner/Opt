#include "CERESWarpingSolver.h"

#ifdef USE_CERES

#include "mLibInclude.h"

const bool performanceTest = false;
//const int linearIterationMin = 100;

#include <cuda_runtime.h>
#include "../../shared/Precision.h"
#include "../../shared/SolverIteration.h"


#define GLOG_NO_ABBREVIATED_SEVERITIES
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
local x = Offset(0,0,0)

--fitting
local constraint = Constraints(0,0,0)	-- float3

local e_fit = x - constraint
e_fit = ad.select(ad.greatereq(constraint(0), -999999.9), e_fit, ad.Vector(0.0, 0.0, 0.0))
terms:insert(w_fitSqrt*e_fit)
*/

struct FitTerm
{
	FitTerm(const vec3f &_constraint, float _weight)
		: constraint(_constraint), weight(_weight) {}

	template <typename T>
	bool operator()(const T* const x, T* residuals) const
	{
		residuals[0] = (x[0] - T(constraint.x)) * T(weight);
		residuals[1] = (x[1] - T(constraint.y)) * T(weight);
		residuals[2] = (x[2] - T(constraint.z)) * T(weight);
		return true;
	}

	static ceres::CostFunction* Create(const vec3f &constraint, float weight)
	{
		return (new ceres::AutoDiffCostFunction<FitTerm, 3, 3>(
			new FitTerm(constraint, weight)));
	}

	vec3f constraint;
	float weight;
};

template<class T>
struct vec3T
{
	vec3T() {}
	vec3T(T _x, T _y, T _z)
	{
		x = _x;
		y = _y;
		z = _z;
	}
	T sqMagintude()
	{
		return x * x + y * y + z * z;
	}
	vec3T operator * (T v)
	{
		return vec3T(v * x, v * y, v * z);
	}
	vec3T operator + (const vec3T &v)
	{
		return vec3T(x + v.x, y + v.y, z + v.z);
	}
	const T& operator [](int k) const
	{
		if (k == 0) return x;
		if (k == 1) return y;
		if (k == 2) return z;
		return x;
	}
	T x, y, z;
};

template<class T>
void evalRot(T CosAlpha, T CosBeta, T CosGamma, T SinAlpha, T SinBeta, T SinGamma, T R[9])
{
	R[0] = CosGamma*CosBeta;
	R[1] = -SinGamma*CosAlpha + CosGamma*SinBeta*SinAlpha;
	R[2] = SinGamma*SinAlpha + CosGamma*SinBeta*CosAlpha;
	R[3] = SinGamma*CosBeta;
	R[4] = CosGamma*CosAlpha + SinGamma*SinBeta*SinAlpha;
	R[5] = -CosGamma*SinAlpha + SinGamma*SinBeta*CosAlpha;
	R[6] = -SinBeta;
	R[7] = CosBeta*SinAlpha;
	R[8] = CosBeta*CosAlpha;
}

template<class T>
void evalR(T alpha, T beta, T gamma, T R[9])
{
	evalRot(cos(alpha), cos(beta), cos(gamma), sin(alpha), sin(beta), sin(gamma), R);
}

template<class T>
vec3T<T> mul(T matrix[9], const vec3T<T> &v)
{
	vec3T<T> result;
	result.x = matrix[0] * v[0] + matrix[1] * v[1] + matrix[2] * v[2];
	result.y = matrix[3] * v[0] + matrix[4] * v[1] + matrix[5] * v[2];
	result.z = matrix[6] * v[0] + matrix[7] * v[1] + matrix[8] * v[2];
	return result;
}

/*
--regularization
local a = Angle(0,0,0)				-- rotation : float3
local R = evalR(a(0), a(1), a(2))	-- rotation : float3x3
local xHat = UrShape(0,0,0)			-- uv-urshape : float3

local offsets = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0,0,-1}}
for iii ,o in ipairs(offsets) do
	local i,j,k = unpack(o)
	local n = Offset(i,j,k)

	local ARAPCost = (x - n) - mul(R, (xHat - UrShape(i,j,k)))
	local ARAPCostF = ad.select(opt.InBounds(0,0,0),	ad.select(opt.InBounds(i,j,k), ARAPCost, ad.Vector(0.0, 0.0, 0.0)), ad.Vector(0.0, 0.0, 0.0))
	terms:insert(w_regSqrt*ARAPCostF)
end
*/

struct RegTerm
{
	RegTerm(const vec3f &_deltaUr, float _weight)
		: deltaUr(_deltaUr), weight(_weight) {}

	template <typename T>
	bool operator()(const T* const xA, const T* const xB, const T* const a, T* residuals) const
	{
		//cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
		T R[9];
		evalR(a[0], a[1], a[2], R);

		vec3T<T> urOut = mul(R, vec3T<T>(T(deltaUr.x), T(deltaUr.y), T(deltaUr.z)));

		T deltaX[3];
		deltaX[0] = xA[0] - xB[0];
		deltaX[1] = xA[1] - xB[1];
		deltaX[2] = xA[2] - xB[2];

		residuals[0] = (deltaX[0] - urOut[0]) * T(weight);
		residuals[1] = (deltaX[1] - urOut[1]) * T(weight);
		residuals[2] = (deltaX[2] - urOut[2]) * T(weight);
		return true;
	}

	static ceres::CostFunction* Create(const vec3f &deltaUr, float weight)
	{
		return (new ceres::AutoDiffCostFunction<RegTerm, 3, 3, 3, 3>(
			new RegTerm(deltaUr, weight)));
	}

	vec3f deltaUr;
	float weight;
};

vec3f toVec(const float3 &v)
{
	return vec3f(v.x, v.y, v.z);
}

CERESWarpingSolver::CERESWarpingSolver(unsigned int width, unsigned int height, unsigned int depth)
{
	m_width = width;
	m_height = height;
	m_depth = depth;

	voxelCount = m_width * m_height * m_depth;
	vertexPosDouble3 = new vec3d[voxelCount];
	anglesDouble3 = new vec3d[voxelCount];

	h_vertexPosFloat3 = new float3[voxelCount];
	h_anglesFloat3 = new float3[voxelCount];
	h_vertexPosFloat3Urshape = new float3[voxelCount];
	h_vertexPosTargetFloat3 = new float3[voxelCount];
}

CERESWarpingSolver::~CERESWarpingSolver()
{

}

void CERESWarpingSolver::solve(
	float3* d_vertexPosFloat3,
	float3* d_anglesFloat3,
	float3* d_vertexPosFloat3Urshape,
	float3* d_vertexPosTargetFloat3,
	float weightFit,
	float weightReg,
    std::vector<SolverIteration>& iters)
{
	float weightFitSqrt = sqrt(weightFit);
	float weightRegSqrt = sqrt(weightReg);

	cutilSafeCall(cudaMemcpy(h_vertexPosFloat3, d_vertexPosFloat3, sizeof(float3)*voxelCount, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_anglesFloat3, d_anglesFloat3, sizeof(float3)*voxelCount, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_vertexPosFloat3Urshape, d_vertexPosFloat3Urshape, sizeof(float3)*voxelCount, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(h_vertexPosTargetFloat3, d_vertexPosTargetFloat3, sizeof(float3)*voxelCount, cudaMemcpyDeviceToHost));

	auto getVoxel = [=](int x, int y, int z) {
		return z * m_width * m_height + y * m_width + x;
	};
	auto voxelValid = [=](int x, int y, int z) {
		return (x >= 0 && x < (int)m_width) &&
			(y >= 0 && y < (int)m_height) &&
			(z >= 0 && z < (int)m_depth);
	};

	for (int i = 0; i < (int)voxelCount; i++)
	{
		vertexPosDouble3[i].x = h_vertexPosFloat3[i].x;
		vertexPosDouble3[i].y = h_vertexPosFloat3[i].y;
		vertexPosDouble3[i].z = h_vertexPosFloat3[i].z;
		anglesDouble3[i].x = h_anglesFloat3[i].x;
		anglesDouble3[i].y = h_anglesFloat3[i].y;
		anglesDouble3[i].z = h_anglesFloat3[i].z;
	}

	Problem problem;

	// add all fit constraints
	//if (mask(i, j) == 0 && constaints(i, j).u >= 0 && constaints(i, j).v >= 0)
	//    fit = (x(i, j) - constraints(i, j)) * w_fitSqrt
	for (int i = 0; i < (int)voxelCount; i++)
	{
		const vec3f constraint = toVec(h_vertexPosTargetFloat3[i]);
		if (constraint.x > -999999.9f)
		{
			ceres::CostFunction* costFunction = FitTerm::Create(constraint, weightFitSqrt);
			vec3d *varStart = vertexPosDouble3 + i;
			problem.AddResidualBlock(costFunction, NULL, (double *)varStart);
		}
	}

	//local offsets = { { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 } }
	vector<vec3i> offsets;
	offsets.push_back(vec3i(1, 0, 0));
	offsets.push_back(vec3i(-1, 0, 0));
	offsets.push_back(vec3i(0, 1, 0));
	offsets.push_back(vec3i(0, -1, 0));
	offsets.push_back(vec3i(0, 0, 1));
	offsets.push_back(vec3i(0, 0, -1));

	for (int x = 0; x < (int)m_width; x++)
		for (int y = 0; y < (int)m_height; y++)
			for (int z = 0; z < (int)m_depth; z++)
			{
				for (vec3i o : offsets)
				{
					const int myIndex = getVoxel(x, y, z);
					const int neighborIndex = getVoxel(x + o.x, y + o.y, z + o.z);
					if (!voxelValid(x + o.x, y + o.y, z + o.z)) continue;

					//const vec3f constraintA = toVec(h_vertexPosTargetFloat3[myIndex]);
					//const vec3f constraintB = toVec(h_vertexPosTargetFloat3[neighborIndex]);
					//if (constraintA.x > -999999.0f && constraintB.x > -999999.0f)
					{
						const vec3f deltaUr = toVec(h_vertexPosFloat3Urshape[myIndex]) - toVec(h_vertexPosFloat3Urshape[neighborIndex]);
						ceres::CostFunction* costFunction = RegTerm::Create(deltaUr, weightRegSqrt);
						vec3d *varStartA = vertexPosDouble3 + myIndex;
						vec3d *varStartB = vertexPosDouble3 + neighborIndex;
						vec3d *angleStartA = anglesDouble3 + myIndex;
						problem.AddResidualBlock(costFunction, NULL, (double*)varStartA, (double*)varStartB, (double*)angleStartA);
					}
				}
			}

	cout << "Solving..." << endl;

	Solver::Options options;
	Solver::Summary summary;

	options.minimizer_progress_to_stdout = !performanceTest;

	//faster methods
	options.num_threads = 8;
	options.num_linear_solver_threads = 8;
	options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY; //7.2s
	//options.trust_region_strategy_type = ceres::TrustRegionStrategyType::DOGLEG;
	//options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; //10.0s

	//slower methods
	//options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR; //40.6s
	//options.linear_solver_type = ceres::LinearSolverType::CGNR; //46.9s

	//options.min_linear_solver_iterations = linearIterationMin;
	options.max_num_iterations = 10000;
	options.function_tolerance = 1e-3;
	options.gradient_tolerance = 1e-4 * options.function_tolerance;

	//options.min_lm_diagonal = 1.0f;
	//options.min_lm_diagonal = options.max_lm_diagonal;
	//options.max_lm_diagonal = 10000000.0;

	//problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
	//cout << "Cost*2 start: " << cost << endl;

	double elapsedTime;
	{
		ml::Timer timer;
		Solve(options, &problem, &summary);
		elapsedTime = timer.getElapsedTimeMS();
	}

	cout << "Solver used: " << summary.linear_solver_type_used << endl;
	cout << "Minimizer iters: " << summary.iterations.size() << endl;
	cout << "Total time: " << elapsedTime << "ms" << endl;

	double iterationTotalTime = 0.0;
	int totalLinearItereations = 0;
	for (auto &i : summary.iterations)
	{
		iterationTotalTime += i.iteration_time_in_seconds;
		totalLinearItereations += i.linear_solver_iterations;
		cout << "Iteration: " << i.linear_solver_iterations << " " << i.iteration_time_in_seconds * 1000.0 << "ms," << " cost: " << i.cost << endl;
	}

    for (auto &i : summary.iterations) {
        iters.push_back(SolverIteration(i.cost, i.iteration_time_in_seconds * 1000.0));
    }


	cout << "Total iteration time: " << iterationTotalTime << endl;
	cout << "Cost per linear solver iteration: " << iterationTotalTime * 1000.0 / totalLinearItereations << "ms" << endl;

	double cost = -1.0;
	problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
	cout << "Cost*2 end: " << cost * 2 << endl;

    m_finalCost = cost;

	cout << summary.FullReport() << endl;

	for (int i = 0; i < (int)voxelCount; i++)
	{
		h_vertexPosFloat3[i].x = (float)vertexPosDouble3[i].x;
		h_vertexPosFloat3[i].y = (float)vertexPosDouble3[i].y;
		h_vertexPosFloat3[i].z = (float)vertexPosDouble3[i].z;
		h_anglesFloat3[i].x = (float)anglesDouble3[i].x;
		h_anglesFloat3[i].y = (float)anglesDouble3[i].y;
		h_anglesFloat3[i].z = (float)anglesDouble3[i].z;
	}

	cutilSafeCall(cudaMemcpy(d_vertexPosFloat3, h_vertexPosFloat3, sizeof(float3)*voxelCount, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_anglesFloat3, h_anglesFloat3, sizeof(float3)*voxelCount, cudaMemcpyHostToDevice));

	//return (float)(summary.total_time_in_seconds * 1000.0);
}

/*
template<class T> void evalR(const T angle, T matrix[4])
{
	T cosAlpha = cos(angle);
	T sinAlpha = sin(angle);
	matrix[0] = cosAlpha;
	matrix[1] = -sinAlpha;
	matrix[2] = sinAlpha;
	matrix[3] = cosAlpha;
}

template<class T> void mul(const T matrix[4], const T vx, const T vy, T out[2])
{
	out[0] = matrix[0] * vx + matrix[1] * vy;
	out[1] = matrix[2] * vx + matrix[3] * vy;
}

struct FitTerm
{
	FitTerm(const vec2f &_constraint, float _weight)
		: constraint(_constraint), weight(_weight) {}

	template <typename T>
	bool operator()(const T* const x, T* residuals) const
	{
		residuals[0] = (x[0] - T(constraint.x)) * T(weight);
		residuals[1] = (x[1] - T(constraint.y)) * T(weight);
		return true;
	}

	static ceres::CostFunction* Create(const vec2f &constraint, float weight)
	{
		return (new ceres::AutoDiffCostFunction<FitTerm, 2, 2>(
			new FitTerm(constraint, weight)));
	}

	vec2f constraint;
	float weight;
};

struct RegTerm
{
	RegTerm(const vec2f &_deltaUr, float _weight)
		: deltaUr(_deltaUr), weight(_weight) {}

	template <typename T>
	bool operator()(const T* const xA, const T* const xB, const T* const a, T* residuals) const
	{
		//cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
		T R[4];
		evalR(a[0], R);

		T urOut[2];
		mul(R, (T)deltaUr.x, (T)deltaUr.y, urOut);

		T deltaX[2];
		deltaX[0] = xA[0] - xB[0];
		deltaX[1] = xA[1] - xB[1];

		residuals[0] = (deltaX[0] - urOut[0]) * T(weight);
		residuals[1] = (deltaX[1] - urOut[1]) * T(weight);
		return true;
	}

	static ceres::CostFunction* Create(const vec2f &deltaUr, float weight)
	{
		return (new ceres::AutoDiffCostFunction<RegTerm, 2, 2, 2, 1>(
			new RegTerm(deltaUr, weight)));
	}

	vec2f deltaUr;
	float weight;
};

float CeresSolverWarping::solve(OPT_FLOAT2* h_x_float, OPT_FLOAT* h_a_float, OPT_FLOAT2* h_urshape, OPT_FLOAT2* h_constraints, OPT_FLOAT* h_mask, float weightFit, float weightReg, std::vector<SolverIteration>& result)
{
	float weightFitSqrt = sqrt(weightFit);
	float weightRegSqrt = sqrt(weightReg);

	Problem problem;

	auto getPixel = [=](int x, int y) {
		return y * m_width + x;
	};

	const int pixelCount = m_width * m_height;
	for (int i = 0; i < pixelCount; i++)
	{
		h_x_double[i].x = h_x_float[i].x;
		h_x_double[i].y = h_x_float[i].y;
		h_a_double[i] = h_a_float[i];
	}

	// add all fit constraints
	//if (mask(i, j) == 0 && constaints(i, j).u >= 0 && constaints(i, j).v >= 0)
	//    fit = (x(i, j) - constraints(i, j)) * w_fitSqrt
	for (int y = 0; y < m_height; y++)
	{
		for (int x = 0; x < m_width; x++)
		{
			float mask = h_mask[getPixel(x, y)];
			const vec2f constraint = toVec(h_constraints[getPixel(x, y)]);
			if (mask == 0.0f && constraint.x >= 0.0f && constraint.y >= 0.0f)
			{
				ceres::CostFunction* costFunction = FitTerm::Create(constraint, weightFitSqrt);
				double2 *varStart = h_x_double + getPixel(x, y);
				problem.AddResidualBlock(costFunction, NULL, (double*)varStart);
			}
		}
	}

	//add all reg constraints
	//if(mask(i, j) == 0 && offset-in-bounds)
	//  cost = (x(i, j) - x(i + ox, j + oy)) - mul(R, urshape(i, j) - urshape(i + ox, j + oy)) * w_regSqrt
	for (int y = 0; y < m_height; y++)
	{
		for (int x = 0; x < m_width; x++)
		{
			float mask = h_mask[getPixel(x, y)];
			if (mask == 0.0f) {
				const vec2i offsets[] = { vec2i(0, 1), vec2i(1, 0), vec2i(0, -1), vec2i(-1, 0) };
				for (vec2i offset : offsets)
				{
					const vec2i oPos = offset + vec2i(x, y);
					float innerMask = h_mask[getPixel(oPos.x, oPos.y)];
					if (innerMask == 0.0f && oPos.x >= 0 && oPos.x < m_width && oPos.y >= 0 && oPos.y < m_height)
					{
						vec2f deltaUr = toVec(h_urshape[getPixel(x, y)]) - toVec(h_urshape[getPixel(oPos.x, oPos.y)]);
						ceres::CostFunction* costFunction = RegTerm::Create(deltaUr, weightRegSqrt);
						double2 *varStartA = h_x_double + getPixel(x, y);
						double2 *varStartB = h_x_double + getPixel(oPos.x, oPos.y);
						problem.AddResidualBlock(costFunction, NULL, (double*)varStartA, (double*)varStartB, h_a_double + getPixel(x, y));
					}
				}
			}
		}
	}

	cout << "Solving..." << endl;

	Solver::Options options;
	Solver::Summary summary;

	// options.minimizer_progress_to_stdout = true;// !performanceTest;

	options.minimizer_progress_to_stdout = true;

	//faster methods
	options.num_threads = 8;
	options.num_linear_solver_threads = 8;
	options.linear_solver_type = ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY; //7.2s


	options.max_num_iterations = 1000;
	//options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR; //10.0s

	//slower methods
	//options.linear_solver_type = ceres::LinearSolverType::ITERATIVE_SCHUR; //40.6s
	//options.linear_solver_type = ceres::LinearSolverType::CGNR; //46.9s

	//options.minimizer_type = ceres::LINE_SEARCH;

	//options.min_linear_solver_iterations = linearIterationMin;

	//options.function_tolerance = 0.01;
	//options.gradient_tolerance = 1e-4 * options.function_tolerance;

	//options.min_lm_diagonal = 1.0f;
	//options.min_lm_diagonal = options.max_lm_diagonal;
	//options.max_lm_diagonal = 10000000.0;

	//problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
	//cout << "Cost*2 start: " << cost << endl;

	// TODO: remove
	//options.linear_solver_type = ceres::LinearSolverType::CGNR; 


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
	//options.initial_trust_region_radius = 1e7;
	//options.max_linear_solver_iterations = 20;

	options.eta = 1e-4;

	options.jacobi_scaling = true;
	//options.preconditioner_type = ceres::PreconditionerType::IDENTITY;


	options.max_num_iterations = 100;

	double elapsedTime;
	{
		ml::Timer timer;
		Solve(options, &problem, &summary);
		elapsedTime = timer.getElapsedTimeMS();
	}

	cout << "Solver used: " << summary.linear_solver_type_used << endl;
	cout << "Minimizer iters: " << summary.iterations.size() << endl;
	cout << "Total time: " << elapsedTime << "ms" << endl;

	double iterationTotalTime = 0.0;
	int totalLinearItereations = 0;
	for (auto &i : summary.iterations)
	{
		iterationTotalTime += i.iteration_time_in_seconds;
		totalLinearItereations += i.linear_solver_iterations;
		cout << "Iteration: " << i.linear_solver_iterations << " " << i.iteration_time_in_seconds * 1000.0 << "ms" << endl;
	}

	for (auto &i : summary.iterations) {
		result.push_back(SolverIteration(i.cost, i.iteration_time_in_seconds * 1000.0));
	}

	cout << "Total iteration time: " << iterationTotalTime << endl;
	cout << "Cost per linear solver iteration: " << iterationTotalTime * 1000.0 / totalLinearItereations << "ms" << endl;

	double cost = -1.0;
	problem.Evaluate(Problem::EvaluateOptions(), &cost, nullptr, nullptr, nullptr);
	cout << "Cost*2 end: " << cost * 2 << endl;

	cout << summary.FullReport() << endl;

	for (int i = 0; i < pixelCount; i++)
	{
		h_x_float[i].x = (float)h_x_double[i].x;
		h_x_float[i].y = (float)h_x_double[i].y;
		h_a_float[i] = (float)h_a_double[i];
	}

	return (float)(summary.total_time_in_seconds * 1000.0);
}


*/
#endif // USE_CERES
