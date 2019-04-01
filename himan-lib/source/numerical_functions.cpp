/**
 * @file numerical_functions.cpp
 */

#include "numerical_functions.h"
#include <Eigen/Dense>
#include <algorithm>

using namespace himan;
using namespace numerical_functions;
using namespace Eigen;

matrix<double> numerical_functions::Filter2D(const matrix<double>& A, const matrix<double>& B)
{
	return Reduce2D(A, B,
	                [](double& val1, double& val2, const double& a, const double& b) {
		                if (IsValid(a * b))
		                {
			                val1 += a * b;
			                val2 += b;
		                }
	                },
	                [](const double& val1, const double& val2) { return val2 == 0.0 ? MissingDouble() : val1 / val2; },
	                0.0, 0.0);
}

matrix<double> numerical_functions::Max2D(const matrix<double>& A, const matrix<double>& B)
{
	return Reduce2D(A, B,
	                [](double& val1, double& val2, const double& a, const double& b) {
		                if (IsValid(a * b))
			                val1 = !(a * b <= val1) ? a : val1;
	                },
	                [](const double& val1, const double& val2) { return val1; }, MissingDouble(), 0.0);
}

matrix<double> numerical_functions::Min2D(const matrix<double>& A, const matrix<double>& B)
{
	return Reduce2D(A, B,
	                [](double& val1, double& val2, const double& a, const double& b) {
		                if (IsValid(a * b))
			                val1 = !(a * b >= val1) ? a : val1;
	                },
	                [](const double& val1, const double& val2) { return val1; }, MissingDouble(), 0.0);
}

template <typename T>
std::pair<std::vector<T>, std::vector<T>> numerical_functions::LegGauss(size_t N, bool computeWeights)
{
	// Set up Eigenvalue problem
	//-------------------------------------------------------------------------------------------------------
	Matrix<T, Dynamic, Dynamic> J(N, N);

	Diagonal<Matrix<T, Dynamic, Dynamic>, 0> Jdiag0(J);
	Diagonal<Matrix<T, Dynamic, Dynamic>, 1> Jdiag1(J);

	for (int n = 0; n < N; ++n)
	{
		Jdiag0[n] = 0.0;
	}

	for (int n = 0; n < N - 1; ++n)
	{
		Jdiag1[n] = static_cast<T>((n + 1) * 1.0 / std::sqrt(2 * (n) + 1) * 1.0 / std::sqrt(2 * (n + 1) + 1));
	}
	//-------------------------------------------------------------------------------------------------------

	// Solve Eigenvalue problem
	//-------------------------------------------------------------------------------------------------------
	SelfAdjointEigenSolver<Matrix<T, Dynamic, Dynamic>> es(N);
	//-------------------------------------------------------------------------------------------------------

	es.computeFromTridiagonal(Jdiag0, Jdiag1, computeWeights ? ComputeEigenvectors : EigenvaluesOnly);

	// Extract Quadrature points and weights from eigenvalues and eigenvectors
	//-------------------------------------------------------------------------------------------------------
	std::vector<T> r(N);
	std::vector<T> w;

	Map<Matrix<T, Dynamic, Dynamic>> R(r.data(), N, 1);
	R = es.eigenvalues().real();

	if (computeWeights)
	{
		w.resize(N);
		Map<Array<T, Dynamic, Dynamic>> W(w.data(), 1, N);

		W = es.eigenvectors().real().row(0);
		W = W * W * 2;
	}
	//-------------------------------------------------------------------------------------------------------

	return std::make_pair(r, w);
}
template std::pair<std::vector<float>, std::vector<float>> numerical_functions::LegGauss(size_t, bool);
template std::pair<std::vector<double>, std::vector<double>> numerical_functions::LegGauss(size_t, bool);

template <typename T>
T numerical_functions::Mean(const std::vector<T>& data)
{
	if (data.size() == 0)
	{
		return himan::MissingValue<T>();
	}

	return std::accumulate(data.begin(), data.end(), 0.0f) / static_cast<T>(data.size());
}

template double numerical_functions::Mean(const std::vector<double>&);
template float numerical_functions::Mean(const std::vector<float>&);

template <typename T>
T numerical_functions::Variance(const std::vector<T>& data)
{
	if (data.size() == 0)
	{
		return himan::MissingValue<T>();
	}

	const auto mean = Mean(data);

	T sum = 0.0f;

	for (const auto& x : data)
	{
		const auto t = x - mean;
		sum += t * t;
	}

	return sum / static_cast<T>(data.size());
}

template double numerical_functions::Variance(const std::vector<double>&);
template float numerical_functions::Variance(const std::vector<float>&);
