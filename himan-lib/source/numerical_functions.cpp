/**
 * @file numerical_functions.cpp
 */

#include "numerical_functions.h"
#include <Eigen/Dense>
#include <algorithm>
#include <limits>
#include <numeric>

using namespace himan;
using namespace numerical_functions;
using namespace Eigen;

template <typename T>
matrix<T> numerical_functions::Filter2D(const matrix<T>& A, const matrix<T>& B, bool useCuda)
{
#ifdef HAVE_CUDA
	if (useCuda)
	{
		return Filter2DGPU(A, B);
	}
#endif
	return Reduce2D<T>(
	    A, B,
	    [](T& val1, T& val2, const T& a, const T& b)
	    {
		    if (IsValid(a * b))
		    {
			    val1 += a * b;
			    val2 += b;
		    }
	    },
	    [](const T& val1, const T& val2) { return val2 == T(0) ? MissingValue<T>() : val1 / val2; }, T(0), T(0));
}

template matrix<double> numerical_functions::Filter2D(const matrix<double>&, const matrix<double>& B, bool);
template matrix<float> numerical_functions::Filter2D(const matrix<float>&, const matrix<float>& B, bool);

template <typename T>
matrix<T> numerical_functions::Max2D(const matrix<T>& A, const matrix<T>& B, bool useCuda)
{
#ifdef HAVE_CUDA
	if (useCuda)
	{
		return Max2DGPU(A, B);
	}
#endif
	return Reduce2D<T>(
	    A, B,
	    [](T& val1, T& val2, const T& a, const T& b)
	    {
		    if (IsValid(a * b))
			    val1 = !(a * b <= val1) ? a : val1;
	    },
	    [](const T& val1, const T& val2) { return val1; }, MissingValue<T>(), T(0));
}

template matrix<double> numerical_functions::Max2D(const matrix<double>&, const matrix<double>& B, bool);
template matrix<float> numerical_functions::Max2D(const matrix<float>&, const matrix<float>& B, bool);

template <typename T>
matrix<T> numerical_functions::Min2D(const matrix<T>& A, const matrix<T>& B, bool useCuda)
{
#ifdef HAVE_CUDA
	if (useCuda)
	{
		return Min2DGPU(A, B);
	}
#endif
	return Reduce2D<T>(
	    A, B,
	    [](T& val1, T& val2, const T& a, const T& b)
	    {
		    if (IsValid(a * b))
			    val1 = !(a * b >= val1) ? a : val1;
	    },
	    [](const T& val1, const T& val2) { return val1; }, MissingValue<T>(), T(0));
}

template matrix<double> numerical_functions::Min2D(const matrix<double>&, const matrix<double>& B, bool);
template matrix<float> numerical_functions::Min2D(const matrix<float>&, const matrix<float>& B, bool);

template <typename T>
matrix<size_t> numerical_functions::IndexMax2D(const matrix<T>& A, const matrix<T>& B)
{
	return FindIndex2D(
	    A, B, [](T& current_max, const T& a, const T& b) { return (a > current_max) & IsValid(b); },
	    std::numeric_limits<T>::lowest());
}

template matrix<size_t> numerical_functions::IndexMax2D(const matrix<float>& A, const matrix<float>& B);
template matrix<size_t> numerical_functions::IndexMax2D(const matrix<double>& A, const matrix<double>& B);

template <typename T>
std::pair<std::vector<T>, std::vector<T>> numerical_functions::LegGauss(size_t N, bool computeWeights)
{
	// Set up Eigenvalue problem
	//-------------------------------------------------------------------------------------------------------
	Matrix<T, Dynamic, Dynamic> J(N, N);

	Diagonal<Matrix<T, Dynamic, Dynamic>, 0> Jdiag0(J);
	Diagonal<Matrix<T, Dynamic, Dynamic>, 1> Jdiag1(J);

	for (size_t n = 0; n < N; ++n)
	{
		Jdiag0[n] = 0.0;
	}

	for (size_t n = 0; n < N - 1; ++n)
	{
		Jdiag1[n] = static_cast<T>(T(n + 1) * 1.0 / std::sqrt(2 * (n) + 1) * 1.0 / std::sqrt(2 * (n + 1) + 1));
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
T numerical_functions::Median(const std::vector<T>& data)
{
	// Need to mutate (sort) the vector
	auto vec = data;
	size_t n = vec.size();
	size_t mid = n / 2;

	// Use nth_element to partially sort
	std::nth_element(vec.begin(), vec.begin() + mid, vec.end());
	T median = vec[mid];

	if (n % 2 == 0)
	{
		// For even number of elements, need the lower middle too
		std::nth_element(vec.begin(), vec.begin() + mid - 1, vec.end());
		median = (median + vec[mid - 1]) / static_cast<T>(2.0);
	}

	return median;
}

template double numerical_functions::Median(const std::vector<double>&);
template float numerical_functions::Median(const std::vector<float>&);

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

template <typename T>
T numerical_functions::RampUp(T lowerLimit, T upperLimit, T value)
{
	auto v = (value - lowerLimit) / (upperLimit - lowerLimit);
	return std::min(std::max(v, static_cast<T>(0.0)), static_cast<T>(1.0));
}

template double numerical_functions::RampUp(double, double, double);
template float numerical_functions::RampUp(float, float, float);

template <typename T>
std::vector<T> numerical_functions::RampUp(T lowerLimit, T upperLimit, const std::vector<T>& values)
{
	std::vector<T> ret;
	ret.reserve(values.size());

	for (size_t i = 0; i < values.size(); i++)
	{
		ret.push_back(RampUp<T>(lowerLimit, upperLimit, values[i]));
	}

	return ret;
}

template std::vector<double> numerical_functions::RampUp(double, double, const std::vector<double>&);
template std::vector<float> numerical_functions::RampUp(float, float, const std::vector<float>&);

template <typename T>
std::vector<T> numerical_functions::RampUp(const std::vector<T>& lowerLimits, const std::vector<T>& upperLimits,
                                           const std::vector<T>& values)
{
	std::vector<T> ret;
	ret.reserve(values.size());

	for (size_t i = 0; i < values.size(); i++)
	{
		ret.push_back(RampUp<T>(lowerLimits[i], upperLimits[i], values[i]));
	}

	return ret;
}

template std::vector<double> numerical_functions::RampUp(const std::vector<double>&, const std::vector<double>&,
                                                         const std::vector<double>&);
template std::vector<float> numerical_functions::RampUp(const std::vector<float>&, const std::vector<float>&,
                                                        const std::vector<float>&);

template <typename T>
T numerical_functions::RampDown(T lowerLimit, T upperLimit, T value)
{
	auto v = (upperLimit - value) / (upperLimit - lowerLimit);
	return std::min(std::max(v, static_cast<T>(0.0)), static_cast<T>(1.0));
}

template double numerical_functions::RampDown(double, double, double);
template float numerical_functions::RampDown(float, float, float);

template <typename T>
std::vector<T> numerical_functions::RampDown(T lowerLimit, T upperLimit, const std::vector<T>& values)
{
	std::vector<T> ret;
	ret.reserve(values.size());

	for (size_t i = 0; i < values.size(); i++)
	{
		ret.push_back(RampDown<T>(lowerLimit, upperLimit, values[i]));
	}

	return ret;
}

template std::vector<double> numerical_functions::RampDown(double, double, const std::vector<double>&);
template std::vector<float> numerical_functions::RampDown(float, float, const std::vector<float>&);

template <typename T>
std::vector<T> numerical_functions::RampDown(const std::vector<T>& lowerLimits, const std::vector<T>& upperLimits,
                                             const std::vector<T>& values)
{
	std::vector<T> ret;
	ret.reserve(values.size());

	for (size_t i = 0; i < values.size(); i++)
	{
		ret.push_back(RampDown<T>(lowerLimits[i], upperLimits[i], values[i]));
	}

	return ret;
}

template std::vector<double> numerical_functions::RampDown(const std::vector<double>&, const std::vector<double>&,
                                                           const std::vector<double>&);
template std::vector<float> numerical_functions::RampDown(const std::vector<float>&, const std::vector<float>&,
                                                          const std::vector<float>&);
