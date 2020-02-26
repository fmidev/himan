#ifndef NUMERICAL_FUNCTIONS_H
#define NUMERICAL_FUNCTIONS_H

#include "cuda_helper.h"
#include "himan_common.h"
#include "plugin_configuration.h"

namespace himan
{
namespace numerical_functions
{
/**
 * @brief Compute convolution of matrix A by matrix B
 * @param A Data
 * @param B Convolution kernel
 * @return Data convolved by kernel
 */

template <typename T>
himan::matrix<T> Filter2D(const himan::matrix<T>& A, const himan::matrix<T>& B, bool useCuda = false);

/**
 * @brief Compute the maximum value in matrix A from the area specified by
 * matrix B
 *
 * Matrix B acts also as a weight; a default boxed maximum search would have all matrix B
 * elements set to 1, but if for example origin needs to be excluded, that value
 * can be set to Missing.
 *
 * @param A Data
 * @param B Kernel
 * @return Maximum data
 */

template <typename T>
himan::matrix<T> Max2D(const himan::matrix<T>& A, const himan::matrix<T>& B, bool useCuda = false);

/**
 * @brief Compute the minimum value in matrix A from the area specified by
 * matrix B
 *
 * Matrix B acts also as a weight; a default boxed minimum search would have all matrix B
 * elements set to 1, but if for example origin needs to be excluded, that value
 * can be set to Missing.
 *
 * @param A Data
 * @param B Kernel
 * @return Minimum data
 */

template <typename T>
himan::matrix<T> Min2D(const himan::matrix<T>& A, const himan::matrix<T>& B, bool useCuda = false);

/**
 * @brief A generalized filter from that filters like Min,Max,Mean etc. can be derived
 * for  matrix A from the area specified by matrix B
 *
 * Matrix B acts also as a weight; a default boxed convolution would have all matrix B
 * elements set to 1, but if for example origin needs to be excluded, that value
 * can be set to Missing.
 *
 * @param A Data
 * @param B Kernel
 * @param f Filter function void f(double& sum_of_values, double& sum_of_weights, double value, double weight)
 * @param g Weight normalization function double g(double& sum_of_values, double& sum_of_weights)
 * @param init1 Initial value of generalized value summation
 * @param init2 Inital value of weight summation
 */

template <typename T, class F>
himan::matrix<size_t> FindIndex2D(const himan::matrix<T>& A, const himan::matrix<T>& B, F&& f, T init1);

/**
 * @brief Compute the index of the maximum value in matrix A from the area specified by
 * matrix B
 *
 * Matrix B acts as a mask; a default boxed maximum search would have all matrix B
 * elements set to 1, but if for example origin needs to be excluded, that value
 * can be set to Missing.
 *
 * @param A Data
 * @param B Kernel
 * @return Index of maximum data
 */

template <typename T>
himan::matrix<size_t> IndexMax2D(const himan::matrix<T>& A, const himan::matrix<T>& B);

/**
 * @brief Compute the probability of some condition in matrix A from the area specified by
 * matrix B
 *
 * Lambda function f is used to decide if given value fills the condition.
 *
 * @param A Data
 * @param B Kernel
 * @return Probability (0 to 1)
 */

template <typename T, class F>
himan::matrix<T> Prob2D(const himan::matrix<T>& A, const himan::matrix<T>& B, F&& f);

/**
 * @brief A generalized filter from that filters like Min,Max,Mean etc. can be derived
 * for  matrix A from the area specified by matrix B
 *
 * Matrix B acts also as a weight; a default boxed convolution would have all matrix B
 * elements set to 1, but if for example origin needs to be excluded, that value
 * can be set to Missing.
 *
 * @param A Data
 * @param B Kernel
 * @param f Filter function void f(double& sum_of_values, double& sum_of_weights, double value, double weight)
 * @param g Weight normalization function double g(double& sum_of_values, double& sum_of_weights)
 * @param init1 Initial value of generalized value summation
 * @param init2 Inital value of weight summation
 */

template <typename T, class F, class G>
himan::matrix<T> Reduce2D(const himan::matrix<T>& A, const himan::matrix<T>& B, F&& f, G&& g, T init1, T init2);

#ifdef HAVE_CUDA
/**
 * @brief Structure passed to CUDA Filter2D kernel containing the input matrix
 * sizes.
 *
 * Set these before calling the CUDA kernel.
 */

struct filter_opts
{
	int aDimX; /**< input matrix width */
	int aDimY; /**< input matrix height */
	int bDimX; /**< convolution kernel width */
	int bDimY; /**< convolution kernel height */
};

template <typename T>
himan::matrix<T> Filter2DGPU(const matrix<T>& A, const matrix<T>& B);

template <typename T>
himan::matrix<T> Max2DGPU(const matrix<T>& A, const matrix<T>& B);

template <typename T>
himan::matrix<T> Min2DGPU(const matrix<T>& A, const matrix<T>& B);

template <typename T, class F>
himan::matrix<T> Prob2DGPU(const himan::matrix<T>& A, const himan::matrix<T>& B, F&& f);

template <typename T>
himan::matrix<T> ProbLimitGt2DGPU(const matrix<T>& A, const matrix<T>& B, T limit);

template <typename T>
himan::matrix<T> ProbLimitGe2DGPU(const matrix<T>& A, const matrix<T>& B, T limit);

template <typename T>
himan::matrix<T> ProbLimitLt2DGPU(const matrix<T>& A, const matrix<T>& B, T limit);

template <typename T>
himan::matrix<T> ProbLimitLe2DGPU(const matrix<T>& A, const matrix<T>& B, T limit);

template <typename T>
himan::matrix<T> ProbLimitEq2DGPU(const matrix<T>& A, const matrix<T>& B, T limit);

template <typename T, class F, class G>
himan::matrix<T> Reduce2DGPU(const himan::matrix<T>& A, const himan::matrix<T>& B, F&& f, G&& g, T init1, T init2);

// HAVE_CUDA
#endif

/**
 * Arange
 *
 * Mimics numpy's arange function:
 *
 * https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.arange.html
 */

template <typename T>
std::vector<std::vector<T>> Arange(const std::vector<T>& start, const std::vector<T>& stop, T step)
{
	std::vector<std::vector<T>> ret(start.size());

	for (size_t i = 0; i < start.size(); i++)
	{
		const T length = (stop[i] - start[i]) / step;

		if (length <= 0 || IsMissing(length) || length >= (std::numeric_limits<T>::max() - T(1)))
		{
			continue;
		}

		ret[i].resize(static_cast<size_t>(ceil(length)));
		ret[i][0] = start[i];
		std::generate(ret[i].begin() + 1, ret[i].end(), [ v = ret[i][0], &step ]() mutable { return v += step; });
	}

	return ret;
}

/**
 * Linspace
 *
 * Mimics numpy's linspace function:
 *
 * https://docs.scipy.org/doc/numpy-1.12.0/reference/generated/numpy.linspace.html
 */

template <typename T>
std::vector<std::vector<T>> Linspace(const std::vector<T>& start, const std::vector<T>& stop, unsigned int length,
                                     bool endpoint = true)
{
	std::vector<std::vector<T>> ret(start.size());

	for (size_t i = 0; i < start.size(); i++)
	{
		const T step = (stop[i] - start[i]) / (length - static_cast<int>(endpoint));

		if (IsMissing(step))
		{
			continue;
		}

		ret[i].resize(static_cast<size_t>(length));
		ret[i][0] = start[i];
		std::generate(ret[i].begin() + 1, ret[i].end(), [ v = ret[i][0], &step ]() mutable { return v += step; });
	}

	return ret;
}

/**
 * LegGauss
 *
 * Gauss-Legendre quadrature. Computes the sample points and weights for Gauss-Legendre quadrature. These sample points
 * and weights will correctly integrate polynomials of degree 2*deg - 1 or less over the interval [-1, 1] with the
 * weight function f( x ) = 1. The computeWeights flag can be set to false if only quadrature points are needed.
 * Implementation is based on numpy's leggauss function following the Golub and Welsh algorithm.
 *
 * https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.polynomial.legendre.leggauss.html
 * https://web.stanford.edu/class/cme335/spr11/S0025-5718-69-99647-1.pdf
 */

template <typename T>
std::pair<std::vector<T>, std::vector<T>> LegGauss(size_t N, bool computeWeights = true);

/**
 * Return unweighted mean value of vector
 */

template <typename T>
T Mean(const std::vector<T>& data);

/**
 * Return variance of vector
 *
 * https://en.wikipedia.org/wiki/Variance
 */

template <typename T>
T Variance(const std::vector<T>& data);

namespace interpolation
{
/*
 * Basic interpolation functions.
 */

template <typename Type>
CUDA_HOST CUDA_DEVICE inline Type Linear(Type factor, Type Y1, Type Y2)
{
	return std::fma(factor, Y2, std::fma(-factor, Y1, Y1));
}

template <typename Type>
CUDA_HOST CUDA_DEVICE inline Type Linear(Type X, Type X1, Type X2, Type Y1, Type Y2)
{
	if (X1 == X2)
	{
		return Y1;
	}

	const Type factor = (X - X1) / (X2 - X1);
	return Linear<Type>(factor, Y1, Y2);
}

template <typename Type>
CUDA_HOST CUDA_DEVICE inline Type BiLinear(Type dx, Type dy, Type a, Type b, Type c, Type d)
{
	// Method below is faster but gives visible interpolation artifacts

	// double ab = Linear(dx, a, b);
	// double cd = Linear(dx, c, d);
	// return Linear(dy, ab, cd);

	// This one gives smooth interpolation surfaces
	return (1 - dx) * (1 - dy) * c + dx * (1 - dy) * d + (1 - dx) * dy * a + dx * dy * b;
}

}  // namespace interpolation

}  // namespace numerical_functions

}  // namespace himan

#include "numerical_functions_impl.h"

#endif /* NUMERICAL_FUNCTIONS_H */
