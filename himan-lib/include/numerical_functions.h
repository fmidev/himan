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

himan::matrix<double> Filter2D(const himan::matrix<double>& A, const himan::matrix<double>& B);

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

himan::matrix<double> Max2D(const himan::matrix<double>& A, const himan::matrix<double>& B);

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
 * @return Miniimum data
 */

himan::matrix<double> Min2D(const himan::matrix<double>& A, const himan::matrix<double>& B);

/*
 * CUDA version of Filter2D
 */

#ifdef HAVE_CUDA

/* Inline CUDA functions for accessing / setting the input matrix elements */
CUDA_DEVICE size_t CudaMatrixIndex(size_t x, size_t y, size_t z, size_t W, size_t H);
CUDA_DEVICE void CudaMatrixSet(darr_t C, size_t x, size_t y, size_t z, size_t W, size_t H, double v);

/**
 * @brief Structure passed to CUDA Filter2D kernel containing the input matrix
 * sizes.
 *
 * Set these before calling the CUDA kernel.
 */

struct filter_opts
{
	int aDimX;           /**< input matrix width */
	int aDimY;           /**< input matrix height */
	int bDimX;           /**< convolution kernel width */
	int bDimY;           /**< convolution kernel height */
	double missingValue; /**< input matrix missing value (used for detecting
	                        missing values in the CUDA kernel) */
};

/**
 * @brief Filter2D CUDA Kernel
 * @param A Input data
 * @param B Convolution kernel
 * @param C Output matrix of the same size as A
 * @param opts Structure filled with the dimensions of the matrices A, B, C
 */

#ifdef __CUDACC__
CUDA_KERNEL void Filter2DCuda(cdarr_t A, cdarr_t B, darr_t C, filter_opts opts)
{
	const double missing = opts.missingValue;

	const int kCenterX = opts.bDimX / 2;
	const int kCenterY = opts.bDimY / 2;

	const int M = opts.aDimX;
	const int N = opts.aDimY;

	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < M && j < N)
	{
		size_t kernelMissingCount = 0;
		double convolutionValue = 0.0;
		double kernelWeightSum = 0.0;

		for (int n = 0; n < opts.bDimY; n++)
		{
			const int nn = opts.bDimY - 1 - n;

			for (int m = 0; m < opts.bDimX; m++)
			{
				const int mm = opts.bDimX - 1 - m;

				const int ii = i + (m - kCenterX);
				const int jj = j + (n - kCenterY);

				if (ii >= 0 && ii < M && jj >= 0 && jj < N)
				{
					const int aIdx = CudaMatrixIndex(ii, jj, 0, M, N);
					const int bIdx = CudaMatrixIndex(mm, nn, 0, opts.bDimX, opts.bDimY);
					const double aVal = A[aIdx];
					const double bVal = B[bIdx];

					if (aVal == missing || IsMissing(aVal))
					{
						kernelMissingCount++;
						continue;
					}
					convolutionValue += aVal * bVal;
					kernelWeightSum += bVal;
				}
			}
		}
		CudaMatrixSet(C, i, j, 0, M, N, kernelWeightSum == 0 ? MissingDouble() : convolutionValue / kernelWeightSum);
	}
}

/**
 * @brief himan::matrix indexing for identical behaviour with the CPU Filter2D
 * @param W width of the matrix
 * @param H height of the matrix
 */
CUDA_DEVICE CUDA_INLINE size_t CudaMatrixIndex(size_t x, size_t y, size_t z, size_t W, size_t H)
{
	return z * W * H + y * W + x;
}

/**
 * @brief Set C at CudaMatrixIndex(x, y, z, W, H) to v
 * @param C matrix to be modified
 * @param v value to be placed at the index
 */
CUDA_DEVICE CUDA_INLINE void CudaMatrixSet(darr_t C, size_t x, size_t y, size_t z, size_t W, size_t H, double v)
{
	const size_t index = CudaMatrixIndex(x, y, z, W, H);
	C[index] = v;
}

// __CUDACC__
#endif

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

		if (length <= 0 || IsMissing(length))
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
std::pair<std::vector<T>, std::vector<T>> LegGauss(size_t N, bool computeWeights);

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

#endif /* NUMERICAL_FUNCTIONS_H */
