/**
 * @file numerical_functions.h
 *
 */

#ifndef NUMERICAL_FUNCTIONS_H
#define NUMERICAL_FUNCTIONS_H

#include "cuda_helper.h"
#include "himan_common.h"
#include "plugin_configuration.h"
#include <valarray>

namespace himan
{
namespace numerical_functions
{
/**
 * @class Integral
 * An object of type integral that can perform vertical integration in the atmosphere from a lower bound to an upper
 * bound height. It can either be used to integrate a single variable (i.e. /int{x(h) dh} or a function /int{f(x(h)) dh}
 */
#if 0
class integral
{
   public:
	integral();
	~integral() {}
	// virtual std::string ClassName() const { return "himan::numerical_functions::integral"; }

	// provide list of parameters that will be integrated over
	void Params(params theParams);

	// provide function of parameters that will be integrated over. TODO I'll document an example how that lambda
	// function has to look like.
	void Function(std::function<std::valarray<double>(const std::vector<std::valarray<double>>&)> theFunction);

	// set bounds
	void LowerBound(const std::valarray<double>& theLowerBound);
	void UpperBound(const std::valarray<double>& theUpperBound);
	void LowerLevelLimit(int theLowestLevel);
	void UpperLevelLimit(int theHighestLevel);
	void SetLevelLimits();
	void ForecastType(forecast_type theType);
	void ForecastTime(forecast_time theTime);
	void LevelType(level theLevel);
	void HeightInMeters(bool theHeightInMeters);
	// return result
	const std::valarray<double>& Result() const;

	// pass configuration to integration object (needed for fetching values)
	std::shared_ptr<const plugin_configuration> itsConfiguration;

	// evaluate the integral expression
	void Evaluate();
	bool Complete();

   private:
	bool itsHeightInMeters;
	int itsLowestLevel;
	int itsHighestLevel;

	forecast_time itsTime;
	forecast_type itsType;
	level itsLevel;

	params itsParams;
	std::function<std::valarray<double>(const std::vector<std::valarray<double>>&)> itsFunction;

	std::valarray<double> itsLowerBound;
	std::valarray<double> itsUpperBound;

	std::vector<bool> itsComplete;

	std::valarray<double> itsResult;  // variable is modified in some Result() const functions

	std::valarray<double> Interpolate(std::valarray<double>, std::valarray<double>, std::valarray<double>,
	                                  std::valarray<double>, std::valarray<double>) const
	    __attribute__((always_inline));
	std::pair<level, level> LevelForHeight(const producer&, double) const;
};

inline std::valarray<double> integral::Interpolate(std::valarray<double> currentLevelValue,
                                                   std::valarray<double> previousLevelValue,
                                                   std::valarray<double> currentLevelHeight,
                                                   std::valarray<double> previousLevelHeight,
                                                   std::valarray<double> itsBound) const
{
	return (previousLevelValue +
	        (currentLevelValue - previousLevelValue) * (itsBound - previousLevelHeight) /
	            (currentLevelHeight - previousLevelHeight));
}
#endif

/**
 * @brief Compute convolution of matrix A by matrix B
 * @param A Data
 * @param B Convolution kernel
 * @return Data convolved by kernel
 */

himan::matrix<double> Filter2D(const himan::matrix<double>& A, const himan::matrix<double>& B);

/**
 * @brief Compute the maximum value in matrix A from the area specified by matrix B
 *
 * Matrix B acts also as a weight; a default boxed maximum search would have all matrix B
 * elements set to 1, but if for example origin needs to be excluded, that value
 * can be set to 0.
 *
 * @param A Data
 * @param B Kernel
 * @return Maximum data
 */

himan::matrix<double> Max2D(const himan::matrix<double>& A, const himan::matrix<double>& B);

/**
 * @brief Compute the minimum value in matrix A from the area specified by matrix B
 *
 * Matrix B acts also as a weight; a default boxed minimum search would have all matrix B
 * elements set to 1, but if for example origin needs to be excluded, that value
 * can be set to 0.
 *
 * @param A Data
 * @param B Kernel
 * @return Maximum data
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
 * @brief Structure passed to CUDA Filter2D kernel containing the input matrix sizes.
 *
 * Set these before calling the CUDA kernel.
 */

struct filter_opts
{
	int aDimX;           /**< input matrix width */
	int aDimY;           /**< input matrix height */
	int bDimX;           /**< convolution kernel width */
	int bDimY;           /**< convolution kernel height */
	double missingValue; /**< input matrix missing value (used for detecting missing values in the CUDA kernel) */
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

	size_t kernelMissingCount = 0;

	if (i < M && j < N)
	{
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

					if (aVal == missing)
					{
						kernelMissingCount++;
						continue;
					}
					convolutionValue += aVal * bVal;
					kernelWeightSum += bVal;
				}
			}
		}
		if (kernelMissingCount < 3)
		{
			CudaMatrixSet(C, i, j, 0, M, N, convolutionValue / kernelWeightSum);
		}
		else
		{
			CudaMatrixSet(C, i, j, 0, M, N, MissingDouble());
		}
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

namespace interpolation
{
/*
 * Basic interpolation functions.
 */

CUDA_HOST CUDA_DEVICE inline double Linear(double factor, double Y1, double Y2)
{
	return fma(factor, Y2, fma(-factor, Y1, Y1));
}

CUDA_HOST CUDA_DEVICE inline double Linear(double X, double X1, double X2, double Y1, double Y2)
{
	double factor = (X - X1) / (X2 - X1);
	return Linear(factor, Y1, Y2);
}

CUDA_HOST CUDA_DEVICE inline double BiLinear(double dx, double dy, double a, double b, double c, double d)
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
