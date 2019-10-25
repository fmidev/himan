/**
 * @file   interpolate.h
 *
 */

#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include "info.h"
#include "latitude_longitude_grid.h"
#include "plugin_configuration.h"
#include "reduced_gaussian_grid.h"
#include <boost/variant.hpp>

#ifndef __NVCC__
#include <Eigen/SparseCore>
#include <mutex>
#endif

namespace himan
{
namespace interpolate
{
#ifndef __NVCC__

/**
 * @brief area_interpolation provides grid to grid interpolation
 * through solving linear equations A*x = y
 * where:
 *   source data => x
 *   target data => y
 * and A containing the linear coefficients to interpolate data
 * from source grid to target grid.
 *
 * @example (using Einstein convention)
 * For a single grid point in target grid problem can be expressed as:
 *   Y = W[1]*X[1] + W[2]*X[2] + ... + W[N-1]*X[N-1] + W[N]*X[N]
 * or
 *   Y = W[i]*X[i]
 * where W[i] denotes the Weight applied to the value X[i] at location i
 * on the source grid.
 *
 * applied to all points in target grid
 *   Y[j] = W[i,j]*X[i]
 * with i denotin source locations and j target locations
 */

template <typename T>
class area_interpolation
{
   public:
	area_interpolation() = default;
	area_interpolation(grid& source, grid& target, HPInterpolationMethod method);
	void Interpolate(base<T>& source, base<T>& target);
	size_t SourceSize() const;
	size_t TargetSize() const;

   private:
	Eigen::SparseMatrix<T, Eigen::RowMajor> itsInterpolation;
};

template class area_interpolation<double>;
template class area_interpolation<float>;

/**
 * @brief interpolator is a cache that holds area_interpolation instances
 * once calculated and offers an interface to access the interpolation from
 * grid to grid. Interpolations are cached based on unique hash values computed
 * from unique grid identifiers and interpolation method.
 */

template <typename T>
class interpolator
{
   public:
	static bool Insert(const base<T>& source, const base<T>& target, HPInterpolationMethod method);
	bool Interpolate(base<T>& source, base<T>& target, HPInterpolationMethod method);

   private:
	static std::mutex interpolatorAccessMutex;
	static std::map<size_t, area_interpolation<T>> cache;
};
template class interpolator<double>;
template class interpolator<float>;

#endif

template <typename T>
bool InterpolateArea(const grid* baseGrid, std::shared_ptr<info<T>> source);

template <typename T>
bool Interpolate(const grid* baseGrid, std::vector<std::shared_ptr<info<T>>>& infos);

bool IsVectorComponent(const std::string& paramName);

HPInterpolationMethod InterpolationMethod(const std::string& paramName, HPInterpolationMethod interpolationMethod);

template <typename T>
void RotateVectorComponents(const grid* from, const grid* to, himan::info<T>& U, himan::info<T>& V, bool useCuda);

template <typename T>
void RotateVectorComponentsCPU(const grid* from, const grid* to, himan::matrix<T>& U, himan::matrix<T>& V);

#ifdef HAVE_CUDA
template <typename T>
void RotateVectorComponentsGPU(const grid* from, const grid* to, himan::matrix<T>& U, himan::matrix<T>& V,
                               cudaStream_t& stream, T* d_u, T* d_v);
#endif

bool IsSupportedGridForRotation(HPGridType type);

/**
 * @brief Provide the non-zero weights and corresponding indices for a
 * single target point from a source grid using bilinear interpolation
 *
 * If point lies outside grid weight will be Missing Value
 */

template <typename T>
std::pair<std::vector<size_t>, std::vector<T>> InterpolationWeights(reduced_gaussian_grid& source, point target);

template <typename T>
std::pair<std::vector<size_t>, std::vector<T>> InterpolationWeights(regular_grid& source, point target);

/**
 * @brief Provide the nearest point index for a single target point with weight 1.0
 *
 * If point lies outside grid weight will be MissingValue
 */

template <typename T>
std::pair<size_t, T> NearestPoint(reduced_gaussian_grid& source, point target);

template <typename T>
std::pair<size_t, T> NearestPoint(regular_grid& source, point target);
}
}

#endif /* INTERPOLATE_H */
