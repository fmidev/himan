/**
 * @file   interpolate.h
 *
 */

#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include "latitude_longitude_grid.h"
#include "plugin_configuration.h"
#include "reduced_gaussian_grid.h"

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

class area_interpolation
{
   public:
        area_interpolation() = default;
        area_interpolation(grid& source, grid& target, HPInterpolationMethod method);
        void Interpolate(base& source, base& target);
        size_t SourceSize() const;
        size_t TargetSize() const;

   private:
        Eigen::SparseMatrix<double, Eigen::RowMajor> itsInterpolation;
};

/**
 * @brief interpolator is a cache that holds area_interpolation instances
 * once calculated and offers an interface to access the interpolation from
 * grid to grid. Interpolations are cached based on unique hash values computed
 * from unique grid identifiers and interpolation method.
 */

class interpolator
{
   public:
        static bool Insert(const base& source, const base& target, HPInterpolationMethod method);
        bool Interpolate(base& source, base& target, HPInterpolationMethod method);

   private:
        static std::mutex interpolatorAccessMutex;
        static std::map<size_t, interpolate::area_interpolation> cache;
};
#endif

bool InterpolateArea(info& base, info_t source);

bool Interpolate(info& base, std::vector<info_t>& infos, bool useCudaForInterpolation = true);

bool IsVectorComponent(const std::string& paramName);

HPInterpolationMethod InterpolationMethod(const std::string& paramName, HPInterpolationMethod interpolationMethod);

void RotateVectorComponents(info& UInfo, info& VInfo, bool useCuda);

void RotateVectorComponentsCPU(info& UInfo, info& VInfo);

#ifdef HAVE_CUDA
void RotateVectorComponentsGPU(info& UInfo, info& VInfo, cudaStream_t& stream, double* d_u, double* d_v);
#endif

bool IsSupportedGridForRotation(HPGridType type);

/**
 * @brief Provide the non-zero weights and corresponding indices for a 
 * single target point from a source grid using bilinear interpolation
 *
 * If point lies outside grid weight will be Missing Value 
 */

std::pair<std::vector<size_t>,std::vector<double>> InterpolationWeights(reduced_gaussian_grid& source, point target);
std::pair<std::vector<size_t>,std::vector<double>> InterpolationWeights(regular_grid& source, point target);

/**
 * @brief Provide the nearest point index for a single target point with weight 1.0
 * 
 * If point lies outside grid weight will be MissingValue
 */

std::pair<size_t,double> NearestPoint(reduced_gaussian_grid& source, point target);
std::pair<size_t,double> NearestPoint(regular_grid& source, point target);

}
}

#endif /* INTERPOLATE_H */
