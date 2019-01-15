/**
 * @file   cuda_plugin_helper.h
 *
 */

#ifndef CUDA_PLUGIN_HELPER_H
#define CUDA_PLUGIN_HELPER_H

#ifdef HAVE_CUDA

#include "cuda_helper.h"
#include "info.h"
#include "packed_data.h"
#include "plugin_configuration.h"

namespace himan
{
#ifdef __NVCC__
template <typename T>
__global__ void Fill(T* devptr, size_t N, T fillValue)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N)
	{
		devptr[idx] = fillValue;
	}
}

namespace cuda
{
template <typename T>
bool Unpack(std::shared_ptr<himan::info<T>> fullInfo, cudaStream_t& stream, T* d_arr);
bool Unpack(std::shared_ptr<himan::info<double>> fullInfo, cudaStream_t& stream, double* d_arr);

template <typename T>
void PrepareInfo(std::shared_ptr<himan::info<T>> info, T* d_ret, cudaStream_t& stream, bool copyToHost = true);

template <typename T>
void ReleaseInfo(std::shared_ptr<himan::info<T>> info, T* d_ret, cudaStream_t& stream);

template <typename T>
std::shared_ptr<himan::info<T>> Fetch(const std::shared_ptr<const plugin_configuration> conf,
                                      const himan::forecast_time& theTime, const himan::level& theLevel,
                                      const himan::param& theParam, const himan::forecast_type& theType,
                                      bool returnPacked = true);

template <typename T>
std::shared_ptr<himan::info<T>> Fetch(const std::shared_ptr<const plugin_configuration> conf,
                                      const himan::forecast_time& theTime, const himan::level& theLevel,
                                      const himan::params& theParams, const himan::forecast_type& theType,
                                      bool returnPacked = true);

}  // namespace cuda

#endif /* __NVCC__ */

}  // namespace himan

#endif /* HAVE_CUDA */

#endif /* CUDA_PLUGIN_HELPER_H */
