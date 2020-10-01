/**
 * @file   windvector_cuda.h
 */

#ifndef WINDVECTOR_CUDA_H
#define WINDVECTOR_CUDA_H

namespace himan
{
namespace plugin
{
enum HPWindVectorTargetType
{
	kUnknownElement = 0,
	kWind,
	kGust,
	kSea,
	kIce
};

#ifdef HAVE_CUDA

namespace windvector_cuda
{
void RunCuda(const std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<info<float>> myTargetInfo,
             const param& UParam, const param& VParam, HPWindVectorTargetType itsTargetType);

double* CacheLongitudeCoordinates(const grid* g, cudaStream_t& stream);
void FreeLongitudeCache();

}  // namespace windvector_cuda

#endif /* HAVE_CUDA */

}  // namespace plugin
}  // namespace himan

#endif /* WINDVECTOR_CUDA_H */
