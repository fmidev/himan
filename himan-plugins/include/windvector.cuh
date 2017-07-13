/**
 * @file   windvector_cuda.h
 */

#ifndef WINDVECTOR_CUDA_H
#define WINDVECTOR_CUDA_H

#ifdef HAVE_CUDA
#include "info_simple.h"
#endif

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
struct options
{
	info_simple* u;
	info_simple* v;
	info_simple* speed;
	info_simple* dir;

	HPWindVectorTargetType target_type;
	size_t missing;
	size_t N;

	options() : u(0), v(0), speed(0), dir(0), target_type(kUnknownElement), missing(0), N(0) {}
};

void Process(options& opts);

}  // namespace windvector_cuda

#endif /* HAVE_CUDA */

}  // namespace plugin
}  // namespace himan

#endif /* WINDVECTOR_CUDA_H */
