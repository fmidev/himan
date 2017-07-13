/**
 * @file dewpoint.h
 *
 */

#ifndef DEWPOINT_H
#define DEWPOINT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "dewpoint.cuh"

namespace himan
{
namespace plugin
{
/**
 * @class dewpoint
 *
 * @brief Calculate dewpoint from T and RH
 *
 * Source: journals.ametsoc.org/doi/pdf/10.1175/BAMS-86-2-225
 */

class dewpoint : public compiled_plugin, private compiled_plugin_base
{
   public:
	dewpoint();

	inline virtual ~dewpoint() {}
	dewpoint(const dewpoint& other) = delete;
	dewpoint& operator=(const dewpoint& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::dewpoint"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 0); }
   private:
	virtual void Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex);
#ifdef HAVE_CUDA
	std::unique_ptr<dewpoint_cuda::options> CudaPrepare(std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> TInfo,
	                                                    std::shared_ptr<info> RHInfo);
#endif
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<dewpoint>(new dewpoint()); }
}  // namespace plugin
}  // namespace himan

#endif /* DEWPOINT_H */
