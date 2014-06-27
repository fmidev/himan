/**
 * @file windvector.h
 *
 * @date Jan 21, 2013
 * @author aalto
 */

#ifndef WINDVECTOR_H
#define WINDVECTOR_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

//#define WINDVECTOR_HEADER_INCLUDE

#include "windvector_cuda.h" // need to have this here because of HPTargetType

//#undef WINDVECTOR_HEADER_INCLUDE

class NFmiArea;

namespace himan
{
namespace plugin
{

class windvector : public compiled_plugin, private compiled_plugin_base
{
public:
    windvector();

    inline virtual ~windvector() {}

    windvector(const windvector& other) = delete;
    windvector& operator=(const windvector& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::windvector";
    }

    virtual HPPluginClass PluginClass() const
    {
        return kCompiled;
    }

    virtual HPVersionNumber Version() const
    {
        return HPVersionNumber(1, 0);
    }
   
private:
    virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
	std::unique_ptr<NFmiArea> ToNewbaseArea(std::shared_ptr<info> myTargetInfo) const;
#ifdef HAVE_CUDA
	void CudaFinish(std::unique_ptr<windvector_cuda::options> opts, std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> TInfo, std::shared_ptr<info> RHInfo);
	std::unique_ptr<windvector_cuda::options> CudaPrepare(std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> UInfo, std::shared_ptr<info> VInfo);
#endif
	
	HPWindVectorTargetType itsCalculationTarget;
	bool itsVectorCalculation;


};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<windvector> (new windvector());
}

} // namespace plugin
} // namespace himan

#endif /* WINDVECTOR */
