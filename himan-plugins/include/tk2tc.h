/*
 * tk2tc.h
 *
 *  Created on: Nov 17, 2012
 *      Author: partio
 */

#ifndef TK2TC_H
#define TK2TC_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "tk2tc_cuda.h"

namespace himan
{
namespace plugin
{

class tk2tc : public compiled_plugin, private compiled_plugin_base
{
public:
    tk2tc();

    inline virtual ~tk2tc() {}

    tk2tc(const tk2tc& other) = delete;
    tk2tc& operator=(const tk2tc& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::tk2tc";
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
#ifdef HAVE_CUDA
	std::unique_ptr<tk2tc_cuda::options> CudaPrepare(std::shared_ptr<info> sourceInfo);
	void CudaFinish(std::unique_ptr<tk2tc_cuda::options> opts, std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> sourceInfo);
#endif

	double itsBase;
	double itsScale;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<tk2tc> (new tk2tc());
}

} // namespace plugin
} // namespace himan

#endif /* TK2TC */
