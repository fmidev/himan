/*
 * transformer.h
 *
 *  Created on: May 07, 2014
 *      Author: Tack
 */

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <string>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
// #include "transformer_cuda.h"
#include "level.h"


namespace himan
{
namespace plugin
{

class transformer : public compiled_plugin, private compiled_plugin_base
{
public:
    transformer();

    inline virtual ~transformer() {}

    transformer(const transformer& other) = delete;
    transformer& operator=(const transformer& other) = delete;

    virtual void Process(std::shared_ptr<const plugin_configuration> conf);

    virtual std::string ClassName() const
    {
        return "himan::plugin::transformer";
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

	// Check and write json parameters needed for transformer plug-in to local variables.
	void set_additional_parameters();
        std::vector<level> LevelsFromString(const std::string& levelType, const std::string& levelValues) const;
/*
#ifdef HAVE_CUDA
	std::unique_ptr<transformer_cuda::options> CudaPrepare(std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> sourceInfo);
	void CudaFinish(std::unique_ptr<transformer_cuda::options> opts, std::shared_ptr<info> myTargetInfo, std::shared_ptr<info> sourceInfo);
#endif
*/

	double itsBase;
	double itsScale;
	std::string itsSourceParam;
	std::string itsTargetParam;
	int itsTargetUnivID;
	std::vector<level> itsSourceLevels;

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<transformer> (new transformer());
}

} // namespace plugin
} // namespace himan

#endif /* TRANSFORMER */
