// vim: set ai shiftwidth=4 softtabstop=4 tabstop=4 expandtab
//
// @file probability.h
//
// @date May 17, 2016
// @author vanhatam
//

#ifndef PROBABILITY_PLUGIN_H
#define PROBABILITY_PLUGIN_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

#include <stdint.h>

namespace himan
{
namespace plugin
{

/// @brief describes how output parameters are calculated from input parameters
struct param_configuration
{
	int targetInfoIndex;
	double threshold;

	param output;

	// Input parameters used for calculating the 'target'.
	// Usually only one parameter is used, but wind probability
	// consists of U and V components.
	param parameter;
	param parameter2;
};

class probability : public compiled_plugin, private compiled_plugin_base
{
public:
	probability();

	virtual ~probability();

	probability(const probability& other) = delete;
	probability& operator=(const probability& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual void WriteToFile(const info& targetInfo, const size_t targetInfoIndex,
	                         write_options opts = write_options());

	virtual std::string ClassName() const
	{
		return "himan::plugin::probability";
	}

	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}

	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(0, 1);
	}

private:
	virtual void Calculate(uint16_t threadIndex, const param_configuration& pc);

	int itsEnsembleSize;
	bool itsUseNormalizedResult;
	std::vector<param_configuration> itsParamConfigurations;
};

extern "C" std::shared_ptr<himan_plugin> create() 
{ 
	return std::make_shared<probability>();
}

}  // plugin
}  // himan

// PROBABILITY_PLUGIN_H
#endif
