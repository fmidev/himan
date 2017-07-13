/**
 * @file fractile.cpp
 *
 **/

#ifndef FRACTILE_PLUGIN_H
#define FRACTILE_PLUGIN_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

#include <stdint.h>

namespace himan
{
namespace plugin
{
class fractile : public compiled_plugin, private compiled_plugin_base
{
   public:
	fractile();

	virtual ~fractile();

	fractile(const fractile& other) = delete;
	fractile& operator=(const fractile& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::fractile"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> myTargetInfo, uint16_t threadIndex);
	std::string itsParamName;
	int itsEnsembleSize;
	HPEnsembleType itsEnsembleType;
	std::vector<double> itsFractiles;
	int itsLag;
	int itsLaggedSteps;
	int itsMaximumMissingForecasts;
};

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<fractile>(); }
}  // plugin
}  // himan

// FRACTILE_PLUGIN_H
#endif
