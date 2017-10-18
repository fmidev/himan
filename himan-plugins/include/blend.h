#ifndef BLEND_H
#define BLEND_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
// Gather producer, geometry, and time information.
struct meta
{
	producer prod;
	std::string geom;
	forecast_type type;
	forecast_time ftime;
	level lvl;
};

class blend : public compiled_plugin, private compiled_plugin_base
{
   public:
	blend();
	virtual ~blend();

	blend(const blend& other) = delete;
	blend& operator=(const blend& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::blend"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 1); }

	virtual void Start();
	void Run(unsigned short threadIndex);

   protected:
	virtual void Calculate(std::shared_ptr<info> targetInfo, unsigned short threadIndex);
	virtual void WriteToFile(const info& targetInfo, write_options opts = write_options()) override;

   private:
	void SetupOutputForecastTypes(std::shared_ptr<info> Info);

	std::vector<meta> itsMetaOpts;
	std::vector<double> itsWeights;
};

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<himan_plugin>(new blend()); }

}  // namespace plugin
}  // namespace himan

// BLEND_H
#endif
