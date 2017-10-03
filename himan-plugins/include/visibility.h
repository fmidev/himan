/*
 * visibility.h
 *
 */

#ifndef VISIBILITY_H
#define VISIBILITY_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class visibility
 *
 * @brief Calculate ...
 *
 */

class visibility : public compiled_plugin, private compiled_plugin_base
{
   public:
	visibility();

	inline virtual ~visibility() {}
	visibility(const visibility& other) = delete;
	visibility& operator=(const visibility& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::visibility"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 0); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
	std::shared_ptr<info> FetchSourceRR(const forecast_time& wantedTime, const level& wantedLevel);
	void VertMax(std::shared_ptr<info> myTargetInfo, std::vector<double>& in, himan::param p, int low, int high);
	void VertMax(std::shared_ptr<info> myTargetInfo, std::vector<double>& in, std::vector<himan::param> p, int low,
	             int high);
	void VertAvg(std::shared_ptr<info> myTargetInfo, std::vector<double>& in, std::vector<himan::param> p, int low,
	             int high);

	void VertTMin(std::shared_ptr<info> myTargetInfo, std::vector<double>& in, int low, int high);
	void VertFFValue(std::shared_ptr<info> myTargetInfo, std::vector<double>& in, std::vector<double>&);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<visibility>(new visibility()); }
}  // namespace plugin
}  // namespace himan

#endif /* VISIBILITY_H */
