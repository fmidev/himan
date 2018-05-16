#ifndef BLEND_H
#define BLEND_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

enum blend_mode
{
	kNone,
	//
	kCalculateBlend,
	kCalculateMAE,
	kCalculateBias
};

class blend : public compiled_plugin, private compiled_plugin_base
{
   public:
	blend();
	virtual ~blend();

	blend(const blend& other) = delete;
	blend& operator=(const blend& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::blend";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}
	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(1, 1);
	}

	virtual void Start();
	void Run(unsigned short threadIndex);

   protected:
	virtual void Calculate(std::shared_ptr<info> targetInfo, unsigned short threadIndex);
	virtual void WriteToFile(const info_t targetInfo, write_options opts = write_options()) override;

   private:
	void CalculateBlend(std::shared_ptr<info> targetInfo, unsigned short threadIndex);
	void CalculateMember(std::shared_ptr<info> targetInfo, unsigned short threadIndex, blend_mode mode);

	matrix<double> CalculateMAE(logger& log, std::shared_ptr<info> targetInfo, const forecast_type& ftype,
	                            const level& lvl, const forecast_time& calcTime);
	matrix<double> CalculateBias(logger& log, std::shared_ptr<info> targetInfo, const forecast_type& ftype,
	                             const level& lvl, const forecast_time& calcTime);

	void SetupOutputForecastTimes(std::shared_ptr<info> Info, const raw_time& latestOrigin,
	                              const forecast_time& current, int maxStep);
	raw_time LatestOriginTimeForProducer(const std::string& producer) const;

	blend_mode itsCalculationMode;
	int itsNumHours;
	std::string itsProducer;
	forecast_type itsProdFtype;
};

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<himan_plugin>(new blend());
}

}  // namespace plugin
}  // namespace himan

// BLEND_H
#endif
