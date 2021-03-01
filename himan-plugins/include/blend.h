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
	kCalculateNone,
	kCalculateBlend,
	kCalculateMAE,
	kCalculateBias
};

struct blend_producer
{
	enum
	{
		kNone = 0,
		kMos = 1,
		kEcmwf = 2,
		kHirlam = 3,
		kMeps = 4,
		kGfs = 5
	};

	blend_producer() : type(), forecastLength(), originTimestep(0)
	{
	}

	blend_producer(const forecast_type& _type, int _forecastLength, int _originTimestep)
	    : type(_type), forecastLength(_forecastLength), originTimestep(_originTimestep)
	{
	}

	bool operator==(const blend_producer& other) const
	{
		return type == other.type && forecastLength == other.forecastLength && originTimestep == other.originTimestep;
	}

	forecast_type type;
	int forecastLength;
	int originTimestep;
};

class blend : public compiled_plugin, private compiled_plugin_base
{
   public:
	blend();
	virtual ~blend() = default;

	blend(const blend& other) = delete;
	blend& operator=(const blend& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::blend";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   protected:
	virtual void Calculate(std::shared_ptr<info<double>> targetInfo, unsigned short threadIndex) override;
	virtual void WriteToFile(const std::shared_ptr<info<double>> targetInfo,
	                         write_options opts = write_options()) override;

   private:
	void CalculateBlend(std::shared_ptr<info<double>> targetInfo, unsigned short threadIndex);
	void CalculateMember(std::shared_ptr<info<double>> targetInfo, unsigned short threadIndex, blend_mode mode);

	matrix<double> CalculateMAE(std::shared_ptr<info<double>> targetInfo, const forecast_time& calcTime);
	matrix<double> CalculateBias(std::shared_ptr<info<double>> targetInfo, const forecast_time& calcTime);

	void SetupOutputForecastTimes(std::shared_ptr<info<double>> Info, const raw_time& latestOrigin,
	                              const forecast_time& current, int maxStep, int originTimeStep);
	bool ParseConfigurationOptions(const std::shared_ptr<const plugin_configuration>& conf);
	std::vector<std::shared_ptr<info<double>>> FetchRawGrids(std::shared_ptr<info<double>> targetInfo,
	                                                         unsigned short threadIdx) const;
	std::vector<std::shared_ptr<info<double>>> FetchMAEAndBiasGrids(std::shared_ptr<info<double>> targetInfo,
	                                                                unsigned short threadIdx, blend_mode type) const;

	std::tuple<std::shared_ptr<info<double>>, std::shared_ptr<info<double>>, std::shared_ptr<info<double>>,
	           std::shared_ptr<info<double>>>
	FetchMAEAndBiasSource(std::shared_ptr<info<double>>& targetInfo, const forecast_time& calcTime,
	                      blend_mode type) const;

	blend_mode itsCalculationMode;
	int itsNumHours;
	forecast_time itsAnalysisTime;  // store observation analysis time
	blend_producer itsBlendProducer;
};

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<blend>();
}

}  // namespace plugin
}  // namespace himan

// BLEND_H
#endif
