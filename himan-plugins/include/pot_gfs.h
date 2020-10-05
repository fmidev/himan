#ifndef POT_GFS_H
#define POT_GFS_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class time_series
 *
 * @brief contains a series of consecutive time instances for a given parameter field
 * This in preliminarily implemented with the POT plugin while the final implementation
 * will be part of himan-lib.
 */

class time_series
{
   public:
	time_series() = default;
	time_series(param theParam, size_t expectedSize);
	~time_series() = default;
	time_series(const time_series& other) = default;
	time_series& operator=(const time_series& other) = default;

	void Fetch(std::shared_ptr<const plugin_configuration> config, forecast_time start_time,
	           const HPTimeResolution& timeRes, int stepSize, int numSteps, const level& forecastLevel,
	           const forecast_type& requestedType, bool readPackedData);

	param Param() const
	{
		return itsParam;
	};

	void Param(const param& theParam)
	{
		itsParam = theParam;
	};
	size_t Size() const
	{
		return itsInfos.size();
	};
	std::vector<info_t>::iterator begin()
	{
		return itsInfos.begin();
	};
	std::vector<info_t>::iterator end()
	{
		return itsInfos.end();
	};

   private:
	param itsParam;
	std::vector<info_t> itsInfos;
};

/**
 * @class pot
 *
 * @brief Calculate the thunder probability
 *
 */

class pot_gfs : public compiled_plugin, private compiled_plugin_base
{
   public:
	pot_gfs();

	inline virtual ~pot_gfs()
	{
	}
	pot_gfs(const pot_gfs& other) = delete;
	pot_gfs& operator=(const pot_gfs& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::pot_gfs";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex);
	bool itsStrictMode;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<pot_gfs>();
}
}  // namespace plugin
}  // namespace himan

#endif /* POT_GFS_PLUGIN_H */
