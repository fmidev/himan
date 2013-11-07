/**
 * @file precipitation.h
 *
 * @date Jan 28, 2012
 * @author: partio
 */

#ifndef PRECIPITATION_H
#define PRECIPITATION_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

/**
 * @class precipitation
 *
 * @brief Calculate 1/3/6/12 hour precipitation from cumulative precipitation or
 * large scale and convective precipitation
 *
 */

class precipitation : public compiled_plugin, private compiled_plugin_base
{
public:
	precipitation();

	inline virtual ~precipitation() {}

	precipitation(const precipitation& other) = delete;
	precipitation& operator=(const precipitation& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::precipitation";
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

	void Run(std::shared_ptr<info>, std::shared_ptr<const plugin_configuration> conf, unsigned short threadIndex);
	void Calculate(std::shared_ptr<info> myTargetInfo, std::shared_ptr<const plugin_configuration> conf, unsigned short threadIndex, std::shared_ptr<info> curRRInfo, std::shared_ptr<info> prevRRInfo);

	std::shared_ptr<himan::info> GetSourceDataForSum(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime, bool& dataFoundFromRRParam);
	std::shared_ptr<himan::info> GetSourceDataForRate(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<const info> myTargetInfo, bool& dataFoundFromRRParam, bool forward);

	/**
	 * @brief Overcoat for fetching source data. By default will first try fetch data
	 * from cumulative precipitation parameter (RR-KGM2), if data is not found function
	 * will try to fetch data from cumulative and large scale precipitation parameters.
	 *
	 * Will pass throw'd exceptions to calling function
	 *
	 * @param conf himan configuration
	 * @param wantedTime Wanted time
	 * @param wantedLevel Wanted level
	 * @param dataFoundFromRRParam If true, data will first be search from cumulative precipitation parameter (RR-KGM2, 50)
	 * @return himan::info contain source data
	 */

	std::shared_ptr<info> FetchSourcePrecipitation(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime, bool& dataFoundFromRRParam);

	/**
	 * @brief Fetching source data from cumulative precipitation parameter (RR-KGM2, 50)
	 *
	 * Will pass throw'd exceptions to calling function

	 * @param conf himan configuration
	 * @param wantedTime Wanted time
	 * @param wantedLevel Wanted level
	 * @return himan::info contain source data
	 */

	std::shared_ptr<info> FetchSourceRR(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime, const level& wantedLevel);

	/**
	 * @brief Fetching source data from convective (RRC-KGM2) and large scale (RRR-KGM2) precipitation parameters
	 *
	 * Will pass throw'd exceptions to calling function

	 * @param conf himan configuration
	 * @param wantedTime Wanted time
	 * @param wantedLevel Wanted level
	 * @return himan::info contain source data
	 */

	std::shared_ptr<info> FetchSourceConvectiveAndLSRR(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime, const level& wantedLevel);

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<himan_plugin>(new precipitation());
}

} // namespace plugin
} // namespace himan

#endif /* PRECIPITATION_H */
