/**
 * @file split_sum.h
 *
 * @date Jan 28, 2012
 * @author: partio
 */

#ifndef SPLIT_SUM_H
#define SPLIT_SUM_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

/**
 * @class split_sum
 *
 * @brief Calculate 1/3/6/12 hour sum from cumulative value or
 * calculate rate
 *
 */

class split_sum : public compiled_plugin, private compiled_plugin_base
{
public:
	split_sum();

	inline virtual ~split_sum() {}

	split_sum(const split_sum& other) = delete;
	split_sum& operator=(const split_sum& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::split_sum";
	}

	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}

	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(1, 1);
	}

private:

	void Run(std::shared_ptr<info>, std::shared_ptr<const plugin_configuration> conf, unsigned short threadIndex);
	void Calculate(std::shared_ptr<info> myTargetInfo, std::shared_ptr<const plugin_configuration> conf, unsigned short threadIndex);

	/**
	 * @brief Fetch source data for a rate calculation.
	 *
	 * With rate we don't necessary know what are the closest time steps wrt to our
	 * own time step
	 *
	 * @param conf himan configuration
	 * @param myTargetInfo Result info of calculation (will not be modified in this function)
	 * @param forward Whether to search forward in time for next data
	 * @return Requested data
	 */

	std::shared_ptr<himan::info> GetSourceDataForRate(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<const info> myTargetInfo, bool forward);

	/**
	 * @brief Fetching source data from cumulative parameter
	 *
	 * @param conf himan configuration
	 * @param wantedTime Wanted time
	 * @param wantedLevel Wanted level
	 * @return himan::info contain source data, empty if no data found
	 */

	std::shared_ptr<info> FetchSourceData(std::shared_ptr<const plugin_configuration> conf, std::shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime);

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<himan_plugin>(new split_sum());
}

} // namespace plugin
} // namespace himan

#endif /* SPLIT_SUM_H */
