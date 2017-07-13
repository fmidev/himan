/**
 * @file split_sum.h
 *
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

	virtual std::string ClassName() const { return "himan::plugin::split_sum"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 2); }
   protected:
	virtual void WriteToFile(const info& targetInfo, write_options opts = write_options()) override;

   private:
	void Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex);
	void DoParam(std::shared_ptr<info> myTargetInfo, std::string myParamName, std::string threadIndex) const;

	/**
	 * @brief Fetch source data for a rate calculation.
	 *
	 * With rate we don't necessary know what are the closest time steps wrt to our
	 * own time step
	 *
	 * @param myTargetInfo Result info of calculation (will not be modified in this function)
	 * @param targetStep Step length between current and previous timestep
	 * @return Requested data, previous and current
	 */

	std::pair<std::shared_ptr<himan::info>, std::shared_ptr<himan::info>> GetSourceDataForRate(
	    std::shared_ptr<const info> myTargetInfo, int step) const;

	/**
	 * @brief Fetching source data from cumulative parameter
	 *
	 * @param conf himan configuration
	 * @param wantedTime Wanted time
	 * @param wantedLevel Wanted level
	 * @return himan::info contain source data, empty if no data found
	 */

	std::shared_ptr<info> FetchSourceData(std::shared_ptr<const info> myTargetInfo,
	                                      const forecast_time& wantedTime) const;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<himan_plugin>(new split_sum()); }
}  // namespace plugin
}  // namespace himan

#endif /* SPLIT_SUM_H */
