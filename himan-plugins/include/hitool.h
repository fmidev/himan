/**
 * @file hitool.h
 *
 * @brief Contains smarttool functions converted to himan
 */

#ifndef HITOOL_H
#define HITOOL_H

#include "auxiliary_plugin.h"
#include "forecast_time.h"
#include "forecast_type.h"
#include "himan_common.h"
#include "info.h"
#include "modifier.h"
#include "plugin_configuration.h"

namespace himan
{
namespace plugin
{
class hitool : public auxiliary_plugin
{
   public:
	hitool();
	explicit hitool(std::shared_ptr<plugin_configuration> conf);

	virtual ~hitool(){};

	virtual std::string ClassName() const override
	{
		return "himan::plugin::hitool";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kAuxiliary;
	}

	/**
	 * @brief Return maximum value in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalMaximum(params, vector<T>, vector<T>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> VerticalMaximum(const std::vector<param>& wantedParamList, T firstLevelValue,
	                               T lastLevelValue) const;

	/**
	 * @brief Return maximum value in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalMaximum(param, vector<T>, vector<T>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> VerticalMaximum(const std::vector<param>& wantedParamList, const std::vector<T>& firstLevelValue,
	                               const std::vector<T>& lastLevelValue) const;

	/**
	 * @brief Find maximum value of a given parameter in a given height range
	 *
	 * Only for hybrid levels
	 *
	 * @param wantedParam Wanted parameter
	 * @param lowerHeight Lowest height for all points, search will start here
	 * @param upperHeight Highest height for all points, search will stop here
	 * @return Maximum value for each point
	 */

	template <typename T>
	std::vector<T> VerticalMaximum(const param& wantedParam, T lowerHeight, T upperHeight) const;

	/**
	 * @brief Find maximum value of a given parameter in a given height range
	 *
	 * Only for hybrid levels
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @return Maximum value for each point
	 */

	template <typename T>
	std::vector<T> VerticalMaximum(const param& wantedParam, const std::vector<T>& firstLevelValue,
	                               const std::vector<T>& lastLevelValue) const;

	/**
	 * @brief Return maximum value in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalMinimum(params, vector<T>, vector<T>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> VerticalMinimum(const std::vector<param>& wantedParamList, T firstLevelValue,
	                               T lastLevelValue) const;

	/**
	 * @brief Return minimum value in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalMinimum(param, vector<T>, vector<T>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> VerticalMinimum(const std::vector<param>& wantedParamList, const std::vector<T>& firstLevelValue,
	                               const std::vector<T>& lastLevelValue) const;

	/**
	 * @brief Find minimum value of a given parameter in a given height range
	 *
	 * @param wantedParam Wanted parameter
	 * @param lowerHeight Lowest height value for all points, search will start here
	 * @param upperHeight Highest height value for all points, search will stop here
	 * @return Minimum value for each point
	 */

	template <typename T>
	std::vector<T> VerticalMinimum(const param& wantedParam, T lowerHeight, T upperHeight) const;

	/**
	 * @brief Find minimum value of a given parameter in a given height range
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @return Minimum value for each point
	 */

	template <typename T>
	std::vector<T> VerticalMinimum(const param& wantedParam, const std::vector<T>& firstLevelValue,
	                               const std::vector<T>& lastLevelValue) const;

	/**
	 * @brief Return sum in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalSum(param, vector<T>, vector<T>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> VerticalSum(const std::vector<param>& wantedParamList, T firstLevelValue, T lastLevelValue) const;

	/**
	 * @brief Return sum in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalSum(param, vector<T>, vector<T>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> VerticalSum(const std::vector<param>& wantedParamList, const std::vector<T>& firstLevelValue,
	                           const std::vector<T>& lastLevelValue) const;

	/**
	 * @brief Return sum in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalSum(param, vector<T>, vector<T>)
	 *
	 */

	template <typename T>
	std::vector<T> VerticalSum(const param& wantedParam, T firstLevelValue, T lastLevelValue) const;

	/**
	 * @brief Calculate sum of values for each point in a given height range
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @return Sum for each point
	 */

	template <typename T>
	std::vector<T> VerticalSum(const param& wantedParam, const std::vector<T>& firstLevelValue,
	                           const std::vector<T>& lastLevelValue) const;

	/**
	 * @brief Return average of values in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalAverage(params, T, T)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> VerticalAverage(const std::vector<param>& wantedParamList, T lowerHeight, T upperHeight) const;

	/**
	 * @brief Return average of values in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalAverage(param, vector<T>, vector<T>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> VerticalAverage(const std::vector<param>& wantedParamList, const std::vector<T>& firstLevelValue,
	                               const std::vector<T>& lastLevelValue) const;

	/**
	 * @brief Calculate average of values for each point in a given height range
	 *
	 * @param wantedParam Wanted parameter
	 * @param lowerHeight Lowest height value for all points, search will start here
	 * @param upperHeight Highest height value for all points, search will stop here
	 * @return Mean for each point
	 */

	template <typename T>
	std::vector<T> VerticalAverage(const param& wantedParam, T lowerHeight, T upperHeight) const;

	/**
	 * @brief Calculate average of values for each point in a given height range
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @return Mean for each point
	 */
	template <typename T>
	std::vector<T> VerticalAverage(const param& wantedParam, const std::vector<T>& firstLevelValue,
	                               const std::vector<T>& lastLevelValue) const;

	/**
	 * @brief Return height for given value for the first parameter found.
	 *
	 * Overcoat for VerticalHeight(params, vector<T>, vector<T>, vector<T>, size_t)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */
	template <typename T>
	std::vector<T> VerticalHeight(const std::vector<param>& wantedParamList, T firstLevelValue, T lastLevelValue,
	                              const std::vector<T>& findValue, int findNth = 1) const;

	/**
	 * @brief Return height for given value for the first parameter found.
	 *
	 * Overcoat for VerticalHeight(param, vector<T>, vector<T>, vector<T>, size_t)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */
	template <typename T>
	std::vector<T> VerticalHeight(const std::vector<param>& wantedParamList, const std::vector<T>& firstLevelValue,
	                              const std::vector<T>& lastLevelValue, const std::vector<T>& findValue,
	                              int findNth = 1) const;

	/**
	 * @brief Return height of a given parameter value.
	 *
	 * Overcoat for VerticalHeight(param, vector<T>, vector<T>, vector<T>, size_t)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 * @return Heights for given values for each point
	 */
	template <typename T>
	std::vector<T> VerticalHeight(const params& wantedParam, T firstLevelValue, T lastLevelValue, T findValue,
	                              int findNth = 1) const;

	/**
	 * @brief Return height of a given parameter value.
	 *
	 * Overcoat for VerticalHeight(param, vector<T>, vector<T>, vector<T>, size_t)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 * @return Heights for given values for each point
	 */
	template <typename T>
	std::vector<T> VerticalHeight(const param& wantedParam, T firstLevelValue, T lastLevelValue, T findValue,
	                              int findNth = 1) const;

	/**
	 * @brief Return height of a given parameter value.
	 *
	 * Overcoat for VerticalHeight(param, vector<T>, vector<T>, vector<T>, size_t)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 * @return Heights for given values for each point
	 */
	template <typename T>
	std::vector<T> VerticalHeight(const param& wantedParam, T firstLevelValue, T lastLevelValue,
	                              const std::vector<T>& findValue, int findNth = 1) const;

	/**
	 * @brief Find height of a given parameter value.
	 *
	 * Only for hybrid levels. A height between two levels is interpolated linearly.
	 *
	 * If findNth > 1 and value is not found (although lower count values are found),
	 * value is set to Missing (unlike in smarttool).
	 * If findNth = -1, all found values are returned
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @param findValue Wanted value for each point
	 * @param findNth Return the height of Nth found value
	 * @return Heights for given values for each point
	 */
	template <typename T>
	std::vector<T> VerticalHeight(const param& wantedParam, const std::vector<T>& firstLevelValue,
	                              const std::vector<T>& lastLevelValue, const std::vector<T>& findValue,
	                              int findNth = 1) const;

	/**
	 * @brief VerticalHeightGreaterThan() is similar to VerticalHeight(), but when searching
	 * for a value it also considers the situation where a search is started and the value
	 * is encountered in the very first height.
	 *
	 * For example when searching for a height where cloudiness is > 50%, regular VerticalHeight()
	 * does not understand the situation where the first value read is already above the threshold
	 * (stratus cloud).
	 */
	template <typename T>
	std::vector<T> VerticalHeightGreaterThan(const param& wantedParam, T firstLevelValue, T lastLevelValue,
	                                         const std::vector<T>& findValue, int findNth = 1) const;
	template <typename T>
	std::vector<T> VerticalHeightGreaterThan(const param& wantedParam, T firstLevelValue, T lastLevelValue, T findValue,
	                                         int findNth = 1) const;
	template <typename T>
	std::vector<T> VerticalHeightGreaterThan(const params& wantedParam, T firstLevelValue, T lastLevelValue,
	                                         T findValue, int findNth = 1) const;
	template <typename T>
	std::vector<T> VerticalHeightGreaterThan(const std::vector<param>& wantedParamList,
	                                         const std::vector<T>& firstLevelValue,
	                                         const std::vector<T>& lastLevelValue, const std::vector<T>& findValue,
	                                         int findNth = 1) const;
	template <typename T>
	std::vector<T> VerticalHeightGreaterThan(const std::vector<param>& wantedParamList, T firstLevelValue,
	                                         T lastLevelValue, const std::vector<T>& findValue, int findNth = 1) const;
	template <typename T>
	std::vector<T> VerticalHeightGreaterThan(const param& wantedParam, const std::vector<T>& firstLevelValue,
	                                         const std::vector<T>& lastLevelValue, const std::vector<T>& findValue,
	                                         int findNth = 1) const;

	/**
	 * @brief For description, look at VerticalHeightGreaterThan()
	 */

	template <typename T>
	std::vector<T> VerticalHeightLessThan(const param& wantedParam, T firstLevelValue, T lastLevelValue,
	                                      const std::vector<T>& findValue, int findNth = 1) const;
	template <typename T>
	std::vector<T> VerticalHeightLessThan(const param& wantedParam, T firstLevelValue, T lastLevelValue, T findValue,
	                                      int findNth = 1) const;
	template <typename T>
	std::vector<T> VerticalHeightLessThan(const params& wantedParam, T firstLevelValue, T lastLevelValue, T findValue,
	                                      int findNth = 1) const;
	template <typename T>
	std::vector<T> VerticalHeightLessThan(const std::vector<param>& wantedParamList,
	                                      const std::vector<T>& firstLevelValue, const std::vector<T>& lastLevelValue,
	                                      const std::vector<T>& findValue, int findNth = 1) const;
	template <typename T>
	std::vector<T> VerticalHeightLessThan(const std::vector<param>& wantedParamList, T firstLevelValue,
	                                      T lastLevelValue, const std::vector<T>& findValue, int findNth = 1) const;
	template <typename T>
	std::vector<T> VerticalHeightLessThan(const param& wantedParam, const std::vector<T>& firstLevelValue,
	                                      const std::vector<T>& lastLevelValue, const std::vector<T>& findValue,
	                                      int findNth = 1) const;

	/**
	 * @brief Return value of parameter from given height for the first parameter found.
	 *
	 * Overcoat for VerticalValue(params, T, T)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> VerticalValue(const std::vector<param>& wantedParamList, T findValue) const;

	/**
	 * @brief Return value of parameter from given height for the first parameter found.
	 *
	 * Overcoat for VerticalValue(param, vector<T>)
	 *
	 * @return Values for given heights
	 */

	template <typename T>
	std::vector<T> VerticalValue(const std::vector<param>& wantedParamList, const std::vector<T>& findValue) const;

	/**
	 * @brief Find value of parameter from given height
	 *
	 * Only for hybrid levels. A value between two levels is interpolated linearly.
	 *
	 * @param wantedParam Wanted parameter
	 * @param findValue Height for all points
	 * @return Values for given height for each point
	 */

	template <typename T>
	std::vector<T> VerticalValue(const param& wantedParam, T findValue) const;

	/**
	 * @brief Find value of parameter from given height
	 *
	 * Only for hybrid levels. A value between two levels is interpolated linearly.
	 *
	 * @param wantedParam Wanted parameter
	 * @param findValue Height for each point
	 * @return Values for given heights for each point
	 */

	template <typename T>
	std::vector<T> VerticalValue(const param& wantedParam, const std::vector<T>& findValue) const;

	/**
	 * @brief Find the number of occurrences of a given parameter value in a given height range for the first parameter
	 * found.
	 *
	 * Overcoat for VerticalCount(param, vector<T>, vector<T>, vector<T>)
	 *
	 * @return Values for given heights
	 */

	template <typename T>
	std::vector<T> VerticalCount(const std::vector<param>& wantedParamList, T firstLevelValue, T lastLevelValue,
	                             T findValue) const;

	/**
	 * @brief Find the number of occurrences of a given parameter value in a given height range for the first parameter
	 * found.
	 *
	 * Overcoat for VerticalCount(param, vector<T>, vector<T>, vector<T>)
	 *
	 * @return Values for given heights
	 */

	template <typename T>
	std::vector<T> VerticalCount(const std::vector<param>& wantedParamList, const std::vector<T>& firstLevelValue,
	                             const std::vector<T>& lastLevelValue, const std::vector<T>& findValue) const;

	/**
	 * @brief Find the number of occurrences of a given parameter value in a given height range.
	 *
	 * Overcoat for VerticalCount(param, vector<T>, vector<T>, vector<T>)
	 *
	 * @return Values for given heights
	 */

	template <typename T>
	std::vector<T> VerticalCount(const param& wantedParam, T firstLevelValue, T lastLevelValue, T findValue) const;

	/**
	 * @brief Find the number of occurrences of a given parameter value in a given height range
	 *
	 * Only for hybrid levels.
	 *
	 * @param wantedParam Wanter parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @param findValue Value to be searched for each point
	 * @return Number of occurrences for each point
	 */

	template <typename T>
	std::vector<T> VerticalCount(const param& wantedParam, const std::vector<T>& firstLevelValue,
	                             const std::vector<T>& lastLevelValue, const std::vector<T>& findValue) const;

	/**
	 * @brief Return the negative and positive area under a curve/vertical profile. First half of return vector contains
	 * positive area, second half negative area.
	 *
	 * Overcoat for PlusMinusArea(params, T, T)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> PlusMinusArea(const std::vector<param>& wantedParamList, T lowerHeight, T upperHeight) const;

	/**
	 * @brief Return the negative and positive area under a curve/vertical profile. First half of return vector contains
	 * positive area, second half negative area.
	 *
	 * Overcoat for PlusMinusArea(param, vector<T>, vector<T>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 */

	template <typename T>
	std::vector<T> PlusMinusArea(const std::vector<param>& wantedParamList, const std::vector<T>& firstLevelValue,
	                             const std::vector<T>& lastLevelValue) const;

	/**
	 * @brief Return the negative and positive area under a curve/vertical profile. First half of return vector contains
	 * positive area, second half negative area.
	 *
	 * @param wantedParam Wanted parameter
	 * @param lowerHeight Lowest height value for all points, search will start here
	 * @param upperHeight Highest height value for all points, search will stop here
	 * @return vector that contains positive area, second half negative area
	 */

	template <typename T>
	std::vector<T> PlusMinusArea(const param& wantedParam, T lowerHeight, T upperHeight) const;

	/**
	 * @brief Calculate negative and positive area under a curve/vertical profile.
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @return vector that contains positive area, second half negative area
	 */

	template <typename T>
	std::vector<T> PlusMinusArea(const param& wantedParam, const std::vector<T>& firstLevelValue,
	                             const std::vector<T>& lastLevelValue) const;

	/**
	 * @brief Set current forecast time
	 * @param theTime Wanted time
	 */

	void Time(const forecast_time& theTime);

	/**
	 * @brief Set current forecast type. Default is deterministic forecast
	 * @param theType
	 */

	void ForecastType(const forecast_type& theType);

	/**
	 * @brief Set configuration
	 * @param conf
	 */

	void Configuration(const std::shared_ptr<const plugin_configuration> conf);

	void HeightUnit(HPParameterUnit theHeightUnit);
	HPParameterUnit HeightUnit() const;

	void LevelType(HPLevelType theType);
	HPLevelType LevelType() const;

	/**
	 * @brief Determine minimum and maximum hybrid levels that can contain data for a specific height
	 *
	 * @param prod Producer from where data is fetched, function will try to determine correct producer even if raw
	 * producer is defined
	 * @param height The height in question, meters or hPa
	 * @param geomName Name of the geometry (different areas have different characteristics)
	 * @return Two level definitions, minimum level that contains this height and maximum level
	 */

	std::pair<level, level> LevelForHeight(const producer& prod, double height, const std::string& geomName) const;

   private:
	std::shared_ptr<modifier> CreateModifier(HPModifierType modifierType) const;

	/**
	 * @brief Aggregate vertical data
	 *
	 * This function is called by all Vertical* functions. Only hybrid levels for
	 * now, will throw if data is not found.
	 *
	 * Actual calculation is done by the modifier (from himan-lib)
	 *
	 * @param mod Modifier type
	 * @param wantedLevelType Wanted level type, only hybrid allowed
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue First (ie. lowest) level value, optional
	 * @param lastLevelValue Last (ie. highest) level value, optional
	 * @param findValue List of values that are searched, optional. This is passed to the modifier.
	 * @return vector
	 */

	template <typename T>
	std::vector<T> VerticalExtremeValue(std::shared_ptr<modifier> mod, const param& wantedParam,
	                                    const std::vector<T>& firstLevelValue = std::vector<T>(),
	                                    const std::vector<T>& lastLevelValue = std::vector<T>(),
	                                    const std::vector<T>& findValue = std::vector<T>()) const;
	template <typename T>
	std::vector<T> VerticalExtremeValue(std::shared_ptr<modifier> mod, const params& wantedParams,
	                                    const std::vector<T>& firstLevelValue = std::vector<T>(),
	                                    const std::vector<T>& lastLevelValue = std::vector<T>(),
	                                    const std::vector<T>& findValue = std::vector<T>()) const;

	template <typename T>
	std::pair<std::shared_ptr<info<T>>, std::shared_ptr<info<T>>> GetData(const level& wantedLevel,
	                                                                      const param& wantedParam,
	                                                                      const forecast_time& wantedTime,
	                                                                      const forecast_type& theType) const;

	std::shared_ptr<const plugin_configuration> itsConfiguration;
	forecast_time itsTime;
	forecast_type itsForecastType;

	/**
	 * @brief Height from ground can be either meters (HPParameterUnit::kM) or hectopascals (kHPa)
	 */

	HPParameterUnit itsHeightUnit;
	mutable HPLevelType itsLevelType;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<hitool>();
}
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* HITOOL_H */
