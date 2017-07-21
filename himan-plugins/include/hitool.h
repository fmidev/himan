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
#include "modifier.h"
#include "plugin_configuration.h"

namespace himan
{
namespace plugin
{
typedef std::pair<std::shared_ptr<info>, std::shared_ptr<info>> valueheight;

class hitool : public auxiliary_plugin
{
   public:
	hitool();
	explicit hitool(std::shared_ptr<plugin_configuration> conf);

	virtual ~hitool(){};

	virtual std::string ClassName() const { return "himan::plugin::hitool"; }
	virtual HPPluginClass PluginClass() const { return kAuxiliary; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 0); }
	/**
	 * @brief Return maximum value in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalMaximum(params, vector<double>, vector<double>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalMaximum(const std::vector<param>& wantedParamList, double firstLevelValue,
	                                    double lastLevelValue) const;

	/**
	 * @brief Return maximum value in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalMaximum(param, vector<double>, vector<double>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalMaximum(const std::vector<param>& wantedParamList,
	                                    const std::vector<double>& firstLevelValue,
	                                    const std::vector<double>& lastLevelValue) const;

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

	std::vector<double> VerticalMaximum(const param& wantedParam, double lowerHeight, double upperHeight) const;

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

	std::vector<double> VerticalMaximum(const param& wantedParam, const std::vector<double>& firstLevelValue,
	                                    const std::vector<double>& lastLevelValue) const;

	/**
	 * @brief Return maximum value in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalMinimum(params, vector<double>, vector<double>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalMinimum(const std::vector<param>& wantedParamList, double firstLevelValue,
	                                    double lastLevelValue) const;

	/**
	 * @brief Return minimum value in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalMinimum(param, vector<double>, vector<double>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalMinimum(const std::vector<param>& wantedParamList,
	                                    const std::vector<double>& firstLevelValue,
	                                    const std::vector<double>& lastLevelValue) const;

	/**
	 * @brief Find minimum value of a given parameter in a given height range
	 *
	 * @param wantedParam Wanted parameter
	 * @param lowerHeight Lowest height value for all points, search will start here
	 * @param upperHeight Highest height value for all points, search will stop here
	 * @return Minimum value for each point
	 */

	std::vector<double> VerticalMinimum(const param& wantedParam, double lowerHeight, double upperHeight) const;

	/**
	 * @brief Find minimum value of a given parameter in a given height range
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @return Minimum value for each point
	 */

	std::vector<double> VerticalMinimum(const param& wantedParam, const std::vector<double>& firstLevelValue,
	                                    const std::vector<double>& lastLevelValue) const;

	/**
	 * @brief Return sum in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalSum(param, vector<double>, vector<double>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalSum(const std::vector<param>& wantedParamList, double firstLevelValue,
	                                double lastLevelValue) const;

	/**
	 * @brief Return sum in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalSum(param, vector<double>, vector<double>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalSum(const std::vector<param>& wantedParamList,
	                                const std::vector<double>& firstLevelValue,
	                                const std::vector<double>& lastLevelValue) const;

	/**
	 * @brief Return sum in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalSum(param, vector<double>, vector<double>)
	 *
	*/

	std::vector<double> VerticalSum(const param& wantedParam, double firstLevelValue, double lastLevelValue) const;

	/**
	 * @brief Calculate sum of values for each point in a given height range
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @return Sum for each point
	 */

	std::vector<double> VerticalSum(const param& wantedParam, const std::vector<double>& firstLevelValue,
	                                const std::vector<double>& lastLevelValue) const;

	/**
	 * @brief Return average of values in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalAverage(params, double, double)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalAverage(const std::vector<param>& wantedParamList, double lowerHeight,
	                                    double upperHeight) const;

	/**
	 * @brief Return average of values in a given height range for the first parameter found.
	 *
	 * Overcoat for VerticalAverage(param, vector<double>, vector<double>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalAverage(const std::vector<param>& wantedParamList,
	                                    const std::vector<double>& firstLevelValue,
	                                    const std::vector<double>& lastLevelValue) const;

	/**
	 * @brief Calculate average of values for each point in a given height range
	 *
	 * @param wantedParam Wanted parameter
	 * @param lowerHeight Lowest height value for all points, search will start here
	 * @param upperHeight Highest height value for all points, search will stop here
	 * @return Mean for each point
	 */

	std::vector<double> VerticalAverage(const param& wantedParam, double lowerHeight, double upperHeight) const;

	/**
	 * @brief Calculate average of values for each point in a given height range
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @return Mean for each point
	 */

	std::vector<double> VerticalAverage(const param& wantedParam, const std::vector<double>& firstLevelValue,
	                                    const std::vector<double>& lastLevelValue) const;

	/**
	 * @brief Return height for given value for the first parameter found.
	 *
	 * Overcoat for VerticalHeight(params, vector<double>, vector<double>, vector<double>, size_t)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalHeight(const std::vector<param>& wantedParamList, double firstLevelValue,
	                                   double lastLevelValue, const std::vector<double>& findValue,
	                                   size_t findNth = 1) const;

	/**
	 * @brief Return height for given value for the first parameter found.
	 *
	 * Overcoat for VerticalHeight(param, vector<double>, vector<double>, vector<double>, size_t)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalHeight(const std::vector<param>& wantedParamList,
	                                   const std::vector<double>& firstLevelValue,
	                                   const std::vector<double>& lastLevelValue, const std::vector<double>& findValue,
	                                   size_t findNth = 1) const;

	/**
	 * @brief Return height of a given parameter value.
	 *
	 * Overcoat for VerticalHeight(param, vector<double>, vector<double>, vector<double>, size_t)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 * @return Heights for given values for each point
	 */

	std::vector<double> VerticalHeight(const params& wantedParam, double firstLevelValue, double lastLevelValue,
	                                   double findValue, size_t findNth = 1) const;

	/**
	 * @brief Return height of a given parameter value.
	 *
	 * Overcoat for VerticalHeight(param, vector<double>, vector<double>, vector<double>, size_t)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 * @return Heights for given values for each point
	 */

	std::vector<double> VerticalHeight(const param& wantedParam, double firstLevelValue, double lastLevelValue,
	                                   double findValue, size_t findNth = 1) const;

	/**
	 * @brief Return height of a given parameter value.
	 *
	 * Overcoat for VerticalHeight(param, vector<double>, vector<double>, vector<double>, size_t)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	 * @return Heights for given values for each point
	 */

	std::vector<double> VerticalHeight(const param& wantedParam, double firstLevelValue, double lastLevelValue,
	                                   const std::vector<double>& findValue, size_t findNth = 1) const;

	/**
	 * @brief Find height of a given parameter value.
	 *
	 * Only for hybrid levels. A height between two levels is interpolated linearly.
	 *
	 * If findNth > 1 and value is not found (although lower count values are found),
	 * value is set to Missing (unlike in smarttool).
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @param findValue Wanted value for each point
	 * @param findNth Return the height of Nth found value
	 * @return Heights for given values for each point
	 */

	std::vector<double> VerticalHeight(const param& wantedParam, const std::vector<double>& firstLevelValue,
	                                   const std::vector<double>& lastLevelValue, const std::vector<double>& findValue,
	                                   size_t findNth = 1) const;

	/**
	 * @brief VerticalHeightGreaterThan() is similar to VerticalHeight(), but when searching
	 * for a value it also considers the situation where a search is started and the value
	 * is encountered in the very first height.
	 *
	 * For example when searching for a height where cloudiness is > 50%, regular VerticalHeight()
	 * does not understand the situation where the first value read is already above the threshold
	 * (stratus cloud).
	 */

	std::vector<double> VerticalHeightGreaterThan(const param& wantedParam, double firstLevelValue,
	                                              double lastLevelValue, const std::vector<double>& findValue,
	                                              size_t findNth = 1) const;
	std::vector<double> VerticalHeightGreaterThan(const param& wantedParam, double firstLevelValue,
	                                              double lastLevelValue, double findValue, size_t findNth = 1) const;
	std::vector<double> VerticalHeightGreaterThan(const params& wantedParam, double firstLevelValue,
	                                              double lastLevelValue, double findValue, size_t findNth = 1) const;
	std::vector<double> VerticalHeightGreaterThan(const std::vector<param>& wantedParamList,
	                                              const std::vector<double>& firstLevelValue,
	                                              const std::vector<double>& lastLevelValue,
	                                              const std::vector<double>& findValue, size_t findNth = 1) const;
	std::vector<double> VerticalHeightGreaterThan(const std::vector<param>& wantedParamList, double firstLevelValue,
	                                              double lastLevelValue, const std::vector<double>& findValue,
	                                              size_t findNth = 1) const;
	std::vector<double> VerticalHeightGreaterThan(const param& wantedParam, const std::vector<double>& firstLevelValue,
	                                              const std::vector<double>& lastLevelValue,
	                                              const std::vector<double>& findValue, size_t findNth = 1) const;

	/**
	 * @brief For description, look at VerticalHeightGreaterThan()
	 */

	std::vector<double> VerticalHeightLessThan(const param& wantedParam, double firstLevelValue, double lastLevelValue,
	                                           const std::vector<double>& findValue, size_t findNth = 1) const;
	std::vector<double> VerticalHeightLessThan(const param& wantedParam, double firstLevelValue, double lastLevelValue,
	                                           double findValue, size_t findNth = 1) const;
	std::vector<double> VerticalHeightLessThan(const params& wantedParam, double firstLevelValue, double lastLevelValue,
	                                           double findValue, size_t findNth = 1) const;
	std::vector<double> VerticalHeightLessThan(const std::vector<param>& wantedParamList,
	                                           const std::vector<double>& firstLevelValue,
	                                           const std::vector<double>& lastLevelValue,
	                                           const std::vector<double>& findValue, size_t findNth = 1) const;
	std::vector<double> VerticalHeightLessThan(const std::vector<param>& wantedParamList, double firstLevelValue,
	                                           double lastLevelValue, const std::vector<double>& findValue,
	                                           size_t findNth = 1) const;

	std::vector<double> VerticalHeightLessThan(const param& wantedParam, const std::vector<double>& firstLevelValue,
	                                           const std::vector<double>& lastLevelValue,
	                                           const std::vector<double>& findValue, size_t findNth = 1) const;

	/**
	 * @brief Return value of parameter from given height for the first parameter found.
	 *
	 * Overcoat for VerticalValue(params, double, double)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> VerticalValue(const std::vector<param>& wantedParamList, double findValue) const;

	/**
	 * @brief Return value of parameter from given height for the first parameter found.
	 *
	 * Overcoat for VerticalValue(param, vector<double>)
	 *
	 * @return Values for given heights
	 */

	std::vector<double> VerticalValue(const std::vector<param>& wantedParamList,
	                                  const std::vector<double>& findValue) const;

	/**
	 * @brief Find value of parameter from given height
	 *
	 * Only for hybrid levels. A value between two levels is interpolated linearly.
	 *
	 * @param wantedParam Wanted parameter
	 * @param findValue Height for all points
	 * @return Values for given height for each point
	 */

	std::vector<double> VerticalValue(const param& wantedParam, double findValue) const;

	/**
	 * @brief Find value of parameter from given height
	 *
	 * Only for hybrid levels. A value between two levels is interpolated linearly.
	 *
	 * @param wantedParam Wanted parameter
	 * @param findValue Height for each point
	 * @return Values for given heights for each point
	 */

	std::vector<double> VerticalValue(const param& wantedParam, const std::vector<double>& findValue) const;

	/**
	 * @brief Find the number of occurrences of a given parameter value in a given height range for the first parameter
	 * found.
	 *
	 * Overcoat for VerticalCount(param, vector<double>, vector<double>, vector<double>)
	 *
	 * @return Values for given heights
	 */

	std::vector<double> VerticalCount(const std::vector<param>& wantedParamList, double firstLevelValue,
	                                  double lastLevelValue, double findValue) const;

	/**
	 * @brief Find the number of occurrences of a given parameter value in a given height range for the first parameter
	 * found.
	 *
	 * Overcoat for VerticalCount(param, vector<double>, vector<double>, vector<double>)
	 *
	 * @return Values for given heights
	 */

	std::vector<double> VerticalCount(const std::vector<param>& wantedParamList,
	                                  const std::vector<double>& firstLevelValue,
	                                  const std::vector<double>& lastLevelValue,
	                                  const std::vector<double>& findValue) const;

	/**
	 * @brief Find the number of occurrences of a given parameter value in a given height range.
	 *
	 * Overcoat for VerticalCount(param, vector<double>, vector<double>, vector<double>)
	 *
	 * @return Values for given heights
	 */

	std::vector<double> VerticalCount(const param& wantedParam, double firstLevelValue, double lastLevelValue,
	                                  double findValue) const;

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

	std::vector<double> VerticalCount(const param& wantedParam, const std::vector<double>& firstLevelValue,
	                                  const std::vector<double>& lastLevelValue,
	                                  const std::vector<double>& findValue) const;

	/**
	 * @brief Return the negative and positive area under a curve/vertical profile. First half of return vector contains
	 * positive area, second half negative area.
	 *
	 * Overcoat for PlusMinusArea(params, double, double)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> PlusMinusArea(const std::vector<param>& wantedParamList, double lowerHeight,
	                                  double upperHeight) const;

	/**
	 * @brief Return the negative and positive area under a curve/vertical profile. First half of return vector contains
	 * positive area, second half negative area.
	 *
	 * Overcoat for PlusMinusArea(param, vector<double>, vector<double>)
	 *
	 * @param wantedParamList List (vector) of wanted parameters
	*/

	std::vector<double> PlusMinusArea(const std::vector<param>& wantedParamList,
	                                  const std::vector<double>& firstLevelValue,
	                                  const std::vector<double>& lastLevelValue) const;

	/**
	 * @brief Return the negative and positive area under a curve/vertical profile. First half of return vector contains
	 * positive area, second half negative area.
	 *
	 * @param wantedParam Wanted parameter
	 * @param lowerHeight Lowest height value for all points, search will start here
	 * @param upperHeight Highest height value for all points, search will stop here
	 * @return vector that contains positive area, second half negative area
	 */

	std::vector<double> PlusMinusArea(const param& wantedParam, double lowerHeight, double upperHeight) const;

	/**
	 * @brief Calculate negative and positive area under a curve/vertical profile.
	 *
	 * @param wantedParam Wanted parameter
	 * @param firstLevelValue Lowest level value for each point, search will start here
	 * @param lastLevelValue Highest level value for each point, search will stop here
	 * @return vector that contains positive area, second half negative area
	 */

	std::vector<double> PlusMinusArea(const param& wantedParam, const std::vector<double>& firstLevelValue,
	                                  const std::vector<double>& lastLevelValue) const;

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

	/**
	 * @brief Determine minimum and maximum hybrid levels that can contain data for a specific height
	 *
	 * @param prod Producer from where data is fetched, function will try to determine correct producer even if raw
	 * producer is defined
	 * @param height The height in question, meters or hPa
	 * @return Two level definitions, minimum level that contains this height and maximum level
	 */

	std::pair<level, level> LevelForHeight(const producer& prod, double height) const;

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

	std::vector<double> VerticalExtremeValue(std::shared_ptr<modifier> mod, HPLevelType wantedLevelType,
	                                         const param& wantedParam,
	                                         const std::vector<double>& firstLevelValue = std::vector<double>(),
	                                         const std::vector<double>& lastLevelValue = std::vector<double>(),
	                                         const std::vector<double>& findValue = std::vector<double>()) const;

	valueheight GetData(const level& wantedLevel, const param& wantedParam, const forecast_time& wantedTime,
	                    const forecast_type& theType) const;

	std::shared_ptr<const plugin_configuration> itsConfiguration;
	forecast_time itsTime;
	forecast_type itsForecastType;

	/**
	 * @brief Height from ground can be either meters (HPParameterUnit::kM) or hectopascals (kHPa)
	 */

	HPParameterUnit itsHeightUnit;
};

#ifndef HIMAN_AUXILIARY_INCLUDE

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<hitool>(); }
#define HIMAN_AUXILIARY_INCLUDE
#endif /* HIMAN_AUXILIARY_INCLUDE */

}  // namespace plugin
}  // namespace himan

#endif /* HITOOL_H */
