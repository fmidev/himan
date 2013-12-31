/**
 * @file hitool.h
 *
 * @date Sep 3, 2013
 * @author partio
 *
 * @brief Contains smarttool functions converted to himan
 */

#ifndef HITOOL_H
#define HITOOL_H

#include "auxiliary_plugin.h"
#include "himan_common.h"
#include "plugin_configuration.h"
#include "modifier.h"

namespace himan
{
namespace plugin
{

typedef std::pair<std::shared_ptr<info>, std::shared_ptr<info>> valueheight;

class hitool : public auxiliary_plugin
{
public:

    hitool();
	hitool(std::shared_ptr<plugin_configuration> conf);

    virtual ~hitool() {};

    virtual std::string ClassName() const
    {
        return "himan::plugin::hitool";
    }

    virtual HPPluginClass PluginClass() const
    {
        return kAuxiliary;
    }

    virtual HPVersionNumber Version() const
    {
        return HPVersionNumber(0, 1);
    }

	/**
	 * @brief Find maximum value of a given parameter in a given height range
	 */

	std::vector<double> VerticalMaximum(const param& wantedParam, const std::vector<double>& firstLevelValue, const std::vector<double>& lastLevelValue) const;

	/**
	 * @brief Find minimum value of a given parameter in a given height range
	 */

	std::vector<double> VerticalMinimum(const param& wantedParam, const std::vector<double>& firstLevelValue, const std::vector<double>& lastLevelValue) const;

	std::vector<double> VerticalSum(const param& wantedParam, const std::vector<double>& firstLevelValue, const std::vector<double>& lastLevelValue) const;
	std::vector<double> VerticalAverage(const param& wantedParam, const std::vector<double>& firstLevelValue, const std::vector<double>& lastLevelValue) const;
	std::vector<double> VerticalHeight(const param& wantedParam, const std::vector<double>& firstLevelValue, const std::vector<double>& lastLevelValue, const std::vector<double>& findValue , size_t findNth = 1) const;

	/**
	 * @brief Find value of parameter from given height
	 *
	 * Only for hybrid levels.
	 * 
	 */
	
	std::vector<double> VerticalValue(const param& wantedParam, const std::vector<double>& findValue) const;

	/**
	 * @brief Find the number of occurrences of a given parameter value in a given height range
	 *
	 * Only for hybrid levels.
	 */

	std::vector<double> VerticalCount(const param& wantedParam, const std::vector<double>& firstLevelValue, const std::vector<double>& lastLevelValue, const std::vector<double>& findValue) const;

	std::shared_ptr<info> Stratus();
	std::shared_ptr<info> FreezingArea();

	void Time(const forecast_time& theTime);
	void Configuration(const std::shared_ptr<const plugin_configuration> conf);

private:
	std::shared_ptr<modifier> CreateModifier(HPModifierType modifierType) const;

	std::vector<double> VerticalExtremeValue(std::shared_ptr<modifier> mod,
							HPLevelType wantedLevelType,
							const param& wantedParam,
							const std::vector<double>& firstLevelValue = std::vector<double>(),
							const std::vector<double>& lastLevelValue = std::vector<double>(),
							const std::vector<double>& findValue = std::vector<double>()) const;

	valueheight GetData(const level& wantedLevel, const param& wantedParam, const forecast_time& wantedTime) const;

	std::shared_ptr<const plugin_configuration> itsConfiguration;
	forecast_time itsTime;
	
};

#ifndef HIMAN_AUXILIARY_INCLUDE

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<hitool> (new hitool());
}

#endif /* HIMAN_AUXILIARY_INCLUDE */

} // namespace plugin
} // namespace himan

#endif /* HITOOL_H */
