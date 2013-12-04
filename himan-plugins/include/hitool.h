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

	std::shared_ptr<info> VerticalMaximum(const param& wantedParam, const std::shared_ptr<info> firstLevelValueInfo, const std::shared_ptr<info> lastLevelValueInfo, size_t findNth = 1) const;

	/**
	 * @brief Find minimum value of a given parameter in a given height range
	 */

	std::shared_ptr<info> VerticalMinimum(const param& wantedParam, const std::shared_ptr<info> firstLevelValueInfo, const std::shared_ptr<info> lastLevelValueInfo, size_t findNth = 1) const;
	std::shared_ptr<info> VerticalAverage(const param& wantedParam, const std::shared_ptr<info> firstLevelValueInfo, const std::shared_ptr<info> lastLevelValueInfo) const;
	std::shared_ptr<info> VerticalHeight(const param& wantedParam, const std::shared_ptr<info> firstLevelValueInfo, const std::shared_ptr<info> lastLevelValueInfo, const std::shared_ptr<info> findValueInfo , size_t findNth = 1) const;

	/**
	 * @brief Find value of parameter from given height
	 *
	 * Only for hybrid levels.
	 * 
	 */
	
	std::shared_ptr<info> VerticalValue(const param& wantedParam, const std::shared_ptr<info> findValueInfo) const;

	/**
	 * @brief Find the number of occurrences of a given parameter value in a given height range
	 *
	 * Only for hybrid levels.
	 */

	std::shared_ptr<info> VerticalCount(const param& wantedParam, const std::shared_ptr<info> firstLevelValueInfo, const std::shared_ptr<info> lastLevelValueInfo, const std::shared_ptr<info> findValueInfo) const;

	std::shared_ptr<info> Stratus(std::shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime);
	std::shared_ptr<info> FreezingArea(std::shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime);


	void Time(const forecast_time& theTime);
	void Configuration(const std::shared_ptr<const plugin_configuration> conf);

private:
	std::shared_ptr<modifier> CreateModifier(HPModifierType modifierType) const;

	std::shared_ptr<info> VerticalExtremeValue(std::shared_ptr<modifier> mod,
							HPLevelType wantedLevelType,
							const param& sourceParam,
							const param& targetParam,
							const std::shared_ptr<info> firstLevelValueInfo = std::shared_ptr<info> (),
							const std::shared_ptr<info> lastLevelValueInfo = std::shared_ptr<info> (),
							const std::shared_ptr<info> findValueInfo = std::shared_ptr<info> (),
							size_t findNth = 1) const;

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
