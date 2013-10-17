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

struct hitool_search_options
{
	himan::param wantedParam;
	himan::forecast_time wantedTime;
	himan::HPLevelType wantedLevelType;
	std::shared_ptr<info> firstLevelValueInfo;
	std::shared_ptr<info> lastLevelValueInfo;
	himan::HPModifierType wantedModifier;
	std::shared_ptr<const plugin_configuration> conf;
	bool returnHeight;
	size_t findNthValue;
	std::shared_ptr<const info> findValueInfo;

	hitool_search_options(const param& theWantedParam,
							const forecast_time& theWantedTime,
							HPLevelType theWantedLevelType,
							std::shared_ptr<info> theFirstLevelValueInfo,
							std::shared_ptr<info> theLastLevelValueInfo,
							HPModifierType theWantedModifier,
							std::shared_ptr<const plugin_configuration> theConf,
							bool theReturnHeight,
							size_t theFindNthValue)
		: wantedParam(theWantedParam)
		, wantedTime(theWantedTime)
		, wantedLevelType(theWantedLevelType)
		, firstLevelValueInfo(theFirstLevelValueInfo)
		, lastLevelValueInfo(theLastLevelValueInfo)
		, wantedModifier(theWantedModifier)
		, returnHeight(theReturnHeight)
		, findNthValue(theFindNthValue)
	{
		conf = theConf;
	}
};

class hitool : public auxiliary_plugin
{
public:

    hitool();

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

	std::shared_ptr<info> VerticalExtremeValue(hitool_search_options& opts);

	double Base() const;
	void Base(double theBase);

	double Scale() const;
	void Scale(double theScale);

	double FirstLevelValueBase() const;
	double LastLevelValueBase() const;

	void FirstLevelValueBase(double theBase);
	void LastLevelValueBase(double theBase);

	std::shared_ptr<info> Stratus(std::shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime);
	std::shared_ptr<info> FreezingArea(std::shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime);

private:
	std::shared_ptr<modifier> CreateModifier(hitool_search_options& opts, std::vector<himan::param>& params);

	valueheight GetData(const std::shared_ptr<const plugin_configuration> conf,
																	const level& wantedLevel,
																	const param& wantedParam,
																	const forecast_time& wantedTime);
	double itsScale;
	double itsBase;
	double itsFirstLevelValueBase;
	double itsLastLevelValueBase;
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
