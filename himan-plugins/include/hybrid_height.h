/**
 * @file hybrid_height.h
 *
 * @date Apr 5, 2013
 * @author peramaki
 */

#ifndef HYBRID_HEIGHT_H
#define HYBRID_HEIGHT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

class hybrid_height : public compiled_plugin, private compiled_plugin_base
{
public:
	hybrid_height();

	inline virtual ~hybrid_height() {}

	hybrid_height(const hybrid_height& other) = delete;
	hybrid_height& operator=(const hybrid_height& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::hybrid_height";
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

	void Run(std::shared_ptr<info> myTargetInfo, std::shared_ptr<const plugin_configuration> theConfiguration, unsigned short threadIndex);
    void Calculate(std::shared_ptr<info> myTargetInfo, std::shared_ptr<const plugin_configuration> theConfiguration, unsigned short threadIndex);

    std::shared_ptr<info> FetchPrevious(std::shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime, const level& wantedLevel, const param& wantedParam);
    int itsBottomLevel;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<hybrid_height> (new hybrid_height());
}

} // namespace plugin
} // namespace himan


#endif /* HYBRID_HEIGHT_H */
