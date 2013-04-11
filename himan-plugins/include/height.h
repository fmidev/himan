/**
 * @file height.h
 *
 * @date Apr 5, 2013
 * @author peramaki
 */

#ifndef HEIGHT_H
#define HEIGHT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

class height : public compiled_plugin, private compiled_plugin_base
{
public:
	height();

	inline virtual ~height() {}

	height(const height& other) = delete;
	height& operator=(const height& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::height";
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

    bool itsUseCuda;
    int itsCudaDeviceCount;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
    return std::shared_ptr<height> (new height());
}

} // namespace plugin
} // namespace himan


#endif /* HEIGHT_H */