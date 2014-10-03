/**
 * @file si.h
 *
 * @date Feb 13, 2014
 * @author partio
 */

#ifndef SI_H
#define SI_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

class si : public compiled_plugin, private compiled_plugin_base
{
public:
	si();

	inline virtual ~si() {}

	si(const si& other) = delete;
	si& operator=(const si& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::si";
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
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
	void ScaleBase(std::shared_ptr<info> anInfo, double scale, double base);
	void LCLAverage(std::shared_ptr<info> myTargetInfo, double fromZ, double toZ);

	int itsBottomLevel;
	int itsTopLevel;

};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	//return std::make_shared<si> ();
	return std::shared_ptr<si> (new si());
}

} // namespace plugin
} // namespace himan

#endif /* SI */
