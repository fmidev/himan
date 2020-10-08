/**
 * @file tpot.h
 *
 */

#ifndef TPOT_H
#define TPOT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
class tpot : public compiled_plugin, private compiled_plugin_base
{
   public:
	tpot();

	inline virtual ~tpot()
	{
	}
	tpot(const tpot& other) = delete;
	tpot& operator=(const tpot& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::tpot";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<double>> myTargetInfo, unsigned short theThreadIndex);

	bool itsThetaCalculation;
	bool itsThetaWCalculation;
	bool itsThetaECalculation;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<tpot>(new tpot());
}
}  // namespace plugin
}  // namespace himan

#endif /* TPOT_H */
