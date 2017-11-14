/**
 * @file hybrid_height.h
 *
 */

#ifndef HYBRID_HEIGHT_H
#define HYBRID_HEIGHT_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include <future>

namespace himan
{
namespace plugin
{
class hybrid_height : public compiled_plugin, private compiled_plugin_base
{
   public:
	hybrid_height();

	virtual ~hybrid_height();

	hybrid_height(const hybrid_height& other) = delete;
	hybrid_height& operator=(const hybrid_height& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::hybrid_height"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(1, 2); }
   protected:
	void RunTimeDimension(himan::info_t myTargetInfo, unsigned short threadIndex) override;

   private:
	virtual void Calculate(std::shared_ptr<info> myTargetInfo, unsigned short threadIndex);
	bool WithHypsometricEquation(info_t& myTargetInfo);
	bool WithGeopotential(info_t& myTargetInfo);
	std::shared_ptr<himan::info> GetSurfacePressure(std::shared_ptr<himan::info>& myTargetInfo);

	int itsBottomLevel;
	bool itsUseGeopotential;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<hybrid_height>(); }
}  // namespace plugin
}  // namespace himan

#endif /* HYBRID_HEIGHT_H */
