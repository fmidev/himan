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

	virtual ~hybrid_height() = default;

	hybrid_height(const hybrid_height& other) = delete;
	hybrid_height& operator=(const hybrid_height& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::hybrid_height";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

	void WriteToFile(const std::shared_ptr<info<float>> targetInfo, write_options opts = write_options()) override;

   private:
	virtual void Calculate(std::shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex) override;
	void WriteSingleGridToFile(const std::shared_ptr<info<float>> targetInfo);
	bool WithHypsometricEquation(std::shared_ptr<info<float>>& myTargetInfo);
	bool WithGeopotential(std::shared_ptr<info<float>>& myTargetInfo);
	std::shared_ptr<himan::info<float>> GetSurfacePressure(std::shared_ptr<himan::info<float>>& myTargetInfo);

	int itsBottomLevel;
	bool itsUseGeopotential;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<hybrid_height>();
}
}  // namespace plugin
}  // namespace himan

#endif /* HYBRID_HEIGHT_H */
