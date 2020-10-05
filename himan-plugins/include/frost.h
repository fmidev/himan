#ifndef FROST_H
#define FROST_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
class frost : public compiled_plugin, private compiled_plugin_base
{
   public:
	frost();

	inline virtual ~frost()
	{
	}
	frost(const frost& other) = delete;
	frost& operator=(const frost& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::frost";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<double>> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<frost>(new frost());
}
}  // namespace plugin
}  // namespace himan

#endif /* FROST_H */
