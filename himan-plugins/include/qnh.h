/*
 * qnh.h
 *
 */

#ifndef QNH_H
#define QNH_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class qnh
 *
 */

class qnh : public compiled_plugin, private compiled_plugin_base
{
   public:
	qnh();

	inline virtual ~qnh()
	{
	}
	qnh(const qnh& other) = delete;
	qnh& operator=(const qnh& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::qnh";
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
	return std::shared_ptr<qnh>(new qnh());
}
}  // namespace plugin
}  // namespace himan

#endif /* qnh_H */
