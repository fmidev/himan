/*
 * @file stability.h
 *
 */

#ifndef STABILITY_H
#define STABILITY_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"
#include "stability.cuh"

namespace himan
{
namespace plugin
{
/**
 * @class stability
 *
 * @brief Calculate k-index and other indexes that describe that stability of the atmosphere.
 *
 */

class stability : public compiled_plugin, private compiled_plugin_base
{
   public:
	stability();

	inline virtual ~stability()
	{
	}
	stability(const stability& other) = delete;
	stability& operator=(const stability& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::stability";
	}
	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}
	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(2, 0);
	}
  protected:
	void RunTimeDimension(info_t myTargetInfo, unsigned short threadIndex) override;
	void WriteToFile(const info& targetInfo, write_options writeOptions = write_options()) override;

   private:
	void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::shared_ptr<stability>(new stability());
}
}  // namespace plugin
}  // namespace himan

#endif /* STABILITY_H */
