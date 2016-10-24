/*
 * roughness.h
 *
 */

#ifndef ROUGHNESS_H
#define ROUGHNESS_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{
/**
 * @class roughness
 *
 * @brief Calculate roughness length from HIRLAM data.
 *        Roughness length = terrain roughness length + vegetation roughness length
 *
 */

class roughness : public compiled_plugin, private compiled_plugin_base
{
public:
	roughness();

	inline virtual ~roughness() {}
	roughness(const roughness& other) = delete;
	roughness& operator=(const roughness& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::roughness"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<roughness>(new roughness()); }

}  // namespace plugin
}  // namespace himan

#endif /* ROUGHNESS_H */
