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

	inline virtual ~stability() {}
	stability(const stability& other) = delete;
	stability& operator=(const stability& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::stability"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(2, 0); }
   private:
	void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short theThreadIndex);

	bool GetSourceData(std::shared_ptr<info>& T850Info, std::shared_ptr<info>& T700Info,
	                   std::shared_ptr<info>& T500Info, std::shared_ptr<info>& TD850Info,
	                   std::shared_ptr<info>& TD700Info, const std::shared_ptr<info>& myTargetInfo,
	                   bool useCudaInThisThread);
	bool GetLISourceData(const std::shared_ptr<info>& myTargetInfo, std::vector<double>& T500mVector,
	                     std::vector<double>& TD500mVector, std::vector<double>& P500mVector);
	bool GetWindShearSourceData(const std::shared_ptr<info>& myTargetInfo, std::vector<double>& U01Vector,
	                            std::vector<double>& V01Vector, std::vector<double>& U06Vector,
	                            std::vector<double>& V06Vector);
	bool GetSRHSourceData(const std::shared_ptr<info>& myTargetInfo, std::vector<double>& Uid,
	                      std::vector<double>& Vid);

	bool itsLICalculation;
	bool itsBSCalculation;
	bool itsSRHCalculation;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::shared_ptr<stability>(new stability()); }
}  // namespace plugin
}  // namespace himan

#endif /* STABILITY_H */
