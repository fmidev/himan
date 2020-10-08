/**
 * @file cape.h
 *
 */

#ifndef CAPE_H
#define CAPE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

// T, Td, P
typedef std::tuple<std::vector<float>, std::vector<float>, std::vector<float>> cape_source;
typedef std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>, std::vector<std::vector<float>>>
    cape_multi_source;
typedef std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>,
                   std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>>
    CAPEdata;

namespace himan
{
namespace plugin
{
class cape : public compiled_plugin, private compiled_plugin_base
{
   public:
	cape();

	inline virtual ~cape() = default;
	cape(const cape& other) = delete;
	cape& operator=(const cape& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf) override;

	virtual std::string ClassName() const override
	{
		return "himan::plugin::cape";
	}
	virtual HPPluginClass PluginClass() const override
	{
		return kCompiled;
	}

   private:
	virtual void Calculate(std::shared_ptr<info<float>> theTargetInfo, unsigned short threadIndex) override;

	std::pair<std::vector<float>, std::vector<float>> GetLCL(const cape_source& source) const;

	std::vector<std::pair<std::vector<float>, std::vector<float>>> GetLFC(std::shared_ptr<info<float>> myTargetInfo,
	                                                                      std::vector<float>& T,
	                                                                      std::vector<float>& P) const;
	std::vector<std::pair<std::vector<float>, std::vector<float>>> GetLFCCPU(std::shared_ptr<info<float>> myTargetInfo,
	                                                                         std::vector<float>& T,
	                                                                         std::vector<float>& P,
	                                                                         std::vector<float>& TenvLCL) const;

	// Functions to fetch different kinds of source data

	cape_source GetSurfaceValues(std::shared_ptr<info<float>> myTargetInfo);

	cape_source Get500mMixingRatioValues(std::shared_ptr<info<float>> myTargetInfo);
	cape_source Get500mMixingRatioValuesCPU(std::shared_ptr<info<float>> myTargetInfo);

	cape_multi_source GetNHighestThetaEValues(std::shared_ptr<info<float>> myTargetInfo, int N) const;
	cape_multi_source GetNHighestThetaEValuesCPU(std::shared_ptr<info<float>> myTargetInfo, int N) const;

	CAPEdata GetCAPE(std::shared_ptr<info<float>> myTargetInfo,
	                 const std::pair<std::vector<float>, std::vector<float>>& LFC) const;
	CAPEdata GetCAPECPU(std::shared_ptr<info<float>> myTargetInfo, const std::vector<float>& T,
	                    const std::vector<float>& P) const;

	std::vector<float> GetCIN(std::shared_ptr<info<float>> myTargetInfo, const std::vector<float>& Tsource,
	                          const std::vector<float>& Psource, const std::vector<float>& PLCL,
	                          const std::vector<float>& PLFC, const std::vector<float>& ZLFC) const;

	std::vector<float> GetCINCPU(std::shared_ptr<info<float>> myTargetInfo, const std::vector<float>& Tsource,
	                             const std::vector<float>& Psource, const std::vector<float>& PLCL,
	                             const std::vector<float>& PLFC, const std::vector<float>& ZLFC) const;

	void MostUnstableCAPE(std::shared_ptr<info<float>> myTargetInfo, short threadIndex) const;
	level itsBottomLevel;
	bool itsUseVirtualTemperature;

	std::vector<level> itsSourceLevels;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<cape>();
}
}  // namespace plugin
}  // namespace himan

#endif /* SI */
