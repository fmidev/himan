/**
 * @file cape.h
 *
 */

#ifndef CAPE_H
#define CAPE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

// T, Td, P
typedef std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> cape_source;

namespace himan
{
namespace plugin
{
class cape : public compiled_plugin, private compiled_plugin_base
{
   public:
	cape();

	inline virtual ~cape() {}
	cape(const cape& other) = delete;
	cape& operator=(const cape& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const { return "himan::plugin::cape"; }
	virtual HPPluginClass PluginClass() const { return kCompiled; }
	virtual HPVersionNumber Version() const { return HPVersionNumber(0, 1); }
   private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short threadIndex);

	std::pair<std::vector<double>, std::vector<double>> GetLCL(std::shared_ptr<info> myTargetInfo,
	                                                           const cape_source& source);

	std::pair<std::vector<double>, std::vector<double>> GetLFC(std::shared_ptr<info> myTargetInfo,
	                                                           std::vector<double>& T, std::vector<double>& P);
	std::pair<std::vector<double>, std::vector<double>> GetLFCCPU(std::shared_ptr<info> myTargetInfo,
	                                                              std::vector<double>& T, std::vector<double>& P,
	                                                              std::vector<double>& TenvLCL);

	// Functions to fetch different kinds of source data

	cape_source GetSurfaceValues(std::shared_ptr<info> myTargetInfo);

	cape_source Get500mMixingRatioValues(std::shared_ptr<info> myTargetInfo);
	cape_source Get500mMixingRatioValuesCPU(std::shared_ptr<info> myTargetInfo);

	cape_source GetHighestThetaEValues(std::shared_ptr<info> myTargetInfo);
	cape_source GetHighestThetaEValuesCPU(std::shared_ptr<info> myTargetInfo);

	void GetCAPE(std::shared_ptr<info> myTargetInfo, const std::pair<std::vector<double>, std::vector<double>>& LFC);
	void GetCAPECPU(std::shared_ptr<info> myTargetInfo, const std::vector<double>& T, const std::vector<double>& P);

	void GetCIN(std::shared_ptr<info> myTargetInfo, const std::vector<double>& Tsource,
	            const std::vector<double>& Psource, const std::vector<double>& TLCL, const std::vector<double>& PLCL,
	            const std::vector<double>& PLFC);
	void GetCINCPU(std::shared_ptr<info> myTargetInfo, const std::vector<double>& Tsource,
	               const std::vector<double>& Psource, const std::vector<double>& TLCL, const std::vector<double>& PLCL,
	               const std::vector<double>& PLFC);

	level itsBottomLevel;

	std::vector<level> itsSourceLevels;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create() { return std::make_shared<cape>(); }
}  // namespace plugin
}  // namespace himan

#endif /* SI */
