/**
 * @file cape.h
 *
 * @date Feb 13, 2014
 * @author partio
 */

#ifndef CAPE_H
#define CAPE_H

#include "compiled_plugin.h"
#include "compiled_plugin_base.h"

namespace himan
{
namespace plugin
{

enum HPSoundingIndexSourceDataType
{
	kUnknown = 0,
	kSurface,			// Use surface data (from 2m)
	k500mAvg,			// Average data of lowest 500m
	k500mAvgMixingRatio,		// Average data of lowest 500m using mixing ratio
	kMaxThetaE			// Find source data with max Theta E with upper limit 500hPa
	
};

const boost::unordered_map<HPSoundingIndexSourceDataType,std::string> HPSoundingIndexSourceDataTypeToString = ba::map_list_of
		(kUnknown, "unknown")
		(kSurface, "surface")
		(k500mAvg, "500m avg")
		(k500mAvgMixingRatio, "500m mix")
		(kMaxThetaE, "max theta e")
		;

class cape : public compiled_plugin, private compiled_plugin_base
{
public:
	cape();

	inline virtual ~cape() {}

	cape(const cape& other) = delete;
	cape& operator=(const cape& other) = delete;

	virtual void Process(std::shared_ptr<const plugin_configuration> conf);

	virtual std::string ClassName() const
	{
		return "himan::plugin::cape";
	}

	virtual HPPluginClass PluginClass() const
	{
		return kCompiled;
	}

	virtual HPVersionNumber Version() const
	{
		return HPVersionNumber(0, 1);
	}

private:
	virtual void Calculate(std::shared_ptr<info> theTargetInfo, unsigned short threadIndex);
	void CalculateVersion(std::shared_ptr<info> theTargetInfo, unsigned short threadIndex, HPSoundingIndexSourceDataType sourceType);
	void ScaleBase(std::shared_ptr<info> anInfo, double scale, double base);
	
	std::pair<std::vector<double>,std::vector<double>> GetLCL(std::shared_ptr<info> myTargetInfo, std::vector<double>& T, std::vector<double>& TD);

	std::pair<std::vector<double>,std::vector<double>> GetLFC(std::shared_ptr<info> myTargetInfo, std::vector<double>& T, std::vector<double>& P);
	std::pair<std::vector<double>,std::vector<double>> GetLFCCPU(std::shared_ptr<info> myTargetInfo, std::vector<double>& T, std::vector<double>& P, std::vector<double>& TenvLCL);
	
	// Functions to fetch different kinds of source data

	std::pair<std::vector<double>,std::vector<double>> GetSurfaceTAndTD(std::shared_ptr<info> myTargetInfo);
	std::pair<std::vector<double>,std::vector<double>> Get500mTAndTD(std::shared_ptr<info> myTargetInfo);
	
	std::pair<std::vector<double>,std::vector<double>> Get500mMixingRatioTAndTD(std::shared_ptr<info> myTargetInfo);
	std::pair<std::vector<double>,std::vector<double>> Get500mMixingRatioTAndTDCPU(std::shared_ptr<info> myTargetInfo);

	std::pair<std::vector<double>,std::vector<double>> GetHighestThetaETAndTD(std::shared_ptr<info> myTargetInfo);
	std::pair<std::vector<double>,std::vector<double>> GetHighestThetaETAndTDCPU(std::shared_ptr<info> myTargetInfo);

	void GetCAPE(std::shared_ptr<info> myTargetInfo, const std::pair<std::vector<double>, std::vector<double>>& LFC, himan::param ELTParam, himan::param ELPParam, himan::param ELZParam, himan::param CAPEParam, himan::param CAPE1040Param, himan::param CAPE3kmParam);
	void GetCAPECPU(std::shared_ptr<info> myTargetInfo, const std::vector<double>& T, const std::vector<double>& P, himan::param ELTParam, himan::param ELPParam, himan::param CAPEParam, himan::param CAPE1040Param, himan::param CAPE3kmParam);

	void GetCIN(std::shared_ptr<info> myTargetInfo, const std::vector<double>& Tsurf, const std::vector<double>& TLCL, const std::vector<double>& PLCL, const std::vector<double>& PLFC, himan::param CINParam);
	void GetCINCPU(std::shared_ptr<info> myTargetInfo, const std::vector<double>& Tsurf, const std::vector<double>& TLCL, const std::vector<double>& PLCL, const std::vector<double>& PLFC, himan::param CINParam);

	level itsBottomLevel;
	
	std::vector<HPSoundingIndexSourceDataType> itsSourceDatas;
};

// the class factory

extern "C" std::shared_ptr<himan_plugin> create()
{
	return std::make_shared<cape> ();
}

} // namespace plugin
} // namespace himan

#endif /* SI */
