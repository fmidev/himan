/**
 * @file transformer.cpp
 *
 * @date May, 2014
 * @author Tack
 */

#include "transformer.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "json_parser.h"
#include "util.h"
#include "level.h"
#include "forecast_time.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#include "cuda_helper.h"

transformer::transformer() : itsBase(0.0), itsScale(1.0), itsTargetUnivID(9999), itsApplyLandSeaMask(false), itsLandSeaMaskThreshold(0.5)
{
	itsClearTextFormula = "target_param = source_param * itsScale + itsBase";
	itsCudaEnabledCalculation = true;

	itsLogger = logger_factory::Instance()->GetLog("transformer");
}

vector<himan::level> transformer::LevelsFromString(const string& levelType, const string& levelValues) const
{
	HPLevelType theLevelType = HPStringToLevelType.at(boost::to_lower_copy(levelType));

	vector<string> levelsStr = util::Split(levelValues, ",", true);

	vector<level> levels;

	for (size_t i = 0; i < levelsStr.size(); i++)
	{
		levels.push_back(level(theLevelType, boost::lexical_cast<float> (levelsStr[i]), levelType));
	}

	return levels;
}

void transformer::SetAdditionalParameters()
{
	std::string itsSourceLevelType;
	std::string SourceLevels;

	if(!itsConfiguration->GetValue("base").empty())
	{
		itsBase = boost::lexical_cast<double>(itsConfiguration->GetValue("base"));
	}
	else
	{
		itsLogger->Warning("Base not specified, using default value 0.0");
	}
	
	if(!itsConfiguration->GetValue("scale").empty())
	{
		itsScale = boost::lexical_cast<double>(itsConfiguration->GetValue("scale"));
	}
	else
	{
		itsLogger->Warning("Scale not specified, using default value 1.0");
	}

	if(!itsConfiguration->GetValue("target_univ_ID").empty())
	{
		itsTargetUnivID = boost::lexical_cast<int>(itsConfiguration->GetValue("target_univ_id"));
	}
	else
	{
		itsLogger->Warning("Target_univ_ID not specified, using default value 9999");
	}
	
	if(!itsConfiguration->GetValue("target_param").empty())
	{
		itsTargetParam = itsConfiguration->GetValue("target_param");
	}
	else
	{
		throw runtime_error("Transformer_plugin: target_param not specified.");
		exit(1);
	}

	if(!itsConfiguration->GetValue("source_param").empty())
	{
		itsSourceParam = itsConfiguration->GetValue("source_param");
	}
	else
	{
		itsSourceParam = itsTargetParam;
		itsLogger->Warning("Source_param not specified, source_param set to target_param");
	}

	if(!itsConfiguration->GetValue("source_level_type").empty())
	{
		itsSourceLevelType = itsConfiguration->GetValue("source_level_type");
	}
	else
	{
		throw runtime_error("Transformer_plugin: source_level_type not specified.");
		exit(1);
	}
	
	if(!itsConfiguration->GetValue("source_levels").empty())
	{
		SourceLevels = itsConfiguration->GetValue("source_levels");
	}
	else
	{
		throw runtime_error("Transformer_plugin: source_level_type not specified.");
		exit(1);
	}
	
	// Check apply land sea mask parameter
	
	if (itsConfiguration->Exists("apply_landsea_mask") && itsConfiguration->GetValue("apply_landsea_mask") == "true")
	{
		itsApplyLandSeaMask = true;
		
		// Check for optional threshold parameter
		if (itsConfiguration->Exists("landsea_mask_threshold"))
		{
			itsLandSeaMaskThreshold = boost::lexical_cast<double> (itsConfiguration->GetValue("landsea_mask_threshold"));
		}
	}
	
	// looks useful to use this function to create source_levels
	itsSourceLevels = LevelsFromString(itsSourceLevelType, SourceLevels);
}

void transformer::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);
	SetAdditionalParameters();

	/*
	 * Set target parameter to T
	 * - name T-C
	 * - univ_id 4
	 * - grib2 descriptor 0'00'000
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param requestedParam(itsTargetParam, itsTargetUnivID);

	// GRIB 2
	if (itsConfiguration->OutputFileType() == kGRIB2)
	{
		if (!itsConfiguration->GetValue("grib_discipline").empty() && !itsConfiguration->GetValue("grib_category").empty() && !itsConfiguration->GetValue("grib_parameter").empty())
		{
			requestedParam.GribDiscipline(boost::lexical_cast<int>(itsConfiguration->GetValue("grib_discipline")));
			requestedParam.GribCategory(boost::lexical_cast<int>(itsConfiguration->GetValue("grib_category")));
			requestedParam.GribParameter(boost::lexical_cast<int>(itsConfiguration->GetValue("grib_parameter")));
		}
		else
		{
			throw runtime_error("Transformer_plugin: Grib2 output requested but Grib2 parameter specifiers for output parameter not given in json file.");
		}
	}	

	// GRIB 1

	theParams.push_back(requestedParam);

	SetParams(theParams);

	Start();
	
}


/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void transformer::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	// Required source parameter

	param InputParam(itsSourceParam);

	auto myThreadedLogger = logger_factory::Instance()->GetLog("transformerThread #" + boost::lexical_cast<string> (threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	auto f = GET_PLUGIN(fetcher);
	
	if (itsApplyLandSeaMask)
	{
		f->ApplyLandSeaMask(true);
		f->LandSeaMaskThreshold(itsLandSeaMaskThreshold);
	}
	
	info_t sourceInfo;
	
	try
	{
		sourceInfo = f->Fetch(itsConfiguration, forecastTime, itsSourceLevels[myTargetInfo->LevelIndex()], InputParam, forecastType, itsConfiguration->UseCudaForPacking());
	}
	catch (HPExceptionType& e)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	SetAB(myTargetInfo, sourceInfo);

	bool levelOnly = (itsScale == 1.0 && itsBase == 0.0);

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda() && !levelOnly)
	{

		deviceType = "GPU";

		auto opts = CudaPrepare(myTargetInfo, sourceInfo);

		transformer_cuda::Process(*opts);

	}
	else
#endif
	{
		deviceType = "CPU";

		LOCKSTEP(myTargetInfo, sourceInfo)
		{

			double value = sourceInfo->Value();

			if (value == kFloatMissing)
			{
				continue;
			}
				
			if (!levelOnly)
			{
				double newValue = value * itsScale + itsBase;

				myTargetInfo->Value(newValue);
			}
			else
			{
				myTargetInfo->Value(value);
			}

		}
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));
}


#ifdef HAVE_CUDA

unique_ptr<transformer_cuda::options> transformer::CudaPrepare( shared_ptr<info> myTargetInfo, shared_ptr<info> sourceInfo)
{
	unique_ptr<transformer_cuda::options> opts(new transformer_cuda::options);

	opts->N = sourceInfo->Data().Size();

	opts->base = itsBase;
	opts->scale = itsScale;

	opts->source = sourceInfo->ToSimple();
	opts->dest = myTargetInfo->ToSimple();

	return opts;
}
#endif

