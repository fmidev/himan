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

using namespace std;
using namespace himan::plugin;

#include "cuda_helper.h"

transformer::transformer() : itsBase(0.0), itsScale(1.0), itsTargetUnivID(9999)
{
	itsClearTextFormula = "target_param = source_param * itsScale + itsBase";
	itsCudaEnabledCalculation = true;

	itsLogger = logger_factory::Instance()->GetLog("transformer");
}

vector<himan::level> transformer::LevelsFromString(const string& levelType, const string& levelValues) const
{
	HPLevelType theLevelType = HPStringToLevelType.at(boost::to_lower_copy(levelType));

	// can cause exception, what will happen then ?

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

	if(itsConfiguration->Options().count("base"))
	{
		itsBase = boost::lexical_cast<double>(itsConfiguration->GetValue("base"));
	}
	else
	{
		itsLogger->Warning("Base not specified, using default value 0.0");
	}
	
	if(itsConfiguration->Options().count("scale"))
	{
		itsScale = boost::lexical_cast<double>(itsConfiguration->GetValue("scale"));
	}
	else
	{
		itsLogger->Warning("Scale not specified, using default value 1.0");
	}

	if(itsConfiguration->Options().count("target_univ_ID"))
	{
		itsTargetUnivID = boost::lexical_cast<int>(itsConfiguration->GetValue("target_univ_id"));
	}
	else
	{
		itsLogger->Warning("Target_univ_ID not specified, using default value 9999");
	}
	
	if(itsConfiguration->Options().count("target_param"))
	{
		itsTargetParam = itsConfiguration->GetValue("target_param");
	}
	else
	{
		throw runtime_error("Transformer_plugin: target_param not specified.");
		exit(1);
	}

	if(itsConfiguration->Options().count("source_param"))
	{
		itsSourceParam = itsConfiguration->GetValue("source_param");
	}
	else
	{
		itsSourceParam = itsTargetParam;
		itsLogger->Warning("Source_param not specified, source_param set to target_param");
	}

	if(itsConfiguration->Options().count("source_level_type"))
	{
		itsSourceLevelType = itsConfiguration->GetValue("source_level_type");
	}
	else
	{
		throw runtime_error("Transformer_plugin: source_level_type not specified.");
		exit(1);
	}
	
	if(itsConfiguration->Options().count("source_levels"))
	{
		SourceLevels = itsConfiguration->GetValue("source_levels");
	}
	else
	{
		throw runtime_error("Transformer_plugin: source_level_type not specified.");
		exit(1);
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
		if (itsConfiguration->Options().count("grib_discipline") && itsConfiguration->Options().count("grib_category") && itsConfiguration->Options().count("grib_parameter"))
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

	myThreadedLogger->Info("Calculating time " + static_cast<string>(*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(threadIndex);

	info_t sourceInfo = Fetch(forecastTime, itsSourceLevels[myTargetInfo->LevelIndex()], InputParam, itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

	if (!sourceInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	SetAB(myTargetInfo, sourceInfo);

	bool levelOnly = (itsScale == 1.0 && itsBase == 0.0);

	string deviceType;

#ifdef HAVE_CUDA

	if (useCudaInThisThread && !levelOnly)
	{

		deviceType = "GPU";

		auto opts = CudaPrepare(myTargetInfo, sourceInfo);

		transformer_cuda::Process(*opts);

		CudaFinish(move(opts), myTargetInfo, sourceInfo);

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

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data()->MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data()->Size()));
}


#ifdef HAVE_CUDA

unique_ptr<transformer_cuda::options> transformer::CudaPrepare( shared_ptr<info> myTargetInfo, shared_ptr<info> sourceInfo)
{
	unique_ptr<transformer_cuda::options> opts(new transformer_cuda::options);

	opts->N = sourceInfo->Data()->Size();

	opts->base = itsBase;
	opts->scale = itsScale;

	opts->source = sourceInfo->ToSimple();
	opts->dest = myTargetInfo->ToSimple();

	return opts;
}

void transformer::CudaFinish(unique_ptr<transformer_cuda::options> opts, shared_ptr<info> myTargetInfo, shared_ptr<info> sourceInfo)
{
	// Copy data back to infos

	CopyDataFromSimpleInfo(myTargetInfo, opts->dest, false);

	if (sourceInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(sourceInfo, opts->source, true);
	}

}

#endif

