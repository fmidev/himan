/**
 * @file transformer.cpp
 */

#include "transformer.h"
#include "forecast_time.h"
#include "interpolate.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"

#include "fetcher.h"

using namespace std;
using namespace himan::plugin;

mutex aggregationMutex;

#include "cuda_helper.h"

transformer::transformer()
    : itsBase(0.0),
      itsScale(1.0),
      itsApplyLandSeaMask(false),
      itsLandSeaMaskThreshold(0.5),
      itsInterpolationMethod(kUnknownInterpolationMethod),
      itsTargetForecastType(kUnknownType),
      itsSourceForecastType(kUnknownType),
      itsRotateVectorComponents(false)
{
	itsCudaEnabledCalculation = true;

	itsLogger = logger("transformer");
}

vector<himan::level> transformer::LevelsFromString(const string& levelType, const string& levelValues) const
{
	HPLevelType theLevelType = HPStringToLevelType.at(boost::to_lower_copy(levelType));

	vector<string> levelsStr = util::Split(levelValues, ",", true);

	vector<level> levels;

	for (size_t i = 0; i < levelsStr.size(); i++)
	{
		levels.push_back(level(theLevelType, stof(levelsStr[i]), levelType));
	}

	return levels;
}

void transformer::SetAdditionalParameters()
{
	std::string itsSourceLevelType;
	std::string SourceLevels;
	std::string targetForecastType;

	if (!itsConfiguration->GetValue("base").empty())
	{
		itsBase = stod(itsConfiguration->GetValue("base"));
	}
	else
	{
		itsLogger.Trace("Base not specified, using default value 0.0");
	}

	if (!itsConfiguration->GetValue("scale").empty())
	{
		itsScale = stod(itsConfiguration->GetValue("scale"));
	}
	else
	{
		itsLogger.Trace("Scale not specified, using default value 1.0");
	}

	if (itsConfiguration->Exists("rotation"))
	{
		itsTargetParam = util::Split(itsConfiguration->GetValue("rotation"), ",", false);
		itsSourceParam = itsTargetParam;
		itsRotateVectorComponents = true;
	}
	else
	{
		if (!itsConfiguration->GetValue("target_param").empty())
		{
			itsTargetParam = vector<string>({itsConfiguration->GetValue("target_param")});
		}
		else
		{
			throw runtime_error("Transformer_plugin: target_param not specified.");
		}
	}

	if (!itsConfiguration->GetValue("source_param").empty())
	{
		itsSourceParam = vector<string>({itsConfiguration->GetValue("source_param")});
	}
	else
	{
		itsSourceParam = itsTargetParam;
		itsLogger.Trace("Source_param not specified, source_param set to target_param");
	}

	if (itsSourceParam.size() != itsTargetParam.size())
	{
		itsLogger.Fatal("Number source params does not match target params");
		himan::Abort();
	}

	if (!itsConfiguration->GetValue("source_level_type").empty())
	{
		itsSourceLevelType = itsConfiguration->GetValue("source_level_type");
	}
	else
	{
		itsLogger.Trace("Source_level not specified, source_level set to target level");
	}

	if (!itsConfiguration->GetValue("source_levels").empty())
	{
		SourceLevels = itsConfiguration->GetValue("source_levels");
	}
	else
	{
		itsLogger.Trace("Source_levels not specified, source_levels set to target levels");
	}

	if (!itsConfiguration->GetValue("target_forecast_type").empty())
	{
		targetForecastType = itsConfiguration->GetValue("target_forecast_type");
	}
	else
	{
		itsLogger.Trace("Target_forecast_type not specified, target_forecast_type set to source forecast type");
	}

	// Check apply land sea mask parameter

	if (itsConfiguration->Exists("apply_landsea_mask") && itsConfiguration->GetValue("apply_landsea_mask") == "true")
	{
		itsApplyLandSeaMask = true;

		// Check for optional threshold parameter
		if (itsConfiguration->Exists("landsea_mask_threshold"))
		{
			itsLandSeaMaskThreshold = stod(itsConfiguration->GetValue("landsea_mask_threshold"));
		}
	}

	if (itsConfiguration->Exists("interpolation"))
	{
		itsInterpolationMethod = HPStringToInterpolationMethod.at(itsConfiguration->GetValue("interpolation"));
	}

	if (!SourceLevels.empty())
	{
		// looks useful to use this function to create source_levels

		itsSourceLevels = LevelsFromString(itsSourceLevelType, SourceLevels);
	}
	else
	{
		// copy levels from target
		auto x = make_shared<info>(*itsInfo);
		for (x->ResetLevel(); x->NextLevel();)
		{
			itsSourceLevels.push_back(x->Level());
		}
	}

	if (!targetForecastType.empty())
	{
		if (targetForecastType == "cf")
		{
			itsTargetForecastType = forecast_type(kEpsControl);
		}
		else if (targetForecastType == "deterministic")
		{
			itsTargetForecastType = forecast_type(kDeterministic);
		}
		else if (targetForecastType == "analysis")
		{
			itsTargetForecastType = forecast_type(kAnalysis);
		}
		else
		{
			// should be 'pfNN'
			auto pos = targetForecastType.find("pf");
			int value = 0;
			if (pos != std::string::npos)
			{
				const string snum = targetForecastType.substr(pos + 2);
				try
				{
					value = std::stoi(snum);
				}
				catch (std::invalid_argument& e)
				{
					throw runtime_error("Transformer_plugin: failed to convert perturbation forecast number");
				}
			}
			else
			{
				throw runtime_error("Transformer_plugin: invalid forecast type specified");
			}
			itsTargetForecastType = forecast_type(kEpsPerturbation, value);
		}
	}
}

void transformer::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);
	SetAdditionalParameters();

	// Need to set this before starting Calculate, since we don't want to fetch with 'targetForecastType'.
	if (itsTargetForecastType.Type() != kUnknownType)
	{
		if (itsInfo->ForecastTypeIterator().Size() > 1)
		{
			throw std::runtime_error("Forecast type iterator can only be set when there's only 1 source forecast type");
		}
		else
		{
			itsInfo->ForecastTypeIterator().First();
			// Copy the original so that we can fetch the right data.
			itsSourceForecastType = itsInfo->ForecastType();
			itsInfo->ForecastTypeIterator().Replace(itsTargetForecastType);
		}
	}

	vector<param> theParams;

	for (const auto& name : itsTargetParam)
	{
		theParams.push_back(param(name));
	}

	if (itsInterpolationMethod != kUnknownInterpolationMethod)
	{
		for (auto& p : theParams)
		{
			p.InterpolationMethod(itsInterpolationMethod);
		}
	}

	SetParams(theParams);

	Start();
}

void transformer::Rotate(info_t myTargetInfo)
{
	itsLogger.Trace("Rotating vector component");

	if (itsSourceParam.size() != 2)
	{
		itsLogger.Error("Two source parameters are needed for rotation");
		return;
	}

	auto a = Fetch(myTargetInfo->Time(), myTargetInfo->Level(), param(itsSourceParam[0]), myTargetInfo->ForecastType(),
	               itsConfiguration->UseCudaForPacking());
	auto b = Fetch(myTargetInfo->Time(), myTargetInfo->Level(), param(itsSourceParam[1]), myTargetInfo->ForecastType(),
	               itsConfiguration->UseCudaForPacking());

	myTargetInfo->ParamIndex(0);
	myTargetInfo->Data().Set(VEC(a));
	myTargetInfo->Grid()->UVRelativeToGrid(a->Grid()->UVRelativeToGrid());

	auto secondInfo = make_shared<info>(*myTargetInfo);
	secondInfo->ParamIndex(1);
	secondInfo->Data().Set(VEC(b));
	secondInfo->Grid()->UVRelativeToGrid(b->Grid()->UVRelativeToGrid());

	interpolate::RotateVectorComponents(*myTargetInfo, *secondInfo, itsConfiguration->UseCudaForInterpolation());
}

void transformer::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("transformerThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	forecast_type forecastType;
	if (itsSourceForecastType.Type() != kUnknownType)
	{
		forecastType = itsSourceForecastType;
	}
	else
	{
		forecastType = myTargetInfo->ForecastType();
	}

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	if (itsRotateVectorComponents)
	{
		Rotate(myTargetInfo);
		return;
	}

	auto f = GET_PLUGIN(fetcher);

	if (itsApplyLandSeaMask)
	{
		f->ApplyLandSeaMask(true);
		f->LandSeaMaskThreshold(itsLandSeaMaskThreshold);
	}

	info_t sourceInfo;

	try
	{
		sourceInfo = f->Fetch(itsConfiguration, forecastTime, itsSourceLevels[myTargetInfo->LevelIndex()],
		                      param(itsSourceParam[0]), forecastType, itsConfiguration->UseCudaForPacking());
	}
	catch (HPExceptionType& e)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	if (itsSourceParam[0] == itsTargetParam[0] && sourceInfo->Param().Aggregation().Type() != kUnknownAggregationType)
	{
		// If source parameter is an aggregation, copy that to target param
		param p = myTargetInfo->Param();
		aggregation a = sourceInfo->Param().Aggregation();
		p.Aggregation(a);

		{
			lock_guard<mutex> lock(aggregationMutex);
			myTargetInfo->ParamIterator().Replace(p);
		}
	}

	SetAB(myTargetInfo, sourceInfo);
	myTargetInfo->Grid()->UVRelativeToGrid(sourceInfo->Grid()->UVRelativeToGrid());

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";

		auto opts = CudaPrepare(myTargetInfo, sourceInfo);

		transformer_cuda::Process(*opts);
	}
	else
#endif
	{
		deviceType = "CPU";

		auto& result = VEC(myTargetInfo);
		const auto& source = VEC(sourceInfo);

		transform(source.begin(), source.end(), result.begin(),
		          [&](const double& value) { return fma(value, itsScale, itsBase); });
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

#ifdef HAVE_CUDA

unique_ptr<transformer_cuda::options> transformer::CudaPrepare(shared_ptr<info> myTargetInfo,
                                                               shared_ptr<info> sourceInfo)
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
