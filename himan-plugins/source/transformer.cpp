/**
 * @file transformer.cpp
 */

#include "transformer.h"
#include "forecast_time.h"
#include "interpolate.h"
#include "level.h"
#include "logger.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "util.h"

#include "fetcher.h"
#include "hitool.h"

using namespace std;
using namespace himan::plugin;

mutex paramMutex;

#ifdef HAVE_CUDA
namespace transformergpu
{
void Process(shared_ptr<const himan::plugin_configuration> conf, shared_ptr<himan::info<double>> myTargetInfo,
             shared_ptr<himan::info<double>> sourceInfo, double scale, double base);
}
#endif

transformer::transformer()
    : itsBase(0.0),
      itsScale(1.0),
      itsApplyLandSeaMask(false),
      itsLandSeaMaskThreshold(0.5),
      itsInterpolationMethod(kUnknownInterpolationMethod),
      itsTargetForecastType(kUnknownType),
      itsSourceForecastType(kUnknownType),
      itsRotateVectorComponents(false),
      itsDoTimeInterpolation(false),
      itsDoLevelInterpolation(false),
      itsChangeMissingTo(himan::MissingDouble()),
      itsWriteEmptyGrid(true),
      itsDecimalPrecision(kHPMissingInt)
{
	itsCudaEnabledCalculation = true;

	itsLogger = logger("transformer");
}

vector<himan::level> transformer::LevelsFromString(const string& levelType, const string& levelValuesStr) const
{
	HPLevelType theLevelType = HPStringToLevelType.at(boost::to_lower_copy(levelType));

	vector<int> levelValues = util::ExpandString(levelValuesStr);

	vector<level> levels;
	levels.reserve(levelValues.size());

	std::transform(levelValues.begin(), levelValues.end(), std::back_inserter(levels),
	               [&](int levelValue) { return level(theLevelType, static_cast<float>(levelValue), levelType); });

	return levels;
}

shared_ptr<himan::info<double>> transformer::InterpolateTime(const forecast_time& ftime, const level& lev,
                                                             const param& par, const forecast_type& ftype) const
{
	// Time interpolation only supported for hour resolution

	itsLogger.Debug("Starting time interpolation");

	// fetch previous data, max 6 hours to past

	forecast_time curtime = ftime;

	shared_ptr<info<double>> prev = nullptr, next = nullptr;

	do
	{
		curtime.ValidDateTime().Adjust(kHourResolution, -1);
		prev = Fetch(curtime, lev, par, ftype, false);
	} while (curtime.Step().Hours() >= max(0l, ftime.Step().Hours() - 6l) && prev == nullptr);

	// fetch next data, max 6 hours to future

	curtime = ftime;

	do
	{
		curtime.ValidDateTime().Adjust(kHourResolution, 1);
		next = Fetch(curtime, lev, par, ftype, false);
	} while (curtime.Step().Hours() <= ftime.Step().Hours() + 6 && next == nullptr);

	if (!prev || !next)
	{
		itsLogger.Error("Time interpolation failed: unable to find previous or next data");
		return nullptr;
	}

	auto interpolated = make_shared<info<double>>(ftype, ftime, lev, par);
	interpolated->Producer(prev->Producer());
	interpolated->Create(prev->Base(), true);

	const auto& prevdata = VEC(prev);
	const auto& nextdata = VEC(next);
	auto& interp = VEC(interpolated);

	const double X = static_cast<double>(ftime.Step().Hours());
	const double X1 = static_cast<double>(prev->Time().Step().Hours());
	const double X2 = static_cast<double>(next->Time().Step().Hours());

	for (size_t i = 0; i < prevdata.size(); i++)
	{
		interp[i] = numerical_functions::interpolation::Linear<double>(X, X1, X2, prevdata[i], nextdata[i]);
	}

	return interpolated;
}

shared_ptr<himan::info<double>> transformer::InterpolateLevel(const forecast_time& ftime, const level& lev,
                                                              const param& par, const forecast_type& ftype) const
{
	// Vertical interpolation only supported if model levels are found for producer

	itsLogger.Debug("Starting vertical interpolation");

	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(ftime);
	h->ForecastType(ftype);

	if (lev.Type() == kPressure)
	{
		h->HeightUnit(kHPa);
	}
	else if (lev.Type() != kHeight)
	{
		itsLogger.Error("Level interpolation allowed only to level types 'height (m)' and 'pressure (hPa)'");
		return nullptr;
	}

	auto data = h->VerticalValue<double>(par, lev.Value());

	auto interpolated = make_shared<info<double>>(ftype, ftime, lev, par);
	interpolated->Producer(itsConfiguration->TargetProducer());

	auto b = make_shared<base<double>>();
	b->grid = shared_ptr<grid>(itsConfiguration->BaseGrid()->Clone());
	interpolated->Create(b, false);

	interpolated->Data().Set(data);

	return interpolated;
}

void transformer::SetAdditionalParameters()
{
	string itsSourceLevelType;
	string SourceLevels;
	string targetForecastType;

	if (!itsConfiguration->GetValue("base").empty())
	{
		itsBase = stod(itsConfiguration->GetValue("base"));
	}
	else
	{
		itsLogger.Trace("base not specified, using default value 0.0");
	}

	if (!itsConfiguration->GetValue("scale").empty())
	{
		itsScale = stod(itsConfiguration->GetValue("scale"));
	}
	else
	{
		itsLogger.Trace("scale not specified, using default value 1.0");
	}

	if (itsConfiguration->Exists("rotation"))
	{
		const auto spl = util::Split(itsConfiguration->GetValue("rotation"), ",");
		for_each(spl.begin(), spl.end(), [&](const std::string& str) { itsTargetParam.emplace_back(str); });

		itsSourceParam = itsTargetParam;
		itsRotateVectorComponents = true;
	}
	else
	{
		if (!itsConfiguration->GetValue("target_param").empty())
		{
			itsTargetParam = vector<param>({param(itsConfiguration->GetValue("target_param"))});
		}
		else
		{
			throw runtime_error("Transformer_plugin: target_param not specified.");
		}
	}

	if (!itsConfiguration->GetValue("target_param_aggregation").empty())
	{
		if (!itsConfiguration->GetValue("target_param_aggregation_period").empty())
		{
			itsTargetParam[0].Aggregation(
			    {HPStringToAggregationType.at(itsConfiguration->GetValue("target_param_aggregation")),
			     time_duration(itsConfiguration->GetValue("target_param_aggregation_period"))});
		}
		else
		{
			itsTargetParam[0].Aggregation(
			    {HPStringToAggregationType.at(itsConfiguration->GetValue("target_param_aggregation"))});
		}
	}

	if (!itsConfiguration->GetValue("target_param_processing_type").empty())
	{
		itsTargetParam[0].ProcessingType(
		    {HPStringToProcessingType.at(itsConfiguration->GetValue("target_param_processing_type"))});
	}

	if (!itsConfiguration->GetValue("source_param").empty())
	{
		itsSourceParam = vector<param>({param(itsConfiguration->GetValue("source_param"))});
	}
	else
	{
		itsSourceParam = itsTargetParam;
		itsLogger.Trace("Source_param not specified, source_param set to target_param");
	}

	if (!itsConfiguration->GetValue("source_param_aggregation").empty())
	{
		itsSourceParam[0].Aggregation(
		    {HPStringToAggregationType.at(itsConfiguration->GetValue("source_param_aggregation"))});
	}

	if (!itsConfiguration->GetValue("source_param_processing_type").empty())
	{
		itsSourceParam[0].ProcessingType(
		    {HPStringToProcessingType.at(itsConfiguration->GetValue("source_param_processing_type"))});
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
		itsLogger.Trace("source_level_type not specified, set to target level type");
	}

	if (!itsConfiguration->GetValue("source_levels").empty())
	{
		SourceLevels = itsConfiguration->GetValue("source_levels");
	}
	else
	{
		itsLogger.Trace("source_levels not specified, set to target levels");
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
		itsSourceLevels = itsLevelIterator.Values();
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
			if (pos != string::npos)
			{
				const string snum = targetForecastType.substr(pos + 2);
				try
				{
					value = stoi(snum);
				}
				catch (invalid_argument& e)
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

	if (itsConfiguration->Exists("time_interpolation"))
	{
		itsDoTimeInterpolation = util::ParseBoolean(itsConfiguration->GetValue("time_interpolation"));
	}

	if (itsConfiguration->Exists("vertical_interpolation"))
	{
		itsDoLevelInterpolation = util::ParseBoolean(itsConfiguration->GetValue("vertical_interpolation"));
	}

	if (itsDoTimeInterpolation && itsDoLevelInterpolation)
	{
		itsLogger.Fatal("Cannot have both 'time_interpolation' and 'vertical_interpolation' defined");
		himan::Abort();
	}

	if (itsConfiguration->Exists("change_missing_value_to"))
	{
		try
		{
			itsChangeMissingTo = stod(itsConfiguration->GetValue("change_missing_value_to"));
		}
		catch (const invalid_argument& e)
		{
			throw runtime_error("Unable to convert " + itsConfiguration->GetValue("change_missing_value_to") +
			                    " to double");
		}
	}

	if (itsConfiguration->Exists("write_empty_grid"))
	{
		itsWriteEmptyGrid = util::ParseBoolean(itsConfiguration->GetValue("write_empty_grid"));
	}

	if (itsConfiguration->Exists("decimal_precision"))
	{
		try
		{
			itsDecimalPrecision = stoi(itsConfiguration->GetValue("decimal_precision"));
		}
		catch (const invalid_argument& e)
		{
			throw runtime_error("Unable to convert " + itsConfiguration->GetValue("decimal_precision") + " to int");
		}
	}
}

void transformer::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);
	SetAdditionalParameters();

	// Need to set this before starting Calculate, since we don't want to fetch with 'targetForecastType'.
	if (itsTargetForecastType.Type() != kUnknownType)
	{
		if (itsForecastTypeIterator.Size() > 1)
		{
			throw runtime_error("Forecast type iterator can only be set when there's only 1 source forecast type");
		}
		else
		{
			itsForecastTypeIterator.First();
			// Copy the original so that we can fetch the right data.
			itsSourceForecastType = itsForecastTypeIterator.At();
			itsForecastTypeIterator.Replace(itsTargetForecastType);
		}
	}

	if (itsInterpolationMethod != kUnknownInterpolationMethod)
	{
		for (auto& p : itsTargetParam)
		{
			p.InterpolationMethod(itsInterpolationMethod);
		}
	}

	SetParams(itsTargetParam);

	Start();
}

void transformer::Rotate(shared_ptr<info<double>> myTargetInfo)
{
	itsLogger.Trace("Rotating vector component");

	if (itsSourceParam.size() != 2)
	{
		itsLogger.Error("Two source parameters are needed for rotation");
		return;
	}

	auto a = Fetch(myTargetInfo->Time(), myTargetInfo->Level(), itsSourceParam[0], myTargetInfo->ForecastType(), false);
	auto b = Fetch(myTargetInfo->Time(), myTargetInfo->Level(), itsSourceParam[1], myTargetInfo->ForecastType(), false);

	if (!a || !b)
	{
		itsLogger.Error("Data not found");
		return;
	}

	myTargetInfo->Index<param>(0);
	myTargetInfo->Data().Set(VEC(a));
	myTargetInfo->Grid()->UVRelativeToGrid(a->Grid()->UVRelativeToGrid());

	auto secondInfo = make_shared<info<double>>(*myTargetInfo);
	secondInfo->Index<param>(1);
	secondInfo->Data().Set(VEC(b));
	secondInfo->Grid()->UVRelativeToGrid(b->Grid()->UVRelativeToGrid());

	interpolate::RotateVectorComponents(a->Grid().get(), myTargetInfo->Grid().get(), *myTargetInfo, *secondInfo, false);
}

void transformer::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
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

	shared_ptr<info<double>> sourceInfo;

	try
	{
		sourceInfo = f->Fetch(itsConfiguration, forecastTime, itsSourceLevels[myTargetInfo->Index<level>()],
		                      itsSourceParam[0], forecastType, itsConfiguration->UseCudaForPacking());
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			if (itsDoTimeInterpolation)
			{
				sourceInfo = InterpolateTime(forecastTime, itsSourceLevels[myTargetInfo->Index<level>()],
				                             itsSourceParam[0], forecastType);
			}
			else if (itsDoLevelInterpolation)
			{
				sourceInfo = InterpolateLevel(forecastTime, itsSourceLevels[myTargetInfo->Index<level>()],
				                              itsSourceParam[0], forecastType);
			}
		}
		if (!sourceInfo)
		{
			myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
			                         static_cast<string>(forecastLevel));
			return;
		}
	}

	if (itsSourceParam[0].Name() == itsTargetParam[0].Name() &&
	    (sourceInfo->Param().Aggregation().Type() != kUnknownAggregationType ||
	     sourceInfo->Param().ProcessingType().Type() != kUnknownProcessingType))
	{
		// If source parameter is an aggregation or processed somehow, copy that
		// information to target param
		param p = myTargetInfo->Param();
		p.Aggregation(sourceInfo->Param().Aggregation());
		p.ProcessingType(sourceInfo->Param().ProcessingType());

		{
			lock_guard<mutex> lock(paramMutex);
			myTargetInfo->Set<param>(p);
		}
	}

	SetAB(myTargetInfo, sourceInfo);
	myTargetInfo->Grid()->UVRelativeToGrid(sourceInfo->Grid()->UVRelativeToGrid());

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";

		transformergpu::Process(itsConfiguration, myTargetInfo, sourceInfo, itsScale, itsBase);
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

	if (!IsMissing(itsChangeMissingTo))
	{
		auto& vec = VEC(myTargetInfo);
		replace_if(
		    vec.begin(), vec.end(), [=](double d) { return IsMissing(d); }, itsChangeMissingTo);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

void transformer::WriteToFile(const shared_ptr<info<double>> targetInfo, write_options writeOptions)
{
	writeOptions.write_empty_grid = itsWriteEmptyGrid;
	writeOptions.precision = itsDecimalPrecision;

	return compiled_plugin_base::WriteToFile(targetInfo, writeOptions);
}
