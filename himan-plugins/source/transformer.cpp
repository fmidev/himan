/**
 * @file transformer.cpp
 */

#include "transformer.h"
#include "fetcher.h"
#include "forecast_time.h"
#include "hitool.h"
#include "interpolate.h"
#include "level.h"
#include "logger.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "querydata.h"
#include "util.h"
#include <fmt/ranges.h>

using namespace std;
using namespace himan::plugin;

mutex paramMutex;

#ifdef HAVE_CUDA
namespace transformergpu
{
template <typename T>
void Process(shared_ptr<const himan::plugin_configuration> conf, shared_ptr<himan::info<T>> myTargetInfo,
             shared_ptr<himan::info<T>> sourceInfo, double scale, double base, T min, T max);
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
      itsChangeMissingTo(""),
      itsWriteEmptyGrid(true),
      itsDecimalPrecision(kHPMissingInt),
      itsParamDefinitionFromConfig(false),
      itsEnsemble(nullptr),
      itsSourceForecastPeriod(),
      itsReadFromPreviousForecastIfNotFound(false),
      itsMinimumValue(himan::MissingDouble()),
      itsMaximumValue(himan::MissingDouble()),
      itsAllowAnySourceForecastType(false)
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

template <typename T>
void SetParamAggregation(const himan::param& src, shared_ptr<himan::info<T>> myTargetInfo)
{
	if (src.Name() == myTargetInfo->Param().Name() && (src.Aggregation().Type() != himan::kUnknownAggregationType ||
	                                                   src.ProcessingType().Type() != himan::kUnknownProcessingType))
	{
		// If source parameter is an aggregation or processed somehow, copy that
		// information to target param
		himan::param p = myTargetInfo->Param();
		p.Aggregation(src.Aggregation());
		p.ProcessingType(src.ProcessingType());

		{
			lock_guard<mutex> lock(paramMutex);
			myTargetInfo->template Set<himan::param>(p);
		}
	}
}

namespace
{
himan::forecast_time GetSourceTime(const himan::forecast_time& targetTime, const himan::time_duration& forecastPeriod)
{
	auto sourceTime = targetTime;

	if (forecastPeriod.Empty() == false)
	{
		sourceTime = himan::forecast_time(targetTime.OriginDateTime(), targetTime.OriginDateTime() + forecastPeriod);
	}

	return sourceTime;
}

}  // namespace

shared_ptr<himan::info<double>> transformer::InterpolateTime(const forecast_time& ftime, const level& lev,
                                                             const param& par, const forecast_type& ftype) const
{
	auto f = GET_PLUGIN(fetcher);
	return f->Fetch(itsConfiguration, ftime, lev, par, ftype, false, false, false, true);
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
	string SourceLevelType;
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

	if (itsConfiguration->Exists("minimum_value"))
	{
		itsMinimumValue = stod(itsConfiguration->GetValue("minimum_value"));
	}

	if (itsConfiguration->Exists("maximum_value"))
	{
		itsMaximumValue = stod(itsConfiguration->GetValue("maximum_value"));
	}

	if ((IsMissing(itsMinimumValue) && IsValid(itsMaximumValue)) ||
	    (IsValid(itsMinimumValue) && IsMissing(itsMaximumValue)))
	{
		itsLogger.Fatal("Both minimum_value and maximum_value must be specified");
		himan::Abort();
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
		itsSourceParam[0].Aggregation(aggregation(itsConfiguration->GetValue("source_param_aggregation")));
	}

	if (!itsConfiguration->GetValue("source_param_processing_type").empty())
	{
		itsSourceParam[0].ProcessingType(processing_type(itsConfiguration->GetValue("source_param_processing_type")));
	}

	if (itsSourceParam.size() != itsTargetParam.size())
	{
		itsLogger.Fatal("Number source params does not match target params");
		himan::Abort();
	}

	if (!itsConfiguration->GetValue("source_level_type").empty())
	{
		SourceLevelType = itsConfiguration->GetValue("source_level_type");
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

	if (!itsConfiguration->GetValue("source_forecast_period").empty())
	{
		itsSourceForecastPeriod = time_duration(itsConfiguration->GetValue("source_forecast_period"));
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
		for (auto& p : itsSourceParam)
		{
			p.InterpolationMethod(itsInterpolationMethod);
		}
	}

	if ((SourceLevels.empty() && !SourceLevelType.empty()) || (!SourceLevels.empty() && SourceLevelType.empty()))
	{
		itsLogger.Warning("'source_levels' and 'source_level_type' are usually both defined or neither is defined");
	}

	if (!SourceLevels.empty())
	{
		// looks useful to use this function to create source_levels

		itsSourceLevels = LevelsFromString(SourceLevelType, SourceLevels);
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
		else if (targetForecastType == "deterministic" || targetForecastType == "det")
		{
			itsTargetForecastType = forecast_type(kDeterministic);
		}
		else if (targetForecastType == "analysis" || targetForecastType == "an")
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
	if (itsDoTimeInterpolation && itsSourceForecastPeriod.Empty() == false)
	{
		itsLogger.Fatal("Cannot have both 'time_interpolation' and 'source_forecast_period' defined");
		himan::Abort();
	}

	if (itsConfiguration->Exists("change_missing_value_to"))
	{
		try
		{
			itsChangeMissingTo = itsConfiguration->GetValue("change_missing_value_to");
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
			throw runtime_error(
			    fmt::format("Unable to convert {} to int", itsConfiguration->GetValue("decimal_precision")));
		}
	}

	const auto grib1_tbl = itsConfiguration->GetValue("grib1_table_number");
	const auto grib1_num = itsConfiguration->GetValue("grib1_parameter_number");

	if (!grib1_tbl.empty() && !grib1_num.empty())
	{
		itsTargetParam[0].GribTableVersion(stoi(grib1_tbl));
		itsTargetParam[0].GribIndicatorOfParameter(stoi(grib1_num));

		itsParamDefinitionFromConfig = true;

		if (itsConfiguration->OutputFileType() != kGRIB1)
		{
			itsLogger.Warning("grib1 metadata set but output file type is not grib1");
		}
	}

	const auto grib2_dis = itsConfiguration->GetValue("grib2_discipline");
	const auto grib2_cat = itsConfiguration->GetValue("grib2_parameter_category");
	const auto grib2_num = itsConfiguration->GetValue("grib2_parameter_number");

	if (!grib2_dis.empty() && !grib2_cat.empty() && !grib2_num.empty())
	{
		itsTargetParam[0].GribCategory(stoi(grib2_cat));
		itsTargetParam[0].GribParameter(stoi(grib2_num));
		itsTargetParam[0].GribDiscipline(stoi(grib2_dis));

		itsParamDefinitionFromConfig = true;

		if (itsConfiguration->OutputFileType() != kGRIB2)
		{
			itsLogger.Warning("grib2 metadata set but output file type is not grib2");
		}
	}

	if (itsConfiguration->Exists("univ_id"))
	{
		itsTargetParam[0].UnivId(stoi(itsConfiguration->GetValue("univ_id")));
		itsParamDefinitionFromConfig = true;

		if (itsConfiguration->OutputFileType() != kQueryData)
		{
			itsLogger.Warning("querydata metadata set but output file type is not querydata");
		}
	}

	if (itsConfiguration->Exists("ensemble_type") || itsConfiguration->Exists("named_ensemble"))
	{
		itsEnsemble = util::CreateEnsembleFromConfiguration(itsConfiguration);
		itsEnsemble->Param(itsSourceParam[0]);
	}

	if (itsEnsemble &&
	    (itsDoTimeInterpolation || itsDoLevelInterpolation || itsRotateVectorComponents || itsApplyLandSeaMask))
	{
		itsLogger.Fatal(
		    "Conflicting options: ensemble and (time/level interpolation, vector component rotation, land "
		    "sea masking)");
		himan::Abort();
	}
	if (itsConfiguration->Exists("read_previous_forecast_if_not_found"))
	{
		itsReadFromPreviousForecastIfNotFound =
		    util::ParseBoolean(itsConfiguration->GetValue("read_previous_forecast_if_not_found"));
	}
	if (itsDoTimeInterpolation && itsReadFromPreviousForecastIfNotFound)
	{
		itsLogger.Fatal("Cannot have both 'time_interpolation' and 'read_previous_forecast_if_not_found' defined");
		himan::Abort();
	}
	if (itsConfiguration->Exists("allow_any_source_forecast_type"))
	{
		itsAllowAnySourceForecastType =
		    util::ParseBoolean(itsConfiguration->GetValue("allow_any_source_forecast_type"));
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

			itsLogger.Info(fmt::format("Notice: overriding forecast_type from configuration file ({}) with {}",
			                           static_cast<string>(itsSourceForecastType),
			                           static_cast<string>(itsTargetForecastType)));
		}
	}

	if (itsEnsemble != nullptr)
	{
		auto ensembleConfiguration = itsEnsemble->DesiredForecasts();
		itsForecastTypeIterator.First();
		itsSourceForecastType = itsForecastTypeIterator.At();

		if (ensembleConfiguration.size() == 0)
		{
			itsLogger.Fatal("Ensemble with zero members");
			himan::Abort();
		}
		itsForecastTypeIterator = forecast_type_iter(ensembleConfiguration);
		itsThreadDistribution = ThreadDistribution::kThreadForTimeAndLevel;

		itsLogger.Info(fmt::format("Notice: overriding forecast_type from configuration file ({}) with {}",
		                           static_cast<string>(itsSourceForecastType), fmt::join(ensembleConfiguration, ", ")));
	}

	if (itsInterpolationMethod != kUnknownInterpolationMethod)
	{
		for (auto& p : itsTargetParam)
		{
			p.InterpolationMethod(itsInterpolationMethod);
		}
	}

	SetParams(itsTargetParam, itsParamDefinitionFromConfig);

	if (itsEnsemble)
	{
		Start<float>();
	}
	else
	{
		Start();
	}
}

void transformer::Rotate(shared_ptr<info<double>> myTargetInfo)
{
	itsLogger.Trace("Rotating vector component");

	if (itsSourceParam.size() != 2)
	{
		itsLogger.Error("Two source parameters are needed for rotation");
		return;
	}

	const auto sourceTime = GetSourceTime(myTargetInfo->Time(), itsSourceForecastPeriod);

	auto a = Fetch(sourceTime, myTargetInfo->Level(), itsSourceParam[0], myTargetInfo->ForecastType(), false);
	auto b = Fetch(sourceTime, myTargetInfo->Level(), itsSourceParam[1], myTargetInfo->ForecastType(), false);

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

	auto target = std::unique_ptr<grid>(myTargetInfo->Grid()->Clone());
	target->UVRelativeToGrid(false);

	interpolate::RotateVectorComponents(a->Grid().get(), target.get(), *myTargetInfo, *secondInfo,
	                                    itsConfiguration->UseCuda());
}

void transformer::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("transformerThread #" + to_string(threadIndex));

	if (!itsEnsemble)
	{
		itsLogger.Error("float mode calculation started without ensemble configuration");
		return;
	}

	unique_ptr<ensemble> myEnsemble;

	switch (itsEnsemble->EnsembleType())
	{
		case kPerturbedEnsemble:
			myEnsemble = make_unique<ensemble>(*itsEnsemble);
			break;
		case kLaggedEnsemble:
			myEnsemble = make_unique<lagged_ensemble>(dynamic_cast<lagged_ensemble&>(*itsEnsemble));
			break;
		default:
			myThreadedLogger.Error("Don't know how to handle this ensemble type");
			return;
	}

	try
	{
		myEnsemble->Fetch(itsConfiguration, myTargetInfo->Time(), myTargetInfo->Level());
	}
	catch (const std::exception& e)
	{
		myThreadedLogger.Error(e.what());
		return;
	}

	string deviceType = "CPU";

	const auto desired = myEnsemble->DesiredForecasts();

	for (size_t i = 0; i < desired.size(); i++)
	{
		shared_ptr<info<float>> sourceInfo = nullptr;

		try
		{
			sourceInfo = myEnsemble->Forecast(i);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				myThreadedLogger.Warning(fmt::format("Skipping forecast type {}", desired[i]));
				continue;
			}
		}

		myTargetInfo->Find<forecast_type>(sourceInfo->ForecastType());

		// No SetAB since template types differ (double vs float)
		if (myTargetInfo->Level().Type() == kHybrid)
		{
			const size_t paramIndex = myTargetInfo->Index<param>();

			for (myTargetInfo->Reset<param>(); myTargetInfo->Next<param>();)
			{
				myTargetInfo->Level().AB(sourceInfo->Level().AB());
			}

			myTargetInfo->Index<param>(paramIndex);
		}

		myTargetInfo->Grid()->UVRelativeToGrid(sourceInfo->Grid()->UVRelativeToGrid());
		SetParamAggregation(sourceInfo->Param(), myTargetInfo);

		auto& result = VEC(myTargetInfo);
		const auto& source = VEC(sourceInfo);
#ifdef HAVE_CUDA

		if (itsConfiguration->UseCuda())
		{
			deviceType = "GPU";

			float min = IsValid(itsMinimumValue) ? static_cast<float>(itsMinimumValue) : MissingValue<float>();
			float max = IsValid(itsMaximumValue) ? static_cast<float>(itsMaximumValue) : MissingValue<float>();

			transformergpu::Process<float>(itsConfiguration, myTargetInfo, sourceInfo, itsScale, itsBase, min, max);
		}
		else
#endif
		{
			transform(source.begin(), source.end(), result.begin(),
			          [&](const float& value) { return fma(value, itsScale, itsBase); });

			if (IsValid(itsMinimumValue))
			{
				float min = IsValid(itsMinimumValue) ? static_cast<float>(itsMinimumValue) : MissingValue<float>();
				float max = IsValid(itsMaximumValue) ? static_cast<float>(itsMaximumValue) : MissingValue<float>();

				for_each(result.begin(), result.end(),
				         [&](float& value)
				         {
					         if (IsValid(value))
					         {
						         value = fmin(fmax(value, min), max);
					         }
				         });
			}
		}
		if (itsChangeMissingTo.empty() == false)
		{
			float newMissing = MissingValue<float>();
			if (itsChangeMissingTo == "mean")
			{
				newMissing = numerical_functions::Mean(util::RemoveMissingValues(VEC(myTargetInfo)));
			}
			else if (itsChangeMissingTo == "median")
			{
				newMissing = numerical_functions::Median(util::RemoveMissingValues(VEC(myTargetInfo)));
			}
			else
			{
				newMissing = stof(itsChangeMissingTo);
			}

			auto& vec = VEC(myTargetInfo);
			replace_if(
			    vec.begin(), vec.end(), [=](float d) { return IsMissing(d); }, newMissing);
		}

		myThreadedLogger.Info(fmt::format("[{}] Missing values: {}/{}", deviceType, myTargetInfo->Data().MissingCount(),
		                                  myTargetInfo->Data().Size()));
	}
}

shared_ptr<himan::info<double>> transformer::FetchSource(shared_ptr<himan::info<double>>& myTargetInfo,
                                                         forecast_time sourceTime, forecast_type forecastType)
{
	const vector<forecast_type> forecastTypes =
	    itsAllowAnySourceForecastType ? vector<forecast_type>{forecast_type(kDeterministic),
	                                                          forecast_type(kEpsControl, 0), forecast_type(kAnalysis)}
	                                  : vector<forecast_type>{forecastType};

	auto f = GET_PLUGIN(fetcher);

	if (itsApplyLandSeaMask)
	{
		f->ApplyLandSeaMask(true);
		f->LandSeaMaskThreshold(itsLandSeaMaskThreshold);
	}

	for (const auto& ftype : forecastTypes)
	{
		try
		{
			if (itsAllowAnySourceForecastType)
			{
				itsLogger.Trace("Any source forecast_type allowed: trying to fetch source data with forecast type " +
				                static_cast<string>(ftype));
			}
			return f->Fetch(itsConfiguration, sourceTime, itsSourceLevels[myTargetInfo->Index<level>()],
			                itsSourceParam[0], ftype, itsConfiguration->UseCudaForPacking(), false,
			                itsReadFromPreviousForecastIfNotFound);
		}
		catch (HPExceptionType& e)
		{
			if (e != kFileDataNotFound)
			{
				throw e;
			}
		}
	}

	return nullptr;
}

void transformer::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("transformerThread #" + to_string(threadIndex));

	if (itsEnsemble)
	{
		itsLogger.Error("double mode calculation started with ensemble configuration");
		return;
	}

	const forecast_time forecastTime = myTargetInfo->Time();
	const auto sourceTime = GetSourceTime(myTargetInfo->Time(), itsSourceForecastPeriod);

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

	myThreadedLogger.Info(fmt::format("Calculating time {} level {}", forecastTime.ValidDateTime(), forecastLevel));

	if (itsRotateVectorComponents)
	{
		Rotate(myTargetInfo);
		myThreadedLogger.Info(fmt::format("[{}] Missing values: {}/{}", itsConfiguration->UseCuda() ? "GPU" : "CPU",
		                                  myTargetInfo->Data().MissingCount(), myTargetInfo->Data().Size()));
		return;
	}

	shared_ptr<info<double>> sourceInfo = FetchSource(myTargetInfo, sourceTime, forecastType);

	if (!sourceInfo)
	{
		if (itsDoTimeInterpolation)
		{
			sourceInfo = InterpolateTime(sourceTime, itsSourceLevels[myTargetInfo->Index<level>()], itsSourceParam[0],
			                             forecastType);
		}
		else if (itsDoLevelInterpolation)
		{
			sourceInfo = InterpolateLevel(sourceTime, itsSourceLevels[myTargetInfo->Index<level>()], itsSourceParam[0],
			                              forecastType);
		}
	}

	if (!sourceInfo)
	{
		myThreadedLogger.Warning(fmt::format("Skipping step {}, level {}", forecastTime.Step(), forecastLevel));
		return;
	}

	SetParamAggregation(sourceInfo->Param(), myTargetInfo);
	SetAB(myTargetInfo, sourceInfo);
	myTargetInfo->Grid()->UVRelativeToGrid(sourceInfo->Grid()->UVRelativeToGrid());

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda() && itsEnsemble == nullptr)
	{
		deviceType = "GPU";

		transformergpu::Process(itsConfiguration, myTargetInfo, sourceInfo, itsScale, itsBase, itsMinimumValue,
		                        itsMaximumValue);
	}
	else
#endif
	{
		deviceType = "CPU";

		auto& result = VEC(myTargetInfo);
		const auto& source = VEC(sourceInfo);

		transform(source.begin(), source.end(), result.begin(),
		          [&](const double& value) { return fma(value, itsScale, itsBase); });

		if (IsValid(itsMinimumValue))
		{
			for_each(result.begin(), result.end(),
			         [&](double& value)
			         {
				         if (IsValid(value))
				         {
					         value = fmin(fmax(value, itsMinimumValue), itsMaximumValue);
				         }
			         });
		}
	}

	if (itsChangeMissingTo.empty() == false)
	{
		double newMissing = MissingValue<double>();
		if (itsChangeMissingTo == "mean")
		{
			newMissing = numerical_functions::Mean(util::RemoveMissingValues(VEC(myTargetInfo)));
		}
		else if (itsChangeMissingTo == "median")
		{
			newMissing = numerical_functions::Median(util::RemoveMissingValues(VEC(myTargetInfo)));
		}
		else
		{
			newMissing = stod(itsChangeMissingTo);
		}

		auto& vec = VEC(myTargetInfo);
		replace_if(
		    vec.begin(), vec.end(), [=](double d) { return IsMissing(d); }, newMissing);
	}

	myThreadedLogger.Info(fmt::format("[{}] Missing values: {}/{}", deviceType, myTargetInfo->Data().MissingCount(),
	                                  myTargetInfo->Data().Size()));
}
