/**
 * @file transformer.cpp
 */

#include "transformer.h"
#include "forecast_time.h"
#include "interpolate.h"
#include "level.h"
#include "logger.h"
#include "newbase/NFmiDataMatrix.h"
#include "newbase/NFmiFastQueryInfo.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "util.h"
#include <gis/CoordinateMatrix.h>

#include "fetcher.h"
#include "hitool.h"
#include "querydata.h"

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
      itsDecimalPrecision(kHPMissingInt),
      itsDoLandscapeInterpolation(false),
      itsParamDefinitionFromConfig(false)
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

namespace
{
template <typename T>
T NewbaseMissingValue();

template <>
float NewbaseMissingValue()
{
	return kFloatMissing;
}

template <typename T, typename U>
NFmiDataMatrix<U> InfoToDataMatrix(const std::shared_ptr<himan::info<T>>& in)
{
	NFmiDataMatrix<U> ret(in->Data().SizeX(), in->Data().SizeY(), NewbaseMissingValue<U>());

	himan::matrix<T> origdata;

	if (dynamic_cast<himan::regular_grid*>(in->Grid().get())->ScanningMode() == himan::kTopLeft)
	{
		origdata = in->Data();
		himan::util::Flip<T>(origdata);
	}

	const std::vector<T>& data = (origdata.Size() == 0) ? VEC(in) : origdata.Values();

	for (size_t i = 0; i < data.size(); i++)
	{
		ret.SetValue(static_cast<int>(i), static_cast<U>(data[i]));
	}
	return ret;
}

template <typename T>
NFmiDataMatrix<bool> LSMToWaterMask(const NFmiDataMatrix<float>& in)
{
	NFmiDataMatrix<bool> ret(in.NX(), in.NY(), false);

	for (size_t i = 0; i < in.NX() * in.NY(); i++)
	{
		const int v = static_cast<int>(in.GetValue(static_cast<int>(i), 0.f));

		switch (v)
		{
			case 210:
				ret.SetValue(static_cast<int>(i), true);
				break;
			default:
				break;
		}
	}

	return ret;
}

Fmi::CoordinateMatrix CreateCoordinateMatrix(const himan::regular_grid* from, const himan::regular_grid* to)
{
	std::vector<himan::point> xy = from->XY(*to);

	const size_t ni = to->Ni();
	const size_t nj = to->Nj();

	Fmi::CoordinateMatrix cm(ni, nj);

	for (size_t j = 0; j < nj; j++)
	{
		for (size_t i = 0; i < ni; i++)
		{
			cm.set<himan::point>(i, j, xy[i + j * ni]);
		}
	}
	return cm;
}
}  // namespace

template <typename T>
shared_ptr<himan::info<T>> transformer::LandscapeInterpolation(const forecast_time& ftime, const level& lvl,
                                                               const param& par, const forecast_type& ftype)
{
	itsLogger.Trace("Executing landscape interpolation");

	// Fetch source data without interpolation (obviously)

	auto f = GET_PLUGIN(fetcher);
	f->DoInterpolation(false);

	auto cnf = make_shared<plugin_configuration>(*itsConfiguration);
	const level zeroH(kHeight, 0);

	// Specify nearest interpolation for these, especially
	// required for LC-N which is not a fraction between 0...1
	// (as name suggests) but a code table.

	const param lc("LC-N", -1, 1, 0, himan::kNearestPoint);
	const param z("Z-M2S2", -1, 1, 0, himan::kNearestPoint);

	auto source = f->Fetch<T>(cnf, ftime, lvl, par, ftype, false);
	auto hgt = f->Fetch<float>(cnf, ftime, zeroH, z, ftype, false);
	auto lr = f->Fetch<float>(cnf, ftime, zeroH, param("LR-KM"), ftype, false);
	auto mask = f->Fetch<float>(cnf, ftime, zeroH, param("LC-0TO1"), ftype, false);  // Model has land cover 0..1

	if (source->Data().MissingCount() > 0 || hgt->Data().MissingCount() > 0 || lr->Data().MissingCount() > 0 ||
	    mask->Data().MissingCount() > 0)
	{
		itsLogger.Debug("Source data has missing values");
	}

	// Interpolate DEM and LSM to our wanted (target) grid

	f->DoInterpolation(true);

	cnf->SourceGeomNames({""});
	cnf->SourceProducers({producer(521, 0, 0, "GLOBCOVER")});

	const raw_time lsmTime("2010-12-21", "%Y-%m-%d");
	auto lsm =
	    f->Fetch<unsigned char>(cnf, forecast_time(lsmTime, lsmTime), zeroH, lc, forecast_type(kAnalysis), false);

	cnf->SourceProducers({producer(520, 0, 0, "VIEWFINDER")});

	const raw_time demTime("2008-09-11", "%Y-%m-%d");
	auto dem = f->Fetch<short>(cnf, forecast_time(demTime, demTime), zeroH, z, forecast_type(kAnalysis), false);

	const size_t lsmMissing = lsm->Data().MissingCount();
	const size_t demMissing = dem->Data().MissingCount();

	if (lsmMissing > 0 || demMissing > 0)
	{
		itsLogger.Warning(fmt::format("Missing values in environment data: DEM {}, LSM {}", demMissing, lsmMissing));
	}

	const Fmi::CoordinateMatrix gp = CreateCoordinateMatrix(dynamic_cast<regular_grid*>(source->Grid().get()),
	                                                        dynamic_cast<regular_grid*>(dem->Grid().get()));

	auto q = GET_PLUGIN(querydata);
	std::shared_ptr<NFmiQueryData> qd = q->CreateQueryData(*source, true);
	NFmiFastQueryInfo qi(qd.get());

	const auto demM = InfoToDataMatrix<short, float>(dem);
	const auto lsmM = LSMToWaterMask<float>(InfoToDataMatrix<unsigned char, float>(lsm));
	const auto dataM = qi.Values();
	const auto hgtM = InfoToDataMatrix<float, float>(hgt);
	const auto lrM = InfoToDataMatrix<float, float>(lr);
	const auto mM = InfoToDataMatrix<float, float>(mask);

	// these should be equal
	itsLogger.Trace(fmt::format("gp        {},{}", gp.width(), gp.height()));
	itsLogger.Trace(fmt::format("dem       {},{}", demM.NX(), demM.NY()));
	itsLogger.Trace(fmt::format("lsm       {},{}", lsmM.NX(), lsmM.NY()));
	itsLogger.Trace(fmt::format("source    {},{}", source->Data().SizeX(), source->Data().SizeY()));

	// these should be equal
	itsLogger.Trace(fmt::format("data      {},{}", dataM.NX(), dataM.NY()));
	itsLogger.Trace(fmt::format("hgtmat    {},{}", hgtM.NX(), hgtM.NY()));
	itsLogger.Trace(fmt::format("lapsemat  {},{}", lrM.NX(), lrM.NY()));
	itsLogger.Trace(fmt::format("maskmat   {},{}", mM.NX(), mM.NY()));

	NFmiDataMatrix<float> lsValues = qi.LandscapeInterpolatedValues(dataM, gp, demM, lsmM, hgtM, lrM, mM);

	auto target = make_shared<info<T>>(ftype, ftime, lvl, par);
	auto b = make_shared<base<T>>();
	b->grid = shared_ptr<grid>(itsConfiguration->BaseGrid()->Clone());

	target->Create(b, true);

	auto& data = VEC(target);

	for (size_t i = 0; i < target->Grid()->Size(); i++)
	{
		const float val = lsValues.GetValue(static_cast<int>(i), kFloatMissing);

		data[i] = (val == kFloatMissing) ? MissingValue<T>() : val;
	}

	return target;
}

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

	if (itsConfiguration->Exists("landscape_interpolation"))
	{
		itsDoLandscapeInterpolation = util::ParseBoolean(itsConfiguration->GetValue("landscape_interpolation"));
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

	const auto grib_kv = itsConfiguration->GetValue("extra_file_metadata");

	if (!grib_kv.empty())
	{
		const auto list = util::Split(grib_kv, ",");
		for (const auto& e : list)
		{
			const auto kv = util::Split(e, "=");
			if (kv.size() != 2)
			{
				itsLogger.Warning(fmt::format("Invalid extra_file_metadata option: {}", e));
				continue;
			}

			itsExtraFileMetadata.emplace(kv[0], kv[1]);
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

	SetParams(itsTargetParam, itsParamDefinitionFromConfig);

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

	auto target = std::unique_ptr<grid>(myTargetInfo->Grid()->Clone());
	target->UVRelativeToGrid(false);

	interpolate::RotateVectorComponents(a->Grid().get(), target.get(), *myTargetInfo, *secondInfo,
	                                    itsConfiguration->UseCuda());
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
		myThreadedLogger.Info(fmt::format("[{}] Missing values: {}/{}", itsConfiguration->UseCuda() ? "GPU" : "CPU",
		                                  myTargetInfo->Data().MissingCount(), myTargetInfo->Data().Size()));
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
		if (itsDoLandscapeInterpolation)
		{
			sourceInfo = LandscapeInterpolation<double>(forecastTime, itsSourceLevels[myTargetInfo->Index<level>()],
			                                            itsSourceParam[0], forecastType);
		}
		else
		{
			sourceInfo = f->Fetch(itsConfiguration, forecastTime, itsSourceLevels[myTargetInfo->Index<level>()],
			                      itsSourceParam[0], forecastType, itsConfiguration->UseCudaForPacking());
		}
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
	writeOptions.extra_metadata = itsExtraFileMetadata;

	return compiled_plugin_base::WriteToFile(targetInfo, writeOptions);
}
