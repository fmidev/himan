#include "blend.h"
#include "fetcher.h"
#include "plugin_factory.h"
#include "radon.h"
#include "writer.h"

#include <algorithm>
#include <mutex>
#include <thread>

//
// Forecast blender
//

namespace himan
{
namespace plugin
{
using namespace std;

static mutex singleFileWriteMutex;

const string kClassName = "himan::plugin::blend";

// 'decaying factor' for bias and mae
const double alpha = 0.05;

const producer kBlendWeightProd(182, 86, 182, "BLENDW");
const producer kBlendRawProd(183, 86, 183, "BLENDR");
const producer kBlendBiasProd(184, 86, 184, "BLENDB");

// When adjusting origin times, we need to check that the resulting time is compatible with the model's
// (used) forecast length.
const int kMosForecastLength = 240;
const int kEcmwfForecastLength = 240;
const int kHirlamForecastLength = 54;
const int kMepsForecastLength = 66;
const int kGfsForecastLength = 240;

const producer kLapsProd(109, 86, 109, "LAPSSCAN");
const string kLapsGeom = "LAPSSCANLARGE";

// Each blend producer is composed of these original producers. We use forecast_types to distinguish them
// from each other, and this way we don't have to create bunch of extra producers.
const blend_producer LAPS(forecast_type(kAnalysis), 0, 1);
const blend_producer MOS(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kMos)), kMosForecastLength,
                         12);
const blend_producer ECMWF(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kEcmwf)),
                           kEcmwfForecastLength, 12);
const blend_producer HIRLAM(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kHirlam)),
                            kHirlamForecastLength, 6);
const blend_producer MEPS(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kMeps)),
                          kMepsForecastLength, 6);
const blend_producer GFS(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kGfs)), kGfsForecastLength,
                         6);

blend::blend() : itsCalculationMode(kCalculateNone), itsNumHours(0), itsAnalysisTime(), itsBlendProducer()
{
	itsLogger = logger("blend");
}

// Create observation analysis time (ie. LAPS analysis time) from analysis hour that's
// given in conf file
forecast_time MakeAnalysisTime(const forecast_time& currentTime, int analysisHour)
{
	const int validHour = stoi(currentTime.ValidDateTime().String("%H"));

	forecast_time analysisFetchTime = currentTime;

	analysisFetchTime.ValidDateTime().Adjust(kHourResolution, -validHour);  // set to 00
	analysisFetchTime.ValidDateTime().Adjust(kHourResolution, analysisHour);

	analysisFetchTime.OriginDateTime() = analysisFetchTime.ValidDateTime();

	return analysisFetchTime;
}

// Read the configuration and set plugin properties:
// - calculation mode (bias, mae, blend) that is to be dispatched in Calculate()
// - producer for bias and mae
// - analysis time (obs)
bool blend::ParseConfigurationOptions(const shared_ptr<const plugin_configuration>& conf)
{
	const string mode = conf->GetValue("mode");

	if (mode == "blend")
	{
		itsCalculationMode = kCalculateBlend;
	}
	else if (mode == "mae")
	{
		itsCalculationMode = kCalculateMAE;
	}
	else if (mode == "bias")
	{
		itsCalculationMode = kCalculateBias;
	}
	else
	{
		itsLogger.Fatal("Invalid blender 'mode' specified: '" + mode + "'");
		return false;
	}

	const string hours = conf->GetValue("hours");

	if ((itsCalculationMode == kCalculateBias || itsCalculationMode == kCalculateMAE) && hours.empty())
	{
		throw std::runtime_error(ClassName() + ": number of previous hours ('hours') for calculation not specified");
	}
	if (itsCalculationMode == kCalculateBias || itsCalculationMode == kCalculateMAE)
	{
		itsNumHours = stoi(hours);
	}

	// Producer for bias and mae calculation (itsProdFtype is only used with these modes)
	const string prod = conf->Exists("producer") ? conf->GetValue("producer") : "";
	if ((itsCalculationMode == kCalculateBias || itsCalculationMode == kCalculateMAE) && prod.empty())
	{
		throw std::runtime_error(ClassName() + ": bias calculation data producer ('producer') not defined");
	}

	if (itsCalculationMode == kCalculateBias || itsCalculationMode == kCalculateMAE)
	{
		if (prod == "ECG")
		{
			itsBlendProducer = ECMWF;
		}
		else if (prod == "HL2")
		{
			itsBlendProducer = HIRLAM;
		}
		else if (prod == "MEPS")
		{
			itsBlendProducer = MEPS;
		}
		else if (prod == "GFS")
		{
			itsBlendProducer = GFS;
		}
		else if (prod == "MOS")
		{
			itsBlendProducer = MOS;
		}
		else
		{
			throw std::runtime_error(ClassName() + ": unknown producer for bias/mae calculation");
		}
	}

	try
	{
		const int ahour = stoi(conf->GetValue("analysis_hour"));
		itsAnalysisTime = MakeAnalysisTime(itsTimeIterator.At(0), ahour);
	}
	catch (const std::invalid_argument& e)
	{
		itsLogger.Fatal("Analysis_hour not defined");
		return false;
	}

	return true;
}

// - parameter

void blend::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	if (ParseConfigurationOptions(conf) == false)
	{
		return;
	}

	// Each parameter has to be processed with a separate process queue invocation.
	const string p = conf->GetValue("param");
	if (p.empty())
	{
		throw std::runtime_error(ClassName() + ": parameter not specified");
	}

	SetParams({param(p)});
	Start();
}

void blend::Calculate(shared_ptr<info<double>> targetInfo, unsigned short threadIndex)
{
	auto log = logger("blendThread#" + to_string(threadIndex));
	const string deviceType = "CPU";
	const forecast_time currentTime = targetInfo->Time();
	const param currentParam = targetInfo->Param();

	switch (itsCalculationMode)
	{
		case kCalculateBlend:
			log.Info("Calculating blend for " + currentParam.Name() + " " +
			         static_cast<string>(currentTime.ValidDateTime()));
			CalculateBlend(targetInfo, threadIndex);
			break;
		case kCalculateMAE:
			log.Info("Calculating weights for " + currentParam.Name() + " " +
			         static_cast<string>(currentTime.ValidDateTime()));
			CalculateMember(targetInfo, threadIndex, kCalculateMAE);
			break;
		case kCalculateBias:
			log.Info("Calculating bias corrected grids for " + currentParam.Name() + " " +
			         static_cast<string>(currentTime.ValidDateTime()));
			CalculateMember(targetInfo, threadIndex, kCalculateBias);
			break;
		default:
			log.Error("Invalid calculation mode");
			himan::Abort();
	}
}

matrix<double> blend::CalculateBias(logger& log, shared_ptr<info<double>> targetInfo, const forecast_time& calcTime)
{
	shared_ptr<plugin_configuration> cnf = make_shared<plugin_configuration>(*itsConfiguration);

	const param& currentParam = targetInfo->Param();
	const forecast_time& currentTime = targetInfo->Time();
	const level& currentLevel = targetInfo->Level();

	info_t analysis = Fetch(itsAnalysisTime, currentLevel, currentParam, LAPS.type, {kLapsGeom}, kLapsProd);

	if (!analysis)
	{
		return matrix<double>();
	}

	matrix<double> currentBias(targetInfo->Data().SizeX(), targetInfo->Data().SizeY(), targetInfo->Data().SizeZ(),
	                           MissingDouble());
	vector<double> forecast;

	forecast_time leadTime(calcTime);

	// MOS doesn't have hours 0, 1, 2. So we'll set this to missing. We don't want to do this with other models, since
	// in these cases it is certainly an error that needs to be looked at and fixed manually.
	if (itsBlendProducer == MOS && currentTime.Step() < 3)
	{
		forecast = vector<double>(targetInfo->Data().Size(), MissingDouble());
	}
	else
	{
		info_t Info = Fetch(leadTime, currentLevel, currentParam, itsBlendProducer.type);

		if (!Info)
		{
			return matrix<double>();
		}

		forecast = VEC(Info);
	}

	// Previous forecast's bias corrected data is optional. If the data is not found we'll set the grid to missing.
	// (This happens, for example, during initialization.)
	forecast_time prevLeadTime(leadTime);
	prevLeadTime.OriginDateTime().Adjust(kHourResolution, -itsBlendProducer.originTimestep);

	vector<double> prevBias;

	if (prevLeadTime.Step() <= itsBlendProducer.forecastLength)
	{
		info_t prev = Fetch(prevLeadTime, currentLevel, currentParam, itsBlendProducer.type,
		                    {itsConfiguration->TargetGeomName()}, kBlendBiasProd);

		if (!prev)
		{
			prevBias = vector<double>(targetInfo->Data().Size(), MissingDouble());
		}
		else
		{
			prevBias = VEC(prev);
		}
	}
	else
	{
		prevBias = vector<double>(targetInfo->Data().Size(), MissingDouble());
	}

	// Introduce shorter names for clarity
	const vector<double>& O = VEC(analysis);
	const vector<double>& BC = prevBias;
	vector<double>& B = currentBias.Values();

	for (size_t i = 0; i < forecast.size(); i++)
	{
		double f = forecast[i];
		double o = O[i];
		double bc = BC[i];

		if (IsMissing(bc))
		{
			bc = 0.0;
		}

		if (IsMissing(f) || IsMissing(o))
		{
			f = 0.0;
			o = 0.0;
		}
		B[i] = (1.0 - alpha) * bc + alpha * (f - o);
	}

	return currentBias;
}

// Follows largely the same format as CalculateBias
matrix<double> blend::CalculateMAE(logger& log, shared_ptr<info<double>> targetInfo, const forecast_time& calcTime)
{
	const param& currentParam = targetInfo->Param();
	const forecast_time& currentTime = targetInfo->Time();
	const level& currentLevel = targetInfo->Level();

	info_t analysis = Fetch(itsAnalysisTime, currentLevel, currentParam, LAPS.type, {kLapsGeom}, kLapsProd);

	if (!analysis)
	{
		return matrix<double>();
	}

	matrix<double> MAE(targetInfo->Data().SizeX(), targetInfo->Data().SizeY(), targetInfo->Data().SizeZ(),
	                   MissingDouble());

	forecast_time leadTime(calcTime);

	info_t bias = Fetch(leadTime, currentLevel, currentParam, itsBlendProducer.type,
	                    itsConfiguration->SourceGeomNames(), kBlendBiasProd);

	if (!bias)
	{
		return matrix<double>();
	}
	// See note pertaining to MOS at CalculateBias.
	vector<double> forecast;
	if (itsBlendProducer == MOS && currentTime.Step() < 3)
	{
		forecast = vector<double>(targetInfo->Data().Size(), MissingDouble());
	}
	else
	{
		info_t Info = Fetch(leadTime, currentLevel, currentParam, itsBlendProducer.type);

		if (Info)
		{
			matrix<double>();
		}

		forecast = VEC(Info);
	}

	vector<double> prevMAE;

	forecast_time prevLeadTime(leadTime);
	prevLeadTime.OriginDateTime().Adjust(kHourResolution, -itsBlendProducer.originTimestep);

	if (prevLeadTime.Step() <= itsBlendProducer.forecastLength)
	{
		info_t prev = Fetch(prevLeadTime, currentLevel, currentParam, itsBlendProducer.type,
		                    {itsConfiguration->TargetGeomName()}, kBlendWeightProd);

		if (!prev)
		{
			prevMAE = vector<double>(targetInfo->Data().Size(), MissingDouble());
		}
		else
		{
			prevMAE = VEC(prev);
		}
	}
	else
	{
		prevMAE = vector<double>(targetInfo->Data().Size(), MissingDouble());
	}

	const vector<double>& O = VEC(analysis);
	const vector<double>& B = VEC(bias);
	vector<double>& mae = MAE.Values();

	for (size_t i = 0; i < mae.size(); i++)
	{
		double o = O[i];
		double f = forecast[i];
		double b = B[i];
		double _prevMAE = prevMAE[i];

		if (IsMissing(f) || IsMissing(o))
		{
			f = 0.0;
			o = 0.0;
		}

		if (IsMissing(_prevMAE))
		{
			_prevMAE = 0.0;
		}

		if (IsMissing(b))
		{
			b = 0.0;
		}

		const double bcf = f - b;
		mae[i] = (1.0 - alpha) * _prevMAE + alpha * std::abs(bcf - o);
	}

	return MAE;
}

void blend::CalculateMember(shared_ptr<info<double>> targetInfo, unsigned short threadIdx, blend_mode mode)
{
	logger log;
	if (mode == kCalculateBias)
	{
		log = logger("calculateBias#" + to_string(threadIdx));
	}
	else
	{
		log = logger("calculateMAE#" + to_string(threadIdx));
	}

	forecast_type forecastType = itsBlendProducer.type;
	level targetLevel = targetInfo->Level();
	forecast_time current = targetInfo->Time();

	const raw_time latestOrigin = targetInfo->Time().OriginDateTime();
	log.Info("Latest origin time for producer: " + latestOrigin.String());

	// Used for fetching raw model output, bias, and weight for models.

	if (itsBlendProducer == MOS || itsBlendProducer == ECMWF)
	{
		// ValidDateTime can be borked on ECMWF and MOS.
		const int validHour = stoi(current.OriginDateTime().String("%H"));
		if (validHour == 6 || validHour == 18)
		{
			current.ValidDateTime().Adjust(kHourResolution, -6);
		}
	}

	// Start from the 'earliest' (set below) point in time, and proceed to current time.
	forecast_time ftime(latestOrigin, current.ValidDateTime());

	const int maxStep = itsBlendProducer.forecastLength;
	const int originTimeStep = itsBlendProducer.originTimestep;

	ftime.OriginDateTime().Adjust(kHourResolution, -itsNumHours);

	// Check that we're not overstepping the forecast length.
	while (ftime.Step() > maxStep)
	{
		ftime.OriginDateTime().Adjust(kHourResolution, originTimeStep);
	}

	// Problem: targetInfo has information for the data that we want to fetch, but because of the convoluted way of
	// calculating everything, this doesn't match with the data we want to write out.
	// Solution: Create a new info and write that out.
	info_t Info = make_shared<info<double>>(*targetInfo);
	vector<forecast_type> ftypes{itsBlendProducer.type};

	if (mode == kCalculateBias)
	{
		Info->Producer(kBlendBiasProd);
	}
	else
	{
		Info->Producer(kBlendWeightProd);
	}

	SetupOutputForecastTimes(Info, latestOrigin, current, maxStep, originTimeStep);
	Info->Set<forecast_type>(ftypes);
	Info->Create(targetInfo->Base(), true);
	Info->First();

	while (true)
	{
		/// LAPS fetching is done via forecast time in |targetInfo|. Each call to Calculate{Bias,MAE} calculates for one
		/// time step.
		///
		/// So we want to adjust ftime based on time.

		// Newest forecast
		if (ftime.Step() < 0)
		{
			log.Trace("End of forecast, breaking");
			break;
		}

		if (ftime.OriginDateTime() > current.OriginDateTime() || ftime.OriginDateTime() > latestOrigin)
		{
			break;
		}

		matrix<double> d;

		if (mode == kCalculateBias)
		{
			d = CalculateBias(log, targetInfo, ftime);
		}
		else
		{
			d = CalculateMAE(log, targetInfo, ftime);
		}

		if (d.Size() > 0)
		{
			if (Info->Find<forecast_type>(forecastType))
			{
				auto b = Info->Base();
				b->data = std::move(d);
				WriteToFile(Info);
			}
			else
			{
				log.Error("Failed to set the correct output forecast type");
				himan::Abort();
			}
		}

		Info->Next<forecast_time>();

		ftime.OriginDateTime().Adjust(kHourResolution, originTimeStep);
	}

	// Remove grid so it won't be written: blend is manually writing out grids with a separate info.
	// Previously blend implemented Start() and prevented it from being written, but this is not
	// allowed now. Blend should also be fixed so that we don't need this workaround.

	auto b = targetInfo->Base();
	b->grid = nullptr;
	b->data.Clear();

	targetInfo->Base(b);
}

std::vector<info_t> blend::FetchRawGrids(shared_ptr<info<double>> targetInfo, unsigned short threadIdx) const
{
	auto log = logger("calculateBlend_FetchRawGrids#" + to_string(threadIdx));

	const forecast_time& currentTime = targetInfo->Time();
	const param& currentParam = targetInfo->Param();
	const level& currentLevel = targetInfo->Level();

	// Fetch previous model runs raw fields for EC and MOS when we're calculating during the 06 and 18 cycles.
	forecast_time ecmosFetchTime = currentTime;
	const int hour = stoi(ecmosFetchTime.OriginDateTime().String("%H"));
	if (hour == 6 || hour == 18)
	{
		ecmosFetchTime.OriginDateTime().Adjust(kHourResolution, -6);
	}

	info_t mosRaw = Fetch(ecmosFetchTime, currentLevel, currentParam, MOS.type);
	info_t ecRaw = Fetch(ecmosFetchTime, currentLevel, currentParam, ECMWF.type);
	info_t mepsRaw = Fetch(currentTime, currentLevel, currentParam, MEPS.type);
	info_t hirlamRaw = Fetch(currentTime, currentLevel, currentParam, HIRLAM.type);
	info_t gfsRaw = Fetch(currentTime, currentLevel, currentParam, GFS.type);

	//
	// We want to return nullptrs here so that we can skip over these entries in the Calculate-loop.
	//

	if (mosRaw)
		log.Info("MOS_raw missing count: " + to_string(mosRaw->Data().MissingCount()));
	else
		log.Info("MOS_raw missing completely");

	if (ecRaw)
		log.Info("EC_raw missing count: " + to_string(ecRaw->Data().MissingCount()));
	else
		log.Info("EC_raw missing completely");

	if (mepsRaw)
		log.Info("MEPS_raw missing count: " + to_string(mepsRaw->Data().MissingCount()));
	else
		log.Info("MEPS_raw missing completely");

	if (hirlamRaw)
		log.Info("HIRLAM_raw missing count: " + to_string(hirlamRaw->Data().MissingCount()));
	else
		log.Info("HIRLAM_raw missing completely");

	if (gfsRaw)
		log.Info("GFS_raw missing count: " + to_string(gfsRaw->Data().MissingCount()));
	else
		log.Info("GFS_raw missing completely");

	return std::vector<info_t>{mosRaw, ecRaw, mepsRaw, hirlamRaw, gfsRaw};
}

namespace
{
// Try to fetch 'historical data' for the given arguments. This is done because Bias and MAE data is calculated for
// 'old data'. Naturally we don't have the appropriate weights for current data or future data (the data we want to
// blend), so we'll scan for the first occurance of a grid with the wanted parameters.
info_t FetchHistorical(logger& log, shared_ptr<plugin_configuration> cnf, const forecast_time& forecastTime,
                       HPTimeResolution stepResolution, const param& parm, const blend_producer& blendProd,
                       const producer& prod)
{
	auto f = GET_PLUGIN(fetcher);
	auto r = GET_PLUGIN(radon);

	const string geomName = cnf->TargetGeomName();

	cnf->SourceGeomNames({geomName});
	cnf->SourceProducers({prod});

	// Newest BC/MAE field that forecasts to 240 hours is -240 hours from current analysis time.
	// Use these fields for applying BC/MAE.
	forecast_time ftime = forecastTime;
	ftime.StepResolution(stepResolution);
	int currentStep = ftime.Step();

	const int maxIterations = blendProd.forecastLength / blendProd.originTimestep;

	for (int i = 0; i < maxIterations; i++)
	{
		ftime.OriginDateTime().Adjust(kHourResolution, -blendProd.originTimestep);
		ftime.ValidDateTime() = ftime.OriginDateTime();
		ftime.ValidDateTime().Adjust(kHourResolution, currentStep);

		search_options opts(ftime, parm, level(kHeight, 2.0), prod, blendProd.type, cnf);
		const vector<string> files = r->Files(opts).first;
		if (files.empty())
		{
			log.Trace("Failed to find matching files for: " + to_string(prod.Id()) + "/" +
			          to_string(blendProd.type.Value()) + " " + ftime.OriginDateTime().String() + " " +
			          ftime.ValidDateTime().String());
		}
		else
		{
			break;
		}
	}

	// ftime is set to the time where we found a matching file
	info_t Info;
	try
	{
		Info = f->Fetch(cnf, ftime, level(kHeight, 2.0), parm, blendProd.type, false);
		log.Info("Found matching file: " + to_string(prod.Id()) + "/" + to_string(blendProd.type.Value()) + " " +
		         ftime.OriginDateTime().String() + " " + ftime.ValidDateTime().String());
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			himan::Abort();
		}
	}

	return Info;
}

std::vector<info_t> FetchBiasGrids(shared_ptr<info<double>> targetInfo, shared_ptr<plugin_configuration> cnf,
                                   unsigned short threadIdx)
{
	auto f = GET_PLUGIN(fetcher);
	auto log = logger("calculateBlend_FetchBiasGrids#" + to_string(threadIdx));

	const forecast_time fetchTime = targetInfo->Time();
	const HPTimeResolution currentResolution = fetchTime.StepResolution();
	const param currentParam = targetInfo->Param();

	info_t mosBias = FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, MOS, kBlendBiasProd);
	info_t ecBias = FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, ECMWF, kBlendBiasProd);
	info_t mepsBias = FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, MEPS, kBlendBiasProd);
	info_t hirlamBias = FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, HIRLAM, kBlendBiasProd);
	info_t gfsBias = FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, GFS, kBlendBiasProd);

	if (mosBias)
		log.Info("MOS_bias missing count: " + to_string(mosBias->Data().MissingCount()));
	else
		log.Info("MOS_bias missing completely");

	if (ecBias)
		log.Info("EC_bias missing count: " + to_string(ecBias->Data().MissingCount()));
	else
		log.Info("EC_bias missing completely");

	if (mepsBias)
		log.Info("MEPS_bias missing count: " + to_string(mepsBias->Data().MissingCount()));
	else
		log.Info("MEPS_bias missing completely");

	if (hirlamBias)
		log.Info("HIRLAM_bias missing count: " + to_string(hirlamBias->Data().MissingCount()));
	else
		log.Info("HIRLAM_bias missing completely");

	if (gfsBias)
		log.Info("GFS_bias missing count: " + to_string(gfsBias->Data().MissingCount()));
	else
		log.Info("GFS_bias missing completely");

	return std::vector<info_t>{mosBias, ecBias, mepsBias, hirlamBias, gfsBias};
}

std::vector<info_t> FetchMAEGrids(shared_ptr<info<double>> targetInfo, shared_ptr<plugin_configuration> cnf,
                                  unsigned short threadIdx)
{
	auto f = GET_PLUGIN(fetcher);
	auto log = logger("calculateBlend_FetchMAEGrids#" + to_string(threadIdx));

	const forecast_time fetchTime = targetInfo->Time();
	const HPTimeResolution currentResolution = fetchTime.StepResolution();
	const param currentParam = targetInfo->Param();

	info_t mos = FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, MOS, kBlendWeightProd);
	info_t ec = FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, ECMWF, kBlendWeightProd);
	info_t hirlam = FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, HIRLAM, kBlendWeightProd);
	info_t meps = FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, MEPS, kBlendWeightProd);
	info_t gfs = FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, GFS, kBlendWeightProd);

	if (mos)
		log.Info("MOS_mae missing count: " + to_string(mos->Data().MissingCount()));
	else
		log.Info("MOS_mae missing completely");

	if (ec)
		log.Info("EC_mae missing count: " + to_string(ec->Data().MissingCount()));
	else
		log.Info("EC_mae missing completely");

	if (meps)
		log.Info("MEPS_mae missing count: " + to_string(meps->Data().MissingCount()));
	else
		log.Info("MEPS_mae missing completely");

	if (hirlam)
		log.Info("HIRLAM_mae missing count: " + to_string(hirlam->Data().MissingCount()));
	else
		log.Info("HIRLAM_mae missing completely");

	if (gfs)
		log.Info("GFS_mae missing count: " + to_string(gfs->Data().MissingCount()));
	else
		log.Info("GFS_mae missing completely");

	return std::vector<info_t>{mos, ec, meps, hirlam, gfs};
}

}  // namespace

void blend::CalculateBlend(shared_ptr<info<double>> targetInfo, unsigned short threadIdx)
{
	auto log = logger("calculateBlend#" + to_string(threadIdx));
	const string deviceType = "CPU";

	const forecast_time currentTime = targetInfo->Time();
	const param currentParam = targetInfo->Param();

	log.Info("Blending " + currentParam.Name() + " " + static_cast<string>(currentTime.ValidDateTime()));

	// NOTE: If one of the grids is missing, we should still put an empty grid there. Since all the stages assume that
	// F[i], B[i], W[i] are all of the same model! This means that the ordering is fixed for the return vector of
	// FetchRawGrids, FetchBiasGrids, FetchMAEGrids.
	vector<info_t> forecasts = FetchRawGrids(targetInfo, threadIdx);

	if (std::all_of(forecasts.begin(), forecasts.end(), [&](info_t i) { return i == nullptr; }))
	{
		log.Error("Failed to acquire any source data");
		return;
	}

	// First reset all the locations for the upcoming loop.
	for (const auto& f : forecasts)
	{
		if (f)
		{
			f->ResetLocation();
		}
	}

	// Load all the precalculated bias factors from BLENDB
	vector<info_t> biases = FetchBiasGrids(targetInfo, make_shared<plugin_configuration>(*itsConfiguration), threadIdx);
	if (std::all_of(biases.begin(), biases.end(), [&](info_t i) { return i == nullptr; }))
	{
		log.Error("Failed to acquire any bias grids");
	}

	for (const auto& b : biases)
	{
		if (b)
		{
			b->ResetLocation();
		}
	}

	// Load all the precalculated weights from BLENDW
	vector<info_t> preweights =
	    FetchMAEGrids(targetInfo, make_shared<plugin_configuration>(*itsConfiguration), threadIdx);

	if (std::all_of(preweights.begin(), preweights.end(), [&](info_t i) { return i == nullptr; }))
	{
		log.Error("Failed to acquire any MAE grids");
	}

	for (auto& w : preweights)
	{
		if (w)
		{
			w->ResetLocation();
		}
	}

	// Used to collect values in the case of missing values.
	vector<double> collectedValues;
	collectedValues.reserve(forecasts.size());

	vector<double> collectedWeights;
	collectedWeights.reserve(forecasts.size());

	vector<double> collectedBiases;
	collectedBiases.reserve(forecasts.size());

	size_t forecastWarnings = 0;
	size_t biasWarnings = 0;
	size_t weightWarnings = 0;

	targetInfo->ResetLocation();
	while (targetInfo->NextLocation())
	{
		double F = 0.0;
		size_t numMissing = 0;

		for (const auto& tup : zip_range(forecasts, preweights, biases))
		{
			info_t f = tup.get<0>();
			info_t w = tup.get<1>();
			info_t b = tup.get<2>();

			if (!f || !f->NextLocation())
			{
				forecastWarnings++;
				continue;
			}

			if (!w || !w->NextLocation())
			{
				weightWarnings++;
				continue;
			}

			if (!b || !b->NextLocation())
			{
				biasWarnings++;
				continue;
			}

			const double v = f->Value();
			const double wv = w->Value();
			const double bb = b->Value();
			if (IsMissing(v) || IsMissing(wv) || IsMissing(bb))
			{
				numMissing++;
				continue;
			}

			collectedValues.push_back(v);
			collectedWeights.push_back(wv);
			collectedBiases.push_back(bb);
		}

		// Used for the actual application of the weights.
		vector<double> weights;
		weights.resize(collectedWeights.size());

		for (size_t i = 0; i < weights.size(); i++)
		{
			double sum = 0.0;
			for (const double& w : collectedWeights)
			{
				// currentWeights already pruned of missing values
				sum += 1.0 / w;  // could be zero
			}

			ASSERT(sum != 0);

			weights[i] = 1.0 / sum;
		}

		if (numMissing == forecasts.size())
		{
			F = MissingDouble();
		}
		else
		{
			double sw = 0.0;
			for (const auto& w : weights)
			{
				sw += w;
			}

			if (sw > 0.0)
			{
				// Finally apply the weights to the forecast values.
				for (const auto& tup : zip_range(weights, collectedValues, collectedBiases))
				{
					const double w = tup.get<0>();
					const double v = tup.get<1>();
					const double b = tup.get<2>();
					F += w * (v - b);
				}

				F /= sw;
			}
			else
			{
				F = MissingDouble();
			}
		}

		targetInfo->Value(F);

		collectedValues.clear();
		collectedBiases.clear();
		collectedWeights.clear();
	}

	log.Warning("Failed to advance forecast iterator position " + to_string(forecastWarnings) + " times");
	log.Warning("Failed to advance bias iterator position " + to_string(biasWarnings) + " times");
	log.Warning("Failed to advance weight iterator position " + to_string(weightWarnings) + " times");

	log.Info("[" + deviceType + "] Missing values: " + to_string(targetInfo->Data().MissingCount()) + "/" +
	         to_string(targetInfo->Data().Size()));
}

void blend::WriteToFile(const info_t targetInfo, write_options writeOptions)
{
	auto aWriter = GET_PLUGIN(writer);

	aWriter->WriteOptions(writeOptions);
	auto tempInfo = make_shared<info<double>>(*targetInfo);

	if (targetInfo->IsValidGrid() == false)
	{
		return;
	}

	if (itsConfiguration->FileWriteOption() == kDatabase || itsConfiguration->FileWriteOption() == kMultipleFiles)
	{
		aWriter->ToFile(tempInfo, itsConfiguration);
	}
	else
	{
		lock_guard<mutex> lock(singleFileWriteMutex);

		aWriter->ToFile(tempInfo, itsConfiguration, itsConfiguration->ConfigurationFile());
	}

	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		DeallocateMemory(*tempInfo);
	}
}

// Sets the forecast times for |Info| to the times we want to calculate for Bias and MAE (we need to go back in time).
void blend::SetupOutputForecastTimes(shared_ptr<info<double>> Info, const raw_time& latestOrigin,
                                     const forecast_time& current, int maxStep, int originTimeStep)
{
	vector<forecast_time> ftimes;

	Info->Reset<forecast_time>();

	forecast_time ftime(latestOrigin, current.ValidDateTime());
	ftime.OriginDateTime().Adjust(kHourResolution, -itsNumHours);

	while (ftime.Step() > maxStep)
	{
		ftime.OriginDateTime().Adjust(kHourResolution, originTimeStep);
	}

	for (int i = 0; i < itsNumHours; i += originTimeStep)
	{
		ftimes.push_back(ftime);
		ftime.OriginDateTime().Adjust(kHourResolution, originTimeStep);
	}
	ftimes.push_back(ftime);

	Info->Set<forecast_time>(ftimes);
}

}  // namespace plugin
}  // namespace himan
