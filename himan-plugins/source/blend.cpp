#include "blend.h"
#include "fetcher.h"
#include "plugin_factory.h"
#include "radon.h"
#include "writer.h"

#include <mutex>
#include <thread>
#include <algorithm>

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
const blend_producer LAPS(forecast_type(kAnalysis), level(kHeight, 2.0), 0, 1);
const blend_producer MOS(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kMos)), level(kHeight, 0.0),
                         kMosForecastLength, 12);
const blend_producer ECMWF(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kEcmwf)),
                           level(kGround, 0.0), kEcmwfForecastLength, 12);
const blend_producer HIRLAM(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kHirlam)),
                            level(kHeight, 2.0), kHirlamForecastLength, 6);
const blend_producer MEPS(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kMeps)),
                          level(kHeight, 2.0), kMepsForecastLength, 6);
const blend_producer GFS(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kGfs)), level(kGround, 0.0),
                         kGfsForecastLength, 6);

blend::blend()
    : itsCalculationMode(kCalculateNone), itsNumHours(0), itsAnalysisHour(0), itsProducer(), itsBlendProducer()
{
	itsLogger = logger("blend");
}

blend::~blend()
{
}

// 2 fetching helper functions are used:
// - FetchProd: specify all information except geometry name, throws exception if data is not found
// - FetchNoExcept: same as FetchProd, except exceptions are caught and nullptr is returned
static info_t FetchProd(shared_ptr<plugin_configuration> cnf, const forecast_time& forecastTime,
                        HPTimeResolution stepResolution, const param& parm, const blend_producer& blendProd,
                        const producer& prod)
{
	auto f = GET_PLUGIN(fetcher);

	const string geomName = cnf->TargetGeomName();

	forecast_time ftime = forecastTime;
	ftime.StepResolution(stepResolution);

	cnf->SourceGeomNames({geomName});
	cnf->SourceProducers({prod});

	return f->Fetch(cnf, ftime, blendProd.lvl, parm, blendProd.type, false);
}

static info_t FetchNoExcept(shared_ptr<plugin_configuration> cnf, const forecast_time& forecastTime,
                            HPTimeResolution stepResolution, const param& parm, const blend_producer& blendProd,
                            const producer& prod, const string& geom)
{
	auto f = GET_PLUGIN(fetcher);

	forecast_time ftime = forecastTime;
	ftime.StepResolution(stepResolution);

	cnf->SourceGeomNames({geom});
	cnf->SourceProducers({prod});

	info_t I;
	try
	{
		I = f->Fetch(cnf, ftime, blendProd.lvl, parm, blendProd.type, false);
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			himan::Abort();
		}
		else
		{
			return nullptr;
		}
	}
	return I;
}

static mutex dimensionMutex;

void blend::Run(unsigned short threadIndex)
{
	// Iterate through timesteps
	auto targetInfo = make_shared<info>(*itsInfo);

	while (itsDimensionsRemaining)
	{
		{
			lock_guard<mutex> lock(dimensionMutex);

			if (itsInfo->NextTime())
			{
				if (!targetInfo->Time(itsInfo->Time()))
				{
					throw std::runtime_error("invalid target time");
				}
			}
			else
			{
				itsDimensionsRemaining = false;
				break;
			}
		}

		Calculate(targetInfo, threadIndex);

		if (itsConfiguration->StatisticsEnabled())
		{
			// NOTE this is likely wrong. (Only counts the current iterator position's missing count.)
			itsConfiguration->Statistics()->AddToMissingCount(targetInfo->Data().MissingCount());
			itsConfiguration->Statistics()->AddToValueCount(targetInfo->Data().Size());
		}

		// NOTE: Each Calculate-function will call WriteToFile manually, since they write out data at different
		// frequencies.
	}
}

raw_time blend::LatestOriginTimeForProducer(const string& producer) const
{
	// These are hardcoded for simplicity.
	int producerId = -1;
	string geom;
	if (producer == "MOS")
	{
		producerId = 120;
		geom = "MOSKRIGING2";
	}
	else if (producer == "ECG")
	{
		producerId = 131;
		geom = "ECGLO0100";
	}
	else if (producer == "HL2")
	{
		producerId = 1;
		geom = "RCR068";
	}
	else if (producer == "MEPS")
	{
		producerId = 4;
		geom = "MEPSSCAN2500";
	}
	else if (producer == "GFS")
	{
		producerId = 53;
		geom = "GFS0250";
	}
	else
	{
		itsLogger.Error("Invalid producer string: " + producer);
		himan::Abort();
	}

	auto r = GET_PLUGIN(radon);

	const string latest = r->RadonDB().GetLatestTime(producerId, geom, 0);
	if (latest.empty())
	{
		itsLogger.Error("Failed to find latest time for producer: " + to_string(producerId) + " and geom: " + geom +
		                " from Radon");
		himan::Abort();
	}

	raw_time raw (latest);

	// With ECMWF and MOS we only want 00 and 12 times. MOS is only calculated for 00 and 12. ECMWF forecast length is
	// shorter for 6 and 18, and that seems to break everything.
	const int hour = stoi(raw.String("%H"));
	if (producer == "ECG" || producer == "MOS")
	{
		if (hour == 6 || hour == 18)
		{
			raw.Adjust(kHourResolution, -6);
		}
	}

	return raw;
}

void blend::Start()
{
	itsThreadCount = static_cast<short>(itsInfo->SizeTimes());

	vector<thread> threads;
	threads.reserve(static_cast<size_t>(itsThreadCount));

	itsInfo->Reset();
	itsInfo->FirstForecastType();
	itsInfo->FirstLevel();
	itsInfo->FirstParam();

	for (unsigned short i = 0; i < itsThreadCount; i++)
	{
		threads.push_back(thread(&blend::Run, this, i));
	}

	for (auto&& t : threads)
	{
		t.join();
	}

	Finish();
}

// Read the configuration and set plugin properties:
// - calculation mode (bias, mae, blend) that is to be dispatched in Calculate()
// - producer for bias and mae
// - analysis hour
// - parameter
void blend::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);
	vector<param> params;

	const string mode = conf->Exists("mode") ? conf->GetValue("mode") : "";
	if (mode.empty())
	{
		throw std::runtime_error(ClassName() + ": blender 'mode' not specified");
	}

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
		throw std::runtime_error(ClassName() + ": invalid blender 'mode' specified");
	}

	const string hours = conf->Exists("hours") ? conf->GetValue("hours") : "";
	if ((itsCalculationMode == kCalculateBias || itsCalculationMode == kCalculateMAE) && hours.empty())
	{
		throw std::runtime_error(ClassName() + ": number of previous hours ('hours') for calculation not specified");
	}
	if (itsCalculationMode == kCalculateBias || itsCalculationMode == kCalculateMAE)
	{
		itsNumHours = stoi(hours);
	}

    // Producer for bias and mae calculation
	const string prod = conf->Exists("producer") ? conf->GetValue("producer") : "";
	if ((itsCalculationMode == kCalculateBias || itsCalculationMode == kCalculateMAE) && prod.empty())
	{
		throw std::runtime_error(ClassName() + ": bias calculation data producer ('producer') not defined");
	}

	if (itsCalculationMode == kCalculateBias || itsCalculationMode == kCalculateMAE)
	{
		itsProducer = prod;
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

	if (itsCalculationMode != kCalculateBlend)
	{
		const string analysisHour = conf->Exists("analysis_hour") ? conf->GetValue("analysis_hour") : "";
		if (analysisHour.empty())
		{
			throw std::runtime_error(ClassName() + ": analysis_hour not defined");
		}
		else
		{
			itsAnalysisHour = stoi(analysisHour);
		}
	}

	// Each parameter has to be processed with a separate process queue invocation.
	const string p = conf->Exists("param") ? conf->GetValue("param") : "";
	if (p.empty())
	{
		throw std::runtime_error(ClassName() + ": parameter not specified");
	}
	params.push_back(param(p));

	PrimaryDimension(kTimeDimension);
	SetParams(params);
	Start();
}

void blend::Calculate(shared_ptr<info> targetInfo, unsigned short threadIndex)
{
	auto f = GET_PLUGIN(fetcher);
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

static forecast_time MakeAnalysisFetchTime(const forecast_time& currentTime, int analysisHour)
{
	const int validHour = stoi(currentTime.ValidDateTime().String("%H"));

	forecast_time analysisFetchTime = currentTime;

	analysisFetchTime.ValidDateTime().Adjust(kHourResolution, -validHour); // set to 00
	analysisFetchTime.ValidDateTime().Adjust(kHourResolution, analysisHour);

	analysisFetchTime.OriginDateTime() = analysisFetchTime.ValidDateTime();

	return analysisFetchTime;
}

matrix<double> blend::CalculateBias(logger& log, shared_ptr<info> targetInfo, const forecast_time& calcTime,
                                    const blend_producer& blendProd)
{
	shared_ptr<plugin_configuration> cnf = make_shared<plugin_configuration>(*itsConfiguration);

	param currentParam = targetInfo->Param();
	forecast_time currentTime = targetInfo->Time();
	forecast_time analysisFetchTime = MakeAnalysisFetchTime(currentTime, itsAnalysisHour);

	HPTimeResolution currentRes = currentTime.StepResolution();

	info_t analysis =
	    FetchNoExcept(cnf, analysisFetchTime, currentRes, currentParam, LAPS, kLapsProd, kLapsGeom);
	if (!analysis)
	{
		log.Error("Analysis (LAPS) data not found");
		himan::Abort();
	}

	matrix<double> currentBias(targetInfo->Data().SizeX(), targetInfo->Data().SizeY(), targetInfo->Data().SizeZ(),
	                           MissingDouble());
	vector<double> forecast;

	forecast_time leadTime(calcTime);

	// MOS doesn't have hours 0, 1, 2. So we'll set this to missing. We don't want to do this with other models, since
	// in these cases it is certainly an error that needs to be looked at and fixed manually.
	const int validHour = stoi(currentTime.ValidDateTime().String("%H"));
	if (itsBlendProducer == MOS && validHour >= 0 && validHour <= 2)
	{
		try
		{
			info_t Info = FetchProd(cnf, leadTime, currentRes, currentParam, itsBlendProducer, kBlendRawProd);
			forecast = VEC(Info);
		}
		catch (HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				forecast = vector<double>(targetInfo->Data().Size(), MissingDouble());	
			}
			else
			{
				throw;
			}
		}
	}
	else
	{
		info_t Info = FetchProd(cnf, leadTime, currentRes, currentParam, itsBlendProducer, kBlendRawProd);
		forecast = VEC(Info);
	}

	// Previous forecast's bias corrected data is optional. If the data is not found we'll set the grid to missing.
	// (This happens, for example, during initialization.)
	forecast_time prevLeadTime(leadTime);
	prevLeadTime.OriginDateTime().Adjust(currentRes, -blendProd.originTimestep);

	vector<double> prevBias;
	try
	{
		info_t temp = FetchProd(cnf, prevLeadTime, currentRes, currentParam, itsBlendProducer, kBlendBiasProd);
		prevBias = VEC(temp);
	}
	catch (HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			prevBias = vector<double>(targetInfo->Data().Size(), MissingDouble());
		}
		else
		{
			throw;
		}
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
matrix<double> blend::CalculateMAE(logger& log, shared_ptr<info> targetInfo, const forecast_time& calcTime,
                                   const blend_producer& blendProd)
{
	shared_ptr<plugin_configuration> cnf = make_shared<plugin_configuration>(*itsConfiguration);

	const string geom = cnf->TargetGeomName();

	param currentParam = targetInfo->Param();
	forecast_time currentTime = targetInfo->Time();
	forecast_time analysisFetchTime = MakeAnalysisFetchTime(currentTime, itsAnalysisHour);


	HPTimeResolution currentRes = currentTime.StepResolution();

	info_t analysis =
	    FetchNoExcept(cnf, analysisFetchTime, currentRes, currentParam, LAPS, kLapsProd, kLapsGeom);
	if (!analysis)
	{
		log.Error("Analysis (LAPS) data not found");
		himan::Abort();
	}

	matrix<double> MAE(targetInfo->Data().SizeX(), targetInfo->Data().SizeY(), targetInfo->Data().SizeZ(),
	                   MissingDouble());

	forecast_time leadTime(calcTime);
	forecast_time prevLeadTime(leadTime);
	prevLeadTime.OriginDateTime().Adjust(currentRes, -blendProd.originTimestep);

	info_t bias = FetchProd(cnf, leadTime, currentRes, currentParam, blendProd, kBlendBiasProd);

	// See note pertaining to MOS at CalculateBias.
	vector<double> forecast;
	const int validHour = stoi(currentTime.ValidDateTime().String("%H"));
	if (itsBlendProducer == MOS && validHour >= 0 && validHour <= 2)
	{
		try
		{
			info_t Info = FetchProd(cnf, leadTime, currentRes, currentParam, blendProd, kBlendRawProd);
			forecast = VEC(Info);
		}
		catch (HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				forecast = vector<double>(targetInfo->Data().Size(), MissingDouble());
			}
			else
			{
				throw;
			}
		}
	}
	else
	{
		info_t Info = FetchProd(cnf, leadTime, currentRes, currentParam, blendProd, kBlendRawProd);
		forecast = VEC(Info);
	}

	vector<double> prevMAE;
	info_t prevMAE_Info =
	    FetchNoExcept(cnf, prevLeadTime, currentRes, currentParam, blendProd, kBlendWeightProd, geom);
	if (!prevMAE_Info)
	{
		prevMAE = vector<double>(targetInfo->Data().Size(), MissingDouble());
	}
	else
	{
		prevMAE = VEC(prevMAE_Info);
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

void blend::CalculateMember(shared_ptr<info> targetInfo, unsigned short threadIdx, blend_mode mode)
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

	raw_time latestOrigin = LatestOriginTimeForProducer(itsProducer);
	log.Info("Latest origin time for producer: " + latestOrigin.String());

	// Used for fetching raw model output, bias, and weight for models.

	if (itsBlendProducer == MOS || itsBlendProducer == ECMWF)
	{
		// ValidDateTime can be borked on ECMWF and MOS.
		const int validHour = stoi(current.OriginDateTime().String("%H"));
		if (validHour == 6 ||  validHour == 18)
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
	info_t Info = make_shared<info>(*targetInfo);
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
	Info->ForecastTypes(ftypes);
	Info->Create(targetInfo->Grid(), true);
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
			d = CalculateBias(log, targetInfo, ftime, itsBlendProducer);
		}
		else
		{
			d = CalculateMAE(log, targetInfo, ftime, itsBlendProducer);
		}

		if (Info->ForecastType(forecastType))
		{
			Info->Grid()->Data(d);
			WriteToFile(Info);
		}
		else
		{
			log.Error("Failed to set the correct output forecast type");
			himan::Abort();
		}

		Info->NextTime();

		ftime.OriginDateTime().Adjust(kHourResolution, originTimeStep);
	}
}

static std::vector<info_t> FetchRawGrids(shared_ptr<info> targetInfo, shared_ptr<plugin_configuration> cnf,
                                         unsigned short threadIdx)
{
	auto f = GET_PLUGIN(fetcher);
	auto log = logger("calculateBlend_FetchRawGrids#" + to_string(threadIdx));

	const forecast_time currentTime = targetInfo->Time();
	const HPTimeResolution currentResolution = currentTime.StepResolution();
	const param currentParam = targetInfo->Param();
	const string geom = cnf->TargetGeomName();

	// Fetch previous model runs raw fields for EC and MOS when we're calculating during the 06 and 18 cycles.
	forecast_time ecmosFetchTime  = currentTime;
	const int hour = stoi(ecmosFetchTime.OriginDateTime().String("%H"));
	if (hour == 6 || hour == 18)
	{
		ecmosFetchTime.OriginDateTime().Adjust(kHourResolution, -6);
	}

	info_t mosRaw = FetchNoExcept(cnf, ecmosFetchTime, currentResolution, currentParam, MOS, kBlendRawProd, geom);
	info_t ecRaw = FetchNoExcept(cnf, ecmosFetchTime, currentResolution, currentParam, ECMWF, kBlendRawProd, geom);
	info_t mepsRaw = FetchNoExcept(cnf, currentTime, currentResolution, currentParam, MEPS, kBlendRawProd, geom);
	info_t hirlamRaw = FetchNoExcept(cnf, currentTime, currentResolution, currentParam, HIRLAM, kBlendRawProd, geom);
	info_t gfsRaw = FetchNoExcept(cnf, currentTime, currentResolution, currentParam, GFS, kBlendRawProd, geom);

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

		search_options opts(ftime, parm, blendProd.lvl, prod, blendProd.type, cnf);
		const vector<string> files = r->Files(opts);
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
		Info = f->Fetch(cnf, ftime, blendProd.lvl, parm, blendProd.type, false);
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

std::vector<info_t> FetchBiasGrids(shared_ptr<info> targetInfo, shared_ptr<plugin_configuration> cnf,
                                   unsigned short threadIdx)
{
	auto f = GET_PLUGIN(fetcher);
	auto log = logger("calculateBlend_FetchBiasGrids#" + to_string(threadIdx));

	const forecast_time fetchTime = targetInfo->Time();
	const HPTimeResolution currentResolution = fetchTime.StepResolution();
	const param currentParam = targetInfo->Param();

	info_t mosBias =
	    FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, MOS, kBlendBiasProd);
	info_t ecBias =
	    FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, ECMWF, kBlendBiasProd);
	info_t mepsBias =
	    FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, MEPS, kBlendBiasProd);
	info_t hirlamBias =
	    FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, HIRLAM, kBlendBiasProd);
	info_t gfsBias =
	    FetchHistorical(log, cnf, fetchTime, currentResolution, currentParam, GFS, kBlendBiasProd);

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

std::vector<info_t> FetchMAEGrids(shared_ptr<info> targetInfo, shared_ptr<plugin_configuration> cnf,
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

void blend::CalculateBlend(shared_ptr<info> targetInfo, unsigned short threadIdx)
{
	auto f = GET_PLUGIN(fetcher);
	auto log = logger("calculateBlend#" + to_string(threadIdx));
	const string deviceType = "CPU";

	const forecast_time currentTime = targetInfo->Time();
	const param currentParam = targetInfo->Param();

	log.Info("Blending " + currentParam.Name() + " " + static_cast<string>(currentTime.ValidDateTime()));

	// NOTE: If one of the grids is missing, we should still put an empty grid there. Since all the stages assume that
	// F[i], B[i], W[i] are all of the same model! This means that the ordering is fixed for the return vector of
	// FetchRawGrids, FetchBiasGrids, FetchMAEGrids.
	vector<info_t> forecasts =
	    FetchRawGrids(targetInfo, make_shared<plugin_configuration>(*itsConfiguration), threadIdx);
	if (std::all_of(forecasts.begin(), forecasts.end(), [&](info_t i) { return i == nullptr; }))
	{
		log.Error("Failed to acquire any source data");
		himan::Abort();
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
	vector<info_t> weights = FetchMAEGrids(targetInfo, make_shared<plugin_configuration>(*itsConfiguration), threadIdx);
	if (std::all_of(weights.begin(), weights.end(), [&](info_t i) { return i == nullptr; }))
	{
		log.Error("Failed to acquire any MAE grids");
	}

	for (auto& w : weights)
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

		for (const auto& tup : zip_range(forecasts, weights, biases))
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

	WriteToFile(targetInfo);
}

void blend::WriteToFile(const info_t targetInfo, write_options writeOptions)
{
	auto aWriter = GET_PLUGIN(writer);

	aWriter->WriteOptions(writeOptions);
	auto tempInfo = make_shared<info>(*targetInfo);

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
void blend::SetupOutputForecastTimes(shared_ptr<info> Info, const raw_time& latestOrigin, const forecast_time& current,
                                     int maxStep, int originTimeStep)
{
	vector<forecast_time> ftimes;

	Info->ResetTime();

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

	Info->Times(ftimes);
}

}  // namespace plugin
}  // namespace himan
