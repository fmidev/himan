#include "blend.h"
#include "fetcher.h"
#include "plugin_factory.h"
#include "radon.h"
#include "writer.h"
#include <algorithm>
#include <mutex>
#include <numeric>
#include <thread>

//
// Forecast blender
//

namespace himan
{
namespace plugin
{
using namespace std;

// 'decaying factor' for bias and mae
const double alpha = 0.05;

const producer kBlendWeightProd(182, 86, 182, "BLENDW");
const producer kBlendRawProd(183, 86, 183, "BLENDR");
const producer kBlendBiasProd(184, 86, 184, "BLENDB");

const producer kObsProd(281, 86, 202, "SMARTMETNWC");
// Default
string kObsGeom = "SMARTMET2500";

// Each blend producer is composed of these original producers. We use forecast_types to distinguish them
// from each other, and this way we don't have to create bunch of extra producers.
const blend_producer OBS(forecast_type(kDeterministic), 0, 1);
const blend_producer MOS(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kMos)), 240, 12);
const blend_producer ECMWF(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kEcmwf)), 240, 12);
const blend_producer HIRLAM(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kHirlam)), 54, 12);
const blend_producer MEPS(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kMeps)), 66, 12);
const blend_producer GFS(forecast_type(kEpsPerturbation, static_cast<float>(blend_producer::kGfs)), 240, 12);

blend::blend() : itsCalculationMode(kCalculateNone), itsAnalysisTime(), itsBlendProducer()
{
	itsLogger = logger("blend");
}

// Create observation analysis time (ie. OBS analysis time) from analysis hour that's
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

// for logging purposes
std::string IdToName(size_t id)
{
	switch (id)
	{
		case 1:
			return "MOS";
		case 2:
			return "ECMWF";
		case 3:
			return "HIRLAM";
		case 4:
			return "MEPS";
		case 5:
			return "GFS";
		default:
			return "UNKNOWN";
	}
}

// Read the configuration and set plugin properties:
// - calculation mode (bias, mae, blend) that is to be dispatched in Calculate()
// - producer for bias and mae
// - analysis time (obs)
bool blend::ParseConfigurationOptions(const shared_ptr<const plugin_configuration>& conf)
{
	// backwards compatibility
	if (conf->GetValue("laps_geometry").empty() == false)
	{
		kObsGeom = conf->GetValue("laps_geometry");
	}

	if (conf->GetValue("obs_geometry").empty() == false)
	{
		kObsGeom = conf->GetValue("obs_geometry");
	}

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
		itsLogger.Fatal(fmt::format("Invalid blender 'mode' specified: '{}'", mode));
		himan::Abort();
	}

	// Producer for bias and mae calculation (itsProdFtype is only used with these modes)
	const string prod = conf->Exists("producer") ? conf->GetValue("producer") : "";
	if ((itsCalculationMode == kCalculateBias || itsCalculationMode == kCalculateMAE) && prod.empty())
	{
		itsLogger.Fatal("Bias calculation data producer ('producer') not defined");
		himan::Abort();
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

	if (itsCalculationMode != kCalculateBlend)
	{
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
	const param currentParam = targetInfo->Param();

	switch (itsCalculationMode)
	{
		case kCalculateBlend:
			CalculateBlend(targetInfo, threadIndex);
			break;
		case kCalculateMAE:
		case kCalculateBias:
			CalculateMember(targetInfo, threadIndex, itsCalculationMode);
			break;
		default:
			log.Error("Invalid calculation mode");
			himan::Abort();
	}
}

tuple<shared_ptr<info<double>>, shared_ptr<info<double>>, shared_ptr<info<double>>, shared_ptr<info<double>>>
blend::FetchMAEAndBiasSource(shared_ptr<info<double>>& targetInfo, const forecast_time& calcTime, blend_mode type) const
{
	const param& currentParam = targetInfo->Param();
	const forecast_time& currentTime = targetInfo->Time();
	const level& currentLevel = targetInfo->Level();

	shared_ptr<info<double>> analysis =
	    Fetch(itsAnalysisTime, currentLevel, currentParam, OBS.type, {kObsGeom}, kObsProd);

	if (!analysis)
	{
		return make_tuple(nullptr, nullptr, nullptr, nullptr);
	}

	// rawTime is the time from where we fetch the raw data
	// it must be always the *latest* raw data from either ahour 00 or 12
	// (matching ahour of calcTime)
	forecast_time rawTime(calcTime);

	ASSERT(rawTime.OriginDateTime().String("%H") == "00" || rawTime.OriginDateTime().String("%H") == "12");

	itsLogger.Debug("Fetching RAW");

	shared_ptr<info<double>> forecast = Fetch(rawTime, currentLevel, currentParam, itsBlendProducer.type);

	if (!forecast)
	{
		return make_tuple(nullptr, nullptr, nullptr, nullptr);
	}

	// Previous forecast's bias corrected data is optional. If the data is not found we'll set the grid to missing.
	// (This happens, for example, during initialization.)

	// previous time is always in relation to _current_ time, not "calcTime"
	forecast_time prevTime(currentTime);

	if (prevTime.OriginDateTime().String("%H") != calcTime.OriginDateTime().String("%H"))
	{
		if (prevTime.OriginDateTime().String("%H") == "00")
		{
			prevTime.OriginDateTime().Adjust(kHourResolution, 12);
			prevTime.ValidDateTime().Adjust(kHourResolution, 12);
		}
		else
		{
			prevTime.OriginDateTime().Adjust(kHourResolution, -12);
			prevTime.ValidDateTime().Adjust(kHourResolution, -12);
		}
	}

	// Adjust valid date time so that step is equivalent to step of calcTime

	prevTime.ValidDateTime().Adjust(kHourResolution,
	                                static_cast<int>(calcTime.Step().Hours() - currentTime.Step().Hours()));
	prevTime.OriginDateTime().Adjust(kHourResolution, -24);
	prevTime.ValidDateTime().Adjust(kHourResolution, -24);

	ASSERT(prevTime.Step() == calcTime.Step());

	if (type == kCalculateBias)
	{
		itsLogger.Debug("Fetching previous BIAS");

		ASSERT(prevTime.OriginDateTime().String("%H") == calcTime.OriginDateTime().String("%H"));
		shared_ptr<info<double>> prev = Fetch(prevTime, currentLevel, currentParam, itsBlendProducer.type,
		                                      {itsConfiguration->TargetGeomName()}, kBlendBiasProd);

		return make_tuple(analysis, forecast, prev, nullptr);
	}
	else if (type == kCalculateMAE)
	{
		itsLogger.Debug("Fetching previous MAE");
		shared_ptr<info<double>> prev = Fetch(prevTime, currentLevel, currentParam, itsBlendProducer.type,
		                                      {itsConfiguration->TargetGeomName()}, kBlendWeightProd);

		// Get latest BIAS
		prevTime.OriginDateTime().Adjust(kHourResolution, 24);
		prevTime.ValidDateTime().Adjust(kHourResolution, 24);

		itsLogger.Info("Fetching latest BIAS");
		shared_ptr<info<double>> bias = Fetch(prevTime, currentLevel, currentParam, itsBlendProducer.type,
		                                      {itsConfiguration->TargetGeomName()}, kBlendBiasProd);

		return make_tuple(analysis, forecast, prev, bias);
	}

	return make_tuple(nullptr, nullptr, nullptr, nullptr);
}

matrix<double> blend::CalculateBias(shared_ptr<info<double>> targetInfo, const forecast_time& calcTime)
{
	auto source = FetchMAEAndBiasSource(targetInfo, calcTime, kCalculateBias);

	auto analysis = get<0>(source);
	auto forecast = get<1>(source);
	auto prev = get<2>(source);

	if (!analysis || !forecast)
	{
		return matrix<double>();
	}

	const vector<double>& O = VEC(analysis);
	const vector<double>& F = VEC(forecast);

	vector<double> BC;

	if (!prev)
	{
		BC.resize(targetInfo->Data().Size(), MissingDouble());
	}
	else
	{
		BC = VEC(prev);
	}

	matrix<double> currentBias(targetInfo->Data().SizeX(), targetInfo->Data().SizeY(), 1, MissingDouble());

	vector<double>& B = currentBias.Values();

	for (size_t i = 0; i < B.size(); i++)
	{
		double f = F[i];
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
matrix<double> blend::CalculateMAE(shared_ptr<info<double>> targetInfo, const forecast_time& calcTime)
{
	auto source = FetchMAEAndBiasSource(targetInfo, calcTime, kCalculateMAE);

	auto analysis = get<0>(source);
	auto forecast = get<1>(source);
	auto prev = get<2>(source);
	auto bias = get<3>(source);

	if (!analysis || !forecast || !bias)
	{
		return matrix<double>();
	}

	const vector<double>& O = VEC(analysis);
	const vector<double>& F = VEC(forecast);
	const vector<double>& B = VEC(bias);

	vector<double> PM;

	if (!prev)
	{
		PM.resize(targetInfo->Data().Size(), MissingDouble());
	}
	else
	{
		PM = VEC(prev);
	}

	matrix<double> currentMAE(targetInfo->Data().SizeX(), targetInfo->Data().SizeY(), 1, MissingDouble());

	vector<double>& M = currentMAE.Values();

	for (size_t i = 0; i < M.size(); i++)
	{
		double o = O[i];
		double f = F[i];
		double b = B[i];
		double pm = PM[i];

		if (IsMissing(f) || IsMissing(o))
		{
			f = 0.0;
			o = 0.0;
		}

		if (IsMissing(pm))
		{
			pm = 0.0;
		}

		if (IsMissing(b))
		{
			b = 0.0;
		}

		const double bcf = f - b;
		M[i] = (1.0 - alpha) * pm + alpha * std::abs(bcf - o);
	}

	return currentMAE;
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

	const forecast_type forecastType = itsBlendProducer.type;
	const forecast_time current = targetInfo->Time();

	const raw_time originDateTime = current.OriginDateTime();

	// Start from the 'earliest' (set below) point in time, and proceed to current time.

	const int maxStep = itsBlendProducer.forecastLength;
	const int originTimeStep = itsBlendProducer.originTimestep;

	// Problem: targetInfo has information for the data that we want to fetch, but because of the convoluted way of
	// calculating everything, this doesn't match with the data we want to write out.
	// Solution: Create a new info and write that out.
	shared_ptr<info<double>> Info = make_shared<info<double>>(*targetInfo);
	vector<forecast_type> ftypes{itsBlendProducer.type};

	if (mode == kCalculateBias)
	{
		Info->Producer(kBlendBiasProd);
	}
	else
	{
		Info->Producer(kBlendWeightProd);
	}

	SetupOutputForecastTimes(Info, originDateTime, current, maxStep, originTimeStep);
	Info->Set<forecast_type>(ftypes);
	Info->Create(targetInfo->Base(), true);
	Info->First();

	for (Info->Reset<forecast_time>(); Info->Next<forecast_time>();)
	{
		auto ftime = Info->Value<forecast_time>();

		if (ftime.Step().Hours() < 0)
		{
			log.Trace("End of forecast, breaking");
			break;
		}

		log.Info(fmt::format("Calculating for member {} analysis_hour {} step {}", Info->ForecastType().Value(),
		                     ftime.OriginDateTime().String("%H"), static_cast<string>(ftime.Step())));

		if (ftime.OriginDateTime() > current.OriginDateTime() || ftime.OriginDateTime() > originDateTime)
		{
			break;
		}

		matrix<double> d;

		if (mode == kCalculateBias)
		{
			d = CalculateBias(targetInfo, ftime);
		}
		else
		{
			d = CalculateMAE(targetInfo, ftime);
		}

		if (d.Size() > 0)
		{
			if (Info->Find<forecast_type>(forecastType))
			{
				auto newI = make_shared<info<double>>(*Info);
				// Adjust origin date time so that it is from "today" with correct ahour
				newI->Time().OriginDateTime(originDateTime);

				long int offset = 0;

				const int latestH = std::stoi(originDateTime.String("%H"));
				const int currentH = std::stoi(ftime.OriginDateTime().String("%H"));

				if (latestH == 0 && currentH == 12)
				{
					newI->Time().OriginDateTime().Adjust(kHourResolution, 12);
					offset = -12;
				}
				else if (latestH == 12 && currentH == 0)
				{
					newI->Time().OriginDateTime().Adjust(kHourResolution, -12);
					offset = 12;
				}

				// Adjust valid date time so that step values remains the same
				newI->Time().ValidDateTime().Adjust(
				    kHourResolution, static_cast<int>(ftime.Step().Hours() - current.Step().Hours() - offset));

				ASSERT(originDateTime.String("%Y%m%d") == newI->Time().OriginDateTime().String("%Y%m%d"));
				ASSERT(ftime.OriginDateTime().String("%H") == newI->Time().OriginDateTime().String("%H"));

				auto b = newI->Base();
				b->data = std::move(d);

				WriteToFile(newI);
			}
			else
			{
				log.Error("Failed to set the correct output forecast type");
				himan::Abort();
			}
		}
	}

	// Remove grid so it won't be written: blend is manually writing out grids with a separate info.
	// Previously blend implemented Start() and prevented it from being written, but this is not
	// allowed now. Blend should also be fixed so that we don't need this workaround.

	auto b = targetInfo->Base();
	b->grid = nullptr;
	b->data.Clear();

	targetInfo->Base(b);
}

std::vector<shared_ptr<info<double>>> blend::FetchRawGrids(shared_ptr<info<double>> targetInfo,
                                                           unsigned short threadIdx) const
{
	auto log = logger("calculateBlend_FetchRawGrids#" + to_string(threadIdx));

	const forecast_time& currentTime = targetInfo->Time();
	const param& currentParam = targetInfo->Param();
	const level& currentLevel = targetInfo->Level();

	std::vector<forecast_type> types = {MOS.type, ECMWF.type, HIRLAM.type, MEPS.type, GFS.type};
	std::vector<shared_ptr<info<double>>> ret(5);

	for (size_t i = 0; i < types.size(); i++)
	{
		shared_ptr<info<double>> raw = Fetch(currentTime, currentLevel, currentParam, types[i]);

		ret[i] = raw;
	}

	for (size_t i = 0; i < ret.size(); i++)
	{
		log.Info(fmt::format("{} RAW missing {}", IdToName(i + 1),
		                     (ret[i]) ? to_string(ret[i]->Data().MissingCount()) : "completely"));
	}

	return ret;
}

std::vector<shared_ptr<info<double>>> blend::FetchMAEAndBiasGrids(shared_ptr<info<double>> targetInfo,
                                                                  unsigned short threadIdx, blend_mode type) const
{
	ASSERT(type == kCalculateMAE || type == kCalculateBias);

	const producer prod = (type == kCalculateMAE) ? kBlendWeightProd : kBlendBiasProd;
	const std::string typestr = (type == kCalculateMAE) ? "MAE" : "BIAS";

	logger log(fmt::format("calculateBlend_Fetch{}Grids#{}", typestr, threadIdx));

	std::vector<forecast_type> types = {MOS.type, ECMWF.type, HIRLAM.type, MEPS.type, GFS.type};
	std::vector<shared_ptr<info<double>>> ret(5);

	// try to fetch bias/mae fields from current day or day before that, ie
	// newest or second newest
	for (size_t i = 0; i < types.size(); i++)
	{
		auto time = targetInfo->Time();

		for (size_t j = 0; j < 2; j++)
		{
			auto info = Fetch(time, targetInfo->Level(), targetInfo->Param(), types[i],
			                  itsConfiguration->SourceGeomNames(), prod);

			if (!info)
			{
				time.OriginDateTime().Adjust(kHourResolution, -24);
				time.ValidDateTime().Adjust(kHourResolution, -24);
				continue;
			}

			ret[i] = info;
			break;
		}
	}

	for (size_t i = 0; i < ret.size(); i++)
	{
		log.Info(fmt::format("{} {} missing {}", IdToName(i + 1), typestr,
		                     (ret[i]) ? to_string(ret[i]->Data().MissingCount()) : "completely"));
	}

	return ret;
}

void blend::CalculateBlend(shared_ptr<info<double>> targetInfo, unsigned short threadIdx)
{
	auto log = logger("calculateBlend#" + to_string(threadIdx));

	const param currentParam = targetInfo->Param();

	// NOTE: If one of the grids is missing, we should still put an empty grid there. Since all the stages assume that
	// F[i], B[i], W[i] are all of the same model! This means that the ordering is fixed for the return vector of
	// FetchRawGrids, FetchBiasGrids, FetchMAEGrids.
	vector<shared_ptr<info<double>>> forecasts = FetchRawGrids(targetInfo, threadIdx);

	if (std::all_of(forecasts.begin(), forecasts.end(), [&](shared_ptr<info<double>> i) { return i == nullptr; }))
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
	vector<shared_ptr<info<double>>> biases = FetchMAEAndBiasGrids(targetInfo, threadIdx, kCalculateBias);
	if (std::all_of(biases.begin(), biases.end(), [&](shared_ptr<info<double>> i) { return i == nullptr; }))
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
	vector<shared_ptr<info<double>>> preweights = FetchMAEAndBiasGrids(targetInfo, threadIdx, kCalculateMAE);

	if (std::all_of(preweights.begin(), preweights.end(), [&](shared_ptr<info<double>> i) { return i == nullptr; }))
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

		const size_t li = targetInfo->LocationIndex();

		for (const auto& tup : zip_range(forecasts, preweights, biases))
		{
			shared_ptr<info<double>> f = tup.get<0>();
			shared_ptr<info<double>> w = tup.get<1>();
			shared_ptr<info<double>> b = tup.get<2>();

			if (!f)
			{
				forecastWarnings++;
				continue;
			}

			if (!w)
			{
				weightWarnings++;
				continue;
			}

			if (!b)
			{
				biasWarnings++;
				continue;
			}

			f->LocationIndex(li);
			w->LocationIndex(li);
			b->LocationIndex(li);

			const double v = f->Value();
			const double wv = w->Value();
			const double bb = b->Value();

			if (IsMissing(v) || IsMissing(wv) || IsMissing(bb) || wv == 0)
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
			const double sum = accumulate(collectedWeights.begin(), collectedWeights.end(), 0.0,
			                              [](double acc, double d) { return acc + 1.0 / d; });

			ASSERT(sum != 0);

			weights[i] = 1.0 / sum;
		}

		if (numMissing == forecasts.size() || weights.size() == 0)
		{
			F = MissingDouble();
		}
		else
		{
			const double sw = accumulate(weights.begin(), weights.end(), 0.0);

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

	log.Info(fmt::format("[CPU] Missing values: {}/{}", targetInfo->Data().MissingCount(), targetInfo->Data().Size()));
}

void blend::WriteToFile(const shared_ptr<info<double>> targetInfo, write_options writeOptions)
{
	auto aWriter = GET_PLUGIN(writer);

	if (itsCalculationMode == kCalculateMAE || itsCalculationMode == kCalculateBias)
	{
		// Add more precision as MAE/Bias fields are not actually what they claim
		// (for example T-K)
		writeOptions.precision = 5;
	}

	aWriter->WriteOptions(writeOptions);
	auto tempInfo = make_shared<info<double>>(*targetInfo);

	if (targetInfo->IsValidGrid() == false)
	{
		return;
	}

	aWriter->ToFile(tempInfo, itsConfiguration);

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

	int numHours = itsBlendProducer.forecastLength;

	// analysis hour must always be 0 or 12
	while (numHours % 12 != 0)
	{
		numHours += 1;
	}
	ftime.OriginDateTime().Adjust(kHourResolution, -numHours);

	while (ftime.Step().Hours() > maxStep)
	{
		ftime.OriginDateTime().Adjust(kHourResolution, 12);
	}

	for (int i = 0; i < numHours; i += originTimeStep)
	{
		ftimes.push_back(ftime);
		ftime.OriginDateTime().Adjust(kHourResolution, 12);
	}
	ftimes.push_back(ftime);
	Info->Set<forecast_time>(ftimes);
}

}  // namespace plugin
}  // namespace himan
