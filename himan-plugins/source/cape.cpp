/**
 * @file cape.cpp
 *
 */

#include "cape.h"
#include "logger.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "util.h"
#include <future>

#include "debug.h"
#include "fetcher.h"
#include "hitool.h"
#include "radon.h"
#include "writer.h"

#include "cape.cuh"
#include "lift.h"

using namespace std;
using namespace himan::plugin;
using namespace himan::numerical_functions;

typedef vector<vector<float>> vec2d;

// parameters are defined in cape.cuh

const himan::level SURFACE(himan::kHeight, 0);
const himan::level M500(himan::kHeightLayer, 500, 0);
const himan::level UNSTABLE(himan::kMaximumThetaE, 0);

void SmoothData(shared_ptr<himan::info<float>> myTargetInfo);
void CheckDataConsistency(vector<float>& LCLZ, vector<float>& LFCT, vector<float>& LFCP, vector<float>& LFCZ,
                          vector<float>& ELT, vector<float>& ELP, vector<float>& ELZ, vector<float>& LastELT,
                          vector<float>& LastELP, vector<float>& LastELZ, vector<float>& CAPE, vector<float>& CAPE1040,
                          vector<float>& CAPE3km, vector<float>& CIN);
void SetDataToInfo(shared_ptr<himan::info<float>> myTargetInfo, vector<float>& LCLT, vector<float>& LCLP,
                   vector<float>& LCLZ, vector<float>& LFCT, vector<float>& LFCP, vector<float>& LFCZ,
                   vector<float>& ELT, vector<float>& ELP, vector<float>& ELZ, vector<float>& LastELT,
                   vector<float>& LastELP, vector<float>& LastELZ, vector<float>& CAPE, vector<float>& CAPE1040,
                   vector<float>& CAPE3km, vector<float>& CIN);
void FinalizeData(shared_ptr<himan::info<float>> myTargetInfo);

vector<float> Sample(const vector<float>& x, const vector<float>& y, const vector<float>& samples)
{
	vector<float> ret(samples.size());

	for (size_t i = 0; i < samples.size(); i++)
	{
		const float& sample = samples[i];

		// const auto it = upper_bound(x.rbegin(), x.rend(), sample);
		const auto it = upper_bound(x.rbegin(), x.rend(), sample, [](float a, float b) { return a <= b; });

		if (it == x.rend())
		{
			continue;
		}

		const long dist = distance(x.begin(), it.base());
		const float x1 = *it;
		const float y1 = y[dist - 1];
		const float x2 = *(it - 1);
		const float y2 = y[dist];

		ret[i] = interpolation::Linear<float>(sample, x1, x2, y1, y2);
	}

	return ret;
}

vec2d Sample(const vec2d& x, const vec2d& y, const vec2d& samples)
{
	vec2d ret(x.size());

	for (size_t i = 0; i < x.size(); i++)
	{
		ret[i] = Sample(x[i], y[i], samples[i]);
	}

	return ret;
}

tuple<vec2d, vec2d, vec2d> GetSampledSourceData(shared_ptr<const himan::plugin_configuration> conf,
                                                shared_ptr<himan::info<float>> myTargetInfo, const vector<float>& P500m,
                                                const vector<float>& Psurface, const himan::level& startLevel,
                                                const himan::level& stopLevel)
{
	using namespace himan;

	// Pre-sample vertical data to make prosessing faster
	// We need temperature and relative humidity interpolated to 1hPa intervals
	// in the lowest 500 meters.

	const size_t N = myTargetInfo->SizeLocations();

	vec2d pressureProfile(N), temperatureProfile(N), humidityProfile(N);

	level curLevel = startLevel;

	const int levelCount = 1 + static_cast<int>(curLevel.Value() - stopLevel.Value());

	for (size_t i = 0; i < N; i++)
	{
		pressureProfile[i].resize(levelCount);
		temperatureProfile[i].resize(levelCount);
		humidityProfile[i].resize(levelCount);
	}

	auto f = GET_PLUGIN(fetcher);

	unsigned int k = 0;

	while (curLevel.Value() >= stopLevel.Value())
	{
		shared_ptr<info<float>> PInfo, TInfo, RHInfo;

		try
		{
			PInfo = f->Fetch<float>(conf, myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType(), false);
			TInfo = f->Fetch<float>(conf, myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType(), false);
			RHInfo = f->Fetch<float>(conf, myTargetInfo->Time(), curLevel, param("RH-PRCNT"),
			                         myTargetInfo->ForecastType(), false);
		}
		catch (const HPExceptionType& e)
		{
			break;
		}
		const auto P = VEC(PInfo);
		const auto T = VEC(TInfo);
		const auto RH = VEC(RHInfo);

		for (size_t i = 0; i < P.size(); i++)
		{
			pressureProfile[i][k] = P[i];
			temperatureProfile[i][k] = T[i];
			humidityProfile[i][k] = RH[i];
		}
		k++;
		curLevel.Value(curLevel.Value() - 1);
	}

	auto Psample = numerical_functions::Arange<float>(Psurface, P500m, -1);
	auto Tsample = Sample(pressureProfile, temperatureProfile, Psample);
	auto RHsample = Sample(pressureProfile, humidityProfile, Psample);

	ASSERT(Psample.size() == Tsample.size() && Tsample.size() == RHsample.size());
	ASSERT(Psample[0].size() == Tsample[0].size() && Tsample[0].size() == RHsample[0].size());

	return make_tuple(Psample, Tsample, RHsample);
}

vector<float> VirtualTemperature(vector<float> T, const vector<float>& P)
{
	for (size_t i = 0; i < T.size(); i++)
	{
		T[i] = himan::metutil::VirtualTemperature_<float>(T[i], P[i]);
		ASSERT(himan::IsMissing(T[i]) || (T[i] > 100 && T[i] < 400));
	}

	return T;
}

float Max(const vector<float>& vec)
{
	float ret = himan::MissingFloat();

	for (const float& val : vec)
	{
		ret = fmax(val, ret);
	}

	return ret;
}

void MultiplyWith(vector<float>& vec, float multiplier)
{
	for (float& val : vec)
	{
		val *= multiplier;
	}
}

template <typename T>
string PrintMean(const vector<T>& vec)
{
	T min = numeric_limits<T>::quiet_NaN(), max = numeric_limits<T>::quiet_NaN(), sum = 0;
	size_t count = 0, missing = 0;

	for (const T& val : vec)
	{
		if (himan::IsMissing(val))
		{
			missing++;
			continue;
		}

		min = fmin(val, min);
		max = fmax(val, max);
		count++;
		sum += val;
	}

	T mean = numeric_limits<T>::quiet_NaN();

	if (count > 0)
	{
		mean = sum / static_cast<float>(count);
	}

	string minstr = ::isnan(min) ? "nan" : to_string(static_cast<int>(min));
	string maxstr = ::isnan(max) ? "nan" : to_string(static_cast<int>(max));
	string meanstr = ::isnan(mean) ? "nan" : to_string(static_cast<int>(mean));

	return "min " + minstr + " max " + maxstr + " mean " + meanstr + " missing " + to_string(missing);
}

void MoistLift(const float* Piter, const float* Titer, const float* Penv, float* Tparcel, size_t size)
{
	// Split MoistLift (integration of a saturated air parcel upwards in atmosphere)
	// to several threads since it is very CPU intensive

	vector<future<void>> futures;

	size_t workers = 6;

	if (size % workers != 0)
	{
		workers = 4;
		if (size % workers != 0)
		{
			workers = 3;
			if (size % workers != 0)
			{
				workers = 1;
			}
		}
	}

	const size_t splitSize = static_cast<size_t>(floor(size / workers));

	for (size_t num = 0; num < workers; num++)
	{
		const size_t start = num * splitSize;
		futures.push_back(async(launch::async,
		                        [&](size_t _start) {
			                        for (size_t i = _start; i < _start + splitSize; i++)
			                        {
				                        Tparcel[i] = himan::metutil::MoistLiftA_<float>(Piter[i], Titer[i], Penv[i]);
			                        }
		                        },
		                        start));
	}

	for (auto& future : futures)
	{
		future.get();
	}
}

cape::cape() : itsBottomLevel(kHybrid, kHPMissingInt), itsUseVirtualTemperature(true)
{
	itsLogger = logger("cape");
}
void cape::Process(shared_ptr<const plugin_configuration> conf)
{
	compiled_plugin_base::Init(conf);

	auto r = GET_PLUGIN(radon);

	if (itsConfiguration->Exists("virtual_temperature"))
	{
		itsUseVirtualTemperature = util::ParseBoolean(itsConfiguration->GetValue("virtual_temperature"));
	}

	itsLogger.Info("Virtual temperature correction is " + string(itsUseVirtualTemperature ? "enabled" : "disabled"));

	itsBottomLevel = level(kHybrid, stoi(r->RadonDB().GetProducerMetaData(itsConfiguration->TargetProducer().Id(),
	                                                                      "last hybrid level number")));

#ifdef HAVE_CUDA
	cape_cuda::itsUseVirtualTemperature = itsUseVirtualTemperature;
	cape_cuda::itsBottomLevel = itsBottomLevel;
#endif

	vector<param> theParams;
	vector<string> sourceDatas;

	if (itsConfiguration->Exists("source_data"))
	{
		sourceDatas = itsConfiguration->GetValueList("source_data");
	}

	if (sourceDatas.size() == 0)
	{
		sourceDatas.push_back("surface");
		sourceDatas.push_back("500m mix");
		sourceDatas.push_back("most unstable");
	}

	theParams.push_back(LCLTParam);
	theParams.push_back(LCLPParam);
	theParams.push_back(LCLZParam);
	theParams.push_back(LFCTParam);
	theParams.push_back(LFCPParam);
	theParams.push_back(LFCZParam);
	theParams.push_back(ELTParam);
	theParams.push_back(ELPParam);
	theParams.push_back(ELZParam);
	theParams.push_back(LastELTParam);
	theParams.push_back(LastELPParam);
	theParams.push_back(LastELZParam);
	theParams.push_back(CAPEParam);
	theParams.push_back(CAPE1040Param);
	theParams.push_back(CAPE3kmParam);
	theParams.push_back(CINParam);

	// Discard the levels defined in json
	itsLevelIterator.Clear();

	for (const auto& source : sourceDatas)
	{
		if (source == "surface")
		{
			itsSourceLevels.push_back(SURFACE);
		}
		else if (source == "500m mix")
		{
			itsSourceLevels.push_back(M500);
		}
		else if (source == "most unstable")
		{
			itsSourceLevels.push_back(UNSTABLE);
			SetParams({LPLTParam, LPLPParam, LPLZParam}, {UNSTABLE});
		}
	}

	SetParams(theParams, itsSourceLevels);

	Start<float>();
}

void cape::MostUnstableCAPE(shared_ptr<info<float>> myTargetInfo, short threadIndex) const
{
	/*
	 * Most unstable CAPE is defined as the highest CAPE value found from a given
	 * location. The 'proper' way of calculating this then would be to calculate
	 * CAPE from all hybrid levels and select the highest. This is computationally
	 * too expensive.
	 *
	 * An approximation is to calculate theta e value for each hybrid level: most
	 * unstable CAPE is most often found from the level where the maximum theta e
	 * value is found.
	 *
	 * Unfortunately this is only an approximation, and it fails sometimes especially
	 * when maximum theta e is found high in the atmosphere in a relatively dry air
	 * and when there's a relatively warm and moist layer near surface.
	 *
	 * In order to alleviate this problem we do 'probing' for the most unstable CAPE
	 * value: instead of extracting a single theta e maximum value and process CAPE
	 * with that information, we calculate up to three local maxima from the theta e
	 * profile. Then we proceed to calculate CAPE separately for each three theta e
	 * maxima data, and in the end select the one that produces largest CAPE value.
	 */

	const size_t N = myTargetInfo->Data().Size();
	logger log("muCAPEThread#" + to_string(threadIndex));

	timer tmr(true);

	const int num = 3;  // number of potential source levels to consider

	auto source = GetNHighestThetaEValues(myTargetInfo, num);

	tmr.Stop();

	const auto& Ts = get<0>(source);
	const auto& TDs = get<1>(source);
	const auto& Ps = get<2>(source);

	if (Ts.empty())
	{
		return;
	}

	log.Debug("Potential source data created in " + to_string(tmr.GetTime()) + " ms");

	// LCLT, LCLP, LFCT, LFCP, ELT, ELP, ELZ, LastELT, LastELP, LastELZ, CAPE, CAPE1040, CAPE3km

	typedef tuple<vector<float>, vector<float>, vector<float>, vector<float>, vector<float>, vector<float>,
	              vector<float>, vector<float>, vector<float>, vector<float>, vector<float>, vector<float>,
	              vector<float>, vector<float>, vector<float>>
	    tuple15f;

	tmr.Start();

	vector<future<tuple15f>> futures;

	// This is where we store the results of the runs
	vector<tuple15f> results;

	// This is where we put thetae values so they are more convenient to read later on
	vec2d refValues(N);

	for (size_t taskIndex = 0; taskIndex < num; taskIndex++)
	{
		log.Debug("Launching async task #" + to_string(taskIndex));

		futures.push_back(async(
		    launch::async,
		    [&myTargetInfo, this](const cape_source& sourceValues, short threadId, size_t taskId) {
			    logger tasklog("muCAPEThread#" + to_string(threadId) + "asyncTask#" + to_string(taskId));

			    timer tasktmr(true);
			    auto LCL = GetLCL(sourceValues);
			    tasktmr.Stop();
			    tasklog.Debug("LCL in " + to_string(tasktmr.GetTime()) + "ms");

			    tasktmr.Start();
			    auto LFC = GetLFC(myTargetInfo, LCL.first, LCL.second);
			    tasktmr.Stop();
			    tasklog.Debug("LFC in " + to_string(tasktmr.GetTime()) + "ms");

			    const auto missingLFCcount =
			        count_if(LFC[0].first.begin(), LFC[0].first.end(), [](const float& f) { return IsMissing(f); });

			    if (LFC[0].first.empty() || static_cast<int>(LFC[0].first.size()) == missingLFCcount)
			    {
				    tasklog.Warning("LFC level not found");
				    return tuple15f();
			    }

			    tasktmr.Start();
			    auto CAPE = GetCAPE(myTargetInfo, LFC[0]);
			    tasktmr.Stop();
			    tasklog.Debug("CAPE in " + to_string(tasktmr.GetTime()) + "ms");

			    return tuple_cat(LCL, LFC[0], LFC[1], CAPE);

		    },
		    make_tuple(Ts[taskIndex], TDs[taskIndex], Ps[taskIndex]), threadIndex, taskIndex));
	}

	for (size_t i = 0; i < futures.size(); i++)
	{
		auto res = futures[i].get();

		if (get<0>(res).empty())
		{
			continue;
		}

		results.push_back(res);

		const auto& muCAPE = get<12>(res);

		for (size_t j = 0; j < N; j++)
		{
			refValues[j].push_back(muCAPE[j]);
		}
	}

	tmr.Stop();
	log.Debug("MUCape produced in " + to_string(tmr.GetTime()) + " ms");

	vector<float> LPLT(N), LPLP(N), LCLT(N), LCLP(N), LFCT(N), LFCP(N), LastLFCT(N), LastLFCP(N), ELT(N), ELP(N),
	    ELZ(N), LastELT(N), LastELP(N), LastELZ(N), CAPE(N), CAPE1040(N), CAPE3km(N);

	// cache trashing
	for (size_t i = 0; i < N; i++)
	{
		const auto& ref = refValues[i];

		// Find the position of the highest mucape
		auto pos = distance(ref.begin(), max_element(ref.begin(), ref.end()));

		LPLT[i] = Ts[pos][i];
		LPLP[i] = Ps[pos][i];
		LCLT[i] = get<0>(results[pos])[i];
		LCLP[i] = get<1>(results[pos])[i];
		LFCT[i] = get<2>(results[pos])[i];
		LFCP[i] = get<3>(results[pos])[i];
		LastLFCT[i] = get<4>(results[pos])[i];
		LastLFCP[i] = get<5>(results[pos])[i];
		ELT[i] = get<6>(results[pos])[i];
		ELP[i] = get<7>(results[pos])[i];
		ELZ[i] = get<8>(results[pos])[i];
		LastELT[i] = get<9>(results[pos])[i];
		LastELP[i] = get<10>(results[pos])[i];
		LastELZ[i] = get<11>(results[pos])[i];
		CAPE[i] = get<12>(results[pos])[i];
		CAPE1040[i] = get<13>(results[pos])[i];
		CAPE3km[i] = get<14>(results[pos])[i];
	}

	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	myTargetInfo->Find<param>(LPLTParam);
	myTargetInfo->Data().Set(LPLT);
	myTargetInfo->Find<param>(LPLPParam);
	myTargetInfo->Data().Set(LPLP);

	log.Debug("Fetching LFC height");
	auto LFCZ = h->VerticalValue<float>(ZParam, LFCP);
	auto LastLFCZ = h->VerticalValue<float>(ZParam, LastLFCP);

	log.Debug("Fetching LPL height");

	auto LPLZ = h->VerticalValue<float>(ZParam, LPLP);

	myTargetInfo->Find<param>(LPLZParam);
	myTargetInfo->Data().Set(LPLZ);

	log.Debug("Processing CIN");

	// Choose which LFC (first or last) to use for CIN processing
	// If LPL is above 500meters, use lowest LFC
	vector<float> CinLFCT(N), CinLFCP(N), CinLFCZ(N);

	for (size_t i = 0; i < N; i++)
	{
		if (LPLZ[i] < 500.)
		{
			CinLFCT[i] = LastLFCT[i];
			CinLFCP[i] = LastLFCP[i];
			CinLFCZ[i] = LastLFCZ[i];
		}
		else
		{
			CinLFCT[i] = LFCT[i];
			CinLFCP[i] = LFCP[i];
			CinLFCZ[i] = LFCZ[i];
		}
	}

	future<vector<float>> CINfut =
	    async(launch::async, &cape::GetCIN, this, myTargetInfo, LPLT, LPLP, LCLP, CinLFCP, CinLFCZ);

	log.Debug("Fetching LCL height");
	auto LCLZ = h->VerticalValue<float>(ZParam, LCLP);

	auto CIN = CINfut.get();
	CheckDataConsistency(LCLZ, LFCT, LFCP, LFCZ, ELT, ELP, ELZ, LastELT, LastELP, LastELZ, CAPE, CAPE1040, CAPE3km,
	                     CIN);
	SetDataToInfo(myTargetInfo, LCLT, LCLP, LCLZ, CinLFCT, CinLFCP, CinLFCZ, ELT, ELP, ELZ, LastELT, LastELP, LastELZ,
	              CAPE, CAPE1040, CAPE3km, CIN);
	SmoothData(myTargetInfo);
	FinalizeData(myTargetInfo);
}

void cape::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
{
	/*
	 * Algorithm:
	 *
	 * 1) Find suitable T and TD values of an air parcel
	 *
	 * 2) Lift this air parcel to LCL height dry adiabatically
	 *
	 * 3) Continue lifting the particle to LFC height moist adiabatically.
	 *
	 * 4) Integrate from LFC to EL to get CAPE
	 *
	 * 5) Integrate from source parcel height to LFC to get CIN
	 */

	const auto sourceLevel = myTargetInfo->Level();

	auto log =
	    logger("capeThread#" + to_string(threadIndex) + "Version" + to_string(static_cast<int>(sourceLevel.Type())));

	log.Info("Calculating source level type " + HPLevelTypeToString.at(sourceLevel.Type()) + " for time " +
	         static_cast<string>(myTargetInfo->Time().ValidDateTime()));

	// 1.

	timer aTimer;
	aTimer.Start();

	cape_source sourceValues;

	switch (sourceLevel.Type())
	{
		case kHeight:
			sourceValues = GetSurfaceValues(myTargetInfo);
			break;

		case kHeightLayer:
			sourceValues = Get500mMixingRatioValues(myTargetInfo);
			break;

		case kMaximumThetaE:
			return MostUnstableCAPE(myTargetInfo, threadIndex);
			break;

		default:
			throw runtime_error("Invalid source level: " + static_cast<string>(sourceLevel));
			break;
	}

	myTargetInfo->Find(sourceLevel);

	if (get<0>(sourceValues).empty())
	{
		return;
	}

	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	aTimer.Stop();

	log.Info("Source data calculated in " + to_string(aTimer.GetTime()) + " ms");

	log.Debug("Source temperature: " + ::PrintMean<float>(get<0>(sourceValues)));
	log.Debug("Source dewpoint: " + ::PrintMean<float>(get<1>(sourceValues)));
	log.Debug("Source pressure: " + ::PrintMean<float>(get<2>(sourceValues)));

	// 2.

	aTimer.Start();

	auto LCL = GetLCL(sourceValues);
	auto& LCLT = LCL.first;
	auto& LCLP = LCL.second;

	aTimer.Stop();

	log.Info("LCL calculated in " + to_string(aTimer.GetTime()) + " ms");

	log.Debug("LCL temperature: " + ::PrintMean<float>(LCL.first));
	log.Debug("LCL pressure: " + ::PrintMean<float>(LCL.second));

	auto LCLZ = h->VerticalValue<float>(ZParam, LCL.second);

	// 3.

	aTimer.Start();

	auto LFC = GetLFC(myTargetInfo, LCL.first, LCL.second);
	const auto& LFCT = LFC[0].first;
	const auto& LFCP = LFC[0].second;
	auto& LastLFCT = LFC[1].first;
	auto& LastLFCP = LFC[1].second;

	aTimer.Stop();
	log.Info("LFC calculated in " + to_string(aTimer.GetTime()) + " ms");

	const auto missingLFCcount = count_if(LFCT.begin(), LFCT.end(), [](const float& f) { return IsMissing(f); });

	if (LFCT.empty() || static_cast<int>(LFCT.size()) == missingLFCcount)
	{
		log.Warning("LFC level not found");
		return;
	}

	log.Debug("LFC temperature: " + ::PrintMean<float>(LFC[0].first));
	log.Debug("LFC pressure: " + ::PrintMean<float>(LFC[0].second));

	auto LFCZ = h->VerticalValue<float>(ZParam, LFCP);
	auto LastLFCZ = h->VerticalValue<float>(ZParam, LastLFCP);

	// 4. & 5.

	aTimer.Start();

	// CAPE is integrated from lowest LFC
	future<CAPEdata> CAPEfut = async(launch::async, &cape::GetCAPE, this, myTargetInfo, LFC[0]);

	// CIN is integrated to highest LFC, because surface&500m mix LPL is always below 650hPa
	future<vector<float>> CINfut = async(launch::async, &cape::GetCIN, this, myTargetInfo, get<0>(sourceValues),
	                                     get<2>(sourceValues), LCL.second, LastLFCP, LastLFCZ);

	auto CAPEresult = CAPEfut.get();
	auto CIN = CINfut.get();

	aTimer.Stop();

	log.Info("CAPE and CIN calculated in " + to_string(aTimer.GetTime()) + " ms");

	auto& ELT = get<0>(CAPEresult);
	auto& ELP = get<1>(CAPEresult);
	auto& ELZ = get<2>(CAPEresult);
	auto& LastELT = get<3>(CAPEresult);
	auto& LastELP = get<4>(CAPEresult);
	auto& LastELZ = get<5>(CAPEresult);
	auto& CAPE = get<6>(CAPEresult);
	auto& CAPE1040 = get<7>(CAPEresult);
	auto& CAPE3km = get<8>(CAPEresult);

	CheckDataConsistency(LCLZ, LastLFCT, LastLFCP, LastLFCZ, ELT, ELP, ELZ, LastELT, LastELP, LastELZ, CAPE, CAPE1040,
	                     CAPE3km, CIN);
	SetDataToInfo(myTargetInfo, LCLT, LCLP, LCLZ, LastLFCT, LastLFCP, LastLFCZ, ELT, ELP, ELZ, LastELT, LastELP,
	              LastELZ, CAPE, CAPE1040, CAPE3km, CIN);
	SmoothData(myTargetInfo);
	FinalizeData(myTargetInfo);
}

void SetDataToInfo(shared_ptr<himan::info<float>> myTargetInfo, vector<float>& LCLT, vector<float>& LCLP,
                   vector<float>& LCLZ, vector<float>& LFCT, vector<float>& LFCP, vector<float>& LFCZ,
                   vector<float>& ELT, vector<float>& ELP, vector<float>& ELZ, vector<float>& LastELT,
                   vector<float>& LastELP, vector<float>& LastELZ, vector<float>& CAPE, vector<float>& CAPE1040,
                   vector<float>& CAPE3km, vector<float>& CIN)
{
	using himan::param;

	myTargetInfo->Find<param>(LCLTParam);
	myTargetInfo->Data().Set(LCLT);

	myTargetInfo->Find<param>(LCLPParam);
	myTargetInfo->Data().Set(LCLP);

	myTargetInfo->Find<param>(LCLZParam);
	myTargetInfo->Data().Set(LCLZ);

	myTargetInfo->Find<param>(LFCTParam);
	myTargetInfo->Data().Set(LFCT);

	myTargetInfo->Find<param>(LFCPParam);
	myTargetInfo->Data().Set(LFCP);

	myTargetInfo->Find<param>(LFCZParam);
	myTargetInfo->Data().Set(LFCZ);

	myTargetInfo->Find<param>(ELTParam);
	myTargetInfo->Data().Set(ELT);

	myTargetInfo->Find<param>(ELPParam);
	myTargetInfo->Data().Set(ELP);

	myTargetInfo->Find<param>(ELZParam);
	myTargetInfo->Data().Set(ELZ);

	myTargetInfo->Find<param>(LastELTParam);
	myTargetInfo->Data().Set(LastELT);

	myTargetInfo->Find<param>(LastELPParam);
	myTargetInfo->Data().Set(LastELP);

	myTargetInfo->Find<param>(LastELZParam);
	myTargetInfo->Data().Set(LastELZ);

	myTargetInfo->Find<param>(CAPEParam);
	myTargetInfo->Data().Set(CAPE);

	myTargetInfo->Find<param>(CAPE1040Param);
	myTargetInfo->Data().Set(CAPE1040);

	myTargetInfo->Find<param>(CAPE3kmParam);
	myTargetInfo->Data().Set(CAPE3km);

	myTargetInfo->Find<param>(CINParam);
	myTargetInfo->Data().Set(CIN);
}

void SmoothData(shared_ptr<himan::info<float>> myTargetInfo)
{
	// Do smoothening for CAPE & CIN parameters
	himan::matrix<float> filter_kernel(3, 3, 1, himan::MissingFloat(), 1.0f / 9.0f);

	auto filter = [&](const himan::param& par) {

		myTargetInfo->Find<himan::param>(par);
		himan::matrix<float> filtered =
		    himan::numerical_functions::Filter2D<double>(myTargetInfo->Data(), filter_kernel);

		myTargetInfo->Base()->data = move(filtered);
	};

	filter(CAPEParam);
	filter(CAPE1040Param);
	filter(CAPE3kmParam);
	filter(CINParam);
}

void CheckDataConsistency(vector<float>& LCLZ, vector<float>& LFCT, vector<float>& LFCP, vector<float>& LFCZ,
                          vector<float>& ELT, vector<float>& ELP, vector<float>& ELZ, vector<float>& LastELT,
                          vector<float>& LastELP, vector<float>& LastELZ, vector<float>& CAPE, vector<float>& CAPE1040,
                          vector<float>& CAPE3km, vector<float>& CIN)
{
	// Check data physical consistency

	for (size_t i = 0; i < LFCZ.size(); i++)
	{
		// Unclosed CAPE areas are not good

		if (CAPE[i] == 0 && himan::IsMissing(ELZ[i]) && !himan::IsMissing(LFCZ[i]))
		{
			CIN[i] = 0;
			LFCZ[i] = himan::MissingFloat();
			LFCP[i] = himan::MissingFloat();
			LFCT[i] = himan::MissingFloat();
		}

		// Due to numeric inaccuracies sometimes LFC is slightly *below*
		// LCL. If the shift is small enough, consider them to be at
		// equal height.

		if ((LCLZ[i] - LFCZ[i]) > 0 && (LCLZ[i] - LFCZ[i]) < 0.1)
		{
			LFCZ[i] = LCLZ[i];
		}

		// The same with LFC & EL.

		if ((LFCZ[i] - ELZ[i]) > 0 && (LFCZ[i] - ELZ[i]) < 0.1)
		{
			ELZ[i] = LFCZ[i];
		}
	}

#ifdef DEBUG
	ASSERT(LFCZ.size() == ELZ.size());
	ASSERT(CAPE.size() == ELZ.size());
	ASSERT(CIN.size() == ELZ.size());

	for (size_t i = 0; i < LFCZ.size(); i++)
	{
		// Check:
		// * If LFC is missing, EL is missing
		// * If LFC is present, EL is present
		// * LFC must be above LCL or equal to it
		// * CAPE must be zero or positive real value
		// * CIN must be missing, zero or negative real value

		ASSERT((himan::IsMissing(LFCZ[i]) && himan::IsMissing(ELZ[i])) ||
		       (!himan::IsMissing(LFCZ[i]) && !himan::IsMissing(ELZ[i])));
		ASSERT(himan::IsMissing(LFCZ[i]) || (LFCZ[i] >= LCLZ[i] && !himan::IsMissing(LCLZ[i])));
		ASSERT(CAPE[i] >= 0);
		ASSERT(himan::IsMissing(CIN[i]) || CIN[i] <= 0);
	}
#endif
}

void FinalizeData(shared_ptr<himan::info<float>> myTargetInfo)
{
	// Do some final tuning for the data like changing zero values to missing
	// based on some arbitrary rules, ie. nothing physical.

	using himan::param;

	myTargetInfo->Find<param>(LFCTParam);
	auto& LFCT = VEC(myTargetInfo);

	myTargetInfo->Find<param>(LFCPParam);
	auto& LFCP = VEC(myTargetInfo);

	myTargetInfo->Find<param>(LFCZParam);
	auto& LFCZ = VEC(myTargetInfo);

	myTargetInfo->Find<param>(ELTParam);
	auto& ELT = VEC(myTargetInfo);

	myTargetInfo->Find<param>(ELPParam);
	auto& ELP = VEC(myTargetInfo);

	myTargetInfo->Find<param>(ELZParam);
	auto& ELZ = VEC(myTargetInfo);

	myTargetInfo->Find<param>(LastELTParam);
	auto& LastELT = VEC(myTargetInfo);

	myTargetInfo->Find<param>(LastELPParam);
	auto& LastELP = VEC(myTargetInfo);

	myTargetInfo->Find<param>(LastELZParam);
	auto& LastELZ = VEC(myTargetInfo);

	myTargetInfo->Find<param>(CAPEParam);
	auto& CAPE = VEC(myTargetInfo);

	myTargetInfo->Find<param>(CAPE1040Param);
	auto& CAPE1040 = VEC(myTargetInfo);

	myTargetInfo->Find<param>(CAPE3kmParam);
	auto& CAPE3km = VEC(myTargetInfo);

	myTargetInfo->Find<param>(CINParam);
	auto& CIN = VEC(myTargetInfo);

	for (size_t i = 0; i < LFCZ.size(); i++)
	{
		// HIMAN-224: CAPE & CIN values smaller than 0.1 are rounded to zero
		// STU-10036: If CAPE is zero set all level parameters and CIN to missing

		if (CAPE[i] < 0.1f)
		{
			CAPE[i] = 0.f;
			CAPE1040[i] = himan::IsMissing(CAPE1040[i]) ? himan::MissingFloat() : 0.f;
			CAPE3km[i] = himan::IsMissing(CAPE3km[i]) ? himan::MissingFloat() : 0.f;
			CIN[i] = himan::MissingFloat();
			LFCZ[i] = himan::MissingFloat();
			LFCP[i] = himan::MissingFloat();
			LFCT[i] = himan::MissingFloat();
			ELZ[i] = himan::MissingFloat();
			ELP[i] = himan::MissingFloat();
			ELT[i] = himan::MissingFloat();
			LastELZ[i] = himan::MissingFloat();
			LastELP[i] = himan::MissingFloat();
			LastELT[i] = himan::MissingFloat();
		}
	}
}

vector<float> cape::GetCIN(shared_ptr<info<float>> myTargetInfo, const vector<float>& Tsource,
                           const vector<float>& Psource, const vector<float>& PLCL, const vector<float>& PLFC,
                           const vector<float>& ZLFC) const
{
#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		return cape_cuda::GetCINGPU(itsConfiguration, myTargetInfo, Tsource, Psource, PLCL, PLFC, ZLFC);
	}
	else
#endif
	{
		return GetCINCPU(myTargetInfo, Tsource, Psource, PLCL, PLFC, ZLFC);
	}
}

vector<float> cape::GetCINCPU(shared_ptr<info<float>> myTargetInfo, const vector<float>& Tsource,
                              const vector<float>& Psource, const vector<float>& PLCL, const vector<float>& PLFC,
                              const vector<float>& ZLFC) const
{
	vector<bool> found(Tsource.size(), false);

	for (size_t i = 0; i < found.size(); i++)
	{
		if (IsMissing(PLFC[i]))
		{
			found[i] = true;
		}
	}

	forecast_time ftime = myTargetInfo->Time();
	forecast_type ftype = myTargetInfo->ForecastType();

	/*
	 * Modus operandi:
	 *
	 * 1. Integrate from source level to LCL dry adiabatically
	 *
	 * This can be done always since LCL is known at all grid points
	 * (that have source data values defined).
	 *
	 * 2. Integrate from LCL to LFC moist adiabatically
	 *
	 * Note! For some points integration will fail (no LFC found)
	 *
	 * We stop integrating at first time CAPE area is found!
	 */

	level curLevel = itsBottomLevel;

	auto prevZenvInfo = Fetch<float>(ftime, curLevel, ZParam, ftype, false);
	auto prevTenvInfo = Fetch<float>(ftime, curLevel, TParam, ftype, false);
	auto prevPenvInfo = Fetch<float>(ftime, curLevel, PParam, ftype, false);

	auto prevZenvVec = VEC(prevZenvInfo);
	auto prevTenvVec = VEC(prevTenvInfo);
	auto prevPenvVec = VEC(prevPenvInfo);

	vector<float> cinh(PLCL.size(), 0);

	size_t foundCount = count(found.begin(), found.end(), true);

	auto Piter = Psource;
	::MultiplyWith(Piter, 100);

	auto PLCLPa = PLCL;
	::MultiplyWith(PLCLPa, 100);

	auto Titer = Tsource;
	auto prevTparcelVec = Tsource;

	curLevel.Value(curLevel.Value() - 1);

	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 100.);

	while (curLevel.Value() > stopLevel.first.Value() && foundCount != found.size())
	{
		auto ZenvInfo = Fetch<float>(ftime, curLevel, ZParam, ftype, false);
		auto TenvInfo = Fetch<float>(ftime, curLevel, TParam, ftype, false);
		auto PenvInfo = Fetch<float>(ftime, curLevel, PParam, ftype, false);

		auto ZenvVec = VEC(ZenvInfo);
		auto TenvVec = VEC(TenvInfo);

		vector<float> TparcelVec(Piter.size());

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = VEC(PenvInfo);
		::MultiplyWith(PenvVec, 100);

		for (size_t i = 0; i < TparcelVec.size(); i++)
		{
			TparcelVec[i] = metutil::LiftLCLA_<float>(Piter[i], Titer[i], PLCLPa[i], PenvVec[i]);
		}

		int i = -1;

		auto& cinhref = cinh;

		for (auto&& tup : zip_range(cinhref, TenvVec, prevTenvVec, PenvVec, prevPenvVec, ZenvVec, prevZenvVec,
		                            TparcelVec, prevTparcelVec, Psource))
		{
			i++;

			if (found[i])
			{
				continue;
			}

			float& cin = tup.get<0>();

			float Tenv = tup.get<1>();  // K
			ASSERT(Tenv >= 100.);

			float prevTenv = tup.get<2>();

			float Penv = tup.get<3>() * 0.01f;  // hPa
			ASSERT(Penv < 1200.);

			float prevPenv = tup.get<4>() * 0.01f;

			float Zenv = tup.get<5>();      // m
			float prevZenv = tup.get<6>();  // m

			float Tparcel = tup.get<7>();  // K
			ASSERT(Tparcel >= 100. || IsMissing(Tparcel));

			float prevTparcel = tup.get<8>();  // K

			float Psrc = tup.get<9>();

			if (Penv > Psrc)
			{
				// Have not reached source level yet
				continue;
			}

			else if (Penv <= PLFC[i])
			{
				// reached max height

				found[i] = true;

				if (IsMissing(prevTparcel) || IsMissing(prevPenv) || IsMissing(prevTenv))
				{
					continue;
				}

				// Integrate the final piece from previous level to LFC level

				// First get LFC height in meters
				Zenv = interpolation::Linear<float>(PLFC[i], prevPenv, Penv, prevZenv, Zenv);

				// LFC environment temperature value
				Tenv = interpolation::Linear<float>(PLFC[i], prevPenv, Penv, prevTenv, Tenv);

				// LFC T parcel value
				Tparcel = interpolation::Linear<float>(PLFC[i], prevPenv, Penv, prevTparcel, Tparcel);

				Penv = PLFC[i];

				if (Zenv < prevZenv)
				{
					prevZenv = Zenv;
				}
			}

			if (IsMissing(Tparcel))
			{
				continue;
			}

			if (Penv < PLCL[i] && itsUseVirtualTemperature)
			{
				// Above LCL, switch to virtual temperature
				Tparcel = metutil::VirtualTemperature_<float>(Tparcel, Penv * 100);
				Tenv = metutil::VirtualTemperature_<float>(Tenv, Penv * 100);
			}

			cin += CAPE::CalcCIN(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

			ASSERT(cin <= 0);
		}

		foundCount = count(found.begin(), found.end(), true);

		itsLogger.Trace("CIN read for " + to_string(foundCount) + "/" + to_string(found.size()) + " gridpoints");

		curLevel.Value(curLevel.Value() - 1);

		prevZenvVec = ZenvVec;
		prevTenvVec = TenvVec;
		prevPenvVec = PenvVec;
		prevTparcelVec = TparcelVec;

		for (size_t j = 0; j < Titer.size(); j++)
		{
			if (found[j])
			{
				Titer[j] = MissingFloat();  // by setting this we prevent MoistLift to integrate particle
			}
		}
	}

	return cinh;
}

CAPEdata cape::GetCAPE(shared_ptr<info<float>> myTargetInfo, const pair<vector<float>, vector<float>>& LFC) const
{
#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		return cape_cuda::GetCAPEGPU(itsConfiguration, myTargetInfo, LFC.first, LFC.second);
	}
	else
#endif
	{
		return GetCAPECPU(myTargetInfo, LFC.first, LFC.second);
	}
}

CAPEdata cape::GetCAPECPU(shared_ptr<info<float>> myTargetInfo, const vector<float>& T, const vector<float>& P) const
{
	ASSERT(T.size() == P.size());

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	// Found count determines if we have calculated all three CAPE variation for a single grid point
	vector<unsigned char> found(T.size(), 0);

	vector<float> CAPE(T.size(), 0);
	vector<float> CAPE1040(T.size(), 0);
	vector<float> CAPE3km(T.size(), 0);
	vector<float> ELT(T.size(), MissingFloat());
	vector<float> ELP(T.size(), MissingFloat());
	vector<float> ELZ(T.size(), MissingFloat());
	vector<float> LastELT(T.size(), MissingFloat());
	vector<float> LastELP(T.size(), MissingFloat());
	vector<float> LastELZ(T.size(), MissingFloat());

	// Unlike LCL, LFC is *not* found for all grid points

	for (size_t i = 0; i < P.size(); i++)
	{
		if (IsMissing(P[i]))
		{
			found[i] = true;
		}
	}

	size_t foundCount = count(found.begin(), found.end(), true);

	// For each grid point find the hybrid level that's below LFC and then pick the lowest level
	// among all grid points

	const float maxP = ::Max(P);

	if (IsMissing(maxP))
	{
		throw runtime_error("CAPE: LFC pressure is missing");
	}
	auto levels = h->LevelForHeight(myTargetInfo->Producer(), maxP);

	level curLevel = levels.first;

	auto prevZenvInfo = Fetch<float>(myTargetInfo->Time(), curLevel, ZParam, myTargetInfo->ForecastType(), false);
	auto prevTenvInfo = Fetch<float>(myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType(), false);
	auto prevPenvInfo = Fetch<float>(myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType(), false);

	auto prevPenvVec = VEC(prevPenvInfo);
	auto prevZenvVec = VEC(prevZenvInfo);

	vector<float> prevTenvVec;

	if (itsUseVirtualTemperature)
	{
		::MultiplyWith(prevPenvVec, 100);
		prevTenvVec = VirtualTemperature(VEC(prevTenvInfo), prevPenvVec);
		::MultiplyWith(prevPenvVec, 0.01f);
	}
	else
	{
		prevTenvVec = VEC(prevTenvInfo);
	}

	curLevel.Value(curLevel.Value());

	auto Piter = P, Titer = T;  // integration variables, virtual correction already made
	vector<float> prevTparcelVec(Titer.size(), himan::MissingFloat());

	// Convert pressure to Pa since metutil-library expects that
	::MultiplyWith(Piter, 100);

	shared_ptr<info<float>> TenvInfo, PenvInfo, ZenvInfo;

	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 50.);

	while (curLevel.Value() > stopLevel.first.Value() && foundCount != found.size())
	{
		// Get environment temperature, pressure and height values for this level
		PenvInfo = Fetch<float>(myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType(), false);
		TenvInfo = Fetch<float>(myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType(), false);
		ZenvInfo = Fetch<float>(myTargetInfo->Time(), curLevel, ZParam, myTargetInfo->ForecastType(), false);

		if (!PenvInfo || !TenvInfo || !ZenvInfo)
		{
			break;
		}

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = VEC(PenvInfo);
		::MultiplyWith(PenvVec, 100);

		vector<float> TparcelVec(P.size());

		::MoistLift(&Piter[0], &Titer[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		vector<float> TenvVec;

		if (itsUseVirtualTemperature)
		{
			TenvVec = VirtualTemperature(VEC(TenvInfo), PenvVec);
		}
		else
		{
			TenvVec = VEC(TenvInfo);
		}

		::MultiplyWith(PenvVec, 0.01f);

		const auto ZenvVec = VEC(ZenvInfo);

		int i = -1;
		for (auto&& tup :
		     zip_range(TenvVec, PenvVec, ZenvVec, prevTenvVec, prevPenvVec, prevZenvVec, TparcelVec, prevTparcelVec))
		{
			i++;

			float Tenv = tup.get<0>();         // K
			float Penv = tup.get<1>();         // hPa
			float Zenv = tup.get<2>();         // m
			float prevTenv = tup.get<3>();     // K
			float prevPenv = tup.get<4>();     // hPa
			float prevZenv = tup.get<5>();     // m
			float Tparcel = tup.get<6>();      // K
			float prevTparcel = tup.get<7>();  // K

			if (found[i])
			{
				continue;
			}
			else if (IsMissing(Penv) || IsMissing(Tenv) || IsMissing(Zenv) || IsMissing(prevZenv) ||
			         IsMissing(Tparcel) || Penv > P[i])
			{
				// Missing data or current grid point is below LFC
				continue;
			}

			// When rising above LFC, get accurate value of Tenv at that level so that even small amounts of CAPE
			// (and EL!) values can be determined.

			if (IsMissing(prevTparcel) && !IsMissing(Tparcel))
			{
				prevTenv =
				    himan::numerical_functions::interpolation::Linear<float>(P[i], prevPenv, Penv, prevTenv, Tenv);
				prevZenv =
				    himan::numerical_functions::interpolation::Linear<float>(P[i], prevPenv, Penv, prevZenv, Zenv);
				prevPenv = P[i];     // LFC pressure
				prevTparcel = T[i];  // LFC temperature

				// If LFC was found close to lower hybrid level, the linear interpolation and moist lift will result
				// to same values. In this case CAPE integration fails as there is no area formed between
				// environment and parcel temperature. The result for this is that LFC is found but EL is not found.
				// To prevent this, warm the parcel value just slightly so that a miniscule CAPE area is formed and
				// EL is found.

				if (fabs(prevTparcel - prevTenv) < 0.0001f)
				{
					prevTparcel += 0.0001f;
				}
			}

			if (curLevel.Value() < 85 && (Tenv - Tparcel) > 25.)
			{
				// Temperature gap between environment and parcel too large --> abort search.
				// Only for values higher in the atmosphere, to avoid the effects of inversion

				found[i] = true;
				continue;
			}

			if (prevZenv < 3000.)
			{
				float C = CAPE::CalcCAPE3km(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

				CAPE3km[i] += C;

				ASSERT(CAPE3km[i] >= 0);
			}

			float C = CAPE::CalcCAPE1040(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

			CAPE1040[i] += C;

			ASSERT(CAPE1040[i] >= 0);

			float CAPEval, ELTval, ELPval, ELZval;

			CAPE::CalcCAPE(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, CAPEval, ELTval,
			               ELPval, ELZval);

			CAPE[i] += CAPEval;
			ASSERT(CAPEval >= 0.);

			if (!IsMissing(ELTval))
			{
				LastELT[i] = ELTval;
				LastELP[i] = ELPval;
				LastELZ[i] = ELZval;

				ELP[i] = fmaxf(ELP[i], LastELP[i]);
				ELZ[i] = fminf(ELZ[i], LastELZ[i]);

				if (IsMissing(ELT[i]))
				{
					ELT[i] = ELTval;
				}
			}
		}

		curLevel.Value(curLevel.Value() - 1);

		foundCount = count(found.begin(), found.end(), true);

		itsLogger.Trace("CAPE read for " + to_string(foundCount) + "/" + to_string(found.size()) + " gridpoints");
		prevZenvVec = ZenvVec;
		prevTenvVec = TenvVec;
		prevPenvVec = PenvVec;
		prevTparcelVec = TparcelVec;
	}

	// If the CAPE area is continued all the way to stopLevel and beyond, we don't have an EL for that
	// (since integration is forcefully stopped)
	// In this case let last level be EL

	for (size_t i = 0; i < CAPE.size(); i++)
	{
		if (CAPE[i] > 0 && IsMissing(ELT[i]))
		{
			ELT[i] = prevTenvVec[i];
			ELP[i] = prevPenvVec[i];
			ELZ[i] = prevZenvVec[i];

			LastELT[i] = ELT[i];
			LastELP[i] = ELP[i];
			LastELZ[i] = ELZ[i];
		}
	}

#ifdef DEBUG
	for (size_t i = 0; i < ELP.size(); i++)
	{
		ASSERT((IsMissing(ELP[i]) && IsMissing(LastELP[i])) || (ELP[i] >= LastELP[i]));
	}
#endif

	return make_tuple(ELT, ELP, ELZ, LastELT, LastELP, LastELZ, CAPE, CAPE1040, CAPE3km);
}

vector<pair<vector<float>, vector<float>>> cape::GetLFC(shared_ptr<info<float>> myTargetInfo, vector<float>& T,
                                                        vector<float>& P) const
{
	auto h = GET_PLUGIN(hitool);

	ASSERT(T.size() == P.size());

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	itsLogger.Trace("Searching environment temperature for starting pressure");

	vector<float> TenvLCL;

	try
	{
		TenvLCL = h->VerticalValue<float>(TParam, P);
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			return {make_pair(vector<float>(), vector<float>()), make_pair(vector<float>(), vector<float>())};
		}

		throw;
	}

	vector<float> _T, _Tenv;

	if (itsUseVirtualTemperature)
	{
		auto PP = P;
		::MultiplyWith(PP, 100);
		_T = VirtualTemperature(T, PP);
		_Tenv = VirtualTemperature(TenvLCL, PP);
	}
	else
	{
		_T = T;
		_Tenv = TenvLCL;
	}

#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		return cape_cuda::GetLFCGPU(itsConfiguration, myTargetInfo, _T, P, _Tenv);
	}
	else
#endif
	{
		return GetLFCCPU(myTargetInfo, _T, P, _Tenv);
	}
}

vector<pair<vector<float>, vector<float>>> cape::GetLFCCPU(shared_ptr<info<float>> myTargetInfo, vector<float>& T,
                                                           vector<float>& P, vector<float>& TenvLCL) const
{
	auto h = GET_PLUGIN(hitool);

	ASSERT(T.size() == P.size());

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	auto Piter = P, Titer = T;

	// Convert pressure to Pa since metutil-library expects that
	::MultiplyWith(Piter, 100);

	vector<bool> found(T.size(), false);

	vector<float> LFCT(T.size(), MissingFloat());
	vector<float> LFCP(T.size(), MissingFloat());
	vector<float> LastLFCT(T.size(), MissingFloat());
	vector<float> LastLFCP(T.size(), MissingFloat());

	for (size_t i = 0; i < TenvLCL.size(); i++)
	{
		// Require dry lifted parcel to be just a fraction higher
		// than environment to be accepted as LFC level.
		// This requirement is important later when CAPE integration
		// starts.

		if ((T[i] - TenvLCL[i]) > 0.0001)
		{
			LFCT[i] = T[i];
			LFCP[i] = P[i];
		}
	}

	// For each grid point find the hybrid level that's below LCL and then pick the lowest level
	// among all grid points; most commonly it's the lowest hybrid level

	const float maxP = ::Max(P);

	if (IsMissing(maxP))
	{
		throw runtime_error("LFC: LCL pressure is missing");
	}

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), maxP);
	level curLevel = levels.first;

	auto prevPenvInfo = Fetch<float>(myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType(), false);
	auto prevPenvVec = VEC(prevPenvInfo);

	auto prevTenvInfo = Fetch<float>(myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType(), false);

	vector<float> prevTenvVec;

	if (itsUseVirtualTemperature)
	{
		auto PP = prevPenvVec;
		::MultiplyWith(PP, 100);

		prevTenvVec = VirtualTemperature(VEC(prevTenvInfo), PP);
	}
	else
	{
		prevTenvVec = VEC(prevTenvInfo);
	}

	curLevel.Value(curLevel.Value() - 1);

	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 250.);
	auto hPa450 = h->LevelForHeight(myTargetInfo->Producer(), 450.);
	vector<float> prevTparcelVec(P.size(), MissingFloat());

	size_t foundCount = 0;

	while (curLevel.Value() > stopLevel.first.Value() && foundCount != found.size())
	{
		// Get environment temperature and pressure values for this level
		auto TenvInfo = Fetch<float>(myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType(), false);
		auto PenvInfo = Fetch<float>(myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType(), false);

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = VEC(PenvInfo);
		::MultiplyWith(PenvVec, 100);

		// Lift the particle from previous level to this level. In the first revolution
		// of this loop the starting level is LCL. If target level level is below current level
		// (ie. we would be lowering the particle) missing value is returned.

		vector<float> TparcelVec(P.size());

		::MoistLift(&Piter[0], &Titer[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		if (prevPenvInfo->Param().Name() == "P-PA")
		{
			::MultiplyWith(prevPenvVec, 0.01f);
		}

		vector<float> TenvVec;

		if (itsUseVirtualTemperature)
		{
			TenvVec = VirtualTemperature(VEC(TenvInfo), PenvVec);
		}
		else
		{
			TenvVec = VEC(TenvInfo);
		}

		::MultiplyWith(PenvVec, 0.01f);

		int i = -1;
		for (auto&& tup : zip_range(TenvVec, PenvVec, prevPenvVec, prevTenvVec, TparcelVec, prevTparcelVec, LFCT, LFCP,
		                            LastLFCT, LastLFCP))
		{
			i++;

			if (found[i])
			{
				continue;
			}

			float Tenv = tup.get<0>();  // K
			ASSERT(Tenv > 100.);

			float Penv = tup.get<1>();  // hPa
			ASSERT(Penv < 1200.);
			ASSERT(P[i] < 1200.);

			float prevPenv = tup.get<2>();  // hPa
			ASSERT(prevPenv < 1200.);

			float prevTenv = tup.get<3>();  // K
			ASSERT(prevTenv > 100.);

			float Tparcel = tup.get<4>();  // K
			ASSERT(Tparcel > 100. || IsMissing(Tparcel));

			float prevTparcel = tup.get<5>();  // K
			ASSERT(Tparcel > 100. || IsMissing(Tparcel));

			if (IsValid(tup.get<6>()) && Penv < 650.)
			{
				found[i] = true;
				continue;
			}

			float& Tresult = (IsMissing(tup.get<6>())) ? tup.get<6>() : tup.get<8>();
			float& Presult = (IsMissing(tup.get<7>())) ? tup.get<7>() : tup.get<9>();

			const float prevdiff = prevTparcel - prevTenv;
			const float diff = Tparcel - Tenv;
			const bool isFirstLFC = (diff >= 0 || fabs(diff) < 1e-4) && IsMissing(prevdiff) && IsMissing(tup.get<6>());
			const bool isLastLFC = (diff >= 0 || fabs(diff) < 1e-4) && (prevdiff < 0 || fabs(prevdiff) < 1e-4);

			if (isFirstLFC || isLastLFC)
			{
				// Parcel is now warmer than environment, we have found LFC and entering CAPE zone

				if (IsMissing(prevTparcel))
				{
					// Previous value is unknown: perhaps LFC is found very close to LCL?
					// Use LCL for previous value.
					prevTparcel = T[i];
				}

				if (diff < 0.01f)
				{
					// The passing of parcel to warmer side of sounding happened quite close
					// to current environment height, use the environment pressure without
					// any interpolation
					Tresult = Tparcel;
					Presult = Penv;
				}
				else if (prevdiff >= 0)
				{
					// Previous environment and parcel temperature are the same: perhaps because
					// we set it so earlier.
					Tresult = prevTparcel;
					Presult = prevPenv;
				}
				else
				{
					// Since Tparcel > Tenv, that means prevTenv > Tparcel > Ten
					// Use this information to linearly interpolate the pressure
					// where the crossing happened.

					auto intersection =
					    CAPE::GetPointOfIntersection(point(Tenv, Penv), point(prevTenv, prevPenv), point(Tparcel, Penv),
					                                 point(prevTparcel, prevPenv));
					Tresult = static_cast<float>(intersection.X());
					Presult = static_cast<float>(intersection.Y());

					if (Presult > prevPenv)
					{
						// Do not allow LFC to be below previous level
						Tresult = prevTparcel;
						Presult = prevPenv;
					}
					else if (IsMissing(Tresult))
					{
						// Intersection not found, use exact level value
						Tresult = Tparcel;
						Presult = Penv;
					}

					ASSERT((Presult <= prevPenv) && (Presult > Penv));
					ASSERT(Tresult > 100 && Tresult < 400);
				}

				ASSERT(!IsMissing(Tresult));
				ASSERT(!IsMissing(Presult));
			}
			else if (curLevel.Value() < hPa450.first.Value() && (Tenv - Tparcel) > 30.)
			{
				// Temperature gap between environment and parcel too large --> abort search.
				// Only for values higher in the atmosphere, to avoid the effects of inversion

				found[i] = true;
			}
		}

		curLevel.Value(curLevel.Value() - 1);

		foundCount = count(found.begin(), found.end(), true);
		itsLogger.Trace("LFC processed for " + to_string(foundCount) + "/" + to_string(found.size()) + " grid points");

		prevPenvVec = PenvVec;
		prevTenvVec = TenvVec;
		prevTparcelVec = TparcelVec;

		for (size_t j = 0; j < Titer.size(); j++)
		{
			if (found[j])
			{
				Titer[j] = MissingFloat();  // by setting this we prevent MoistLift to integrate particle
			}
		}
	}

	// If higher LFC is not found, set it to be the same as lower LFC. Also prevent that higher LFC is not actually
	// closer to ground than lower LFC.
	// This makes CIN integration setup easier later.

	for (size_t i = 0; i < LFCT.size(); i++)
	{
		if (IsMissing(LastLFCT[i]) || LastLFCP[i] > LFCP[i])
		{
			LastLFCT[i] = LFCT[i];
			LastLFCP[i] = LFCP[i];
		}
	}

	return {make_pair(LFCT, LFCP), make_pair(LastLFCT, LastLFCP)};
}

pair<vector<float>, vector<float>> cape::GetLCL(const cape_source& sourceValues) const
{
	vector<float> TLCL(get<0>(sourceValues).size(), MissingFloat());
	vector<float> PLCL = TLCL;

	// Need surface pressure

	for (auto&& tup : zip_range(get<0>(sourceValues), get<1>(sourceValues), get<2>(sourceValues), TLCL, PLCL))
	{
		float T = tup.get<0>();
		float TD = tup.get<1>();
		float P = tup.get<2>() * 100.f;  // Pa
		float& Tresult = tup.get<3>();
		float& Presult = tup.get<4>();

		auto lcl = metutil::LCLA_<float>(P, T, TD);

		Tresult = lcl.T;                              // K
		Presult = 0.01f * ((lcl.P > P) ? P : lcl.P);  // hPa
	}

	for (auto& val : PLCL)
	{
		val = fmaxf(val, 100.f);
	}

	return make_pair(TLCL, PLCL);
}

cape_source cape::GetSurfaceValues(shared_ptr<info<float>> myTargetInfo)
{
	/*
	 * 1. Get temperature and relative humidity from lowest hybrid level.
	 * 2. Calculate dewpoint
	 * 3. Return temperature and dewpoint
	 */

	auto TInfo = Fetch<float>(myTargetInfo->Time(), itsBottomLevel, TParam, myTargetInfo->ForecastType(), false);
	auto RHInfo =
	    Fetch<float>(myTargetInfo->Time(), itsBottomLevel, param("RH-PRCNT"), myTargetInfo->ForecastType(), false);
	auto PInfo = Fetch<float>(myTargetInfo->Time(), itsBottomLevel, PParam, myTargetInfo->ForecastType(), false);

	if (!TInfo || !RHInfo || !PInfo)
	{
		return make_tuple(vector<float>(), vector<float>(), vector<float>());
	}

	const auto& T = VEC(TInfo);
	const auto& RH = VEC(RHInfo);

	vector<float> TD(T.size(), MissingFloat());

	for (size_t i = 0; i < TD.size(); i++)
	{
		TD[i] = metutil::DewPointFromRH_<float>(T[i], RH[i]);
	}

	return make_tuple(T, TD, VEC(PInfo));
}

cape_source cape::Get500mMixingRatioValues(shared_ptr<info<float>> myTargetInfo)
{
/*
 * 1. Calculate potential temperature and mixing ratio for vertical profile
 *    0...500m for every 2 hPa
 * 2. Take an average from all values
 * 3. Calculate temperature from potential temperature, and dewpoint temperature
 *    from temperature and mixing ratio
 * 4. Return the two calculated values
 */

#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		return cape_cuda::Get500mMixingRatioValuesGPU(itsConfiguration, myTargetInfo);
	}
	else
#endif
	{
		return Get500mMixingRatioValuesCPU(myTargetInfo);
	}
}

cape_source cape::Get500mMixingRatioValuesCPU(shared_ptr<info<float>> myTargetInfo)
{
	modifier_mean tp, mr;
	level curLevel = itsBottomLevel;
	const size_t N = myTargetInfo->SizeLocations();

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());

	tp.HeightInMeters(false);
	mr.HeightInMeters(false);

	auto PInfo = Fetch<double>(myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType(), false);

	if (!PInfo)
	{
		return make_tuple(vector<float>(), vector<float>(), vector<float>());
	}
	else
	{
		// Himan specialty: empty data grid

		size_t miss = 0;
		for (auto& val : VEC(PInfo))
		{
			if (IsMissing(val))
			{
				miss++;
			}
		}

		if (PInfo->Data().MissingCount() == PInfo->Data().Size())
		{
			return make_tuple(vector<float>(), vector<float>(), vector<float>());
		}
	}

	auto curP = VEC(PInfo);

	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 500.);
	auto P500m = h->VerticalValue<double>(PParam, 500.);

	auto sourceData = GetSampledSourceData(itsConfiguration, myTargetInfo, util::Convert<double, float>(P500m),
	                                       util::Convert<double, float>(curP), curLevel, stopLevel.second);

	h->HeightUnit(kHPa);

	tp.LowerHeight(curP);
	mr.LowerHeight(curP);

	tp.UpperHeight(P500m);
	mr.UpperHeight(P500m);

	unsigned int k = 0;

	const auto& Psample = get<0>(sourceData);
	const auto& Tsample = get<1>(sourceData);
	const auto& RHsample = get<2>(sourceData);

	while (true)
	{
		vector<float> Tpot(N, MissingFloat());
		vector<float> MR(N, MissingFloat());
		vector<double> Pres(N, MissingDouble());

		for (size_t i = 0; i < N; i++)
		{
			if (k >= Psample[i].size())
			{
				continue;
			}

			const float& T = Tsample[i][k];
			const float& RH = RHsample[i][k];
			const float& P = Psample[i][k];

			if (IsMissing(T) || IsMissing(P) || IsMissing(RH))
			{
				continue;
			}

			ASSERT(T > 150 && T < 350);
			ASSERT(RH > 0 && RH < 102);
			ASSERT(P > 100);

			Tpot[i] = metutil::Theta_(T, 100 * P);
			MR[i] = metutil::smarttool::MixingRatio_(T, RH, 100 * P);
			Pres[i] = static_cast<double>(P);
		}

		if (static_cast<unsigned int>(
		        count_if(Pres.begin(), Pres.end(), [](const double& v) { return IsMissing(v); })) == Pres.size())
		{
			break;
		}

		tp.Process(util::Convert<float, double>(Tpot), Pres);
		mr.Process(util::Convert<float, double>(MR), Pres);

		k++;
	}

	auto Tpot = util::Convert<double, float>(tp.Result());
	auto MR = util::Convert<double, float>(mr.Result());

	auto PsurfInfo = Fetch<float>(myTargetInfo->Time(), itsBottomLevel, PParam, myTargetInfo->ForecastType(), false);
	auto PSurf = VEC(PsurfInfo);

	vector<float> T(Tpot.size());

	for (size_t i = 0; i < Tpot.size(); i++)
	{
		ASSERT((PSurf[i] > 100 && PSurf[i] < 1500) || IsMissing(PSurf[i]));
		if (!IsMissing(Tpot[i]) && !IsMissing(PSurf[i]))
		{
			T[i] = Tpot[i] * std::pow((PSurf[i] / 1000.f), 0.2854f);
		}
	}

	vector<float> TD(T.size(), MissingFloat());

	for (size_t i = 0; i < MR.size(); i++)
	{
		if (!IsMissing(T[i]) && !IsMissing(MR[i]) && !IsMissing(curP[i]))
		{
			const float Es = metutil::Es_<float>(T[i]);  // Saturated water vapor pressure
			const float E = metutil::E_<float>(MR[i], 100 * PSurf[i]);

			const float RH = fminf(102., E / Es * 100);

			TD[i] = metutil::DewPointFromRH_<float>(T[i], RH);
		}
	}

	return make_tuple(T, TD, PSurf);
}

cape_multi_source cape::GetNHighestThetaEValues(shared_ptr<info<float>> myTargetInfo, int n) const
{
#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		return cape_cuda::GetNHighestThetaEValuesGPU(itsConfiguration, myTargetInfo, n);
	}
	else
#endif
	{
		return GetNHighestThetaEValuesCPU(myTargetInfo, n);
	}
}

vector<float> Max1D(const vector<float>& v, size_t mask_len)
{
	ASSERT(mask_len % 2 == 1);
	ASSERT(!v.empty());

	vector<float> ret(v.size());

	const size_t half = mask_len / 2;

	// beginning

	for (size_t i = 0; i < half; i++)
	{
		float maxv = numeric_limits<float>::lowest();

		for (size_t j = 0; j <= half + i; j++)
		{
			maxv = fmaxf(maxv, v[j]);
		}
		ret[i] = maxv;
	}

	// center

	for (size_t i = half; i < v.size() - half; i++)
	{
		float maxv = numeric_limits<float>::lowest();

		for (size_t j = i - half; j <= i + half; j++)
		{
			maxv = fmaxf(maxv, v[j]);
		}
		ret[i] = maxv;
	}

	// end

	for (size_t i = v.size() - half; i < v.size(); i++)
	{
		float maxv = numeric_limits<float>::lowest();

		for (size_t j = i - half; j < v.size(); j++)
		{
			maxv = fmaxf(maxv, v[j]);
		}

		ret[i] = maxv;
	}

	return ret;
};

vector<pair<size_t, float>> MaximaLocation(const vector<float>& vals, const vector<float>& max)
{
	vector<pair<size_t, float>> ml;

	for (size_t i = 0; i < vals.size(); i++)
	{
		if (vals[i] == max[i])
		{
			ml.push_back(std::make_pair(i, vals[i]));
		}
	}

	// sort maxima values so that the biggest value of thetae comes first
	// (we want to make sure that if there are more than 3 maximas, we use the
	// three largest ones for further processing)

	sort(ml.begin(), ml.end(),
	     [](const pair<size_t, float>& a, const pair<size_t, float>& b) -> bool { return a.second > b.second; });

	// remove duplicates (sometimes two consecutive levels have the same theta e
	// value which also happens to be a maxima
	auto newEnd = unique(ml.begin(), ml.end(), [](const pair<size_t, float>& a, const pair<size_t, float>& b) {
		return a.second == b.second;
	});

	ml.erase(newEnd, ml.end());

	return ml;
}

cape_multi_source cape::GetNHighestThetaEValuesCPU(shared_ptr<info<float>> myTargetInfo, int n) const
{
	// Note: this function has not optimized for performance
	const size_t N = myTargetInfo->Data().Size();
	vector<bool> found(N, false);

	level curLevel = itsBottomLevel;

	vector<float> prevT, prevRH, prevP;
	vec2d ThetaEProfile, TProfile, TDProfile, PProfile;

	// 1. Create a vertical profile (for each grid point) that consists of
	// - temperature
	// - relative humidity
	// - theta e value

	while (true)
	{
		auto TInfo = Fetch<float>(myTargetInfo->Time(), curLevel, TParam, myTargetInfo->ForecastType(), false);
		auto RHInfo =
		    Fetch<float>(myTargetInfo->Time(), curLevel, param("RH-PRCNT"), myTargetInfo->ForecastType(), false);
		auto PInfo = Fetch<float>(myTargetInfo->Time(), curLevel, PParam, myTargetInfo->ForecastType(), false);

		if (!TInfo || !RHInfo || !PInfo)
		{
			return cape_multi_source();
		}

		int i = -1;

		auto curT = VEC(TInfo);
		auto curP = VEC(PInfo);
		const auto& curRH = VEC(RHInfo);

		vector<float> TD(N, MissingFloat()), ThetaE(N, MissingFloat());

		for (auto&& tup : zip_range(curT, curRH, curP))
		{
			i++;

			if (found[i])
			{
				continue;
			}

			float& T = tup.get<0>();
			float RH = tup.get<1>();
			float& P = tup.get<2>();

			if (IsMissing(P))
			{
				found[i] = true;
				continue;
			}

			if (P < mucape_search_limit)
			{
				found[i] = true;  // Make sure this is the last time we access this grid point

				if (prevP.empty())
				{
					// Lowest grid point located above search limit, hmm...
					continue;
				}

				// Linearly interpolate temperature and humidity values to search limit, to check
				// if highest theta e is found there

				T = interpolation::Linear<float>(mucape_search_limit, P, prevP[i], T, prevT[i]);
				RH = interpolation::Linear<float>(mucape_search_limit, P, prevP[i], RH, prevRH[i]);

				P = mucape_search_limit;
			}

			TD[i] = metutil::DewPointFromRH_(T, RH);
			ThetaE[i] = metutil::smarttool::ThetaE_<float>(T, RH, P * 100);
		}

		TProfile.push_back(curT);
		PProfile.push_back(curP);
		TDProfile.push_back(TD);
		ThetaEProfile.push_back(ThetaE);

		size_t foundCount = count(found.begin(), found.end(), true);

		if (foundCount == found.size())
		{
			break;
		}

		itsLogger.Trace("Max ThetaE processed for " + to_string(foundCount) + "/" + to_string(found.size()) +
		                " grid points");

		curLevel.Value(curLevel.Value() - 1);

		prevP = curP;
		prevT = curT;
		prevRH = curRH;
	}

	// 2. Find local maximas from thetae values
	// Select up to n local theta e maximas, and pick the temperature, dewpoint and
	// pressure values found from those levels.

	vec2d Tret(n), TDret(n), Pret(n);

	// Vector initialization is set to missing value, as not all grid points will have
	// three maximas
	for (size_t j = 0; j < static_cast<size_t>(n); j++)
	{
		Tret[j].resize(N, MissingFloat());
		TDret[j].resize(N, MissingFloat());
		Pret[j].resize(N, MissingFloat());
	}

	const size_t K = ThetaEProfile.size();

	for (size_t i = 0; i < N; i++)
	{
		// Collect data for single gridpoint
		vector<float> ThetaE(K);

		for (size_t k = 0; k < K; k++)
		{
			ThetaE[k] = ThetaEProfile[k][i];
		}

		const auto max = Max1D(ThetaE, 5);
		auto ml = MaximaLocation(ThetaE, max);

		// erase values above maxima search limit

		ml.erase(
		    remove_if(ml.begin(), ml.end(),
		              [&](const pair<size_t, float>& a) { return PProfile[a.first][i] < mucape_maxima_search_limit; }),
		    ml.end());

		// Copy values from max theta e levels for further processing

		for (size_t j = 0; j < min(static_cast<size_t>(n), ml.size()); j++)
		{
			Tret[j][i] = TProfile[ml[j].first][i];
			TDret[j][i] = TDProfile[ml[j].first][i];
			Pret[j][i] = PProfile[ml[j].first][i];

			ASSERT(IsValid(Tret[j][i]));
			ASSERT(IsValid(TDret[j][i]));
			ASSERT(IsValid(Pret[j][i]));
		}
	}

	return make_tuple(Tret, TDret, Pret);
}
