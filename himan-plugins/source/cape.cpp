/**
 * @file cape.cpp
 *
 */

#include "cape.h"
#include "logger.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "util.h"
#include <boost/thread.hpp>
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

extern mutex dimensionMutex;

// parameters are defined in cape.cuh

const himan::level SURFACE(himan::kHeight, 0);
const himan::level M500(himan::kHeightLayer, 500, 0);
const himan::level UNSTABLE(himan::kMaximumThetaE, 0);

vector<float> Convert(const vector<double>& arr)
{
	vector<float> ret(arr.size());
	copy(arr.begin(), arr.end(), ret.begin());

	replace_if(ret.begin(), ret.end(), [](const float& val) { return ::isnan(val); }, himan::MissingFloat());
	return ret;
}

vector<double> Convert(const vector<float>& arr)
{
	vector<double> ret(arr.size());
	copy(arr.begin(), arr.end(), ret.begin());

	replace_if(ret.begin(), ret.end(), [](const double& val) { return ::isnan(val); }, himan::MissingDouble());
	return ret;
}

vector<float> VirtualTemperature(vector<float> T, const std::vector<float>& P)
{
	for (size_t i = 0; i < T.size(); i++)
	{
		T[i] = himan::metutil::VirtualTemperature_<float>(T[i], P[i]);
		ASSERT(T[i] > 100 && T[i] < 400);
	}

	return T;
}

float Max(const vector<float>& vec)
{
	float ret = nanf("");

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
		                        [&](size_t start) {
			                        for (size_t i = start; i < start + splitSize; i++)
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
void cape::Process(std::shared_ptr<const plugin_configuration> conf)
{
	compiled_plugin_base::Init(conf);

	auto r = GET_PLUGIN(radon);

	if (itsConfiguration->Exists("virtual_temperature"))
	{
		itsUseVirtualTemperature = util::ParseBoolean(itsConfiguration->GetValue("virtual_temperature"));
	}

	itsLogger.Info("Virtual temperature correction is " + string(itsUseVirtualTemperature ? "enabled" : "disabled"));

	itsBottomLevel = level(kHybrid, stoi(r->RadonDB().GetProducerMetaData(itsConfiguration->SourceProducer().Id(),
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

	PrimaryDimension(kTimeDimension);
	// Discard the levels defined in json
	itsInfo->LevelIterator().Clear();

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

	Start();
}

void cape::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
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
	 * This is done by lifting the parcel a certain height at a time (500 Pa),
	 * and calculating the new temperature from the lifted height. If parcel
	 * temperature is larger than environment temperature, we have reached LFC.
	 * Environment temperature is therefore fetched every iteration of integration
	 * algorithm.
	 *
	 * 4) Integrate from LFC to EL
	 *
	 * 5) Integrate from surface to LFC to find CIN
	 */

	auto sourceLevel = myTargetInfo->Level();

	auto mySubThreadedLogger =
	    logger("capeThread#" + to_string(threadIndex) + "Version" + to_string(static_cast<int>(sourceLevel.Type())));

	mySubThreadedLogger.Info("Calculating source level type " + HPLevelTypeToString.at(sourceLevel.Type()) +
	                         " for time " + static_cast<string>(myTargetInfo->Time().ValidDateTime()));

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
			sourceValues = GetHighestThetaEValues(myTargetInfo);
			break;

		default:
			throw runtime_error("Invalid source level: " + static_cast<std::string>(sourceLevel));
			break;
	}

	myTargetInfo->Level(sourceLevel);

	if (get<0>(sourceValues).empty())
	{
		return;
	}

	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	if (sourceLevel.Type() == kMaximumThetaE)
	{
		myTargetInfo->Param(LPLTParam);
		myTargetInfo->Data().Set(Convert(get<0>(sourceValues)));
		myTargetInfo->Param(LPLPParam);
		myTargetInfo->Data().Set(Convert(get<2>(sourceValues)));

		auto height = h->VerticalValue(param("HL-M"), Convert(get<1>(sourceValues)));

		myTargetInfo->Param(LPLZParam);
		myTargetInfo->Data().Set(height);
	}

	aTimer.Stop();

	mySubThreadedLogger.Info("Source data calculated in " + to_string(aTimer.GetTime()) + " ms");

	mySubThreadedLogger.Debug("Source temperature: " + ::PrintMean<float>(get<0>(sourceValues)));
	mySubThreadedLogger.Debug("Source dewpoint: " + ::PrintMean<float>(get<1>(sourceValues)));
	mySubThreadedLogger.Debug("Source pressure: " + ::PrintMean<float>(get<2>(sourceValues)));

	// 2.

	aTimer.Start();

	auto LCL = GetLCL(myTargetInfo, sourceValues);

	aTimer.Stop();

	mySubThreadedLogger.Info("LCL calculated in " + to_string(aTimer.GetTime()) + " ms");

	mySubThreadedLogger.Debug("LCL temperature: " + ::PrintMean<float>(LCL.first));
	mySubThreadedLogger.Debug("LCL pressure: " + ::PrintMean<float>(LCL.second));

	myTargetInfo->Param(LCLTParam);
	myTargetInfo->Data().Set(Convert(LCL.first));

	myTargetInfo->Param(LCLPParam);
	myTargetInfo->Data().Set(Convert(LCL.second));

	auto LCLZ = h->VerticalValue(param("HL-M"), Convert(LCL.second));

	myTargetInfo->Param(LCLZParam);
	myTargetInfo->Data().Set(LCLZ);

	// 3.

	aTimer.Start();

	auto LFC = GetLFC(myTargetInfo, LCL.first, LCL.second);

	aTimer.Stop();

	mySubThreadedLogger.Info("LFC calculated in " + to_string(aTimer.GetTime()) + " ms");

	if (LFC.first.empty())
	{
		return;
	}

	mySubThreadedLogger.Debug("LFC temperature: " + ::PrintMean<float>(LFC.first));
	mySubThreadedLogger.Debug("LFC pressure: " + ::PrintMean<float>(LFC.second));

	myTargetInfo->Param(LFCTParam);
	myTargetInfo->Data().Set(Convert(LFC.first));

	myTargetInfo->Param(LFCPParam);
	myTargetInfo->Data().Set(Convert(LFC.second));

	auto LFCZ = h->VerticalValue(param("HL-M"), Convert(LFC.second));

	myTargetInfo->Param(LFCZParam);
	myTargetInfo->Data().Set(LFCZ);

	// 4. & 5.

	aTimer.Start();

	auto capeInfo = make_shared<info>(*myTargetInfo);
	boost::thread t1(&cape::GetCAPE, this, boost::ref(capeInfo), LFC);

	auto cinInfo = make_shared<info>(*myTargetInfo);
	boost::thread t2(&cape::GetCIN, this, boost::ref(cinInfo), get<0>(sourceValues), get<2>(sourceValues), LCL.first,
	                 LCL.second, Convert(LCLZ), LFC.second, Convert(LFCZ));

	t1.join();
	t2.join();

	aTimer.Stop();

	mySubThreadedLogger.Info("CAPE and CIN calculated in " + to_string(aTimer.GetTime()) + " ms");

	// Sometimes CAPE area is infinitely small -- so that CAPE is zero but LFC is found. In this case set all derivative
	// parameters missing.

	capeInfo->Param(LFCZParam);
	auto& lfcz_ = VEC(capeInfo);
	capeInfo->Param(LFCPParam);
	auto& lfcp_ = VEC(capeInfo);
	capeInfo->Param(LFCTParam);
	auto& lfct_ = VEC(capeInfo);
	cinInfo->Param(CINParam);
	auto& cin_ = VEC(cinInfo);
	capeInfo->Param(ELZParam);
	const auto& elz_ = VEC(capeInfo);
	capeInfo->Param(CAPEParam);
	const auto& cape_ = VEC(capeInfo);

	for (size_t i = 0; i < lfcz_.size(); i++)
	{
		if (cape_[i] == 0 && himan::IsMissing(elz_[i]) && !himan::IsMissing(lfcz_[i]))
		{
			cin_[i] = 0;
			lfcz_[i] = MissingDouble();
			lfcp_[i] = MissingDouble();
			lfct_[i] = MissingDouble();
		}
	}

#ifdef DEBUG
	ASSERT(lfcz_.size() == elz_.size());
	ASSERT(cape_.size() == elz_.size());
	ASSERT(cin_.size() == elz_.size());

	for (size_t i = 0; i < lfcz_.size(); i++)
	{
		// Check:
		// * If LFC is missing, EL is missing
		// * If LFC is present, EL is present
		// * If both are present, LFC must be below EL
		// * CAPE must be zero or positive real value
		// * CIN must be zero or negative real value
		ASSERT((IsMissing(lfcz_[i]) && IsMissing(elz_[i])) ||
		       (!IsMissing(lfcz_[i]) && !IsMissing(elz_[i]) && (lfcz_[i] < elz_[i])));
		ASSERT(cape_[i] >= 0);
		ASSERT(cin_[i] <= 0);
	}
#endif

	// Do smoothening for CAPE & CIN parameters
	mySubThreadedLogger.Trace("Smoothening");

	himan::matrix<double> filter_kernel(3, 3, 1, MissingDouble(), 1. / 9.);

	capeInfo->Param(CAPEParam);
	himan::matrix<double> filtered = numerical_functions::Filter2D(capeInfo->Data(), filter_kernel);
	capeInfo->Grid()->Data(filtered);

	capeInfo->Param(CAPE1040Param);
	filtered = numerical_functions::Filter2D(capeInfo->Data(), filter_kernel);
	capeInfo->Grid()->Data(filtered);

	capeInfo->Param(CAPE3kmParam);
	filtered = numerical_functions::Filter2D(capeInfo->Data(), filter_kernel);
	capeInfo->Grid()->Data(filtered);

	capeInfo->Param(CINParam);
	filtered = numerical_functions::Filter2D(capeInfo->Data(), filter_kernel);
	capeInfo->Grid()->Data(filtered);

	capeInfo->Param(CAPEParam);
	mySubThreadedLogger.Debug("CAPE: " + ::PrintMean<double>(VEC(capeInfo)));
	capeInfo->Param(CAPE1040Param);
	mySubThreadedLogger.Debug("CAPE1040: " + ::PrintMean<double>(VEC(capeInfo)));
	capeInfo->Param(CAPE3kmParam);
	mySubThreadedLogger.Debug("CAPE3km: " + ::PrintMean<double>(VEC(capeInfo)));
	cinInfo->Param(CINParam);
	mySubThreadedLogger.Debug("CIN: " + ::PrintMean<double>(VEC(cinInfo)));
}

void cape::GetCIN(shared_ptr<info> myTargetInfo, const vector<float>& Tsource, const vector<float>& Psource,
                  const vector<float>& TLCL, const vector<float>& PLCL, const vector<float>& ZLCL,
                  const vector<float>& PLFC, const vector<float>& ZLFC)
{
#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		cape_cuda::GetCINGPU(itsConfiguration, myTargetInfo, Tsource, Psource, TLCL, PLCL, ZLCL, PLFC, ZLFC);
	}
	else
#endif
	{
		GetCINCPU(myTargetInfo, Tsource, Psource, TLCL, PLCL, ZLCL, PLFC, ZLFC);
	}
}

void cape::GetCINCPU(shared_ptr<info> myTargetInfo, const vector<float>& Tsource, const vector<float>& Psource,
                     const vector<float>& TLCL, const vector<float>& PLCL, const vector<float>& ZLCL,
                     const vector<float>& PLFC, const vector<float>& ZLFC)
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

	auto prevZenvInfo = Fetch(ftime, curLevel, param("HL-M"), ftype, false);
	auto prevTenvInfo = Fetch(ftime, curLevel, param("T-K"), ftype, false);
	auto prevPenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);

	auto prevZenvVec = Convert(VEC(prevZenvInfo));
	auto prevTenvVec = Convert(VEC(prevTenvInfo));
	auto prevPenvVec = Convert(VEC(prevPenvInfo));

	std::vector<float> cinh(PLCL.size(), 0);

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

	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 200.);

	while (curLevel.Value() > stopLevel.first.Value() && foundCount != found.size())
	{
		auto ZenvInfo = Fetch(ftime, curLevel, param("HL-M"), ftype, false);
		auto TenvInfo = Fetch(ftime, curLevel, param("T-K"), ftype, false);
		auto PenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);

		auto ZenvVec = Convert(VEC(ZenvInfo));
		auto TenvVec = Convert(VEC(TenvInfo));

		vector<float> TparcelVec(Piter.size());

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = Convert(VEC(PenvInfo));
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

			float prevPenv = tup.get<4>();

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

		for (size_t i = 0; i < Titer.size(); i++)
		{
			if (!IsMissing(TparcelVec[i]) && !IsMissing(PenvVec[i]))
			{
				Titer[i] = TparcelVec[i];
				Piter[i] = PenvVec[i];
			}

			if (found[i])
			{
				Titer[i] = MissingFloat();  // by setting this we prevent MoistLift to integrate particle
			}
		}
	}

	myTargetInfo->Param(CINParam);
	myTargetInfo->Data().Set(Convert(cinh));
}

void cape::GetCAPE(shared_ptr<info> myTargetInfo, const pair<vector<float>, vector<float>>& LFC)
{
#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		cape_cuda::GetCAPEGPU(itsConfiguration, myTargetInfo, LFC.first, LFC.second);
	}
	else
#endif
	{
		GetCAPECPU(myTargetInfo, LFC.first, LFC.second);
	}
}

void cape::GetCAPECPU(shared_ptr<info> myTargetInfo, const vector<float>& T, const vector<float>& P)
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

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));

	level curLevel = levels.first;

	auto prevZenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);
	auto prevTenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
	auto prevPenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

	auto prevPenvVec = Convert(VEC(prevPenvInfo));
	auto prevZenvVec = Convert(VEC(prevZenvInfo));

	vector<float> prevTenvVec;

	if (itsUseVirtualTemperature)
	{
		::MultiplyWith(prevPenvVec, 100);
		prevTenvVec = VirtualTemperature(Convert(VEC(prevTenvInfo)), prevPenvVec);
		::MultiplyWith(prevPenvVec, 0.01f);
	}
	else
	{
		prevTenvVec = Convert(VEC(prevTenvInfo));
	}

	curLevel.Value(curLevel.Value());

	auto Piter = P, Titer = T;  // integration variables, virtual correction already made
	vector<float> prevTparcelVec(Titer.size(), himan::MissingFloat());

	// Convert pressure to Pa since metutil-library expects that
	::MultiplyWith(Piter, 100);

	info_t TenvInfo, PenvInfo, ZenvInfo;

	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 50.);

	while (curLevel.Value() > stopLevel.first.Value() && foundCount != found.size())
	{
		// Get environment temperature, pressure and height values for this level
		PenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
		TenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		ZenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);

		if (!PenvInfo || !TenvInfo || !ZenvInfo)
		{
			break;
		}

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = Convert(VEC(PenvInfo));
		::MultiplyWith(PenvVec, 100);

		vector<float> TparcelVec(P.size());

		::MoistLift(&Piter[0], &Titer[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		vector<float> TenvVec;

		if (itsUseVirtualTemperature)
		{
			TenvVec = VirtualTemperature(Convert(VEC(TenvInfo)), PenvVec);
		}
		else
		{
			TenvVec = Convert(VEC(TenvInfo));
		}

		const auto ZenvVec = Convert(VEC(ZenvInfo));

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
				// to same values. In this case CAPE integration fails as there is no area formed between environment
				// and parcel temperature. The result for this is that LFC is found but EL is not found. To prevent
				// this, warm the parcel value just slightly so that a miniscule CAPE area is formed and EL is found.

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
		prevZenvInfo = ZenvInfo;
		prevTenvVec = TenvVec;
		prevPenvInfo = PenvInfo;
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
	myTargetInfo->Param(ELTParam);
	myTargetInfo->Data().Set(Convert(ELT));

	myTargetInfo->Param(ELPParam);
	myTargetInfo->Data().Set(Convert(ELP));

	myTargetInfo->Param(ELZParam);
	myTargetInfo->Data().Set(Convert(ELZ));

	myTargetInfo->Param(LastELTParam);
	myTargetInfo->Data().Set(Convert(LastELT));

	myTargetInfo->Param(LastELPParam);
	myTargetInfo->Data().Set(Convert(LastELP));

	myTargetInfo->Param(LastELZParam);
	myTargetInfo->Data().Set(Convert(LastELZ));

	myTargetInfo->Param(CAPEParam);
	myTargetInfo->Data().Set(Convert(CAPE));

	myTargetInfo->Param(CAPE1040Param);
	myTargetInfo->Data().Set(Convert(CAPE1040));

	myTargetInfo->Param(CAPE3kmParam);
	myTargetInfo->Data().Set(Convert(CAPE3km));
}

pair<vector<float>, vector<float>> cape::GetLFC(shared_ptr<info> myTargetInfo, vector<float>& T, vector<float>& P)
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
		TenvLCL = Convert(h->VerticalValue(param("T-K"), Convert(P)));
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			return make_pair(vector<float>(), vector<float>());
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

pair<vector<float>, vector<float>> cape::GetLFCCPU(shared_ptr<info> myTargetInfo, vector<float>& T, vector<float>& P,
                                                   vector<float>& TenvLCL)
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

	for (size_t i = 0; i < TenvLCL.size(); i++)
	{
		// Require dry lifted parcel to be just a fraction higher
		// than environment to be accepted as LFC level.
		// This requirement is important later when CAPE integration
		// starts.

		if ((T[i] - TenvLCL[i]) > 0.001)
		{
			found[i] = true;
			LFCT[i] = T[i];
			LFCP[i] = P[i];
			Piter[i] = MissingFloat();
		}
	}

	size_t foundCount = count(found.begin(), found.end(), true);

	itsLogger.Debug("Found " + to_string(foundCount) + " gridpoints that have LCL=LFC");

	// For each grid point find the hybrid level that's below LCL and then pick the lowest level
	// among all grid points; most commonly it's the lowest hybrid level

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));
	level curLevel = levels.first;

	auto prevPenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
	auto prevPenvVec = Convert(VEC(prevPenvInfo));

	auto prevTenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);

	vector<float> prevTenvVec;

	if (itsUseVirtualTemperature)
	{
		auto PP = prevPenvVec;
		::MultiplyWith(PP, 100);

		prevTenvVec = VirtualTemperature(Convert(VEC(prevTenvInfo)), PP);
	}
	else
	{
		prevTenvVec = Convert(VEC(prevTenvInfo));
	}

	curLevel.Value(curLevel.Value() - 1);

	auto stopLevel = h->LevelForHeight(myTargetInfo->Producer(), 150.);
	auto hPa450 = h->LevelForHeight(myTargetInfo->Producer(), 450.);
	vector<float> prevTparcelVec(P.size(), MissingFloat());

	while (curLevel.Value() > stopLevel.first.Value() && foundCount != found.size())
	{
		// Get environment temperature and pressure values for this level
		auto TenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto PenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = Convert(VEC(PenvInfo));
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
			TenvVec = VirtualTemperature(Convert(VEC(TenvInfo)), PenvVec);
		}
		else
		{
			TenvVec = Convert(VEC(TenvInfo));
		}

		::MultiplyWith(PenvVec, 0.01f);

		int i = -1;
		for (auto&& tup : zip_range(TenvVec, PenvVec, prevPenvVec, prevTenvVec, TparcelVec, prevTparcelVec, LFCT, LFCP))
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

			float& Tresult = tup.get<6>();
			float& Presult = tup.get<7>();

			const float diff = Tparcel - Tenv;

			if (diff >= 0)
			{
				// Parcel is now warmer than environment, we have found LFC and entering CAPE zone

				found[i] = true;

				if (IsMissing(prevTparcel))
				{
					// Previous value is unknown: perhaps LFC is found very close to ground?
					// Use LCL for previous value.
					prevTparcel = T[i];
				}

				if (diff < 0.1f)
				{
					// The passing of parcel to warmer side of sounding happened quite close
					// to current environment height, use the environment pressure without
					// any interpolation
					Tresult = Tparcel;
					Presult = Penv;
				}
				else if (prevTparcel - prevTenv >= 0)
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

		for (size_t i = 0; i < Titer.size(); i++)
		{
			if (found[i])
			{
				Titer[i] = MissingFloat();  // by setting this we prevent MoistLift to integrate particle
			}
		}
	}

	return make_pair(LFCT, LFCP);
}

pair<vector<float>, vector<float>> cape::GetLCL(shared_ptr<info> myTargetInfo, const cape_source& sourceValues)
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
		val = fmaxf(val, 250.f);
	}

	return make_pair(TLCL, PLCL);
}

cape_source cape::GetSurfaceValues(shared_ptr<info> myTargetInfo)
{
	/*
	 * 1. Get temperature and relative humidity from lowest hybrid level.
	 * 2. Calculate dewpoint
	 * 3. Return temperature and dewpoint
	 */

	auto TInfo = Fetch(myTargetInfo->Time(), itsBottomLevel, param("T-K"), myTargetInfo->ForecastType(), false);
	auto RHInfo = Fetch(myTargetInfo->Time(), itsBottomLevel, param("RH-PRCNT"), myTargetInfo->ForecastType(), false);
	auto PInfo = Fetch(myTargetInfo->Time(), itsBottomLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

	if (!TInfo || !RHInfo || !PInfo)
	{
		return make_tuple(vector<float>(), vector<float>(), vector<float>());
	}

	const auto T = Convert(VEC(TInfo));
	const auto RH = Convert(VEC(RHInfo));

	vector<float> TD(T.size(), MissingFloat());

	for (size_t i = 0; i < TD.size(); i++)
	{
		TD[i] = metutil::DewPointFromRH_<float>(T[i], RH[i]);
	}

	return make_tuple(T, TD, Convert(VEC(PInfo)));
}

cape_source cape::Get500mMixingRatioValues(shared_ptr<info> myTargetInfo)
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

cape_source cape::Get500mMixingRatioValuesCPU(shared_ptr<info> myTargetInfo)
{
	modifier_mean tp, mr;
	level curLevel = itsBottomLevel;

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());

	tp.HeightInMeters(false);
	mr.HeightInMeters(false);

	auto PInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

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

	auto P = VEC(PInfo);

	auto P500m = h->VerticalValue(param("P-HPA"), 500.);

	h->HeightUnit(kHPa);

	tp.LowerHeight(P);
	mr.LowerHeight(P);

	tp.UpperHeight(P500m);
	mr.UpperHeight(P500m);

	vector<bool> found(myTargetInfo->Data().Size(), false);
	size_t foundCount = 0;

	while (foundCount != found.size())
	{
		auto T = h->VerticalValue(param("T-K"), P);
		auto RH = h->VerticalValue(param("RH-PRCNT"), P);

		vector<double> Tpot(T.size(), MissingDouble());
		vector<double> MR(T.size(), MissingDouble());

		for (size_t i = 0; i < T.size(); i++)
		{
			if (found[i] || IsMissingDouble(T[i]) || IsMissingDouble(P[i]) || IsMissingDouble(RH[i]))
			{
				continue;
			}
			ASSERT(T[i] > 150 && T[i] < 350);
			ASSERT(P[i] > 100 && P[i] < 1500);
			ASSERT(RH[i] > 0 && RH[i] < 102);

			Tpot[i] = metutil::Theta_<double>(T[i], 100 * P[i]);
			MR[i] = metutil::smarttool::MixingRatio_<double>(T[i], RH[i], 100 * P[i]);
		}

		tp.Process(Tpot, P);
		mr.Process(MR, P);

		foundCount = tp.HeightsCrossed();

		ASSERT(tp.HeightsCrossed() == mr.HeightsCrossed());

		itsLogger.Debug("Data read " + to_string(foundCount) + "/" + to_string(found.size()) + " gridpoints");

		for (size_t i = 0; i < found.size(); i++)
		{
			ASSERT((P[i] > 100 && P[i] < 1500) || IsMissingDouble(P[i]));

			if (found[i])
			{
				P[i] = MissingDouble();  // disable processing of this
			}
			else if (!IsMissingDouble(P[i]))
			{
				P[i] -= 2.0;
			}
		}
	}

	auto Tpot = Convert(tp.Result());
	auto MR = Convert(mr.Result());

	auto PsurfInfo = Fetch(myTargetInfo->Time(), itsBottomLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
	auto PSurf = Convert(VEC(PsurfInfo));

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
		if (!IsMissing(T[i]) && !IsMissing(MR[i]) && !IsMissing(P[i]))
		{
			float Es = metutil::Es_<float>(T[i]);  // Saturated water vapor pressure
			float E = metutil::E_<float>(MR[i], 100 * PSurf[i]);

			float RH = E / Es * 100;
			TD[i] = metutil::DewPointFromRH_<float>(T[i], RH);
		}
	}

	return make_tuple(T, TD, PSurf);
}

cape_source cape::GetHighestThetaEValues(shared_ptr<info> myTargetInfo)
{
/*
 * 1. Calculate equivalent potential temperature for all hybrid levels
 *    below 600hPa
 * 2. Take temperature and relative humidity from the level that had
 *    highest theta e
 * 3. Calculate dewpoint temperature from temperature and relative humidity.
 * 4. Return temperature and dewpoint
 */

#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		return cape_cuda::GetHighestThetaEValuesGPU(itsConfiguration, myTargetInfo);
	}
	else
#endif
	{
		return GetHighestThetaEValuesCPU(myTargetInfo);
	}
}

cape_source cape::GetHighestThetaEValuesCPU(shared_ptr<info> myTargetInfo)
{
	vector<bool> found(myTargetInfo->Data().Size(), false);

	vector<float> maxThetaE(myTargetInfo->Data().Size(), -1);
	vector<float> Ttheta(myTargetInfo->Data().Size(), MissingFloat());
	auto TDtheta = Ttheta;
	auto Ptheta = Ttheta;

	level curLevel = itsBottomLevel;

	vector<float> prevT, prevRH, prevP;

	while (true)
	{
		auto TInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto RHInfo = Fetch(myTargetInfo->Time(), curLevel, param("RH-PRCNT"), myTargetInfo->ForecastType(), false);
		auto PInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

		if (!TInfo || !RHInfo || !PInfo)
		{
			return make_tuple(vector<float>(), vector<float>(), vector<float>());
		}

		int i = -1;

		const auto curT = Convert(VEC(TInfo));
		const auto curP = Convert(VEC(PInfo));
		const auto curRH = Convert(VEC(RHInfo));

		for (auto&& tup : zip_range(curT, curRH, curP, maxThetaE, Ttheta, TDtheta, Ptheta))
		{
			i++;

			if (found[i])
			{
				continue;
			}

			float T = tup.get<0>();
			float RH = tup.get<1>();
			float P = tup.get<2>();
			float& refThetaE = tup.get<3>();
			float& Tresult = tup.get<4>();
			float& TDresult = tup.get<5>();
			float& Presult = tup.get<6>();

			if (IsMissing(P))
			{
				found[i] = true;
				continue;
			}

			if (P < 600.)
			{
				found[i] = true;  // Make sure this is the last time we access this grid point

				if (prevP.empty())
				{
					// Lowest grid point located above 600hPa, hmm...
					continue;
				}

				// Linearly interpolate temperature and humidity values to 600hPa, to check
				// if highest theta e is found there

				T = interpolation::Linear<float>(600., P, prevP[i], T, prevT[i]);
				RH = interpolation::Linear<float>(600., P, prevP[i], RH, prevRH[i]);

				P = 600.f;
			}

			const float TD = metutil::DewPointFromRH_(T, RH);
			const float ThetaE = metutil::smarttool::ThetaE_<float>(T, RH, P * 100);
			ASSERT(ThetaE >= 0);

			if ((ThetaE - refThetaE) > 0.0001)  // epsilon added for numerical stability
			{
				refThetaE = ThetaE;
				Tresult = T;
				TDresult = TD;
				Presult = P;

				ASSERT(TDresult > 100);
			}
		}

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

	return make_tuple(Ttheta, TDtheta, Ptheta);
}
