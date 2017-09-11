/**
 * @file cape.cpp
 *
 */

#include "cape.h"
#include "logger.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "util.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <future>

#include "fetcher.h"
#include "hitool.h"
#include "radon.h"

#include "cape.cuh"

const unsigned char FCAPE = (1 << 2);
const unsigned char FCAPE3km = (1 << 0);

using namespace std;
using namespace himan::plugin;
using namespace himan::numerical_functions;
using himan::IsMissingDouble;

#ifdef DEBUG
#define DumpVector(A, B) himan::util::DumpVector(A, B)
#else
#define DumpVector(A, B)
#endif

extern mutex dimensionMutex;

const himan::param LCLTParam("LCL-K", 4, 0, 0, 0);
const himan::param LCLPParam("LCL-HPA", 4720, 0, 3, 0);
const himan::param LCLZParam("LCL-M", 4726, 0, 3, 6);
const himan::param LFCTParam("LFC-K", 4, 0, 0, 0);
const himan::param LFCPParam("LFC-HPA", 4721, 0, 3, 0);
const himan::param LFCZParam("LFC-M", 4727, 0, 3, 6);
const himan::param ELTParam("EL-K", 4, 0, 0, 0);
const himan::param ELPParam("EL-HPA", 4722, 0, 3, 0);
const himan::param ELZParam("EL-M", 4728, 0, 3, 6);
const himan::param CAPEParam("CAPE-JKG", 4723, 0, 7, 6);
const himan::param CAPE1040Param("CAPE1040-JKG", 4729, 0, 7, 6);
const himan::param CAPE3kmParam("CAPE3KM-JKG", 4724, 0, 7, 6);
const himan::param CINParam("CIN-JKG", 4725, 0, 7, 7);

const himan::level SURFACE(himan::kHeight, 0);
const himan::level M500(himan::kHeightLayer, 500, 0);
const himan::level UNSTABLE(himan::kMaximumThetaE, 0);

double Max(const vector<double>& vec)
{
	double ret = -1e38;

	for (const double& val : vec)
	{
		if (!IsMissingDouble(val) && val > ret) ret = val;
	}

	if (ret == -1e38) ret = himan::MissingDouble();

	return ret;
}

void MultiplyWith(vector<double>& vec, double multiplier)
{
	for (double& val : vec)
	{
		if (!IsMissingDouble(val)) val *= multiplier;
	}
}

string PrintMean(const vector<double>& vec)
{
	double min = 1e38, max = -1e38, sum = 0;
	size_t count = 0, missing = 0;

	for (const double& val : vec)
	{
		if (IsMissingDouble(val))
		{
			missing++;
			continue;
		}

		min = (val < min) ? val : min;
		max = (val > max) ? val : max;
		count++;
		sum += val;
	}

	double mean = numeric_limits<double>::quiet_NaN();

	if (count > 0)
	{
		mean = sum / static_cast<double>(count);
	}

	string minstr = (min == 1e38) ? "nan" : to_string(static_cast<int>(min));
	string maxstr = (max == -1e38) ? "nan" : to_string(static_cast<int>(max));
	string meanstr = (mean != mean) ? "nan" : to_string(static_cast<int>(mean));

	return "min " + minstr + " max " + maxstr + " mean " + meanstr + " missing " + to_string(missing);
}

void MoistLift(const double* Piter, const double* Titer, const double* Penv, double* Tparcel, size_t size)
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
			                        himan::metutil::MoistLiftA(&Piter[start], &Titer[start], &Penv[start],
			                                                   &Tparcel[start], splitSize);
			                    },
		                        start));
	}

	for (auto& future : futures)
	{
		future.get();
	}
}

cape::cape() : itsBottomLevel(kHybrid, kHPMissingInt)
{
	itsLogger = logger("cape");
}

void cape::Process(std::shared_ptr<const plugin_configuration> conf)
{
	compiled_plugin_base::Init(conf);

	auto r = GET_PLUGIN(radon);

	itsBottomLevel = level(kHybrid, stoi(r->RadonDB().GetProducerMetaData(itsConfiguration->SourceProducer().Id(),
	                                                                      "last hybrid level number")));

#ifdef HAVE_CUDA
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
	theParams.push_back(CAPEParam);
	theParams.push_back(CAPE1040Param);
	theParams.push_back(CAPE3kmParam);
	theParams.push_back(CINParam);

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
		}
	}

	// disregard the level information provided by user

	itsConfiguration->Info()->Levels(itsSourceLevels);

	SetParams(theParams);

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

	auto mySubThreadedLogger = logger("siThread#" + boost::lexical_cast<string>(threadIndex) + "Version" +
	                                  boost::lexical_cast<string>(static_cast<int>(sourceLevel.Type())));

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

	if (get<0>(sourceValues).empty()) return;

	aTimer.Stop();

	mySubThreadedLogger.Info("Source data calculated in " + boost::lexical_cast<string>(aTimer.GetTime()) + " ms");

	mySubThreadedLogger.Debug("Source temperature: " + ::PrintMean(get<0>(sourceValues)));
	mySubThreadedLogger.Debug("Source dewpoint: " + ::PrintMean(get<1>(sourceValues)));
	mySubThreadedLogger.Debug("Source pressure: " + ::PrintMean(get<2>(sourceValues)));

	// 2.

	aTimer.Start();

	auto LCL = GetLCL(myTargetInfo, sourceValues);

	aTimer.Stop();

	mySubThreadedLogger.Info("LCL calculated in " + boost::lexical_cast<string>(aTimer.GetTime()) + " ms");

	mySubThreadedLogger.Debug("LCL temperature: " + ::PrintMean(LCL.first));
	mySubThreadedLogger.Debug("LCL pressure: " + ::PrintMean(LCL.second));

	myTargetInfo->Param(LCLTParam);
	myTargetInfo->Data().Set(LCL.first);

	myTargetInfo->Param(LCLPParam);
	myTargetInfo->Data().Set(LCL.second);

	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	auto height = h->VerticalValue(param("HL-M"), LCL.second);

	myTargetInfo->Param(LCLZParam);
	myTargetInfo->Data().Set(height);

	// 3.

	aTimer.Start();

	auto LFC = GetLFC(myTargetInfo, LCL.first, LCL.second);

	aTimer.Stop();

	mySubThreadedLogger.Info("LFC calculated in " + boost::lexical_cast<string>(aTimer.GetTime()) + " ms");

	if (LFC.first.empty())
	{
		return;
	}

	mySubThreadedLogger.Debug("LFC temperature: " + ::PrintMean(LFC.first));
	mySubThreadedLogger.Debug("LFC pressure: " + ::PrintMean(LFC.second));

	myTargetInfo->Param(LFCTParam);
	myTargetInfo->Data().Set(LFC.first);

	myTargetInfo->Param(LFCPParam);
	myTargetInfo->Data().Set(LFC.second);

	height = h->VerticalValue(param("HL-M"), LFC.second);

	myTargetInfo->Param(LFCZParam);
	myTargetInfo->Data().Set(height);

	// 4. & 5.

	aTimer.Start();

	auto capeInfo = make_shared<info>(*myTargetInfo);
	boost::thread t1(&cape::GetCAPE, this, boost::ref(capeInfo), LFC, ELTParam, ELPParam, ELZParam, CAPEParam,
	                 CAPE1040Param, CAPE3kmParam);

	auto cinInfo = make_shared<info>(*myTargetInfo);
	boost::thread t2(&cape::GetCIN, this, boost::ref(cinInfo), get<0>(sourceValues), get<2>(sourceValues), LCL.first,
	                 LCL.second, LFC.second, CINParam);

	t1.join();
	t2.join();

	aTimer.Stop();

	mySubThreadedLogger.Info("CAPE and CIN calculated in " + boost::lexical_cast<string>(aTimer.GetTime()) + " ms");

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
	auto elz_ = VEC(capeInfo);
	capeInfo->Param(CAPEParam);
	auto cape_ = VEC(capeInfo);

	for (size_t i = 0; i < lfcz_.size(); i++)
	{
		if (cape_[i] == 0 && IsMissingDouble(elz_[i]) && !IsMissingDouble(lfcz_[i]))
		{
			cin_[i] = 0;
			lfcz_[i] = MissingDouble();
			lfcp_[i] = MissingDouble();
			lfct_[i] = MissingDouble();
		}
	}

#ifdef DEBUG
	assert(lfcz_.size() == elz_.size());
	assert(cape_.size() == elz_.size());
	assert(cin_.size() == elz_.size());

	for (size_t i = 0; i < lfcz_.size(); i++)
	{
		// Check:
		// * If LFC is missing, EL is missing
		// * If LFC is present, EL is present
		// * If both are present, LFC must be below EL
		// * CAPE must be zero or positive real value
		// * CIN must be zero or negative real value
		assert((IsMissingDouble(lfcz_[i]) && IsMissingDouble(elz_[i])) ||
		       (!IsMissingDouble(lfcz_[i]) && !IsMissingDouble(elz_[i]) && (lfcz_[i] < elz_[i])));
		assert(cape_[i] >= 0);
		assert(cin_[i] <= 0);
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
	mySubThreadedLogger.Debug("CAPE: " + ::PrintMean(VEC(capeInfo)));
	capeInfo->Param(CAPE1040Param);
	mySubThreadedLogger.Debug("CAPE1040: " + ::PrintMean(VEC(capeInfo)));
	capeInfo->Param(CAPE3kmParam);
	mySubThreadedLogger.Debug("CAPE3km: " + ::PrintMean(VEC(capeInfo)));
	cinInfo->Param(CINParam);
	mySubThreadedLogger.Debug("CIN: " + ::PrintMean(VEC(cinInfo)));
}

void cape::GetCIN(shared_ptr<info> myTargetInfo, const vector<double>& Tsource, const vector<double>& Psource,
                  const vector<double>& TLCL, const vector<double>& PLCL, const vector<double>& PLFC, param CINParam)
{
#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		cape_cuda::GetCINGPU(itsConfiguration, myTargetInfo, Tsource, Psource, TLCL, PLCL, PLFC, CINParam);
	}
	else
#endif
	{
		GetCINCPU(myTargetInfo, Tsource, Psource, TLCL, PLCL, PLFC, CINParam);
	}
}

void cape::GetCINCPU(shared_ptr<info> myTargetInfo, const vector<double>& Tsource, const vector<double>& Psource,
                     const vector<double>& TLCL, const vector<double>& PLCL, const vector<double>& PLFC, param CINParam)
{
	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	vector<bool> found(Tsource.size(), false);

	for (size_t i = 0; i < found.size(); i++)
	{
		if (IsMissingDouble(PLFC[i]))
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

	// Get LCL and LFC heights in meters

	auto ZLCL = h->VerticalValue(param("HL-M"), PLCL);
	auto ZLFC = h->VerticalValue(param("HL-M"), PLFC);

	level curLevel = itsBottomLevel;

	auto prevZenvInfo = Fetch(ftime, curLevel, param("HL-M"), ftype, false);
	auto prevTenvInfo = Fetch(ftime, curLevel, param("T-K"), ftype, false);
	auto prevPenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);

	std::vector<double> cinh(PLCL.size(), 0);

	size_t foundCount = count(found.begin(), found.end(), true);

	auto Piter = Psource;
	::MultiplyWith(Piter, 100);

	auto PLCLPa = PLCL;
	::MultiplyWith(PLCLPa, 100);

	auto Titer = Tsource;
	auto prevTparcelVec = Tsource;

	curLevel.Value(curLevel.Value() - 1);

	auto hPa100 = h->LevelForHeight(myTargetInfo->Producer(), 100.);

	while (curLevel.Value() > hPa100.first.Value() && foundCount != found.size())
	{
		auto ZenvInfo = Fetch(ftime, curLevel, param("HL-M"), ftype, false);
		auto TenvInfo = Fetch(ftime, curLevel, param("T-K"), ftype, false);
		auto PenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);

		vector<double> TparcelVec(Piter.size(), MissingDouble());

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = PenvInfo->Data().Values();
		::MultiplyWith(PenvVec, 100);

		metutil::LiftLCL(&Piter[0], &Titer[0], &PLCLPa[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		int i = -1;

		auto& cinhref = cinh;

		for (auto&& tup : zip_range(cinhref, VEC(TenvInfo), VEC(prevTenvInfo), VEC(PenvInfo), VEC(prevPenvInfo),
		                            VEC(ZenvInfo), VEC(prevZenvInfo), TparcelVec, prevTparcelVec, Psource))
		{
			i++;

			if (found[i]) continue;

			double& cin = tup.get<0>();

			double Tenv = tup.get<1>();  // K
			assert(Tenv >= 100.);

			double prevTenv = tup.get<2>();

			double Penv = tup.get<3>();  // hPa
			assert(Penv < 1200.);

			double prevPenv = tup.get<4>();

			double Zenv = tup.get<5>();      // m
			double prevZenv = tup.get<6>();  // m

			double Tparcel = tup.get<7>();  // K
			assert(Tparcel >= 100. || IsMissingDouble(Tparcel));

			double prevTparcel = tup.get<8>();  // K

			double Psrc = tup.get<9>();

			if (Penv > Psrc)
			{
				// Have not reached source level yet
				continue;
			}
			else if (Penv <= PLFC[i])
			{
				// reached max height

				found[i] = true;

				if (IsMissingDouble(prevTparcel) || IsMissingDouble(prevPenv) || IsMissingDouble(prevTenv))
				{
					continue;
				}

				// Integrate the final piece from previous level to LFC level

				// First get LFC height in meters
				Zenv = himan::numerical_functions::interpolation::Linear(PLFC[i], prevPenv, Penv, prevZenv, Zenv);

				// LFC environment temperature value
				Tenv = himan::numerical_functions::interpolation::Linear(PLFC[i], prevPenv, Penv, prevTenv, Tenv);

				// LFC T parcel value
				Tparcel =
				    himan::numerical_functions::interpolation::Linear(PLFC[i], prevPenv, Penv, prevTparcel, Tparcel);

				Penv = PLFC[i];
				assert(Zenv > prevZenv);
			}

			if (IsMissingDouble(Tparcel))
			{
				continue;
			}

			if (Penv < PLCL[i])
			{
				// Above LCL, switch to virtual temperature
				Tparcel = metutil::VirtualTemperature_(Tparcel, Penv * 100);
				Tenv = metutil::VirtualTemperature_(Tenv, Penv * 100);
			}

			cin += CAPE::CalcCIN(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);
			assert(cin <= 0);
		}

		foundCount = count(found.begin(), found.end(), true);

		itsLogger.Trace("CIN read for " + boost::lexical_cast<string>(foundCount) + "/" +
		                boost::lexical_cast<string>(found.size()) + " gridpoints");

		curLevel.Value(curLevel.Value() - 1);

		prevZenvInfo = ZenvInfo;
		prevTenvInfo = TenvInfo;
		prevPenvInfo = PenvInfo;
		prevTparcelVec = TparcelVec;

		for (size_t i = 0; i < Titer.size(); i++)
		{
			if (!IsMissingDouble(TparcelVec[i]) && !IsMissingDouble(PenvVec[i]))
			{
				Titer[i] = TparcelVec[i];
				Piter[i] = PenvVec[i];
			}

			if (found[i]) Titer[i] = MissingDouble();  // by setting this we prevent MoistLift to integrate particle
		}
	}

	myTargetInfo->Param(CINParam);
	myTargetInfo->Data().Set(cinh);
}

void cape::GetCAPE(shared_ptr<info> myTargetInfo, const pair<vector<double>, vector<double>>& LFC, param ELTParam,
                   param ELPParam, param ELZParam, param CAPEParam, param CAPE1040Param, param CAPE3kmParam)
{
#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		cape_cuda::GetCAPEGPU(itsConfiguration, myTargetInfo, LFC.first, LFC.second, ELTParam, ELPParam, CAPEParam,
		                      CAPE1040Param, CAPE3kmParam);
	}
	else
#endif
	{
		GetCAPECPU(myTargetInfo, LFC.first, LFC.second, ELTParam, ELPParam, CAPEParam, CAPE1040Param, CAPE3kmParam);
	}

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	myTargetInfo->Param(ELPParam);
	auto height = h->VerticalValue(param("HL-M"), VEC(myTargetInfo));

	myTargetInfo->Param(ELZParam);
	myTargetInfo->Data().Set(height);
}

void cape::GetCAPECPU(shared_ptr<info> myTargetInfo, const vector<double>& T, const vector<double>& P, param ELTParam,
                      param ELPParam, param CAPEParam, param CAPE1040Param, param CAPE3kmParam)
{
	assert(T.size() == P.size());

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	// Found count determines if we have calculated all three CAPE variation for a single grid point
	vector<unsigned char> found(T.size(), 0);

	vector<double> CAPE(T.size(), 0);
	vector<double> CAPE1040(T.size(), 0);
	vector<double> CAPE3km(T.size(), 0);
	vector<double> ELT(T.size(), MissingDouble());
	vector<double> ELP(T.size(), MissingDouble());

	// Unlike LCL, LFC is *not* found for all grid points

	size_t foundCount = 0;

	for (size_t i = 0; i < P.size(); i++)
	{
		if (IsMissingDouble(P[i]))
		{
			found[i] |= FCAPE;
			foundCount++;
		}
	}

	// For each grid point find the hybrid level that's below LFC and then pick the lowest level
	// among all grid points

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));

	level curLevel = levels.first;

	auto prevZenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);
	auto prevTenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
	auto prevPenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

	curLevel.Value(curLevel.Value());

	auto Piter = P, Titer = T;  // integration variables
	auto prevTparcelVec = Titer;

	// Convert pressure to Pa since metutil-library expects that
	::MultiplyWith(Piter, 100);

	info_t TenvInfo, PenvInfo, ZenvInfo;

	auto hPa100 = h->LevelForHeight(myTargetInfo->Producer(), 100.);

	while (curLevel.Value() > hPa100.first.Value() && foundCount != found.size())
	{
		// Get environment temperature, pressure and height values for this level
		PenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
		TenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		ZenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = PenvInfo->Data().Values();
		::MultiplyWith(PenvVec, 100);

		vector<double> TparcelVec(P.size(), MissingDouble());

		::MoistLift(&Piter[0], &Titer[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		int i = -1;
		for (auto&& tup : zip_range(VEC(PenvInfo), VEC(ZenvInfo), VEC(prevZenvInfo), VEC(prevTenvInfo),
		                            VEC(prevPenvInfo), VEC(TenvInfo), TparcelVec, prevTparcelVec))
		{
			i++;

			double Tenv = tup.get<5>();  // K
			assert(Tenv > 100.);

			double prevTenv = tup.get<3>();  // K
			assert(prevTenv > 100.);

			double Penv = tup.get<0>();  // hPa
			assert(Penv < 1200.);

			double prevPenv = tup.get<4>();  // hPa
			assert(prevPenv < 1200.);

			double Zenv = tup.get<1>();      // m
			double prevZenv = tup.get<2>();  // m

			double Tparcel = tup.get<6>();  // K
			assert(Tparcel > 100.);

			double prevTparcel = tup.get<7>();  // K
			assert(prevTparcel > 100.);

			if (found[i] & FCAPE)
			{
				continue;
			}
			else if (IsMissingDouble(Penv) || IsMissingDouble(Tenv) || IsMissingDouble(Zenv) ||
			         IsMissingDouble(prevZenv) || IsMissingDouble(Tparcel) || Penv > P[i])
			{
				// Missing data or current grid point is below LFC
				continue;
			}

			// When rising above LFC, get accurate value of Tenv at that level so that even small amounts of CAPE
			// (and EL!) values can be determined.

			if (IsMissingDouble(prevTparcel) && !IsMissingDouble(Tparcel))
			{
				prevTenv = himan::numerical_functions::interpolation::Linear(P[i], prevPenv, Penv, prevTenv, Tenv);
				prevZenv = himan::numerical_functions::interpolation::Linear(P[i], prevPenv, Penv, prevZenv, Zenv);
				prevPenv = P[i];     // LFC pressure
				prevTparcel = T[i];  // LFC temperature

				// If LFC was found close to lower hybrid level, the linear interpolation and moist lift will result
				// to same values. In this case CAPE integration fails as there is no area formed between environment
				// and parcel temperature. The result for this is that LFC is found but EL is not found. To prevent
				// this, warm the parcel value just slightly so that a miniscule CAPE area is formed and EL is found.

				if (fabs(prevTparcel - prevTenv) < 0.0001)
				{
					prevTparcel += 0.0001;
				}
			}

			if (curLevel.Value() < 85 && (Tenv - Tparcel) > 25.)
			{
				// Temperature gap between environment and parcel too large --> abort search.
				// Only for values higher in the atmosphere, to avoid the effects of inversion

				found[i] |= FCAPE;
				continue;
			}

			if (prevZenv >= 3000. && Zenv >= 3000.)
			{
				found[i] |= FCAPE3km;
			}

			if ((found[i] & FCAPE3km) == 0)
			{
				double C = CAPE::CalcCAPE3km(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

				CAPE3km[i] += C;

				assert(CAPE3km[i] < 3000.);  // 3000J/kg, not 3000m
				assert(CAPE3km[i] >= 0);
			}

			double C = CAPE::CalcCAPE1040(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

			CAPE1040[i] += C;

			assert(CAPE1040[i] < 5000.);
			assert(CAPE1040[i] >= 0);

			double CAPEval, ELTval, ELPval;

			CAPE::CalcCAPE(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, CAPEval, ELTval,
			               ELPval);

			CAPE[i] += CAPEval;

			assert(CAPEval >= 0.);
			assert(CAPE[i] < 8000);

			if (!IsMissingDouble(ELTval))
			{
				ELT[i] = ELTval;
				ELP[i] = ELPval;
			}
		}

		curLevel.Value(curLevel.Value() - 1);

		foundCount = 0;
		for (auto& val : found)
		{
			if (val & FCAPE) foundCount++;
		}

		itsLogger.Trace("CAPE read for " + boost::lexical_cast<string>(foundCount) + "/" +
		                boost::lexical_cast<string>(found.size()) + " gridpoints");
		prevZenvInfo = ZenvInfo;
		prevTenvInfo = TenvInfo;
		prevPenvInfo = PenvInfo;
		prevTparcelVec = TparcelVec;
	}

	// If the CAPE area is continued all the way to level 60 and beyond, we don't have an EL for that
	// (since integration is forcefully stopped)
	// In this case level 60 = EL

	for (size_t i = 0; i < CAPE.size(); i++)
	{
		if (CAPE[i] > 0 && IsMissingDouble(ELT[i]))
		{
			TenvInfo->LocationIndex(i);
			PenvInfo->LocationIndex(i);

			ELT[i] = TenvInfo->Value();
			ELP[i] = PenvInfo->Value();
		}
	}

	myTargetInfo->Param(ELTParam);
	myTargetInfo->Data().Set(ELT);

	myTargetInfo->Param(ELPParam);
	myTargetInfo->Data().Set(ELP);

	myTargetInfo->Param(CAPEParam);
	myTargetInfo->Data().Set(CAPE);

	myTargetInfo->Param(CAPE1040Param);
	myTargetInfo->Data().Set(CAPE1040);

	myTargetInfo->Param(CAPE3kmParam);
	myTargetInfo->Data().Set(CAPE3km);
}

pair<vector<double>, vector<double>> cape::GetLFC(shared_ptr<info> myTargetInfo, vector<double>& T, vector<double>& P)
{
	auto h = GET_PLUGIN(hitool);

	assert(T.size() == P.size());

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	itsLogger.Trace("Searching environment temperature for starting pressure");

	vector<double> TenvLCL;

	try
	{
		TenvLCL = h->VerticalValue(param("T-K"), P);
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			return make_pair(vector<double>(), vector<double>());
		}

		throw;
	}

#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda())
	{
		return cape_cuda::GetLFCGPU(itsConfiguration, myTargetInfo, T, P, TenvLCL);
	}
	else
#endif
	{
		return GetLFCCPU(myTargetInfo, T, P, TenvLCL);
	}
}

pair<vector<double>, vector<double>> cape::GetLFCCPU(shared_ptr<info> myTargetInfo, vector<double>& T,
                                                     vector<double>& P, vector<double>& TenvLCL)
{
	auto h = GET_PLUGIN(hitool);

	assert(T.size() == P.size());

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());
	h->HeightUnit(kHPa);

	auto Piter = P, Titer = T;  // integration variables

	// Convert pressure to Pa since metutil-library expects that
	::MultiplyWith(Piter, 100);

	vector<bool> found(T.size(), false);

	vector<double> LFCT(T.size(), MissingDouble());
	vector<double> LFCP(T.size(), MissingDouble());

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
			Piter[i] = MissingDouble();
		}
	}

	size_t foundCount = count(found.begin(), found.end(), true);

	itsLogger.Debug("Found " + boost::lexical_cast<string>(foundCount) + " gridpoints that have LCL=LFC");

	// For each grid point find the hybrid level that's below LCL and then pick the lowest level
	// among all grid points; most commonly it's the lowest hybrid level

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));
	level curLevel = levels.first;

	auto prevPenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
	auto prevTenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);

	curLevel.Value(curLevel.Value() - 1);

	auto hPa150 = h->LevelForHeight(myTargetInfo->Producer(), 150.);
	auto hPa450 = h->LevelForHeight(myTargetInfo->Producer(), 450.);
	vector<double> prevTparcelVec(P.size(), MissingDouble());

	while (curLevel.Value() > hPa150.first.Value() && foundCount != found.size())
	{
		// Get environment temperature and pressure values for this level
		auto TenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto PenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = PenvInfo->Data().Values();
		::MultiplyWith(PenvVec, 100);

		// Lift the particle from previous level to this level. In the first revolution
		// of this loop the starting level is LCL. If target level level is below current level
		// (ie. we would be lowering the particle) missing value is returned.

		vector<double> TparcelVec(P.size(), MissingDouble());

		::MoistLift(&Piter[0], &Titer[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		double scale = 1;
		if (prevPenvInfo->Param().Name() == "P-PA") scale = 0.01;

		int i = -1;
		for (auto&& tup : zip_range(VEC(TenvInfo), VEC(PenvInfo), VEC(prevPenvInfo), VEC(prevTenvInfo), TparcelVec,
		                            prevTparcelVec, LFCT, LFCP))
		{
			i++;

			if (found[i]) continue;

			double Tenv = tup.get<0>();  // K
			assert(Tenv > 100.);

			double Penv = tup.get<1>();  // hPa
			assert(Penv < 1200.);
			assert(P[i] < 1200.);

			double prevPenv = tup.get<2>() * scale;
			assert(prevPenv < 1200.);

			double prevTenv = tup.get<3>();  // K
			assert(prevTenv > 100.);

			double Tparcel = tup.get<4>();  // K
			assert(Tparcel > 100. || IsMissingDouble(Tparcel));

			double prevTparcel = tup.get<5>();  // K
			assert(Tparcel > 100. || IsMissingDouble(Tparcel));

			double& Tresult = tup.get<6>();
			double& Presult = tup.get<7>();

			if (IsMissingDouble(Tparcel) || Penv > P[i] + 30)
			{
				continue;
			}

			if (Tparcel > Tenv)
			{
				// Parcel is now warmer than environment, we have found LFC and entering CAPE zone

				found[i] = true;

				if (IsMissingDouble(prevTparcel))
				{
					// Previous value is unknown: perhaps LFC is found very close to ground?
					// Use LCL for previous value.
					prevTparcel = T[i];
				}

				if (fabs(prevTparcel - prevTenv) < 0.0001)
				{
					Tresult = Tparcel;
					Presult = Penv;
				}
				else
				{
					auto intersection =
					    CAPE::GetPointOfIntersection(point(Tenv, Penv), point(prevTenv, prevPenv), point(Tparcel, Penv),
					                                 point(prevTparcel, prevPenv));
					Tresult = intersection.X();
					Presult = intersection.Y();

					if (IsMissingDouble(Tresult))
					{
						// Intersection not found, use exact level value
						Tresult = Tenv;
						Presult = Penv;
					}
				}

				assert(!IsMissingDouble(Tresult));
				assert(!IsMissingDouble(Presult));
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
		itsLogger.Trace("LFC processed for " + boost::lexical_cast<string>(foundCount) + "/" +
		                boost::lexical_cast<string>(found.size()) + " grid points");

		prevPenvInfo = PenvInfo;
		prevTenvInfo = TenvInfo;

		for (size_t i = 0; i < Titer.size(); i++)
		{
			if (found[i]) Titer[i] = MissingDouble();  // by setting this we prevent MoistLift to integrate particle
		}

		prevTparcelVec = TparcelVec;
	}
	return make_pair(LFCT, LFCP);
}

pair<vector<double>, vector<double>> cape::GetLCL(shared_ptr<info> myTargetInfo, const cape_source& sourceValues)
{
	vector<double> TLCL(get<0>(sourceValues).size(), MissingDouble());
	vector<double> PLCL = TLCL;

	// Need surface pressure

	double Pscale = 100.;  // P should be Pa

	for (auto&& tup : zip_range(get<0>(sourceValues), get<1>(sourceValues), get<2>(sourceValues), TLCL, PLCL))
	{
		double T = tup.get<0>();
		double TD = tup.get<1>();
		double P = tup.get<2>() * Pscale;  // Pa
		double& Tresult = tup.get<3>();
		double& Presult = tup.get<4>();

		auto lcl = metutil::LCLA_(P, T, TD);

		Tresult = lcl.T;  // K

		if (!IsMissingDouble(lcl.P))
		{
			Presult = 0.01 * ((lcl.P > P) ? P : lcl.P);  // hPa
		}
	}

	for (auto& val : PLCL)
	{
		val = fmax(val, 250.);
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
		return make_tuple(vector<double>(), vector<double>(), vector<double>());
	}

	auto T = VEC(TInfo);
	auto RH = VEC(RHInfo);

	vector<double> TD(T.size(), MissingDouble());

	for (size_t i = 0; i < TD.size(); i++)
	{
		if (!IsMissingDouble(T[i]) && !IsMissingDouble(RH[i]))
		{
			TD[i] = metutil::DewPointFromRH_(T[i], RH[i]);
		}
	}

	return make_tuple(T, TD, VEC(PInfo));
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
		return make_tuple(vector<double>(), vector<double>(), vector<double>());
	}
	else
	{
		// Himan specialty: empty data grid

		size_t miss = 0;
		for (auto& val : VEC(PInfo))
		{
			if (IsMissingDouble(val)) miss++;
		}

		if (PInfo->Data().MissingCount() == PInfo->Data().Size())
		{
			return make_tuple(vector<double>(), vector<double>(), vector<double>());
		}
	}

	auto P = PInfo->Data().Values();

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
			if (found[i]) continue;
			if (IsMissingDouble(T[i]) || IsMissingDouble(P[i]) || IsMissingDouble(RH[i])) continue;

			assert(T[i] > 150 && T[i] < 350);
			assert(P[i] > 100 && P[i] < 1500);
			assert(RH[i] > 0 && RH[i] < 102);

			Tpot[i] = metutil::Theta_(T[i], 100 * P[i]);
			MR[i] = metutil::smarttool::MixingRatio_(T[i], RH[i], 100 * P[i]);
		}

		tp.Process(Tpot, P);
		mr.Process(MR, P);

		foundCount = tp.HeightsCrossed();

		assert(tp.HeightsCrossed() == mr.HeightsCrossed());

		itsLogger.Debug("Data read " + boost::lexical_cast<string>(foundCount) + "/" +
		                boost::lexical_cast<string>(found.size()) + " gridpoints");

		for (size_t i = 0; i < found.size(); i++)
		{
			assert((P[i] > 100 && P[i] < 1500) || IsMissingDouble(P[i]));

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

	auto Tpot = tp.Result();
	auto MR = mr.Result();

	auto Psurf = Fetch(myTargetInfo->Time(), itsBottomLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
	P = Psurf->Data().Values();

	vector<double> T(Tpot.size(), MissingDouble());

	for (size_t i = 0; i < Tpot.size(); i++)
	{
		assert((P[i] > 100 && P[i] < 1500) || IsMissingDouble(P[i]));
		if (!IsMissingDouble(Tpot[i]) && !IsMissingDouble(P[i]))
		{
			T[i] = Tpot[i] * pow((P[i] / 1000.), 0.2854);
		}
	}

	vector<double> TD(T.size(), MissingDouble());

	for (size_t i = 0; i < MR.size(); i++)
	{
		if (!IsMissingDouble(T[i]) && !IsMissingDouble(MR[i]) && !IsMissingDouble(P[i]))
		{
			double Es = metutil::Es_(T[i]);  // Saturated water vapor pressure
			double E = metutil::E_(MR[i], 100 * P[i]);

			double RH = E / Es * 100;
			TD[i] = metutil::DewPointFromRH_(T[i], RH);
		}
	}

	return make_tuple(T, TD, P);
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

	vector<double> maxThetaE(myTargetInfo->Data().Size(), -1);
	vector<double> Ttheta(myTargetInfo->Data().Size(), MissingDouble());
	auto TDtheta = Ttheta;
	auto Ptheta = Ttheta;

	level curLevel = itsBottomLevel;

	info_t prevTInfo, prevRHInfo, prevPInfo;

	while (true)
	{
		auto TInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto RHInfo = Fetch(myTargetInfo->Time(), curLevel, param("RH-PRCNT"), myTargetInfo->ForecastType(), false);
		auto PInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

		if (!TInfo || !RHInfo || !PInfo)
		{
			return make_tuple(vector<double>(), vector<double>(), vector<double>());
		}

		int i = -1;

		for (auto&& tup : zip_range(VEC(TInfo), VEC(RHInfo), VEC(PInfo), maxThetaE, Ttheta, TDtheta, Ptheta))
		{
			i++;

			if (found[i]) continue;

			double T = tup.get<0>();
			double RH = tup.get<1>();
			double P = tup.get<2>();
			double& refThetaE = tup.get<3>();
			double& Tresult = tup.get<4>();
			double& TDresult = tup.get<5>();
			double& Presult = tup.get<6>();

			if (IsMissingDouble(P))
			{
				found[i] = true;
				continue;
			}

			if (P < 600.)
			{
				found[i] = true;  // Make sure this is the last time we access this grid point

				if (!prevPInfo || !prevTInfo || !prevRHInfo)
				{
					// Lowest grid point located above 600hPa, hmm...
					continue;
				}

				// Cut search if reach level 600hPa
				prevPInfo->LocationIndex(i);
				prevTInfo->LocationIndex(i);
				prevRHInfo->LocationIndex(i);

				// Linearly interpolate temperature and humidity values to 600hPa, to check
				// if highest theta e is found there

				T = interpolation::Linear(600., P, prevPInfo->Value(), T, prevTInfo->Value());
				RH = interpolation::Linear(600., P, prevPInfo->Value(), RH, prevRHInfo->Value());

				P = 600.;
			}

			double TD = metutil::DewPointFromRH_(T, RH);
			double ThetaE = metutil::smarttool::ThetaE_(T, RH, P * 100);
			assert(ThetaE >= 0);

			if (ThetaE >= refThetaE)
			{
				refThetaE = ThetaE;
				Tresult = T;
				TDresult = TD;
				Presult = P;

				assert(TDresult > 100);
			}
		}

		size_t foundCount = count(found.begin(), found.end(), true);

		if (foundCount == found.size())
		{
			break;
		}

		itsLogger.Trace("Max ThetaE processed for " + boost::lexical_cast<string>(foundCount) + "/" +
		                boost::lexical_cast<string>(found.size()) + " grid points");

		curLevel.Value(curLevel.Value() - 1);

		prevPInfo = PInfo;
		prevTInfo = TInfo;
		prevRHInfo = RHInfo;
	}

	return make_tuple(Ttheta, TDtheta, Ptheta);
}
