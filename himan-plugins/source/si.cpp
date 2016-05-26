/**
 * @file si.cpp
 *
 * @date Feb 13, 2014
 * @author partio
 */

#include "si.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <future>
#include "NFmiInterpolation.h"
#include <boost/thread.hpp>
#include "util.h"
#include "numerical_functions.h"

#include "neons.h"
#include "fetcher.h"
#include "querydata.h"
#include "hitool.h"

#include "si_cuda.h"

const unsigned char FCAPE		= (1 << 2);
const unsigned char FCAPE3km	= (1 << 0);

using namespace std;
using namespace himan::plugin;

#ifdef POINTDEBUG
himan::point debugPoint(25.47, 37.03);
#endif

#ifdef DEBUG
#define DumpVector(A, B) himan::util::DumpVector(A, B)
#else
#define DumpVector(A, B)
#endif

const himan::param SBLCLT("LCL-K", 4);
const himan::param SBLCLP("LCL-HPA", 4720);
const himan::param SBLCLZ("LCL-M", 4726);
const himan::param SBLFCT("LFC-K", 4);
const himan::param SBLFCP("LFC-HPA", 4721);
const himan::param SBLFCZ("LFC-M", 4727);
const himan::param SBELT("EL-K", 4);
const himan::param SBELP("EL-HPA", 4722);
const himan::param SBELZ("EL-M", 4728);
const himan::param SBCAPE("CAPE-JKG", 4723);
const himan::param SBCAPE1040("CAPE1040-JKG", 4729);
const himan::param SBCAPE3km("CAPE3KM-JKG", 4724);
const himan::param SBCIN("CIN-JKG", 4725);

const himan::param SB500LCLT("LCL500-K", 4);
const himan::param SB500LCLP("LCL500-HPA", 4730);
const himan::param SB500LCLZ("LCL500-M", 4736);
const himan::param SB500LFCT("LFC500-K", 4);
const himan::param SB500LFCP("LFC500-HPA", 4731);
const himan::param SB500LFCZ("LFC500-M", 4737);
const himan::param SB500ELT("EL500-K", 4);
const himan::param SB500ELP("EL500-HPA", 4732);
const himan::param SB500ELZ("EL500-M", 4738);
const himan::param SB500CAPE("CAPE500-JKG", 4733);
const himan::param SB500CAPE1040("CAPE5001040", 4739);
const himan::param SB500CAPE3km("CAPE5003KM", 4734);
const himan::param SB500CIN("CIN500-JKG", 4735);

const himan::param MULCLT("LCLMU-K", 4);
const himan::param MULCLP("LCLMU-HPA", 4740);
const himan::param MULCLZ("LCLMU-M", 4746);
const himan::param MULFCT("LFCMU-K", 4);
const himan::param MULFCP("LFCMU-HPA", 4741);
const himan::param MULFCZ("LFCMU-M", 4747);
const himan::param MUELT("ELMU-K", 4);
const himan::param MUELP("ELMU-HPA", 4742);
const himan::param MUELZ("ELMU-M", 4748);
const himan::param MUCAPE("CAPEMU-JKG", 4743);
const himan::param MUCAPE1040("CAPEMU1040", 4749);
const himan::param MUCAPE3km("CAPEMU3KM", 4744);
const himan::param MUCIN("CINMU-JKG", 4745);

double Max(const vector<double>& vec)
{ 
	double ret = -1e38;

	for(const double& val : vec)
	{
		if (val != kFloatMissing && val > ret) ret = val;
	}

	if (ret == -1e38) ret = kFloatMissing;

	return ret;
}

void MultiplyWith(vector<double>& vec, double multiplier)
{
	for(double& val : vec)
	{
		if (val != kFloatMissing) val *= multiplier;
	}
}

std::string PrintMean(const vector<double>& vec)
{
	double min = 1e38, max = -1e38, sum = 0;
	size_t count = 0, missing = 0;

	for(const double& val : vec)
	{
		if (val == himan::kFloatMissing)
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
		mean = sum / static_cast<double> (count);
	}
	
	return "min " + boost::lexical_cast<string> (static_cast<int> (min)) + " max " + boost::lexical_cast<string> (static_cast<int> (max)) + " mean " + boost::lexical_cast<string> (static_cast<int> (mean)) + " missing " + boost::lexical_cast<string> (missing);
}

si::si() : itsBottomLevel(kHybrid, kHPMissingInt)
{
	itsClearTextFormula = "<multiple algorithms>";

	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("si"));
}

void si::Process(std::shared_ptr<const plugin_configuration> conf)
{
	compiled_plugin_base::Init(conf);

	/*
	 * Set target parameters:
	 * - name 
	 * - univ_id 
	 * - grib2 descriptor 0'00'000
	 *
	 */

	auto theNeons = GET_PLUGIN(neons);

	itsBottomLevel = level(kHybrid, boost::lexical_cast<int> (theNeons->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "last hybrid level number")));

#ifdef HAVE_CUDA
	si_cuda::itsBottomLevel = itsBottomLevel;
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

	for (const auto& source : sourceDatas)
	{
		if (source == "surface")
		{
			theParams.push_back(SBLCLT);
			theParams.push_back(SBLCLP);
			theParams.push_back(SBLCLZ);
			theParams.push_back(SBLFCT);
			theParams.push_back(SBLFCP);
			theParams.push_back(SBLFCZ);
			theParams.push_back(SBELT);
			theParams.push_back(SBELP);
			theParams.push_back(SBELZ);
			theParams.push_back(SBCAPE);
			theParams.push_back(SBCAPE1040);
			theParams.push_back(SBCAPE3km);
			theParams.push_back(SBCIN);
			itsSourceDatas.push_back(kSurface);
		}
		else if (source == "500m mix")
		{
			theParams.push_back(SB500LCLT);
			theParams.push_back(SB500LCLP);
			theParams.push_back(SB500LCLZ);
			theParams.push_back(SB500LFCT);
			theParams.push_back(SB500LFCP);
			theParams.push_back(SB500LFCZ);
			theParams.push_back(SB500ELT);
			theParams.push_back(SB500ELP);
			theParams.push_back(SB500ELZ);
			theParams.push_back(SB500CAPE);
			theParams.push_back(SB500CAPE1040);
			theParams.push_back(SB500CAPE3km);
			theParams.push_back(SB500CIN);	
			itsSourceDatas.push_back(k500mAvgMixingRatio);
		}
		else if (source == "most unstable")
		{
			theParams.push_back(MULCLT);
			theParams.push_back(MULCLP);
			theParams.push_back(MULCLZ);
			theParams.push_back(MULFCT);
			theParams.push_back(MULFCP);
			theParams.push_back(MULFCZ);
			theParams.push_back(MUELT);
			theParams.push_back(MUELP);
			theParams.push_back(MUELZ);
			theParams.push_back(MUCAPE);
			theParams.push_back(MUCAPE1040);
			theParams.push_back(MUCAPE3km);
			theParams.push_back(MUCIN);	
			itsSourceDatas.push_back(kMaxThetaE);
		}
	}
	
	SetParams(theParams);
	
	Start();
	
}



void si::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	boost::thread_group g;
	
	for (auto sourceData : itsSourceDatas)
	{
		switch (sourceData)
		{
			case kSurface:
				g.add_thread(new boost::thread(&si::CalculateVersion, this, boost::ref(myTargetInfo), threadIndex, kSurface));
				break;
			case k500mAvgMixingRatio:
				g.add_thread(new boost::thread(&si::CalculateVersion, this, boost::ref(myTargetInfo), threadIndex, k500mAvgMixingRatio));
				break;
			case kMaxThetaE:
				g.add_thread(new boost::thread(&si::CalculateVersion, this, boost::ref(myTargetInfo), threadIndex, kMaxThetaE));
				break;
			default:
				throw runtime_error("Invalid source type");
				break;
		}
		
	}
	
	g.join_all();
	
}


void si::CalculateVersion(shared_ptr<info> myTargetInfoOrig, unsigned short threadIndex, HPSoundingIndexSourceDataType sourceType)
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

	/* HIMAN-117
	 * 
	 * There is a possible (probable?) race-condition when myTargetInfo is accessed from several
	 * subthreads at the same time. The problem arises when the calculation is finished and data is
	 * set to info: One thread might set the descriptors to a position, get suspended by the kernel
	 * and another thread re-sets the desriptors. Then the original thread is resumed and it sets
	 * the data to wrong parameter.
	 * 
	 * Fix: Create a new info for each thread.
	 */
	
	auto myTargetInfo = make_shared<info> (*myTargetInfoOrig);

	auto mySubThreadedLogger = logger_factory::Instance()->GetLog("siThread#" + boost::lexical_cast<string> (threadIndex) + "Version" + boost::lexical_cast<string> (static_cast<int> (sourceType)));

	mySubThreadedLogger->Info("Calculating source type " + HPSoundingIndexSourceDataTypeToString.at(sourceType) + " for time " + static_cast<string>(myTargetInfo->Time().ValidDateTime()));
	
	// 1. 
	
	auto timer = timer_factory::Instance()->GetTimer();
	timer->Start();
	
	pair<vector<double>, vector<double>> TandTD;
	
	param LCLTParam, LCLPParam, LCLZParam;
	param LFCTParam, LFCPParam, LFCZParam;
	param ELPParam, ELTParam, ELZParam;
	param CINParam, CAPEParam, CAPE1040Param, CAPE3kmParam;
	
	switch (sourceType)
	{
		case kSurface:
			TandTD = GetSurfaceTAndTD(myTargetInfo);
			LCLTParam = SBLCLT;
			LCLPParam = SBLCLP;
			LCLZParam = SBLCLZ;
			LFCTParam = SBLFCT;
			LFCPParam = SBLFCP;
			LFCZParam = SBLCLZ;
			CAPEParam = SBCAPE;
			CAPE1040Param = SBCAPE1040;
			CAPE3kmParam = SBCAPE3km;
			CINParam = SBCIN;
			ELPParam = SBELP;
			ELTParam = SBELT;
			ELZParam = SBELZ;
			break;
		
		case k500mAvg:
			throw runtime_error("Source type 500m avg not implemented");
			break;
		
		case k500mAvgMixingRatio:
			TandTD = Get500mMixingRatioTAndTD(myTargetInfo);
			LCLTParam = SB500LCLT;
			LCLPParam = SB500LCLP;
			LCLZParam = SB500LCLZ;
			LFCTParam = SB500LFCT;
			LFCPParam = SB500LFCP;
			LFCZParam = SB500LFCZ;
			CAPEParam = SB500CAPE;
			CAPE1040Param = SB500CAPE1040;
			CAPE3kmParam = SB500CAPE3km;
			CINParam = SB500CIN;
			ELPParam = SB500ELP;
			ELTParam = SB500ELT;
			ELZParam = SB500ELZ;
			break;
			
		case kMaxThetaE:
			TandTD = GetHighestThetaETAndTD(myTargetInfo);
			LCLTParam = MULCLT;
			LCLPParam = MULCLP;
			LCLZParam = MULCLZ;
			LFCTParam = MULFCT;
			LFCPParam = MULFCP;
			LFCZParam = MULFCZ;
			CAPEParam = MUCAPE;
			CAPE1040Param = MUCAPE1040;
			CAPE3kmParam = MUCAPE3km;
			CINParam = MUCIN;
			ELPParam = MUELP;
			ELTParam = MUELT;
			ELZParam = MUELZ;
			break;
		
		default:
			throw runtime_error("Invalid source data type");
			break;
	}
	
	if (TandTD.first.empty()) return;
	
	timer->Stop();

	mySubThreadedLogger->Info("Source data calculated in " + boost::lexical_cast<string> (timer->GetTime()) + " ms");

	mySubThreadedLogger->Debug("Surface temperature: " + ::PrintMean(TandTD.first));
	mySubThreadedLogger->Debug("Surface dewpoint: " + ::PrintMean(TandTD.second));

	// 2.
	
	timer->Start();
	
	auto LCL = GetLCL(myTargetInfo, TandTD.first, TandTD.second);
	
	timer->Stop();
	
	mySubThreadedLogger->Info("LCL calculated in " + boost::lexical_cast<string> (timer->GetTime()) + " ms");

	mySubThreadedLogger->Debug("LCL temperature: " + ::PrintMean(LCL.first));
	mySubThreadedLogger->Debug("LCL pressure: " + ::PrintMean(LCL.second));

	myTargetInfo->Param(LCLTParam);
	myTargetInfo->Data().Set(LCL.first);
	
	myTargetInfo->Param(LCLPParam);
	myTargetInfo->Data().Set(LCL.second);
	
	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	auto height = h->VerticalValue(param("HL-M"), LCL.second);

	myTargetInfo->Param(LCLZParam);
	myTargetInfo->Data().Set(height);


	// 3.

	timer->Start();
	
	auto LFC = GetLFC(myTargetInfo, LCL.first, LCL.second);

	timer->Stop();

	mySubThreadedLogger->Info("LFC calculated in " + boost::lexical_cast<string> (timer->GetTime()) + " ms");

	if (LFC.first.empty())
	{
		return;
	}

	mySubThreadedLogger->Debug("LFC temperature: " + ::PrintMean(LFC.first));
	mySubThreadedLogger->Debug("LFC pressure: " + ::PrintMean(LFC.second));

	myTargetInfo->Param(LFCTParam);
	myTargetInfo->Data().Set(LFC.first);
	
	myTargetInfo->Param(LFCPParam);
	myTargetInfo->Data().Set(LFC.second);
	
	height = h->VerticalValue(param("HL-M"), LFC.second);

	myTargetInfo->Param(LFCZParam);
	myTargetInfo->Data().Set(height);

	// 4. & 5.

	timer->Start();

	auto capeInfo = make_shared<info> (*myTargetInfo);
	boost::thread t1(&si::GetCAPE, this, boost::ref(capeInfo), LFC, ELTParam, ELPParam, ELZParam, CAPEParam, CAPE1040Param, CAPE3kmParam);

	auto cinInfo = make_shared<info> (*myTargetInfo);
	boost::thread t2(&si::GetCIN, this, boost::ref(cinInfo), TandTD.first, LCL.first, LCL.second, LFC.second, CINParam);

	t1.join();
	t2.join();
	
	timer->Stop();

	mySubThreadedLogger->Info("CAPE and CIN calculated in " + boost::lexical_cast<string> (timer->GetTime()) + " ms");

	// Do smoothening for CAPE & CIN parameters
	// Calculate average of nearest 4 points + the point in question
	mySubThreadedLogger->Trace("Smoothening");
	
	himan::matrix<double> filter_kernel(3,3,1,kFloatMissing);
	// C was row-major... right?
	filter_kernel.Set({0, 0.2, 0, 0.2, 0.2, 0.2, 0, 0.2, 0});
	
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
	mySubThreadedLogger->Debug("CAPE: " + ::PrintMean(VEC(capeInfo)));
	capeInfo->Param(CAPE1040Param);
	mySubThreadedLogger->Debug("CAPE1040: " + ::PrintMean(VEC(capeInfo)));
	capeInfo->Param(CAPE3kmParam);
	mySubThreadedLogger->Debug("CAPE3km: " + ::PrintMean(VEC(capeInfo)));
	cinInfo->Param(CINParam);
	mySubThreadedLogger->Debug("CIN: " + ::PrintMean(VEC(cinInfo)));
	
}

void si::GetCIN(shared_ptr<info> myTargetInfo, const vector<double>& Tsurf, const vector<double>& TLCL, const vector<double>& PLCL, const vector<double>& PLFC, param CINParam)
{
	if (itsConfiguration->UseCuda())
	{
		si_cuda::GetCINGPU(itsConfiguration, myTargetInfo, Tsurf, TLCL, PLCL, PLFC, CINParam);
	}
	else
	{
		GetCINCPU(myTargetInfo, Tsurf, TLCL, PLCL, PLFC, CINParam);
	}
}

void si::GetCINCPU(shared_ptr<info> myTargetInfo, const vector<double>& Tsurf, const vector<double>& TLCL, const vector<double>& PLCL, const vector<double>& PLFC, param CINParam)
{
	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	vector<bool> found(Tsurf.size(), false);

	forecast_time ftime = myTargetInfo->Time();
	forecast_type ftype = myTargetInfo->ForecastType();

	/*
	 * Modus operandi:
	 * 
	 * 1. Integrate from ground to LCL dry adiabatically
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
	
	auto basePenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);
	auto prevZenvInfo = Fetch(ftime, curLevel, param("HL-M"), ftype, false);
	auto prevTenvInfo = Fetch(ftime, curLevel, param("T-K"), ftype, false);
	auto prevPenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);
	
	std::vector<double> cinh(PLCL.size(), 0);
	
	size_t foundCount = count(found.begin(), found.end(), true);
	
	auto Piter = basePenvInfo->Data().Values();
	::MultiplyWith(Piter, 100);
	
	auto PLCLPa = PLCL;
	::MultiplyWith(PLCLPa, 100);
	
	auto Titer = Tsurf;
	auto prevTparcelVec = Tsurf;

	curLevel.Value(curLevel.Value()-1);

	auto hPa100 = h->LevelForHeight(myTargetInfo->Producer(), 100.);

	while (curLevel.Value() > hPa100.first.Value() && foundCount != found.size())
	{
		auto ZenvInfo = Fetch(ftime, curLevel, param("HL-M"), ftype, false);
		auto TenvInfo = Fetch(ftime, curLevel, param("T-K"), ftype, false);
		auto PenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);
		
		vector<double> TparcelVec(Piter.size(), kFloatMissing);

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = PenvInfo->Data().Values();
		::MultiplyWith(PenvVec, 100);

		metutil::LiftLCL(&Piter[0], &Titer[0], &PLCLPa[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		int i = -1;

		for (auto&& tup : zip_range(VEC(TenvInfo), VEC(PenvInfo), VEC(ZenvInfo), VEC(prevZenvInfo), Titer))
		{
			i++;

			if (found[i]) continue;

			double Tenv = tup.get<0> (); // K
			assert(Tenv >= 100.);
			
			double Penv = tup.get<1>(); // hPa
			assert(Penv < 1200.);
			
			double Zenv = tup.get<2>(); // m
			double prevZenv = tup.get<3>(); // m
						
			double Tparcel = tup.get<4>(); // K
			assert(Tparcel >= 100.);
			
			assert(PLFC[i] < 1200. || PLFC[i] == kFloatMissing);
					
			if (PLFC[i] == kFloatMissing)
			{
				found[i] = true;
				continue;
			}
			else if (Penv <= PLFC[i])
			{
				// reached max height
				// TODO: final piece integration
				found[i] = true;
				continue;
			}
			
			if (Penv < PLCL[i])
			{
				// Above LCL, switch to virtual temperature
				
				if (Tparcel == kFloatMissing) continue;
				
				Tparcel = metutil::VirtualTemperature_(Tparcel, Penv * 100);
				Tenv = metutil::VirtualTemperature_(Tenv, Penv * 100);
			}

			if (Tparcel <= Tenv)
			{
				cinh[i] += constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);

				assert(cinh[i] <= 0);
			}
			else if (cinh[i] != 0)
			{
				// Parcel buoyant --> cape layer, no more CIN. We stop integration here.
				// TODO: final piece integration
				found[i] = true;
			}
		}

		foundCount = count(found.begin(), found.end(), true);

		itsLogger->Trace("CIN read for " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " gridpoints");

		curLevel.Value(curLevel.Value()-1);
		prevZenvInfo = ZenvInfo;
		prevTparcelVec = TparcelVec;

		prevZenvInfo = ZenvInfo;
		prevTenvInfo = TenvInfo;
		prevPenvInfo = PenvInfo;
		prevTparcelVec = TparcelVec;

		for (size_t i = 0; i < Titer.size(); i++)
		{
			// preserve starting position for those grid points that have value
			if (TparcelVec[i] != kFloatMissing && PenvVec[i] != kFloatMissing)
			{
				Titer[i] = TparcelVec[i];
				Piter[i] = PenvVec[i];
			}

			if (found[i]) Titer[i] = kFloatMissing; // by setting this we prevent MoistLift to integrate particle

		}
	} 

	myTargetInfo->Param(CINParam);
	myTargetInfo->Data().Set(cinh);
	
}

void si::GetCAPE(shared_ptr<info> myTargetInfo, const pair<vector<double>,vector<double>>& LFC, param ELTParam, param ELPParam, param ELZParam, param CAPEParam, param CAPE1040Param, param CAPE3kmParam)
{
	if (itsConfiguration->UseCuda())
	{
		si_cuda::GetCAPEGPU(itsConfiguration, myTargetInfo, LFC.first, LFC.second, ELTParam, ELPParam, CAPEParam, CAPE1040Param, CAPE3kmParam);
	}
	else
	{
		GetCAPECPU(myTargetInfo, LFC.first, LFC.second, ELTParam, ELPParam, CAPEParam, CAPE1040Param, CAPE3kmParam);
	}
	
	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);
	
	myTargetInfo->Param(ELPParam);
	auto height = h->VerticalValue(param("HL-M"), VEC(myTargetInfo));

	myTargetInfo->Param(ELZParam);
	myTargetInfo->Data().Set(height);

}

void si::GetCAPECPU(shared_ptr<info> myTargetInfo, const vector<double>& T, const vector<double>& P, param ELTParam, param ELPParam, param CAPEParam, param CAPE1040Param, param CAPE3kmParam)
{
	assert(T.size() == P.size());
	
	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	// Found count determines if we have calculated all three CAPE variation for a single grid point
	vector<unsigned char> found(T.size(), 0);

	vector<double> CAPE(T.size(), 0);
	vector<double> CAPE1040(T.size(), 0);
	vector<double> CAPE3km(T.size(), 0);
	vector<double> ELT(T.size(), kFloatMissing);
	vector<double> ELP(T.size(), kFloatMissing);
	
	// Unlike LCL, LFC is *not* found for all grid points

	size_t foundCount = 0;

	for (size_t i = 0; i < P.size(); i++)
	{
		if (P[i] == kFloatMissing)
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
	
	curLevel.Value(curLevel.Value() - 1);	
	
	auto Piter = P, Titer = T; // integration variables
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

		vector<double> TparcelVec(P.size(), kFloatMissing);

		metutil::MoistLift(&Piter[0], &Titer[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		int i = -1;
		for (auto&& tup : zip_range(VEC(PenvInfo), VEC(ZenvInfo), VEC(prevZenvInfo), VEC(prevTenvInfo), VEC(prevPenvInfo), VEC(TenvInfo), TparcelVec, prevTparcelVec))
		{

			i++;

			double Tenv = tup.get<5>(); // K
			assert(Tenv > 100.);
			
			double prevTenv = tup.get<3>(); // K
			assert(prevTenv > 100.);
			
			double Penv = tup.get<0>(); // hPa
			assert(Penv < 1200.);
			
			double prevPenv = tup.get<4>(); // hPa
			assert(prevPenv < 1200.);
			
			double Zenv = tup.get<1>(); // m
			double prevZenv = tup.get<2>(); // m
			
			double Tparcel = tup.get<6>(); // K
			assert(Tparcel > 100. || Tparcel == kFloatMissing);
			
			double prevTparcel = tup.get<7>(); // K
			assert(prevTparcel > 100. || Tparcel == kFloatMissing);
			
#ifdef POINTDEBUG
			myTargetInfo->LocationIndex(i);
			point currentPoint = myTargetInfo->LatLon();
			
			if (fabs(currentPoint.X() - debugPoint.X()) < 0.08 && fabs(currentPoint.Y() - debugPoint.Y()) < 0.08)
			{
				std::cout << "LatLon\t" << currentPoint.X() << "," << currentPoint.Y() << std::endl
							<< "Tparcel\t" << Tparcel << std::endl
							<< "Tenv\t" << Tenv << std::endl
							<< "startP\t" << P[i] << std::endl
							<< "Penv\t" << Penv << std::endl
							<< "CAPE\t" << CAPE[i] << std::endl
							<< "CAPE1040\t" << CAPE1040[i] << std::endl
							<< "MoistLift" << std::endl
							<< "currP\t" << Piter[i] << std::endl
							<< "currT\t" << Titer[i] << std::endl
							<< "targetP\t" << PenvVec[i] << std::endl
							<< "result\t" << TparcelVec[i] << std::endl
							<< "------\n";
			}
#endif
			if (found[i] & FCAPE)
			{
				continue;
			}
			else if (Penv == kFloatMissing || Tenv == kFloatMissing || Zenv == kFloatMissing || prevZenv == kFloatMissing || Tparcel == kFloatMissing || prevTparcel == kFloatMissing || Penv > P[i])
			{
				// Missing data or current grid point is below LFC
				continue;
			}
			else if (curLevel.Value() < 85 && (Tenv - Tparcel) > 25.)
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
				
				assert(CAPE3km[i] < 3000.); // 3000J/kg, not 3000m
				assert(CAPE3km[i] >= 0);
			}

			double C = CAPE::CalcCAPE1040(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

			CAPE1040[i] += C;

			assert(CAPE1040[i] < 5000.);
			assert(CAPE1040[i] >= 0);
			
			double CAPEval, ELTval, ELPval;
			
			CAPE::CalcCAPE(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv, CAPEval, ELTval, ELPval);

			CAPE[i] += CAPEval;
			
			assert(CAPEval >= 0.);				
			assert(CAPE[i] < 8000);
			
			if (ELTval != kFloatMissing)
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

		itsLogger->Trace("CAPE read for " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " gridpoints");
		prevZenvInfo = ZenvInfo;
		prevTenvInfo = TenvInfo;
		prevPenvInfo = PenvInfo;
		prevTparcelVec = TparcelVec;

		for (size_t i = 0; i < Titer.size(); i++)
		{
			// preserve starting position for those grid points that have value
			if (TparcelVec[i] != kFloatMissing && PenvVec[i] != kFloatMissing)
			{
				Titer[i] = TparcelVec[i];
				Piter[i] = PenvVec[i];
			}
		}
	}
	
	// If the CAPE area is continued all the way to level 60 and beyond, we don't have an EL for that
	// (since integration is forcefully stopped)
	// In this case level 60 = EL
	
	for (size_t i = 0; i < CAPE.size(); i++)
	{
		if (CAPE[i] > 0 && ELT[i] == kFloatMissing)
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

pair<vector<double>,vector<double>> si::GetLFC(shared_ptr<info> myTargetInfo, vector<double>& T, vector<double>& P)
{

	auto h = GET_PLUGIN(hitool);

	assert(T.size() == P.size());
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);
	
	// The arguments given to this function are LCL temperature and pressure
	// Often LFC height is the same as LCL height, check that now
	
	itsLogger->Trace("Searching environment temperature for LCL");

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

		throw e;
	}
	
	if (itsConfiguration->UseCuda())
	{
		return si_cuda::GetLFCGPU(itsConfiguration, myTargetInfo, T, P, TenvLCL);
	}
	else
	{
		return GetLFCCPU(myTargetInfo, T, P, TenvLCL);
	}

}

pair<vector<double>,vector<double>> si::GetLFCCPU(shared_ptr<info> myTargetInfo, vector<double>& T, vector<double>& P, vector<double>& TenvLCL)
{

	auto h = GET_PLUGIN(hitool);

	assert(T.size() == P.size());
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	auto Piter = P, Titer = T; // integration variables
	
	// Convert pressure to Pa since metutil-library expects that
	::MultiplyWith(Piter, 100);

	vector<bool> found(T.size(), false);
	
	vector<double> LFCT(T.size(), kFloatMissing);
	vector<double> LFCP(T.size(), kFloatMissing);

	for (size_t i = 0; i < TenvLCL.size(); i++)
	{
		if (T[i] >= TenvLCL[i])
		{
			found[i] = true;
			LFCT[i] = T[i];
			LFCP[i] = P[i];
			Piter[i] = kFloatMissing;
		}
	}
	
	size_t foundCount = count(found.begin(), found.end(), true);

	itsLogger->Debug("Found " + boost::lexical_cast<string> (foundCount) + " gridpoints that have LCL=LFC");

	// For each grid point find the hybrid level that's below LCL and then pick the lowest level
	// among all grid points; most commonly it's the lowest hybrid level

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::Max(P));
	level curLevel = levels.first;

	auto prevPenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
	auto prevTenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);

	curLevel.Value(curLevel.Value()-1);

	auto hPa150 = h->LevelForHeight(myTargetInfo->Producer(), 150.);
	auto hPa450 = h->LevelForHeight(myTargetInfo->Producer(), 450.);

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

		vector<double> TparcelVec(P.size(), kFloatMissing);
		metutil::MoistLift(&Piter[0], &Titer[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		int i = -1;
		for (auto&& tup : zip_range(VEC(TenvInfo), VEC(PenvInfo), VEC(prevPenvInfo), VEC(prevTenvInfo), TparcelVec, LFCT, LFCP))
		{
			i++;
	
			if (found[i]) continue;

			double Tenv = tup.get<0> (); // K
			assert(Tenv > 100.);

			double Penv = tup.get<1> (); // hPa
			assert(Penv < 1200.);
			assert(P[i] < 1200.);
			
			double Tparcel = tup.get<4> (); // K
			assert(Tparcel > 100.);
			
			double& Tresult = tup.get<5> ();
			double& Presult = tup.get<6> ();
	
#ifdef POINTDEBUG
			myTargetInfo->LocationIndex(i);
			point currentPoint = myTargetInfo->LatLon();
			
			if (fabs(currentPoint.X() - debugPoint.X()) < 0.08 && fabs(currentPoint.Y() - debugPoint.Y()) < 0.08)
			{
				std::cout << "LatLon\t" << currentPoint.X() << "," << currentPoint.Y() << std::endl
							<< "Tparcel\t" << Tparcel << std::endl
							<< "Tenv\t" << Tenv << std::endl
							<< "LCLP\t" << P[i] << std::endl
							<< "Penv\t" << Penv << std::endl
							<< "Tresult\t" << Tresult << std::endl
							<< "Presult\t" << Presult << std::endl
							<< "------\n";
			}
#endif
			if (Tparcel == kFloatMissing || Penv > P[i])
			{
				continue;
			}

			if (Tparcel >= Tenv)
			{
				// Parcel is now warmer than environment, we have found LFC and entering CAPE zone

				found[i] = true;

				// We have no specific information on the precise height where the temperature has crossed
				// Or we could if we'd integrate it but it makes the calculation more complex. So maybe in the
				// future. For now just take an average of upper and lower level values.
				
				double prevTenv = tup.get<3> (); // K
				assert(prevTenv > 100.);

				Tresult = (Tenv + prevTenv) * 0.5;

				double prevPenv = tup.get<2> ();
				if (prevPenvInfo->Param().Name() == "P-PA") prevPenv *= 0.01;

				// Never allow LFC pressure to be bigger than LCL pressure; bound lower level (with larger pressure value)
				// to LCL level if it below LCL

				prevPenv = min(prevPenv, P[i]);
				
				Presult = (Penv + prevPenv) * 0.5;
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
		itsLogger->Trace("LFC processed for " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " grid points");

		prevPenvInfo = PenvInfo;
		prevTenvInfo = TenvInfo;

		for (size_t i = 0; i < Titer.size(); i++)
		{
			// preserve starting position for those grid points that have value
			if (TparcelVec[i] != kFloatMissing && PenvVec[i] != kFloatMissing)
			{
				Titer[i] = TparcelVec[i];
				Piter[i] = PenvVec[i];
			}
			if (found[i]) Titer[i] = kFloatMissing; // by setting this we prevent MoistLift to integrate particle
		}
	}

	return make_pair(LFCT, LFCP);
}

pair<vector<double>,vector<double>> si::GetLCL(shared_ptr<info> myTargetInfo, vector<double>& Tsurf, vector<double>& TDsurf)
{
	vector<double> TLCL(Tsurf.size(), kFloatMissing);
	vector<double> PLCL = TLCL;

	// Need surface pressure

	const params PParams({param("PGR-PA"), param("P-PA")});

	auto Psurf = Fetch(myTargetInfo->Time(), level(kHeight, 0), PParams, myTargetInfo->ForecastType(), false);

	double Pscale = 1.; // P should be Pa

	
	if (!Psurf)
	{
		itsLogger->Warning("Surface pressure not found, trying lowest hybrid level pressure");
		Psurf = Fetch(myTargetInfo->Time(), itsBottomLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
		
		if (!Psurf)
		{
			throw runtime_error("Pressure data not found");
		}
		
		Pscale = 100.;
	}

	assert(Tsurf.size() == VEC(Psurf).size());

	for (auto&& tup : zip_range(Tsurf, TDsurf, VEC(Psurf), TLCL, PLCL))
	{	
		double T = tup.get<0> ();
		double TD = tup.get<1> ();
		double P = tup.get<2> () * Pscale; // Pa
		double& Tresult = tup.get<3> ();
		double& Presult = tup.get<4> ();
		
		auto lcl = metutil::LCLA_(P, T, TD);
		
		Tresult = lcl.T; // K
		
		if (lcl.P != kFloatMissing)
		{
			Presult = 0.01 * ((lcl.P > P) ? P : lcl.P); // hPa
		}
	}

	for (size_t i = 0; i < PLCL.size(); i++)
	{
		if (PLCL[i] < 250.) PLCL[i] = 250.;
	}

	return make_pair(TLCL,PLCL);
	
}

pair<vector<double>,vector<double>> si::GetSurfaceTAndTD(shared_ptr<info> myTargetInfo)
{
	auto TInfo = Fetch(myTargetInfo->Time(), itsBottomLevel, param("T-K"), myTargetInfo->ForecastType(), false);
	auto RHInfo = Fetch(myTargetInfo->Time(), itsBottomLevel, param("RH-PRCNT"), myTargetInfo->ForecastType(), false);
	
	if (!TInfo || !RHInfo)
	{
		return make_pair(vector<double>(),vector<double>());
	}

	auto T = VEC(TInfo);
	auto RH = VEC(RHInfo);

	vector<double> TD(T.size(), kFloatMissing);

	for (size_t i = 0; i < TD.size(); i++)
	{
		if (T[i] != kFloatMissing && RH[i] != kFloatMissing)
		{
			TD[i] = metutil::DewPointFromRH_(T[i], RH[i]);
		}
	}

	return make_pair(T,TD);

}

pair<vector<double>,vector<double>> si::Get500mMixingRatioTAndTD(shared_ptr<info> myTargetInfo)
{
	if (false && itsConfiguration->UseCuda())
	{
		return si_cuda::Get500mMixingRatioTAndTDGPU(itsConfiguration, myTargetInfo);
	}
	else
	{
		return Get500mMixingRatioTAndTDCPU(myTargetInfo);
	}
}

pair<vector<double>,vector<double>> si::Get500mMixingRatioTAndTDCPU(shared_ptr<info> myTargetInfo)
{
	modifier_mean tp, mr;
	level curLevel = itsBottomLevel;

	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

	itsLogger->Info("Calculating T&Td in smarttool compatibility mode");

	tp.HeightInMeters(false);
	mr.HeightInMeters(false);

	auto PInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

	if (!PInfo)
	{
		return make_pair(vector<double>(), vector<double>());
	}
	else
	{
		// Himan specialty: empty data grid
		
		size_t miss = 0;
		for (auto& val : VEC(PInfo))
		{
			if (val == kFloatMissing) miss++;
		}

		if (PInfo->Data().MissingCount() == PInfo->Data().Size())
		{
			return make_pair(vector<double>(), vector<double>());
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

		vector<double> Tpot(T.size(), kFloatMissing);
		vector<double> MR(T.size(), kFloatMissing);

		for (size_t i = 0; i < T.size(); i++)
		{
			if (found[i]) continue;

			assert((T[i] > 150 && T[i] < 350) || T[i] == kFloatMissing);
			assert((P[i] > 100 && P[i] < 1500) || P[i] == kFloatMissing);
			assert((RH[i] > 0 && RH[i] < 102) || RH[i] == kFloatMissing);
			
			Tpot[i] = metutil::Theta_(T[i], 100*P[i]);
			MR[i] = [&](){
				
				if (T[i] == kFloatMissing || P[i] == kFloatMissing || RH[i] == kFloatMissing) return kFloatMissing;

				// es				
				const double b = 17.2694;
				const double e0 = 6.11; // 6.11 <- 0.611 [kPa]
				const double T1 = 273.16; // [K]
				const double T2 = 35.86; // [K]

				double nume = b * (T[i]-T1);
				double deno = (T[i]-T2);

				double es = e0 * ::exp(nume/deno);
				
				// e
				double e = RH[i] * es / 100;
				
				// w
				double w = 0.622 * e/P[i] * 1000;
				
				assert(w < 60);
	
				return w;
			}();
		}

		tp.Process(Tpot, P);
		mr.Process(MR, P);

		foundCount = tp.HeightsCrossed();

		assert(tp.HeightsCrossed() == mr.HeightsCrossed());

		itsLogger->Debug("Data read " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " gridpoints");

		for (size_t i = 0; i < found.size(); i++)
		{
			assert((P[i] > 100 && P[i] < 1500) || P[i] == kFloatMissing);
			
			if (found[i])
			{
				P[i] = kFloatMissing; // disable processing of this
			}
			else if (P[i] != kFloatMissing)
			{
				P[i] -= 2.0;
			}
		}
	}
	
	auto Tpot = tp.Result();
	auto MR = mr.Result();

	auto Psurf = Fetch(myTargetInfo->Time(), itsBottomLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
	P = Psurf->Data().Values();

	vector<double> T(Tpot.size(), kFloatMissing);			

	for (size_t i = 0; i < Tpot.size(); i++)
	{
		assert((P[i] > 100 && P[i] < 1500) || P[i] == kFloatMissing);
		if (Tpot[i] != kFloatMissing && P[i] != kFloatMissing)
		{
			T[i] = Tpot[i] * pow((P[i]/1000.), 0.2854);
		}
	}

	vector<double> TD(T.size(), kFloatMissing);

	for (size_t i = 0; i < MR.size(); i++)
	{
		if (T[i] != kFloatMissing && MR[i] != kFloatMissing && P[i] != kFloatMissing)
		{
			double Es = metutil::Es_(T[i]); // Saturated water vapor pressure
			double E = metutil::E_(MR[i], 100*P[i]);

			double RH = E/Es * 100;
			TD[i] = metutil::DewPointFromRH_(T[i], RH);
		}
	}

	return make_pair(T,TD);
}

pair<vector<double>,vector<double>> si::GetHighestThetaETAndTD(shared_ptr<info> myTargetInfo)
{
	if (itsConfiguration->UseCuda())
	{
		return si_cuda::GetHighestThetaETAndTDGPU(itsConfiguration, myTargetInfo);
	}
	else
	{
		return GetHighestThetaETAndTDCPU(myTargetInfo);
	}
}

pair<vector<double>,vector<double>> si::GetHighestThetaETAndTDCPU(shared_ptr<info> myTargetInfo)
{
	vector<bool> found(myTargetInfo->Data().Size(), false);
	
	vector<double> maxThetaE(myTargetInfo->Data().Size(), -1);
	vector<double> Tsurf(myTargetInfo->Data().Size(), kFloatMissing);
	auto TDsurf = Tsurf;
	
	level curLevel = itsBottomLevel;
	
	info_t prevTInfo, prevRHInfo, prevPInfo;
	
	while (true)
	{
		auto TInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto RHInfo = Fetch(myTargetInfo->Time(), curLevel, param("RH-PRCNT"), myTargetInfo->ForecastType(), false);
		auto PInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
		
		if (!TInfo || !RHInfo || !PInfo)
		{
			return make_pair(vector<double>(),vector<double>());
		}

		int i = -1;

		for (auto&& tup : zip_range(VEC(TInfo), VEC(RHInfo), VEC(PInfo), maxThetaE, Tsurf, TDsurf))
		{
			i++;

			if (found[i]) continue;
			
			double T			= tup.get<0> ();
			double RH			= tup.get<1> ();
			double P			= tup.get<2> ();
			double& refThetaE	= tup.get<3> ();
			double& Tresult		= tup.get<4> ();
			double& TDresult	= tup.get<5> ();

			if (P == kFloatMissing)
			{
				found[i] = true;
				continue;
			}

			if (P < 600.)
			{
				// Cut search if reach level 600hPa
				prevPInfo->LocationIndex(i);
				prevTInfo->LocationIndex(i);
				prevRHInfo->LocationIndex(i);
				
				// Linearly interpolate temperature and humidity values to 600hPa, to check
				// if highest theta e is found there
				
				T = NFmiInterpolation::Linear(600., P, prevPInfo->Value(), T, prevTInfo->Value());
				RH = NFmiInterpolation::Linear(600., P, prevPInfo->Value(), RH, prevRHInfo->Value());
				
				found[i] = true; // Make sure this is the last time we access this grid point
				
				P = 600.;
			}
				
			double TD = metutil::DewPointFromRH_(T, RH);
			double ThetaE = metutil::ThetaE_(T, TD, P*100);
			
#ifdef POINTDEBUG
			myTargetInfo->LocationIndex(i);
			point currentPoint = myTargetInfo->LatLon();
			
			if (fabs(currentPoint.X() - debugPoint.X()) < 0.08 && fabs(currentPoint.Y() - debugPoint.Y()) < 0.08)
			{
				std::cout << "LatLon\t" << currentPoint.X() << "," << currentPoint.Y() << std::endl
							<< "T\t\t" << T << std::endl
							<< "RH\t\t" << RH << std::endl
							<< "P\t\t" << P << std::endl
							<< "ThetaE\t\t" << ThetaE << std::endl
							<< "refThetaE\t" << refThetaE << std::endl
							<< "Tresult\t\t" << Tresult << std::endl
							<< "TDresult\t" << TDresult << std::endl
							<< "------\n";
			}
#endif
			assert(ThetaE >= 0);
			
			if (ThetaE >= refThetaE)
			{
				refThetaE = ThetaE;
				Tresult = T;
				TDresult = TD;

				assert(TDresult > 100);
			}
		}
		
		size_t foundCount = count(found.begin(), found.end(), true);

		if (foundCount == found.size())
		{
			break;
		}

		itsLogger->Trace("Max ThetaE processed for " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " grid points");

		curLevel.Value(curLevel.Value()-1);
		
		prevPInfo = PInfo;
		prevTInfo = TInfo;
		prevRHInfo = RHInfo;
	}
	
	for (size_t i = 0; i < Tsurf.size(); i++)
	{
		if (Tsurf[i] == 0.) Tsurf[i] = kFloatMissing;
		if (TDsurf[i] == 0.) TDsurf[i] = kFloatMissing;
	}

	return make_pair(Tsurf,TDsurf);

}
