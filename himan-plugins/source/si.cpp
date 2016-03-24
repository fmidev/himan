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

#define FAST_MATH
#include "metutil.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"
#include "fetcher.h"
#include "querydata.h"
#include "hitool.h"

#undef HIMAN_AUXILIARY_INCLUDE

const unsigned char FCAPE		= (1 << 2);
const unsigned char FCAPE1040	= (1 << 1);
const unsigned char FCAPE3km	= (1 << 0);

using namespace std;
using namespace himan::plugin;

#ifdef POINTDEBUG
himan::point debugPoint(25.47, 37.03);
#endif

#ifdef DEBUG
std::vector<double> MUCAPEZonesEntered;
std::vector<double> MUCAPE1040ZonesEntered;
std::vector<double> MUCAPE3kmZonesEntered;
size_t MUCAPEZoneIndex = 0;
#define DumpVector(A, B) himan::util::DumpVector(A, B)
#else
#define DumpVector(A, B)
#endif

namespace CAPE
{
double IntegrateHeightAreaLeavingParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Zenv, double prevZenv, double areaUpperLimit);
tuple<double,double,double> IntegrateLeavingParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv, double prevPenv, double Zenv, double prevZenv);
double IntegrateEnteringParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Zenv, double prevZenv);
double IntegrateTemperatureAreaEnteringParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Zenv, double prevZenv, double areaColderLimit, double areaWarmerLimit);
double IntegrateTemperatureAreaLeavingParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Zenv, double prevZenv, double areaColderLimit, double areaWarmerLimit);

double Min(const vector<double>& vec);
double Max(const vector<double>& vec);
void MultiplyWith(vector<double>& vec, double multiplier);
void AddTo(vector<double>& vec, double incr);
}

const himan::param SBLCLT("LCL-K");
const himan::param SBLCLP("LCL-HPA", 68);
const himan::param SBLFCT("LFC-K");
const himan::param SBLFCP("LFC-HPA");
const himan::param SBELT("EL-K");
const himan::param SBELP("EL-HPA");
const himan::param SBCAPE("CAPE-JKG", 59);
const himan::param SBCAPE1040("CAPE1040-JKG", 59);
const himan::param SBCAPE3km("CAPE3KM-JKG", 59);
const himan::param SBCIN("CIN-JKG", 66);

const himan::param SB500LCLT("LCL500-K");
const himan::param SB500LCLP("LCL500-HPA", 68);
const himan::param SB500LFCT("LFC500-K");
const himan::param SB500LFCP("LFC500-HPA");
const himan::param SB500ELT("EL500-K");
const himan::param SB500ELP("EL500-HPA");
const himan::param SB500CAPE("CAPE500-JKG", 59);
const himan::param SB500CAPE1040("CAPE5001040", 59);
const himan::param SB500CAPE3km("CAPE5003KM", 59);
const himan::param SB500CIN("CIN500-JKG", 66);

const himan::param MULCLT("LCLMU-K");
const himan::param MULCLP("LCLMU-HPA", 68);
const himan::param MULFCT("LFCMU-K");
const himan::param MULFCP("LFCMU-HPA");
const himan::param MUELT("ELMU-K");
const himan::param MUELP("ELMU-HPA");
const himan::param MUCAPE("CAPEMU-JKG", 59);
const himan::param MUCAPE1040("CAPEMU1040", 59);
const himan::param MUCAPE3km("CAPEMU3KM", 59);
const himan::param MUCIN("CINMU-JKG", 66);

#ifdef DEBUG
const himan::param MUCAPEZoneCount("CAPEMUZONES");
const himan::param MUCAPE1040ZoneCount("CAPEMU1040ZONES");
const himan::param MUCAPE3kmZoneCount("CAPEMU3KMZONES");
#endif

si::si() : itsBottomLevel(kHPMissingInt)
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

	itsBottomLevel = boost::lexical_cast<int> (theNeons->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "last hybrid level number"));
	itsTopLevel = boost::lexical_cast<int> (theNeons->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "first hybrid level number"));

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
			theParams.push_back(SBLFCT);
			theParams.push_back(SBLFCP);
			theParams.push_back(SBELT);
			theParams.push_back(SBELP);
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
			theParams.push_back(SB500LFCT);
			theParams.push_back(SB500LFCP);
			theParams.push_back(SB500ELT);
			theParams.push_back(SB500ELP);
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
			theParams.push_back(MULFCT);
			theParams.push_back(MULFCP);
			theParams.push_back(MUELT);
			theParams.push_back(MUELP);
			theParams.push_back(MUCAPE);
			theParams.push_back(MUCAPE1040);
			theParams.push_back(MUCAPE3km);
			theParams.push_back(MUCIN);	

#ifdef DEBUG
			theParams.push_back(MUCAPEZoneCount);
			theParams.push_back(MUCAPE1040ZoneCount);
			theParams.push_back(MUCAPE3kmZoneCount);	
#endif
			itsSourceDatas.push_back(kMaxThetaE);
		}
	}
	
	SetParams(theParams);
	
	Start();
	
}



void si::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger_factory::Instance()->GetLog("siThread #" + boost::lexical_cast<string> (threadIndex));

	myThreadedLogger->Info("Calculating time " + static_cast<string>(myTargetInfo->Time().ValidDateTime()) + " level " + static_cast<string> (myTargetInfo->Level()));

	boost::thread_group g;
	
	for (auto sourceData : itsSourceDatas)
	{
		switch (sourceData)
		{
			case kSurface:
				g.add_thread(new boost::thread(&si::CalculateVersion, this, boost::ref(myTargetInfo), kSurface));
				break;
			case k500mAvgMixingRatio:
				g.add_thread(new boost::thread(&si::CalculateVersion, this, boost::ref(myTargetInfo), k500mAvgMixingRatio));
				break;
			case kMaxThetaE:
				g.add_thread(new boost::thread(&si::CalculateVersion, this, boost::ref(myTargetInfo), kMaxThetaE));
				break;
			default:
				throw runtime_error("Invalid source type");
				break;
		}
		
	}
	
	g.join_all();
	
}


void si::CalculateVersion(shared_ptr<info> myTargetInfo, HPSoundingIndexSourceDataType sourceType)
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

	auto mySubThreadedLogger = logger_factory::Instance()->GetLog("siVersionThread" + boost::lexical_cast<string> (static_cast<int> (sourceType)));

	mySubThreadedLogger->Info("Calculating source type " + boost::lexical_cast<string> (static_cast<int> (sourceType)));
	
	// 1. 
	
	auto timer = timer_factory::Instance()->GetTimer();
	timer->Start();
	
	cout << "\n--- T AND TD --\n" << endl;	

	pair<vector<double>, vector<double>> TandTD;
	
	param LCLTParam, LCLPParam, LFCTParam, LFCPParam, ELPParam, ELTParam;
	param CINParam, CAPEParam, CAPE1040Param, CAPE3kmParam;
	
	switch (sourceType)
	{
		case kSurface:
			TandTD = GetSurfaceTAndTD(myTargetInfo);
			LCLTParam = SBLCLT;
			LCLPParam = SBLCLP;
			LFCTParam = SBLFCT;
			LFCPParam = SBLFCP;
			CAPEParam = SBCAPE;
			CAPE1040Param = SBCAPE1040;
			CAPE3kmParam = SBCAPE3km;
			CINParam = SBCIN;
			ELPParam = SBELP;
			ELTParam = SBELT;
			break;
		
		case k500mAvg:
			TandTD = Get500mTAndTD(myTargetInfo);
			break;
		
		case k500mAvgMixingRatio:
			TandTD = Get500mMixingRatioTAndTD(myTargetInfo);
			LCLTParam = SB500LCLT;
			LCLPParam = SB500LCLP;
			LFCTParam = SB500LFCT;
			LFCPParam = SB500LFCP;
			CAPEParam = SB500CAPE;
			CAPE1040Param = SB500CAPE1040;
			CAPE3kmParam = SB500CAPE3km;
			CINParam = SB500CIN;
			ELPParam = SB500ELP;
			ELTParam = SB500ELT;
			break;
			
		case kMaxThetaE:
			TandTD = GetHighestThetaETAndTD(myTargetInfo);
			LCLTParam = MULCLT;
			LCLPParam = MULCLP;
			LFCTParam = MULFCT;
			LFCPParam = MULFCP;
			CAPEParam = MUCAPE;
			CAPE1040Param = MUCAPE1040;
			CAPE3kmParam = MUCAPE3km;
			CINParam = MUCIN;
			ELPParam = MUELP;
			ELTParam = MUELT;

			break;
		
		default:
			throw runtime_error("Invalid source data type");
			break;
	}
	
	if (TandTD.first.empty()) return;
	
	timer->Stop();

	mySubThreadedLogger->Info("Source data calculated in " + boost::lexical_cast<string> (timer->GetTime()) + " ms");

	DumpVector(get<0>(TandTD), "T");
	DumpVector(get<1>(TandTD), "TD");

	// 2.
	
	cout << "\n--- LCL --\n" << endl;
	
	timer->Start();
	
	auto LCL = GetLCL(myTargetInfo, TandTD.first, TandTD.second);
	
	timer->Stop();
	
	mySubThreadedLogger->Info("LCL calculated in " + boost::lexical_cast<string> (timer->GetTime()) + " ms");

	myTargetInfo->Param(LCLTParam);
	myTargetInfo->Data().Set(LCL.first);
	
	myTargetInfo->Param(LCLPParam);
	myTargetInfo->Data().Set(LCL.second);
	
	DumpVector(LCL.first, "LCL T");
	DumpVector(LCL.second, "LCL P");

	// 3.
	
	cout << "\n--- LFC --\n" << endl;

	timer->Start();
	
	auto LFC = GetLFC(myTargetInfo, LCL.first, LCL.second);

	timer->Stop();

	mySubThreadedLogger->Info("LFC calculated in " + boost::lexical_cast<string> (timer->GetTime()) + " ms");

	if (LFC.first.empty())
	{
		return;
	}

	myTargetInfo->Param(LFCTParam);
	myTargetInfo->Data().Set(LFC.first);
	
	myTargetInfo->Param(LFCPParam);
	myTargetInfo->Data().Set(LFC.second);

	DumpVector(LFC.first, "LFC T");
	DumpVector(LFC.second, "LFC P");

	// 4. & 5.

	timer->Start();

	cout << "\n--- CAPE AND CIN --\n" << endl;
	auto capeInfo = make_shared<info> (*myTargetInfo);
	boost::thread t1(&si::GetCAPE, this, boost::ref(capeInfo), LFC.first, LFC.second, ELTParam, ELPParam, CAPEParam, CAPE1040Param, CAPE3kmParam);

	auto cinInfo = make_shared<info> (*myTargetInfo);
	boost::thread t2(&si::GetCIN, this, boost::ref(cinInfo), TandTD.first, LCL.first, LCL.second, LFC.second, CINParam);

	t1.join();
	t2.join();
	
	timer->Stop();

	mySubThreadedLogger->Info("CAPE and CIN calculated in " + boost::lexical_cast<string> (timer->GetTime()) + " ms");

#ifdef DEBUG
	myTargetInfo->Param(MUCAPEZoneCount);
	myTargetInfo->Data().Set(MUCAPEZonesEntered);
	
	myTargetInfo->Param(MUCAPE1040ZoneCount);
	myTargetInfo->Data().Set(MUCAPE1040ZonesEntered);
	
	myTargetInfo->Param(MUCAPE3kmZoneCount);
	myTargetInfo->Data().Set(MUCAPE3kmZonesEntered);
#endif

	// Do smoothening for CAPE & CIN parameters
	// Calculate average of nearest 4 points + the point in question
	
	himan::matrix<double> filter_kernel(3,3,1,kFloatMissing);
	// C was row-major... right?
	filter_kernel.Set({0, 0.2, 0, 0.2, 0.2, 0.2, 0, 0.2, 0});
	
	capeInfo->Param(CAPEParam);
	himan::matrix<double> filtered = util::Filter2D(capeInfo->Data(), filter_kernel);
	capeInfo->Grid()->Data(filtered);

	capeInfo->Param(CAPE1040Param);
	filtered = util::Filter2D(capeInfo->Data(), filter_kernel);
	capeInfo->Grid()->Data(filtered);

	capeInfo->Param(CAPE3kmParam);
	filtered = util::Filter2D(capeInfo->Data(), filter_kernel);
	capeInfo->Grid()->Data(filtered);

	capeInfo->Param(CINParam);
	filtered = util::Filter2D(capeInfo->Data(), filter_kernel);
	capeInfo->Grid()->Data(filtered);

}

void si::GetCIN(shared_ptr<info> myTargetInfo, const vector<double>& Tsurf, const vector<double>& TLCL, const vector<double>& PLCL, const vector<double>& PLFC, param CINParam)
{
	const params PParams({param("PGR-PA"), param("P-PA")});
	
	//auto PsurfInfo = Fetch(myTargetInfo->Time(), level(kHeight, 0), PParams, myTargetInfo->ForecastType(), false);
	
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

	itsLogger->Debug("Fetching LCL metric height");

	DumpVector(ZLCL, "LCL Z");
	
	itsLogger->Debug("Fetching LFC metric height");
	DumpVector(ZLFC, "LFC Z");

	level curLevel(kHybrid, 137);
	
	auto basePenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);
	auto prevZenvInfo = Fetch(ftime, curLevel, param("HL-M"), ftype, false);
	auto prevTenvInfo = Fetch(ftime, curLevel, param("T-K"), ftype, false);
	auto prevPenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);
	
	std::vector<double> cinh(PLCL.size(), 0);
	
	size_t foundCount = count(found.begin(), found.end(), true);
	
	auto Piter = basePenvInfo->Data().Values();
	CAPE::MultiplyWith(Piter, 100);
	
	auto PLCLPa = PLCL;
	CAPE::MultiplyWith(PLCLPa, 100);
	
	auto Titer = Tsurf;
	auto prevTparcelVec = Tsurf;

	curLevel.Value(curLevel.Value()-1);
	
	while (curLevel.Value() > 60 && foundCount != found.size())
	{

		auto ZenvInfo = Fetch(ftime, curLevel, param("HL-M"), ftype, false);
		auto TenvInfo = Fetch(ftime, curLevel, param("T-K"), ftype, false);
		auto PenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);
		
		vector<double> TparcelVec(Piter.size(), kFloatMissing);

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = PenvInfo->Data().Values();
		CAPE::MultiplyWith(PenvVec, 100);

		metutil::LiftLCL(&Piter[0], &Titer[0], &PLCLPa[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		int i = -1;

		for (auto&& tup : zip_range(VEC(TenvInfo), VEC(PenvInfo), VEC(ZenvInfo), VEC(prevZenvInfo), VEC(basePenvInfo)))
		{
			i++;

			if (found[i]) continue;

			double Tenv = tup.get<0> (); // K
			assert(Tenv >= 100.);
			
			double Penv = tup.get<1>(); // hPa
			assert(Penv < 1200.);
			
			double Pbase = tup.get<4>(); // hPa
			assert(Pbase < 1200.);
			
			assert(PLFC[i] < 1200. || PLFC[i] == kFloatMissing);
			
			double Zenv = tup.get<2>(); // m
			double prevZenv = tup.get<3>(); // m
			
			double Tparcel = kFloatMissing;
			
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
			else if (curLevel.Value() < 85 && (Tenv - Tparcel) > 30.)
			{
				// Temperature gap between environment and parcel too large --> abort search.
				// Only for values higher in the atmosphere, to avoid the effects of inversion

				found[i] = true;
				continue;
			}
			
			if (Penv > PLCL[i])
			{
				// Below LCL --> Lift to cloud base
				assert(Tsurf[i] >0);
				assert(Tsurf[i] < 500);
				Tparcel = metutil::DryLift_(Pbase*100, Tsurf[i], Penv * 100);
			}
			else
			{
				// Above LCL --> Integrate to current hybrid level height
				Tparcel = metutil::MoistLift_(Pbase*100, Tsurf[i], Penv * 100);
				
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

		itsLogger->Debug("CIN read for " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " gridpoints");

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

double CalcCAPE1040(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv, double prevPenv, double Zenv, double prevZenv)
{
	double C = 0;

	assert(Tenv != kFloatMissing && Penv != kFloatMissing && Tparcel != kFloatMissing);

	if (Tparcel < Tenv && prevTparcel < prevTenv)
	{
		// No CAPE
		return C;
	}
		
	Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv*100);
	Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);

	double coldColderLimit = 233.15;
	double coldWarmerLimit = 263.15;

	if (Tparcel > Tenv)
	{
		// Parcel is buoyant at current height
		
		if (Tenv >= coldColderLimit && Tenv <= coldWarmerLimit)
		{
			// Parcel is inside cold area at current height
			
			if (prevTenv > coldWarmerLimit || prevTenv < coldColderLimit)
			{
				// Entering cold cape area from either warmer or colder area
#ifdef DEBUG
				//CAPE1040ZonesEntered[CAPEZoneIndex] += 1;
#endif
				
				C = CAPE::IntegrateTemperatureAreaEnteringParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv, coldColderLimit, coldWarmerLimit);
			}
			else
			{
				// Firmly in the cold zone
				C = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
				assert(C >= 0.);
				
			}
		}
		else if ((prevTenv > coldColderLimit && prevTenv < coldWarmerLimit) // At previous height conditions were suitable (TODO: buoyancy is not checked!)
				&& (Tenv < coldColderLimit || Tenv > coldWarmerLimit))
		{
			// Current env temperature is too cold or too warm			
			C = CAPE::IntegrateTemperatureAreaLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv, coldColderLimit, coldWarmerLimit);			
		}
	}
	else if (prevTparcel >= prevTenv) 
	{
		// No buoyancy anymore at current height, but
		// we HAD buoyancy: we just exited from a CAPE zone
			
		if (prevTenv >= coldColderLimit && prevTenv <= coldWarmerLimit)
		{
			/* Just left cold CAPE zone for an warmer or colder area */
			auto val = CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);
			C = get<0>(val);
		}
	}
	
	return C;
}

double CalcCAPE3km(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv, double prevPenv, double Zenv, double prevZenv)
{
	double C = 0.;

	Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv*100);
	Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);
			
	if (Tparcel > Tenv)
	{
		// Have buoyancy at current height

		if (Zenv <= 3000.)
		{
		
			if (prevTparcel >= prevTenv)
			{
				// Firmly in the zone
				C = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
			
			}
			else if  (prevTparcel < prevTenv)
			{
				// Just entered CAPE zone
				C = CAPE::IntegrateEnteringParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv);
			}
		}
		else if (prevZenv <= 3000.)
		{
			
			// Parcel has risen over 3km
			// Integrate from previous level to 3km (if parcel is buoyant the whole height)
			
			if (prevTparcel >= prevTenv)
			{
				C = CAPE::IntegrateHeightAreaLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv, 3000);
			}
		}
	}
	else
	{
		// Exited CAPE zone, no buoyancy at this height
		
		if (prevTparcel >= prevTenv)
		{
			if (Zenv <= 3000.)
			{
				// Integrate from previous height to intersection
				auto val = CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);
				C = get<0>(val);
			}
			
			else
			{
				// Integrate from previous height to 3km
				C = CAPE::IntegrateHeightAreaLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv, 3000);
			}
		}
	}

	return C;
}

tuple<double,double,double> CalcCAPE(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv, double prevPenv, double Zenv, double prevZenv)
{
	double CAPE = 0.;
	double ELT = kFloatMissing;
	double ELP = kFloatMissing;
	
	assert(Tenv != kFloatMissing && Penv != kFloatMissing && Tparcel != kFloatMissing);

	if (Tparcel < Tenv && prevTparcel < prevTenv)
	{
		// No CAPE
		return make_tuple(CAPE,ELT,ELP);
	}

	Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv*100);
	Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);

	if (Tparcel >= Tenv && prevTparcel >= Tenv)
	{
		// We are fully in a CAPE zone
		CAPE = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
		assert(CAPE >= 0);
	}
	else if (Tparcel >= Tenv && prevTparcel < prevTenv)
	{		

#ifdef DEBUG
		//MUCAPEZonesEntered[MUCAPEZoneIndex] += 1;
#endif
		CAPE = CAPE::IntegrateEnteringParcel(Tenv, prevTenv, Tparcel, prevTparcel, Zenv, prevZenv);

	}
	else if (Tparcel < Tenv && prevTparcel >= prevTenv)
	{
		auto val = CAPE::IntegrateLeavingParcel(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);
		CAPE = get<0>(val);
		ELT = get<1>(val);
		ELP = get<2>(val);
		
	}
	
	return make_tuple(CAPE,ELT,ELP);
}
			
void si::GetCAPE(shared_ptr<info> myTargetInfo, const vector<double>& T, const vector<double>& P, param ELTParam, param ELPParam, param CAPEParam, param CAPE1040Param, param CAPE3kmParam)
{
	assert(T.size() == P.size());

#ifdef DEBUG
	MUCAPEZonesEntered.resize(T.size(), 0);
	MUCAPE1040ZonesEntered.resize(T.size(), 0);
	MUCAPE3kmZonesEntered.resize(T.size(), 0);
#endif
	
	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	vector<unsigned char> found(T.size(), 0);

	vector<double> CAPE(T.size(), 0);
	vector<double> CAPE1040(T.size(), 0);
	vector<double> CAPE3km(T.size(), 0);
	vector<double> ELT(T.size(), kFloatMissing);
	vector<double> ELP(T.size(), kFloatMissing);
	
#ifdef DEBUG
	vector<double> CAPEZones(T.size(), 0);
#endif

	// Unlike LCL, LFC is *not* found for all grid points

	for (size_t i = 0; i < P.size(); i++)
	{
		if (P[i] == kFloatMissing)
		{
			found[i] |= FCAPE;
		}
	}

	// Found count determines if we have calculated all three CAPE variation for a single grid point
	
	size_t foundCount = 0;
	
	for (auto& val : found)
	{
		if (val & FCAPE) foundCount++;
	}
	
	// For each grid point find the hybrid level that's below LFC and then pick the lowest level
	// among all grid points
		
	auto levels = h->LevelForHeight(myTargetInfo->Producer(), CAPE::Max(P));
	
	level curLevel = levels.first;
	
	auto prevZenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);
	auto prevTenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
	auto prevPenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
	
	curLevel.Value(curLevel.Value() - 1);	
	
	auto Piter = P, Titer = T; // integration variables
	auto prevTparcelVec = Titer;
	
	// Convert pressure to Pa since metutil-library expects that
	CAPE::MultiplyWith(Piter, 100);
	
	info_t TenvInfo, PenvInfo, ZenvInfo;
	
	while (curLevel.Value() > 60 && foundCount != found.size())
	{
		// Get environment temperature, pressure and height values for this level
		PenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
		TenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		ZenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = PenvInfo->Data().Values();
		CAPE::MultiplyWith(PenvVec, 100);

		vector<double> TparcelVec(P.size(), kFloatMissing);

		metutil::MoistLift(&Piter[0], &Titer[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());

		int i = -1;
		for (auto&& tup : zip_range(VEC(PenvInfo), VEC(ZenvInfo), VEC(prevZenvInfo), VEC(prevTenvInfo), VEC(prevPenvInfo), VEC(TenvInfo), TparcelVec, prevTparcelVec))
		{

			i++;

#ifdef DEBUG
			MUCAPEZoneIndex = i;
#endif

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
			if (Penv == kFloatMissing || Tenv == kFloatMissing || Zenv == kFloatMissing || prevZenv == kFloatMissing || Tparcel == kFloatMissing || prevTparcel == kFloatMissing || Penv > P[i])
			{
				// Missing data or current grid point is below LFC
				continue;
			}

			if (curLevel.Value() < 85 && (Tenv - Tparcel) > 30.)
			{
				// Temperature gap between environment and parcel too large --> abort search.
				// Only for values higher in the atmosphere, to avoid the effects of inversion

				found[i] = true;
			}
		
			if (prevZenv >= 3000. && Zenv >= 3000.)
			{
				found[i] |= FCAPE3km;
			}
		
			if ((found[i] & FCAPE3km) == 0)
			{
				double C = CalcCAPE3km(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

				CAPE3km[i] += C;
				
				assert(CAPE3km[i] < 3000.); // 3000J/kg, not 3000m
				assert(CAPE3km[i] >= 0);
			}

			double C = CalcCAPE1040(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

			CAPE1040[i] += C;

			assert(CAPE1040[i] < 5000.);
			assert(CAPE1040[i] >= 0);
			
			auto val = CalcCAPE(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);
			
			CAPE[i] += get<0>(val);
			
			assert(get<0>(val) >= 0.);				
			assert(CAPE[i] < 8000);
			
			if (get<1>(val) != kFloatMissing)
			{
				ELT[i] = get<1>(val);
				ELP[i] = get<2>(val);
			}
		}

		curLevel.Value(curLevel.Value() - 1);		

		itsLogger->Debug("CAPE read for " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " gridpoints");
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
	
	
#ifdef DEBUG
	DumpVector(MUCAPEZonesEntered, "CAPEZonesEntered");
	DumpVector(MUCAPE1040ZonesEntered, "CAPE1040ZonesEntered");
#endif

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
	
	itsLogger->Info("Searching environment temperature for LCL");

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

	auto Piter = P, Titer = T; // integration variables
	
	// Convert pressure to Pa since metutil-library expects that
	CAPE::MultiplyWith(Piter, 100);

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

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), CAPE::Max(P));

	level curLevel = levels.first;

	auto prevPenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
	auto prevTenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);

	curLevel.Value(curLevel.Value()-1);

	//auto prevPenvInfo = Fetch(myTargetInfo->Time(), level(kGround, 0), param("P-PA"), myTargetInfo->ForecastType(), false);
	//auto prevTenvInfo = Fetch(myTargetInfo->Time(), level(kGround, 0), param("T-K"), myTargetInfo->ForecastType(), false);

	while (curLevel.Value() > 70 && foundCount != found.size())
	{	
		// Get environment temperature and pressure values for this level
		auto TenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto PenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = PenvInfo->Data().Values();		
		CAPE::MultiplyWith(PenvVec, 100);
		
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

				double prevP = tup.get<2> ();
				if (prevPenvInfo->Param().Name() == "P-PA") prevP *= 0.01;

				// Never allow LFC pressure to be bigger than LCL pressure; bound lower level (with larger pressure value)
				// to LCL level if it below LCL

				prevP = min(prevP, P[i]);
				
				Presult = (Penv + prevP) * 0.5;
			}
			else if (curLevel.Value() < 95 && (Tenv - Tparcel) > 30.)
			{
				// Temperature gap between environment and parcel too large --> abort search.
				// Only for values higher in the atmosphere, to avoid the effects of inversion

				found[i] = true;
			}
		}
		
		curLevel.Value(curLevel.Value() - 1);	
	
		foundCount = count(found.begin(), found.end(), true);
		itsLogger->Debug("LFC processed for " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " grid points");

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

#ifndef NDEBUG
	for (size_t i = 0; i < LFCP.size(); i++) assert(LFCP[i] == kFloatMissing || LFCP[i]<=P[i]);
#endif

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
		Psurf = Fetch(myTargetInfo->Time(), level(kHybrid, 137), param("P-HPA"), myTargetInfo->ForecastType(), false);
		
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
		if (PLCL[i] < 150.) PLCL[i] = 150.;
	}

	return make_pair(TLCL,PLCL);
	
}

pair<vector<double>,vector<double>> si::GetSurfaceTAndTD(shared_ptr<info> myTargetInfo)
{
	auto TInfo = Fetch(myTargetInfo->Time(), level(himan::kHeight,2), param("T-K"), myTargetInfo->ForecastType(), false);
	auto TDInfo = Fetch(myTargetInfo->Time(), level(himan::kHeight,2), param("TD-C"), myTargetInfo->ForecastType(), false);
	
	if (!TInfo || !TDInfo)
	{
		return make_pair(vector<double>(),vector<double>());
	}
	auto T = TInfo->Data().Values();
	auto TD = TDInfo->Data().Values();

	return make_pair(T,TD);

}

pair<vector<double>,vector<double>> si::Get500mTAndTD(shared_ptr<info> myTargetInfo)
{
	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

	auto T = h->VerticalAverage(param("T-K"), 0, 500);
	auto RH = h->VerticalAverage(param("RH-PRCNT"), 0, 500);

	auto TD = T;

	for (size_t i = 0; i < T.size(); i++)
	{
		TD[i] = metutil::DewPointFromRH_(T[i], RH[i]);
	}

	return make_pair(T,TD);

}

pair<vector<double>,vector<double>> si::Get500mMixingRatioTAndTD(shared_ptr<info> myTargetInfo)
{
	modifier_mean tp, mr;
	level curLevel(kHybrid, 137);

	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());

#if 1
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

			Tpot[i] = metutil::Theta_(T[i], 100*P[i]);
			//MR[i] = metutil::MixingRatio_(T[i], 100*P[i]);
			MR[i] = [&](){
				
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
	
				return w;
			}();
		}

		tp.Process(Tpot, P);
		mr.Process(MR, P);

		foundCount = tp.HeightsCrossed();

		assert(tp.HeightsCrossed() == mr.HeightsCrossed());

		//itsLogger->Debug("Data read " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " gridpoints");

		for (size_t i = 0; i < found.size(); i++)
		{
			assert(P[i] > 100 && P[i] < 1500);
			
			if (found[i])
			{
				P[i] = kFloatMissing; // disable processing of this
			}
			else
			{
				P[i] -= 2.0;
			}
		}
	}

#endif	
#if 0
	itsLogger->Info("Calculating T&Td himan style");
	vector<double> zero(myTargetInfo->Data().Size(), 0);
	vector<double> m500(zero.size(), 500.);

	tp.LowerHeight(zero);
	mr.LowerHeight(zero);

	tp.UpperHeight(m500);
	mr.UpperHeight(m500);

	while (!tp.CalculationFinished() && !mr.CalculationFinished())
	{
		auto TInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto RHInfo = Fetch(myTargetInfo->Time(), curLevel, param("RH-PRCNT"), myTargetInfo->ForecastType(), false);
		auto PInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
		auto ZInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);

		auto T = TInfo->Data().Values();

		auto RH = RHInfo->Data().Values();
		auto P = PInfo->Data().Values();

		vector<double> Tpot(T.size(), kFloatMissing);
		vector<double> MR(T.size(), kFloatMissing);

		for (size_t i = 0; i < T.size(); i++)
		{
			assert(T[i] != kFloatMissing);
			assert(P[i] != kFloatMissing);
	
			Tpot[i] = metutil::Theta_(T[i], 100*P[i]);
			//MR[i] = metutil::MixingRatio_(T[i], 100*P[i]);
			MR[i] = [&](){
				
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
	
				return w;
			}();
		}

		tp.Process(Tpot, ZInfo->Data().Values());
		mr.Process(MR, ZInfo->Data().Values());

		curLevel.Value(curLevel.Value()-1);
	}
#endif
	
	auto Tpot = tp.Result();
	auto MR = mr.Result();

	DumpVector(Tpot, "Tpot");
	DumpVector(MR, "MR");

	auto Psurf = Fetch(myTargetInfo->Time(), level(kHybrid, 137), param("P-HPA"), myTargetInfo->ForecastType(), false);
	P = Psurf->Data().Values();

	vector<double> T(Tpot.size(), kFloatMissing);			

	for (size_t i = 0; i < Tpot.size(); i++)
	{
		T[i] = Tpot[i] * pow((P[i]/1000.), 0.2854);
	}

	vector<double> TD(T.size(), kFloatMissing);

	for (size_t i = 0; i < MR.size(); i++)
	{
		double Es = metutil::Es_(T[i]); // Saturated water vapor pressure
		double E = metutil::E_(MR[i], 100*P[i]);

		double RH = E/Es * 100;
		TD[i] = metutil::DewPointFromRH_(T[i], RH);
	}

	return make_pair(T,TD);
}

pair<vector<double>,vector<double>> si::GetHighestThetaETAndTD(shared_ptr<info> myTargetInfo)
{
	vector<bool> found(myTargetInfo->Data().Size(), false);
	
	vector<double> maxThetaE(myTargetInfo->Data().Size(), -1);
	vector<double> Tsurf(myTargetInfo->Data().Size(), kFloatMissing);
	auto TDsurf = Tsurf;
	
	level curLevel(kHybrid, 137);
	
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
				found[i] = true;
				continue;
			}
			
			double ThetaE = metutil::ThetaE_(T, P*100);

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
				
				if (RH == 0.) 
				{
					RH = 0.1;
				}
				
				TDresult = metutil::DewPointFromRH_(T, RH);
				
				assert(TDresult > 100);
			}
		}
		
		size_t foundCount = count(found.begin(), found.end(), true);

		if (foundCount == found.size())
		{
			break;
		}
		
		itsLogger->Debug("Max ThetaE processed for " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " grid points");

		curLevel.Value(curLevel.Value()-1);
	}
	
	for (size_t i = 0; i < Tsurf.size(); i++)
	{
		if (Tsurf[i] == 0.) Tsurf[i] = kFloatMissing;
		if (TDsurf[i] == 0.) TDsurf[i] = kFloatMissing;
	}

	return make_pair(Tsurf,TDsurf);

}

/* 
 * Namespace CAPE
 * 
 * Holds static functions for integrating different parts of the CAPE area
 */

namespace CAPE
{
himan::point GetPointOfIntersection(const himan::point& a1, const himan::point& a2, const himan::point& b1, const himan::point& b2)
{

	double x1 = a1.X(), x2 = a2.X(), x3 = b1.X(), x4 = b2.X();
	double y1 = a1.Y(), y2 = a2.Y(), y3 = b1.Y(), y4 = b2.Y();

	double d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);

	himan::point null(kFloatMissing,kFloatMissing);

	if (d == 0)
	{
		// parallel lines
		return null;
	}

	double pre = (x1*y2 - y1*x2);
	double post = (x3*y4 - y3*x4);

	// Intersection x & y
	double x = (pre * (x3 - x4) - (x1 - x2) * post) / d;
	double y = (pre * (y3 - y4) - (y1 - y2) * post) / d;

	if (x < min(x1, x2) || x > max(x1, x2) || x < min(x3, x4) || x > max(x3, x4))
	{
		return null;
	}

	if (y < min(y1, y2) || y > max(y1, y2) || y < min(y3, y4) || y > max(y3, y4))
	{
		return null;
	}

	return himan::point(x,y);
}
	
double IntegrateEnteringParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Zenv, double prevZenv)
{
	/*
	 *  We just entered CAPE zone.
	 * 
	 *                                             Hybrid level n == Zenv
	 *                        This point is Tenv --> ======== <-- this point is Tparcel  
	 *                                                \####/
	 *                                                 \##/
	 *                                                  \/  <-- This point is going to be new prevZenv that we get from intersectionWithZ.
	 *                                                  /\      At this point obviously Tenv = Tparcel.
	 *                                                 /  \
	 *          This line is the raising particle --> /    \ <-- This line is the environment temperature
	 *                                               ========
	 *                                             Hybrid level n+1 == prevZenv
	 * 
	 *  We want to calculate only the upper triangle!
	 * 
	 *  Summary:
	 *  1. Calculate intersection of lines in order to get the height of the point where Tparcel == Tenv. This point is going 
	 *     to be the new prevZenv.
	 *  2. Calculate integral using dz = Zenv - prevZenv, for temperatures use the values from Hybrid level n.
    */
	
	using himan::point;
		
	auto intersection = CAPE::GetPointOfIntersection(point(Tenv, Zenv), point(prevTenv, prevZenv), point(Tparcel, Zenv), point(prevTparcel, prevZenv));
	prevZenv = intersection.Y();
		
	double CAPE = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
	
	assert(CAPE >= 0);
	assert(CAPE < 150);
	
	return CAPE;
}

tuple<double,double,double> IntegrateLeavingParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv, double prevPenv, double Zenv, double prevZenv)
{
	/*
	 *  We just left CAPE zone.
	 * 
	 *                                             Hybrid level n == Zenv
	 *                                               ========  
	 *                                                 \  /
	 *                                                  \/  <-- This point is going to be new Zenv that we get from intersectionWithZ.
	 *                                                  /\      At this point obviously Tenv = Tparcel.
	 *                                                 /##\
	 *   This line is the environment temperature --> /####\ <-- this line is the raising particle 
	 *                   This point is prevTenv -->  ========  <-- This point is prevTparcel
	 *                                             Hybrid level n+1 == prevZenv
	 *  
	 *  We want to calculate only the lower triangle!
	 * 
	 *  Summary:
	 *  1. Calculate intersection of lines in order to get the height of the point where Tparcel == Tenv. This point is going 
	 *     to be the new Zenv.
	 *  2. Calculate integral using dz = ZenvNew - prevZenv, for temperatures use the values from Hybrid level n+1.
     */

	using himan::point;

	auto intersectionZ = CAPE::GetPointOfIntersection(point(Tenv, Zenv), point(prevTenv, prevZenv), point(Tparcel, Zenv), point(prevTparcel, prevZenv));
	auto intersectionP = CAPE::GetPointOfIntersection(point(Tenv, Penv), point(prevTenv, prevPenv), point(Tparcel, Penv), point(prevTparcel, prevPenv));

	Zenv = intersectionZ.Y();
	assert(fabs(intersectionZ.X() - intersectionP.X()) < 1.);
	double CAPE = himan::constants::kG * (Zenv - prevZenv) * ((prevTparcel - prevTenv) / prevTenv);

	assert(CAPE >= 0);	
	assert(CAPE < 150);
	
	return make_tuple(CAPE, intersectionP.X(), intersectionP.Y());
}

double IntegrateTemperatureAreaEnteringParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Zenv, double prevZenv, double areaColderLimit, double areaWarmerLimit)
{
	/*
	 * Just entered valid CAPE zone from a non-valid area. 
	 * 
	 * Note! Parcel is buoyant at both areas!
	 *
	 * In this example we entered a cold cape area (-) from a warmer area (+).
	 *
	 *          ##########
	 *           \-----|  
	 *            \----|  
	 *             \---|  
	 *              \++|  
	 *               \+|  
	 *          ##########
	 *        
	 * 
	 *  We want to calculate only the '-' area!
	 * 
	 *  Summary:
	 *  1. Calculate the point where the env temperature crosses to cold area (ie. 263.15K). This point lies somewhere 
	 *     between the two levels, and it's found with linear interpolation. The result should be new value for prevZ. 
	 *     Note that the interpolation is done to the virtual temperatures, so that we don't have to interpolate pressure again!
	 *  2. Sometimes Tparcel is colder than Tenv at that height where Tenv crosser to colder area --> not in CAPE zone. 
	 *     In that case we must find the first height where Tenv >= 263.15 and Tparcel >= Tenv.
	 *  3. Calculate integral using dz = Zenv - prevZenv, for temperatures use the values from Hybrid level n.
	 */

	double areaLimit;
	bool fromWarmerToCold = true;
	
	if (prevTenv > Tenv)
	{
		// Entering area from a warmer zone
		areaLimit = areaWarmerLimit;
	}
	else
	{
		// Entering area from a colder zone
		areaLimit = areaColderLimit;
		fromWarmerToCold = false;
	}

	double newPrevZenv = NFmiInterpolation::Linear(areaLimit, Tenv, prevTenv, Zenv, prevZenv);
	double newTparcel = NFmiInterpolation::Linear(newPrevZenv, Zenv, prevZenv, Tparcel, prevTparcel);

	if (newTparcel < areaLimit)
	{
	   // Tparcel has to be warmer than environment, otherwise no CAPE

	   for (int i = 0; i < 20; i++)
	   {
		   areaLimit += (fromWarmerToCold) ? -0.1 : 0.1;
		   
		   newPrevZenv = NFmiInterpolation::Linear(areaLimit, Tenv, prevTenv, Zenv, prevZenv);
		   newTparcel = NFmiInterpolation::Linear(newPrevZenv, Zenv, prevZenv, Tparcel, prevTparcel);				

		   if (newPrevZenv >= Zenv)
		   {
			   // Lower height reached upper height
			   return 0;
		   }
		   else if (newTparcel >= areaLimit )
		   {
			   // Found correct height
			   break;
		   }
	   }

	   if (newTparcel <= areaLimit)
	   {
		   // Unable to find the height where env temp is cold enough AND Tparcel is warmer than Tenv
		   return 0;
	   }
   }

   assert(Tparcel >= Tenv);
   assert(Zenv >= newPrevZenv);

   double CAPE = himan::constants::kG * (Zenv - newPrevZenv) * ((Tparcel - Tenv) / Tenv);

   assert(Zenv >= prevZenv);
   assert(CAPE >= 0.);
   assert(CAPE < 150.);
   
   return CAPE;
}

double IntegrateTemperatureAreaLeavingParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Zenv, double prevZenv, double areaColderLimit, double areaWarmerLimit)
{
	/* 
	 * Just left valid CAPE zone to a non-valid area. 
	 * 
	 * Note! Parcel is buoyant at both areas!
	 * 
	 *      ##########      ##########
	 *         \==|           \++|
	 *          \==\           \++\
	 *           \--\           \--\
	 *            \-|            \-|
	 *            |-|            |-|
	 *       ##########     ##########
	 *        
	 * 
	 *  We want to calculate only the '-' area!
	 */
	
	double areaLimit;
	bool fromColdToWarmer = true;
	
	if (prevTenv < Tenv)
	{
		// Entering to a warmer area
		areaLimit = areaWarmerLimit;
	}
	else
	{
		// Entering to a colder area
		areaLimit = areaColderLimit;
		fromColdToWarmer = false;
	}

	double newZenv = NFmiInterpolation::Linear(areaLimit, Tenv, prevTenv, Zenv, prevZenv);
	double newTparcel = NFmiInterpolation::Linear(newZenv, Zenv, prevZenv, Tparcel, prevTparcel);

	if (newTparcel <= areaLimit)
	{
		// Tparcel has to be warmer than environment, otherwise no CAPE

		for (int i = 0; i < 20; i++)
		{
			areaLimit += (fromColdToWarmer) ? -0.1 : 0.1;

			newZenv = NFmiInterpolation::Linear(areaLimit, Tenv, prevTenv, Zenv, prevZenv);
			newTparcel = NFmiInterpolation::Linear(newZenv, Zenv, prevZenv, Tparcel, prevTparcel);				

			if (newZenv <= prevZenv)
			{
				// Lower height reached upper height
				return 0;
			}
			else if (newTparcel >= areaLimit )
			{
				// Found correct height
				break;
			}
		}

		if (newTparcel <= areaLimit)
		{
			// Unable to find the height where env temp is cold enough AND Tparcel is warmer than Tenv
			return 0;
		}
	}

	assert(Tparcel >= Tenv);
	assert(newZenv <= Zenv);
	assert(newZenv >= prevZenv);

	double CAPE = himan::constants::kG * (Zenv - prevZenv) * ((newTparcel - areaLimit) / areaLimit);

	assert(CAPE >= 0.);	
	assert(CAPE < 150.);
	
	return CAPE;
}

double IntegrateHeightAreaLeavingParcel(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Zenv, double prevZenv, double areaUpperLimit)
{
	/* 
	 * Just left valid CAPE zone to a non-valid area. 
	 * 
	 * Note! Parcel is buoyant at both areas!
	 * 
	 * In this example parcel is lifted to over 3km.
	 * 
	 *       =========
	 *         \  |
	 *          \  \
	 *           \##\  <-- 3km height
	 *            \#|
	 *            |#|
	 *       =========
	 *        
	 * 
	 *  We want to calculate only the '#' area!
	 */
		
	double newTenv = NFmiInterpolation::Linear(areaUpperLimit, Zenv, prevZenv, Tenv, prevTenv);
	double newTparcel = NFmiInterpolation::Linear(areaUpperLimit, Zenv, prevZenv, Tparcel, prevTparcel);

	if (newTparcel <= newTenv)
	{
		// Tparcel has to be warmer than environment, otherwise no CAPE

		for (int i = 0; i < 20; i++)
		{
			areaUpperLimit -= 10;

			newTenv = NFmiInterpolation::Linear(areaUpperLimit, Zenv, prevZenv, Tenv, prevTenv);
			newTparcel = NFmiInterpolation::Linear(areaUpperLimit, Zenv, prevZenv, Tparcel, prevTparcel);				

			if (areaUpperLimit <= prevZenv)
			{
				// Lower height reached upper height
				return 0;
			}
			else if (newTparcel > newTenv)
			{
				// Found correct height
				break;
			}
		}

		if (newTparcel <= newTenv)
		{
			// Unable to find the height where env temp is cold enough AND Tparcel is warmer than Tenv
			return 0;
		}
	}

	assert(newTparcel >= newTenv);
	assert(areaUpperLimit > prevZenv);

	double CAPE = himan::constants::kG * (areaUpperLimit - prevZenv) * ((prevTparcel - prevTenv) / prevTenv);

	assert(CAPE >= 0.);		
	assert(CAPE < 150);
	
	return CAPE;
}

double Min(const vector<double>& vec)
{ 
	double ret = 1e38;

	for (const double& val : vec)
	{
		if (val != himan::kFloatMissing && val < ret) ret = val;
	}

	if (ret == 1e38) ret = himan::kFloatMissing;

	return ret;
}

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

void AddTo(vector<double>& vec, double incr)
{
	for(double& val : vec)
	{
		if (val != kFloatMissing) val += incr;
	}
}

} // namespace CAPE
