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
#include "metutil.h"
#include <future>
#include "NFmiInterpolation.h"
#include <boost/thread.hpp>

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

double min(const vector<double>& vec)
{ 
	double ret = 1e38;

	for (const double& val : vec)
	{
		if (val != himan::kFloatMissing && val < ret) ret = val;
	}

	if (ret == 1e38) ret = himan::kFloatMissing;

	return ret;
}

double max(const vector<double>& vec)
{ 
	double ret = -1e38;

	for(const double& val : vec)
	{
		if (val != kFloatMissing && val > ret) ret = val;
	}

	if (ret == -1e38) ret = kFloatMissing;

	return ret;
}

void multiply_with(vector<double>& vec, double multiplier)
{
	for(double& val : vec)
	{
		if (val != kFloatMissing) val *= multiplier;
	}
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
const himan::param SB500CAPE1040("CAPE5001040-JKG", 59);
const himan::param SB500CAPE3km("CAPE5003KM-JKG", 59);
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

si::si() : itsBottomLevel(kHPMissingInt), itsSourceData(kUnknown)
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
	
	if (itsConfiguration->Exists("source-data"))
	{
		auto source = itsConfiguration->GetValue("source-data");
		
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
			itsSourceData = kSurface;
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
			itsSourceData = k500mAvgMixingRatio;
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
			itsSourceData = kMaxThetaE;
		}
		else
		{
			throw runtime_error("Invalid source data type: " + source);
		}
	}
	else
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
	}
	
	SetParams(theParams);
	
	Start();
	
}

void DumpVector(const vector<double>& vec, const string& name)
{

	double min = 1e38, max = -1e38, sum = 0;
	size_t count = 0, missing = 0;

	for(const double& val : vec)
	{
		if (val == kFloatMissing)
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

	cout << name << "\tmin " << min << " max " << max << " mean " << mean << " count " << count << " missing " << missing << endl;

	int binn = 10;
	
	double binw = (max-min)/10;

	double binmin = min;
	double binmax = binmin + binw;

	cout << "distribution:" << endl;

	for (int i = 1; i <= binn; i++)
	{
		if (i == binn) binmax += 0.001;

		size_t count = 0;

		for (const double& val : vec)
		{
			if (val == kFloatMissing) continue;

			if (val >= binmin && val < binmax)
			{
				count++;
			}
		}

		if (i == binn) binmax -= 0.001;

		cout << binmin << ":" << binmax << " " << count << std::endl;

		binmin += binw;
		binmax += binw;

	}
}

void si::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger_factory::Instance()->GetLog("siThread #" + boost::lexical_cast<string> (threadIndex));

	myThreadedLogger->Info("Calculating time " + static_cast<string>(myTargetInfo->Time().ValidDateTime()) + " level " + static_cast<string> (myTargetInfo->Level()));

	boost::thread_group g;
	
	switch (itsSourceData)
	{
		case kUnknown:
			g.add_thread(new boost::thread(&si::CalculateVersion, this, boost::ref(myTargetInfo), kSurface));
			g.add_thread(new boost::thread(&si::CalculateVersion, this, boost::ref(myTargetInfo), k500mAvgMixingRatio));
			g.add_thread(new boost::thread(&si::CalculateVersion, this, boost::ref(myTargetInfo), kMaxThetaE));
			break;
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

	myTargetInfo->Param(LFCTParam);
	myTargetInfo->Data().Set(LFC.first);
	
	myTargetInfo->Param(LFCPParam);
	myTargetInfo->Data().Set(LFC.second);

	DumpVector(LFC.first, "LFC T");
	DumpVector(LFC.second, "LFC P");

	// 4. & 5.

	tuple<vector<double>,vector<double>,vector<double>,vector<double>,vector<double>> CAPE;
	vector<double> CIN;

	timer->Start();

	if (false && itsSourceData != kUnknown)
	{

		cout << "\n--- CAPE --\n" << endl;
		boost::thread t1(&si::GetCAPE, this, boost::ref(myTargetInfo), LFC.first, LFC.second);

		cout << "\n--- CIN --\n" << endl;
		boost::thread t2(&si::GetCIN, this, boost::ref(myTargetInfo), TandTD.first, LCL.first, LCL.second, LFC.second);

		t1.join();
		t2.join();
	
	}
	else
	{
		cout << "\n--- CAPE --\n" << endl;
			
		CAPE = GetCAPE(myTargetInfo, LFC.first, LFC.second);
/*
		cout << "\n--- CIN --\n" << endl;
	
		CIN = GetCIN(myTargetInfo, TandTD.first, LCL.first, LCL.second, LFC.second);
 */
	}
	
	timer->Stop();

	mySubThreadedLogger->Info("CAPE and CIN calculated in " + boost::lexical_cast<string> (timer->GetTime()) + " ms");

	myTargetInfo->Param(ELTParam);
	myTargetInfo->Data().Set(get<0> (CAPE));
	
	myTargetInfo->Param(ELPParam);
	myTargetInfo->Data().Set(get<1> (CAPE));

	myTargetInfo->Param(CAPEParam);
	myTargetInfo->Data().Set(get<2> (CAPE));
	
	myTargetInfo->Param(CAPE1040Param);
	myTargetInfo->Data().Set(get<3> (CAPE));

	myTargetInfo->Param(CAPE3kmParam);
	myTargetInfo->Data().Set(get<4> (CAPE));

	DumpVector(get<0> (CAPE), "EL T");
	DumpVector(get<1> (CAPE), "EL P");
	DumpVector(get<2> (CAPE), "CAPE");
	DumpVector(get<3> (CAPE), "CAPE 1040");
	DumpVector(get<4> (CAPE), "CAPE 3km");
/*
	DumpVector(CIN, "CIN");

	myTargetInfo->Param(CINParam);
	myTargetInfo->Data().Set(CIN);
*/
	if (sourceType == kMaxThetaE)
	{
		// If calculating most unstable CAPE, and the value of CAPE is zero, set it to missing
		// with all its helper parameters
		
		myTargetInfo->Param(CAPEParam);
		
		for (myTargetInfo->ResetLocation(); myTargetInfo->NextLocation();)
		{			
			double CAPEval = myTargetInfo->Value();
			
			if (CAPEval <= 0.001)
			{
				myTargetInfo->Value(kFloatMissing);

				myTargetInfo->Param(ELTParam);
				myTargetInfo->Value(kFloatMissing);

				myTargetInfo->Param(ELPParam);
				myTargetInfo->Value(kFloatMissing);
	
				myTargetInfo->Param(CAPE1040Param);
				myTargetInfo->Value(kFloatMissing);

				myTargetInfo->Param(CAPE3kmParam);
				myTargetInfo->Value(kFloatMissing);
				
				myTargetInfo->Param(LFCTParam);
				myTargetInfo->Value(kFloatMissing);
				
				myTargetInfo->Param(LFCPParam);
				myTargetInfo->Value(kFloatMissing);
				
				myTargetInfo->Param(LCLTParam);
				myTargetInfo->Value(kFloatMissing);
				
				myTargetInfo->Param(LCLPParam);
				myTargetInfo->Value(kFloatMissing);
			}
		}
	}
	
}

vector<double> si::GetCIN(shared_ptr<info> myTargetInfo, const vector<double>& Tsurf, const vector<double>& TLCL, const vector<double>& PLCL, const vector<double>& PLFC)
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
	 * This can be done always since LCL is known at all grid points.
	 * 
	 * 2. Integrate from LCL to LFC moist adiabatically
	 * 
	 * This can be done to only those grid points that have LFC defined.
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
	multiply_with(Piter, 100);
	
	auto PLCLPa = PLCL;
	multiply_with(PLCLPa, 100);
	
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
		multiply_with(PenvVec, 100);

		metutil::LiftLCL(&Piter[0], &Titer[0], &PLCLPa[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());
		
		int i = -1;
		//LOCKSTEP(TenvInfo, PenvInfo, ZenvInfo, prevZenvInfo, basePenvInfo)
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
			
			if (Penv >= PLCL[i] && PLFC[i] == kFloatMissing)
			{
				found[i] = true;
				continue;
			}
			
			if (Penv <= PLFC[i])
			{
				// reached max height
				// TODO: final piece integration
				found[i] = true;
				continue;
			}
			else if (Penv > PLCL[i])
			{
				Tparcel = metutil::DryLift_(Pbase*100, Tsurf[i], Penv * 100);
			}
			else
			{
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
				// cape layer, no more CIN
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

			if (found[i] & FCAPE) Titer[i] = kFloatMissing; // by setting this we prevent MoistLift to integrate particle

		}
	} 

	return cinh;
	
}

double CalcCAPE1040(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv, double prevPenv, double Zenv, double prevZenv)
{
	double C = 0;

	if (Tparcel >= Tenv)
	{
		if (Tenv >= 233.15 && Tenv <= 263.15 && prevTenv > 263.15)
		{
			// Just entered cold CAPE zone, the direction is hard coded from warmer to colder

			if (prevTparcel == kFloatMissing)
			{
				Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv*100);
				Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);

				return himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
			}

			Zenv = NFmiInterpolation::Linear(263.15, Tenv, prevTenv, Zenv, prevZenv);
			Penv = NFmiInterpolation::Linear(263.15, Tenv, prevTenv, Penv, prevPenv);
			Tparcel = NFmiInterpolation::Linear(263.15, Tenv, prevTenv, Tparcel, prevTparcel);

			Tenv = himan::metutil::VirtualTemperature_(263.15, Penv*100);
			Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);

			// problem : 
			// interpolating Tparcel to same level as env 263.15 makes Tparcel temperature
			// lower than 263.15 --> not in CAPE zone

			// This could be solved by integrating Tparcel to exactly level where Tenv is 263.15
			// but that might be expensive

			if (Tparcel >= Tenv)
			{
				Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv*100);
				Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);

				C = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
				assert(Zenv >= prevZenv);
				assert(C >= 0.);
			}
		}
		else if (prevTenv >= 233.15 && prevTenv <= 263.15 && Tenv < 233.15)
		{
			// Just exited cold CAPE zone to an even colder area

			if (prevTparcel == kFloatMissing)
			{
				return 0;
			}
			
			Zenv = NFmiInterpolation::Linear(233.15, prevTenv, Tenv, prevZenv, Zenv);
			Penv = NFmiInterpolation::Linear(233.15, prevTenv, Tenv, prevPenv, Penv);
			Tparcel = NFmiInterpolation::Linear(233.15, prevTenv, Tenv, prevTparcel, Tparcel);

			Tenv = himan::metutil::VirtualTemperature_(233.15, Penv*100);


		}
		else if (Tenv >= 233.15 && Tenv <= 263.15)
		{
			// Firmly in the cold CAPE zone
			C = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
		}
	}
	else
	{
		// Out of general CAPE zone
		// Calculate average between two levels to get an approximate height

		Tenv = (Tenv + prevTenv) * 0.5;
		Penv = (Penv + prevPenv) * 0.5;
		Zenv = (Zenv + prevZenv) * 0.5;
		
		if (Tenv >= 233.15 && Tenv <= 263.15)
		{
			Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv*100);
			Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);

			if (Tparcel >= Tenv)
			{
				// Approximation worked and particle is still warmer than environment in the 
				// -10 .. -40 temperature zone
				C = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
				assert(C >= 0);
			}
		}
	}
	
	return C;
}

double CalcCAPE3km(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv, double prevPenv, double Zenv, double prevZenv)
{
	double C= 0.;

	if (Tparcel >= Tenv)
	{
		// In general CAPE zone

		if (Zenv <= 3000.)
		{
			Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv*100);
			Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);

			assert(Tparcel >= Tenv);
			C = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
			assert(C >= 0);
		}
		else if (prevZenv <= 3000.)
		{
			// Interpolate the final piece of CAPE area just below 3000m		
			// Interpolate without virtual temp

			if (prevTparcel == kFloatMissing)
			{
				// Unable to interpolate value since previous temperature is missing
				return 0;
			}
			
			Tparcel = NFmiInterpolation::Linear(3000., prevZenv, Zenv, prevTparcel, Tparcel);
			Tenv = NFmiInterpolation::Linear(3000., prevZenv, Zenv, prevTenv, Tenv);
			Penv = NFmiInterpolation::Linear(3000., prevZenv, Zenv, prevPenv, Penv);

			Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv*100);
			Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);

			if (Tparcel >= Tenv)
			{
				C = himan::constants::kG * (3000. - prevZenv) * ((Tparcel - Tenv) / Tenv);
				assert(C >= 0);
			}
		}
	}
	else if (prevZenv < 3000.)
	{
		// Out of general CAPE zone
		// Calculate average between two levels to get an approximate height

		Tenv = (Tenv + prevTenv) * 0.5;
		Penv = (Penv + prevPenv) * 0.5;
		Zenv = (Zenv + prevZenv) * 0.5;

		if (Zenv < 3000.)
		{
			// If average Zenv is higher than 3000m, we cannot calculate the missing
			// part of CAPE zone since the location is only an approximation.

			Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv*100);
			Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);
		
			if (Tparcel >= Tenv)
			{
				// Approximation worked and particle is still warmer than environment
				C = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
				assert(C >= 0);
			}
		}		
	}

	return C;
}


double CalcCAPE(double Tenv, double prevTenv, double Tparcel, double prevTparcel, double Penv, double prevPenv, double Zenv, double prevZenv)
{
	double C= 0.;

	if (Tparcel >= Tenv)
	{
		Tenv = himan::metutil::VirtualTemperature_(Tenv, Penv*100);
		Tparcel = himan::metutil::VirtualTemperature_(Tparcel, Penv*100);

		assert(Tparcel >= Tenv);

		C = himan::constants::kG * (Zenv - prevZenv) * ((Tparcel - Tenv) / Tenv);
	}

	return C;
}
			
tuple<vector<double>, vector<double>, vector<double>, vector<double>, vector<double>> si::GetCAPE(shared_ptr<info> myTargetInfo, const vector<double>& T, const vector<double>& P)
{
	assert(T.size() == P.size());

	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	vector<unsigned char> found(T.size(), 0);
	
	// CAPE is initialized to -1: in the data we make a difference between
	// zero CAPE and unknown CAPE (kFloatMissing)

	vector<double> CAPE(T.size(), -1);
	vector<double> CAPE1040(T.size(), -1);
	vector<double> CAPE3km(T.size(), -1);
	vector<double> ELT(T.size(), kFloatMissing);
	vector<double> ELP(T.size(), kFloatMissing);

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
		
	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::max(P));
	
	level curLevel = levels.first;
	
	auto prevZenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);
	auto prevTenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
	auto prevPenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
	
	curLevel.Value(curLevel.Value() - 1);	
	
	auto Piter = P, Titer = T; // integration variables
	auto prevTparcelVec = Titer;
	
	// Convert pressure to Pa since metutil-library expects that
	multiply_with(Piter, 100);

	while (curLevel.Value() > 60 && foundCount != found.size())
	{
		// Get environment temperature, pressure and height values for this level
		auto PenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
		auto TenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto ZenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);

		// Convert pressure to Pa since metutil-library expects that
		auto PenvVec = PenvInfo->Data().Values();
		multiply_with(PenvVec, 100);

		vector<double> TparcelVec(P.size(), kFloatMissing);

		metutil::MoistLift(&Piter[0], &Titer[0], &PenvVec[0], &TparcelVec[0], TparcelVec.size());
	
		int i = -1;
		for (auto&& tup : zip_range(VEC(PenvInfo), VEC(ZenvInfo), VEC(prevZenvInfo), VEC(prevTenvInfo), VEC(prevPenvInfo), VEC(TenvInfo), TparcelVec, prevTparcelVec))
		//LOCKSTEP(PenvInfo, ZenvInfo, prevZenvInfo, prevTenvInfo, prevPenvInfo, TenvInfo)
		{

			i++;
			// CAPE is a superset of CAPE1040 and CAPE3km
			if (found[i] & FCAPE) continue;

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
			
			if (Penv == kFloatMissing || Tenv == kFloatMissing || Zenv == kFloatMissing || prevZenv == kFloatMissing || Tparcel == kFloatMissing || Penv > P[i])
			{
				// Missing data or current grid point is below LFC
				continue;
			}

			if (prevZenv >= 3000. && Zenv >= 3000.)
			{
				found[i] |= FCAPE3km;
			}
			
			if (prevTenv < 200. && Tenv < 200.)
			{
				found[i] |= FCAPE1040;
			}
			
			if ((found[i] & FCAPE3km) == 0)
			{
				double C = CalcCAPE3km(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);

				CAPE3km[i] = max(CAPE3km[i], 0.);
				CAPE3km[i] += C;
				
				assert(CAPE3km[i] < 3000.);
				assert(CAPE3km[i] >= -1.);
			}

			if ((found[i] & FCAPE1040) == 0)
			{
				double C = CalcCAPE1040(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);
					
				CAPE1040[i] = max(CAPE1040[i], 0.);					
				CAPE1040[i] += C;

				assert(CAPE1040[i] < 5000.);
				assert(CAPE1040[i] >= -1.);
			}
			
			double C = CalcCAPE(Tenv, prevTenv, Tparcel, prevTparcel, Penv, prevPenv, Zenv, prevZenv);
			
			if (C >= 0)
			{
				CAPE[i] = max(CAPE[i], 0.);					
				CAPE[i] += C;
			}
			
			assert(C >= 0.);				
			assert(CAPE[i] < 8000);
	
			if (Tparcel <= Tenv && CAPE[i] != -1)
			{
				// We are exiting CAPE zone

				found[i] |= FCAPE;

				if (prevTparcel == kFloatMissing)
				{
					// Nothing to do here, we have never entered CAPE zone
					continue;
				}

				// Do simple linear interpolation to get EL values
				// EL is the estimate of the exact value where CAPE zone ends

				double _ELT = (Tenv + prevTenv) * 0.5;
				double _ELP = (Penv + prevPenv) * 0.5;

				ELP[i] = _ELP;
				ELT[i] = _ELT;	

				// Interpolate the final piece of CAPE area between previous level and EL
				_ELT = metutil::VirtualTemperature_(_ELT, _ELP*100); // environment
					
				// Linear interpolation of parcel temperature
				Tparcel = (Tparcel + prevTparcel) * 0.5;			
				Tparcel = metutil::VirtualTemperature_(Tparcel, _ELP*100);

				if(Tparcel >= _ELT)
				{
					// Linear interpolation of parcel height
					Zenv = (Zenv + prevZenv) * 0.5;

					double C = constants::kG * (Zenv - prevZenv) * ((Tparcel - _ELT) / _ELT);
					CAPE[i] += C;
				}
				
				assert(CAPE[i] < 8000);

			}
		}

		curLevel.Value(curLevel.Value() - 1);		

		itsLogger->Debug("CAPE read " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " gridpoints");
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
			
			if (found[i] & FCAPE) Titer[i] = kFloatMissing; // by setting this we prevent MoistLift to integrate particle

		}
	}

	for (size_t i = 0; i < CAPE.size();i++)
	{
		if (CAPE[i] == -1) CAPE[i] = kFloatMissing;
		if (CAPE3km[i] == -1) CAPE3km[i] = kFloatMissing;
		if (CAPE1040[i] == -1) CAPE1040[i] = kFloatMissing;
	}
	
	return make_tuple (ELT, ELP, CAPE, CAPE1040, CAPE3km);
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

	auto TenvLCL = h->VerticalValue(param("T-K"), P);

	auto Piter = P, Titer = T; // integration variables
	
	// Convert pressure to Pa since metutil-library expects that
	multiply_with(Piter, 100);

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

	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::max(P));

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
		multiply_with(PenvVec, 100);
		
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
		Presult = 0.01 * ((lcl.P > P) ? P : lcl.P); // hPa
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
		throw runtime_error("Surface temperature and/or dewpoint not found");
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

#if 1
	itsLogger->Info("Calculating T&Td in smarttool compatibility mode");
	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	tp.HeightInMeters(false);
	mr.HeightInMeters(false);
	
	auto PInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
	auto P = PInfo->Data().Values();	

	tp.LowerHeight(P);
	mr.LowerHeight(P);

	auto P500m = h->VerticalValue(param("P-HPA"), 500.);

	tp.UpperHeight(P500m);
	mr.UpperHeight(P500m);
	
	vector<bool> found(myTargetInfo->Data().Size(), false);

	while (true)
	{
		auto T = h->VerticalValue(param("T-K"), P);
		
		vector<double> Tpot(T.size(), kFloatMissing);
		vector<double> MR(T.size(), kFloatMissing);

		for (size_t i = 0; i < T.size(); i++)
		{
			if (found[i]) continue;

			Tpot[i] = metutil::Theta_(T[i], 100*P[i]);
			MR[i] = metutil::MixingRatio_(T[i], 100*P[i]);
		}
		DumpVector(MR, "mr");

		tp.Process(Tpot, P);
		mr.Process(MR, P);

		auto Z = h->VerticalValue(param("HL-M"), P);
		
		for (size_t i = 0; i < Z.size(); i++)
		{
			if (found[i]) continue;
			if (Z[i] > 500.) found[i] = true;
		}
		
		auto foundCount = static_cast<size_t> (count(found.begin(), found.end(), true));
		itsLogger->Debug("Data read " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " gridpoints");

		if (foundCount == found.size()) break;

		transform(P.begin(), P.end(), P.begin(), bind2nd(minus<double>(), 1.0));
	}
		
#else
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
			Tpot[i] = metutil::Theta_(T[i], 100*P[i]);
			MR[i] = metutil::MixingRatio_(T[i], 100*P[i]);
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

	// Need surface pressure

	const params PParams({param("PGR-PA"), param("P-PA")});

	auto Psurf = Fetch(myTargetInfo->Time(), level(kHeight, 0), PParams, myTargetInfo->ForecastType(), false);
	P = Psurf->Data().Values();

	vector<double> T(Tpot.size(), kFloatMissing);			

	for (size_t i = 0; i < Tpot.size(); i++)
	{
		T[i] = Tpot[i] * pow((P[i]/100000.), 0.2854);
	}

	vector<double> TD(T.size(), kFloatMissing);

	for (size_t i = 0; i < MR.size(); i++)
	{
		double Es = metutil::Es_(T[i]); // Saturated water vapor pressure
		double E = metutil::E_(MR[i], P[i]);

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
			throw kFileDataNotFound;
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
