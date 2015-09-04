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
#include <boost/foreach.hpp>
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

using namespace std;
using namespace himan::plugin;

double min(const vector<double>& vec)
{ 
	double ret = 1e38;

	BOOST_FOREACH(const double& val, vec)
	{
		if (val != himan::kFloatMissing && val < ret) ret = val;
	}

	if (ret == 1e38) ret = himan::kFloatMissing;

	return ret;
}

double max(const vector<double>& vec)
{ 
	double ret = -1e38;

	BOOST_FOREACH(const double& val, vec)
	{
		if (val != kFloatMissing && val > ret) ret = val;
	}

	if (ret == -1e38) ret = kFloatMissing;

	return ret;
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
const himan::param MUCAPE1040("CAPEMU1040-JKG", 59);
const himan::param MUCAPE3km("CAPEMU3KM-JKG", 59);
const himan::param MUCIN("CINMU-JKG", 66);

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
	theParams.push_back(SB500CIN);

	theParams.push_back(MULCLT);
	theParams.push_back(MULCLP);
	theParams.push_back(MULFCT);
	theParams.push_back(MULFCP);
	theParams.push_back(MUELT);
	theParams.push_back(MUELP);
	theParams.push_back(MUCAPE);
	theParams.push_back(MUCIN);
	
	SetParams(theParams);
	
	Start();
	
}

void DumpVector(const vector<double>& vec, const string& name)
{

	double min = 1e38, max = -1e38, sum = 0;
	size_t count = 0, missing = 0;

	BOOST_FOREACH(double val, vec)
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

}

void si::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger_factory::Instance()->GetLog("siThread #" + boost::lexical_cast<string> (threadIndex));

	myThreadedLogger->Info("Calculating time " + static_cast<string>(myTargetInfo->Time().ValidDateTime()) + " level " + static_cast<string> (myTargetInfo->Level()));
	
	// Spinoff thread calculates surface data
	
	//boost::thread t1(&si::CalculateVersion, this, boost::ref(myTargetInfo), kSurface);
	
	// Spinoff thread calculate 500m data
	
	boost::thread t2(&si::CalculateVersion, this, boost::ref(myTargetInfo), k500mAvgMixingRatio);
	
	// Spinoff thread calculates most unstable data
	
	//boost::thread t3(&si::CalculateVersion, this, boost::ref(myTargetInfo), kMaxThetaE);
	
	//t1.join(); 
	t2.join(); 
	//t3.join();
	
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
	
	DumpVector(get<0>(TandTD), "T");
	DumpVector(get<1>(TandTD), "TD");
	
	// 2.
	
	cout << "\n--- LCL --\n" << endl;
	
	auto LCL = GetLCL(myTargetInfo, TandTD.first, TandTD.second);
	
	myTargetInfo->Param(LCLTParam);
	myTargetInfo->Data().Set(LCL.first);
	
	myTargetInfo->Param(LCLPParam);
	myTargetInfo->Data().Set(LCL.second);
	
	DumpVector(LCL.first, "LCL T");
	DumpVector(LCL.second, "LCL P");

	// 3.
	
	cout << "\n--- LFC --\n" << endl;

	auto LFC = GetLFC(myTargetInfo, LCL.first, LCL.second);

	myTargetInfo->Param(LFCTParam);
	myTargetInfo->Data().Set(LFC.first);
	
	myTargetInfo->Param(LFCPParam);
	myTargetInfo->Data().Set(LFC.second);

	DumpVector(LFC.first, "LFC T");
	DumpVector(LFC.second, "LFC P");

	// 4.

 	cout << "\n--- CAPE --\n" << endl;
	
	auto CAPE = GetCAPE(myTargetInfo, LFC.first, LFC.second, kCAPE);

	myTargetInfo->Param(CAPEParam);
	myTargetInfo->Data().Set(get<2> (CAPE));
	
	myTargetInfo->Param(ELTParam);
	myTargetInfo->Data().Set(get<0> (CAPE));
	
	myTargetInfo->Param(ELPParam);
	myTargetInfo->Data().Set(get<1> (CAPE));

	CAPE = GetCAPE(myTargetInfo, LFC.first, LFC.second, kCAPE1040);

	myTargetInfo->Param(CAPE1040Param);
	myTargetInfo->Data().Set(get<2> (CAPE));

	CAPE = GetCAPE(myTargetInfo, LFC.first, LFC.second, kCAPE3km);

	myTargetInfo->Param(CAPE3kmParam);
	myTargetInfo->Data().Set(get<2> (CAPE));

	// 5. 

	cout << "\n--- CIN --\n" << endl;
	
	auto CIN = GetCIN(myTargetInfo, TandTD.first, LCL.first, LCL.second, LFC.second);

	DumpVector(CIN, "CIN");
	myTargetInfo->Param(CINParam);
	myTargetInfo->Data().Set(CIN);

}

vector<double> si::GetCIN(shared_ptr<info> myTargetInfo, const vector<double>& Tsurf, const vector<double>& TLCL, const vector<double>& PLCL, const vector<double>& PLFC)
{
	const params PParams({param("PGR-PA"), param("P-PA")});
	
	auto PsurfInfo = Fetch(myTargetInfo->Time(), level(kHeight, 0), PParams, myTargetInfo->ForecastType(), false);
	
	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	vector<bool> found(Tsurf.size(), false);

	forecast_time ftime = myTargetInfo->Time();
	forecast_type ftype = myTargetInfo->ForecastType();

	modifier_plusminusarea m;

	/*
	 * Modus operandi:
	 * 
	 * 1. Integrate from ground to LCL dry adiabatically
	 * 2. Integrate from LCL to LFC moist adiabatically
	 * 
	 * Use modifier_plusminusarea for integration.
	 */
	
	// Get LCL and LFC heights in meters

	DumpVector(PLCL, "PLCL");

	auto ZLCL = h->VerticalValue(param("HL-M"), PLCL);
	auto ZLFC = h->VerticalValue(param("HL-M"), PLFC);

	itsLogger->Debug("Fetching LCL metric height");

	DumpVector(ZLCL, "ZLCL");
	
	itsLogger->Debug("Fetching LFC metric height");
	DumpVector(ZLFC, "ZLFC");

	// If LFC is not defined for some points, do not calculate CIN for
	// those either.
	
	for (size_t i = 0; i < PLFC.size(); i++)
	{
		if (ZLCL[i] == kFloatMissing || ZLFC[i] == kFloatMissing)
		{
			found[i] = true;
			ZLCL[i] = kFloatMissing;
			ZLFC[i] = kFloatMissing;
		}
	}
	
	vector<double> zero (PLCL.size(), 0);
	m.LowerHeight(zero);
	m.UpperHeight(ZLCL);
	
	level curLevel(kHybrid, 137);
	
	//while (!m.CalculationFinished())
	
	info_t prevZInfo;
	std::vector<double> cinh(PLCL.size(), 0);
	
	for (size_t i=0; i<found.size();i++) if (found[i]) cinh[i] = kFloatMissing;
	
	// For moist lift we need thetaE
	
	vector<double> thetaE(PLFC.size(), kFloatMissing);
	
	for (size_t i = 0; i < thetaE.size() && PsurfInfo->NextLocation(); i++)
	{
		if (TLCL[i] != kFloatMissing && PLCL[i] != kFloatMissing && PLFC[i] != kFloatMissing )
		{
			thetaE[i] = metutil::ThetaE_(TLCL[i], 100*PLCL[i]);
		}
	}

	while (true)
	{
		auto ZenvInfo = Fetch(ftime, curLevel, param("HL-M"), ftype, false);
		
		if (curLevel.Value() == 137)
		{
			prevZInfo = ZenvInfo;
			curLevel.Value(curLevel.Value()-1);
			continue;
		}
		
		auto TenvInfo = Fetch(ftime, curLevel, param("T-K"), ftype, false);
		auto PenvInfo = Fetch(ftime, curLevel, param("P-HPA"), ftype, false);
		
		vector<double> Tdiff(PLCL.size(), kFloatMissing);
		
		LOCKSTEP(TenvInfo, PenvInfo, ZenvInfo, PsurfInfo, prevZInfo)
		{
			size_t i = TenvInfo->LocationIndex();

			if (found[i]) continue;

			double Tenv = TenvInfo->Value();

			double Tparcel = kFloatMissing;
			
			if (PenvInfo->Value() <= PLFC[i])
			{
				// reached max height
				found[i] = true;
				continue;
			}
			else if (PenvInfo->Value() > PLCL[i])
			{
				Tparcel = metutil::DryLift_(PsurfInfo->Value(), Tsurf[i], PenvInfo->Value() * 100);
			}
			else
			{
				Tparcel = metutil::Tw_(thetaE[i], PenvInfo->Value() * 100);
				if (Tparcel == kFloatMissing) continue;
				
				Tparcel = metutil::VirtualTemperature_(Tparcel, PenvInfo->Value() * 100);
				Tenv = metutil::VirtualTemperature_(Tenv, PenvInfo->Value() * 100);
			}

			if (Tparcel <= Tenv)
			{
				cinh[i] += constants::kG * (ZenvInfo->Value() - prevZInfo->Value()) * ((Tparcel - Tenv) / Tenv);
				assert(cinh[i] <= 0);
			}
			else if (cinh[i] != 0)
			{
				// cape layer, no more CIN
				found[i] = true;
			}
		}

		size_t numfound = static_cast<unsigned int> (count(found.begin(), found.end(), true));
		
		std::cout << "cinh done for " << numfound << "/" << found.size() << " gridpoints level " << curLevel.Value() << "\n";
		
		if (numfound == found.size())
		{
			break;
		}

		curLevel.Value(curLevel.Value()-1);
		prevZInfo = ZenvInfo;
	} 
	
	return cinh;
	
}

tuple<vector<double>, vector<double>, vector<double>> si::GetCAPE(shared_ptr<info> myTargetInfo, const vector<double>& T, const vector<double>& P, HPSoundingIndexCAPEVariation CAPEVariation)
{
	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	vector<bool> found(T.size(), false);
	vector<double> CAPE(T.size(), -1);
	vector<double> ELT(T.size(), kFloatMissing);
	vector<double> ELP(T.size(), kFloatMissing);

	// Use Davies-Jones Tw formula 
	
	auto thetaE = P;

	for (size_t i = 0; i < thetaE.size(); i++)
	{
		if (T[i] != kFloatMissing && P[i] != kFloatMissing)
		{
			thetaE[i] = metutil::ThetaE_(T[i], 100*P[i]);
		}
	}
	
	// For each grid point find next hybrid level that's below the LFC
		
	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::max(P));
	
	level curLevel = levels.first;
	curLevel.Value(curLevel.Value() - 1);
	
	size_t foundCount = count(found.begin(), found.end(), true);

	info_t prevZInfo, prevTInfo, prevPInfo;
	std::vector<double> prevTw;

	while (curLevel.Value() > 45 && foundCount != found.size())
	{
		
		cout << "Current level: " << curLevel.Value() << "\n";
	
		if (!prevZInfo)
		{
			prevZInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);
			prevTInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
			prevPInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
			
			curLevel.Value(curLevel.Value() - 1);		

			continue;
		}
		
		// Get environment temperature values

		auto PenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
		auto TenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto ZenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("HL-M"), myTargetInfo->ForecastType(), false);
		
		// Penv is hPa
		
		vector<double> Penv(P.size(), kFloatMissing);

		for (size_t i = 0; i < P.size() && PenvInfo->NextLocation(); i++)
		{
			if (PenvInfo->Value() == kFloatMissing) continue;
			
			Penv[i] = PenvInfo->Value() * 100;
		}
		
		vector<double> Tw(P.size(), kFloatMissing);
		
		metutil::Tw(&thetaE[0], &Penv[0], &Tw[0], thetaE.size());
		
		assert(T.size() == P.size());
		
		LOCKSTEP(PenvInfo, ZenvInfo, prevZInfo, prevTInfo, prevPInfo, TenvInfo)
		{
			size_t i = PenvInfo->LocationIndex();

			if (found[i]) continue;

			double Tenv = TenvInfo->Value();
			double _Penv = Penv[i];
			
			double Zenv = ZenvInfo->Value();
			double ZenvPrev = prevZInfo->Value();
			double _Tw = Tw[i];
				
			if (_Penv == kFloatMissing || Tenv == kFloatMissing || Zenv == kFloatMissing || ZenvPrev == kFloatMissing || _Tw == kFloatMissing)
			{
				found[i] = true;
				continue;
			}
			
			if (_Penv > P[i]*100) 
			{
				// Current grid point is below LFC
				continue;
			}

			if (CAPEVariation == kCAPE3km && Zenv > 3000.)
			{
				// Interpolate the final piece of CAPE area just below 3000m
				assert(prevTw.size());
				_Tw = NFmiInterpolation::Linear(3000., ZenvPrev, Zenv, prevTw[i], _Tw);
				Tenv = NFmiInterpolation::Linear(3000., ZenvPrev, Zenv, prevTInfo->Value(), Tenv);
				_Penv = NFmiInterpolation::Linear(3000., ZenvPrev, Zenv, prevPInfo->Value(), _Penv);

				Tenv = metutil::VirtualTemperature_(Tenv, _Penv);
				_Tw = metutil::VirtualTemperature_(_Tw, _Penv);

				if (CAPE[i] == -1) CAPE[i] = 0;

				CAPE[i] += constants::kG * (3000. - ZenvPrev) * ((_Tw - Tenv) / Tenv);
				
				found[i] = true;
				continue;
			}
			
			if (CAPEVariation == kCAPE1040 && (Tenv < 233.15 || Tenv > 263.15))
			{
				continue;
			}

			if (_Tw >= Tenv)
			{
				Tenv = metutil::VirtualTemperature_(Tenv, _Penv);
				_Tw = metutil::VirtualTemperature_(_Tw, _Penv);

				if (CAPE[i] == -1) CAPE[i] = 0;

				CAPE[i] += constants::kG * (Zenv - ZenvPrev) * ((_Tw - Tenv) / Tenv);
				assert(CAPE[i] < 8000);
			}
			else 
			{
				// Do simple linear interpolation to get EL values

				double _ELT = (Tenv + prevTInfo->Value()) / 2;
				double _ELP = (prevPInfo->Value() + _Penv) / 2;
				
				if (CAPEVariation == kCAPE)
				{
					ELP[i] = _ELP;
					ELT[i] = _ELT;	
				}

				// Interpolate the final piece of CAPE area just below EL
				assert(prevTw.size());

				_ELT = metutil::VirtualTemperature_(_ELT, _ELP);
				_Tw = metutil::VirtualTemperature_((_Tw + prevTw[i])/2, _Penv);
	
				if (CAPE[i] == -1) CAPE[i] = 0;

				CAPE[i] += constants::kG * 0.5 * (Zenv - ZenvPrev) * ((_Tw - _ELT) / _ELT);
				assert(CAPE[i] < 8000);
				
				found[i] = true;
			}
		}
		//DumpVector(CAPE, "CAPE");
		curLevel.Value(curLevel.Value() - 1);		
		
		foundCount = count(found.begin(), found.end(), true);
		itsLogger->Debug("CAPE read " + boost::lexical_cast<string> (foundCount) + "/" + boost::lexical_cast<string> (found.size()) + " gridpoints");
		prevZInfo = ZenvInfo;
		prevTInfo = TenvInfo;
		prevPInfo = PenvInfo;
		prevTw = Tw;
	}/*
	for (size_t i = 0;i< CAPE.size();i++){
		if(CAPE[i] > 7000){
			std::cout << "CAPE " << CAPE[i] << "@" << i << std::endl;exit(1);
		}
	}*/

	for (size_t i = 0;i< CAPE.size();i++) if (CAPE[i] == -1) CAPE[i] = kFloatMissing;

	return make_tuple (ELT, ELP, CAPE);
}

pair<vector<double>,vector<double>> si::GetLFC(shared_ptr<info> myTargetInfo, vector<double>& T, vector<double>& P)
{
	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->HeightUnit(kHPa);

	vector<bool> found(T.size(), false);
	
	vector<double> LFCT(T.size(), kFloatMissing);
	vector<double> LFCP(T.size(), kFloatMissing);
	
	auto f = GET_PLUGIN(fetcher);
	
	// Check LCL conditions, if LCL = LFC
	
	auto TenvLCL = h->VerticalValue(param("T-K"), P);
	
	for (size_t i = 0; i < TenvLCL.size(); i++)
	{
		if ((T[i] >= TenvLCL[i]) || (TenvLCL[i] - T[i]) < 0.01) // fuzzy factor
		{
			found[i] = true;
			LFCT[i] = T[i];
			LFCP[i] = P[i];
		}
	}

	itsLogger->Debug("Found " + boost::lexical_cast<string> (count(found.begin(), found.end(), true)) + " gridpoints that have LCL=LFC");

	auto Pint = P;
	
	auto thetaE = P;

	for (size_t i = 0; i < thetaE.size(); i++)
	{
		thetaE[i] = metutil::ThetaE_(T[i], 100*P[i]);
	}
		
	auto levels = h->LevelForHeight(myTargetInfo->Producer(), ::max(P));
	
	level curLevel = levels.first;
	curLevel.Value(curLevel.Value());

	size_t foundCount = count(found.begin(), found.end(), true);

	auto prevPInfo = Fetch(myTargetInfo->Time(), level(kGround, 0), param("P-PA"), myTargetInfo->ForecastType(), false);
	auto prevTInfo = Fetch(myTargetInfo->Time(), level(kGround, 0), param("T-K"), myTargetInfo->ForecastType(), false);
		
	while (curLevel.Value() > 55 && foundCount != found.size())
	{	
	
		// Get environment temperature values

		auto TenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto PenvInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);

		// Penv is hPa
		
		vector<double> Penv(PenvInfo->Data().Size(), kFloatMissing);

		PenvInfo->ResetLocation();
		for (size_t i = 0; PenvInfo->NextLocation(); i++)
		{
			double v = PenvInfo->Value();
			
			if (v != kFloatMissing) Penv[i] = (v * 100);
		}
		
		vector<double> Tw(P.size(), kFloatMissing);
		
		metutil::Tw(&thetaE[0], &Penv[0], &Tw[0], thetaE.size());
	
		TenvInfo->ResetLocation();
		PenvInfo->ResetLocation();
		
		assert(T.size() == P.size());

		LOCKSTEP(TenvInfo, PenvInfo, prevPInfo, prevTInfo)
		{
			size_t i = TenvInfo->LocationIndex();

			if (found[i]) continue;

			double Penv = PenvInfo->Value();
			double Tenv = TenvInfo->Value();

//#define SMARTTOOL_COMPATIBILITY

#ifndef SMARTTOOL_COMPATIBILITY
			// In smarttool:
			// " jos löytyi LFC, mutta sen arvo on alempana kuin LCL, laitetaan  LFC:n arvoksi LCL"

			if (Penv > P[i]) 
			{
				continue;
			}
#endif

			if (Tw[i] > Tenv)
			{
				
				found[i] = true;
				LFCT[i] = (Tenv + prevTInfo->Value()) / 2;
				
				// Never allow LFC pressure to be bigger than LCL pressure
				
				double prevP = prevPInfo->Value();
				if (prevPInfo->Param().Name() == "P-PA") prevP *= 0.01;
				if (prevP >= P[i]) prevP = P[i];
				
				LFCP[i] = (Penv + prevP) / 2;

			}
		}
		
		curLevel.Value(curLevel.Value() - 1);		
		
		foundCount = count(found.begin(), found.end(), true);

		prevPInfo = PenvInfo;
		prevTInfo = TenvInfo;
	}
	
#ifdef SMARTTOOL_COMPATIBILITY
	
	for (size_t i = 0; i < T.size(); i++)
	{
		if (LFCP[i] != kFloatMissing && LFCP[i] > P[i])
		{
			LFCP[i] = P[i];
			LFCT[i] = T[i];
		}
	}
#endif
	return make_pair(LFCT, LFCP);
}

#if 0
namespace smarttool {

const double gTMR_alfa = 0.0498646455;
const double gTMR_beta = 2.4082965;
const double gTMR_gamma = 0.0915;
const double gTMR_gamma2 = 38.9114;
const double gTMR_gamma3 = 1.2035;
const double gTpot2tConstant1 = 0.2854;
const double gKelvinChange = 273.16;



double CalcRH(double T, double Td)
{
	double RH = 100 * ::pow((112-0.1*T+Td)/(112+0.9*T) ,8);
	return RH;
}

double CalcE(double RH, double es)
{
	double e = RH * es / 100;
	return e;
}

double CalcEs2(double Tcelsius)
{
	const double b = 17.2694;
	const double e0 = 6.11; // 6.11 <- 0.611 [kPa]
	const double T1 = 273.16; // [K]
	const double T2 = 35.86; // [K]

	double T = Tcelsius + T1;
	double nume = b * (T-T1);
	double deno = (T-T2);

	double es = e0 * ::exp(nume/deno);
	return es;
}

double CalcW(double e, double P)
{
	double w = 0.622 * e/P * 1000;
	return w;
}

double CalcMixingRatio(double T, double Td, double P)
{
	double RH = CalcRH(T, Td);
	double es = CalcEs2(T);
	double e = CalcE(RH, es);
	double w = CalcW(e, P);
	return w;
}

double Tpot2t(double tpot, double p)
{
	// HUOM! pot lämpötila muutetaan ensin kelvineiksi ja lopuksi tulos muutetaan takaisin celsiuksiksi
	return ( (gKelvinChange + tpot) * ::pow(p/1000, gTpot2tConstant1) ) - gKelvinChange;
}

double T2tpot(double T, double P)
{
	const double T0 = 273.16; // kelvin asteikon muunnos
	return ((T+T0) * ::pow(1000/P, 0.2854)) - T0;
}

double TMR(double W, double P)
{
	double X   =  ::log10( W * P / (622.+ W) );
	double TMR = ::pow(10., ( gTMR_alfa * X + gTMR_beta )) - 7.07475 + gTMR_gamma2 * ( ::pow((::pow(10.,( gTMR_gamma * X )) - gTMR_gamma3 ), 2 ));
	return TMR - 273.16; // HUOM! lopussa muutetaan kuitenkin celsiuksiksi!!
}

double IterateMixMoistDiffWithNewtonMethod(double W, double Tpot, double P, double &diffOut)
{
	double P2 = P + 0.001;
	double tmr1 = TMR(W, P);
	double tmr2 = TMR(W, P2);
	double Tw1 = Tpot2t(Tpot, P);
	double Tw2 = Tpot2t(Tpot, P2);
	double tmrDeri = (tmr2 - tmr1)/(P2-P);
	double TwDeri = (Tw2 - Tw1)/(P2-P);
	double mixMoistDiff = tmr1 - Tw1;
	diffOut = mixMoistDiff;
	double mixMoistDiffDerivate = tmrDeri - TwDeri;
	return P - (mixMoistDiff / mixMoistDiffDerivate);
}

lcl_t CalcLCLPressureFast(double T, double Td, double P)
{
	double lastLCL = 900; // aloitetaan haku jostain korkeudesta

	T -= himan::constants::kKelvin;
	Td -= himan::constants::kKelvin;
	
	int iterationCount = 0; // Tämän voi poistaa profiloinnin jälkeen
	double lclPressure = kFloatMissing;
	// 2. Laske sekoitussuhde pinnalla
	double w = CalcMixingRatio(T, Td, P);
	double tpot = T2tpot(T, P); // pitää laskea mitä lämpötilaa vastaa pinnan 'potentiaalilämpötila'
	double currentP = lastLCL;
	double diff = 99999;
	int maxIterations = 20;
	// Etsi newtonin menetelmällä LCL pressure
	do
	{
		iterationCount++;
		currentP = IterateMixMoistDiffWithNewtonMethod(w, tpot, currentP, diff);
		if(::fabs(diff) < 0.01)
			break;
		if(currentP < 100) // most unstable tapauksissa etsintä piste saattaa pompata tosi ylös
		{ // tässä on paineen arvoksi tullut niin pieni että nostetaan sitä takaisin ylös ja jatketaan etsintöjä
			currentP = 100;
		}
	}while(iterationCount < maxIterations);

	// laske tarkempi paine jos viitsit lastP ja currentP;n avulla interpoloimalla
	if(iterationCount < maxIterations && currentP != kFloatMissing)
		lastLCL = currentP;
	else if(iterationCount >= maxIterations)
		currentP = kFloatMissing;
	lclPressure = currentP;
	lcl_t lcl;
	lcl.P = lclPressure;
	return lcl;
}
}
#endif
pair<vector<double>,vector<double>> si::GetLCL(shared_ptr<info> myTargetInfo, vector<double>& Tsurf, vector<double>& TDsurf)
{
	vector<double> T(Tsurf.size(), kFloatMissing);
	vector<double> P = T;
	
	// Need surface pressure

	const params PParams({param("PGR-PA"), param("P-PA")});

	auto Psurf = Fetch(myTargetInfo->Time(), level(kHeight, 0), PParams, myTargetInfo->ForecastType(), false);

	if (!Psurf)
	{
		throw runtime_error("Surface pressure not found");
	}
	
	Psurf->ResetLocation();

	for (size_t i = 0; i < T.size() && Psurf->NextLocation(); i++)
	{
		double _Psurf = Psurf->Value();
		auto lcl = metutil::LCLA_(_Psurf, Tsurf[i], TDsurf[i]);
		T[i] = lcl.T;
		P[i] = (lcl.P > _Psurf) ? 0.01 * _Psurf : 0.01 * lcl.P; // hPa

/*		
		auto lcl = smarttool::CalcLCLPressureFast(Tsurf[i], TDsurf[i], _Psurf);
		P[i] = (lcl.P > _Psurf) ? 0.01 * _Psurf : 0.01 * lcl.P; // hPa
		T[i] = himan::metutil::DryLift_(_Psurf, Tsurf[i], lcl.P);
*/		
	}

	return make_pair(T,P);
	
}

pair<vector<double>,vector<double>> si::GetSurfaceTAndTD(shared_ptr<info> myTargetInfo)
{
	auto TInfo = Fetch(myTargetInfo->Time(), level(himan::kHeight,2), param("T-K"), myTargetInfo->ForecastType(), false);
	auto TDInfo = Fetch(myTargetInfo->Time(), level(himan::kHeight,2), param("TD-C"), myTargetInfo->ForecastType(), false);
	
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
			
	vector<double> zero(myTargetInfo->Data().Size(), 0);
	vector<double> m500(zero.size(), 500.);

	tp.LowerHeight(zero);
	mr.LowerHeight(zero);

	tp.UpperHeight(m500);
	mr.UpperHeight(m500);

	level curLevel(kHybrid, 137);

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

	auto Tpot = tp.Result();
	auto MR = mr.Result();

	DumpVector(Tpot, "Tpot");
	DumpVector(MR, "MR");

	// Need surface pressure

	const params PParams({param("PGR-PA"), param("P-PA")});

	auto Psurf = Fetch(myTargetInfo->Time(), level(kHeight, 0), PParams, myTargetInfo->ForecastType(), false);
	auto P = Psurf->Data().Values();

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
	vector<double> T(myTargetInfo->Data().Size(), kFloatMissing);
	auto TD = T;
	
	level curLevel(kHybrid, 137);
	
	while (true)
	{
		auto TInfo = Fetch(myTargetInfo->Time(), curLevel, param("T-K"), myTargetInfo->ForecastType(), false);
		auto RHInfo = Fetch(myTargetInfo->Time(), curLevel, param("RH-PRCNT"), myTargetInfo->ForecastType(), false);
		auto PInfo = Fetch(myTargetInfo->Time(), curLevel, param("P-HPA"), myTargetInfo->ForecastType(), false);
		LOCKSTEP(TInfo, RHInfo, PInfo)
		{
			size_t i = TInfo->LocationIndex();
			
			if (found[i]) continue;
			
			double T_ = TInfo->Value();
			double RH = RHInfo->Value();
			double P = PInfo->Value();
			
			if (P < 500.)
			{
				found[i] = true;
				continue;
			}
			
			double ThetaE = metutil::ThetaE_(T_, P*100);
			
			if (ThetaE >= maxThetaE[i])
			{
				maxThetaE[i] = ThetaE;
				T[i] = T_;
				if (RH == 0.) RH=0.1;
				TD[i] = metutil::DewPointFromRH_(T_, RH);
				if (TD[i] < 100) std::cout << T_ << " " << RH << std::endl;
				assert(TD[i] > 100);
			}
		}
		
		if (static_cast<size_t> (count(found.begin(), found.end(), true)) == found.size())
		{
			break;
		}

		curLevel.Value(curLevel.Value()-1);
	}
	
	for (size_t i = 0; i < T.size(); i++)
	{
		if (T[i] == 0.) T[i] = kFloatMissing;
		if (TD[i] == 0.) TD[i] = kFloatMissing;
	}
	return make_pair(T,TD);

}