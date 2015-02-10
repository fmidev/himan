/**
 * @file preform_hybrid.cpp
 *
 * @date Sep 5, 2013
 * @author partio
 */

#define AND &&
#define OR ||
#define MISS kFloatMissing

#include "preform_hybrid.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "level.h"
#include "forecast_time.h"
#include <boost/foreach.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "hitool.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;
using namespace himan::plugin;

// Korkein sallittu pilven alarajan korkeus, jotta kysessä stratus [m]
const double baseLimit = 300.;

// Vaadittu min. stratuksen paksuus tihkussa ja jäätävässä tihkussa [m]
const double stLimit = 500.;
const double fzStLimit = 800.;

// Kylmin sallittu stratuksen topin T ja kylmin sallittu st:n keskim. T [C] jäätävässä tihkussa
const double stTlimit = -12.;

// Kynnysarvo "riittävän yhtenäisen/ei kerroksittaisen" stratuksen keskim. N-arvolle [%]
const double Nlimit = 70.;

// Vaadittu 2m lämpötilaväli (oltava näiden välissä) [C] jäätävässä tihkussa
const double sfcMax = 0.;
const double sfcMin = -10.;

// Max. sallittu keskim. RH-arvo (suht. kosteus) stratuksen yläpuoliselle kerrokselle [%] (jäätävässä) tihkussa
const double dryLimit = 70.;

// Raja-arvot tihkun ja jäätävän tihkun max intensiteetille [mm/h]
// (pienemmällä jäätävän tihkun raja-arvolla voi hieman rajoittaa sen esiintymistä)
const double dzLim = 0.3;
const double fzdzLim = 0.2;

// Raja-arvot pinnan pakkaskerroksen (MA) ja sen yläpuolisen sulamiskerroksen (PA) pinta-alalle jäätävässä sateessa [mC]
const double fzraMA = -100.;
const double fzraPA = 100.;

// Pinnan yläpuolisen plussakerroksen pinta-alan raja-arvot [mC, "metriastetta"]:
const double waterArea = 300;  // alkup. PK:n arvo oli 300
const double snowArea = 50;	// alkup. PK:n arvo oli 50

// Max sallittu nousuliike st:ssa [mm/s]
const double wMax = 50;

preform_hybrid::preform_hybrid()
{
	itsClearTextFormula = "<algorithm>";

	itsLogger = logger_factory::Instance()->GetLog("preform_hybrid");

}

void preform_hybrid::Process(std::shared_ptr<const plugin_configuration> conf)
{
	// Initialize plugin

	Init(conf);

	/*
	 * !!! HUOM !!!
	 *
	 * GRIB2 precipitation type <> FMI precipitation form
	 *
	 * FMI:
	 * 0 = tihku, 1 = vesi, 2 = räntä, 3 = lumi, 4 = jäätävä tihku, 5 = jäätävä sade
	 *
	 * GRIB2:
	 * 0 = Reserved, 1 = Rain, 2 = Thunderstorm, 3 = Freezing Rain, 4 = Mixed/Ice, 5 = Snow
	 *
	 */

	if (itsConfiguration->OutputFileType() == kGRIB2)
	{
		itsLogger->Error("GRIB2 output requested, conversion between FMI precipitation form and GRIB2 precipitation type is not lossless");
		return;
	}


	// Feikkiparametrinimi ja -numero koska alkuperainen on preform_pressurelle varattu!
	// Uusi neons-rakenne ehka sallii meidan tallentaa eri laskentatavoilla tuotetut
	// parametrit samalle numerolle

	SetParams({param("PRECFORM2-N", 10059)});

	Start();
	
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void preform_hybrid::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	assert(fzStLimit >= stLimit);

	// Required source parameters

	params RRParam({ param("RR-1-MM"), param("RRR-KGM2")}); // one hour prec OR precipitation rate (HHsade)
	const param TParam("T-K");
	const param RHParam("RH-PRCNT");

	level surface0mLevel(kHeight, 0);
	level surface2mLevel(kHeight, 2);

	auto myThreadedLogger = logger_factory::Instance()->GetLog("preformHybridThread #" + boost::lexical_cast<string> (threadIndex));

	const param stratusBaseParam("STRATUS-BASE-M");
	const param stratusTopParam("STRATUS-TOP-M");
	const param stratusTopTempParam("STRATUS-TOP-T-K");
	const param stratusMeanTempParam("STRATUS-MEAN-T-K");
	const param stratusMeanCloudinessParam("STRATUS-MEAN-N-PRCNT");
	const param stratusUpperLayerRHParam("STRATUS-UPPER-LAYER-RH-PRCNT");
	const param stratusVerticalVelocityParam("STRATUS-VERTICAL-VELOCITY-MMS");

	const param minusAreaParam("MINUS-AREA-MC");
	const param plusAreaParam("PLUS-AREA-MC");

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	h->Time(forecastTime);

	// Source infos

	info_t RRInfo = Fetch(forecastTime, surface0mLevel, RRParam, false);
	info_t TInfo = Fetch(forecastTime, surface0mLevel, TParam, false);
	info_t RHInfo = Fetch(forecastTime, surface2mLevel, RHParam, false);

	if (!RRInfo || !TInfo || !RHInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	info_t stratus;
	info_t freezingArea;

	/*
	 * Spinoff thread will calculate freezing area while main thread calculates
	 * stratus.
	 *
	 * The constructor of std::thread (and boost::thread) deduces argument types
	 * and stores them *by value*.
	 */
		
	boost::thread t(&preform_hybrid::FreezingArea, this, itsConfiguration, myTargetInfo->Time(), boost::ref(freezingArea));

	Stratus(itsConfiguration, myTargetInfo->Time(), stratus);

	t.join();

	if (!stratus || !freezingArea)
	{
		myThreadedLogger->Error("hitool calculation failed, unable to proceed");
		return;
	}

	freezingArea->First();
	stratus->First();

	myThreadedLogger->Info("Stratus and freezing area calculated");
		
	string deviceType = "CPU";

	assert(myTargetInfo->SizeLocations() == stratus->SizeLocations());
	assert(myTargetInfo->SizeLocations() == freezingArea->SizeLocations());
	assert(myTargetInfo->SizeLocations() == TInfo->SizeLocations());
	assert(myTargetInfo->SizeLocations() == RRInfo->SizeLocations());
	assert(myTargetInfo->SizeLocations() == RHInfo->SizeLocations());

	LOCKSTEP (myTargetInfo, stratus, freezingArea, TInfo, RRInfo, RHInfo)
	{

		stratus->Param(stratusBaseParam);
		double base = stratus->Value();

		stratus->Param(stratusTopParam);
		double top = stratus->Value();

		stratus->Param(stratusUpperLayerRHParam);
		double upperLayerRH = stratus->Value();

		stratus->Param(stratusVerticalVelocityParam);
		double wAvg = stratus->Value();

		stratus->Param(stratusMeanCloudinessParam);
		double Navg = stratus->Value();

		stratus->Param(stratusMeanTempParam);
		double stTavg = stratus->Value();

		stratus->Param(stratusTopTempParam);
		double Ttop = stratus->Value();

		freezingArea->Param(plusAreaParam);
		double plusArea = freezingArea->Value();

		freezingArea->Param(minusAreaParam);
		double minusArea = freezingArea->Value();

		double RR = RRInfo->Value();
		double T = TInfo->Value();
		double RH = RHInfo->Value();

		if (RR == kFloatMissing || RR == 0 || T == kFloatMissing || RH == kFloatMissing)
		{
			// No rain --> no rain type
			continue;
		}

		double PreForm = kFloatMissing;

		// Unit conversions

		T -= himan::constants::kKelvin; // K --> C
		RH *= 100; // 0..1 --> %

		if (Ttop != kFloatMissing)
		{
			Ttop -= himan::constants::kKelvin;
		}

		if (stTavg != kFloatMissing)
		{
			stTavg -= himan::constants::kKelvin;
		}

		if (Navg != kFloatMissing)
		{
			Navg *= 100; // --> %
		}
/*
		cout	<< "base\t\t" << base << endl
				<< "top\t\t" << top << endl
				<< "Navg\t\t" << Navg << endl
				<< "upperLayerRH\t" << upperLayerRH << endl
				<< "RR\t\t" << RR << endl
				<< "stTavg\t\t" << stTavg << endl
				<< "T\t\t" << T << endl
				<< "RH\t\t" << RH << endl
				<< "plusArea1\t" << plusArea1 << endl
				<< "minusArea\t" << minusArea << endl
				<< "wAvg\t\t" << wAvg << endl
				<< "baseLimit\t" << baseLimit << endl
				<< "topLimit\t" << stLimit << endl
				<< "Nlimit\t\t" << Nlimit << endl
				<< "dryLimit\t" << dryLimit << endl
				<< "waterArea\t" << waterArea << endl
				<< "snowArea\t" << snowArea << endl
				<< "wMax\t\t" << wMax << endl
				<< "sfcMin\t\t" << sfcMin << endl
				<< "sfcMax\t\t" << sfcMax << endl
				<< "fzdzLim\t" << fzdzLim << endl
				<< "fzStLimit\t" << fzStLimit << endl
				<< "fzraPA\t\t" << fzraPA << endl
				<< "fzraMA\t\t" << fzraMA << endl;
*/
		bool thickStratusWithLightPrecipitation = (	base			!= kFloatMissing &&
													top				!= kFloatMissing &&
													Navg			!= kFloatMissing &&
													upperLayerRH	!= kFloatMissing &&
													RR				<= dzLim &&
													base			< baseLimit &&
													(top - base)	> stLimit &&
													Navg			> Nlimit &&
													upperLayerRH	< dryLimit);

		// Start algorithm
		// Possible values for preform: 0 = tihku, 1 = vesi, 2 = räntä, 3 = lumi, 4 = jäätävä tihku, 5 = jäätävä sade

		// 1. jäätävää tihkua? (tai lumijyväsiä)

		if (	base			!= kFloatMissing &&
				top				!= kFloatMissing &&
				upperLayerRH	!= kFloatMissing &&
				wAvg			!= kFloatMissing &&
				Navg			!= kFloatMissing &&
				stTavg			!= kFloatMissing &&
				Ttop			!= kFloatMissing)
		{

			if ((RR <= fzdzLim) AND
				(base < baseLimit) AND
				((top - base) >= fzStLimit) AND
				(wAvg < wMax) AND
				(wAvg >= 0) AND
				(Navg > Nlimit) AND
				(Ttop > stTlimit) AND
				(stTavg > stTlimit) AND
				(T > sfcMin) AND
				(T <= sfcMax) AND
				(upperLayerRH < dryLimit))
			{
				PreForm = kFreezingDrizzle;
			}
		}

		// 2. jäätävää vesisadetta? (tai jääjyväsiä (ice pellets), jos pakkaskerros hyvin paksu, ja/tai sulamiskerros ohut)
		// Löytyykö riittävän paksut: pakkaskerros pinnasta ja sen yläpuolelta plussakerros?
		// (Heikoimmat intensiteetit pois, RR>0.1 tms?)

		if (PreForm == MISS AND
			plusArea != MISS AND
			minusArea != MISS AND
			RR > 0.1 AND
			plusArea > fzraPA AND
			minusArea < fzraMA AND
			T <= 0 AND
			((upperLayerRH > dryLimit) OR (upperLayerRH == MISS)))
		{
			PreForm = kFreezingRain;
		}

		// Tihkua tai vettä jos "riitävän paksu lämmin kerros pinnan yläpuolella"

		if (PreForm == MISS)
		{
			if (plusArea != MISS AND plusArea > waterArea)
			{
				// Tihkua jos riittävän paksu stratus heikolla sateen intensiteetillä ja yläpuolella kuiva kerros
				// AND (ConvPre=0) poistettu alla olevasta (ConvPre mm/h puuttuu EC:stä; Hirlam-versiossa pidetään mukana)
				if (thickStratusWithLightPrecipitation)
				{
					PreForm = kDrizzle;
				}
				else
				{
					PreForm = kRain;
				}
				
				// Lisäys, jolla korjataan vesi/tihku lumeksi, jos pintakerros pakkasella (mutta jäätävän sateen/tihkun kriteerit eivät toteudu, 
				// esim. paksu plussakerros pakkas-st/sc:n yllä)
				if (minusArea != MISS OR plusArea < snowArea)
				{
					PreForm = kSnow;
				}
			}

			// Räntää jos "ei liian paksu lämmin kerros pinnan yläpuolella"

			if (plusArea != MISS && plusArea >= snowArea AND plusArea <= waterArea)
			{
				PreForm = kSleet;
				
				// lisäys, jolla korjataan räntä lumeksi, kun pintakerros pakkasella tai vain ohuelti plussalla
				
				if (minusArea != MISS OR plusArea < snowArea)
				{
					PreForm = kSnow;
				}
			}

			// Muuten lunta (PlusArea<50: "korkeintaan ohut lämmin kerros pinnan yläpuolella")
			// 20.2.2014: Ehto "OR T<0" poistettu (muuten ajoittain lunta, jos hyvin ohut pakkaskerros pinnassa)

			if (plusArea == MISS OR plusArea < snowArea)
			{
				PreForm = kSnow;
			}
		}

		// FINISHED

		myTargetInfo->Value(PreForm);

	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}

void preform_hybrid::FreezingArea(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, shared_ptr<info>& result)
{
	auto h = GET_PLUGIN(hitool);
	
	h->Configuration(conf);
	h->Time(ftime);

	const param minusAreaParam("MINUS-AREA-MC"); // metriastetta, mC
	const param plusAreaParam("PLUS-AREA-MC"); // metriastetta, mC

	vector<param> params = { minusAreaParam, plusAreaParam };
	vector<forecast_time> times = { ftime };
	vector<level> levels = { level(kHeight, 0, "HEIGHT") };

	auto ret = make_shared<info> (*conf->Info());
	ret->Params(params);
	ret->Levels(levels);
	ret->Times(times);
	ret->Create();

	vector<double> constData1(ret->Data().Size(), 0);

	auto constData2 = constData1;
	fill(constData2.begin(), constData2.end(), 5000);

	auto constData3 = constData1;
	fill(constData3.begin(), constData3.end(), himan::constants::kKelvin); // 0C

	vector<double> numZeroLevels, zeroLevel1, zeroLevel2, zeroLevel3, zeroLevel4;
	vector<double> Tavg1, Tavg2_two, Tavg2_three, Tavg3, Tavg2_four;
	vector<double> plusArea, minusArea;

	auto logger = logger_factory::Instance()->GetLog("preform_hybrid-freezing_area");
	
	try
	{
		// 0-kohtien lkm pinnasta (yläraja 5km, jotta ylinkin nollakohta varmasti löytyy)

		param wantedParam ("T-K");

		logger->Info("Counting number of zero levels");

		numZeroLevels = h->VerticalCount(wantedParam, constData1, constData2, constData3);

#ifdef DEBUG
		for (size_t i = 0; i < numZeroLevels.size(); i++)
		{
			assert(numZeroLevels[i] != kFloatMissing);
		}
#endif

		/* Check which values we have. Will slow down processing a bit but
		 * will make subsequent code much easier to understand.
		 */

		bool haveOne = false;
		bool haveTwo = false;
		bool haveThree = false;
		bool haveFour = false;

		for (size_t i = 0; i < numZeroLevels.size(); i++)
		{
			size_t val = static_cast<size_t> (numZeroLevels[i]);

			if (val == 1)
			{
				haveOne = true;
			}
			else if (val == 2)
			{
				haveTwo = true;
			}
			else if (val == 3)
			{
				haveThree = true;
			}
			else if (val >= 4)
			{
				haveFour = true;
			}

			if (haveOne && haveTwo && haveThree && haveFour)
			{
				break;
			}
		}

		// Get necessary source data based on loop data above

		zeroLevel1.resize(numZeroLevels.size(), kFloatMissing);

		zeroLevel2 = zeroLevel1;
		zeroLevel3 = zeroLevel1;
		zeroLevel4 = zeroLevel1;
		Tavg1 = zeroLevel1;
		Tavg2_two = zeroLevel1;
		Tavg2_three = zeroLevel1;
		Tavg2_four = zeroLevel1;
		Tavg3 = zeroLevel1;
		plusArea = zeroLevel1;
		minusArea = zeroLevel1;

		if (haveOne)
		{
			logger->Info("Searching for first zero level height and value");

			zeroLevel1 = h->VerticalHeight(wantedParam, constData1, constData2, constData3, 1);

			logger->Info("Searching for average temperature between ground level and first zero level");

			Tavg1 = h->VerticalAverage(wantedParam, constData1, zeroLevel1);

		}

		if (haveTwo)
		{
			logger->Info("Searching for second zero level height and value");

			assert(haveOne);

			zeroLevel2 = h->VerticalHeight(wantedParam, constData1, constData2, constData3, 2);

			logger->Info("Searching for average temperature between first and second zero level");

			Tavg2_two = h->VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);
		}

		if (haveThree)
		{
			logger->Info("Searching for third zero level height and value");

			assert(haveOne && haveTwo);

			zeroLevel3 = h->VerticalHeight(wantedParam, constData1, constData2, constData3, 3);

			logger->Info("Searching for average temperature between second and third zero level");

			Tavg2_three = h->VerticalAverage(wantedParam, zeroLevel2, zeroLevel3);

			logger->Info("Searching for average temperature between first and third zero level");

			Tavg3 = h->VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);
		}

		if (haveFour)
		{
			logger->Info("Searching for fourth zero level height and value");

			zeroLevel4 = h->VerticalHeight(wantedParam, constData1, constData2, constData3, 4);

			logger->Info("Searching for average temperature between third and fourth zero level");

			Tavg2_four = h->VerticalAverage(wantedParam, zeroLevel3, zeroLevel4);

		}
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("FreezingArea() caught exception " + boost::lexical_cast<string> (e));
		}
		else
		{
			return;
		}
	}

	for (size_t i = 0; i < numZeroLevels.size(); i++)
	{
		short numZeroLevel = static_cast<short> (numZeroLevels[i]);

		// nollarajoja parillinen määrä (pintakerros pakkasella)
		// nollarajoja on siis vähintään kaksi

		double pa = kFloatMissing, ma = kFloatMissing;

		if (numZeroLevel%2 == 0)
		{
			double zl1 = zeroLevel1[i], zl2 = zeroLevel2[i];
			double ta1 = Tavg1[i], ta2 = Tavg2_two[i];
			
			double paloft = kFloatMissing;

			if (zl1 != kFloatMissing && zl2 != kFloatMissing 
					&& ta1 != kFloatMissing && ta2 != kFloatMissing)
			{
				ma = zl1 * (ta1 - constants::kKelvin);
				paloft = (zl2 - zl1) * (ta2 - constants::kKelvin);
			}

			// Jos ylempänä toinen T>0 kerros, lasketaan myös sen koko (vähintään 4 nollarajaa)
			// (mahdollisista vielä ylempänä olevista plussakerroksista ei välitetä)
			
			if (numZeroLevel >= 4)
			{
				double zl3 = zeroLevel3[i], zl4 = zeroLevel4[i];
				ta2 = Tavg2_four[i];

				if (zl3 != kFloatMissing && zl4 != kFloatMissing && ta2 != kFloatMissing)
				{
					paloft = paloft + (zl4 - zl3) * (ta2 - constants::kKelvin);
				}
			}

			pa = paloft;

		}
		
		// nollarajoja pariton määrä (pintakerros plussalla)

		else if (numZeroLevel%2 == 1)
		{
			double zl1 = zeroLevel1[i], ta1 = Tavg1[i];
			double pasfc = kFloatMissing, paloft = kFloatMissing;

			if (zl1 != kFloatMissing && ta1 != kFloatMissing)
			{
				pasfc = zl1 * (ta1 - constants::kKelvin);
				pa = pasfc;
			}

			// Jos ylempänä toinen T>0 kerros, lasketaan myös sen koko (vähintään 3 nollarajaa)
			// (mahdollisista vielä ylempänä olevista plussakerroksista ei välitetä)

			if (numZeroLevel >= 3)
			{
				double zl2 = zeroLevel2[i], zl3 = zeroLevel3[i];
				double ta2 = Tavg2_three[i], ta3 = Tavg3[i];

				if (zl2 != kFloatMissing && zl3 != kFloatMissing &&
						ta2 != kFloatMissing && ta3 != kFloatMissing)
				{
					paloft = (zl3 - zl2) * (ta2 - constants::kKelvin);
					pa = pasfc + paloft;
				}
			}
		}

		plusArea[i] = pa;
		minusArea[i] = ma;
	}

	ret->Param(minusAreaParam);
	ret->Data().Set(minusArea);

	ret->Param(plusAreaParam);
	ret->Data().Set(plusArea);

	result = ret;

}

void preform_hybrid::Stratus(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, shared_ptr<info>& result)
{
	auto h = GET_PLUGIN(hitool);

	h->Configuration(conf);
	h->Time(ftime);

	// Vaadittu min. stratuksen paksuus tihkussa [m]
	const double stLimit = 500.;

	// Kerroksen paksuus pinnasta [m], josta etsitään stratusta (min. BaseLimit+FZstLimit)
	const double layer = 2000.;

	// N-kynnysarvo vaaditulle min. stratuksen määrälle [%] (50=yli puoli taivasta):
	const double stCover = 0.5;

	// Kynnysarvo vaaditulle stratuksen yläpuolisen kuivan kerroksen paksuudelle [m] (jäätävässä) tihkussa:
	const double drydz = 1500.;

	param baseParam("STRATUS-BASE-M");
	baseParam.Unit(kM);

	param topParam("STRATUS-TOP-M");
	topParam.Unit(kM);

	param topTempParam("STRATUS-TOP-T-K");
	topTempParam.Unit(kK);

	param meanTempParam("STRATUS-MEAN-T-K");
	meanTempParam.Unit(kK);

	param meanCloudinessParam("STRATUS-MEAN-N-PRCNT");
	meanCloudinessParam.Unit(kPrcnt);

	param upperLayerRHParam("STRATUS-UPPER-LAYER-RH-PRCNT");
	upperLayerRHParam.Unit(kPrcnt);

	param verticalVelocityParam("STRATUS-VERTICAL-VELOCITY-MMS");
	verticalVelocityParam.Unit(kMs);

	vector<param> params = { baseParam, topParam, topTempParam, meanTempParam, meanCloudinessParam, upperLayerRHParam, verticalVelocityParam };
	vector<forecast_time> times = { ftime };
	vector<level> levels = { level(kHeight, 0, "HEIGHT") };

	auto ret = make_shared<info> (*conf->Info());
	ret->Params(params);
	ret->Levels(levels);
	ret->Times(times);
	ret->Create();

	vector<double> constData1(ret->Data().Size(), 0);

	auto constData2 = constData1;
	auto logger = logger_factory::Instance()->GetLog("preform_hybrid-stratus");

	try
	{
		// N-kynnysarvot stratuksen ala- ja ylärajalle [%] (tarkkaa stCover arvoa ei aina löydy)

		vector<param> wantedParamList({param("N-0TO1"), param("N-PRCNT")});

		/**
		 * Etsitään parametrin N minimiarvo korkeusvälillä 0 .. stLimit (=500)
		 */

		logger->Info("Searching for stratus lower limit");

		auto baseThreshold = h->VerticalMinimum(wantedParamList, 0, stLimit);

		for (size_t i = 0; i < baseThreshold.size(); i++)
		{
			assert(baseThreshold[i] != kFloatMissing);

			if (baseThreshold[i] < stCover)
			{
				baseThreshold[i] = stCover;
			}
		}

		ret->Param(baseParam);
		ret->Data().Set(baseThreshold);

		/**
		 * Etsitään parametrin N minimiarvo korkeusvälillä stLimit (=500) .. layer (=2000)
		 */

		logger->Info("Searching for stratus upper limit");

		auto topThreshold = h->VerticalMinimum(wantedParamList, stLimit, layer);

		for (size_t i = 0; i < topThreshold.size(); i++)
		{
			assert(topThreshold[i] != kFloatMissing);
			if (topThreshold[i] < stCover)
			{
				topThreshold[i] = stCover;
			}
		}

		// Stratus Base/top [m]
		// _findh: 0 = viimeinen löytyvä arvo pinnasta ylöspäin, 1 = ensimmäinen löytyvä arvo
		// (Huom. vertz-funktio hakee tarkkaa arvoa, jota ei aina löydy esim. heti pinnasta lähtevälle
		//  stratukselle; joskus siis tuloksena on virheellisesti Base=top)

		/**
		 * Etsitään parametrin N ensimmäisen baseThreshold-arvon korkeus väliltä 0 .. layer (=2000)
		 */

		logger->Info("Searching for stratus base accurate value");

		auto stratusBase = h->VerticalHeight(wantedParamList, 0, stLimit, baseThreshold);

		//VAR Base = VERTZ_FINDH(N_EC,0,Layer,BaseThreshold,1)

		if (!ret->Param(baseParam))
		{
			throw runtime_error("Impossible error");
		}

		ret->Data().Set(stratusBase);

		size_t missing = 0;

		for (size_t i = 0; i < stratusBase.size(); i++)
		{
			if (stratusBase[i] == kFloatMissing)
			{
				missing++;
			}
		}

		logger->Debug("Stratus base number of missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (stratusBase.size()));

		/**
		 * Etsitään parametrin N viimeisen topThreshold-arvon korkeus väliltä 0 .. layer (=2000)
		 */

		logger->Info("Searching for stratus top accurate value");
		auto stratusTop = h->VerticalHeight(wantedParamList, stLimit, layer, topThreshold, 0);

		ret->Param(topParam);
		ret->Data().Set(stratusTop);

#ifdef DEBUG

		missing = 0;

		for (size_t i = 0; i < stratusTop.size(); i++)
		{
			if (stratusTop[i] == kFloatMissing)
			{
				missing++;
			}
		}

		logger->Debug("Stratus top number of missing values: " + boost::lexical_cast<string> (missing)+ "/" + boost::lexical_cast<string> (stratusTop.size()));

#endif
		// Keskimääräinen RH stratuksen yläpuolisessa kerroksessa (jäätävä tihku)

		logger->Info("Searching for humidity in layers above stratus top");

		param wantedParam = param("RH-PRCNT");

		assert(constData1.size() == constData2.size() && constData1.size() == stratusTop.size());

		for (size_t i = 0; i < constData1.size(); i++)
		{
			if (stratusTop[i] == kFloatMissing)
			{
				constData1[i] = kFloatMissing;
				constData2[i] = kFloatMissing;
			}
			else
			{
				constData1[i] = stratusTop[i] + 100;
				constData2[i] = stratusTop[i] + drydz;
			}
		}

		//VERTZ_AVG(RH_EC,Top+100,Top+DRYdz)
		auto upperLayerRH = h->VerticalAverage(wantedParam, constData1, constData2);

		ret->Param(upperLayerRHParam);
		ret->Data().Set(upperLayerRH);

#ifdef DEBUG
		missing = 0;

		for (size_t i = 0; i < upperLayerRH.size(); i++)
		{
			if (upperLayerRH[i] == kFloatMissing)
			{
				missing++;
			}
		}

		logger->Debug("Upper layer RH number of missing values: " + boost::lexical_cast<string> (missing)+ "/" + boost::lexical_cast<string> (upperLayerRH.size()));
#endif
		//VERTZ_AVG(N_EC,Base,Top)

		logger->Info("Searching for stratus mean cloudiness");

		auto stratusMeanN = h->VerticalAverage(wantedParamList, stratusBase, stratusTop);

#ifdef DEBUG
		missing = 0;

		for (size_t i = 0; i < stratusMeanN.size(); i++)
		{
			if (stratusMeanN[i] == kFloatMissing)
			{
				missing++;
			}
		}

		logger->Debug("Stratus mean cloudiness number of missing values: " + boost::lexical_cast<string> (missing)+ "/" + boost::lexical_cast<string> (stratusMeanN.size()));
#endif

		ret->Param(meanCloudinessParam);
		ret->Data().Set(stratusMeanN);

		logger->Info("Searching for stratus top temperature");

		// Stratuksen Topin lämpötila (jäätävä tihku)
		//VAR TTop = VERTZ_GET(T_EC,Top)

		wantedParam = { param("T-K") };

		auto stratusTopTemp = h->VerticalValue(wantedParam, stratusTop);
#ifdef DEBUG
		missing = 0;

		for (size_t i = 0; i < stratusTopTemp.size(); i++)
		{
			if (stratusTopTemp[i] == kFloatMissing)
			{
				missing++;
			}
		}

		logger->Debug("Stratus top temperature number of missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (stratusTopTemp.size()));
#endif
		ret->Param(topTempParam);
		ret->Data().Set(stratusTopTemp);

		logger->Info("Searching for stratus mean temperature");

		// St:n keskimääräinen lämpötila (poissulkemaan kylmät <-10C stratukset, joiden toppi >-10C) (jäätävä tihku)
		//VAR stTavg = VERTZ_AVG(T_EC,Base+50,Top-50)

		for (size_t i = 0; i < constData1.size(); i++)
		{
			double lower = stratusBase[i];
			double upper = stratusTop[i];

			if (lower == kFloatMissing || upper == kFloatMissing)
			{
				constData1[i] = kFloatMissing;
				constData2[i] = kFloatMissing;
			}
			else if (fabs(lower-upper) < 100)
			{
				constData1[i] = lower;
				constData2[i] = upper;
			}
			else
			{
				constData1[i] = lower + 50;
				constData2[i] = upper - 50;
			}
		}

		auto stratusMeanTemp = h->VerticalAverage(wantedParam, constData1, constData2);
#ifdef DEBUG
		missing = 0;

		for (size_t i = 0; i < stratusMeanTemp.size(); i++)
		{
			if (stratusMeanTemp[i] == kFloatMissing)
			{
				missing++;
			}
		}

		logger->Debug("Stratus mean temperature number of missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (stratusMeanTemp.size()));
#endif
		ret->Param(meanTempParam);
		ret->Data().Set(stratusMeanTemp);

		// Keskimääräinen vertikaalinopeus st:ssa [mm/s]
		//VAR wAvg = VERTZ_AVG(W_EC,Base,Top)

		logger->Info("Searching for mean vertical velocity in stratus");

		wantedParam = param("VV-MMS");

		vector<double> stratusVerticalVelocity;

		try
		{
			stratusVerticalVelocity = h->VerticalAverage(wantedParam, stratusBase, stratusTop);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				logger->Debug("Trying for param VV-MS");
				wantedParam = param("VV-MS");

				stratusVerticalVelocity = h->VerticalAverage(wantedParam, stratusBase, stratusTop);

				BOOST_FOREACH(double& d, stratusVerticalVelocity)
				{
					d *= 1000;
				}
			}
		}

#ifdef DEBUG
		missing = 0;

		for (size_t i = 0; i < stratusVerticalVelocity.size(); i++)
		{
			if (stratusVerticalVelocity[i] == kFloatMissing)
			{
				missing++;
			}
		}

		logger->Debug("Stratus vertical velocity number of missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (stratusVerticalVelocity.size()));
#endif

		ret->Param(verticalVelocityParam);
		ret->Data().Set(stratusVerticalVelocity);
		
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Stratus() caught exception " + boost::lexical_cast<string> (e));
		}
		else
		{
			return;
		}
	}

	result = ret;

}
