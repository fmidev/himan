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

#define HIMAN_AUXILIARY_INCLUDE

#include "hitool.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;
using namespace himan::plugin;

void Stratus(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, shared_ptr<info>& result);
void FreezingArea(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, shared_ptr<info>& result);

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

	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(itsConfiguration);

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

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
		
	boost::thread t(&FreezingArea, itsConfiguration, myTargetInfo->Time(), boost::ref(freezingArea));

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

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data()->MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data()->Size()));

}

void FreezingArea(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, shared_ptr<info>& result)
{
	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(conf);
	h->Time(ftime);

	try
	{
		result = h->FreezingArea();
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("FreezingArea() caught exception " + boost::lexical_cast<string> (e));
		}
	}

}

void Stratus(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, shared_ptr<info>& result)
{
	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));
	
	h->Configuration(conf);
	h->Time(ftime);

	try
	{
		result = h->Stratus();
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("FreezingArea() caught exception " + boost::lexical_cast<string> (e));
		}
	}

}