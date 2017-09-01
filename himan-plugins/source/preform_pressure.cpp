/**
 * @file preform_pressure.cpp
 */

#define AND &&
#define OR ||

#include "preform_pressure.h"
#include "forecast_time.h"
#include "level.h"
#include <boost/lexical_cast.hpp>
#include <limits>
#include "logger.h"

using namespace std;
using namespace himan::plugin;

// Olomuotopäättely pinta/peruspainepintadatalla tiivistettynä (tässä järjestyksessä):
//
// 0. Mallissa sadetta (RR>0; RR = rainfall + snowfall)
// 1. Jäätävää tihkua, jos -10<T2m<=0C, pakkasstratus (~-10...-0) jossa nousuliikettä, sade heikkoa, ei satavaa
// keskipilveä
// 2. Jäätävää vesisadetta, jos T2m<=0C ja pinnan yläpuolella on T>0C kerros, jossa kosteaa (pilveä)
// 3. Lunta, jos snowfall/RR>0.8, tai T<=0C
// 4. Räntää, jos 0.15<snowfall/RR<0.8
// 5. Vettä tai tihkua, jos snowfall/RR<0.15
// 5a. Tihkua, jos stratusta pienellä sadeintensiteetillä, eikä keskipilveä

// Mallin lumisateen osuuden raja-arvot (kokonaissateesta) lumi/räntä/vesiolomuodoille

const double waterLim = 0.15;
const double snowLim = 0.8;

// Vaadittu 2m lämpötilaväli (oltava näiden välissä) [C] jäätävässä tihkussa
const double sfcMax = 0.;
const double sfcMin = -10.;

// Kylmin sallittu stratuksen topin T ja kylmin sallittu st:n keskim. T [C] jäätävässä tihkussa
const double stTlimit = -12.;

// Raja-arvot tihkun ja jäätävän tihkun max intensiteetille [mm/h]
// (pienemmällä jäätävän tihkun raja-arvolla voi hieman rajoittaa sen esiintymistä)
const double dzLim = 0.3;
const double fzdzLim = 0.2;

// Max sallittu nousuliike stratuksessa [mm/s] jäätävässä tihkussa (vähentää fzdz esiintymistä)
const double wMax = 50.;

// 925 tai 850hPa:n stratuksen ~vähimmäispaksuus [hPa] jäätävässä tihkussa, ja
// sulamiskerroksen (tai sen alapuolisen pakkaskerroksen) vähimmäispaksuus [hPa] jäätävässä sateessa
// (olettaen, että stratuksen/sulamis-/pakkaskerroksen top on 925/850hPa:ssa)
const double stH = 15.;

// Suht. kosteuden raja-arvo alapilvelle (925/850/700hPa) [%]
const double rhLim = 90.;

// define missing int value
const int missingInt = numeric_limits<int>::min();

bool IsMissingInt(int val) { return val == missingInt; }
preform_pressure::preform_pressure()
{
	itsLogger = logger("preform_pressure");
}

void preform_pressure::Process(std::shared_ptr<const plugin_configuration> conf)
{
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
		itsLogger.Error(
		    "GRIB2 output requested, conversion between FMI precipitation form and GRIB2 precipitation type is not "
		    "lossless");
		return;
	}

	SetParams({param("PRECFORM-N", 57)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void preform_pressure::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	// Required source parameters

	const param TParam("T-K");
	const param RHParam("RH-PRCNT");
	const param SNRParam("SNR-KGM2");

	// Source precipitation parameter is "precipitation rate" which by my definition is always either
	// a) one hour precipitation, if forecast time step >= 1 hour
	// b) forecast time step, if step < 1 hour (for example 15 minutes with harmonie)
	//
	// For backwards compatibility also support one hour precipitation

	params RRParams({param("RRR-KGM2"), param("RR-1-MM")});

	const params PParams({param("PGR-PA"), param("P-PA")});
	const params WParams({param("VV-MMS"), param("VV-MS")});

	level groundLevel(kHeight, 2);

	level surface0mLevel(kHeight, 0);
	level surface2mLevel(kHeight, 2);
	level P700(kPressure, 700);
	level P850(kPressure, 850);
	level P925(kPressure, 925);
	level P1000(kPressure, 1000);

	auto myThreadedLogger =
	    logger("preformPressureThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
						  static_cast<string>(forecastLevel));

	// Source infos

	info_t TInfo = Fetch(forecastTime, groundLevel, TParam, forecastType, false);
	info_t T700Info = Fetch(forecastTime, P700, TParam, forecastType, false);
	info_t T850Info = Fetch(forecastTime, P850, TParam, forecastType, false);
	info_t T925Info = Fetch(forecastTime, P925, TParam, forecastType, false);

	info_t RHInfo = Fetch(forecastTime, surface2mLevel, RHParam, forecastType, false);
	info_t RH700Info = Fetch(forecastTime, P700, RHParam, forecastType, false);
	info_t RH850Info = Fetch(forecastTime, P850, RHParam, forecastType, false);
	info_t RH925Info = Fetch(forecastTime, P925, RHParam, forecastType, false);

	info_t W925Info = Fetch(forecastTime, P925, WParams, forecastType, false);
	info_t W850Info = Fetch(forecastTime, P850, WParams, forecastType, false);

	info_t RRInfo = Fetch(forecastTime, surface0mLevel, RRParams, forecastType, false);
	info_t PInfo = Fetch(forecastTime, surface0mLevel, PParams, forecastType, false);

	info_t SNRInfo = Fetch(forecastTime, surface0mLevel, SNRParam, forecastType, false);

	if (!TInfo || !T700Info || !T850Info || !T925Info || !RHInfo || !RH700Info || !RH850Info || !RH925Info ||
	    !W925Info || !W850Info || !RRInfo || !PInfo || !SNRInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	assert(TInfo->Param().Unit() == kK);
	assert(T700Info->Param().Unit() == kK);
	assert(T850Info->Param().Unit() == kK);
	assert(T925Info->Param().Unit() == kK);

	double WScale = 1;

	if (W850Info->Param().Name() == "VV-MS")
	{
		WScale = 1000;
	}

	assert(W850Info->Param().Name() == W925Info->Param().Name());

	// In Hirlam parameter name is RH-PRCNT but data is still 0 .. 1
	double RHScale = 100;

	if (RHInfo->Producer().Process() == 240 || RHInfo->Producer().Process() == 243)
	{
		// himan-calculated RH has values 0 .. 100
		RHScale = 1;
	}

	int DRIZZLE = 0;
	int RAIN = 1;
	int SLEET = 2;
	int SNOW = 3;
	int FREEZING_DRIZZLE = 4;
	int FREEZING_RAIN = 5;
	LOCKSTEP(myTargetInfo, TInfo, T700Info, T850Info, T925Info, RHInfo, RH700Info, RH850Info, RH925Info, W925Info,
	         W850Info, RRInfo, PInfo, SNRInfo)
	{
		double RR = RRInfo->Value();

		// No rain --> no rain type

		if (RR == 0 || IsMissing(RR))
		{
			continue;
		}

		double T = TInfo->Value();
		double T700 = T700Info->Value();
		double T850 = T850Info->Value();
		double T925 = T925Info->Value();

		double RH = RHInfo->Value();
		double RH700 = RH700Info->Value();
		double RH850 = RH850Info->Value();
		double RH925 = RH925Info->Value();

		double W850 = W850Info->Value();
		double W925 = W925Info->Value();

		double P = PInfo->Value();

		double SNR = SNRInfo->Value();

		if (IsMissingValue({T, T850, T925, RH, RH700, RH925, RH850, T700, P, W925, W850}))
		{
			continue;
		}

		int PreForm = missingInt;

		// Unit conversions

		//<! TODO: Kertoimet tietokannasta!

		T -= himan::constants::kKelvin;
		T700 -= himan::constants::kKelvin;
		T850 -= himan::constants::kKelvin;
		T925 -= himan::constants::kKelvin;

		RH *= RHScale;
		RH700 *= RHScale;
		RH850 *= RHScale;
		RH925 *= RHScale;

		P *= 0.01;  // ground pressure is always Pa in model

		W850 *= WScale;
		W925 *= WScale;
		/*
		        cout	<< "T\t\t" << T << endl
		                << "T700\t\t" << T700 << endl
		                << "T850\t\t" << T850 << endl
		                << "T925\t\t" << T925 << endl
		                << "RH\t\t" << RH << endl
		                << "RH700\t\t" << RH700 << endl
		                << "RH850\t\t" << RH850 << endl
		                << "RH925\t\t" << RH925 << endl
		                << "P\t\t" << P << endl
		                << "W850\t\t" << W850 << endl
		                << "W925\t\t" << W925 << endl
		                << "RR\t\t" << RR << endl
		                << "stH\t\t" << stH << endl
		                << "sfcMin\t\t" << sfcMin << endl
		                << "sfcMax\t\t" << sfcMax << endl
		                << "fzdzLim\t\t" << fzdzLim << endl
		                << "wMax\t\t" << wMax << endl
		                << "stTlimit\t" << stTlimit << endl
		                << "SNR\t\t" << SNR << endl;

		        abort();
		*/
		// (0=tihku, 1=vesi, 2=räntä, 3=lumi, 4=jäätävä tihku, 5=jäätävä sade)

		// jäätävää tihkua: "-10<T2m<=0, pakkasstratus (pinnassa/sen yläpuolella pakkasta & kosteaa), päällä ei
		// (satavaa) keskipilveä, sade heikkoa"

		if ((T <= sfcMax)AND(T > sfcMin) AND(RH700 < 80) AND(RH > 90) AND(RR <= fzdzLim))
		{
			// ollaanko korkeintaan ~750m merenpinnasta (pintapaine>925),
			// tai kun Psfc ei löydy (riittävän paksu/jäätävä) stratus 925hPa:ssa, jossa nousuliikettä?

			if (P > (925 + stH)AND RH925 > rhLim AND T925<0 AND T925> stTlimit AND W925 > 0 AND W925 < wMax)
			{
				PreForm = FREEZING_DRIZZLE;
			}

			// ollaanko ~750-1500m merenpinnasta (925<pintapaine<850)?
			// (riittävän paksu/jäätävä) stratus 850hPa:ssa, jossa nousuliikettä?

			if ((P <= 925 + stH)AND(P > 850 + stH) AND(RH850 > rhLim) AND(T850 < 0) AND(T850 > stTlimit) AND(W850 > 0)
			        AND(W850 < wMax))
			{
				PreForm = FREEZING_DRIZZLE;
			}
		}

		// jäätävää vesisadetta: "pinnassa pakkasta ja sulamiskerros pinnan lähellä"

		if (IsMissingInt(PreForm) AND(T <= 0) AND((T925 > 0)OR(T850 > 0) OR(T700 > 0)))
		{
			// ollaanko korkeintaan ~750m merenpinnasta (pintapaine>925), tai kun Psfc ei löydy?
			// (riittävän paksu) sulamiskerros ja pilveä 925/850hPa:ssa?

			if (P > (925 + stH)AND((T925 > 0 AND RH925 >= rhLim)OR(T850 > 0 AND RH850 >= rhLim)))
			{
				PreForm = FREEZING_RAIN;
			}

			// ollaanko ~750-1500m merenpinnasta (925<pintapaine<850)?
			// (riittävän paksu) sulamiskerros 850hPa:ssa (tai pakkaskerros sen alla)?

			if (P <= (925 + stH)AND P > (850 + stH)AND T850 > 0 AND RH850 >= rhLim)
			{
				PreForm = FREEZING_RAIN;
			}

			// ollaanko ~1500-3000m merenpinnasta (850<pintapaine<700)?
			// (riittävän paksu) sulamiskerros 700hPa:ssa ja pilveä 700hPa:ssa

			if (P <= 850 + stH AND P > 700 + stH AND T700 > 0 AND RH700 >= rhLim)
			{
				PreForm = FREEZING_RAIN;
			}
		}

		double SNR_RR = 0;  // oletuksena kaikki sade vetta

		if (!IsMissing(SNR))
		{
			// lasketaan oikea suhde vain jos lumidataa on (kesalla ei ole)
			SNR_RR = SNR / RR;
		}

		// lumisadetta: snowfall >=80% kokonaissateesta

		if (IsMissingInt(PreForm) AND(SNR_RR >= snowLim OR T <= 0))
		{
			PreForm = SNOW;
		}

		// räntää: snowfall 15...80% kokonaissateesta
		if (IsMissingInt(PreForm) AND(SNR_RR > waterLim) AND(SNR_RR < snowLim))
		{
			PreForm = SLEET;
		}

		// tihkua tai vesisadetta: Rain>=85% kokonaissateesta
		if (IsMissingInt(PreForm) AND(SNR_RR) <= waterLim)
		{
			// tihkua: "ei (satavaa) keskipilveä, pinnan lähellä kosteaa (stratus), sade heikkoa"
			if ((RH700 < 80)AND(RH > 90) AND(RR <= dzLim))
			{
				// ollaanko korkeintaan ~750m merenpinnasta (pintapaine>925),
				// tai kun Psfc ei (enää) löydy (eli ei mp-dataa, 6-10vrk)?
				// stratus 925hPa:ssa?

				if ((P > 925)AND(RH925 > rhLim))
				{
					PreForm = DRIZZLE;
				}

				// ollaanko ~750-1500m merenpinnasta (925<pintapaine<850)?
				// stratus 850hPa:ssa?

				if ((P <= 925)AND(P > 850) AND(RH850 > rhLim))
				{
					PreForm = DRIZZLE;
				}
			}

			// muuten vesisadetta:
			if (IsMissingInt(PreForm))
			{
				PreForm = RAIN;
			}
		}
		if (!IsMissingInt(PreForm))
		{
			myTargetInfo->Value(PreForm);
		}
	}

	myThreadedLogger.Info("[CPU] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) + "/" +
	                      to_string(myTargetInfo->Data().Size()));
}
