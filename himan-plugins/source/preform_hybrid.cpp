/**
 * @file preform_hybrid.cpp
 *
 */

#define AND &&
#define OR ||

#include "preform_hybrid.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <future>
#include <iostream>
#include <thread>

#include "hitool.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

// 0. Mallissa sadetta (RR>0; RR = rainfall + snowfall, [RR]=mm/h)
//
// 1. Jäätävää tihkua, jos
// RR <= 0.2
// -10C < T2m <= 0C
// stratus (base<305m ja määrä >4/8)
// stratuksessa (heikkoa) nousuliikettä (0<wAvg<50mm/s)
// stratus riittävän paksu (dz>700m)
// stratus Ttop > -12C
// stratus avgT > -12C
// kuiva kerros (paksuus>1.5km, jossa N<30%) stratuksen yläpuolella
//
// 2. Jäätävää vesisadetta, jos
// T2m <= 0C
// riittävän paksu/lämmin (pinta-ala>100mC) sulamiskerros pinnan yläpuolella
// riittävän paksu/kylmä (pinta-ala<-100mC) pakkaskerros pinnassa sulamiskerroksen alapuolella
// jos on stratus, sulamiskerros sen yllä ei saa olla kuiva
//
// 3. Tihkua tai vettä, jos
// riittävän paksu ja lämmin plussakerros pinnan yläpuolella (pinta-ala>200mC)
//
// 3.1 Tihkua, jos
// RR <= 0.3
// stratus (base<305m ja määrä >4/8)
// stratus riittävän paksu (dz>400m)
// kuiva kerros (dz>1.5km, jossa N<30%) stratuksen yläpuolella
//
// 3.2 Muuten vettä
//
// 3.3 Jos pinnan plussakerroksessa on kuivaa (rhAvg<rhMelt), muutetaan olomuoto vedestä rännäksi
//
// 4. Räntää, jos
// ei liian paksu/lämmin plussakerros pinnan yläpuolella (50mC<pinta-ala<200mC)
//
// 4.1 Jos pinnan plussakerroksessa on kuivaa (rhAvg<rhMelt), muutetaan olomuoto rännästä lumeksi
//
// 5. Muuten lunta
// korkeintaan ohut plussakerros pinnan yläpuolella (area < 50mC)

// Korkein sallittu pilven alarajan korkeus, jotta kyseessä stratus [m] (305m=1000ft)
const double baseLimit = 305;

// Vaadittu min. stratuksen paksuus tihkussa ja jäätävässä tihkussa [m]
const double stLimit = 400.;
const double fzStLimit = 700.;

// Kylmin sallittu stratuksen topin T ja kylmin sallittu st:n keskim. T [C] jäätävässä tihkussa
const double stTlimit = -12.;

// Vaadittu 2m lämpötilaväli (oltava näiden välissä) [C] jäätävässä tihkussa
const double sfcMax = 0.;
const double sfcMin = -10.;

// Max. sallittu pilven (keskimääräinen) määrä (N) stratuksen yläpuolisessa kerroksessa [%] (jäätävässä) tihkussa
// (käytetään myös jäätävässä sateessa vaadittuna pilven minimimääränä)
const double dryNlim = 0.3;

// Kynnysarvo vaaditulle stratuksen yläpuolisen kuivan kerroksen paksuudelle [m] jää]
// Raja-arvot tihkun ja jäätävän tihkun max intensiteetille [mm/h]
// (pienemmällä jäätävän tihkun raja-arvolla voi hieman rajoittaa sen esiintymistä)
const double dzLim = 0.3;
const double fzdzLim = 0.2;

// Raja-arvot pinnan pakkaskerroksen (MA) ja sen yläpuolisen sulamiskerroksen (PA) pinta-alalle jäätävässä sateessa [mC]
const double fzraMA = -100.;
const double fzraPA = 100.;

// Pinnan yläpuolisen plussakerroksen pinta-alan raja-arvot [mC, "metriastetta"]:
const double waterArea = 200.;  // alkup. PK:n arvo oli 300
const double snowArea = 50.;    // alkup. PK:n arvo oli 50

// Max sallittu nousuliike st:ssa [mm/s]
const double wMax = 50.;

const param stratusBaseParam("STRATUS-BASE-M");
const param stratusTopParam("STRATUS-TOP-M");
const param stratusTopTempParam("STRATUS-TOP-T-K");
const param stratusMeanTempParam("STRATUS-MEAN-T-K");
const param stratusUpperLayerNParam("STRATUS-UPPER-LAYER-N-PRCNT");
const param stratusVerticalVelocityParam("STRATUS-VERTICAL-VELOCITY-MMS");

const param minusAreaParam("MINUS-AREA-MC");        // metriastetta, mC
const param plusAreaParam("PLUS-AREA-MC");          // metriastetta, mC
const param plusAreaSfcParam("PLUS-AREA-SFC-MC");   // metriastetta, mC
const param numZeroLevelsParam("NUMZEROLEVELS-N");  // nollakohtien lkm
const param rhAvgParam("RHAVG-PRCNT");
const param rhAvgUpperParam("RHAVG-UPPER-PRCNT");
const param rhMeltParam("RHMELT-PRCNT");
const param rhMeltUpperParam("RHMELT-UPPER-PRCNT");

const param TParam("T-K");
const param RHParam("RH-PRCNT");

preform_hybrid::preform_hybrid()
{
	itsLogger = logger("preform_hybrid");
}
void preform_hybrid::Process(std::shared_ptr<const plugin_configuration> conf)
{
	// Initialize plugin

	Init(conf);

	vector<param> params({param("PRECFORM2-N", 1206, 0, 1, 19)});

	if (itsConfiguration->Exists("potential_precipitation_form") &&
	    itsConfiguration->GetValue("potential_precipitation_form") == "true")
	{
		params.push_back(param("POTPRECF-N", 1226, 0, 1, 254));
	}

	SetParams(params);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void preform_hybrid::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	ASSERT(fzStLimit >= stLimit);

	// Required source parameters

	params RRParam({param("RRR-KGM2"), param("RR-1-MM")});  // one hour prec OR precipitation rate (HHsade)

	level surface0mLevel(kHeight, 0);
	level surface2mLevel(kHeight, 2);

	auto myThreadedLogger = logger("preformHybridThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	// Source infos

	shared_ptr<info<double>> RRInfo = Fetch(forecastTime, surface0mLevel, RRParam, forecastType, false);
	shared_ptr<info<double>> TInfo = Fetch(forecastTime, surface2mLevel, TParam, forecastType, false);

	if (!RRInfo || !TInfo)
	{
		myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	shared_ptr<info<double>> stratus;
	shared_ptr<info<double>> freezingArea;

	/*
	 * Spinoff thread will calculate freezing area while main thread calculates
	 * stratus.
	 *
	 * The constructor of std::thread (and boost::thread) deduces argument types
	 * and stores them *by value*.
	 */

	thread t(&preform_hybrid::FreezingArea, this, itsConfiguration, forecastTime, forecastType, ref(freezingArea),
	         myTargetInfo->Base());

	Stratus(itsConfiguration, forecastTime, forecastType, stratus, myTargetInfo->Base());

	t.join();

	if (!stratus)
	{
		myThreadedLogger.Error("stratus calculation failed, unable to proceed");
		return;
	}

	if (!freezingArea)
	{
		myThreadedLogger.Error("freezingArea calculation failed, unable to proceed");
		return;
	}

	freezingArea->First();
	stratus->First();

	myThreadedLogger.Info("Stratus and freezing area calculated");

	const string deviceType = "CPU";

	ASSERT(myTargetInfo->SizeLocations() == stratus->SizeLocations());
	ASSERT(myTargetInfo->SizeLocations() == freezingArea->SizeLocations());
	ASSERT(myTargetInfo->SizeLocations() == TInfo->SizeLocations());
	ASSERT(myTargetInfo->SizeLocations() == RRInfo->SizeLocations());

	myTargetInfo->First<param>();

	const bool noPotentialPrecipitationForm = (myTargetInfo->Size<param>() == 1);

	const int DRIZZLE = 0;
	const int RAIN = 1;
	const int SLEET = 2;
	const int SNOW = 3;
	const int FREEZING_DRIZZLE = 4;
	const int FREEZING_RAIN = 5;

	LOCKSTEP(myTargetInfo, stratus, freezingArea, TInfo, RRInfo)
	{
		stratus->Find<param>(stratusBaseParam);
		double base = stratus->Value();

		stratus->Find<param>(stratusTopParam);
		double top = stratus->Value();

		stratus->Find<param>(stratusUpperLayerNParam);
		double upperLayerN = stratus->Value();

		stratus->Find<param>(stratusVerticalVelocityParam);
		double wAvg = stratus->Value();

		stratus->Find<param>(stratusMeanTempParam);
		double stTavg = stratus->Value();

		stratus->Find<param>(stratusTopTempParam);
		double Ttop = stratus->Value();

		freezingArea->Find<param>(plusAreaParam);
		double plusArea = freezingArea->Value();

		freezingArea->Find<param>(plusAreaSfcParam);
		double plusAreaSfc = freezingArea->Value();

		freezingArea->Find<param>(minusAreaParam);
		double minusArea = freezingArea->Value();

		freezingArea->Find<param>(numZeroLevelsParam);
		double nZeroLevel = freezingArea->Value();

		freezingArea->Find<param>(rhAvgParam);
		double rhAvg = freezingArea->Value();

		freezingArea->Find<param>(rhAvgUpperParam);
		double rhAvgUpper = freezingArea->Value();

		freezingArea->Find<param>(rhMeltParam);
		double rhMelt = freezingArea->Value();

		freezingArea->Find<param>(rhMeltUpperParam);
		double rhMeltUpper = freezingArea->Value();

		double RR = RRInfo->Value();
		double T = TInfo->Value();

		if (IsMissing(RR) || IsMissing(T))
		{
			continue;
		}

		if (RR == 0 && noPotentialPrecipitationForm)
		{
			continue;
		}

		double PreForm = MissingDouble();

		// Unit conversions

		T -= himan::constants::kKelvin;  // K --> C
		Ttop -= himan::constants::kKelvin;
		stTavg -= himan::constants::kKelvin;

		ASSERT(T >= -80 && T < 80);
		ASSERT(!noPotentialPrecipitationForm || RR > 0);

		// Start algorithm
		// Possible values for preform: 0 = tihku, 1 = vesi, 2 = räntä, 3 = lumi, 4 = jäätävä tihku, 5 = jäätävä sade

		// 1. jäätävää tihkua? (tai lumijyväsiä)

		if (RR <=
		    fzdzLim AND
		        base<baseLimit AND(top - base) >= fzStLimit AND wAvg<wMax AND wAvg >= 0 AND Ttop> stTlimit AND stTavg>
		            stTlimit AND T > sfcMin AND T <= sfcMax AND upperLayerN < dryNlim)
		{
			PreForm = FREEZING_DRIZZLE;
		}

		// 2. jäätävää vesisadetta? (tai jääjyväsiä (ice pellets), jos pakkaskerros hyvin paksu, ja/tai sulamiskerros
		// ohut)
		// Löytyykö riittävän paksut: pakkaskerros pinnasta ja sen yläpuolelta plussakerros, jossa pilveä/ei liian
		// kuivaa?
		// (Huom. hyvin paksu pakkaskerros (tai ohut sulamiskerros) -> oikeasti jääjyväsiä/ice pellets fzra sijaan)

		if (IsMissing(PreForm) AND plusArea >
		    fzraPA AND minusArea<fzraMA AND T <= 0 AND(IsMissing(upperLayerN) OR upperLayerN > dryNlim) AND rhAvgUpper>
		        rhMeltUpper)
		{
			PreForm = FREEZING_RAIN;
		}

		// 3. Lunta, räntää, tihkua vai vettä? PK:n koodia mukaillen

		if (IsMissing(PreForm))
		{
			// Tihkua tai vettä jos "riitävän paksu lämmin kerros pinnan yläpuolella"

			if (plusArea > waterArea)
			{
				// Tihkua jos riittävän paksu stratus heikolla sateen intensiteetillä ja yläpuolella kuiva kerros
				// AND (ConvPre=0) poistettu alla olevasta (ConvPre mm/h puuttuu EC:stä; Hirlam-versiossa pidetään
				// mukana)
				if (RR <= dzLim && base < baseLimit && (top - base) > stLimit && upperLayerN < dryNlim)
				{
					PreForm = DRIZZLE;
				}
				else
				{
					PreForm = RAIN;
				}

				// Jos pinnan plussakerroksessa on kuivaa, korjataan olomuodoksi räntä veden sijaan

				if (nZeroLevel == 1 AND rhAvg < rhMelt AND plusArea < 2600)
				{
					PreForm = SLEET;
				}

				// Lisäys, jolla korjataan vesi/tihku lumeksi, jos pintakerros pakkasella (mutta jäätävän sateen/tihkun
				// kriteerit eivät toteudu,
				// esim. paksu plussakerros pakkas-st/sc:n yllä)
				if (!IsMissing(minusArea) OR plusAreaSfc < snowArea)
				{
					PreForm = SNOW;
				}
			}

			// Räntää jos "ei liian paksu lämmin kerros pinnan yläpuolella"

			if (plusArea >= snowArea AND plusArea <= waterArea)
			{
				PreForm = SLEET;

				// Jos pinnan plussakerroksessa on kuivaa, korjataan olomuodoksi lumi rännän sijaan

				if (nZeroLevel == 1 AND rhAvg < rhMelt)
				{
					PreForm = SNOW;
				}

				// lisäys, jolla korjataan räntä lumeksi, kun pintakerros pakkasella tai vain ohuelti plussalla

				else if (!IsMissing(minusArea) OR plusAreaSfc < snowArea)
				{
					PreForm = SNOW;
				}
			}

			// Muuten lunta (PlusArea<50: "korkeintaan ohut lämmin kerros pinnan yläpuolella")

			if (IsMissing(plusArea) OR plusArea < snowArea)
			{
				PreForm = SNOW;
			}
		}

		// FINISHED

		if (RR == 0)
		{
			// If RR is zero, we can only have potential prec form
			myTargetInfo->Index<param>(1);
			myTargetInfo->Value(PreForm);
		}
		else
		{
			// If there is precipitation, we have at least regular prec form
			myTargetInfo->Index<param>(0);
			myTargetInfo->Value(PreForm);

			if (!noPotentialPrecipitationForm)
			{
				// Also potential prec form
				myTargetInfo->Index<param>(1);
				myTargetInfo->Value(PreForm);
			}
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

void preform_hybrid::FreezingArea(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime,
                                  const forecast_type& ftype, shared_ptr<info<double>>& result,
                                  shared_ptr<base<double>> baseGrid)
{
	timer t(true);

	auto h = GET_PLUGIN(hitool);

	h->Configuration(conf);
	h->Time(ftime);
	h->ForecastType(ftype);

	vector<param> params = {minusAreaParam, plusAreaParam,   plusAreaSfcParam, numZeroLevelsParam,
	                        rhAvgParam,     rhAvgUpperParam, rhMeltParam,      rhMeltUpperParam};
	vector<forecast_time> times = {ftime};
	vector<level> levels = {level(kHeight, 0, "HEIGHT")};

	auto ret = make_shared<info<double>>();

	ret->Set<param>(params);
	ret->Set<level>(levels);
	ret->Set<forecast_time>(times);
	ret->Set<forecast_type>({forecast_type()});
	ret->Create(baseGrid);

	const size_t N = ret->Data().Size();
	vector<double> zerom(N, 0);
	vector<double> tenkm(N, 10000.);
	vector<double> zerodeg(N, himan::constants::kKelvin);  // 0C

	vector<double> numZeroLevels(N), zeroLevel1(N), zeroLevel2(N), zeroLevel3(N), zeroLevel4(N);
	vector<double> Tavg23, Tavg34;
	vector<double> plusArea, minusArea, plusAreaSfc;
	vector<double> rhAvgUpper23;

	auto log = logger("preform_hybrid-freezing_area");

	future<vector<double>> futTavg01, futTavg12, futrhAvg01, futrhAvgUpper12;

	try
	{
		// 0-kohtien lkm pinnasta (yläraja 10km, jotta ylinkin nollakohta varmasti löytyy)

		const auto zeroLevels = h->VerticalHeight<double>(TParam, zerom, tenkm, zerodeg, -1);
		const size_t maxNumZeroLevels = zeroLevels.size() / zerom.size();

		for (size_t i = 0; i < N; i++)
		{
			for (size_t j = 0; j < maxNumZeroLevels; j++)
			{
				const double val = zeroLevels[i + j * N];

				if (IsMissing(val))
				{
					continue;
				}

				numZeroLevels[i] += 1;

				switch (j)
				{
					case 0:
						zeroLevel1[i] = val;
						break;
					case 1:
						zeroLevel2[i] = val;
						break;
					case 2:
						zeroLevel3[i] = val;
						break;
					case 3:
						zeroLevel4[i] = val;
						break;
					default:
						break;
				}
			}
		}

		ret->Find<param>(numZeroLevelsParam);
		ret->Data().Set(numZeroLevels);

		// 1. nollarajan alapuolisen, 2/3. nollarajojen välisen, ja koko T>0 alueen koko [mC, "metriastetta"]
		plusArea = zeroLevel1;
		plusAreaSfc = zeroLevel1;

		// Mahdollisen pinta- tai 1/2. nollarajojen välisen pakkaskerroksen koko [mC, "metriastetta"]
		minusArea = zeroLevel1;

		futTavg01 = async(launch::async, [&]() { return h->VerticalAverage<double>(TParam, zerom, zeroLevel1); });

		// Keskimääräinen RH nollarajan alapuolisessa plussakerroksessa
		futrhAvg01 = async(launch::async, [&]() { return h->VerticalAverage<double>(RHParam, zerom, zeroLevel1); });

		// Only the first zero layer with at least one non-missing element is required
		// to be present. All other zero layers (2,3,4) are optional.

		try
		{
			// Values between zero levels 1 <--> 2
			futTavg12 =
			    async(launch::async, [&]() { return h->VerticalAverage<double>(TParam, zeroLevel1, zeroLevel2); });

			// Keskimääräinen RH pakkaskerroksen yläpuolisessa plussakerroksessa
			futrhAvgUpper12 =
			    async(launch::async, [&]() { return h->VerticalAverage<double>(RHParam, zeroLevel1, zeroLevel2); });

			// 2 <--> 3
			Tavg23 = h->VerticalAverage<double>(TParam, zeroLevel2, zeroLevel3);

			// Keskimääräinen RH ylemmässä plussakerroksessa
			rhAvgUpper23 = h->VerticalAverage<double>(RHParam, zeroLevel2, zeroLevel3);

			// 3 <--> 4
			Tavg34 = h->VerticalAverage<double>(TParam, zeroLevel3, zeroLevel4);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				log.Debug("Some zero level not found from entire data");
			}
		}
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("FreezingArea() caught exception " + to_string(e));
		}
		else
		{
			return;
		}
	}

	// Keskimääräinen rhMelt nollarajan alapuolisessa plussakerroksessa, ja pakkaskerroksen yläpuolisessa
	// plussakerroksessa
	vector<double> rhMeltUpper(N, MissingDouble());
	vector<double> rhMelt(N, MissingDouble());

	// Keskimääräinen RH nollarajan alapuolisessa plussakerroksessa, ja pakkaskerroksen yläpuolisessa plussakerroksessa
	vector<double> rhAvgUpper(N, MissingDouble());
	vector<double> rhAvg(N, MissingDouble());

	auto Tavg01 = futTavg01.get();
	auto Tavg12 = futTavg12.get();
	auto rhAvg01 = futrhAvg01.get();
	auto rhAvgUpper12 = futrhAvgUpper12.get();

	for (size_t i = 0; i < numZeroLevels.size(); i++)
	{
		const short numZeroLevel = static_cast<short>(numZeroLevels[i]);

		double pa = MissingDouble(), ma = MissingDouble(), pasfc = MissingDouble();

		// Kommentteja Simolta nollakohtien lukumäärään:
		// * Nollakohtien löytymättömyys ei ole ongelma, sillä tällöin olomuoto on aina lumi tai jäätävä tihku
		// * Parittomilla nollakohdilla ei voi tulla koskaan jäätävää sadetta koska pintakerros on plussalla
		//   (jolloin ainakin yleensä pätee oletus ettei jäätävää sadetta voi esiintyä)

		// nollarajoja parillinen määrä (pintakerros pakkasella)

		if (numZeroLevel % 2 == 0)
		{
			double zl1 = zeroLevel1[i], zl2 = zeroLevel2[i];
			double ta1 = Tavg01[i] - constants::kKelvin;
			double ta2 = Tavg12[i] - constants::kKelvin;

			ma = zl1 * ta1;
			double paloft = (zl2 - zl1) * ta2;

			// Keskimääräinen rhMelt pakkaskerroksen yläpuolisessa plussakerroksessa
			rhMeltUpper[i] = 9.5 * exp((-17.27 * ta2) / (ta2 + 238.3)) * (10.5 - ta2);

			rhAvgUpper[i] = rhAvgUpper12[i];

			// Jos ylempänä toinen T>0 kerros, lasketaan myös sen koko (vähintään 4 nollarajaa) ja lisätään se alemman
			// kerroksen kokoon
			// (mahdollisista vielä ylempänä olevista plussakerroksista ei välitetä)
			// (tässäkin pitäisi tarkkaan ottaen tutkia rhAvgUpper, eli sulaako kerroksen läpi putoava lumi)

			if (numZeroLevel >= 4)
			{
				double zl3 = zeroLevel3[i], zl4 = zeroLevel4[i];
				ta2 = Tavg34[i] - constants::kKelvin;

				paloft += (zl4 - zl3) * ta2;
			}

			pa = paloft;
		}

		// nollarajoja pariton määrä (pintakerros plussalla)

		else if (numZeroLevel % 2 == 1)
		{
			double zl1 = zeroLevel1[i], ta1 = Tavg01[i] - constants::kKelvin;

			pasfc = zl1 * ta1;
			pa = pasfc;

			// Lisäys 3.2.2015: Suhteellisen kosteuden (raja-) arvot pinnan plussakerroksessa:
			// Jos nollarajan alapuolisessa plussakerroksessa on kuivaa, asetetaan olomuodoksi lumi
			// (Shaviv, 2006: http://www.sciencebits.com/SnowAboveFreezing)

			// rhMelt = suht. kosteuden raja-arvo, jota pienemmillä kosteuksilla lumihiutaleet eivät sula

			// Keskimääräinen rhMelt nollarajan alapuolisessa plussakerroksessa

			rhMelt[i] = 9.5 * exp((-17.27 * ta1) / (ta1 + 238.3)) * (10.5 - ta1);

			// Keskimääräinen RH nollarajan alapuolisessa plussakerroksessa
			rhAvg[i] = rhAvg01[i];

			// Jos ylempänä toinen T>0 kerros, lasketaan myös sen koko (vähintään 3 nollarajaa)
			// (mahdollisista vielä ylempänä olevista plussakerroksista ei välitetä)

			if (numZeroLevel >= 3)
			{
				double zl2 = zeroLevel2[i], zl3 = zeroLevel3[i];
				double ta2 = Tavg23[i] - constants::kKelvin;

				double paloft = (zl3 - zl2) * ta2;

				// Keskimääräinen rhMelt ylemmässä plussakerroksessa
				rhMeltUpper[i] = 9.5 * exp((-17.27 * ta2) / (ta2 + 238.3)) * (10.5 - ta2);

				// Keskimääräinen RH ylemmässä plussakerroksessa
				rhAvgUpper[i] = rhAvgUpper23[i];

				if (rhAvgUpper[i] > rhMeltUpper[i])
				{
					pa = pasfc + paloft;
				}
			}
		}

		plusArea[i] = pa;
		plusAreaSfc[i] = pasfc;
		minusArea[i] = ma;
	}

#ifdef DEBUG
	util::DumpVector(minusArea);
	util::DumpVector(plusArea);
	util::DumpVector(plusAreaSfc);
	util::DumpVector(rhAvg);
	util::DumpVector(rhAvgUpper);
	util::DumpVector(rhMelt);
	util::DumpVector(rhMeltUpper);
#endif

	ret->Find<param>(minusAreaParam);
	ret->Data().Set(minusArea);

	ret->Find<param>(plusAreaParam);
	ret->Data().Set(plusArea);

	ret->Find<param>(plusAreaSfcParam);
	ret->Data().Set(plusAreaSfc);

	ret->Find<param>(rhAvgParam);
	ret->Data().Set(rhAvg);

	ret->Find<param>(rhAvgUpperParam);
	ret->Data().Set(rhAvgUpper);

	ret->Find<param>(rhMeltParam);
	ret->Data().Set(rhMelt);

	ret->Find<param>(rhMeltUpperParam);
	ret->Data().Set(rhMeltUpper);

	t.Stop();
	log.Debug("Freezing area processed in " + to_string(t.GetTime()) + " ms");

	result = ret;
}

vector<double> Add(vector<double> vec, double a)
{
	for_each(vec.begin(), vec.end(), [a](double& d) { d += a; });
	return vec;
}

void preform_hybrid::Stratus(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime,
                             const forecast_type& ftype, shared_ptr<info<double>>& result,
                             shared_ptr<base<double>> baseGrid)
{
	timer t(true);
	auto h = GET_PLUGIN(hitool);

	h->Configuration(conf);
	h->Time(ftime);
	h->ForecastType(ftype);

	// Kerroksen paksuus pinnasta [m], josta etsitään stratusta (min. BaseLimit+FZstLimit)
	const double layer = 2500.;

	// N-kynnysarvo vaaditulle min. stratuksen määrälle [%] (50=yli puoli taivasta):
	const double stCover = 0.5;

	// Kynnysarvo vaaditulle stratuksen yläpuolisen kuivan kerroksen paksuudelle [m] (jäätävässä) tihkussa:
	const double drydz = 1500.;

	vector<param> params = {stratusBaseParam,     stratusTopParam,         stratusTopTempParam,
	                        stratusMeanTempParam, stratusUpperLayerNParam, stratusVerticalVelocityParam};
	vector<forecast_time> times = {ftime};
	vector<level> levels = {level(kHeight, 0, "HEIGHT")};

	auto ret = make_shared<info<double>>();

	ret->Set<param>(params);
	ret->Set<level>(levels);
	ret->Set<forecast_time>(times);
	ret->Set<forecast_type>({forecast_type()});  // doesn't matter what we put here
	ret->Create(baseGrid);

	const vector<double> layerVec(ret->Data().Size(), layer);
	const vector<double> stCoverVec(layerVec.size(), stCover);

	logger log("preform_hybrid-stratus");

	try
	{
		// Base = ensimmäinen korkeus, missä N>stCover
		vector<param> wantedParamList({param("N-0TO1"), param("N-PRCNT")});

		const auto base = h->VerticalHeightGreaterThan<double>(wantedParamList, 0, baseLimit, stCover);

		ret->Find<param>(stratusBaseParam);
		ret->Data().Set(base);

		const auto basePlus10 = Add(base, 10.);

		auto top = h->VerticalHeightLessThan<double>(wantedParamList, basePlus10, layerVec, stCoverVec);

		// Mahdollinen toinen matala pilvikerros stratuksen yläpuolella
		const auto topPlus10 = Add(top, 10.);
		const auto base2 = h->VerticalHeightGreaterThan<double>(wantedParamList, topPlus10, layerVec, stCoverVec);

		// Top2 = seuraava (pilvi)korkeus, missä N<stCover
		const auto top2 = h->VerticalHeightLessThan<double>(wantedParamList, base2, layerVec, stCoverVec);

		// Jos toinen pilvikerros alle 45m (150ft) korkeammalla, katsotaan sen olevan samaa stratusta
		for (size_t i = 0; i < base.size(); i++)
		{
			if ((base2[i] - top[i]) < 45.)
			{
				top[i] = top2[i];
			}
		}

		// Stratuksella oltava sekä Base että Top (jos Top:ia ei löydy kerroksesta Layer)
		for (size_t i = 0; i < base.size(); i++)
		{
			if (IsMissing(base[i]) == false && IsMissing(top[i]) == true)
			{
				top[i] = layer;
			}
		}

		ret->Find<param>(stratusTopParam);
		ret->Data().Set(top);

		// Stratuksen Topin lämpötila (jäätävä tihku)
		auto futTtop = async(launch::async, [&]() { return h->VerticalValue<double>(TParam, top); });

		// Stratuksen keskimääräinen lämpötila (poissulkemaan
		// kylmät <-10C stratukset, joiden toppi >-10C) (jäätävä tihku)
		auto topMinus10 = Add(top, -10);

		auto futstTavg =
		    async(launch::async, [&]() { return h->VerticalAverage<double>(TParam, basePlus10, topMinus10); });

		// Keskimääräinen pilven määrä [%] stratuksen yläpuolisessa kerroksessa
		auto topPlus30 = Add(top, 30);
		auto topPlusDrydz = Add(top, drydz);

		auto futupperLayerN = async(
		    launch::async, [&]() { return h->VerticalAverage<double>(wantedParamList, topPlus30, topPlusDrydz); });

		// Keskimääräinen vertikaalinopeus st:ssa [mm/s]
		vector<double> wAvg;

		try
		{
			wAvg = h->VerticalAverage<double>(param("VV-MMS"), base, top);
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				wAvg = h->VerticalAverage<double>(param("VV-MS"), base, top);

				for_each(wAvg.begin(), wAvg.end(), [](double& d) { d *= 1000.; });
			}
		}

		ret->Find<param>(stratusVerticalVelocityParam);
		ret->Data().Set(wAvg);

		auto Ttop = futTtop.get();
		ret->Find<param>(stratusTopTempParam);
		ret->Data().Set(Ttop);

		auto stTavg = futstTavg.get();
		ret->Find<param>(stratusMeanTempParam);
		ret->Data().Set(stTavg);

		auto upperLayerN = futupperLayerN.get();
		ret->Find<param>(stratusUpperLayerNParam);
		ret->Data().Set(upperLayerN);
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Stratus() caught exception " + to_string(e));
		}
	}

	t.Stop();
	log.Debug("Stratus processed in " + to_string(t.GetTime()) + " ms");
	result = ret;
}
