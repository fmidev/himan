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
#include <boost/thread.hpp>
#include <iostream>

#include "hitool.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

// 0. Mallissa sadetta (RR>0; RR = rainfall + snowfall, [RR]=mm/h)
//
// 1. Jäätävää tihkua, jos
//     RR <= 0.2
//     -10 < T2m < 0
//     sade ei konvektiivista (ConvPre mm/h = 0)
//     stratus (base<300m ja määrä vähintään 5/8)
//     stratuksessa (heikkoa) nousuliikettä (0<wAvg<50mm/s)
//     stratus riittävän paksu (dz>800m)
//     stratus Ttop > -12C
//     stratus avgT > -12C
//     -10C < T2m < 0C
//     kuiva kerros (paksuus>1.5km, jossa N<30%) stratuksen yläpuolella
//
// 2. Jäätävää vesisadetta, jos
//     T2m <= 0
//     riittävän paksu/lämmin (pinta-ala>100mC) sulamiskerros pinnan yläpuolella
//     riittävän paksu/kylmä (pinta-ala<-100mC) pakkaskerros pinnassa sulamiskerroksen alapuolella
//     jos on stratus, sulamiskerros sen yllä ei saa olla kuiva
//
// 3. Tihkua tai vettä, jos
//     riittävän paksu ja lämmin plussakerros pinnan yläpuolella
//
//   3.1 Tihkua, jos
//        RR <= 0.3
//        stratus (base<300m ja määrä vähintään 5/8)
//        stratus riittävän paksu (dz>500m)
//        kuiva kerros (dz>1.5km, jossa N<30%) stratuksen yläpuolella
//
//   3.2 Muuten vettä
//
//   3.3 Jos pinnan plussakerroksessa on kuivaa (rhAvg<rhMelt), muutetaan olomuoto vedestä rännäksi
//
// 4. Räntää, jos
//     ei liian paksu/lämmin plussakerros pinnan yläpuolella
//
//   4.1 Jos pinnan plussakerroksessa on kuivaa (rhAvg<rhMelt), muutetaan olomuoto rännästä lumeksi
//
// 5. Muuten lunta
//     korkeintaan ohut plussakerros pinnan yläpuolella

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
const param stratusMeanCloudinessParam("STRATUS-MEAN-N-PRCNT");
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

void preform_hybrid::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	assert(fzStLimit >= stLimit);

	// Required source parameters

	params RRParam({param("RR-1-MM"), param("RRR-KGM2")});  // one hour prec OR precipitation rate (HHsade)
	const param TParam("T-K");
	const param RHParam("RH-PRCNT");

	level surface0mLevel(kHeight, 0);
	level surface2mLevel(kHeight, 2);

	auto myThreadedLogger = logger("preformHybridThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	// Source infos

	info_t RRInfo = Fetch(forecastTime, surface0mLevel, RRParam, forecastType, false);
	info_t TInfo = Fetch(forecastTime, surface2mLevel, TParam, forecastType, false);

	if (!RRInfo || !TInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
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

	boost::thread t(&preform_hybrid::FreezingArea, this, itsConfiguration, forecastTime, forecastType,
	                boost::ref(freezingArea), myTargetInfo->Grid());

	Stratus(itsConfiguration, forecastTime, forecastType, stratus, myTargetInfo->Grid());

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

	string deviceType = "CPU";

	assert(myTargetInfo->SizeLocations() == stratus->SizeLocations());
	assert(myTargetInfo->SizeLocations() == freezingArea->SizeLocations());
	assert(myTargetInfo->SizeLocations() == TInfo->SizeLocations());
	assert(myTargetInfo->SizeLocations() == RRInfo->SizeLocations());

	myTargetInfo->FirstParam();

	bool noPotentialPrecipitationForm = (myTargetInfo->SizeParams() == 1);

	int DRIZZLE = 0;
	int RAIN = 1;
	int SLEET = 2;
	int SNOW = 3;
	int FREEZING_DRIZZLE = 4;
	int FREEZING_RAIN = 5;

	LOCKSTEP(myTargetInfo, stratus, freezingArea, TInfo, RRInfo)
	{
		stratus->Param(stratusBaseParam);
		double base = stratus->Value();

		stratus->Param(stratusTopParam);
		double top = stratus->Value();

		stratus->Param(stratusUpperLayerNParam);
		double upperLayerN = stratus->Value();

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

		freezingArea->Param(plusAreaSfcParam);
		double plusAreaSfc = freezingArea->Value();

		freezingArea->Param(minusAreaParam);
		double minusArea = freezingArea->Value();

		freezingArea->Param(numZeroLevelsParam);
		double nZeroLevel = freezingArea->Value();

		freezingArea->Param(rhAvgParam);
		double rhAvg = freezingArea->Value();

		freezingArea->Param(rhAvgUpperParam);
		double rhAvgUpper = freezingArea->Value();

		freezingArea->Param(rhMeltParam);
		double rhMelt = freezingArea->Value();

		freezingArea->Param(rhMeltUpperParam);
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

		if (!IsMissing(Ttop))
		{
			Ttop -= himan::constants::kKelvin;
		}

		if (!IsMissing(stTavg))
		{
			stTavg -= himan::constants::kKelvin;
		}

		if (!IsMissing(Navg))
		{
			Navg *= 100;  // --> %
		}

		assert(T >= -80 && T < 80);
		assert(!noPotentialPrecipitationForm || RR > 0);
		assert(IsMissing(Navg) || (Navg >= 0 && Navg <= 100));

		// Start algorithm
		// Possible values for preform: 0 = tihku, 1 = vesi, 2 = räntä, 3 = lumi, 4 = jäätävä tihku, 5 = jäätävä sade

		// 1. jäätävää tihkua? (tai lumijyväsiä)

		if (!IsMissing(base) AND !IsMissing(top) AND !IsMissing(upperLayerN) AND !IsMissing(wAvg) AND !IsMissing(Navg)
		        AND !IsMissing(stTavg) AND !IsMissing(Ttop) AND RR <=
		    fzdzLim AND
		        base<baseLimit AND(top - base) >= fzStLimit AND wAvg<wMax AND wAvg >= 0 AND Navg> Nlimit AND Ttop>
		            stTlimit AND stTavg > stTlimit AND T > sfcMin AND T <= sfcMax AND upperLayerN < dryNlim)
		{
			PreForm = FREEZING_DRIZZLE;
		}

		// 2. jäätävää vesisadetta? (tai jääjyväsiä (ice pellets), jos pakkaskerros hyvin paksu, ja/tai sulamiskerros
		// ohut)
		// Löytyykö riittävän paksut: pakkaskerros pinnasta ja sen yläpuolelta plussakerros, jossa pilveä/ei liian
		// kuivaa?
		// (Huom. hyvin paksu pakkaskerros (tai ohut sulamiskerros) -> oikeasti jääjyväsiä/ice pellets fzra sijaan)

		if (IsMissing(PreForm) AND !IsMissing(plusArea) AND !IsMissing(minusArea) AND !IsMissing(rhAvgUpper)
		        AND !IsMissing(rhMeltUpper) AND plusArea >
		    fzraPA AND minusArea<fzraMA AND T <= 0 AND(IsMissing(upperLayerN) OR upperLayerN > dryNlim) AND rhAvgUpper>
		        rhMeltUpper)
		{
			PreForm = FREEZING_RAIN;
		}

		// 3. Lunta, räntää, tihkua vai vettä? PK:n koodia mukaillen

		if (IsMissing(PreForm))
		{
			// Tihkua tai vettä jos "riitävän paksu lämmin kerros pinnan yläpuolella"

			if (!IsMissing(plusArea) AND plusArea > waterArea)
			{
				// Tihkua jos riittävän paksu stratus heikolla sateen intensiteetillä ja yläpuolella kuiva kerros
				// AND (ConvPre=0) poistettu alla olevasta (ConvPre mm/h puuttuu EC:stä; Hirlam-versiossa pidetään
				// mukana)
				if (!IsMissing(base) && !IsMissing(top) && !IsMissing(Navg) && !IsMissing(upperLayerN) && RR <= dzLim &&
				    base < baseLimit && (top - base) > stLimit && Navg > Nlimit && upperLayerN < dryNlim)
				{
					PreForm = DRIZZLE;
				}
				else
				{
					PreForm = RAIN;
				}

				// Jos pinnan plussakerroksessa on kuivaa, korjataan olomuodoksi räntä veden sijaan

				if (!IsMissing(nZeroLevel) AND !IsMissing(rhAvg) AND !IsMissing(rhMelt)
				        AND nZeroLevel == 1 AND rhAvg < rhMelt AND plusArea < 4000)
				{
					PreForm = SLEET;
				}

				// Lisäys, jolla korjataan vesi/tihku lumeksi, jos pintakerros pakkasella (mutta jäätävän sateen/tihkun
				// kriteerit eivät toteudu,
				// esim. paksu plussakerros pakkas-st/sc:n yllä)
				if (!IsMissing(minusArea) OR(!IsMissing(plusAreaSfc) AND plusAreaSfc < snowArea))
				{
					PreForm = SNOW;
				}
			}

			// Räntää jos "ei liian paksu lämmin kerros pinnan yläpuolella"

			if (!IsMissing(plusArea) AND plusArea >= snowArea AND plusArea <= waterArea)
			{
				PreForm = SLEET;

				// Jos pinnan plussakerroksessa on kuivaa, korjataan olomuodoksi lumi rännän sijaan

				if (!IsMissing(nZeroLevel) AND !IsMissing(rhAvg) AND !IsMissing(rhMelt)
				        AND nZeroLevel == 1 AND rhAvg < rhMelt)
				{
					PreForm = SNOW;
				}

				// lisäys, jolla korjataan räntä lumeksi, kun pintakerros pakkasella tai vain ohuelti plussalla

				if (!IsMissing(minusArea) OR(!IsMissing(plusAreaSfc) AND plusAreaSfc < snowArea))
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
			myTargetInfo->ParamIndex(1);
			myTargetInfo->Value(PreForm);
		}
		else
		{
			// If there is precipitation, we have at least regular prec form
			myTargetInfo->ParamIndex(0);
			myTargetInfo->Value(PreForm);

			if (!noPotentialPrecipitationForm)
			{
				// Also potential prec form
				myTargetInfo->ParamIndex(1);
				myTargetInfo->Value(PreForm);
			}
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

void preform_hybrid::FreezingArea(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime,
                                  const forecast_type& ftype, shared_ptr<info>& result, const grid* baseGrid)
{
	auto h = GET_PLUGIN(hitool);

	h->Configuration(conf);
	h->Time(ftime);
	h->ForecastType(ftype);

	vector<param> params = {minusAreaParam, plusAreaParam,   plusAreaSfcParam, numZeroLevelsParam,
	                        rhAvgParam,     rhAvgUpperParam, rhMeltParam,      rhMeltUpperParam};
	vector<forecast_time> times = {ftime};
	vector<level> levels = {level(kHeight, 0, "HEIGHT")};

	auto ret = make_shared<info>(*conf->Info());
	ret->Params(params);
	ret->Levels(levels);
	ret->Times(times);
	ret->Create(baseGrid);

	vector<double> constData1(ret->Data().Size(), 0);

	auto constData2 = constData1;
	fill(constData2.begin(), constData2.end(), 5000);

	auto constData3 = constData1;
	fill(constData3.begin(), constData3.end(), himan::constants::kKelvin);  // 0C

	vector<double> numZeroLevels, zeroLevel1, zeroLevel2, zeroLevel3, zeroLevel4;
	vector<double> Tavg01, Tavg12, Tavg23, Tavg34;
	vector<double> plusArea, minusArea, plusAreaSfc;
	vector<double> rhAvg01, rhAvgUpper12, rhAvgUpper23;

	auto log = logger("preform_hybrid-freezing_area");

	try
	{
		// 0-kohtien lkm pinnasta (yläraja 5km, jotta ylinkin nollakohta varmasti löytyy)

		param wantedParam("T-K");

		log.Trace("Counting number of zero levels");

		numZeroLevels = h->VerticalCount(wantedParam, constData1, constData2, constData3);

		ret->Param(numZeroLevelsParam);
		ret->Data().Set(numZeroLevels);

#ifdef DEBUG
		for (size_t i = 0; i < numZeroLevels.size(); i++)
		{
			assert(!IsMissing(numZeroLevels[i]));
		}

		util::DumpVector(numZeroLevels, "num zero levels");
#endif

		zeroLevel1.resize(numZeroLevels.size(), MissingDouble());
		zeroLevel2.resize(numZeroLevels.size(), MissingDouble());
		zeroLevel3.resize(numZeroLevels.size(), MissingDouble());
		zeroLevel4.resize(numZeroLevels.size(), MissingDouble());

		rhAvgUpper12.resize(numZeroLevels.size(), MissingDouble());
		rhAvgUpper23.resize(numZeroLevels.size(), MissingDouble());

		// Keskim. lämpötila 1. nollarajan alapuolella, 1/2. ja 2/3. nollarajojen välisissä kerroksissa [C]
		Tavg01.resize(numZeroLevels.size(), MissingDouble());
		Tavg12.resize(numZeroLevels.size(), MissingDouble());
		Tavg23.resize(numZeroLevels.size(), MissingDouble());
		Tavg34.resize(numZeroLevels.size(), MissingDouble());

		// 1. nollarajan alapuolisen, 2/3. nollarajojen välisen, ja koko T>0 alueen koko [mC, "metriastetta"]
		plusArea = zeroLevel1;
		plusAreaSfc = zeroLevel1;

		// Mahdollisen pinta- tai 1/2. nollarajojen välisen pakkaskerroksen koko [mC, "metriastetta"]
		minusArea = zeroLevel1;

		log.Trace("Searching for first zero level height");
		zeroLevel1 = h->VerticalHeight(wantedParam, constData1, constData2, constData3, 1);

#ifdef DEBUG
		util::DumpVector(zeroLevel1, "zero level 1");
#endif

		log.Trace("Searching for average temperature between ground level and first zero level");
		Tavg01 = h->VerticalAverage(wantedParam, constData1, zeroLevel1);

#ifdef DEBUG
		util::DumpVector(Tavg01, "tavg 01");
#endif

		wantedParam = param("RH-PRCNT");

		log.Trace("Searching for average humidity between ground and first zero level");
		// Keskimääräinen RH nollarajan alapuolisessa plussakerroksessa
		rhAvg01 = h->VerticalAverage(wantedParam, constData1, zeroLevel1);

#ifdef DEBUG
		util::DumpVector(rhAvg01, "rh avg 01");
#endif

		// Only the first zero layer with at least one non-missing element is required
		// to be present. All other zero layers (2,3,4) are optional.

		try
		{
			// Values between zero levels 1 <--> 2
			wantedParam = param("T-K");

			log.Trace("Searching for second zero level height");
			zeroLevel2 = h->VerticalHeight(wantedParam, constData1, constData2, constData3, 2);

#ifdef DEBUG
			util::DumpVector(zeroLevel2, "zero level 2");
#endif

			log.Trace("Searching for average temperature between first and second zero level");
			Tavg12 = h->VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);

#ifdef DEBUG
			util::DumpVector(Tavg12, "tavg 12");
#endif

			log.Trace("Searching for average humidity between first and second zero level");

			// Keskimääräinen RH pakkaskerroksen yläpuolisessa plussakerroksessa
			wantedParam = param("RH-PRCNT");

			rhAvgUpper12 = h->VerticalAverage(wantedParam, zeroLevel1, zeroLevel2);

#ifdef DEBUG
			util::DumpVector(rhAvgUpper12, "rh avg upper 12");
#endif

			// 2 <--> 3
			wantedParam = param("T-K");

			log.Trace("Searching for third zero level height");
			zeroLevel3 = h->VerticalHeight(wantedParam, constData1, constData2, constData3, 3);

#ifdef DEBUG
			util::DumpVector(zeroLevel3, "zero level 3");
#endif

			log.Trace("Searching for average temperature between second and third zero level");
			Tavg23 = h->VerticalAverage(wantedParam, zeroLevel2, zeroLevel3);

#ifdef DEBUG
			util::DumpVector(Tavg23, "tavg 23");
#endif

			wantedParam = param("RH-PRCNT");

			log.Trace("Searching for average humidity between second and third zero level");

			// Keskimääräinen RH ylemmässä plussakerroksessa
			rhAvgUpper23 = h->VerticalAverage(wantedParam, zeroLevel2, zeroLevel3);

#ifdef DEBUG
			util::DumpVector(rhAvgUpper23, "rh avg upper 23");
#endif

			// 3 <--> 4
			wantedParam = param("T-K");

			log.Trace("Searching for fourth zero level height");
			zeroLevel4 = h->VerticalHeight(wantedParam, constData1, constData2, constData3, 4);

#ifdef DEBUG
			util::DumpVector(zeroLevel4, "zero level 4");
#endif

			log.Trace("Searching for average temperature between third and fourth zero level");
			Tavg34 = h->VerticalAverage(wantedParam, zeroLevel3, zeroLevel4);

#ifdef DEBUG
			util::DumpVector(Tavg34, "tavg 34");
#endif
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
	vector<double> rhMeltUpper(rhAvg01.size(), MissingDouble());
	vector<double> rhMelt(rhAvg01.size(), MissingDouble());

	// Keskimääräinen RH nollarajan alapuolisessa plussakerroksessa, ja pakkaskerroksen yläpuolisessa plussakerroksessa
	vector<double> rhAvgUpper(rhAvg01.size(), MissingDouble());
	vector<double> rhAvg(rhAvg01.size(), MissingDouble());

	for (size_t i = 0; i < numZeroLevels.size(); i++)
	{
		short numZeroLevel = static_cast<short>(numZeroLevels[i]);

		double pa = MissingDouble(), ma = MissingDouble(), pasfc = MissingDouble();

		// Kommentteja Simolta nollakohtien lukumäärään:
		// * Nollakohtien löytymättömyys ei ole ongelma, sillä tällöin olomuoto on aina lumi tai jäätävä tihku
		// * Parittomilla nollakohdilla ei voi tulla koskaan jäätävää sadetta koska pintakerros on plussalla
		//   (jolloin ainakin yleensä pätee oletus ettei jäätävää sadetta voi esiintyä)

		// nollarajoja parillinen määrä (pintakerros pakkasella)

		if (numZeroLevel % 2 == 0)
		{
			double zl1 = zeroLevel1[i], zl2 = zeroLevel2[i];
			double ta1 = Tavg01[i], ta2 = Tavg12[i];

			double paloft = MissingDouble();

			if (!IsMissing(zl1) && !IsMissing(zl2) && !IsMissing(ta1) && !IsMissing(ta2))
			{
				ta1 -= constants::kKelvin;
				ta2 -= constants::kKelvin;
				ma = zl1 * ta1;
				paloft = (zl2 - zl1) * ta2;

				// Keskimääräinen rhMelt pakkaskerroksen yläpuolisessa plussakerroksessa
				rhMeltUpper[i] = 9.5 * exp((-17.27 * ta2) / (ta2 + 238.3)) * (10.5 - ta2);
			}

			rhAvgUpper[i] = rhAvgUpper12[i];

			// Jos ylempänä toinen T>0 kerros, lasketaan myös sen koko (vähintään 4 nollarajaa) ja lisätään se alemman
			// kerroksen kokoon
			// (mahdollisista vielä ylempänä olevista plussakerroksista ei välitetä)
			// (tässäkin pitäisi tarkkaan ottaen tutkia rhAvgUpper, eli sulaako kerroksen läpi putoava lumi)

			if (numZeroLevel >= 4)
			{
				double zl3 = zeroLevel3[i], zl4 = zeroLevel4[i];
				ta2 = Tavg34[i];

				if (!IsMissing(zl3) && !IsMissing(zl4) && !IsMissing(ta2) && !IsMissing(paloft))
				{
					paloft = paloft + (zl4 - zl3) * (ta2 - constants::kKelvin);
				}
			}

			pa = paloft;
		}

		// nollarajoja pariton määrä (pintakerros plussalla)

		else if (numZeroLevel % 2 == 1)
		{
			double zl1 = zeroLevel1[i], ta1 = Tavg01[i];
			double paloft = MissingDouble();

			if (!IsMissing(zl1) && !IsMissing(ta1))
			{
				ta1 -= constants::kKelvin;
				pasfc = zl1 * ta1;
				pa = pasfc;

				// Lisäys 3.2.2015: Suhteellisen kosteuden (raja-) arvot pinnan plussakerroksessa:
				// Jos nollarajan alapuolisessa plussakerroksessa on kuivaa, asetetaan olomuodoksi lumi
				// (Shaviv, 2006: http://www.sciencebits.com/SnowAboveFreezing)

				// rhMelt = suht. kosteuden raja-arvo, jota pienemmillä kosteuksilla lumihiutaleet eivät sula

				// Keskimääräinen rhMelt nollarajan alapuolisessa plussakerroksessa

				rhMelt[i] = 9.5 * exp((-17.27 * ta1) / (ta1 + 238.3)) * (10.5 - ta1);
			}

			// Keskimääräinen RH nollarajan alapuolisessa plussakerroksessa
			rhAvg[i] = rhAvg01[i];

			// Jos ylempänä toinen T>0 kerros, lasketaan myös sen koko (vähintään 3 nollarajaa)
			// (mahdollisista vielä ylempänä olevista plussakerroksista ei välitetä)

			if (numZeroLevel >= 3)
			{
				double zl2 = zeroLevel2[i], zl3 = zeroLevel3[i];
				double ta2 = Tavg23[i];

				if (!IsMissing(zl2) && !IsMissing(zl3) && !IsMissing(ta2))
				{
					ta2 -= constants::kKelvin;

					paloft = (zl3 - zl2) * ta2;

					// Keskimääräinen rhMelt ylemmässä plussakerroksessa
					rhMeltUpper[i] = 9.5 * exp((-17.27 * ta2) / (ta2 + 238.3)) * (10.5 - ta2);

					// Keskimääräinen RH ylemmässä plussakerroksessa
					rhAvgUpper[i] = rhAvgUpper23[i];

					if (!IsMissing(rhAvgUpper[i]) AND !IsMissing(rhMeltUpper[i]) AND rhAvgUpper[i] > rhMeltUpper[i] &&
					    !IsMissing(pasfc))
					{
						pa = pasfc + paloft;
					}
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

	ret->Param(minusAreaParam);
	ret->Data().Set(minusArea);

	ret->Param(plusAreaParam);
	ret->Data().Set(plusArea);

	ret->Param(plusAreaSfcParam);
	ret->Data().Set(plusAreaSfc);

	ret->Param(rhAvgParam);
	ret->Data().Set(rhAvg);

	ret->Param(rhAvgUpperParam);
	ret->Data().Set(rhAvgUpper);

	ret->Param(rhMeltParam);
	ret->Data().Set(rhMelt);

	ret->Param(rhMeltUpperParam);
	ret->Data().Set(rhMeltUpper);

	result = ret;
}

void preform_hybrid::Stratus(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime,
                             const forecast_type& ftype, shared_ptr<info>& result, const grid* baseGrid)
{
	auto h = GET_PLUGIN(hitool);

	h->Configuration(conf);
	h->Time(ftime);
	h->ForecastType(ftype);

	// Kerroksen paksuus pinnasta [m], josta etsitään stratusta (min. BaseLimit+FZstLimit)
	const double layer = 2000.;

	// N-kynnysarvo vaaditulle min. stratuksen määrälle [%] (50=yli puoli taivasta):
	const double stCover = 0.5;

	// Kynnysarvo vaaditulle stratuksen yläpuolisen kuivan kerroksen paksuudelle [m] (jäätävässä) tihkussa:
	const double drydz = 1500.;

	vector<param> params = {
	    stratusBaseParam,           stratusTopParam,         stratusTopTempParam,         stratusMeanTempParam,
	    stratusMeanCloudinessParam, stratusUpperLayerNParam, stratusVerticalVelocityParam};
	vector<forecast_time> times = {ftime};
	vector<level> levels = {level(kHeight, 0, "HEIGHT")};

	auto ret = make_shared<info>(*conf->Info());
	ret->Params(params);
	ret->Levels(levels);
	ret->Times(times);
	ret->Create(baseGrid);

	vector<double> constData1(ret->Data().Size(), 0);

	auto constData2 = constData1;
	auto log = logger("preform_hybrid-stratus");

	try
	{
		// baseThreshold ja topThreshold:
		// (Paikan mukaan vaihtuva) N-kynnysarvo stratuksen ala- ja ylärajalle [%] (tarkkaa stCover arvoa ei aina löydy)

		vector<param> wantedParamList({param("N-0TO1"), param("N-PRCNT")});

		log.Info("Searching for stratus lower limit");

		auto baseThreshold = h->VerticalMinimum(wantedParamList, 0, stLimit);

		for (size_t i = 0; i < baseThreshold.size(); i++)
		{
			assert(!IsMissing(baseThreshold[i]));

			if (baseThreshold[i] < stCover)
			{
				baseThreshold[i] = stCover;
			}
		}

#ifdef DEBUG
		util::DumpVector(baseThreshold);
#endif

		ret->Param(stratusBaseParam);
		ret->Data().Set(baseThreshold);

		log.Info("Searching for stratus upper limit");

		auto topThreshold = h->VerticalMinimum(wantedParamList, stLimit, layer);

		for (size_t i = 0; i < topThreshold.size(); i++)
		{
			assert(!IsMissing(topThreshold[i]));
			if (topThreshold[i] < stCover)
			{
				topThreshold[i] = stCover;
			}
		}

#ifdef DEBUG
		util::DumpVector(topThreshold);
#endif

		// Stratus Base/top [m]
		// _findh: 0 = viimeinen löytyvä arvo pinnasta ylöspäin, 1 = ensimmäinen löytyvä arvo

		log.Info("Searching for stratus base accurate value");

		auto stratusBase = h->VerticalHeight(wantedParamList, 0, stLimit, baseThreshold);

		ret->Param(stratusBaseParam);
		ret->Data().Set(stratusBase);

#ifdef DEBUG
		util::DumpVector(stratusBase);
#endif

		log.Info("Searching for stratus top accurate value");
		auto stratusTop = h->VerticalHeight(wantedParamList, stLimit, layer, topThreshold, 0);

		ret->Param(stratusTopParam);
		ret->Data().Set(stratusTop);

#ifdef DEBUG
		util::DumpVector(stratusTop);
#endif

		try
		{
			log.Info("Searching for cloudiness in layers above stratus top");

			assert(constData1.size() == constData2.size() && constData1.size() == stratusTop.size());

			for (size_t i = 0; i < constData1.size(); i++)
			{
				if (IsMissing(stratusTop[i]))
				{
					constData1[i] = MissingDouble();
					constData2[i] = MissingDouble();
				}
				else
				{
					constData1[i] = stratusTop[i] + 100;
					constData2[i] = stratusTop[i] + drydz;
				}
			}

			auto upperLayerN = h->VerticalAverage(wantedParamList, constData1, constData2);

			ret->Param(stratusUpperLayerNParam);
			ret->Data().Set(upperLayerN);

#ifdef DEBUG
			util::DumpVector(upperLayerN);
#endif

			log.Info("Searching for stratus mean cloudiness");

			// Stratuksen keskimääräinen N
			auto stratusMeanN = h->VerticalAverage(wantedParamList, stratusBase, stratusTop);

#ifdef DEBUG
			util::DumpVector(stratusMeanN);
#endif

			ret->Param(stratusMeanCloudinessParam);
			ret->Data().Set(stratusMeanN);

			log.Info("Searching for stratus top temperature");

			param wantedParam("T-K");

			// Stratuksen Topin lämpötila (jäätävä tihku)
			auto stratusTopTemp = h->VerticalValue(wantedParam, stratusTop);

#ifdef DEBUG
			util::DumpVector(stratusTopTemp);
#endif

			ret->Param(stratusTopTempParam);
			ret->Data().Set(stratusTopTemp);

			log.Info("Searching for stratus mean temperature");

			for (size_t i = 0; i < constData1.size(); i++)
			{
				double lower = stratusBase[i];
				double upper = stratusTop[i];

				if (IsMissing(lower) || IsMissing(upper))
				{
					constData1[i] = MissingDouble();
					constData2[i] = MissingDouble();
				}
				else if (fabs(lower - upper) < 100)
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

			// St:n keskimääräinen lämpötila (poissulkemaan kylmät <-10C stratukset, joiden toppi >-10C) (jäätävä tihku)
			auto stratusMeanTemp = h->VerticalAverage(wantedParam, constData1, constData2);

#ifdef DEBUG
			util::DumpVector(stratusMeanTemp);
#endif

			ret->Param(stratusMeanTempParam);
			ret->Data().Set(stratusMeanTemp);

			log.Info("Searching for mean vertical velocity in stratus");

			wantedParam = param("VV-MMS");

			vector<double> stratusVerticalVelocity;

			try
			{
				// Keskimääräinen vertikaalinopeus st:ssa [mm/s]
				stratusVerticalVelocity = h->VerticalAverage(wantedParam, stratusBase, stratusTop);
			}
			catch (const HPExceptionType& e)
			{
				if (e == kFileDataNotFound)
				{
					log.Debug("Trying for param VV-MS");
					wantedParam = param("VV-MS");

					stratusVerticalVelocity = h->VerticalAverage(wantedParam, stratusBase, stratusTop);

					for (double& d : stratusVerticalVelocity)
					{
						if (!IsMissing(d)) d *= 1000;
					}
				}
			}

#ifdef DEBUG
			util::DumpVector(stratusVerticalVelocity, "stratus vertical velocity");
#endif

			ret->Param(stratusVerticalVelocityParam);
			ret->Data().Set(stratusVerticalVelocity);
		}
		catch (const HPExceptionType& e)
		{
			if (e != kFileDataNotFound)
			{
				throw runtime_error("Stratus() caught exception " + to_string(e));
			}
		}
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Stratus() caught exception " + to_string(e));
		}
		else
		{
			return;
		}
	}

	result = ret;
}
