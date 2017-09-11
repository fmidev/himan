/**
 * @file visibility
 */

#include "visibility.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"

#include "fetcher.h"
#include "hitool.h"

using namespace std;
using namespace himan;
using namespace himan::plugin;

const double defaultVis = 40000;

// Lumi- tai räntäsateen intensiteetin "tehostus" [mm/h]
// Tällä siis tarkoitus saada mallin heikotkin lumisadeintensiteetit huonontamaan näkyvyyttä
const double pseudoRR = 0.13;

// Raja-arvo merkittävälle sumupilven määrälle [0..1]
const double stLimit = 0.55;

// Sumupilven max korkeus [m], 305m = 1000ft
const double stMaxH = 305.;

const himan::params PFParams({himan::param("PRECFORM2-N"), himan::param("PRECFORM-N")});
const himan::params RHParam({himan::param("RH-PRCNT"), himan::param("RH-0TO1")});
const himan::params RRParam({himan::param("RR-1-MM"), himan::param("RRR-KGM2")});
const himan::params NParam({himan::param("N-PRCNT"), himan::param("N-0TO1")});

// ..and their levels
const himan::level NLevel(himan::kHeight, 0, "HEIGHT");
const himan::level RHLevel(himan::kHeight, 2, "HEIGHT");

double VisibilityInRain(double stN, double stH, double RR, double RH, int PF);
double VisibilityInMist(double stN, double stH, double RR, double RH);

visibility::visibility()
{
	itsLogger = logger("visibility");
}

void visibility::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("VV2-M", 407)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void visibility::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("visibilityThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	info_t RHInfo = Fetch(forecastTime, RHLevel, RHParam, forecastType, false);
	info_t PFInfo = Fetch(forecastTime, NLevel, PFParams, forecastType, false);
	info_t RRInfo = Fetch(forecastTime, NLevel, RRParam, forecastType, false);

	if (!RRInfo || !RHInfo || !PFInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	double RHScale = 1;

	if (itsConfiguration->SourceProducer().Id() == 1 || itsConfiguration->SourceProducer().Id() == 199 ||
	    itsConfiguration->SourceProducer().Id() == 4)
	{
		RHScale = 100;
	}

	auto h = GET_PLUGIN(hitool);

	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());

	// Alle 304m (1000ft) sumupilven (max) määrä [0..1]
	auto stN = h->VerticalMaximum(NParam, 0, stMaxH);

	// Stratus <15m (~0-50ft)
	auto st15 = h->VerticalAverage(NParam, 0, 15);

	// Stratus 15-45m (~50-150ft)
	auto st45 = h->VerticalAverage(NParam, 15, 45);

	// Sumupilven korkeus [m]
	auto stH = h->VerticalHeightGreaterThan(NParam, 0, stMaxH, stLimit);

	// Jos sumupilveä ohut kerros (vain) ~alimmalla mallipinnalla, jätetään alin kerros huomioimatta
	// (ehkä mieluummin ylempää keskiarvo, jottei tällöin mahdollinen ylempi st-kerros huononna näkyvyyttä liikaa?)
	auto stHup = h->VerticalHeightGreaterThan(NParam, 25, stMaxH, stLimit);
	auto stNup = h->VerticalAverage(NParam, 15, stMaxH);

	assert(stH.size() == stHup.size());
	assert(st15.size() == stH.size());

	for (size_t i = 0; i < stH.size(); i++)
	{
		if (st15[i] > stLimit && st45[i] < stLimit)
		{
			stN[i] = stNup[i];
			stH[i] = stHup[i];
		}
	}

	string deviceType = "CPU";

	auto& target = VEC(myTargetInfo);

	for (auto&& tup : zip_range(target, VEC(PFInfo), VEC(RRInfo), VEC(RHInfo), stN, stH))
	{
		double& result = tup.get<0>();
		double PF = tup.get<1>();
		double RR = tup.get<2>();
		double RH = tup.get<3>();
		double stratN = tup.get<4>();
		double stratH = tup.get<5>();

		if (IsMissing(RR) || IsMissing(RH) )
		{
			continue;
		}

		assert(stratN <= 1.0);
		assert(RR < 50);

		stratN *= 100;  // Note! Cloudiness is scaled to percents

		RH *= RHScale;

		assert(RH < 102.);

		double visPre = defaultVis;

		if (RR > 0 && IsValid(PF))
		{
			visPre = VisibilityInRain(stratN, stratH, RR, RH, static_cast<int>(PF));
		}

		double visMist = VisibilityInMist(stratN, stratH, RR, RH);

		// Choose lower visibility to be the end result

		result = fmin(visMist, visPre);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

double VisibilityInRain(double stN, double stH, double RR, double RH, int PF)
{
	double visPre = defaultVis;

	// Näkyvyyden utuisuuskerroin sateessa 2m suhteellisen kosteuden perusteella [0,85...8,5, kun 100%<rh<10%]
	// (jos rh<85%, näkyvyyttä parannetaan)

	const double RHpre = 85. / RH;

	// Näkyvyyden utuisuuskerroin sateessa matalan sumupilven (alle 1000ft) määrän perusteella [1...0,85, kun
	// 50<N<100%]
	// Alle 50% sumupilven määrä ei siis huononna näkyvyyttä sateessa

	const double stNpre = (stN >= 50) ? log(50) / log(stN) : 1;

	// Näkyvyyden utuisuuskerroin sateessa matalan sumupilvikorkeuden perusteella [0,68...1, kun stH=12...152m]
	// Yli 152m (500ft) korkeudella oleva sumupilvi ei siis huononna näkyvyyttä sateessa

	const double stHpre = (stH < 152) ? pow((stH / 152), 0.15) : 1;

	switch (PF)
	{
		case 0:
		case 4:
			// Tihku (tai jäätävä tihku)
			// Nakyvyys intensiteetin perusteella
			visPre = 1. / RR * 500;
			break;

		case 1:
		case 5:
			// Vesisade (tai jäätävä vesisade)
			// Nakyvyys intensiteetin perusteella
			// (kaava antaa ehkä turhan huonoja <4000m näkyvyyksiä reippaassa RR>4 vesisateessa)
			visPre = 1 / RR * 6000 + 2000;
			break;

		case 3:
			// Snow
			// Näkyvyys intensiteetin perusteella
			visPre = 1 / (RR + pseudoRR) * 1400;
			break;

		case 2:
			// Sleet
			// Näkyvyys intensiteetin perusteella
			visPre = 1 / (RR + pseudoRR) * 2000;
			break;
		default:
			throw runtime_error("New unhandled precipitation form value: " + to_string(PF));
	}

	// Mahdollinen lisähuononnus utuisuuden perusteella
	visPre = visPre * RHpre * stNpre * stHpre;

	return fmin(visPre, defaultVis);
}

double VisibilityInMist(double stN, double stH, double RR, double RH)
{
	double visMist = defaultVis;

	// Näkyvyyden utuisuuskerroin udussa/sumussa sumupilven määrän perusteella [7,5...0,68, kun stN = 0...100%]
	const double stNmist = 75. / (stN + 10);

	// Nakyvyyden utuisuuskerroin udussa/sumussa sumupilvikorkeuden perusteella [ 0,47...1, kun par500 = 50...999ft]
	const double stHmist = (stH < 305) ? pow((stH / 0.3048 / 1000), 0.25) : 1;

	// Nakyvyys udussa/sumussa, lasketaan myos heikossa sateessa (tarkoituksena tasoittaa suuria
	// nakyvyysgradientteja sateen reunalla)
	// (ehka syyta rajata vain tilanteisiin, jossa sateen perusteella saatu nakyvyys oli viela >8000?)

	if ((RR < 0.5 || IsMissing(RR)) && RH > 80)
	{
		// Näkyvyys udussa/sumussa
		// Yksinkertaistetty kaava, eli lasketaan samalla tavalla riippumatta siitä, onko pakkasta vai ei
		// (pakkaset eri tavalla voisi olla parempi tapa)

		// Alkuarvo suht. kosteuden perusteella [700-31400m, kun 100<=RH<80]
		visMist = pow((101 - RH), 1.25) * 700;
		// Lisamuokkaus sumupilven maaran ja korkeuden perusteella
		visMist = visMist * stNmist * stHmist;
	}

	return fmin(visMist, defaultVis);
}
