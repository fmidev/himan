#include <math.h>

#include "pot.h"

#include "fetcher.h"
#include "forecast_time.h"
#include "hitool.h"
#include "info.h"
#include "level.h"
#include "logger.h"
#include "matrix.h"
#include "numerical_functions.h"
#include "plugin_factory.h"

using namespace std;
using namespace himan::plugin;

pot::pot()
{
	itsLogger = logger("pot");
}

void pot::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param POT("POT-PRCNT", aggregation(), processing_type(kProbability));

	POT.Unit(kPrcnt);

	SetParams({POT});

	Start();
}

void pot::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	/*
	 * Required source parameters
	 */

	const param CapeParamHiman("CAPE1040-JKG");
	const level MU(kMaximumThetaE, 0);
	const level ML(kHeightLayer, 500, 0);
	const param RainParam("RRR-KGM2");
	const param ELHeight("EL-LAST-M");
	const param LCLHeight("LCL-M");
	const param LCLTemp("LCL-K");
	const param LPLHeight("LPL-M");

	forecast_type forecastType = myTargetInfo->ForecastType();
	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	auto myThreadedLogger = logger("pot_pluginThread #" + to_string(threadIndex));

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	shared_ptr<info<double>> CAPEMuInfo, CAPEMlInfo, CbTopMaxInfo, RRInfo, LclMuInfo, LclMlInfo, LclMuTempInfo,
	    LclMlTempInfo, LplMuInfo, LplMlInfo;

	// Fetch params
	CAPEMuInfo = Fetch(forecastTime, MU, CapeParamHiman, forecastType, false);
	CAPEMlInfo = Fetch(forecastTime, ML, CapeParamHiman, forecastType, false);

	CbTopMaxInfo = Fetch(forecastTime, MU, ELHeight, forecastType, false);

	LclMuInfo = Fetch(forecastTime, MU, LCLHeight, forecastType, false);
	LclMuTempInfo = Fetch(forecastTime, MU, LCLTemp, forecastType, false);
	LclMlInfo = Fetch(forecastTime, ML, LCLHeight, forecastType, false);
	LclMlTempInfo = Fetch(forecastTime, ML, LCLTemp, forecastType, false);

	LplMuInfo = Fetch(forecastTime, MU, LPLHeight, forecastType, false);

	RRInfo = Fetch(forecastTime, forecastLevel, RainParam, forecastType, false);

	if (!CAPEMuInfo || !CAPEMlInfo || !CbTopMaxInfo || !LclMuInfo || !LclMuTempInfo || !LclMlInfo || !LclMlTempInfo ||
	    !LplMuInfo || !RRInfo)
	{
		myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	const double smallRadius = 20;
	const double largeRadius = 35;

	int smallFilterSizeX = 3;
	int smallFilterSizeY = 3;
	int largeFilterSizeX = 5;
	int largeFilterSizeY = 5;

	const double di = dynamic_pointer_cast<regular_grid>(myTargetInfo->Grid())->Di();
	const double dj = dynamic_pointer_cast<regular_grid>(myTargetInfo->Grid())->Dj();

	switch (myTargetInfo->Grid()->Type())
	{
		case kLatitudeLongitude:
		case kRotatedLatitudeLongitude:
			smallFilterSizeX = static_cast<int>((smallRadius / di / 111.0));
			smallFilterSizeY = static_cast<int>((smallRadius / dj / 111.0));
			largeFilterSizeX = static_cast<int>((largeRadius / di / 111.0));
			largeFilterSizeY = static_cast<int>((largeRadius / dj / 111.0));
			break;
		case kStereographic:
		case kLambertConformalConic:
			smallFilterSizeX = static_cast<int>(round(smallRadius / di * 1000.0));
			smallFilterSizeY = static_cast<int>(round(smallRadius / dj * 1000.0));
			largeFilterSizeX = static_cast<int>(round(largeRadius / di * 1000.0));
			largeFilterSizeY = static_cast<int>(round(largeRadius / dj * 1000.0));
			break;
		default:
			break;
	}

	// make sure filters have at least size 1;
	smallFilterSizeX = smallFilterSizeX < 1 ? 1 : smallFilterSizeX;
	smallFilterSizeY = smallFilterSizeY < 1 ? 1 : smallFilterSizeY;
	largeFilterSizeX = largeFilterSizeX < 1 ? 1 : largeFilterSizeX;
	largeFilterSizeY = largeFilterSizeY < 1 ? 1 : largeFilterSizeY;

	// filters
	himan::matrix<double> small_filter_kernel(smallFilterSizeX, smallFilterSizeY, 1, MissingDouble(), 1.0);
	himan::matrix<double> large_filter_kernel(largeFilterSizeX, largeFilterSizeY, 1, MissingDouble(), 1.0);

	// Find nearby (20km radius) gridpoint X that gives maximum most unstable CAPE
	// The algorithm then uses CAPE, LPL, EL, LCL and CIN values from the specific gridpoint X
	himan::matrix<size_t> filtered_CAPE_indices =
	    numerical_functions::IndexMax2D(CAPEMuInfo->Data(), small_filter_kernel);

	// There are two paths in the algorithm depending on LPL:
	// If convection is approx. surface based, LPL(mu) < 250, use mean-layer (ml) parameters
	// If convection is elevated, LPL(mu) > 250, use most unstable (mu) parameters
	std::vector<double> Cape, Lcl, LclT;
	Cape.reserve(myTargetInfo->Data().Size());
	Lcl.reserve(myTargetInfo->Data().Size());
	LclT.reserve(myTargetInfo->Data().Size());

	for (size_t idx : filtered_CAPE_indices.Values())
	{
		if (LplMuInfo->Data().At(idx) < 250)
		{
			Cape.push_back(CAPEMlInfo->Data().At(idx));
			Lcl.push_back(LclMlInfo->Data().At(idx));
			LclT.push_back(LclMlTempInfo->Data().At(idx));
		}
		else
		{
			Cape.push_back(CAPEMuInfo->Data().At(idx));
			Lcl.push_back(LclMuInfo->Data().At(idx));
			LclT.push_back(LclMuTempInfo->Data().At(idx));
		}
	}

	// Lift filtering
	himan::matrix<double> filtered_PoLift = numerical_functions::Prob2D<double>(
	    RRInfo->Data(), large_filter_kernel, [](const double& x) { return x >= 0.1; });

	// hitool to find Cb/LCL Top temps
	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());

	vector<double> CbTopTemp;

	try
	{
		CbTopTemp = h->VerticalValue<double>(param("T-K"), VEC(CbTopMaxInfo));
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			return;
		}

		throw;
	}

	string deviceType = "CPU";

	// starting point of the algorithm POT v2.5
	for (auto&& tup :
	     zip_range(VEC(myTargetInfo), Cape, filtered_PoLift.Values(), VEC(CbTopMaxInfo), CbTopTemp, Lcl, LclT))
	{
		double& POT = tup.get<0>();
		const double& CAPE = tup.get<1>();
		const double& PoLift = tup.get<2>();
		const double& Cb_top = tup.get<3>();
		const double& Cb_top_temp = tup.get<4>() - himan::constants::kKelvin;
		const double& LCL = tup.get<5>();
		const double& LCL_temp = tup.get<6>() - himan::constants::kKelvin;

		double PoThermoDyn = 0;  // Probability of ThermoDynamics = todennäköisyys ukkosta suosivalle termodynamiikalle
		double PoColdTop = 0;    // Probability of Cold Top, riittävän kylmä pilven toppi
		double PoMixedPhase = 0;  // Probability of Mixed Phase, Todennäköisyys sekafaasikerrokseen
		double PoDepth = 0;       // Probability of Depth, konvektiota tulee tapahtua riittävän paksussa kerroksessa

		const double verticalVelocity = sqrt(2 * CAPE);

		// Relaatio pystynopeuden ja todennäköisyyden välillä:
		// Todennäköisyys kasvaa 0->1, kun pystynopeus kasvaa 5->30 m/s
		if (verticalVelocity >= 5 && verticalVelocity <= 25)
		{
			PoThermoDyn = 0.05 * verticalVelocity - 0.25;
		}

		if (verticalVelocity > 25)
		{
			PoThermoDyn = 1;
		}

		// Salamoinnin kannalta tarpeeksi kylmän pilven topin todennäköisyys kasvaa
		// 0 --> 1, kun pilven topin lämpötila laskee -15C --> -30C
		if (Cb_top_temp <= -15 && Cb_top_temp >= -30)
		{
			PoColdTop = -0.06666667 * Cb_top_temp - 1;
		}

		if (Cb_top_temp < -30)
		{
			PoColdTop = 1;
		}

		// Probability of Mixed Phase
		// Konvektiopilvessä tulee olla tarpeeksi paksu sekafaasikerros, jotta sähköistyminen voi tapahtua.
		// Näin ollen pilven pohjan korkeudella (LCL-tasolla) lämpötila ei saa olla kylmempi kuin ~ -12C
		if (LCL_temp >= -10 && LCL_temp <= 0)
		{
			PoMixedPhase = 0.1 * LCL_temp + 1;
		}

		if (LCL_temp > 0)
		{
			PoMixedPhase = 1;
		}

		// Probability of Depth
		// Konvektion tulee tapahtua riittävän paksussa kerroksessa LFC-->EL

		double depth = Cb_top - LCL;

		if (depth >= 2500.0 && depth <= 5000.0)
		{
			PoDepth = 0.0004 * depth - 1;
		}

		if (depth > 5000.0)
		{
			PoDepth = 1;
		}

		POT = PoLift * PoThermoDyn * PoColdTop * PoMixedPhase * PoDepth * 100;
	}

	const himan::matrix<double> filter_kernel(3, 3, 1, MissingDouble(), 1 / 9.);

	const auto smooth_pot =
	    numerical_functions::Filter2D<double>(myTargetInfo->Data(), filter_kernel, itsConfiguration->UseCuda());

	myTargetInfo->Base()->data = move(smooth_pot);

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
