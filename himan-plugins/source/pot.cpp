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

pot::pot() : itsStrictMode(false)
{
	itsLogger = logger("pot");
}

void pot::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param POT("POT-PRCNT");

	POT.Unit(kPrcnt);

	if (itsConfiguration->GetValue("strict") == "true")
	{
		itsStrictMode = true;
	}

	SetParams({POT});

	Start();
}

void pot::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	/*
	 * Required source parameters
	 */

	const param CapeParamEC("CAPE-JKG");
	const param CapeParamHiman("CAPE1040-JKG");
	const level MU(kMaximumThetaE, 0);
	const param RainParam("RRR-KGM2");
	const param ELHeight("EL-LAST-M");
	const param LCLHeight("LCL-M");
	const param LCLTemp("LCL-K");
	const param LFCHeight("LFC-M");

	forecast_type forecastType = myTargetInfo->ForecastType();
	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	auto myThreadedLogger = logger("pot_pluginThread #" + to_string(threadIndex));

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	info_t CAPEMaxInfo, CbTopMaxInfo, LfcMinInfo, RRInfo, LclInfo, LclTempInfo;

	// Fetch params
	CAPEMaxInfo = Fetch(forecastTime, MU, CapeParamHiman, forecastType, false);
	if (!CAPEMaxInfo)
		CAPEMaxInfo = Fetch(forecastTime, forecastLevel, CapeParamEC, forecastType, false);
	CbTopMaxInfo = Fetch(forecastTime, MU, ELHeight, forecastType, false);
	LfcMinInfo = Fetch(forecastTime, MU, LFCHeight, forecastType, false);
	LclInfo = Fetch(forecastTime, MU, LCLHeight, forecastType, false);
	LclTempInfo = Fetch(forecastTime, MU, LCLTemp, forecastType, false);
	RRInfo = Fetch(forecastTime, forecastLevel, RainParam, forecastType, false);

	if (!CAPEMaxInfo || !CbTopMaxInfo || !LfcMinInfo || !LclInfo || !LclTempInfo || !RRInfo)
	{
		myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	const double smallRadius = 35;
	const double largeRadius = 62;

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

	// filters
	himan::matrix<double> small_filter_kernel(smallFilterSizeX, smallFilterSizeY, 1, MissingDouble(), 1.0);
	himan::matrix<double> large_filter_kernel(largeFilterSizeX, largeFilterSizeY, 1, MissingDouble(),
	                                          1.0 / (largeFilterSizeX * largeFilterSizeY));

	// Cape filtering
	himan::matrix<double> filtered_CAPE = numerical_functions::Max2D(CAPEMaxInfo->Data(), small_filter_kernel);

	// Cb_top filtering
	himan::matrix<double> filtered_CbTop = numerical_functions::Max2D(CbTopMaxInfo->Data(), small_filter_kernel);

	// LFC filtering
	himan::matrix<double> filtered_LFC = numerical_functions::Min2D(LfcMinInfo->Data(), small_filter_kernel);

	// Lift filtering
	himan::matrix<double> filtered_PoLift =
	    numerical_functions::Reduce2D(RRInfo->Data(), large_filter_kernel,
	                                  [=](double& val1, double& val2, const double& a, const double& b) {
		                                  val2++;
		                                  if (IsValid(a) && a >= 0.1)
			                                  val1 += b;
	                                  },
	                                  [](const double& val1, const double& val2) { return val1; }, 0.0, 0.0);

	// hitool to find Cb/LCL Top temps
	auto h = GET_PLUGIN(hitool);
	h->Configuration(itsConfiguration);
	h->Time(myTargetInfo->Time());
	h->ForecastType(myTargetInfo->ForecastType());

	vector<double> CbTopTemp;

	try
	{
		CbTopTemp = h->VerticalValue<double>(param("T-K"), filtered_CbTop.Values());
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
	for (auto&& tup : zip_range(VEC(myTargetInfo), filtered_CAPE.Values(), filtered_PoLift.Values(),
	                            filtered_CbTop.Values(), CbTopTemp, filtered_LFC.Values(), VEC(LclTempInfo)))
	{
		double& POT = tup.get<0>();
		const double& CAPE = tup.get<1>();
		const double& PoLift = tup.get<2>();
		const double& Cb_top = tup.get<3>();
		const double& Cb_top_temp = tup.get<4>() - himan::constants::kKelvin;
		const double& LFC = tup.get<5>();
		const double& LCL_temp = tup.get<6>() - himan::constants::kKelvin;

		double PoThermoDyn = 0;  // Probability of ThermoDynamics = todennäköisyys ukkosta suosivalle termodynamiikalle
		double PoColdTop = 0;    // Probability of Cold Top, riittävän kylmä pilven toppi
		double PoMixedPhase = 0;  // Probability of Mixed Phase, Todennäköisyys sekafaasikerrokseen
		double PoDepth = 0;       // Probability of Depth, konvektiota tulee tapahtua riittävän paksussa kerroksessa

		const double verticalVelocity = sqrt(2 * CAPE);

		// Relaatio pystynopeuden ja todennäköisyyden välillä:
		// Todennäköisyys kasvaa 0->1, kun pystynopeus kasvaa 5->30 m/s
		if (verticalVelocity >= 5 && verticalVelocity <= 30)
		{
			PoThermoDyn = 0.04 * verticalVelocity - 0.2;
		}

		if (verticalVelocity > 30)
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
		if (LCL_temp >= -12 && LCL_temp <= 0)
		{
			PoMixedPhase = 0.0833333 * LCL_temp + 1;
		}

		if (LCL_temp > 0)
		{
			PoMixedPhase = 1;
		}

		// Probability of Depth
		// Konvektion tulee tapahtua riittävän paksussa kerroksessa LFC-->EL

		double depth = Cb_top - LFC;

		if (depth >= 2000.0 && depth <= 4000.0)
		{
			PoDepth = 0.0005 * depth - 1;
		}

		if (depth > 4000.0)
		{
			PoDepth = 1;
		}

		POT = PoLift * PoThermoDyn * PoColdTop * PoMixedPhase * PoDepth * 100;
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
