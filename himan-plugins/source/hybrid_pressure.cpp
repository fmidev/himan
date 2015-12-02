/**
 * @file hybrid_pressure.cpp
 *
 *  @date: Mar 23, 2013
 *  @author aaltom
 */

#include "hybrid_pressure.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "level.h"
#include "forecast_time.h"

using namespace std;
using namespace himan::plugin;

hybrid_pressure::hybrid_pressure()
{
	// Vertkoord_A and Vertkoord_B refer to full hybrid-level coefficients
	itsClearTextFormula = "P = Vertkoord_A + P0 * Vertkoord_B";

	itsLogger = logger_factory::Instance()->GetLog("hybrid_pressure");

}

void hybrid_pressure::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("P-HPA", 1, 0, 3, 0)});

	Start();

}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void hybrid_pressure::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{

	params PParam{ param("P-PA"), param("P-HPA")};
	const param TParam("T-K");
	level PLevel(himan::kHeight, 0, "HEIGHT");

	bool isECMWF = (itsConfiguration->SourceProducer().Id() == 131); // Note! This only checks the *current* source producer

	if (isECMWF)
	{
		// For EC we calculate surface pressure from LNSP parameter
		PParam = { param("LNSP-N") };
		PLevel = level(himan::kHybrid, 1);
	}

	auto myThreadedLogger = logger_factory::Instance()->GetLog("hybrid_pressureThread #" + boost::lexical_cast<string> (theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	info_t PInfo = Fetch(forecastTime, PLevel, PParam, forecastType, false);
	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, false);

	if (!PInfo || !TInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}
	
	SetAB(myTargetInfo, TInfo);

	/* 
	 * Vertical coordinates for full hybrid levels.
	 * For Hirlam data, coefficients A and B are already interpolated to full level coefficients in the grib-file.
	 * For Harmonie and ECMWF interpolation is done, when reading data from the grib-file. (NFmiGribMessage::PV)
	 */

	std::vector<double> ab = TInfo->Grid()->AB();

   	double A = ab[0];
   	double B = ab[1];

#define ZIP

#ifdef ZIP
	auto& target = VEC(myTargetInfo);

	for (auto&& tup : zip_range(target, VEC(PInfo)))
	{
		double& result  = tup.get<0>();
		double P        = tup.get<1>();

		if (P == kFloatMissing)
		{
			continue;
		}

		if (isECMWF)
		{
			P = exp (P);
		}

		result = 0.01 * (A + P * B);
	}

#else
	LOCKSTEP(myTargetInfo, PInfo)
	{
		double P = PInfo->Value();
		
		if (P == kFloatMissing)
		{
			continue;
		}

		if (isECMWF)
		{
			P = exp (P);
		}
		
		double hybrid_pressure = 0.01 * (A + P * B);

		myTargetInfo->Value(hybrid_pressure);
	}
#endif
	myThreadedLogger->Info("[CPU] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}
