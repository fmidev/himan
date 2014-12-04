/**
 * @file density.cpp
 *
 * Computes density for dry air from pressure and temperature using the ideal gas law.
 *
 * @date Mar 11, 2014
 * @author Tack
 */

#include "density.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include "level.h"
#include "forecast_time.h"

using namespace std;
using namespace himan::plugin;

const string itsName("density");

density::density()
{
	itsClearTextFormula = "rho = P / (Rd * T)";
	itsLogger = logger_factory::Instance()->GetLog(itsName);
}

void density::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("RHO-KGM3", 63, 0, 3, 10)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void density::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	// Required source parameters

	const params PParam = { param("P-PA"), param("P-HPA") };
	const param TParam("T-K");

	auto myThreadedLogger = logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	double PScale = 1;

	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, false);

	info_t PInfo;
	bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

	if(!isPressureLevel)
	{
		PInfo = Fetch(forecastTime, forecastLevel, PParam, false);
	}

	if (!TInfo || (!isPressureLevel && !PInfo))
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}


	if (!isPressureLevel && (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA"))
	{
		PScale = 100;
	}

	SetAB(myTargetInfo, TInfo);
		
	string deviceType = "CPU";

	if (PInfo)
	{
		PInfo->ResetLocation();
	}

	LOCKSTEP(myTargetInfo, TInfo)
	{

		double P = kFloatMissing;
		double T = TInfo->Value();

		if (isPressureLevel)
		{
			P = 100 * myTargetInfo->Level().Value();
		}
		else
		{
			PInfo->NextLocation();
			P = PInfo->Value();
		}

		if (P == kFloatMissing || T == kFloatMissing )
		{
			continue;
		}

		// actual calculation of the density using the ideal gas law
		double rho = P * PScale / (constants::kRd * T);

		myTargetInfo->Value(rho);
		
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}
