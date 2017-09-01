/**
 * @file icing.cpp
 *
 */

#include "icing.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"

using namespace std;
using namespace himan::plugin;

icing::icing()
{
	itsLogger = logger("icing");
}

void icing::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("ICING-N", 480, 0, 19, 7)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void icing::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{
	// Required source parameters

	const param TParam("T-K");
	const params VvParam = {param("VV-MS"), param("VV-MMS")};
	const param ClParam("CLDWAT-KGKG");

	auto myThreadedLogger = logger("icingThread #" + to_string(theThreadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
						  static_cast<string>(forecastLevel));

	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, false);
	info_t VvInfo = Fetch(forecastTime, forecastLevel, VvParam, forecastType, false);
	info_t ClInfo = Fetch(forecastTime, forecastLevel, ClParam, forecastType, false);

	if (!TInfo || !VvInfo || !ClInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	double VvScale = 1;  // Assume we'll have VV-MMS
	double ClScale = 1000;

	if (VvInfo->Param().Name() == "VV-MS")
	{
		VvScale = 1000;
	}

	assert(TInfo->Grid()->AB() == VvInfo->Grid()->AB() && TInfo->Grid()->AB() == ClInfo->Grid()->AB());

	SetAB(myTargetInfo, TInfo);

	string deviceType = "CPU";

	auto& target = VEC(myTargetInfo);

	// LOCKSTEP(myTargetInfo, TInfo, VvInfo, ClInfo)
	for (auto&& tup : zip_range(target, VEC(TInfo), VEC(VvInfo), VEC(ClInfo)))
	{
		double& result = tup.get<0>();
		double T = tup.get<1>();
		double Vv = tup.get<2>();
		double Cl = tup.get<3>();

		if (IsMissingValue({T, Vv, Cl}))
		{
			continue;
		}

		double Icing;
		double TBase = constants::kKelvin;
		int vCor = kHPMissingInt;
		int tCor = kHPMissingInt;

		T = T - TBase;
		Vv *= VvScale;
		Cl *= ClScale;

		// Vertical velocity correction factor

		if (Vv < 0)
		{
			vCor = -1;
		}
		else if ((Vv >= 0) && (Vv <= 50))
		{
			vCor = 0;
		}
		else if ((Vv >= 50) && (Vv <= 100))
		{
			vCor = 1;
		}
		else if ((Vv >= 100) && (Vv <= 200))
		{
			vCor = 2;
		}
		else if ((Vv >= 200) && (Vv <= 300))
		{
			vCor = 3;
		}
		else if ((Vv >= 300) && (Vv <= 1000))
		{
			vCor = 4;
		}
		else if (Vv > 1000)
		{
			vCor = 5;
		}

		// Temperature correction factor

		if ((T <= 0) && (T > -1))
		{
			tCor = -2;
		}
		else if ((T <= -1) && (T > -2))
		{
			tCor = -1;
		}
		else if ((T <= -2) && (T > -3))
		{
			tCor = 0;
		}
		else if ((T <= -3) && (T > -12))
		{
			tCor = 1;
		}
		else if ((T <= -12) && (T > -15))
		{
			tCor = 0;
		}
		else if ((T <= -15) && (T > -18))
		{
			tCor = -1;
		}
		else if (T < -18)
		{
			tCor = -2;
		}
		else
		{
			tCor = 0;
		}

		if ((Cl <= 0) || (T > 0))
		{
			Icing = 0;
		}
		else
		{
			Icing = round(log(Cl) + 6) + vCor + tCor;
		}

		// Maximum and minimum values for index

		if (Icing > 15)
		{
			Icing = 15;
		}

		if (Icing < 0)
		{
			Icing = 0;
		}

		result = Icing;
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
