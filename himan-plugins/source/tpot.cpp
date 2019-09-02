/**
 * @file tpot.cpp
 *
 * @brief Plugin to calculate potential temperature, pseudo-adiabatic
 * potential temperature or equivalent potential temperature.
 */

#include "tpot.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"

using namespace std;
using namespace himan::plugin;

#include "moisture.h"

#ifdef HAVE_CUDA
namespace tpotgpu
{
void Process(std::shared_ptr<const himan::plugin_configuration> conf, std::shared_ptr<himan::info<double>> myTargetInfo,
             bool theta, bool thetaw, bool thetae);
}
#endif

tpot::tpot() : itsThetaCalculation(false), itsThetaWCalculation(false), itsThetaECalculation(false)
{
	itsCudaEnabledCalculation = true;

	itsLogger = logger("tpot");
}

void tpot::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to potential temperature
	 */

	vector<param> theParams;

	if (itsConfiguration->Exists("theta") && itsConfiguration->GetValue("theta") == "true")
	{
		itsThetaCalculation = true;

		itsLogger.Trace("Theta calculation requested");

		param p("TP-K", 8, 0, 0, 2);

		theParams.push_back(p);
	}

	if (itsConfiguration->Exists("thetaw") && itsConfiguration->GetValue("thetaw") == "true")
	{
		itsThetaWCalculation = true;

		itsLogger.Trace("ThetaW calculation requested");

		// Sharing GRIB2 number with thetae!

		param p("TPW-K", 9, 0, 0, 3);

		theParams.push_back(p);
	}

	if (itsConfiguration->Exists("thetae") && itsConfiguration->GetValue("thetae") == "true")
	{
		itsThetaECalculation = true;

		itsLogger.Trace("ThetaE calculation requested");

		param p("TPE-K", 129, 0, 0, 3);

		// Sharing GRIB2 number with thetaw!

		theParams.push_back(p);
	}

	if (theParams.size() == 0)
	{
		// By default assume we'll calculate theta

		itsThetaCalculation = true;

		param p("TP-K", 8, 0, 0, 2);

		theParams.push_back(p);
	}

	SetParams(theParams);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void tpot::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("tpotThread #" + to_string(threadIndex));

	const param TParam("T-K");
	const params PParam = {param("P-PA"), param("P-HPA")};
	const param TDParam("TD-K");

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	double PScale = 1;
	double TBase = 0;
	double TDBase = 0;

	info_t TInfo = Fetch(forecastTime, forecastLevel, param("T-K"), myTargetInfo->ForecastType(),
	                     itsConfiguration->UseCudaForPacking());
	info_t TDInfo, PInfo;

	bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

	if (!isPressureLevel)
	{
		PInfo = Fetch(forecastTime, forecastLevel, PParam, myTargetInfo->ForecastType(),
		              itsConfiguration->UseCudaForPacking());

		if (PInfo && (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA"))
		{
			PScale = 100;
		}
	}

	if (itsThetaWCalculation || itsThetaECalculation)
	{
		TDInfo = Fetch(forecastTime, forecastLevel, TDParam, myTargetInfo->ForecastType(),
		               itsConfiguration->UseCudaForPacking());
	}

	if (!TInfo || (!isPressureLevel && !PInfo) || ((itsThetaWCalculation || itsThetaECalculation) && !TDInfo))
	{
		myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	SetAB(myTargetInfo, TInfo);

	if (TInfo->Param().Unit() == kC)
	{
		TBase = -himan::constants::kKelvin;
	}

	if (TDInfo && TDInfo->Param().Unit() == kC)
	{
		TDBase = -himan::constants::kKelvin;
	}

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";

		tpotgpu::Process(itsConfiguration, myTargetInfo, itsThetaCalculation, itsThetaWCalculation,
		                 itsThetaECalculation);
	}
	else
#endif
	{
		deviceType = "CPU";

		if (PInfo)
		{
			PInfo->ResetLocation();
		}

		if (TDInfo)
		{
			TDInfo->ResetLocation();
		}

		LOCKSTEP(myTargetInfo, TInfo)
		{
			double T = TInfo->Value() + TBase;  // to Kelvin

			double P, TD = MissingDouble();

			if (isPressureLevel)
			{
				P = myTargetInfo->Level().Value() * 100;
			}
			else
			{
				PInfo->NextLocation();
				P = PInfo->Value() * PScale;  // to Pa
			}

			if (itsThetaWCalculation || itsThetaECalculation)
			{
				TDInfo->NextLocation();
				TD = TDInfo->Value() + TDBase;  // to Kelvin
				ASSERT(TD >= 80. || IsMissing(TD));
			}

			if (itsThetaCalculation)
			{
				double value = metutil::Theta_<double>(T, P);

				myTargetInfo->Find<param>(param("TP-K"));

				myTargetInfo->Value(value);
			}

			if (itsThetaWCalculation)
			{
				double value = metutil::ThetaW_<double>(metutil::ThetaE_<double>(T, TD, P));

				myTargetInfo->Find<param>(param("TPW-K"));

				myTargetInfo->Value(value);
			}

			if (itsThetaECalculation)
			{
				double value = metutil::ThetaE_<double>(T, TD, P);

				myTargetInfo->Find<param>(param("TPE-K"));

				myTargetInfo->Value(value);
			}
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
