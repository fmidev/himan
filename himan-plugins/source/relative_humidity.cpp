/**
 * @file relative_humidity.cpp
 *
 */

#include "relative_humidity.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "moisture.h"

using namespace std;
using namespace himan::plugin;

void WithTD(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t TDInfo);
void WithQ(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t QInfo, himan::info_t PInfo, double PScale);
void WithQ(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t QInfo, double P);

#ifdef HAVE_CUDA
extern void ProcessHumidityGPU(std::shared_ptr<const himan::plugin_configuration> conf,
                               std::shared_ptr<himan::info<double>> myTargetInfo);
#endif

relative_humidity::relative_humidity()
{
	itsCudaEnabledCalculation = true;

	itsLogger = logger("relative_humidity");
}

void relative_humidity::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("RH-PRCNT", 13, 0, 1, 1)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void relative_humidity::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	const param TParam("T-K");
	const params PParams = {param("P-HPA"), param("P-PA")};
	const param QParam("Q-KGKG");
	const param TDParam("TD-K");

	auto myThreadedLogger = logger("relative_humidityThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";

		ProcessHumidityGPU(itsConfiguration, myTargetInfo);
	}
	else
#endif
	{
		deviceType = "CPU";

		// Temperature is always needed

		info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, itsConfiguration->UseCudaForPacking());

		if (!TInfo)
		{
			itsLogger.Warning("Skipping step " + to_string(myTargetInfo->Time().Step()) + ", level " +
			                  to_string(myTargetInfo->Level().Value()));
			return;
		}

		// First try to calculate using Q and P

		info_t QInfo = Fetch(forecastTime, forecastLevel, QParam, forecastType, itsConfiguration->UseCudaForPacking());

		if (!QInfo)
		{
			auto TDInfo =
			    Fetch(forecastTime, forecastLevel, TDParam, forecastType, itsConfiguration->UseCudaForPacking());

			if (!TDInfo)
			{
				myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
				                         static_cast<string>(forecastLevel));
				return;
			}

			WithTD(myTargetInfo, TInfo, TDInfo);
		}
		else if (myTargetInfo->Level().Type() == kPressure)
		{
			WithQ(myTargetInfo, TInfo, QInfo, myTargetInfo->Level().Value());  // Pressure is needed as hPa, no scaling
		}
		else
		{
			auto PInfo =
			    Fetch(forecastTime, forecastLevel, PParams, forecastType, itsConfiguration->UseCudaForPacking());

			if (!PInfo)
			{
				myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
				                         static_cast<string>(forecastLevel));
				return;
			}

			//	ASSERT(!PInfo || TInfo->Grid()->AB() == PInfo->Grid()->AB());
			//	ASSERT(!TDInfo || TInfo->Grid()->AB() == TDInfo->Grid()->AB());
			//	ASSERT(!QInfo || TInfo->Grid()->AB() == QInfo->Grid()->AB());
			double PScale = 1;

			if (PInfo->Param().Name() == "P-PA")
			{
				PScale = 0.01;
			}

			WithQ(myTargetInfo, TInfo, QInfo, PInfo, PScale);
		}

		SetAB(myTargetInfo, TInfo);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

void WithQ(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t QInfo, double P)
{
	// Pressure needs to be hPa and temperature C

	for (auto&& tup : zip_range(VEC(myTargetInfo), VEC(TInfo), VEC(QInfo)))
	{
		double& result = tup.get<0>();
		const double T = tup.get<1>();
		const double Q = tup.get<2>();

		const double es = himan::metutil::Es_<double>(T) * 0.01;

		result = (P * Q / himan::constants::kEp / es) * (P - es) / (P - Q * P / himan::constants::kEp);

		if (himan::IsMissing(result))
		{
			continue;
		}

		result = fmin(fmax(0.0, result), 1.0) * 100;  // scale to range 0 .. 100
	}
}

void WithQ(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t QInfo, himan::info_t PInfo, double PScale)
{
	// Pressure needs to be hPa and temperature C

	for (auto&& tup : zip_range(VEC(myTargetInfo), VEC(TInfo), VEC(QInfo), VEC(PInfo)))
	{
		double& result = tup.get<0>();
		const double T = tup.get<1>();
		const double Q = tup.get<2>();
		const double P = tup.get<3>() * PScale;

		const double es = himan::metutil::Es_<double>(T) * 0.01;

		result = (P * Q / himan::constants::kEp / es) * (P - es) / (P - Q * P / himan::constants::kEp);

		if (himan::IsMissing(result))
		{
			continue;
		}

		result = fmin(fmax(0.0, result), 1.0) * 100;  // scale to range 0 .. 100
	}
}

void WithTD(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t TDInfo)
{
	const double b = 17.27;
	const double c = 237.3;
	const double d = 1.8;

	for (auto&& tup : zip_range(VEC(myTargetInfo), VEC(TInfo), VEC(TDInfo)))
	{
		double& result = tup.get<0>();
		const double T = tup.get<1>() - himan::constants::kKelvin;
		const double TD = tup.get<2>() - himan::constants::kKelvin;

		result = exp(d + b * (TD / (TD + c))) / exp(d + b * (T / (T + c)));

		if (himan::IsMissing(result))
		{
			continue;
		}

		result = fmin(fmax(0.0, result), 1.0) * 100;  // scale to range 0 .. 100
	}
}
