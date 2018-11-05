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

void WithTD(shared_ptr<himan::info<float>> myTargetInfo, shared_ptr<himan::info<float>> TInfo,
            shared_ptr<himan::info<float>> TDInfo);
void WithQ(shared_ptr<himan::info<float>> myTargetInfo, shared_ptr<himan::info<float>> TInfo,
           shared_ptr<himan::info<float>> QInfo, shared_ptr<himan::info<float>> PInfo, float PScale);
void WithQ(shared_ptr<himan::info<float>> myTargetInfo, shared_ptr<himan::info<float>> TInfo,
           shared_ptr<himan::info<float>> QInfo, float P);

#ifdef HAVE_CUDA
extern void ProcessHumidityGPU(std::shared_ptr<const himan::plugin_configuration> conf,
                               std::shared_ptr<himan::info<float>> myTargetInfo);
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

	Start<float>();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void relative_humidity::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
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

		auto TInfo =
		    Fetch<float>(forecastTime, forecastLevel, TParam, forecastType, itsConfiguration->UseCudaForPacking());

		if (!TInfo)
		{
			itsLogger.Warning("Skipping step " + to_string(myTargetInfo->Time().Step()) + ", level " +
			                  to_string(myTargetInfo->Level().Value()));
			return;
		}

		// First try to calculate using Q and P

		auto QInfo =
		    Fetch<float>(forecastTime, forecastLevel, QParam, forecastType, itsConfiguration->UseCudaForPacking());

		if (!QInfo)
		{
			auto TDInfo =
			    Fetch<float>(forecastTime, forecastLevel, TDParam, forecastType, itsConfiguration->UseCudaForPacking());

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
			WithQ(myTargetInfo, TInfo, QInfo,
			      static_cast<float>(myTargetInfo->Level().Value()));  // Pressure is needed as hPa, no scaling
		}
		else
		{
			auto PInfo =
			    Fetch<float>(forecastTime, forecastLevel, PParams, forecastType, itsConfiguration->UseCudaForPacking());

			if (!PInfo)
			{
				myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
				                         static_cast<string>(forecastLevel));
				return;
			}

			float PScale = 1;

			if (PInfo->Param().Name() == "P-PA")
			{
				PScale = 0.01f;
			}

			WithQ(myTargetInfo, TInfo, QInfo, PInfo, PScale);
		}

		SetAB(myTargetInfo, TInfo);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

void WithQ(shared_ptr<himan::info<float>> myTargetInfo, shared_ptr<himan::info<float>> TInfo,
           shared_ptr<himan::info<float>> QInfo, float P)
{
	// Pressure needs to be hPa and temperature C

	const float ep = static_cast<float>(himan::constants::kEp);

	for (auto&& tup : zip_range(VEC(myTargetInfo), VEC(TInfo), VEC(QInfo)))
	{
		float& result = tup.get<0>();
		const float T = tup.get<1>();
		const float Q = tup.get<2>();

		const float es = himan::metutil::Es_<float>(T) * 0.01f;

		result = (P * Q / ep / es) * (P - es) / (P - Q * P / ep);

		if (himan::IsMissing(result))
		{
			continue;
		}

		result = fminf(fmaxf(0.0f, result), 1.0f) * 100;  // scale to range 0 .. 100
	}
}

void WithQ(shared_ptr<himan::info<float>> myTargetInfo, shared_ptr<himan::info<float>> TInfo,
           shared_ptr<himan::info<float>> QInfo, shared_ptr<himan::info<float>> PInfo, float PScale)
{
	// Pressure needs to be hPa and temperature C

	const float ep = static_cast<float>(himan::constants::kEp);

	for (auto&& tup : zip_range(VEC(myTargetInfo), VEC(TInfo), VEC(QInfo), VEC(PInfo)))
	{
		float& result = tup.get<0>();
		const float T = tup.get<1>();
		const float Q = tup.get<2>();
		const float P = tup.get<3>() * PScale;

		const float es = himan::metutil::Es_<float>(T) * 0.01f;

		result = (P * Q / ep / es) * (P - es) / (P - Q * P / ep);

		if (himan::IsMissing(result))
		{
			continue;
		}

		result = fminf(fmaxf(0.0f, result), 1.0f) * 100;  // scale to range 0 .. 100
	}
}

void WithTD(shared_ptr<himan::info<float>> myTargetInfo, shared_ptr<himan::info<float>> TInfo,
            shared_ptr<himan::info<float>> TDInfo)
{
	const float b = 17.27f;
	const float c = 237.3f;
	const float d = 1.8f;
	const float k = static_cast<float>(himan::constants::kKelvin);

	for (auto&& tup : zip_range(VEC(myTargetInfo), VEC(TInfo), VEC(TDInfo)))
	{
		float& result = tup.get<0>();
		const float T = tup.get<1>() - k;
		const float TD = tup.get<2>() - k;

		result = exp(d + b * (TD / (TD + c))) / exp(d + b * (T / (T + c)));

		if (himan::IsMissing(result))
		{
			continue;
		}

		result = fminf(fmaxf(0.0f, result), 1.0f) * 100;  // scale to range 0 .. 100
	}
}
