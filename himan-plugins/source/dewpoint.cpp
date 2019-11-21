/**
 * @file dewpoint.cpp
 *
 */

#include "dewpoint.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "moisture.h"

using namespace std;
using namespace himan::plugin;

#ifdef HAVE_CUDA
namespace dewpointgpu
{
extern void Process(std::shared_ptr<const himan::plugin_configuration> conf,
                    std::shared_ptr<himan::info<double>> myTargetInfo);
}
#endif

dewpoint::dewpoint()
{
	itsCudaEnabledCalculation = true;

	itsLogger = logger("dewpoint");
}

void dewpoint::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to dewpoint.
	 *
	 */

	param requestedParam("TD-K", 10, 0, 0, 6);
	requestedParam.Unit(kK);

	SetParams({requestedParam});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void dewpoint::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	const param TParam("T-K");
	const params RHParam = {param("RH-PRCNT"), param("RH-0TO1")};

	auto myThreadedLogger = logger("dewpointThread #" + to_string(threadIndex));

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

		dewpointgpu::Process(itsConfiguration, myTargetInfo);
	}
	else
#endif
	{
		double TBase = 0;
		double RHScale = 1;

		info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, itsConfiguration->UseCudaForPacking());
		info_t RHInfo =
		    Fetch(forecastTime, forecastLevel, RHParam, forecastType, itsConfiguration->UseCudaForPacking());

		if (!TInfo || !RHInfo)
		{
			myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
			                         static_cast<string>(forecastLevel));
			return;
		}

		SetAB(myTargetInfo, TInfo);

		if (RHInfo->Param().Name() == "RH-0TO1")
		{
			RHScale = 100.0;
		}

		// Formula assumes T == Celsius

		if (TInfo->Param().Unit() == kC)
		{
			TBase = himan::constants::kKelvin;
		}

		deviceType = "CPU";
		auto& target = VEC(myTargetInfo);
		const auto& TVec = VEC(TInfo);
		const auto& RHVec = VEC(RHInfo);

		for (auto&& tup : zip_range(target, TVec, RHVec))
		{
			double& result = tup.get<0>();

			const double T = tup.get<1>();
			const double RH = tup.get<2>();

			result = metutil::DewPointFromRH_<double>(T + TBase, RH * RHScale);
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
