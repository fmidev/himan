/**
 * @file dewpoint.cpp
 *
 */

#include "dewpoint.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"

using namespace std;
using namespace himan::plugin;

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

void dewpoint::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	const param TParam("T-K");
	const params RHParam = {param("RH-PRCNT"), param("RH-0TO1")};

	auto myThreadedLogger = logger("dewpointThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	double TBase = 0;
	double RHScale = 1;

	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, itsConfiguration->UseCudaForPacking());
	info_t RHInfo = Fetch(forecastTime, forecastLevel, RHParam, forecastType, itsConfiguration->UseCudaForPacking());

	if (!TInfo || !RHInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	assert(TInfo->Grid()->AB() == RHInfo->Grid()->AB());

	SetAB(myTargetInfo, TInfo);

	// Special case for harmonie & MEPS
	if (RHInfo->Param().Unit() != kPrcnt || itsConfiguration->SourceProducer().Id() == 199)
	{
		itsLogger.Warning("Unable to determine RH unit, assuming 0 .. 1");
		RHScale = 100.0;
	}

	// Formula assumes T == Celsius

	if (TInfo->Param().Unit() == kC)
	{
		TBase = himan::constants::kKelvin;
	}

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";

		auto opts = CudaPrepare(myTargetInfo, TInfo, RHInfo);

		dewpoint_cuda::Process(*opts);
	}
	else
#endif
	{
		deviceType = "CPU";

		LOCKSTEP(myTargetInfo, TInfo, RHInfo)
		{
			double T = TInfo->Value();
			double RH = RHInfo->Value();

			T += TBase;
			RH *= RHScale;

			double TD = metutil::DewPointFromRH_(T, RH);

			myTargetInfo->Value(TD);
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

#ifdef HAVE_CUDA

unique_ptr<dewpoint_cuda::options> dewpoint::CudaPrepare(shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo,
                                                         shared_ptr<info> RHInfo)
{
	unique_ptr<dewpoint_cuda::options> opts(new dewpoint_cuda::options);

	opts->t = TInfo->ToSimple();
	opts->rh = RHInfo->ToSimple();
	opts->td = myTargetInfo->ToSimple();

	opts->N = opts->td->size_x * opts->td->size_y;

	if (TInfo->Param().Unit() == kC)
	{
		opts->t_base = himan::constants::kKelvin;
	}

	if (RHInfo->Param().Unit() != kPrcnt || itsConfiguration->SourceProducer().Id() == 199)
	{
		// If unit cannot be detected, assume the values are from 0 .. 1
		opts->rh_scale = 100;
	}

	return opts;
}

#endif
