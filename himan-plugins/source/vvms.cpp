/**
 * @file vvms.cpp
 *
 */
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "vvms.h"

using namespace std;
using namespace himan::plugin;

#include "cuda_helper.h"
#include "vvms.cuh"

// Required source parameters

const himan::param TParam("T-K");
const himan::params PParam = {himan::param("P-PA"), himan::param("P-HPA")};
const himan::param VVParam("VV-PAS");

vvms::vvms() : itsScale(1)
{
	itsCudaEnabledCalculation = true;

	itsLogger = logger("vvms");
}

void vvms::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to vertical velocity
	 */

	param theRequestedParam("VV-MS", 143);

	if (itsConfiguration->Exists("millimeters") && itsConfiguration->GetValue("millimeters") == "true")
	{
		theRequestedParam = param("VV-MMS", 43, 0, 2, 9);
		itsScale = 1000;
	}

	SetParams({theRequestedParam});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void vvms::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("vvmsThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	double PScale = 1;
	double TBase = 0;

	/*
	 * If vvms is calculated for pressure levels, the P value
	 * equals to level value. Otherwise we have to fetch P
	 * separately.
	 */

	info_t PInfo;

	bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

	info_t VVInfo = Fetch(forecastTime, forecastLevel, VVParam, forecastType, itsConfiguration->UseCudaForPacking());
	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, itsConfiguration->UseCudaForPacking());

	if (!isPressureLevel)
	{
		// Source info for P
		PInfo = Fetch(forecastTime, forecastLevel, PParam, forecastType, itsConfiguration->UseCudaForPacking());
	}

	if (!VVInfo || !TInfo || (!isPressureLevel && !PInfo))
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	if (PInfo && (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA"))
	{
		PScale = 100;
	}

	assert(TInfo->Grid()->AB() == VVInfo->Grid()->AB() &&
	       (isPressureLevel || PInfo->Grid()->AB() == TInfo->Grid()->AB()));

	SetAB(myTargetInfo, TInfo);

	if (TInfo->Param().Unit() == kC)
	{
		TBase = himan::constants::kKelvin;
	}

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";

		auto opts = CudaPrepare(myTargetInfo, TInfo, VVInfo, PInfo);

		vvms_cuda::Process(*opts);
	}
	else
#endif
	{
		deviceType = "CPU";

		if (PInfo)
		{
			PInfo->ResetLocation();
		}

		// Assume pressure level calculation

		double P = 100 * myTargetInfo->Level().Value();

		LOCKSTEP(myTargetInfo, TInfo, VVInfo)
		{
			double T = TInfo->Value();
			double VV = VVInfo->Value();

			if (!isPressureLevel)
			{
				PInfo->NextLocation();
				P = PInfo->Value();
			}

			double w = itsScale * (287 * -VV * (T + TBase) / (himan::constants::kG * P * PScale));

			myTargetInfo->Value(w);
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

#ifdef HAVE_CUDA

unique_ptr<vvms_cuda::options> vvms::CudaPrepare(shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo,
                                                 shared_ptr<info> VVInfo, shared_ptr<info> PInfo)
{
	unique_ptr<vvms_cuda::options> opts(new vvms_cuda::options);

	opts->is_constant_pressure = (myTargetInfo->Level().Type() == kPressure);

	opts->t = TInfo->ToSimple();
	opts->vv = VVInfo->ToSimple();
	opts->vv_ms = myTargetInfo->ToSimple();

	if (!opts->is_constant_pressure)
	{
		opts->p = PInfo->ToSimple();

		if (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA")
		{
			opts->p_scale = 100;
		}
	}
	else
	{
		opts->p_const = myTargetInfo->Level().Value() * 100;  // Pa
	}

	if (TInfo->Param().Unit() == kC)
	{
		opts->t_base = himan::constants::kKelvin;
	}

	opts->N = TInfo->Grid()->Size();

	opts->vv_ms_scale = itsScale;

	return opts;
}
#endif
