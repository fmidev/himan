/**
 * @file vvms.cpp
 *
 */
#include "vvms.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"

using namespace std;
using namespace himan::plugin;

#ifdef HAVE_CUDA
namespace vvmsgpu
{
extern void Process(std::shared_ptr<const himan::plugin_configuration> conf,
                    std::shared_ptr<himan::info<double>> myTargetInfo);
}
#endif
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

void vvms::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	auto myThreadedLogger = logger("vvmsThread #" + to_string(threadIndex));

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

		vvmsgpu::Process(itsConfiguration, myTargetInfo);
	}
	else
#endif
	{
		deviceType = "CPU";

		double PScale = 1;

		/*
		 * If vvms is calculated for pressure levels, the P value
		 * equals to level value. Otherwise we have to fetch P
		 * separately.
		 */

		info_t PInfo;

		bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

		info_t VVInfo = Fetch(forecastTime, forecastLevel, VVParam, forecastType, false);
		info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, false);

		if (!isPressureLevel)
		{
			// Source info for P
			PInfo = Fetch(forecastTime, forecastLevel, PParam, forecastType, false);
		}

		if (!VVInfo || !TInfo || (!isPressureLevel && !PInfo))
		{
			myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
			                         static_cast<string>(forecastLevel));
			return;
		}

		if (PInfo && (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA"))
		{
			PScale = 100;
		}

		ASSERT(TInfo->Grid()->AB() == VVInfo->Grid()->AB() &&
		       (isPressureLevel || PInfo->Grid()->AB() == TInfo->Grid()->AB()));

		SetAB(myTargetInfo, TInfo);

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

			double w = itsScale * (287 * -VV * T / (himan::constants::kG * P * PScale));

			myTargetInfo->Value(w);
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
