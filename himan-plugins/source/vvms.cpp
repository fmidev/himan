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
                    std::shared_ptr<himan::info<float>> myTargetInfo);
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

	Start<float>();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void vvms::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
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

		float PScale = 1.f;

		/*
		 * If vvms is calculated for pressure levels, the P value
		 * equals to level value. Otherwise we have to fetch P
		 * separately.
		 */

		shared_ptr<info<float>> PInfo;

		bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

		auto VVInfo = Fetch<float>(forecastTime, forecastLevel, VVParam, forecastType, false);
		auto TInfo = Fetch<float>(forecastTime, forecastLevel, TParam, forecastType, false);

		if (!isPressureLevel)
		{
			// Source info for P
			PInfo = Fetch<float>(forecastTime, forecastLevel, PParam, forecastType, false);
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

		SetAB(myTargetInfo, TInfo);

		vector<float> PData;
		if (PInfo)
		{
			PData = VEC(PInfo);
		}
		else
		{
			PData =
			    vector<float>(myTargetInfo->Data().Size(), 100.f * static_cast<float>(myTargetInfo->Level().Value()));
		}

		auto& result = VEC(myTargetInfo);
		const float gravity = static_cast<float>(himan::constants::kG);

		for (auto&& tup : zip_range(result, VEC(TInfo), VEC(VVInfo), PData))
		{
			float& vv = tup.get<0>();
			const float& t = tup.get<1>();
			const float& vvpas = tup.get<2>();
			const float& p = PScale * tup.get<3>();

			vv = itsScale * (287.f * -vvpas * t / (gravity * p));
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
