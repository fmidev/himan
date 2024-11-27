/**
 * @file vvms.cpp
 *
 */
#include "vvms.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "util.h"

using namespace std;
using namespace himan::plugin;

#ifdef HAVE_CUDA
namespace vvmsgpu
{
extern void Process(shared_ptr<const himan::plugin_configuration> conf, shared_ptr<himan::info<float>> myTargetInfo,
                    bool reverse, float scale);
}
#endif
// Required source parameters

const himan::param TParam("T-K");
const himan::params PParam = {himan::param("P-PA"), himan::param("P-HPA")};
himan::param VVParam("VV-PAS");

vvms::vvms() : itsScale(1), itsReverseCalculation(false)
{
	itsCudaEnabledCalculation = true;

	itsLogger = logger("vvms");
}

void vvms::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to vertical velocity
	 */

	param targetParam("VV-MS");

	if (itsConfiguration->Exists("millimeters") && util::ParseBoolean(itsConfiguration->GetValue("millimeters")))
	{
		targetParam = param("VV-MMS");
		itsScale = 1000;
	}

	if (itsConfiguration->Exists("reverse") && util::ParseBoolean(itsConfiguration->GetValue("reverse")))
	{
		targetParam = param("VV-PAS");
		itsReverseCalculation = true;
		VVParam = param("VV-MS");

		if (itsConfiguration->Exists("millimeters") && util::ParseBoolean(itsConfiguration->GetValue("millimeters")))
		{
			VVParam = param("VV-MMS");
			itsScale = 1000;
		}
	}

	SetParams({targetParam});

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

	myThreadedLogger.Info(fmt::format("Calculating time {} level {}", forecastTime.ValidDateTime(), forecastLevel));

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";

		vvmsgpu::Process(itsConfiguration, myTargetInfo, itsReverseCalculation, static_cast<float>(itsScale));
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
		const float density = static_cast<float>(himan::constants::kRd);
		if (itsReverseCalculation == false)
		{
			for (auto&& tup : zip_range(result, VEC(TInfo), VEC(VVInfo), PData))
			{
				float& vv = tup.get<0>();
				const float& t = tup.get<1>();
				const float& vvpas = tup.get<2>();
				const float& p = PScale * tup.get<3>();

				vv = itsScale * (density * -vvpas * t / (gravity * p));
			}
		}
		else
		{
			for (auto&& tup : zip_range(result, VEC(TInfo), VEC(VVInfo), PData))
			{
				float& vv = tup.get<0>();
				const float& t = tup.get<1>();
				const float& vv_ms = tup.get<2>();
				const float& p = PScale * tup.get<3>();

				vv = (vv_ms / itsScale) * (gravity * p) / (density * -t);
			}
		}
	}
	myThreadedLogger.Info(fmt::format("[{}] Missing values: {}/{}", deviceType, myTargetInfo->Data().MissingCount(),
	                                  myTargetInfo->Data().Size()));
}
