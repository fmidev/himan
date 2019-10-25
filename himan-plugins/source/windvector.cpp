/**
 * @file windvector.cpp
 */

#include "windvector.h"
#include "forecast_time.h"
#include "interpolate.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "stereographic_grid.h"
#include "util.h"
#include <iostream>
#include <math.h>

#include "cache.h"
#include "fetcher.h"

using namespace std;
using namespace himan::plugin;

#include "cuda_helper.h"

typedef tuple<float, float, float, float> coefficients;

windvector::windvector() : itsCalculationTarget(kUnknownElement), itsVectorCalculation(false)
{
	itsCudaEnabledCalculation = true;

	itsLogger = logger("windvector");
}

void windvector::Process(const std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to windvector
	 */

	vector<param> theParams;

	param requestedDirParam;
	param requestedSpeedParam;
	param requestedVectorParam;

	if (itsConfiguration->Exists("do_vector") && itsConfiguration->GetValue("do_vector") == "true")
	{
		itsVectorCalculation = true;
	}

	if (itsConfiguration->Exists("for_ice") && itsConfiguration->GetValue("for_ice") == "true")
	{
		requestedSpeedParam = param("IFF-MS", 389, 10, 2, 3);
		requestedDirParam = param("IDD-D", 390, 10, 2, 2);

		itsCalculationTarget = kIce;

		if (itsVectorCalculation)
		{
			itsLogger.Warning("Unable to calculate vector for ice");
		}
	}
	else if (itsConfiguration->Exists("for_sea") && itsConfiguration->GetValue("for_sea") == "true")
	{
		requestedSpeedParam = param("SFF-MS", 163, 10, 1, 1);
		requestedDirParam = param("SDD-D", 164, 10, 1, 0);

		itsCalculationTarget = kSea;

		if (itsVectorCalculation)
		{
			itsLogger.Warning("Unable to calculate vector for sea");
		}
	}
	else if (itsConfiguration->Exists("for_gust") && itsConfiguration->GetValue("for_gust") == "true")
	{
		requestedSpeedParam = param("FFG-MS", 417, 0, 2, 22);

		itsCalculationTarget = kGust;

		if (itsVectorCalculation)
		{
			itsLogger.Warning("Unable to calculate vector for wind gust");
		}
	}
	else
	{
		// By default assume we'll calculate for wind

		requestedDirParam = param("DD-D", 20, 0, 2, 0);
		requestedSpeedParam = param("FF-MS", 21, 0, 2, 1);

		if (itsVectorCalculation)
		{
			requestedVectorParam = param("DF-MS", 22);
		}

		itsCalculationTarget = kWind;
	}

	theParams.push_back(requestedSpeedParam);

	if (itsCalculationTarget != kGust)
	{
		theParams.push_back(requestedDirParam);
	}

	if (itsVectorCalculation)
	{
		theParams.push_back(requestedVectorParam);
	}

	SetParams(theParams);

	Start<float>();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void windvector::Calculate(shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
{
	// Required source parameters

	param UParam;
	param VParam;

	float directionOffset = 180;  // For wind direction add this

	switch (itsCalculationTarget)
	{
		case kSea:
			UParam = param("WVELU-MS");
			VParam = param("WVELV-MS");
			directionOffset = 0;
			break;

		case kIce:
			UParam = param("IVELU-MS");
			VParam = param("IVELV-MS");
			directionOffset = 0;
			break;

		case kGust:
			UParam = param("WGU-MS");
			VParam = param("WGV-MS");
			break;

		case kWind:
			UParam = param("U-MS");
			VParam = param("V-MS");
			break;

		default:
			throw runtime_error("Invalid calculation target element: " +
			                    to_string(static_cast<int>(itsCalculationTarget)));
			break;
	}

	auto myThreadedLogger = logger("windvectorThread #" + to_string(threadIndex));

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

		windvector_cuda::RunCuda(itsConfiguration, myTargetInfo, UParam, VParam, itsCalculationTarget);
	}
	else
#endif
	{
		deviceType = "CPU";

		auto UInfo = FetchOne(forecastTime, forecastLevel, UParam, forecastType, itsConfiguration->UseCudaForPacking());
		auto VInfo = FetchOne(forecastTime, forecastLevel, VParam, forecastType, itsConfiguration->UseCudaForPacking());

		if (!UInfo || !VInfo)
		{
			myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
			                         static_cast<string>(forecastLevel));
			return;
		}

		for (myTargetInfo->Reset<param>(); myTargetInfo->Next<param>();)
		{
			SetAB(myTargetInfo, UInfo);
		}

		ASSERT(UInfo->Grid()->Type() == VInfo->Grid()->Type());

#ifdef HAVE_CUDA
		if (UInfo->PackedData()->HasData())
		{
			util::Unpack<float>({UInfo, VInfo}, false);
		}
#endif

		// We need to make sure that vector components are rotated to earth-relative form -- fetcher
		// does not do it if source projection == target projection
		// By using a dummy latlon area we make sure that rotation is only done to earth relative form

		latitude_longitude_grid x;

		interpolate::RotateVectorComponents(UInfo->Grid().get(), &x, *UInfo, *VInfo, itsConfiguration->UseCuda());

		auto c = GET_PLUGIN(cache);

		c->Replace(UInfo);
		c->Replace(VInfo);

		myTargetInfo->Index<param>(0);

		auto& FFVec = VEC(myTargetInfo);
		vector<float> DDVec(FFVec.size(), MissingFloat());

		for (auto&& tup : zip_range(FFVec, DDVec, VEC(UInfo), VEC(VInfo)))
		{
			float& speed = tup.get<0>();
			float& dir = tup.get<1>();
			float U = tup.get<2>();
			float V = tup.get<3>();

			if (IsMissingValue({U, V}))
			{
				continue;
			}

			speed = sqrt(U * U + V * V);

			if (itsCalculationTarget == kGust)
			{
				continue;
			}

			dir = static_cast<float>(himan::constants::kRad) * atan2(U, V) + directionOffset;

			// reduce the angle
			dir = fmodf(dir, 360);

			// force it to be the positive remainder, so that 0 <= dir < 360
			dir = round(fmodf((dir + 360), 360));
		}

		if (myTargetInfo->Size<param>() > 1)
		{
			myTargetInfo->Index<param>(1);
			myTargetInfo->Data().Set(DDVec);
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

shared_ptr<himan::info<float>> windvector::FetchOne(const forecast_time& theTime, const level& theLevel,
                                                    const param& theParam, const forecast_type& theType,
                                                    bool returnPacked) const
{
	auto f = GET_PLUGIN(fetcher);
	f->DoVectorComponentRotation(true);

	shared_ptr<info<float>> ret;

	try
	{
		ret = f->Fetch<float>(itsConfiguration, theTime, theLevel, theParam, theType,
		                      itsConfiguration->UseCudaForPacking());

#ifdef HAVE_CUDA
		if (!returnPacked && ret->PackedData()->HasData())
		{
			util::Unpack<float>({ret}, false);

			auto c = GET_PLUGIN(cache);

			c->Insert(ret);
		}
#endif
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error(ClassName() + ": Unable to proceed");
		}
	}

	return ret;
}
