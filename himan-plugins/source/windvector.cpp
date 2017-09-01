/**
 * @file windvector.cpp
 */

#include "windvector.h"
#include "forecast_time.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "stereographic_grid.h"
#include "util.h"
#include <boost/thread.hpp>
#include <iostream>
#include <math.h>

#include "cache.h"
#include "fetcher.h"

using namespace std;
using namespace himan::plugin;

#include "cuda_helper.h"

typedef tuple<double, double, double, double> coefficients;

boost::thread_specific_ptr<map<size_t, coefficients>> myCoefficientCache;

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

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void windvector::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	if (!myCoefficientCache.get())
	{
		myCoefficientCache.reset(new map<size_t, coefficients>());
	}

	// Required source parameters

	param UParam;
	param VParam;

	double directionOffset = 180;  // For wind direction add this

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

	info_t UInfo = Fetch(forecastTime, forecastLevel, UParam, forecastType, itsConfiguration->UseCudaForPacking());
	info_t VInfo = Fetch(forecastTime, forecastLevel, VParam, forecastType, itsConfiguration->UseCudaForPacking());

	if (!UInfo || !VInfo)
	{
		myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                         static_cast<string>(forecastLevel));
		return;
	}

	assert(UInfo->Grid()->AB() == VInfo->Grid()->AB());

	for (myTargetInfo->ResetParam(); myTargetInfo->NextParam();)
	{
		SetAB(myTargetInfo, UInfo);
	}

	assert(UInfo->Grid()->Type() == VInfo->Grid()->Type());

	string deviceType;

#ifdef HAVE_CUDA
	if (itsConfiguration->UseCuda() &&
	    (UInfo->Grid()->Type() == kLatitudeLongitude || UInfo->Grid()->Type() == kRotatedLatitudeLongitude ||
	     UInfo->Grid()->Type() == kLambertConformalConic))
	{
		deviceType = "GPU";

		auto opts = CudaPrepare(myTargetInfo, UInfo, VInfo);

		windvector_cuda::Process(*opts);
	}
	else
#endif
	{
		deviceType = "CPU";

#ifdef HAVE_CUDA
		Unpack({UInfo, VInfo});
#endif
		myTargetInfo->ParamIndex(0);

		auto& FFVec = VEC(myTargetInfo);
		vector<double> DDVec(FFVec.size(), MissingDouble());

		for (auto&& tup : zip_range(FFVec, DDVec, VEC(UInfo), VEC(VInfo)))
		{
			double& speed = tup.get<0>();
			double& dir = tup.get<1>();
			double U = tup.get<2>();
			double V = tup.get<3>();

			if (IsMissingValue({U, V}))
			{
				continue;
			}

			speed = sqrt(U * U + V * V);

			if (itsCalculationTarget == kGust)
			{
				continue;
			}

			dir = himan::constants::kRad * atan2(U, V) + directionOffset;

			// reduce the angle
			dir = fmod(dir, 360);

			// force it to be the positive remainder, so that 0 <= dir < 360
			dir = round(fmod((dir + 360), 360));
		}

		if (myTargetInfo->SizeParams() > 1)
		{
			myTargetInfo->ParamIndex(1);
			myTargetInfo->Data().Set(DDVec);
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

shared_ptr<himan::info> windvector::Fetch(const forecast_time& theTime, const level& theLevel, const param& theParam,
                                          const forecast_type& theType, bool returnPacked) const
{
	auto f = GET_PLUGIN(fetcher);
	f->DoVectorComponentRotation(true);

	info_t ret;

	try
	{
		ret = f->Fetch(itsConfiguration, theTime, theLevel, theParam, theType, itsConfiguration->UseCudaForPacking());

#ifdef HAVE_CUDA
		if (!returnPacked && ret->Grid()->IsPackedData())
		{
			assert(ret->Grid()->PackedData().ClassName() == "simple_packed");

			util::Unpack({ret->Grid()});

			auto c = GET_PLUGIN(cache);

			c->Insert(*ret);
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

#ifdef HAVE_CUDA

unique_ptr<windvector_cuda::options> windvector::CudaPrepare(shared_ptr<info> myTargetInfo, shared_ptr<info> UInfo,
                                                             shared_ptr<info> VInfo)
{
	unique_ptr<windvector_cuda::options> opts(new windvector_cuda::options);

	opts->N = UInfo->Grid()->Size();

	opts->target_type = itsCalculationTarget;

	opts->u = UInfo->ToSimple();
	opts->v = VInfo->ToSimple();

	myTargetInfo->ParamIndex(0);
	opts->speed = myTargetInfo->ToSimple();

	if (opts->target_type != kGust)
	{
		myTargetInfo->ParamIndex(1);
		opts->dir = myTargetInfo->ToSimple();
	}

	return opts;
}

#endif
