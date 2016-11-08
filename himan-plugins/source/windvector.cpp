/**
 * @file windvector.cpp
 */

#include "windvector.h"
#include "forecast_time.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "level.h"
#include "logger_factory.h"
#include "plugin_factory.h"
#include "stereographic_grid.h"
#include "util.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <iostream>
#include <math.h>

#include "querydata.h"

using namespace std;
using namespace himan::plugin;

#include "cuda_helper.h"

typedef tuple<double, double, double, double> coefficients;

boost::thread_specific_ptr<map<size_t, coefficients>> myCoefficientCache;

windvector::windvector() : itsCalculationTarget(kUnknownElement), itsVectorCalculation(false)
{
	itsClearTextFormula =
	    "speed = sqrt(U*U+V*V) ; direction = round(180/PI * atan2(U,V) + offset) ; vector = round(dir/10) + 100 * "
	    "round(speed)";
	itsCudaEnabledCalculation = true;

	itsLogger = logger_factory::Instance()->GetLog("windvector");
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
			itsLogger->Warning("Unable to calculate vector for ice");
		}
	}
	else if (itsConfiguration->Exists("for_sea") && itsConfiguration->GetValue("for_sea") == "true")
	{
		requestedSpeedParam = param("SFF-MS", 163, 10, 1, 1);
		requestedDirParam = param("SDD-D", 164, 10, 1, 0);

		itsCalculationTarget = kSea;

		if (itsVectorCalculation)
		{
			itsLogger->Warning("Unable to calculate vector for sea");
		}
	}
	else if (itsConfiguration->Exists("for_gust") && itsConfiguration->GetValue("for_gust") == "true")
	{
		requestedSpeedParam = param("FFG-MS", 417, 0, 2, 22);

		itsCalculationTarget = kGust;

		if (itsVectorCalculation)
		{
			itsLogger->Warning("Unable to calculate vector for wind gust");
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
			                    boost::lexical_cast<string>(static_cast<int>(itsCalculationTarget)));
			break;
	}

	auto myThreadedLogger =
	    logger_factory::Instance()->GetLog("windvectorThread #" + boost::lexical_cast<string>(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	info_t UInfo = Fetch(forecastTime, forecastLevel, UParam, forecastType, itsConfiguration->UseCudaForPacking());
	info_t VInfo = Fetch(forecastTime, forecastLevel, VParam, forecastType, itsConfiguration->UseCudaForPacking());

	if (!UInfo || !VInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string>(forecastTime.Step()) + ", level " +
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

		delete[] opts->lon;
		delete[] opts->lat;
	}
	else
#endif
	{
		deviceType = "CPU";

		Unpack({UInfo, VInfo});

		// Rotate to earth normal if projected

		auto& UVec = VEC(UInfo);
		auto& VVec = VEC(VInfo);

		switch (UInfo->Grid()->Type())
		{
			case kRotatedLatitudeLongitude:
			{
				const rotated_latitude_longitude_grid* rotArea =
				    dynamic_cast<rotated_latitude_longitude_grid*>(UInfo->Grid());
				const point southPole = dynamic_cast<rotated_latitude_longitude_grid*>(UInfo->Grid())->SouthPole();

				for (UInfo->ResetLocation(); UInfo->NextLocation();)
				{
					size_t i = UInfo->LocationIndex();

					double& U = UVec[i];
					double& V = VVec[i];

					if (U == kFloatMissing || V == kFloatMissing)
					{
						continue;
					}

					const point rotPoint = rotArea->RotatedLatLon(i);
					const point regPoint = rotArea->LatLon(i);

					auto coeffs = util::EarthRelativeUVCoefficients(regPoint, rotPoint, southPole);

					double newU = get<0>(coeffs) * U + get<1>(coeffs) * V;
					double newV = get<2>(coeffs) * U + get<3>(coeffs) * V;
					U = newU;
					V = newV;
				}
			}
			break;

			case kLambertConformalConic:
			{
				const lambert_conformal_grid* lcc = dynamic_cast<lambert_conformal_grid*>(UInfo->Grid());
				const double latin1 = lcc->StandardParallel1();
				const double latin2 = lcc->StandardParallel2();

				double cone;
				if (latin1 == latin2)
				{
					cone = sin(latin1 * constants::kDeg);
				}
				else
				{
					cone = (log(cos(latin1 * constants::kDeg)) - log(cos(latin2 * constants::kDeg))) /
					       (log(tan((90 - fabs(latin1)) * constants::kDeg * 0.5)) -
					        log(tan(90 - fabs(latin2)) * constants::kDeg * 0.5));
				}

				const double orientation = lcc->Orientation();

				for (UInfo->ResetLocation(); UInfo->NextLocation();)
				{
					size_t i = UInfo->LocationIndex();

					double U = UVec[i];
					double V = VVec[i];

					// http://www.mcs.anl.gov/~emconsta/wind_conversion.txt

					double londiff = UInfo->LatLon().X() - orientation;
					assert(londiff >= -180 && londiff <= 180);
					assert(UInfo->LatLon().Y() >= 0);

					const double angle = cone * londiff * constants::kDeg;
					double sinx, cosx;
					sincos(angle, &sinx, &cosx);

					// This output is 1:1 with python basemaplib, but is different
					// than the results received using algorithm from meteorological software.
					// (below)

					// double newU = cosx * U - sinx * V;
					// double newV = sinx * U + cosx * V;

					UVec[i] = -1 * cosx * U + sinx * V;
					VVec[i] = -1 * -sinx * U + cosx * V;
				}
			}
			break;

			case kStereographic:
			{
				// The same as lambert but with cone = 1

				const stereographic_grid* sc = dynamic_cast<stereographic_grid*>(UInfo->Grid());
				const double orientation = sc->Orientation();

				for (UInfo->ResetLocation(); UInfo->NextLocation();)
				{
					size_t i = UInfo->LocationIndex();

					double U = UVec[i];
					double V = VVec[i];

					const double angle = (UInfo->LatLon().X() - orientation) * constants::kDeg;
					double sinx, cosx;

					sincos(angle, &sinx, &cosx);

					UVec[i] = -1 * cosx * U + sinx * V;
					VVec[i] = -1 * -sinx * U + cosx * V;
				}
			}
			break;
			default:
				break;
		}

		myTargetInfo->ParamIndex(0);

		auto& FFVec = VEC(myTargetInfo);
		vector<double> DDVec(FFVec.size(), kFloatMissing);

		for (auto&& tup : zip_range(FFVec, DDVec, UVec, VVec))
		{
			double& speed = tup.get<0>();
			double& dir = tup.get<1>();
			double U = tup.get<2>();
			double V = tup.get<3>();

			if (U == kFloatMissing || V == kFloatMissing)
			{
				continue;
			}

			speed = sqrt(U * U + V * V);

			if (itsCalculationTarget == kGust)
			{
				continue;
			}

			if (speed > 0)
			{
				dir = himan::constants::kRad * atan2(U, V) + directionOffset;

				// reduce the angle
				dir = fmod(dir, 360);

				// force it to be the positive remainder, so that 0 <= dir < 360
				dir = round(fmod((dir + 360), 360));
			}
		}

		if (myTargetInfo->SizeParams() > 1)
		{
			myTargetInfo->ParamIndex(1);
			myTargetInfo->Data().Set(DDVec);
		}
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " +
	                       boost::lexical_cast<string>(myTargetInfo->Data().MissingCount()) + "/" +
	                       boost::lexical_cast<string>(myTargetInfo->Data().Size()));
}

#ifdef HAVE_CUDA

unique_ptr<windvector_cuda::options> windvector::CudaPrepare(shared_ptr<info> myTargetInfo, shared_ptr<info> UInfo,
                                                             shared_ptr<info> VInfo)
{
	unique_ptr<windvector_cuda::options> opts(new windvector_cuda::options);

	opts->need_grid_rotation = false;
	opts->N = UInfo->Grid()->Size();

	if (UInfo->Grid()->Type() == kRotatedLatitudeLongitude)
	{
		opts->need_grid_rotation = dynamic_cast<rotated_latitude_longitude_grid*>(UInfo->Grid())->UVRelativeToGrid();
	}
	else if (UInfo->Grid()->Type() == kLambertConformalConic)
	{
		opts->need_grid_rotation = dynamic_cast<lambert_conformal_grid*>(UInfo->Grid())->UVRelativeToGrid();

		opts->lon = new double[opts->N];
		opts->lat = new double[opts->N];

		for (UInfo->ResetLocation(); UInfo->NextLocation();)
		{
			auto ll = UInfo->LatLon();
			opts->lon[UInfo->LocationIndex()] = ll.X();
			opts->lat[UInfo->LocationIndex()] = ll.Y();
		}
	}

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
