/**
 * @file windvector.cpp
 */

#include "windvector.h"
#include "NFmiRotatedLatLonArea.h"
#include "NFmiStereographicArea.h"
#include "forecast_time.h"
#include "latitude_longitude_grid.h"
#include "level.h"
#include "logger_factory.h"
#include "plugin_factory.h"
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
	    (UInfo->Grid()->Type() == kLatitudeLongitude || UInfo->Grid()->Type() == kRotatedLatitudeLongitude))
	{
		deviceType = "GPU";

		auto opts = CudaPrepare(myTargetInfo, UInfo, VInfo);

		windvector_cuda::Process(*opts);
	}
	else
#endif
	{
		deviceType = "CPU";

		unique_ptr<NFmiArea> sourceArea = ToNewbaseArea(UInfo);
		unique_ptr<NFmiArea> targetArea = ToNewbaseArea(myTargetInfo);

		// 1. Rotate to earth normal

		auto& UVec = VEC(UInfo);
		auto& VVec = VEC(VInfo);

		switch (UInfo->Grid()->Type())
		{
			case kRotatedLatitudeLongitude:
			{
				const NFmiRotatedLatLonArea* rotArea = dynamic_cast<NFmiRotatedLatLonArea*>(sourceArea.get());
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

					const point regPoint = UInfo->LatLon();
					const NFmiPoint rp = rotArea->ToRotLatLon(NFmiPoint(regPoint.X(), regPoint.Y()));
					const point rotPoint(rp.X(), rp.Y());

					auto coeffs = util::EarthRelativeUVCoefficients(regPoint, rotPoint, southPole);

					U = get<0>(coeffs) * U + get<1>(coeffs) * V;
					V = get<2>(coeffs) * U + get<3>(coeffs) * V;
				}
			}
			break;

			case kLambertConformalConic:
			{
			}
			break;

			default:
				break;
		}

		// 2. Rotate to target projection normal, if needed

		switch (myTargetInfo->Grid()->Type())
		{
			default:
				break;

			case kStereographic:
			{
				/*
				 * This modification of the PA,PB,PC,PD coefficients has been
				 * copied from INTROT.F.
				 */
				const NFmiRotatedLatLonArea* rotArea = dynamic_cast<NFmiRotatedLatLonArea*>(sourceArea.get());
				const point southPole = dynamic_cast<rotated_latitude_longitude_grid*>(UInfo->Grid())->SouthPole();

				const double ang = reinterpret_cast<NFmiStereographicArea*>(targetArea.get())->CentralLongitude();

				for (UInfo->ResetLocation(); UInfo->NextLocation();)
				{
					size_t i = UInfo->LocationIndex();

					double& U = UVec[i];
					double& V = VVec[i];

					const point regPoint = UInfo->LatLon();
					const NFmiPoint rp = rotArea->ToRotLatLon(NFmiPoint(regPoint.X(), regPoint.Y()));
					const point rotPoint(rp.X(), rp.Y());

					const double cLon = regPoint.X();

					double cosL, sinL;
					sincos((ang - cLon) * himan::constants::kDeg, &sinL, &cosL);

					auto coeffs = util::EarthRelativeUVCoefficients(
					    regPoint, rotPoint, dynamic_cast<rotated_latitude_longitude_grid*>(UInfo->Grid())->SouthPole());

					double PA = get<0>(coeffs) * cosL - get<1>(coeffs) * sinL;
					double PB = get<0>(coeffs) * sinL + get<1>(coeffs) * cosL;
					double PC = get<2>(coeffs) * cosL - get<3>(coeffs) * sinL;
					double PD = get<2>(coeffs) * sinL + get<3>(coeffs) * cosL;

					U = PA * U + PB * V;
					V = PC * U + PD * V;
				}
			}
			break;
		}

		// 3. Calculate speed and direction
		std::cout << UVec[0] << "\n";
		myTargetInfo->ParamIndex(0);

		auto& FFVec = VEC(myTargetInfo);
		vector<double> _DDVec(FFVec.size(), kFloatMissing);

		if (myTargetInfo->SizeParams() > 1)
		{
			myTargetInfo->ParamIndex(1);
			_DDVec = VEC(myTargetInfo);
		}

		auto& DDVec = _DDVec;

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
	}

// const point regPoint = myTargetInfo->LatLon();
//			throw runtime_error("Rotation of stereographic UV coordinates not confirmed yet");
#if 0
				double j;

				if (myTargetInfo->Grid()->ScanningMode() == kBottomLeft) //opts.j_scans_positive)
				{
					j = floor(static_cast<double> (myTargetInfo->LocationIndex() / myTargetInfo->Data().SizeX()));
				}
				else if (myTargetInfo->Grid()->ScanningMode() == kTopLeft)
				{
					j = static_cast<double> (myTargetInfo->Grid()->Nj()) - floor(static_cast<double> (myTargetInfo->LocationIndex()) / static_cast<double> (myTargetInfo->Data().SizeX()));
				}
				else
				{
					throw runtime_error("Unsupported projection: " + string(HPScanningModeToString.at(myTargetInfo->Grid()->ScanningMode())));
				}

				double i = static_cast<double> (myTargetInfo->LocationIndex()) - j * static_cast<double> (myTargetInfo->Grid()->Ni());

				i /= static_cast<double> (myTargetInfo->Data().SizeX());
				j /= static_cast<double> (myTargetInfo->Data().SizeY());

				NFmiPoint ll = reinterpret_cast<NFmiStereographicArea*> (targetArea.get())->ToLatLon(NFmiPoint(i,j));
				point regPoint(ll.X(), ll.Y());
				
				double angle = 180. + reinterpret_cast<NFmiStereographicArea*> (targetArea.get())->CentralLongitude();
				double lon = regPoint.X();

				lon = lon - (angle - 180.);

				point regUV = util::UVToGeographical(lon, point(U,V));

				// Wind speed should the same with both forms of U and V

				assert(fabs(sqrt(U*U+V*V) - sqrt(regUV.X()*regUV.X() + regUV.Y() * regUV.Y())) < 0.001);
					
				U = regUV.X();
				V = regUV.Y();
#endif

	size_t missing = 0, total = 0;

	for (myTargetInfo->ResetParam(); myTargetInfo->NextParam();)
	{
		missing += myTargetInfo->Data().MissingCount();
		total += myTargetInfo->Data().Size();
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string>(missing) + "/" +
	                       boost::lexical_cast<string>(total));
}

#ifdef HAVE_CUDA

unique_ptr<windvector_cuda::options> windvector::CudaPrepare(shared_ptr<info> myTargetInfo, shared_ptr<info> UInfo,
                                                             shared_ptr<info> VInfo)
{
	unique_ptr<windvector_cuda::options> opts(new windvector_cuda::options);

	opts->vector_calculation = false;
	opts->need_grid_rotation = false;

	if (UInfo->Grid()->Type() == kRotatedLatitudeLongitude)
	{
		opts->need_grid_rotation = dynamic_cast<rotated_latitude_longitude_grid*>(UInfo->Grid())->UVRelativeToGrid();
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

	if (opts->vector_calculation)
	{
		myTargetInfo->ParamIndex(2);
		opts->vector = myTargetInfo->ToSimple();
	}

	opts->N = opts->speed->size_x * opts->speed->size_y;

	return opts;
}

#endif

unique_ptr<NFmiArea> windvector::ToNewbaseArea(shared_ptr<info> myTargetInfo) const
{
	auto q = GET_PLUGIN(querydata);

	auto hdesc = q->CreateHPlaceDescriptor(*myTargetInfo, true);

	unique_ptr<NFmiArea> theArea = unique_ptr<NFmiArea>(hdesc.Area()->Clone());

	assert(theArea);

	return theArea;
}
