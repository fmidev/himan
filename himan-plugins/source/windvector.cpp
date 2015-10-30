/**
 * @file windvector.cpp
 *
 *  Created on: Jan 21, 2013
 *  @author aaltom
 */

#include "windvector.h"
#include <iostream>
#include <boost/lexical_cast.hpp>
#include "util.h"
#include <math.h>
#include "logger_factory.h"
#include "level.h"
#include "forecast_time.h"
#include "NFmiArea.h"
#include "NFmiRotatedLatLonArea.h"
#include "NFmiStereographicArea.h"
#include <boost/thread.hpp>

using namespace std;
using namespace himan::plugin;

#include "cuda_helper.h"

typedef tuple<double,double,double,double> coefficients;

// std::thread_local is implemented only in g++ 4.8 !

boost::thread_specific_ptr <map<size_t, coefficients>> myCoefficientCache;

windvector::windvector()
	: itsCalculationTarget(kUnknownElement)
	, itsVectorCalculation(false)
{
	itsClearTextFormula = "speed = sqrt(U*U+V*V) ; direction = round(180/PI * atan2(U,V) + offset) ; vector = round(dir/10) + 100 * round(speed)";
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

	double directionOffset = 180; // For wind direction add this

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
			throw runtime_error("Invalid calculation target element: " + boost::lexical_cast<string> (static_cast<int> (itsCalculationTarget)));
			break;
	}
	
	auto myThreadedLogger = logger_factory::Instance()->GetLog("windvectorThread #" + boost::lexical_cast<string> (threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger->Info("Calculating time " + static_cast<string> (forecastTime.ValidDateTime()) +
								" level " + static_cast<string> (forecastLevel));

	info_t UInfo = Fetch(forecastTime, forecastLevel, UParam, forecastType, itsConfiguration->UseCudaForPacking());
	info_t VInfo = Fetch(forecastTime, forecastLevel, VParam, forecastType, itsConfiguration->UseCudaForPacking());

	if (!UInfo || !VInfo)
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	assert(UInfo->Grid()->AB() == VInfo->Grid()->AB());

	for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
	{
		SetAB(myTargetInfo, UInfo);
	}		
				
	// if source producer is Hirlam, we must de-stagger U and V grid
	// edit: Nope, do not de-stagger but interpolate

	/*if (conf->SourceProducer().Id() == 1 && sourceLevel.Type() != kHeight)
	{
		UInfo->Grid()->Stagger(-0.5, 0);
		VInfo->Grid()->Stagger(0, -0.5);
	}*/

	assert(UInfo->Grid()->Projection() == VInfo->Grid()->Projection());

	string deviceType;

#ifdef HAVE_CUDA
	bool needStereographicGridRotation = (UInfo->Grid()->Projection() == kStereographicProjection && UInfo->Grid()->UVRelativeToGrid());

	if (itsConfiguration->UseCuda() && !needStereographicGridRotation)
	{
		deviceType = "GPU";

		assert(UInfo->Grid()->Projection() == kLatLonProjection || UInfo->Grid()->Projection() == kRotatedLatLonProjection);
			
		auto opts = CudaPrepare(myTargetInfo, UInfo, VInfo);

		windvector_cuda::Process(*opts);

	}
	else
#endif
	{
		deviceType = "CPU";

		//UGrid->InterpolationMethod(kNearestPoint);
		//VGrid->InterpolationMethod(kNearestPoint);

		unique_ptr<NFmiArea> sourceArea = ToNewbaseArea(UInfo);
		unique_ptr<NFmiArea> targetArea = ToNewbaseArea(myTargetInfo);

		LOCKSTEP(myTargetInfo, UInfo, VInfo)
		{
			double U = UInfo->Value();
			double V = VInfo->Value();

			if (U == kFloatMissing || V == kFloatMissing)
			{
				continue;
			}

			/*
			 * Speed can be calculated with rotated U and V components
			 */
				
			double speed = sqrt(U*U + V*V);
	
			/*
			 * The order of parameters in infos is and must be always:
			 * index 0 : speed parameter
			 * index 1 : direction parameter (not available for gust)
			 * index 2 : vector parameter (optional)
			 */

			myTargetInfo->ParamIndex(0);
			myTargetInfo->Value(speed);

			if (itsCalculationTarget == kGust)
			{
				continue;
			}

			if (UInfo->Grid()->Projection() == kRotatedLatLonProjection)
			{
				const point regPoint = myTargetInfo->LatLon();

				const NFmiPoint rp = reinterpret_cast<NFmiRotatedLatLonArea*> ((sourceArea.get()))->ToRotLatLon(NFmiPoint(regPoint.X(), regPoint.Y()));
				const point rotPoint(rp.X(), rp.Y());

				// We use UGrid area to do to the rotation even though UGrid area might be
				// different from VGrid area (ie. Hirlam), but that does not matter

				double newU = kFloatMissing, newV = kFloatMissing;

				if (myTargetInfo->Grid()->Projection() == kRotatedLatLonProjection || myTargetInfo->Grid()->Projection() == kLatLonProjection)
				{

					/*
					* 1. Get coordinates of current grid point in earth-relative form
					* 2. Get coordinates of current grid point in grid-relative form
					* 3. Call function UVToEarthRelative() that transforms U and V from grid-relative
					*    to earth-relative
					*/

					coefficients coeffs;

					if (myCoefficientCache->count(myTargetInfo->LocationIndex()))
					{
						coeffs = (*myCoefficientCache)[myTargetInfo->LocationIndex()];
					}
					else
					{
						coeffs = util::EarthRelativeUVCoefficients(regPoint, rotPoint, UInfo->Grid()->SouthPole());
						(*myCoefficientCache)[myTargetInfo->LocationIndex()] = coeffs;
					}

					newU = get<0> (coeffs) * U + get<1> (coeffs) * V;
					newV = get<2> (coeffs) * U + get<3> (coeffs) * V;
						
				}
				else if (myTargetInfo->Grid()->Projection() == kStereographicProjection)
				{
					/*
					 * This modification of the PA,PB,PC,PD coefficients has been
					 * copied from INTROT.F.
					 */

					double cosL, sinL;

					double ang = reinterpret_cast<NFmiStereographicArea*> (targetArea.get())->CentralLongitude();
					double cLon = regPoint.X();
						
					sincos((ang - cLon) * himan::constants::kDeg, &sinL, &cosL);

					coefficients coeffs;

					if (myCoefficientCache->count(myTargetInfo->LocationIndex()))
					{
							coeffs = (*myCoefficientCache)[myTargetInfo->LocationIndex()];
					}
					else
					{
						coeffs = util::EarthRelativeUVCoefficients(regPoint, rotPoint, UInfo->Grid()->SouthPole());
						(*myCoefficientCache)[myTargetInfo->LocationIndex()] = coeffs;
					}

					double PA = get<0> (coeffs) * cosL - get<1> (coeffs) * sinL;
					double PB = get<0> (coeffs) * sinL + get<1> (coeffs) * cosL;
					double PC = get<2> (coeffs) * cosL - get<3> (coeffs) * sinL;
					double PD = get<2> (coeffs) * sinL + get<3> (coeffs) * cosL;
						
					newU = PA * U + PB * V;
					newV = PC * U + PD * V;

				}
				else
				{
					myThreadedLogger->Error("Invalid target projection: " + string(HPProjectionTypeToString.at(myTargetInfo->Grid()->Projection())));
					return;
				}

				// Wind speed should the same with both forms of U and V

				assert(fabs(sqrt(U*U+V*V) - sqrt(newU*newU + newV * newV)) < 0.01);

				U = newU;
				V = newV;

			}
			else if (UInfo->Grid()->Projection() == kStereographicProjection)
			{
				//const point regPoint = myTargetInfo->LatLon();
				throw runtime_error("Rotation of stereographic UV coordinates not confirmed yet");
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
			}

			double dir = 0;

			if (speed > 0)
			{
				dir = himan::constants::kRad * atan2(U,V) + directionOffset;

				// reduce the angle
				dir = fmod(dir,360);
					
				// force it to be the positive remainder, so that 0 <= dir < 360
				dir = fmod((dir + 360), 360);

			}

#ifdef HIL_PP_DD_COMPATIBILITY_MODE
				
			double windVector = round(dir/10) + 100 * round(speed);
			dir = 10 * (static_cast<int> (round(windVector)) % 100);
				
#endif
			myTargetInfo->ParamIndex(1);
			myTargetInfo->Value(round(dir));
				
			if (itsVectorCalculation)
			{

#ifndef HIL_PP_DD_COMPATIBILITY_MODE
				double windVector = round(dir/10) + 100 * round(speed);
#endif

				myTargetInfo->ParamIndex(2);

				myTargetInfo->Value(windVector);
			
			}
		}
	}

	size_t missing = 0, total = 0;

	for (myTargetInfo->ResetParam(); myTargetInfo->NextParam();)
	{
		missing += myTargetInfo->Data().MissingCount();
		total += myTargetInfo->Data().Size();
	}
	
	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missing) + "/" + boost::lexical_cast<string> (total));

}

#ifdef HAVE_CUDA

unique_ptr<windvector_cuda::options> windvector::CudaPrepare(shared_ptr<info> myTargetInfo, shared_ptr<info> UInfo, shared_ptr<info> VInfo)
{
	unique_ptr<windvector_cuda::options> opts(new windvector_cuda::options);

	opts->vector_calculation = itsVectorCalculation;
	opts->need_grid_rotation = (UInfo->Grid()->Projection() == kRotatedLatLonProjection && UInfo->Grid()->UVRelativeToGrid());
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

	opts->N = opts->speed->size_x*opts->speed->size_y;

	return opts;
}

#endif

unique_ptr<NFmiArea> windvector::ToNewbaseArea(shared_ptr<info> myTargetInfo) const
{

	unique_ptr<NFmiArea> theArea;

	// Newbase does not understand grib2 longitude coordinates

	double bottomLeftLongitude = myTargetInfo->Grid()->BottomLeft().X();
	double topRightLongitude = myTargetInfo->Grid()->TopRight().X();

	if (bottomLeftLongitude > 180 || topRightLongitude > 180)
	{
		bottomLeftLongitude -= 180;
		topRightLongitude -= 180;
	}

	switch (myTargetInfo->Grid()->Projection())
	{
		case kLatLonProjection:
		{
			theArea = unique_ptr<NFmiLatLonArea> (new NFmiLatLonArea(NFmiPoint(bottomLeftLongitude, myTargetInfo->Grid()->BottomLeft().Y()),
										 NFmiPoint(topRightLongitude, myTargetInfo->Grid()->TopRight().Y())));

			break;
		}

		case kRotatedLatLonProjection:
		{
			theArea = unique_ptr<NFmiRotatedLatLonArea> (new NFmiRotatedLatLonArea(NFmiPoint(bottomLeftLongitude, myTargetInfo->Grid()->BottomLeft().Y()),
												NFmiPoint(topRightLongitude, myTargetInfo->Grid()->TopRight().Y()),
												NFmiPoint(myTargetInfo->Grid()->SouthPole().X(), myTargetInfo->Grid()->SouthPole().Y()),
												NFmiPoint(0.,0.), // default values
												NFmiPoint(1.,1.), // default values
												true));
			break;
		}

		case kStereographicProjection:
		{
			theArea = unique_ptr<NFmiStereographicArea> (new NFmiStereographicArea(NFmiPoint(bottomLeftLongitude, myTargetInfo->Grid()->BottomLeft().Y()),
												NFmiPoint(topRightLongitude, myTargetInfo->Grid()->TopRight().Y()),
												myTargetInfo->Grid()->Orientation()));
			break;

		}

		default:
			throw runtime_error(ClassName() + ": No supported projection found");
			break;
	}

	assert(theArea);

	return theArea;

}
