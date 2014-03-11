/**
 * @file windvector.cpp
 *
 *  Created on: Jan 21, 2013
 *  @author aaltom
 */

#include "windvector.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"
#include <math.h>
#include "NFmiRotatedLatLonArea.h"
#include "NFmiStereographicArea.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

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

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("windvector"));

}

void windvector::Process(const std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to windvector
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
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
		requestedSpeedParam = param("IFF-MS", 389);
		requestedDirParam = param("IDD-D", 390);

		// GRIB 2

		requestedSpeedParam.GribDiscipline(10);
		requestedSpeedParam.GribCategory(2);
		requestedSpeedParam.GribParameter(3);

		requestedDirParam.GribDiscipline(10);
		requestedDirParam.GribCategory(2);
		requestedDirParam.GribParameter(2);

		itsCalculationTarget = kIce;

		if (itsConfiguration->Exists("do_vector") && itsConfiguration->GetValue("do_vector") == "true")
		{
			itsLogger->Error("Unable to calculate vector for ice");
		}

		theParams.push_back(requestedSpeedParam);
		theParams.push_back(requestedDirParam);
	}
	else if (itsConfiguration->Exists("for_sea") && itsConfiguration->GetValue("for_sea") == "true")
	{
		requestedSpeedParam = param("SFF-MS", 163);
		requestedDirParam = param("SDD-D", 164);

		requestedSpeedParam.GribDiscipline(10);
		requestedSpeedParam.GribCategory(1);
		requestedSpeedParam.GribParameter(1);

		requestedDirParam.GribDiscipline(10);
		requestedDirParam.GribCategory(1);
		requestedDirParam.GribParameter(0);

		itsCalculationTarget = kSea;

		if (itsConfiguration->Exists("do_vector") && itsConfiguration->GetValue("do_vector") == "true")
		{
			itsLogger->Error("Unable to calculate vector for sea");
		}

		theParams.push_back(requestedSpeedParam);
		theParams.push_back(requestedDirParam);
	}
	else if (itsConfiguration->Exists("for_gust") && itsConfiguration->GetValue("for_gust") == "true")
	{
		requestedSpeedParam = param("FFG-MS", 417);
		
		requestedSpeedParam.GribDiscipline(0);
		requestedSpeedParam.GribCategory(2);
		requestedSpeedParam.GribParameter(22);

		itsCalculationTarget = kGust;

		if (itsConfiguration->Exists("do_vector") && itsConfiguration->GetValue("do_vector") == "true")
		{
			itsLogger->Error("Unable to calculate vector for wind gust");
		}

		theParams.push_back(requestedSpeedParam);

	}
	else
	{
		// By default assume we'll calculate for wind

		requestedDirParam = param("DD-D", 20);
		requestedDirParam.GribDiscipline(0);
		requestedDirParam.GribCategory(2);
		requestedDirParam.GribParameter(0);

		requestedSpeedParam = param("FF-MS", 21);
		requestedSpeedParam.GribDiscipline(0);
		requestedSpeedParam.GribCategory(2);
		requestedSpeedParam.GribParameter(1);

		theParams.push_back(requestedSpeedParam);
		theParams.push_back(requestedDirParam);

		if (itsVectorCalculation)
		{
			requestedVectorParam = param("DF-MS", 22);
			theParams.push_back(requestedVectorParam);
		}
		
		itsCalculationTarget = kWind;
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

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

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
	
	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("windvectorThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->ParamIndex(0);

	bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(itsConfiguration, threadIndex);

	if (useCudaInThisThread)
	{
		myThreadedLogger->Debug("Will use Cuda");
	}
	
	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> UInfo;
		shared_ptr<info> VInfo;

		try
		{
			// Source info for U
			UInfo = theFetcher->Fetch(itsConfiguration,
										myTargetInfo->Time(),
										myTargetInfo->Level(),
										UParam,
										itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
				
			// Source info for V
			VInfo = theFetcher->Fetch(itsConfiguration,
										myTargetInfo->Time(),
										myTargetInfo->Level(),
										VParam,
										itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
				
		}
		catch (HPExceptionType& e)
		{
		
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value

					if (itsConfiguration->StatisticsEnabled())
					{
						itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
						itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
					}

					continue;
					break;

				default:
					throw runtime_error(ClassName() + ": Unable to proceed");
					break;
				}
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

		size_t missingCount = 0;
		size_t count = 0;

		bool equalGrids = (*myTargetInfo->Grid() == *UInfo->Grid() && *myTargetInfo->Grid() == *VInfo->Grid());

		assert(UInfo->Grid()->Projection() == VInfo->Grid()->Projection());

		string deviceType;

#ifdef HAVE_CUDA
		bool needStereographicGridRotation = (UInfo->Grid()->Projection() == kStereographicProjection && UInfo->Grid()->UVRelativeToGrid());

		// If we read packed data but grids are not equal we cannot use cuda
		// for calculations (our cuda routines do not know how to interpolate)

		if (!equalGrids && (UInfo->Grid()->IsPackedData() || VInfo->Grid()->IsPackedData()))
		{
			myThreadedLogger->Debug("Unpacking for CPU calculation");
			Unpack({UInfo, VInfo});
		}

		if (useCudaInThisThread && equalGrids && !needStereographicGridRotation)
		{
			deviceType = "GPU";

			assert(UInfo->Grid()->Projection() == kLatLonProjection || UInfo->Grid()->Projection() == kRotatedLatLonProjection);
			
			auto opts = CudaPrepare(myTargetInfo, UInfo, VInfo);

			windvector_cuda::Process(*opts);

			count = opts->N;
			missingCount = opts->missing;

			CudaFinish(move(opts), myTargetInfo, UInfo, VInfo);

		}
		else
#endif
		{
			deviceType = "CPU";

			shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

			shared_ptr<NFmiGrid> UGrid(UInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> VGrid(VInfo->Grid()->ToNewbaseGrid());

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();
			//UGrid->InterpolationMethod(kNearestPoint);
			//VGrid->InterpolationMethod(kNearestPoint);

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{
				count++;

				double U = kFloatMissing;
				double V = kFloatMissing;

				InterpolateToPoint(targetGrid, UGrid, equalGrids, U);
				InterpolateToPoint(targetGrid, VGrid, equalGrids, V);

				if (U == kFloatMissing || V == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->ParamIndex(0);
					myTargetInfo->Value(kFloatMissing);

					if (itsCalculationTarget != kGust)
					{
						myTargetInfo->ParamIndex(1);
						myTargetInfo->Value(kFloatMissing);
					}

					if (itsVectorCalculation)
					{
						myTargetInfo->ParamIndex(2);
						myTargetInfo->Value(kFloatMissing);
					}

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
				
				if (UGrid->Area()->ClassId() == kNFmiRotatedLatLonArea)
				{
					const point regPoint(targetGrid->LatLon());

					// We use UGrid area to do to the rotation even though UGrid area might be
					// different from VGrid area (ie. Hirlam), but that does not matter
					
					const point rotPoint(reinterpret_cast<NFmiRotatedLatLonArea*> (UGrid->Area())->ToRotLatLon(regPoint.ToNFmiPoint()));

					double newU = kFloatMissing, newV = kFloatMissing;

					if (targetGrid->Area()->ClassId() == kNFmiRotatedLatLonArea || targetGrid->Area()->ClassId() == kNFmiLatLonArea)
					{

						/*
						* 1. Get coordinates of current grid point in earth-relative form
						* 2. Get coordinates of current grid point in grid-relative form
						* 3. Call function UVToEarthRelative() that transforms U and V from grid-relative
						*    to earth-relative
						*
						* NFmiRotatedLatLonArea will give the normal latlon coordinates with LatLon()
						* function, so we need force the regular point to rotated point with ToRotLatLon().
						*
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
					else if (targetGrid->Area()->ClassId() == kNFmiStereographicArea)
					{
						/*
						 * This modification of the PA,PB,PC,PD coefficients has been
						 * copied from INTROT.F.
						 */

						double cosL, sinL;

						double ang = reinterpret_cast<NFmiStereographicArea*> (targetGrid->Area())->CentralLongitude();
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

					assert(fabs(sqrt(U*U+V*V) - sqrt(newU*newU + newV * newV)) < 0.001);

					U = newU;
					V = newV;

				}
				else if (UGrid->Area()->ClassId() != kNFmiLatLonArea)
				{
					myThreadedLogger->Error("Invalid source projection: " + string(HPProjectionTypeToString.at(UInfo->Grid()->Projection())));
					return;
					
					/*
					 * This code should work in theory but it's not enabled because it is not tested.

					assert(UGrid->Area()->ClassId() == kNFmiStereographicArea);

					double centralLongitude = (reinterpret_cast<NFmiStereographicArea*> (targetGrid->Area())->CentralLongitude());

					point regUV = util::UVToGeographical(centralLongitude, point(U,V));

					// Wind speed should the same with both forms of U and V

					assert(fabs(sqrt(U*U+V*V) - sqrt(regUV.X()*regUV.X() + regUV.Y() * regUV.Y())) < 0.001);

					U = regUV.X();
					V = regUV.Y();
					*/
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

					if (!myTargetInfo->Value(windVector))
					{
						throw runtime_error(ClassName() + ": Failed to set value to matrix");
					}
				}
			}

			for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
			{
				SwapTo(myTargetInfo, kBottomLeft);
			}
		}
		
		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(missingCount);
			itsConfiguration->Statistics()->AddToValueCount(count);
		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));


		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}

#ifdef HAVE_CUDA

unique_ptr<windvector_cuda::options> windvector::CudaPrepare(shared_ptr<info> myTargetInfo, shared_ptr<info> UInfo, shared_ptr<info> VInfo)
{
	unique_ptr<windvector_cuda::options> opts(new windvector_cuda::options);

	opts->vector_calculation = itsVectorCalculation;
	opts->need_grid_rotation = (UInfo->Grid()->Projection() == kRotatedLatLonProjection && UInfo->Grid()->UVRelativeToGrid());;
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

void windvector::CudaFinish(unique_ptr<windvector_cuda::options> opts, shared_ptr<info> myTargetInfo, shared_ptr<info> UInfo, shared_ptr<info> VInfo)
{
	// Copy data back to infos

	myTargetInfo->ParamIndex(0);
	CopyDataFromSimpleInfo(myTargetInfo, opts->speed, false);
	
	if (itsCalculationTarget != kGust)
	{
		myTargetInfo->ParamIndex(1);
		CopyDataFromSimpleInfo(myTargetInfo, opts->dir, false);
	}

	if (itsVectorCalculation)
	{
		myTargetInfo->ParamIndex(2);
		CopyDataFromSimpleInfo(myTargetInfo, opts->vector, false);
	}

	assert(UInfo->Grid()->ScanningMode() == VInfo->Grid()->ScanningMode());

	for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
	{
		SwapTo(myTargetInfo, UInfo->Grid()->ScanningMode());
	}

	// Copy unpacked data to source info in case
	// some other thread/plugin calls for this same data.
	// Clear packed data now that it's been unpacked

	if (UInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(UInfo, opts->u, true);
	}

	if (VInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(VInfo, opts->v, true);
	}

	// opts is destroyed after leaving this function
}

#endif