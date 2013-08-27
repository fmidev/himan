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

#include "windvector_cuda.h"
#include "cuda_helper.h"

const double kRadToDeg = 57.29577951307855; // 180 / PI

windvector::windvector()
	: itsCalculationTarget(kUnknownElement)
	, itsVectorCalculation(false)
{
	itsClearTextFormula = "speed = sqrt(U*U+V*V) ; direction = round(180/PI * atan2(U,V) + offset) ; vector = round(dir/10) + 100 * round(speed)";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("windvector"));

}

void windvector::Process(const std::shared_ptr<const plugin_configuration> conf)
{
	unique_ptr<timer> aTimer;

	// Get number of threads to use

	unsigned short threadCount = ThreadCount(conf->ThreadCount());

	if (conf->StatisticsEnabled())
	{
		aTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
		aTimer->Start();
		conf->Statistics()->UsedThreadCount(threadCount);
		conf->Statistics()->UsedGPUCount(conf->CudaDeviceCount());
	}

	boost::thread_group g;

	shared_ptr<info> targetInfo = conf->Info();

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

	if (conf->Exists("do_vector") && conf->GetValue("do_vector") == "true")
	{
		itsVectorCalculation = true;
	}

	if (conf->Exists("for_ice") && conf->GetValue("for_ice") == "true")
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

		if (conf->Exists("do_vector") && conf->GetValue("do_vector") == "true")
		{
			itsLogger->Error("Unable to calculate vector for ice");
		}

		theParams.push_back(requestedSpeedParam);
		theParams.push_back(requestedDirParam);
	}
	else if (conf->Exists("for_sea") && conf->GetValue("for_sea") == "true")
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

		if (conf->Exists("do_vector") && conf->GetValue("do_vector") == "true")
		{
			itsLogger->Error("Unable to calculate vector for sea");
		}

		theParams.push_back(requestedSpeedParam);
		theParams.push_back(requestedDirParam);
	}
	else if (conf->Exists("for_gust") && conf->GetValue("for_gust") == "true")
	{
		requestedSpeedParam = param("FFG-MS", 417);
		
		requestedSpeedParam.GribDiscipline(0);
		requestedSpeedParam.GribCategory(2);
		requestedSpeedParam.GribParameter(22);

		itsCalculationTarget = kGust;

		if (conf->Exists("do_vector") && conf->GetValue("do_vector") == "true")
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

	if (conf->OutputFileType() == kGRIB1)
	{
		StoreGrib1ParameterDefinitions(theParams, targetInfo->Producer().TableVersion());
	}

	targetInfo->Params(theParams);

	/*
	 * Create data structures.
	 */

	targetInfo->Create();

	/*
	 * Initialize parent class functions for dimension handling
	 */

	Dimension(conf->LeadingDimension());
	FeederInfo(shared_ptr<info> (new info(*targetInfo)));
	FeederInfo()->ParamIndex(0); // Set index to first param (it doesn't matter which one, as long as its set

	if (conf->StatisticsEnabled())
	{
		aTimer->Stop();
		conf->Statistics()->AddToInitTime(aTimer->GetTime());

		aTimer->Start();

	}

	/*
	 * Each thread will have a copy of the target info.
	 */

	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		boost::thread* t = new boost::thread(&windvector::Run,
								this,
								shared_ptr<info> (new info(*targetInfo)),
								conf,
								i + 1);

		g.add_thread(t);

	}

	g.join_all();

	if (conf->StatisticsEnabled())
	{
		aTimer->Stop();
		conf->Statistics()->AddToProcessingTime(aTimer->GetTime());
	}

	WriteToFile(conf, targetInfo);

}

void windvector::Run(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
{
	while (AdjustLeadingDimension(myTargetInfo))
	{
		Calculate(myTargetInfo, conf, threadIndex);
	}
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void windvector::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

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

	// Fetch source level definition

	level sourceLevel = compiled_plugin_base::LevelTransform(conf->SourceProducer(), UParam, myTargetInfo->PeakLevel(0));

	bool useCudaInThisThread = conf->UseCuda() && threadIndex <= conf->CudaDeviceCount();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> UInfo;
		shared_ptr<info> VInfo;

		try
		{
			// Source info for U
			UInfo = theFetcher->Fetch(conf,
										myTargetInfo->Time(),
										sourceLevel,
										UParam,
										conf->UseCudaForPacking() && useCudaInThisThread);
				
			// Source info for V
			VInfo = theFetcher->Fetch(conf,
										myTargetInfo->Time(),
										sourceLevel,
										VParam,
										conf->UseCudaForPacking() && useCudaInThisThread);
				
		}
		catch (HPExceptionType e)
		{
		
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value

					if (conf->StatisticsEnabled())
					{
						conf->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
						conf->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
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

		/*if (conf->SourceProducer().Id() == 1 && sourceLevel.Type() != kHeight)
		{
			UInfo->Grid()->Stagger(-0.5, 0);
			VInfo->Grid()->Stagger(0, -0.5);
		}*/

		size_t missingCount = 0;
		size_t count = 0;

		bool equalGrids = (*myTargetInfo->Grid() == *UInfo->Grid() && *myTargetInfo->Grid() == *VInfo->Grid());

		assert(UInfo->Grid()->Projection() == VInfo->Grid()->Projection());

		bool needRotLatLonGridRotation = (UInfo->Grid()->Projection() == kRotatedLatLonProjection && UInfo->Grid()->UVRelativeToGrid());
		bool needStereographicGridRotation = (UInfo->Grid()->Projection() == kStereographicProjection && UInfo->Grid()->UVRelativeToGrid());

		string deviceType;

#ifdef HAVE_CUDA
		if (useCudaInThisThread && equalGrids && !needStereographicGridRotation)
		{
			deviceType = "GPU";

			assert(UInfo->Grid()->Projection() == kLatLonProjection || UInfo->Grid()->Projection() == kRotatedLatLonProjection);
			
			windvector_cuda::windvector_cuda_options opts;
			windvector_cuda::windvector_cuda_data datas;

			if (myTargetInfo->Grid()->ScanningMode() == kTopLeft)
			{
				opts.jScansPositive = false;
			}
			else if (myTargetInfo->Grid()->ScanningMode() != kBottomLeft)
			{
				throw runtime_error(ClassName() + ": Invalid scanning mode for Cuda: " + string(HPScanningModeToString.at(myTargetInfo->Grid()->ScanningMode())));
			}

			opts.sizeX = myTargetInfo->Grid()->Data()->SizeX();
			opts.sizeY = myTargetInfo->Grid()->Data()->SizeY();

			size_t N = opts.sizeX*opts.sizeY;

			if (UInfo->Grid()->DataIsPacked())
			{
				assert(UInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> u = dynamic_pointer_cast<simple_packed> (UInfo->Grid()->PackedData());

				datas.pU = *(u);

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.u), N * sizeof(double), cudaHostAllocMapped));

				opts.pU = true;

			}
			else
			{
				datas.u = const_cast<double*> (UInfo->Data()->Values());
			}

			if (VInfo->Grid()->DataIsPacked())
			{
				assert(VInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> v = dynamic_pointer_cast<simple_packed> (VInfo->Grid()->PackedData());

				datas.pV = *(v);

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.v), N * sizeof(double), cudaHostAllocMapped));

				opts.pV = true;

			}
			else
			{
				datas.v = const_cast<double*> (VInfo->Data()->Values());
			}

			opts.vectorCalculation = itsVectorCalculation;
			opts.needRotLatLonGridRotation = needRotLatLonGridRotation;
			opts.targetType = itsCalculationTarget;

			CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.speed), N * sizeof(double), cudaHostAllocMapped));

			if (itsCalculationTarget != kGust)
			{
				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.dir), N * sizeof(double), cudaHostAllocMapped));
			}

			if (opts.vectorCalculation)
			{
				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.vector), N * sizeof(double), cudaHostAllocMapped));
			}

			opts.firstLatitude = myTargetInfo->Grid()->FirstGridPoint().Y();
			opts.firstLongitude = myTargetInfo->Grid()->FirstGridPoint().X();
			opts.southPoleLat = myTargetInfo->Grid()->SouthPole().Y();
			opts.southPoleLon = myTargetInfo->Grid()->SouthPole().X();

			opts.di = myTargetInfo->Grid()->Di();
			opts.dj = myTargetInfo->Grid()->Dj();

			opts.cudaDeviceIndex = static_cast<unsigned short> (threadIndex-1);

			windvector_cuda::DoCuda(opts, datas);

			count = N;
			missingCount = opts.missingValuesCount;

			myTargetInfo->ParamIndex(0);
			myTargetInfo->Data()->Set(datas.speed, N);

			if (itsCalculationTarget != kGust)
			{
				myTargetInfo->ParamIndex(1);
				myTargetInfo->Data()->Set(datas.dir, N);
			}

			if (itsVectorCalculation)
			{
				myTargetInfo->ParamIndex(2);
				myTargetInfo->Data()->Set(datas.vector, N);
			}

			// Copy unpacked data to source info in case
			// some other thread/plugin calls for this same data.
			// Clear packed data now that it's been unpacked

			if (UInfo->Grid()->DataIsPacked())
			{
				UInfo->Data()->Set(datas.u, N);
				UInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.u));
			}

			if (VInfo->Grid()->DataIsPacked())
			{
				VInfo->Data()->Set(datas.v, N);
				VInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.v));
			}			
						
			CUDA_CHECK(cudaFreeHost(datas.speed));

			if (opts.targetType != kGust)
			{
				CUDA_CHECK(cudaFreeHost(datas.dir));
			}

			if (opts.vectorCalculation)
			{
				CUDA_CHECK(cudaFreeHost(datas.vector));
			}
			
			assert(UInfo->Grid()->ScanningMode() == VInfo->Grid()->ScanningMode());
			
			for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
			{
				SwapTo(myTargetInfo, UInfo->Grid()->ScanningMode());
			}

		}
		else
#endif
		{
			deviceType = "CPU";

			shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

			shared_ptr<NFmiGrid> UGrid(UInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> VGrid(VInfo->Grid()->ToNewbaseGrid());

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			UGrid->InterpolationMethod(kNearestPoint);
			VGrid->InterpolationMethod(kNearestPoint);

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

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
				
				if (needRotLatLonGridRotation)
				{
					/*
					 * 1. Get coordinates of current grid point in earth-relative form
					 * 2. Get coordinates of current grid point in grid-relative form
					 * 3. Call function UVToEarthRelative() that transforms U and V from grid-relative
					 *    to earth-relative
					 */

					assert(UGrid->Area()->ClassId() == kNFmiRotatedLatLonArea);

					const point regPoint(targetGrid->LatLon());

					const point rotPoint(reinterpret_cast<NFmiRotatedLatLonArea*> (UGrid->Area())->ToRotLatLon(regPoint.ToNFmiPoint()));

					point regUV = util::UVToEarthRelative(regPoint, rotPoint, UInfo->Grid()->SouthPole(), point(U,V));

					// Wind speed should the same with both forms of U and V

					assert(fabs(sqrt(U*U+V*V) - sqrt(regUV.X()*regUV.X() + regUV.Y() * regUV.Y())) < 0.001);

					U = regUV.X();
					V = regUV.Y();

				}
				else if (needStereographicGridRotation)
				{
					assert(UGrid->Area()->ClassId() == kNFmiStereographicArea);

					double centralLongitude = (reinterpret_cast<NFmiStereographicArea*> (targetGrid->Area())->CentralLongitude());

					point regUV = util::UVToGeographical(centralLongitude, point(U,V));

					// Wind speed should the same with both forms of U and V

					assert(fabs(sqrt(U*U+V*V) - sqrt(regUV.X()*regUV.X() + regUV.Y() * regUV.Y())) < 0.005);

				}

				double dir = 0;

				if (speed > 0)
				{
					dir = kRadToDeg * atan2(U,V) + directionOffset;

					// reduce the angle
					dir = fmod(dir,360);
					
					// force it to be the positive remainder, so that 0 <= dir < 360
					dir = fmod((dir + 360), 360);

				}

#ifndef HIL_PP_DD_COMPATIBILITY_MODE
				myTargetInfo->ParamIndex(1);
				myTargetInfo->Value(round(dir));
#endif
				
				if (itsVectorCalculation)
				{

					double windVector = round(dir/10) + 100 * round(speed);

					myTargetInfo->ParamIndex(2);

					if (!myTargetInfo->Value(windVector))
					{
						throw runtime_error(ClassName() + ": Failed to set value to matrix");
					}
	
#ifdef HIL_PP_DD_COMPATIBILITY_MODE
					dir = 10 * (static_cast<int> (round(windVector)) % 100);

					myTargetInfo->ParamIndex(1);
					myTargetInfo->Value(dir);
#endif
				}
			}

			for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
			{

				SwapTo(myTargetInfo, kBottomLeft);
			}
		}
		
		if (conf->StatisticsEnabled())
		{
			conf->Statistics()->AddToMissingCount(missingCount);
			conf->Statistics()->AddToValueCount(count);
		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));


		if (conf->FileWriteOption() != kSingleFile)
		{
			WriteToFile(conf, myTargetInfo);
		}
	}
}
