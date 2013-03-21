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
#include "windvector_cuda_options.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "writer.h"
#include "pcuda.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#include "cuda_extern.h"

const double kRadToDeg = 57.295779513082; // 180 / PI

windvector::windvector()
	: itsUseCuda(false)
	, itsSeaCalculation(false)
	, itsIceCalculation(false)
	, itsWindCalculation(false)
	, itsWindGustCalculation(false)
	, itsVectorCalculation(false)
{
	itsClearTextFormula = "speed = sqrt(U*U+V*V) ; direction = round(180/PI * atan2(U,V) + offset) ; vector = round(dir/10) + 100 * round(speed)";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("windvector"));

}

void windvector::Process(std::shared_ptr<const plugin_configuration> conf)
{

	shared_ptr<plugin::pcuda> c = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

	if (c && c->HaveCuda())
	{
		string msg = "I possess the powers of CUDA ";

		if (!conf->UseCuda())
		{
			msg += ", but I won't use them";
		}
		else
		{
			msg += ", and I'm not afraid to use them";
			itsUseCuda = true;
		}

		itsLogger->Info(msg);
		itsCudaDeviceCount = c->DeviceCount();

	}

	// Get number of threads to use

	unsigned short threadCount = ThreadCount(conf->ThreadCount());

	if (conf->StatisticsEnabled())
	{
		conf->Statistics()->UsedThreadCount(threadCount);
		conf->Statistics()->UsedCudaCount(itsCudaDeviceCount);
	}

	boost::thread_group g;

	shared_ptr<info> targetInfo = conf->Info();

	/*
	 * Get producer information from neons
	 */

	if (conf->FileWriteOption() == kNeons)
	{
		shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

		map<string,string> prodInfo = n->ProducerInfo(targetInfo->Producer().Id());

		if (!prodInfo.empty())
		{
			producer prod(targetInfo->Producer().Id());

			prod.Process(boost::lexical_cast<long> (prodInfo["process"]));
			prod.Centre(boost::lexical_cast<long> (prodInfo["centre"]));
			prod.Name(prodInfo["name"]);

			targetInfo->Producer(prod);
		}

	}

	/*
	 * Set target parameter to windvector
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	// By default assume we'll calculate for wind

	param requestedDirParam("DD-D", 20);
	requestedDirParam.GribDiscipline(0);
	requestedDirParam.GribCategory(2);
	requestedDirParam.GribParameter(0);

	param requestedSpeedParam("FF-MS", 21);
	requestedSpeedParam.GribDiscipline(0);
	requestedSpeedParam.GribCategory(2);
	requestedSpeedParam.GribParameter(1);

	param requestedVectorParam("DF-MS", 22);

	if (conf->Exists("do_vector") && conf->GetValue("do_vector") == "true")
	{
		itsVectorCalculation = true;
	}
	
	if (conf->Exists("for_ice") && conf->GetValue("for_ice") == "true")
	{
		requestedSpeedParam = param("IFF-MS", 389);
		requestedDirParam = param("IDD-D", 390);

		requestedSpeedParam.GribDiscipline(10);
		requestedSpeedParam.GribCategory(2);
		requestedSpeedParam.GribParameter(3);

		requestedDirParam.GribDiscipline(10);
		requestedDirParam.GribCategory(2);
		requestedDirParam.GribParameter(2);

		itsIceCalculation = true;

		if (conf->Exists("do_vector") && conf->GetValue("do_vector") == "true")
		{
			itsLogger->Error("Unable to calculate vector for ice");
		}
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
		
		itsSeaCalculation = true;

		if (conf->Exists("do_vector") && conf->GetValue("do_vector") == "true")
		{
			itsLogger->Error("Unable to calculate vector for sea");
		}
	}
	else if (conf->Exists("for_gust") && conf->GetValue("for_gust") == "true")
	{
		requestedSpeedParam = param("WG-MS", 417);
		
		requestedSpeedParam.GribDiscipline(0);
		requestedSpeedParam.GribCategory(2);
		requestedSpeedParam.GribParameter(22);

		itsWindGustCalculation = true;

		if (conf->Exists("do_vector") && conf->GetValue("do_vector") == "true")
		{
			itsLogger->Error("Unable to calculate vector for wind gust");
		}
	}
	else
	{
		itsWindCalculation = true;
	}

	theParams.push_back(requestedSpeedParam);

	if (!itsWindGustCalculation)
	{
		theParams.push_back(requestedDirParam);
	}

	if (itsVectorCalculation)
	{
		theParams.push_back(requestedVectorParam);
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

	/*
	 * Each thread will have a copy of the target info.
	 */

	vector<shared_ptr<info> > targetInfos;

	targetInfos.resize(threadCount);

	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		targetInfos[i] = shared_ptr<info> (new info(*targetInfo));

		boost::thread* t = new boost::thread(&windvector::Run,
								this,
								targetInfos[i],
								conf,
								i + 1);

		g.add_thread(t);

	}

	g.join_all();

	if (conf->FileWriteOption() == kSingleFile)
	{

		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		string theOutputFile = conf->ConfigurationFile();

		theWriter->ToFile(targetInfo, conf, theOutputFile);

	}
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

	param UParam("U-MS");
	param VParam("V-MS");

	double directionOffset = 180; // For wind direction add this

	if (itsSeaCalculation)
	{
		UParam = param("WVELU-MS");
		VParam = param("WVELV-MS");
		directionOffset = 0;
	}
	else if (itsIceCalculation)
	{
		UParam = param("IVELU-MS");
		VParam = param("IVELV-MS");
		directionOffset = 0;
	}
	else if (itsWindGustCalculation)
	{
		UParam = param("WGU-MS");
		VParam = param("WGV-MS");
	}

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("windvectorThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->ParamIndex(0);

	// Fetch source level definition

	level sourceLevel = compiled_plugin_base::LevelTransform(conf->SourceProducer(), UParam, myTargetInfo->PeakLevel(0));

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
								 UParam);
				
			// Source info for V
			VInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 sourceLevel,
								 VParam);
				
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


		unique_ptr<timer> processTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());

		if (conf->StatisticsEnabled())
		{
			processTimer->Start();
		}
		
		// if source producer is Hirlam, we must de-stagger U and V grid

		/*if (conf->SourceProducer().Id() == 1 && sourceLevel.Type() != kHeight)
		{
			UInfo->Grid()->Stagger(-0.5, 0);
			VInfo->Grid()->Stagger(0, -0.5);
		}*/

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> UGrid(UInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> VGrid(VInfo->Grid()->ToNewbaseGrid());

		UGrid->InterpolationMethod(kNearestPoint);
		VGrid->InterpolationMethod(kNearestPoint);

		int missingCount = 0;
		int count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *UInfo->Grid() && *myTargetInfo->Grid() == *VInfo->Grid());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		assert(UInfo->Grid()->Projection() == VInfo->Grid()->Projection());

		bool needRotLatLonGridRotation = (UInfo->Grid()->Projection() == kRotatedLatLonProjection && UInfo->Grid()->UVRelativeToGrid());
		bool needStereographicGridRotation = (UInfo->Grid()->Projection() == kStereographicProjection && UInfo->Grid()->UVRelativeToGrid());

		string deviceType;

		if (itsUseCuda && equalGrids && !needStereographicGridRotation && threadIndex <= itsCudaDeviceCount)
		{
			deviceType = "GPU";

			assert(UInfo->Grid()->Projection() == kLatLonProjection || UInfo->Grid()->ScanningMode() == kBottomLeft);
			assert(UInfo->Grid()->Projection() == kLatLonProjection || UInfo->Grid()->Projection() == kRotatedLatLonProjection);
			
			windvector_cuda_options opts;

			opts.Uin = const_cast<float*> (UGrid->DataPool()->Data());
			opts.Vin = const_cast<float*> (VGrid->DataPool()->Data());

			opts.sizeX = myTargetInfo->Grid()->Data()->SizeX();
			opts.sizeY = myTargetInfo->Grid()->Data()->SizeY();

			opts.vectorCalculation = itsVectorCalculation;
			opts.needRotLatLonGridRotation = needRotLatLonGridRotation;
			opts.dirCalculation = !itsWindGustCalculation; // direction not calculated for gust

			if (itsVectorCalculation)
			{
				opts.dataOut = new float[3*opts.sizeX*opts.sizeY];
			}
			else
			{
				opts.dataOut = new float[2*opts.sizeX*opts.sizeY];
			}

			opts.firstLatitude = myTargetInfo->Grid()->FirstGridPoint().Y();
			opts.firstLongitude = myTargetInfo->Grid()->FirstGridPoint().X();
			opts.southPoleLat = myTargetInfo->Grid()->SouthPole().Y();
			opts.southPoleLon = myTargetInfo->Grid()->SouthPole().X();

			opts.di = myTargetInfo->Grid()->Di();
			opts.dj = myTargetInfo->Grid()->Dj();

			opts.CudaDeviceIndex = threadIndex-1;

			windvector_cuda::DoCuda(opts);

			size_t N = opts.sizeX*opts.sizeY;

			double* FFdata = new double[N];
			double* DDdata = new double[N];
			double* DFdata;
			
			if (itsVectorCalculation)
			{
				DFdata = new double[N];
			}

			for (size_t i = 0; i < N; i++)
			{
				count++;
				
				FFdata[i] = static_cast<double> (opts.dataOut[i]);
				DDdata[i] = static_cast<double> (opts.dataOut[i+N]);

				if (FFdata[i] == kFloatMissing || DDdata[i] == kFloatMissing)
				{
					missingCount++;

					// Make sure both are missing
					FFdata[i] = kFloatMissing;
					DDdata[i] = kFloatMissing;
				}

				if (itsVectorCalculation)
				{
					// No need to check missing value here, if it is missing then it is
					DFdata[i] = static_cast<double> (opts.dataOut[i+2*N]);
				}
			}

			myTargetInfo->ParamIndex(0);
			myTargetInfo->Data()->Set(FFdata, N);

			myTargetInfo->ParamIndex(1);
			myTargetInfo->Data()->Set(DDdata, N);

			if (itsVectorCalculation)
			{
				myTargetInfo->ParamIndex(2);
				myTargetInfo->Data()->Set(DFdata, N);

				delete [] DFdata;
			}

			delete [] FFdata;
			delete [] DDdata;
			
			delete [] opts.dataOut;
		}
		else
		{
			deviceType = "CPU";

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
					myTargetInfo->ParamIndex(1);
					myTargetInfo->Value(kFloatMissing);

					if (itsVectorCalculation)
					{
						myTargetInfo->ParamIndex(2);
						myTargetInfo->Value(kFloatMissing);
					}

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

					assert(fabs(sqrt(U*U+V*V) - sqrt(regUV.X()*regUV.X() + regUV.Y() * regUV.Y())) < 0.001);

				}

				double speed = sqrt(U*U + V*V);

				double dir = 0;

				if (!itsWindGustCalculation && speed > 0)
				{
					dir = kRadToDeg * atan2(U,V) + directionOffset;

					// reduce the angle
					dir = fmod(dir,360);
					
					// force it to be the positive remainder, so that 0 <= dir < 360
					dir = fmod((dir + 360), 360);

				}

				/*
				 * The order of parameters in infos is and must be always:
				 * index 0 : speed parameter
				 * index 1 : direction parameter (not available for gust)
				 * index 2 : vector parameter (optional)
				 */

				myTargetInfo->ParamIndex(0);
				myTargetInfo->Value(speed);

#ifndef HIL_PP_DD_COMPATIBILITY_MODE
				if (!itsWindGustCalculation)
				{
					myTargetInfo->ParamIndex(1);
					myTargetInfo->Value(round(dir));
				}
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
		}
		
		if (conf->StatisticsEnabled())
		{
			processTimer->Stop();
			conf->Statistics()->AddToProcessingTime(processTimer->GetTime());

#ifdef DEBUG
			itsLogger->Debug("Calculation took " + boost::lexical_cast<string> (processTimer->GetTime()) + " microseconds on "  + deviceType);
#endif

			conf->Statistics()->AddToMissingCount(missingCount);
			conf->Statistics()->AddToValueCount(count);

		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (conf->FileWriteOption() == kNeons || conf->FileWriteOption() == kMultipleFiles)
		{
			shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

			shared_ptr<info> tempInfo(new info(*myTargetInfo));

			for (tempInfo->ResetParam(); tempInfo->NextParam(); )
			{
				theWriter->ToFile(tempInfo, conf);
			}
		}
	}
}
