/**
 * @file cloud_type.cpp
 *
 * @date Jun 13, 2013
 * @author peramaki
 */

#include "cloud_type.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("cloud_type");

cloud_type::cloud_type()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void cloud_type::Process(std::shared_ptr<const plugin_configuration> conf)
{

	unique_ptr<timer> aTimer;

	// Get number of threads to use

	short threadCount = ThreadCount(conf->ThreadCount());

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
	 * Set target parameter to potential temperature
	 * - name CLDSYM-N
	 * - univ_id 328
	 * - grib2 descriptor 0'6'8
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("CLDSYM-N", 328);

	// GRIB 2
	
	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(6);
	theRequestedParam.GribParameter(8);

	theParams.push_back(theRequestedParam);
	
	// GRIB 1

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
	FeederInfo()->Param(theRequestedParam);

	if (conf->StatisticsEnabled())
	{
		aTimer->Stop();
		conf->Statistics()->AddToInitTime(aTimer->GetTime());
		aTimer->Start();
	}

	/*
	 * Each thread will have a copy of the target info.
	 */

	for (short i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		boost::thread* t = new boost::thread(&cloud_type::Run,
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

void cloud_type::Run(shared_ptr<info> myTargetInfo,
				shared_ptr<const plugin_configuration> conf,
				unsigned short threadIndex)
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

void cloud_type::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	param RHParam("RH-PRCNT");
	param NParam("N-0TO1");
	param KParam("KINDEX-N");

	level T2mLevel(himan::kHeight, 2, "HEIGHT");
	level NKLevel(himan::kHeight, 0, "HEIGHT");
	level RH850Level(himan::kPressure, 850, "PRESSURE");
	level RH700Level(himan::kPressure, 700, "PRESSURE");
	level RH500Level(himan::kPressure, 500, "PRESSURE");
	
	

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> T2mInfo;
		shared_ptr<info> NInfo;
		shared_ptr<info> KInfo;
		shared_ptr<info> T850Info;
		shared_ptr<info> RH850Info;
		shared_ptr<info> RH700Info;
		shared_ptr<info> RH500Info;
		
		try
		{
			// Source info for T2m
			T2mInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 T2mLevel,
								 TParam);
			// Source info for N
			NInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 NKLevel,
								 NParam);
			// Source info for kIndex
			KInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 NKLevel,
								 KParam);
			// Source info for T850
			T850Info = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 RH850Level,
								 TParam);	
			// Source info for RH850
			RH850Info = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 RH850Level,
								 RHParam);				
			// Source info for RH700
			RH700Info = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 RH700Level,
								 RHParam);
			// Source info for RH500
			RH500Info = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 RH500Level,
								 RHParam);

		}
		catch (HPExceptionType e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing);

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

		size_t missingCount = 0;
		size_t count = 0;

		/*
		 * Converting original grid-data to newbase grid
		 *
		 */

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T2mGrid(T2mInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> NGrid(NInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> KGrid(KInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T850Grid(T850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RH850Grid(RH850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RH700Grid(RH700Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RH500Grid(RH500Info->Grid()->ToNewbaseGrid());

		bool equalGrids = (*myTargetInfo->Grid() == *RH850Info->Grid() &&
							*myTargetInfo->Grid() == *RH700Info->Grid() &&
							*myTargetInfo->Grid() == *RH500Info->Grid() &&
							*myTargetInfo->Grid() == *T850Info->Grid() &&
							*myTargetInfo->Grid() == *T2mInfo->Grid() &&
							*myTargetInfo->Grid() == *NInfo->Grid() &&
							*myTargetInfo->Grid() == *KInfo->Grid());

		string deviceType = "CPU";

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();
		T2mGrid->Reset();
		NGrid->Reset();
		KGrid->Reset();
		T850Grid->Reset();
		RH850Grid->Reset();
		RH700Grid->Reset();
		RH500Grid->Reset();

		while ( myTargetInfo->NextLocation() && 
				targetGrid->Next() &&
				T2mGrid->Next() &&
				NGrid->Next() &&
				KGrid->Next() &&
				T850Grid->Next() &&
				RH850Grid->Next() &&
				RH700Grid->Next() &&
				RH500Grid->Next() )
		{

			count++;

			double T2m = kFloatMissing;
			double N = kFloatMissing;
			double kIndex = kFloatMissing;
			double T850 = kFloatMissing;
			double RH850 = kFloatMissing;
			double RH700 = kFloatMissing;
			double RH500 = kFloatMissing;

			InterpolateToPoint(targetGrid, T2mGrid, equalGrids, T2m);
			InterpolateToPoint(targetGrid, NGrid, equalGrids, N);
			InterpolateToPoint(targetGrid, KGrid, equalGrids, kIndex);
			InterpolateToPoint(targetGrid, T850Grid, equalGrids, T850);
			InterpolateToPoint(targetGrid, RH850Grid, equalGrids, RH850);
			InterpolateToPoint(targetGrid, RH700Grid, equalGrids, RH700);
			InterpolateToPoint(targetGrid, RH500Grid, equalGrids, RH500);

			if ( T2m == kFloatMissing || N == kFloatMissing || kIndex == kFloatMissing || T850 == kFloatMissing || RH850 == kFloatMissing || RH700 == kFloatMissing || RH500 == kFloatMissing )
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);
				continue;
			}

			//error codes from fortran
			int cloudCode = 704;
			//int cloudType = 1;

			double TBase = 273.15;
			T2m = T2m - TBase;
			T850 = T850 - TBase;
			
			int lowConvection = util::LowConvection(T2m, T850);
			
			//data comes as 0..1 instead of 0-100%
			N *= 100;
			RH500 *= 100;
			RH700 *= 100;
			RH850 *= 100;
			
			if ( N > 90 )
			//Jos N = 90…100 % (pilvistä), niin
			{
  				if ( RH500 > 65 )
  				{
  					cloudCode = 3502;
  				}
  				else 
  				{
  					cloudCode = 3306;
  				}

  				if ( RH700 > 80)
  				{
  					cloudCode = 3405;
  				}

  				if ( RH850 > 60 )
  				{
  					if ( RH700 > 80 )
  					{
  						cloudCode = 3604;
  					//	cloudType = 3;
						if (!myTargetInfo->Value(cloudCode))
						{
							throw runtime_error(ClassName() + ": Failed to set value to matrix");
						}
						continue;
  					}
  					else
  					{
  						cloudCode = 3307;
  					//	cloudType= 2;
  					}
  				}

  				//jos RH500 > 65, tulos 3502 (yläpilvi)
 				//ellei, niin tulos 3306 (yhtenäinen alapilvi)
 				//Jos kuitenkin RH700 > 80, tulos 3405 (keskipilvi)
 				//Jos kuitenkin RH850 > 60, niin
        			//jos RH700 > 80, tulos 3604 (paksut kerrospilvet) tyyppi = 3 (sade) > ulos
			    		//ellei, niin tulos 3307 (alapilvi) tyyppi = 2
				
				if ( kIndex > 25 )
				{
					cloudCode = 3309;
				//	cloudType = 4;
				}
				
				else if ( kIndex > 20)
				{
					cloudCode = 2303;
				//	cloudType = 4;
				}
				
				else if ( kIndex > 15 )
				{
					cloudCode = 2302;
				//	cloudType = 4;
				}
				else if ( lowConvection == 1 )
				{
					cloudCode = 2303;
				//	cloudType = 4;
				}
				/*
				Jos kIndex > 25, niin tulos 3309 (iso kuuropilvi), tyyppi 4
				Jos kIndex > 20, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4
				Jos kIndex > 15, niin tulos 2302 (konvektiopilvi), tyyppi 4
				Jos lowConvection = 1, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4
				*/
			
			}
			else if ( N > 50 )
			//Jos N = 50…90 % (puolipilvistä tai pilvistä), niin
			{
				if ( RH500 > 65 )
				{
					cloudCode = 2501;
				}

				else 
				{
					cloudCode = 2305;
				}

				if ( RH700 > 80 )
				{
					cloudCode = 2403;
				}

				if ( RH850 > 80 ) 
				{
					cloudCode = 2307;

				//	if ( N > 70 )
				//		cloudType = 2;
				}
      			/*	jos RH500 > 65, tulos 2501 (cirrus)
      				ellei, niin tulos 2305 (stratocumulus)
      				Jos kuitenkin RH700 > 80, tulos 2403 (keskipilvi)
      				Jos kuitenkin RH850 > 80, tulos 2307 (matala alapilvi)
            			ja jos N > 70 %, tyyppi 2
            	*/

            	if ( kIndex > 25 )
				{
					cloudCode = 3309;
				//	cloudType = 4;
				}
				
				else if ( kIndex > 20 )
				{
					cloudCode = 2303;
				//	cloudType = 4;
				}
				
				else if ( kIndex > 15 )
				{
					cloudCode = 2302;
				//	cloudType = 4;
				}
				else if ( lowConvection == 1 )
				{
					cloudCode = 2303;
				//	cloudType = 4;
				}
				/*
				Jos kIndex > 25, niin tulos 3309 (iso kuuropilvi), tyyppi 4
		    	Jos kIndex > 20, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4		
		    	Jos kIndex > 15, niin tulos 2302 (konvektiopilvi), tyyppi 4		
				Jos lowConvection = 1, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4
				*/
			}
			else if ( N > 10 )
			//Jos N = 10… 50 % (hajanaista pilvisyyttä)
			{
				if ( RH500 > 60 )
				{
					cloudCode = 1501;
				}

				else
				{
					cloudCode = 1305;
				}
				
				if ( RH700 > 70 )
				{					
					cloudCode = 1403;
				}

				if ( RH850 > 80 )
				{
					cloudCode = 1305;
				}
				
      			//	jos RH500 > 60, niin tulos 1501 (ohutta cirrusta), muuten tulos 1305 (alapilveä)
      			//	Jos RH700 > 70, tulos 1403 (keskipilveä)
      			//	Jos RH850 > 80, tulos 1305 (alapilveä)

				if ( kIndex > 25 )
				{
					cloudCode = 1309;
				//	cloudType = 4;
				}

				else if ( kIndex > 20 )
				{
					cloudCode = 1303;
				//	cloudType = 4;
				}

				else if ( kIndex > 15 )
				{
					cloudCode = 1302;
				//	cloudType = 4;
				}

				else if ( lowConvection == 2 )
				{
					cloudCode = 1301;
				}

				else if ( lowConvection == 1 )
				{
					cloudCode = 1303;
				//	cloudType = 4;
				}

				/*
				Jos kIndex > 25, niin tulos 1309 (korkea kuuropilvi), tyyppi 4
				Jos kIndex > 20, niin tulos 1303 (korkea konvektiopilvi), tyyppi 4
				Jos kIndex > 15, niin tulos 1302 (konvektiopilvi), tyyppi 4
				Jos lowConvection = 2, niin tulos 1301 (poutapilvi)
				Jos lowConvection = 1, niin tulos 1303 (korkea konvektiopilvi), tyyppi 4
				*/
			}
			else
				//Jos N 0…10 %
			{
      			//tulos 0. Jos lowConvection = 1, tulos 1303, tyyppi 4
      			cloudCode = 0;
      			if ( lowConvection == 1 )
      			{
      				cloudCode = 1303;
      			//	cloudType = 4;
      			}
      		}
			//Jälkikäsittely:

      		/*if ( cloudType == 2 && T850 < -9 )
      			cloudType = 5;

      		else if ( cloudType == 4 )
      		{
      			if (kIndex >= 37)
      				cloudType = 45;

      			else if (kIndex >= 27)
      				cloudType = 35;
      		}*/
				/*Jos tyyppi = 2 ja T850 < -9, tyyppi = 5 (lumisade alapilvistä)
				Jos tyyppi = 4, ja
      				kIndex > 37, tyyppi = 45
      				kIndex > 27, tyyppi = 35
      				*/
		

			if (!myTargetInfo->Value(cloudCode))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
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
		 * */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (conf->FileWriteOption() != kSingleFile)
		{
			WriteToFile(conf, myTargetInfo);
		}
	}
}
