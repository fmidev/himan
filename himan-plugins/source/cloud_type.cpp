/**
 * @file cloud_type
 *
 * Template for future plugins.
 *
 * @date Jun 13, 2013
 * @author peramaki
 */

#include "cloud_type.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "writer.h"
#include "neons.h"
#include "pcuda.h"

#undef HIMAN_AUXILIARY_INCLUDE

#ifdef DEBUG
#include "timer_factory.h"
#endif

using namespace std;
using namespace himan::plugin;

#include "cuda_extern.h"

const string itsName("cloud_type");

cloud_type::cloud_type() : itsUseCuda(false)
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void cloud_type::Process(std::shared_ptr<const plugin_configuration> conf)
{

	shared_ptr<plugin::pcuda> c = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

	if (c->HaveCuda())
	{
		string msg = "I possess the powers of CUDA";

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
		conf->Statistics()->UsedGPUCount(itsCudaDeviceCount);
	}

	boost::thread_group g;

	shared_ptr<info> targetInfo = conf->Info();

	/*
	 * Set target parameter to potential temperature
	 * - name CLDSYM-N
	 * - univ_id 328
	 * - grib2 descriptor X'Y'Z
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
	

	// GRIB 1

 	if (conf->OutputFileType() == kGRIB1)
	{
		shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		long parm_id = n->NeonsDB().GetGridParameterId(targetInfo->Producer().TableVersion(), theRequestedParam.Name());
		theRequestedParam.GribIndicatorOfParameter(parm_id);
		theRequestedParam.GribTableVersion(targetInfo->Producer().TableVersion());

	}

	// ----

	theParams.push_back(theRequestedParam);

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

	/*
	 * Each thread will have a copy of the target info.
	 */

	vector<shared_ptr<info> > targetInfos;

	targetInfos.resize(threadCount);

	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		targetInfos[i] = shared_ptr<info> (new info(*targetInfo));

		boost::thread* t = new boost::thread(&cloud_type::Run,
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

		int missingCount = 0;
		int count = 0;

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

		while (myTargetInfo->NextLocation() && targetGrid->Next())
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

			int cloudCode = 0;
			int cloudType = 0;
			T2m = T2m - 273.15;
			T850 = T850 - 273.15;
			int MATAKO = DoMatako(T2m, T850);
			N *= 100;
			bool skip = false;

			if ( N > 90 )
			//Jos N = 90…100 % (pilvistä), niin
			{
  				if ( RH500 > 65 )
  				{
  					cloudCode = 3502;
  					//cloudType = 4;//?;
  				}
  				else 
  				{
  					cloudCode = 3306;
  					//cloudType = 4;//?;
  				}

  				if ( RH700 > 80)
  				{
  					cloudCode = 3405;
  					//cloudType = 4;//?;
  				}

  				if ( RH850 > 60 )
  				{
  					if ( RH700 > 60 )
  					{
  						cloudCode = 3604;
  						cloudType = 3;
  						skip = true;
  					}
  					else
  					{
  						cloudCode = 3307;
  						cloudType= 2;
  					}
  				}

  				//jos RH500 > 65, tulos 3502 (yläpilvi)
 				//ellei, niin tulos 3306 (yhtenäinen alapilvi)
 				//Jos kuitenkin RH700 > 80, tulos 3405 (keskipilvi)
 				//Jos kuitenkin RH850 > 60, niin
        			//jos RH700 > 80, tulos 3604 (paksut kerrospilvet) tyyppi = 3 (sade) > ulos
			    		//ellei, niin tulos 3307 (alapilvi) tyyppi = 2
  				if (!skip)
  				{
	            	if ( kIndex > 25 )
					{
						cloudCode = 3309;
						cloudType = 4;
					}
					
					else if ( kIndex > 20)
					{
						cloudCode = 2303;
						cloudType = 4;
					}
					
					else if ( kIndex > 15 )
					{
						cloudCode = 2302;
						cloudType = 4;
					}
					if ( MATAKO == 1 )
					{
						cloudCode = 2303;
						cloudType = 4;
					}
					/*
					Jos kIndex > 25, niin tulos 3309 (iso kuuropilvi), tyyppi 4
					Jos kIndex > 20, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4
					Jos kIndex > 15, niin tulos 2302 (konvektiopilvi), tyyppi 4
					Jos MATAKO = 1, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4
					*/
				}
			}
			else if ( N > 50 )
			//Jos N = 50…90 % (puolipilvistä tai pilvistä), niin
			{
				if ( RH500 > 65 )
				{
					cloudCode = 2501;
					cloudType = 4;
				}

				else 
				{
					cloudCode = 2305;
					//cloudType = 4;//?;
				}

				if ( RH700 > 80 )
				{
					cloudCode = 2403;
					//cloudType = 4;//?;
				}

				if ( RH850 > 80 ) 
				{
					cloudCode = 2307;
					//cloudType = 4;//;

					if ( N > 70 )
						cloudType = 2;
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
					cloudType = 4;
				}
				
				else if ( kIndex > 20 )
				{
					cloudCode = 2303;
					cloudType = 4;
				}
				
				else if ( kIndex > 15 )
				{
					cloudCode = 2302;
					cloudType = 4;
				}
				if ( MATAKO == 1 )
				{
					cloudCode = 2303;
					cloudType = 4;
				}
				/*
				Jos kIndex > 25, niin tulos 3309 (iso kuuropilvi), tyyppi 4
		    	Jos kIndex > 20, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4		
		    	Jos kIndex > 15, niin tulos 2302 (konvektiopilvi), tyyppi 4		
				Jos MATAKO = 1, niin tulos 2303 (korkea konvektiopilvi), tyyppi 4
				*/
			}
			else if ( N > 10 )
			//Jos N = 10… 50 % (hajanaista pilvisyyttä)
			{
				if ( RH850 > 80 )
				{
					cloudCode = 1305;
					//cloudType = 4;//?;
				}
				
				else if ( RH700 > 70 )
				{					
					cloudCode = 1403;
					//cloudType = 4;//?;
				}

				else if ( RH500 > 60 )
				{
					cloudCode = 1501;
					//cloudType = 4;//?;
				}

				else
				{
					cloudCode = 1305;
					//cloudType = 4;//?;
				}
			
      			//	jos RH500 > 60, niin tulos 1501 (ohutta cirrusta), muuten tulos 1305 (alapilveä)
      			//	Jos RH700 > 70, tulos 1403 (keskipilveä)
      			//	Jos RH850 > 80, tulos 1305 (alapilveä)

				if ( kIndex > 25 )
				{
					cloudCode = 1309;
					cloudType = 4;
				}

				else if ( kIndex > 20 )
				{
					cloudCode = 1303;
					cloudType = 4;
				}

				else if ( kIndex > 15 )
				{
					cloudCode = 1302;
					cloudType = 4;
				}

				if ( MATAKO == 2 )
				{
					cloudCode = 1301;
					//cloudType = 4;//?;
				}

				if ( MATAKO == 1 )
				{
					cloudCode = 1303;
					cloudType = 4;
				}

				/*
				Jos kIndex > 25, niin tulos 1309 (korkea kuuropilvi), tyyppi 4
				Jos kIndex > 20, niin tulos 1303 (korkea konvektiopilvi), tyyppi 4
				Jos kIndex > 15, niin tulos 1302 (konvektiopilvi), tyyppi 4
				Jos MATAKO = 2, niin tulos 1301 (poutapilvi)
				Jos MATAKO = 1, niin tulos 1303 (korkea konvektiopilvi), tyyppi 4
				*/
				}
				else if ( N >= 0 || N < 10 )
				//Jos N 0…10 %
				{
      			//tulos 0. Jos MATAKO = 1, tulos 1303, tyyppi 4
      			cloudCode = 0;
      			if ( MATAKO == 1 )
      			{
      				cloudCode = 1303;
      				cloudType = 4;
      			}
      		}
			//Jälkikäsittely:

      		if ( cloudType == 2 && T850 < -9 )
      			cloudType = 5;

      		else if ( cloudType == 4 )
      		{
      			if (kIndex > 37)
      				cloudType = 45;

      			else if (kIndex > 27)
      				cloudType = 35;
      		}
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

		if (conf->FileWriteOption() == kNeons || conf->FileWriteOption() == kMultipleFiles)
		{
			shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

			theWriter->ToFile(shared_ptr<info> (new info(*myTargetInfo)), conf);
		}
	}
}

int cloud_type::DoMatako(double T2m, double T850)
{
	//double T2mC = T2m + 273.15;

	if ( T2m >= 8 && T2m - T850 >= 10)
		return 2;
	
	if ( T2m > 0 && T850 > 0 && T2m - T850 >= 10)
		return 1;
	
	return 0;
	//Menetelmä: Jos T2 >= 8 ja T2 - T850 >= 10, tulos on 2, jos T2 > 0 ja T850 > 0 ja T2 - T850 >= 10, tulos on 1
}