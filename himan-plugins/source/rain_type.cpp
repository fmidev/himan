/**
 * @file rain_type
 *
 *
 * @date Apr 10, 2013
 * @author partio, peramaki, aalto
 */

#include "rain_type.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include "util.h"
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

const string itsName("rain_type");

rain_type::rain_type() : itsUseCuda(false)
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void rain_type::Process(std::shared_ptr<const plugin_configuration> conf)
{
	unique_ptr<timer> initTimer;

	if (conf->StatisticsEnabled())
	{
		initTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
		initTimer->Start();
	}

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
	 * - name PARM_NAME
	 * - univ_id UNIV_ID
	 * - grib2 descriptor X'Y'Z
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("HSADE1-N", 52);

	// GRIB 2

	/*
	 * theRequestedParam.GribDiscipline(X);
	 * theRequestedParam.GribCategory(Y);
	 * theRequestedParam.GribParameter(Z);
	 */

	// GRIB 1

	/*
	 * GRIB 1 parameters go here
	 *
	 */
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

	if (conf->StatisticsEnabled())
	{
		initTimer->Stop();
		conf->Statistics()->AddToInitTime(initTimer->GetTime());
	}

	/*
	 * Each thread will have a copy of the target info.
	 */

	unique_ptr<timer> processTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());

	if (conf->StatisticsEnabled())
	{
		processTimer->Start();
	}

	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		boost::thread* t = new boost::thread(&rain_type::Run,
								this,
								shared_ptr<info> (new info(*targetInfo)),
								conf,
								i + 1);

		g.add_thread(t);

	}

	g.join_all();

	if (conf->StatisticsEnabled())
	{
		processTimer->Stop();
		conf->Statistics()->AddToProcessingTime(processTimer->GetTime());
	}

	if (conf->FileWriteOption() == kSingleFile)
	{

		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		string theOutputFile = conf->ConfigurationFile();

		theWriter->ToFile(targetInfo, conf, theOutputFile);

	}
}

void rain_type::Run(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
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

void rain_type::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	/*
	 * eg. param PParam("P-Pa"); for pressure in pascals
	 *
	 */

	param ZParam("Z-M2S2");
	param TParam("T-K");
	param PParam("P-PA");
	param CloudParam("CLDSYM-N");
	param PrecParam("RR-1-MM");
	param KindexParam("KINDEX-N");

    level Z850Level(himan::kPressure, 850, "PRESSURE");
    level T2Level(himan::kHeight, 2, "HEIGHT");
    level PLevel(himan::kHeight, 0, "HEIGHT");


	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		long paramStep = 1;
		shared_ptr<info> PInfo;
		shared_ptr<info> Z850Info;
        shared_ptr<info> T850Info;
		shared_ptr<info> TInfo;
		shared_ptr<info> RRInfo;
		shared_ptr<info> CloudInfo;
		shared_ptr<info> KindexInfo;
		shared_ptr<info> prevRRInfo;
		
		try
		{
			// Fetch previous rain. Calculate average from these.
			try
			{
				forecast_time prevTimeStep = myTargetInfo->Time();
				prevTimeStep.ValidDateTime()->Adjust(myTargetInfo->Time().StepResolution(), -paramStep);				

				prevRRInfo = FetchSourceRR(conf,prevTimeStep,myTargetInfo->Level());

			}
			catch (HPExceptionType e)
			{
				switch (e)
				{
					case kFileDataNotFound:
						myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
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

			/*
			 *	Parameter infos are made here
			 *
			 */
			PInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 PLevel,
								 PParam);
			Z850Info = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 Z850Level,
								 ZParam);
			TInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 T2Level,
								 TParam);
			RRInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 PLevel,
								 PrecParam);
			CloudInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 PLevel,
								 CloudParam);
			KindexInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 PLevel,
								 KindexParam);
            T850Info = theFetcher->Fetch(conf,
                                 myTargetInfo->Time(),
                                 Z850Level,
                                 TParam);

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
		shared_ptr<NFmiGrid> PGrid(PInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> Z850Grid(Z850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RRGrid(RRInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> CloudGrid(CloudInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> KindexGrid(KindexInfo->Grid()->ToNewbaseGrid());
        shared_ptr<NFmiGrid> T850Grid(T850Info->Grid()->ToNewbaseGrid());
        shared_ptr<NFmiGrid> PrevRRGrid(prevRRInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = (		*myTargetInfo->Grid() == *PInfo->Grid() &&
                                *myTargetInfo->Grid() == *Z850Info->Grid() &&
                                *myTargetInfo->Grid() == *TInfo->Grid() &&
								*myTargetInfo->Grid() == *RRInfo->Grid() &&
								*myTargetInfo->Grid() == *CloudInfo->Grid());


		string deviceType;

			deviceType = "CPU";

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{

				count++;

				/*
				 * interpolation happens here
				 *
				 */
			
				double rain; // sateen saakoodi
				double T;
                double T850;
				double Z850;
				double P;
				double cloudType;
				double cloud;
				double reltopo;
				double RR;
				double prevRR;
				double kindex;

				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
				InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
				InterpolateToPoint(targetGrid, Z850Grid, equalGrids, Z850);
				InterpolateToPoint(targetGrid, RRGrid, equalGrids, RR);
				InterpolateToPoint(targetGrid, CloudGrid, equalGrids, cloud);
				InterpolateToPoint(targetGrid, KindexGrid, equalGrids, kindex);
                InterpolateToPoint(targetGrid, T850Grid, equalGrids, T850);
                InterpolateToPoint(targetGrid, PrevRRGrid, equalGrids, prevRR);
			
				if (T == kFloatMissing )
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				// Koska P1 on 1000Mba, pitää z tehdä paineesta
				// Sen kautta sitten lasketaan reltopo
				reltopo = util::RelativeTopography(1000, 850, P, Z850);
				
				double rainPeriod = RR - prevRR;

				if ( rainPeriod <= 0)
					rainPeriod = 0;
            
				// Laske intensiteetti ensin, sitten päättele WaWa-koodi
				// Voi olla, että tässä on väärä sade

				if (rainPeriod > 0.01 && rainPeriod < 0.1 ) {
				  
					rain = 60;
				}
				else if (rainPeriod > 0.1 && rainPeriod < 1 ) 
				{
                   	rain = 61;
				}
				else if (rainPeriod > 1 && rainPeriod < 3 ) 
				{
                    rain = 63;
				}
				else if (rainPeriod > 3 ) 
				{
                    rain = 65;
				}
				
				// Pilvikoodista päätellään pilvityyppi
				// Päättelyketju vielä puutteellinen, logiikan voi ehkä siirtää 
				// cloud_type plugarista

				if (cloud == 3307) 
				{
					cloudType = 2;  // sade alapilvesta
				}
				else if (cloud == 3604) 
				{
				    cloudType = 3;	// sade paksusta pilvesta
				}
				else if (cloud == 3309 || cloud == 2303 || cloud == 2302 || cloud == 2303
					 || cloud == 2307 || cloud == 3309 || cloud == 2303 || cloud == 2302
					 || cloud == 1309 || cloud == 1303 || cloud == 1302)
				{
				   cloudType = 4; 	// kuuropilvi
				}

				// Ukkoset

				if ( cloudType == 2 && T850 < -9 )
      			           cloudType = 5;  // lumisade

      		    if ( cloudType == 4 )
      		    {
      			    if (kindex >= 37)
      				    cloudType = 45;  // ukkossade

      			    else if (kindex >= 27)
      				    cloudType = 35; // ukkossade
      		    }

				
				// Sitten itse HSADE

				if (rain >= 60 && rain <= 65) 
				{
					if (cloudType == 3) // Jatkuva sade
					{
              
						if (reltopo < 1288) 
						{           
                       		rain = rain + 10;  // Lumi
                     	}
                     	else if (reltopo > 1300) 
                     	{
					   		rain = 60;   // Vesi
					 	}
					 	else 
					 	{
                       		rain = 68;  // Räntä
                     	}
                  	}
                 	else if (cloudType == 45) // Kuuroja + voimakasta ukkosta
                	{               
                    	if (reltopo < 1285) 
                    	{

							if (rain >= 63) //Lumikuuroja
					   		{
					  	 		rain = 86;
                       		}
					   		else
					   		{
					  	 		rain = 85;
					   		}
                    	}
				    	else 
				    	{
    				  		rain = 97;  // Kesällä ukkosta
				    	}
                	}
                	else if (cloudType == 35)   // Kuuroja + ukkosta
                	{
			        	if (reltopo < 1285) 
			        	{
            
					  		if (rain >= 63)   // Lumikuuroja
					  		{     
					  			rain = 86;
                      		}
					  		else 
					  		{
					  			rain = 85;
					  		}
                    	}
				    	else 
				    	{
					  		rain = 95;  // Kesällä ukkosta
				    	}
                  	}
                 	else if (cloudType == 4) // Kuuroja - ukkosta
                	{

                   		if (reltopo < 1285) 
                    	{
          
					  		if (rain >= 63) 
					  		{
	    			  			rain = 86;
                      		}
					  		else 
					  		{
					  			rain = 85;
					 	 	}
				    	}
				    	else  
				    	{
					  		if (rain >= 63) // Vesikuuroja
					  		{          
					  	 		rain = 82;           
					  		}
					  		else 
					  		{
					  	 		rain = 80;
					 	 	}
						}
                  	}
                	else if (cloudType == 2)   // Tihkua
                	{  
                    	if (rain <= 61) // Sademäärä ei saa olla suuri
                    	{
                 	  		if (reltopo < 1288) 
                 	  		{
                 		 		rain = 78;  // Lumikiteitä
                 	  		}
                 	  		else 
                 	  		{
                 	    		rain = rain - 10;  // Tihkua
                 	  		}
                    	}
                  	}
                  	else if (cloudType == 5) //Lumisadetta alapilvistä
                  	{  
                    	rain = rain + 10;
                  	}
                  	else // Hetkellisen sateen virhe, siis poutaa
                  	{
                    	rain = 0;
                  	}

                }

                if (reltopo > 1289) // Lopuksi jäätävä sade
                {
                	if (rain >= 60 && rain <= 61 && T <= 270.15)
                  	{
                    	rain = 66;
                	}
                	else if (rain >= 62 && rain <= 65 && T <= 270.15)
                	{
				    	rain = 66;
                	}
                	else if (rain >= 50 && rain <= 51 && T <= 270.15)
                	{
				    	rain = 56;
                	}
                	else if (rain >= 52 && rain <= 55 && T <= 270.15)
                  	{
				    	rain = 57;
                	}

                }

				if (!myTargetInfo->Value(rain))
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

shared_ptr<himan::info> rain_type::FetchSourceRR(shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime, const level& wantedLevel)
{
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	try
	{
		return f->Fetch(conf,
						wantedTime,
						wantedLevel,
						param("RR-1-MM"));
   	}
	catch (HPExceptionType e)
	{
		throw e;
	}

}
