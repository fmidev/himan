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

#undef HIMAN_AUXILIARY_INCLUDE

#ifdef DEBUG
#include "timer_factory.h"
#endif

using namespace std;
using namespace himan::plugin;

const string itsName("rain_type");

rain_type::rain_type()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void rain_type::Process(std::shared_ptr<const plugin_configuration> conf)
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

	theParams.push_back(theRequestedParam);

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
		aTimer->Stop();
		conf->Statistics()->AddToProcessingTime(aTimer->GetTime());
	}

	if (conf->FileWriteOption() == kSingleFile)
	{
		WriteToFile(conf, targetInfo);
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
	param NParam("N-0TO1");
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

		int paramStep = 1;
		shared_ptr<info> PInfo;
		shared_ptr<info> Z850Info;
        shared_ptr<info> T850Info;
		shared_ptr<info> NInfo;
		shared_ptr<info> TInfo;
		shared_ptr<info> CloudInfo;
		shared_ptr<info> KindexInfo;
		shared_ptr<info> prevRRInfo;
		shared_ptr<info> nextRRInfo;
		
		try
		{
			// Fetch previous rain. Calculate average from these.
			try
			{
				forecast_time prevTimeStep = myTargetInfo->Time();
				prevTimeStep.ValidDateTime()->Adjust(myTargetInfo->Time().StepResolution(), -paramStep);				
				prevRRInfo = FetchSourceRR(conf,prevTimeStep,myTargetInfo->Level());

				forecast_time nextTimeStep = myTargetInfo->Time();
				nextTimeStep.ValidDateTime()->Adjust(myTargetInfo->Time().StepResolution(), paramStep);				
				nextRRInfo = FetchSourceRR(conf,prevTimeStep,myTargetInfo->Level());

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
			NInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 PLevel,
								 NParam);
			TInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 T2Level,
								 TParam);
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
		
		size_t missingCount = 0;
		size_t count = 0;

		/*
		 * Converting original grid-data to newbase grid
		 *
		 */

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> PGrid(PInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> Z850Grid(Z850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> NGrid(NInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> CloudGrid(CloudInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> KindexGrid(KindexInfo->Grid()->ToNewbaseGrid());
        shared_ptr<NFmiGrid> T850Grid(T850Info->Grid()->ToNewbaseGrid());
        shared_ptr<NFmiGrid> PrevRRGrid(prevRRInfo->Grid()->ToNewbaseGrid());
        shared_ptr<NFmiGrid> NextRRGrid(nextRRInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = (		*myTargetInfo->Grid() == *PInfo->Grid() &&
                                *myTargetInfo->Grid() == *Z850Info->Grid() &&
                                *myTargetInfo->Grid() == *TInfo->Grid() &&
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

				double N;
				double T;
                double T850;
				double Z850;
				double P;
				double cloudType = 704; // hil_pp:n oletusarvo
				double cloud;
				double reltopo;
				double prevRR;
				double nextRR;
				double kindex;

				InterpolateToPoint(targetGrid, NGrid, equalGrids, N);
				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
				InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
				InterpolateToPoint(targetGrid, Z850Grid, equalGrids, Z850);
				InterpolateToPoint(targetGrid, CloudGrid, equalGrids, cloud);
				InterpolateToPoint(targetGrid, KindexGrid, equalGrids, kindex);
                InterpolateToPoint(targetGrid, T850Grid, equalGrids, T850);
                InterpolateToPoint(targetGrid, PrevRRGrid, equalGrids, prevRR);
                InterpolateToPoint(targetGrid, NextRRGrid, equalGrids, nextRR);
			
				if (T == kFloatMissing )
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				// Koska P1 on 1000Mba, pitää z tehdä paineesta
				// Sen kautta sitten lasketaan reltopo
				reltopo = util::RelativeTopography(1000, 850, P, Z850);
				
				double rain = 0; // sateen saakoodi, oletus pouta

				// Laske intensiteetti ensin, sitten päättele WaWa-koodi
				// Voi olla, että tässä on väärä sade

				
				if (nextRR > 0.01 && prevRR > 0.01 )
				{
					rain = 60;
				}
				if (nextRR > 0.1 && prevRR > 0.1 ) 
				{
                   	rain = 61;
				}
				if (nextRR > 1 && prevRR > 1 ) 
				{
                    rain = 63;
				}
				if (nextRR > 3 && prevRR > 3) 
				{
                    rain = 65;
				}

				// Pilvikoodista päätellään pilvityyppi
				// Päättelyketju vielä puutteellinen, logiikan voi ehkä siirtää 
				// cloud_type plugarista

				N *= 100;

				if (cloud == 3307 ) 
				{
					cloudType = 2;  // sade alapilvesta
				}
				else if (cloud == 2307 && N > 70 )
					cloudType = 2;

				else if (cloud == 3604) 
				{
				    cloudType = 3;	// sade paksusta pilvesta
				}
				else if (cloud == 3309 || cloud == 2303 || cloud == 2302
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
					   		//rain = rain;   // Vesi
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

                

	                if (reltopo >= 1289) // Lopuksi jäätävä sade
	                {
	                	if (rain >= 60 && rain <= 61 && T <= 270.15)
	                  	{
	                    	rain = 66;
	                	}
	                	else if (rain >= 62 && rain <= 65 && T <= 270.15)
	                	{
					    	rain = 67;
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

		if (conf->FileWriteOption() != kSingleFile)
		{
			WriteToFile(conf, myTargetInfo);
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
