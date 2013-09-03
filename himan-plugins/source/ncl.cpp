/**
 * @file ncl.cpp
 *
 * Template for future plugins.
 *
 * @date Apr 10, 2013
 * @author peramaki
 */

#include "ncl.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("ncl");

ncl::ncl()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void ncl::Process(std::shared_ptr<const plugin_configuration> conf)
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
	
	// GRIB 2
	
	param theRequestedParam;
	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(3);
	theRequestedParam.GribParameter(6);

	if (conf->Exists("temp") && conf->GetValue("temp") == "-20" )
	{
    	theRequestedParam.Name("HM20C-M");
    	theRequestedParam.UnivId(28);
    	targetTemperature = -20;
    }

    if (conf->Exists("temp") && conf->GetValue("temp") == "0" )
	{
    	theRequestedParam.Name("H0C-M");
    	theRequestedParam.UnivId(270);
    	targetTemperature = 0;
    	
    }

	/*
	 * GRIB 1 parameters go here
	 *
	 */

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


	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		boost::thread* t = new boost::thread(&ncl::Run,
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

void ncl::Run(shared_ptr<info> myTargetInfo,
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

void ncl::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	param HParam("HL-M");
	param TParam("T-K");

	int levelNumber = 65;

	level HLevel(himan::kHybrid, static_cast<float> (levelNumber), "HYBRID");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> HInfo;
		shared_ptr<info> TInfo;
		shared_ptr<info> prevHInfo;
		shared_ptr<info> prevTInfo;

		try
		{

			HInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 HLevel,
								 HParam);
			
			TInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 HLevel,
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

		bool firstLevel = true;

		myTargetInfo->Data()->Fill(0);

			
		
		
		HInfo->ResetLocation();
		TInfo->ResetLocation();

		level curLevel = HLevel;
		level prevLevel;
		
		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

		string deviceType;
		deviceType = "CPU";


		while (--levelNumber > 0)
		{

			targetGrid->Reset();		
			myTargetInfo->FirstLocation();	
			
			 //minneköhän tää pitäis pistää
			//itsLogger->Debug("kierros woop " +  boost::lexical_cast<string> (count) );

			shared_ptr<NFmiGrid> HGrid(HInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> prevHGrid;
			shared_ptr<NFmiGrid> prevTGrid;

			if (!firstLevel)
			{
				prevHGrid = shared_ptr<NFmiGrid>(prevHInfo->Grid()->ToNewbaseGrid());
				prevTGrid = shared_ptr<NFmiGrid>(prevTInfo->Grid()->ToNewbaseGrid());
			}


			bool equalGrids = (	*myTargetInfo->Grid() == *HInfo->Grid() 
								&& *myTargetInfo->Grid() == *TInfo->Grid() 
								&& ( firstLevel || ( *myTargetInfo->Grid() == *prevHInfo->Grid() && *myTargetInfo->Grid() == *prevTInfo->Grid() ) ) );

			while ( myTargetInfo->NextLocation() && targetGrid->Next() && HGrid->Next() && TGrid->Next() && ( firstLevel || (prevHGrid->Next() && prevTGrid->Next() ) ) )
			{
				count++;
		

				double height = kFloatMissing;
				double temp = kFloatMissing;
				double prevHeight = kFloatMissing;
				double prevTemp = kFloatMissing;

				double targetHeight = myTargetInfo->Value();

				assert(targetGrid->Size() == myTargetInfo->Data()->Size());

				InterpolateToPoint(targetGrid, HGrid, equalGrids, height);
				InterpolateToPoint(targetGrid, TGrid, equalGrids, temp);

				if (!firstLevel)
				{
					InterpolateToPoint(targetGrid, prevHGrid, equalGrids, prevHeight);
					InterpolateToPoint(targetGrid, prevTGrid, equalGrids, prevTemp);
				}

				if (height == kFloatMissing || temp == kFloatMissing || (!firstLevel && ( prevHeight == kFloatMissing || prevTemp == kFloatMissing)))
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				temp -= 273.15;
				prevTemp -= 273.15;

				if (targetHeight == 0)
				{
					//itsLogger->Debug("level: " + boost::lexical_cast<string> (height));
					
					if (temp < targetTemperature)
					{
						if (!firstLevel)
						{
							double p_rel = (targetTemperature - temp) / (prevTemp - temp);
							targetHeight = height + (prevHeight - height) * p_rel;
						}
						else
						{
							targetHeight = kFloatMissing;
						}
					}
				}
				//Inversiotilanteessa pelastetaan vielä pisteitä uudelleen laskentaan
				else if (targetHeight != 0 && temp > targetTemperature)
				{
					targetHeight = 0;
				}

		
				//itsLogger->Debug("level: " + boost::lexical_cast<string> (targetHeight));
				if (!myTargetInfo->Value(targetHeight))
				{
					throw runtime_error(ClassName() + ": Failed to set value to matrix");
				}

			}

			prevLevel = curLevel;
			curLevel = level(himan::kHybrid, static_cast<float> (levelNumber), "HYBRID");
			
			HInfo = FetchPrevious(conf, myTargetInfo->Time(), curLevel, HParam);
			TInfo = FetchPrevious(conf, myTargetInfo->Time(), curLevel, TParam);
			
			prevHInfo = FetchPrevious(conf, myTargetInfo->Time(), prevLevel, HParam);
			prevTInfo = FetchPrevious(conf, myTargetInfo->Time(), prevLevel, TParam);


			firstLevel = false;
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
shared_ptr<himan::info> ncl::FetchPrevious(shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime, const level& wantedLevel, const param& wantedParam)
{
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	try
	{
		return f->Fetch(conf,
						wantedTime,
						wantedLevel,
						wantedParam);
   	}
	catch (HPExceptionType e)
	{
		throw e;
	}

}