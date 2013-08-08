/**
 * @file seaicing.cpp
 *
 *  Created on: Jan 03, 2013
 *  @author aaltom
 */

#include "seaicing.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "writer.h"
#include "pcuda.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#include "cuda_extern.h"

seaicing::seaicing() : itsUseCuda(false)
{
	itsClearTextFormula = "SeaIcing = FF * ( -0.35 -T2m ) / ( 1 + 0.3 * ( T0 + 0.35 ))";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("seaicing"));

}

void seaicing::Process(std::shared_ptr<const plugin_configuration> conf)
{

	unique_ptr<timer> initTimer;

	if (conf->StatisticsEnabled())
	{
		initTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
		initTimer->Start();
	}
	
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

	}

	// Get number of threads to use

	unsigned short threadCount = ThreadCount(conf->ThreadCount());
        
        if (conf->StatisticsEnabled())
	{
		conf->Statistics()->UsedThreadCount(threadCount);
	//	conf->Statistics()->UsedCudaCount(itsCudaDeviceCount);
	}

	if (conf->StatisticsEnabled())
	{
		conf->Statistics()->UsedThreadCount(threadCount);
		conf->Statistics()->UsedGPUCount(0);
	}
	
	boost::thread_group g;

	shared_ptr<info> targetInfo = conf->Info();

	/*
	 * Set target parameter to seaicing
	 * - name ICEIND-N
	 * - univ_id 190
	 * - grib2 descriptor 0'00'002
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from conf but why bother?)
	 *
	 */

	vector<param> theParams;

	param requestedParam("ICEIND-N", 190);

	// GRIB 2
	requestedParam.GribDiscipline(0);
	requestedParam.GribCategory(0);
	requestedParam.GribParameter(2);

	// GRIB 1

	if (conf->OutputFileType() == kGRIB1)
	{
		shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		long parm_id = n->NeonsDB().GetGridParameterId(targetInfo->Producer().TableVersion(), requestedParam.Name());
		requestedParam.GribIndicatorOfParameter(parm_id);
		requestedParam.GribTableVersion(targetInfo->Producer().TableVersion());

	}
	
	theParams.push_back(requestedParam);

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
	FeederInfo()->Param(requestedParam);

	if (conf->StatisticsEnabled())
	{
		initTimer->Stop();
		conf->Statistics()->AddToInitTime(initTimer->GetTime());
	}

	/*
	 * Each thread will have a copy of the target info.
	 */

	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		boost::thread* t = new boost::thread(&seaicing::Run,
											 this,
											 shared_ptr<info> (new info(*targetInfo)),
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

void seaicing::Run(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short theThreadIndex)
{
	while (AdjustLeadingDimension(myTargetInfo))
	{
		Calculate(myTargetInfo, conf, theThreadIndex);
	}
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void seaicing::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));
        
        param TParam("T-K");
        level TLevel(himan::kHeight, 2, "HEIGHT");
	param FfParam("FF-MS");  // 10 meter wind
	level FfLevel(himan::kHeight, 10, "HEIGHT");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("seaicingThread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> TInfo;
		shared_ptr<info> TgInfo;
		shared_ptr<info> FfInfo;

		try
		{
			// Source info for T
			TInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 TLevel,
								 TParam);
				
			// Source info for Tg
			TgInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 TParam);

			// Source info for FF
			FfInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 FfLevel,
								 FfParam);
				
		}
		catch (HPExceptionType& e)
		{
			//HPExceptionType t = static_cast<HPExceptionType> (e);

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
		
		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TgGrid(TgInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> FfGrid(FfInfo->Grid()->ToNewbaseGrid());

		int missingCount = 0;
		int count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() &&
							*myTargetInfo->Grid() == *TgInfo->Grid() &&
							*myTargetInfo->Grid() == *FfInfo->Grid());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		string deviceType = "CPU";
		
		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{
			count++;

			double T = kFloatMissing;
			double Tg = kFloatMissing;
			double Ff = kFloatMissing;

			InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
			InterpolateToPoint(targetGrid, TgGrid, equalGrids, Tg);
			InterpolateToPoint(targetGrid, FfGrid, equalGrids, Ff);

			if (T == kFloatMissing || Tg == kFloatMissing || Ff == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(-10);  // No missing values
				continue;
			}

			double seaIcing;
			double TBase = 273.15;

			T = T - TBase;
			Tg = Tg - TBase;

			if (Tg < -2 )
			{
				seaIcing = -10;
			}
			else
			{
				seaIcing = Ff * ( -0.35 -T ) / ( 1 + 0.3 * ( Tg + 0.35 ));

				if (seaIcing > 100)
				{
					seaIcing = 100;
				}
			}

			if (!myTargetInfo->Value(seaIcing))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

		}
                
                if (conf->StatisticsEnabled())
		{
			processTimer->Stop();
			conf->Statistics()->AddToProcessingTime(processTimer->GetTime());


			conf->Statistics()->AddToMissingCount(missingCount);
			conf->Statistics()->AddToValueCount(count);

		}

		/*
		 * Newbase normalizes scanning mode to bottom left -- if that's not what
		 * the target scanning mode is, we have to swap the data back.
		 */

		SwapTo(myTargetInfo, kBottomLeft);

		if (conf->StatisticsEnabled())
		{
			processTimer->Stop();
			conf->Statistics()->AddToProcessingTime(processTimer->GetTime());

#ifdef DEBUG
			itsLogger->Debug("Calculation took " + boost::lexical_cast<string> (processTimer->GetTime()) + " microseconds on " + deviceType);
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

			theWriter->ToFile(shared_ptr<info> (new info(*myTargetInfo)), conf);
		}
	}
}
