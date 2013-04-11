/**
 * @file height.cpp
 *
 * @date Apr 5, 2013
 * @author peramaki
 */

#include "height.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <math.h>

#define HIMAN_COMPILED_INCLUDE

#include "hybrid_pressure.h"

#undef HIMAN_COMPILED_INCLUDE

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

const string itsName("height");

height::height() : itsUseCuda(false), itsCudaDeviceCount(0)
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void height::Process(std::shared_ptr<const plugin_configuration> conf)
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
	 * Set target parameter to TODO
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("H0C-M", 270);

	// GRIB 2

	theRequestedParam.GribParameter(0);

	// GRIB 1?

	// tähän GRIB 1

	theParams.push_back(theRequestedParam);

	targetInfo->Params(theParams);

	/*
	 * Create data structures.
	 */

	targetInfo->Create();


	/*
	 * Initialize parent class functions for dimension handling
	 */

	/*
	 * Each thread will have a copy of the target info.
	 */
}

void height::Run(shared_ptr<info> myTargetInfo,
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

void height::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param PParam("P-Pa"); //maanpintapaine
	param TParam("T-K"); //2m-lämpötila
	level H2(kHeight, 2);

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	shared_ptr<hybrid_pressure> itsHybridPressure = dynamic_pointer_cast<hybrid_pressure> (plugin_factory::Instance()->Plugin("hybrid_pressure"));
	shared_ptr<plugin_configuration> hybridConf(new plugin_configuration(*conf));
	hybridConf->Name("hybrid_pressure");
	hybridConf->Options(conf->Options());

	itsHybridPressure->Process(hybridConf);

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	shared_ptr<info> PInfoOld;
	shared_ptr<info> T2mInfoOld;
	shared_ptr<info> TInfoOld;

	double POld(kFloatMissing);
	//double T2mOld;
	double TOld(kFloatMissing);
	
	bool firstFetch(true);

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));


		shared_ptr<info> PInfo;
		shared_ptr<info> T2mInfo;
		shared_ptr<info> TInfo;
		try
		{
			// Source info for P
			PInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 PParam);
				
			// Source info for 2m T
			T2mInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 H2,
								 TParam);

			// Source info for Hybrid
			TInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 TParam);			

			if (firstFetch)
			{
				PInfoOld = PInfo;
				//T2mInfoOld = T2mInfo;
				TInfoOld = TInfo;
			}

		}
		catch (HPExceptionType e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					//warning vai info, tk2tc:ssä on warning, tpot, icing ja kindeks sisältää infon
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

		int missingCount = 0;
		int count = 0;

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> PGrid(PInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T2mGrid(T2mInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());

		shared_ptr<NFmiGrid> PGridOld(PInfoOld->Grid()->ToNewbaseGrid());
		//shared_ptr<NFmiGrid> T2mGridOld(T2mInfoOld->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGridOld(TInfoOld->Grid()->ToNewbaseGrid());

		bool equalGrids = ( *myTargetInfo->Grid() == *PInfo->Grid() && *myTargetInfo->Grid() == *T2mInfo->Grid() && *myTargetInfo->Grid() == *TInfo->Grid() && 
							*myTargetInfo->Grid() == *PInfoOld->Grid() && *myTargetInfo->Grid() == *TInfoOld->Grid());

		unique_ptr<timer> processTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());

		if (conf->StatisticsEnabled())
		{
			processTimer->Start();
		}

		string deviceType;


		if (itsUseCuda && equalGrids && threadIndex <= itsCudaDeviceCount)
		{
	
			deviceType = "GPU";

		}
		else
		{

			deviceType = "CPU";

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{

				count++;

				//interpolointi

				double T = kFloatMissing;
				double T2m = kFloatMissing;
				double P = kFloatMissing;

				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
				InterpolateToPoint(targetGrid, T2mGrid, equalGrids, T2m);
				InterpolateToPoint(targetGrid, PGrid, equalGrids, P);


				if (T == kFloatMissing || T2m == kFloatMissing || P == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}


				if (firstFetch)
				{
					TOld = T;
					//T2mOld = T2m;
					POld = P;
					firstFetch = false;
				}

				//laskenta
				double Tave = ( T + TOld ) /2;
				double deltaZ = (287 / 9.81) * Tave * log(POld / P);

				if (!myTargetInfo->Value(deltaZ))
				{
					throw runtime_error(ClassName() + ": Failed to set value to matrix");
				}
			}

		}

		PInfoOld = PInfo;
		T2mInfoOld = T2mInfo;
		TInfoOld = TInfo;

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