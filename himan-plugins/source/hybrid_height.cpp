/**
 * @file hybrid_height.cpp
 *
 * @date Apr 5, 2013
 * @author peramaki
 */

#include "hybrid_height.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <math.h>

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

const string itsName("hybrid_height");

hybrid_height::hybrid_height() : itsUseCuda(false), itsCudaDeviceCount(0)
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void hybrid_height::Process(std::shared_ptr<const plugin_configuration> conf)
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
	 * Set target parameter to H0C-M
	 * - name H0C-M
	 * - univ_id 270
	 * 
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("H0C-M", 270);

	// GRIB 2

	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(3);
	theRequestedParam.GribParameter(13);

	// GRIB 1?


	if (conf->OutputFileType() == kGRIB1)
	{
		shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		long parm_id = n->NeonsDB().GetGridParameterId(targetInfo->Producer().TableVersion(), theRequestedParam.Name());
		theRequestedParam.GribIndicatorOfParameter(parm_id);
		theRequestedParam.GribTableVersion(targetInfo->Producer().TableVersion());

	}

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

		boost::thread* t = new boost::thread(&hybrid_height::Run,
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

void hybrid_height::Run(shared_ptr<info> myTargetInfo,
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

void hybrid_height::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param PParam("P-Pa"); //maanpintapaine
	param TParam("T-K"); //lämpötila
	//level H2(kHeight, 2);

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	//shared_ptr<info> PInfoPrevious;
	//shared_ptr<info> T2mInfoPrevious;
	//shared_ptr<info> TInfoPrevious;

	double PPrevious(kFloatMissing);
	//double T2mPrevious;
	double TPrevious(kFloatMissing);
	
	bool firstFetch(true);

	double TotalHeight(0);

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));


		shared_ptr<info> PInfo;
		//shared_ptr<info> T2mInfo;
		shared_ptr<info> TInfo;
		try
		{
			// Source info for P
			PInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 PParam);
				
			/* Source info for 2m T
			T2mInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 H2,
								 TParam);
			*/
			// Source info for Hybrid
			TInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 TParam);			

			/*if (firstFetch)
			{
				PInfoPrevious = PInfo;
				//T2mInfoPrevious = T2mInfo;
				TInfoPrevious = TInfo;
			}*/

		}
		catch (HPExceptionType e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					//warning vai info, tk2twc:ssä on warning, tpot, icing ja kindeks sisältää infon
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
		//shared_ptr<NFmiGrid> T2mGrid(T2mInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());

		//shared_ptr<NFmiGrid> PGridPrevious(PInfoPrevious->Grid()->ToNewbaseGrid());
		//shared_ptr<NFmiGrid> T2mGridPrevious(T2mInfoPrevious->Grid()->ToNewbaseGrid());
		//shared_ptr<NFmiGrid> TGridPrevious(TInfoPrevious->Grid()->ToNewbaseGrid());

		bool equalGrids = ( *myTargetInfo->Grid() == *PInfo->Grid() && *myTargetInfo->Grid() == *TInfo->Grid() );
							//*myTargetInfo->Grid() == *PInfoPrevious->Grid() && *myTargetInfo->Grid() == *TInfoPrevious->Grid());

		unique_ptr<timer> processTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());

		if (conf->StatisticsEnabled())
		{
			processTimer->Start();
		}

		string deviceType;


		/*
		if (itsUseCuda && equalGrids && threadIndex <= itsCudaDeviceCount)
		{
	
			deviceType = "GPU";

		}
		else
		{
		*/
		deviceType = "CPU";

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{

			count++;

			double T = kFloatMissing;
			//double T2m = kFloatMissing;
			double P = kFloatMissing;

			InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
			//InterpolateToPoint(targetGrid, T2mGrid, equalGrids, T2m);
			InterpolateToPoint(targetGrid, PGrid, equalGrids, P);


			if (T == kFloatMissing || P == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);
				continue;
			}


			if (firstFetch)
			{
				TPrevious = T;
				//T2mPrevious = T2m;
				PPrevious = P;
				firstFetch = false;
			}

			//laskenta
			double Tave = ( T + TPrevious ) /2;
			double deltaZ = (287 / 9.81) * Tave * log(PPrevious / P);

			TotalHeight += deltaZ;

			if (!myTargetInfo->Value(TotalHeight))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}
			TPrevious = T;
			PPrevious = P;
		
		}

		//} cuda

		//PInfoPrevious = PInfo;
		//T2mInfoPrevious = T2mInfo;
		//TInfoPrevious = TInfo;

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