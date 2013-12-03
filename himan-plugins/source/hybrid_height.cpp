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
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("hybrid_height");

hybrid_height::hybrid_height()
{
	itsClearTextFormula = "HEIGHT = prevH + (287/9.81) * (T+prevT)/2 * log(prevP / P)";
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void hybrid_height::Process(std::shared_ptr<const plugin_configuration> conf)
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
	 * Set target parameter to HL-M
	 * - name HL-M
	 * - univ_id 3
	 * 
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("HL-M", 3);

	// GRIB 2

	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(3);
	theRequestedParam.GribParameter(6);

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
	/*
	 * For hybrid height we must go through the levels backwards.
	 */
	FeederInfo()->LevelOrder(kBottomToTop);
	shared_ptr<neons> theNeons = dynamic_pointer_cast <neons> (plugin_factory::Instance()->Plugin("neons"));
	itsBottomLevel = boost::lexical_cast<int> (theNeons->ProducerMetaData(230, "last hybrid level number"));

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

		boost::thread* t = new boost::thread(&hybrid_height::Run,
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

void hybrid_height::Run(shared_ptr<info> myTargetInfo,
				shared_ptr<const plugin_configuration> conf,
				unsigned short threadIndex)
{

	//myTargetInfo->LevelOrder(kBottomToTop);
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

	param GPParam("P-PA");
	param PParam("P-HPA");
	param TParam("T-K");
	
	level H2(himan::kHeight, 2, "HEIGHT");
	level H0(himan::kHeight, 0, "HEIGHT");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	/*
		pitääkö tunnistaa tuottaja?
	*/
	level prevLevel;

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		bool firstLevel(false);
		
		//only works with hirlam for now
		//itsLogger->Debug("level: " 
		if ( myTargetInfo->Level().Value() == itsBottomLevel )
		{
			firstLevel = true;
		}
		else
		{
			prevLevel = level(myTargetInfo->Level());
			prevLevel.Value(myTargetInfo->Level().Value() + 1);

			prevLevel.Index(prevLevel.Index() + 1);
		}



		shared_ptr<info> PInfo;
		shared_ptr<info> TInfo;			
		shared_ptr<info> prevPInfo;
		shared_ptr<info> prevTInfo;
		shared_ptr<info> prevHInfo;

		try
		{

			forecast_time& fTime = myTargetInfo->Time();
			if (!firstLevel)
			{
				prevTInfo = FetchPrevious(conf, fTime, prevLevel, param("T-K"));
				prevPInfo = FetchPrevious(conf, fTime, prevLevel, param("P-HPA"));
				prevHInfo = FetchPrevious(conf, fTime, prevLevel, param("HL-M"));
			}
			else 
			{
				prevPInfo = FetchPrevious(conf, fTime, H0, param("P-PA"));
				prevTInfo = FetchPrevious(conf, fTime, H2, param("T-K"));
			}
			//prevLevel = myTargetInfo->Level();

			PInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 PParam);
				
			TInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
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

		assert(PInfo->Grid()->AB() == TInfo->Grid()->AB());

		SetAB(myTargetInfo, TInfo);

		size_t missingCount = 0;
		size_t count = 0;

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> PGrid(PInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> prevPGrid(prevPInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> prevTGrid(prevTInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> prevHGrid;
		
		if (!firstLevel )
			prevHGrid = shared_ptr<NFmiGrid>(prevHInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = ( *myTargetInfo->Grid() == *prevTInfo->Grid() && *myTargetInfo->Grid() == *prevPInfo->Grid() && *myTargetInfo->Grid() == *PInfo->Grid() && *myTargetInfo->Grid() == *TInfo->Grid() ); //&& *myTargetInfo->Grid() == *T2mInfo->Grid() && *myTargetInfo->Grid() == *P0mInfo->Grid() );

		if (!firstLevel)
			equalGrids = ( equalGrids && *myTargetInfo->Grid() == *prevHInfo->Grid() );

		string deviceType = "CPU";

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		prevPGrid->Reset();
		prevTGrid->Reset();
		if (!firstLevel)
			prevHGrid->Reset();

		while (	myTargetInfo->NextLocation() && 
				targetGrid->Next() && 
				prevTGrid->Next() && 
				prevPGrid->Next() )
		{

			count++;

			double T = kFloatMissing;
			double P = kFloatMissing;
			double prevP = kFloatMissing;
			double prevT = kFloatMissing;
			double prevH = kFloatMissing;

			InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
			InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
			InterpolateToPoint(targetGrid, prevPGrid, equalGrids, prevP);		
			InterpolateToPoint(targetGrid, prevTGrid, equalGrids, prevT);
		
			if (!firstLevel)
			{
				prevHGrid->Next();
				InterpolateToPoint(targetGrid, prevHGrid, equalGrids, prevH);
				
				if (prevH == kFloatMissing )
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}
			}

			if (prevT == kFloatMissing || prevP == kFloatMissing || T == kFloatMissing || P == kFloatMissing )
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);
				continue;
			}


			if (firstLevel)
			{
				prevP /= 100.f;
			}

			double Tave = ( T + prevT ) / 2;
			double deltaZ = (287 / 9.81) * Tave * log(prevP / P);

			double totalHeight(0);

			if (firstLevel)
			{
				totalHeight = deltaZ;		
			}
			else
			{	
				totalHeight = prevH + deltaZ;
			}

			if (!myTargetInfo->Value(totalHeight))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}
		
		}

		firstLevel = false;

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

shared_ptr<himan::info> hybrid_height::FetchPrevious(shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime, const level& wantedLevel, const param& wantedParam)
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
