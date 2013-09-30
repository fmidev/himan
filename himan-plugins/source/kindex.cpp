/**
 * @file kindex.cpp
 *
 *  @date: Jan 23, 2013
 *  @author aaltom
 */

#include "kindex.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

kindex::kindex()
{
	itsClearTextFormula = "Kindex = T850 - T500 + TD850 - ( T700 - TD700 )";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("kindex"));

}

void kindex::Process(std::shared_ptr<const plugin_configuration> conf)
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
	 * Set target parameter to Kindex
	 * - name KINDEX-N
	 * - univ_id 80
	 * 
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from conf but why bother?)
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("KINDEX-N", 80);

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

		boost::thread* t = new boost::thread(&kindex::Run,
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

void kindex::Run(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short theThreadIndex)
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

void kindex::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	param TdParam("TD-C");  

	level T850Level(himan::kPressure, 850, "PRESSURE");
	level T700Level(himan::kPressure, 700, "PRESSURE");
	level T500Level(himan::kPressure, 500, "PRESSURE");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("kindexThread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		//myTargetInfo->Data()->Resize(conf->Ni(), conf->Nj());

		shared_ptr<info> T850Info;
		shared_ptr<info> T700Info;
		shared_ptr<info> T500Info;
		shared_ptr<info> Td850Info;
		shared_ptr<info> Td700Info;

		try
		{
			// Source info for T850
			T850Info = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 T850Level,
								 TParam);				
			// Source info for T700
			T700Info = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 T700Level,
								 TParam);
			// Source info for T500
			T500Info = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 T500Level,
								 TParam);
			// Source info for Td850
			Td850Info = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 T850Level,
								 TdParam);
			// Source info for Td700
			Td700Info = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 T700Level,
								 TdParam);
				
		}
		catch (HPExceptionType& e)
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

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T850Grid(T850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T700Grid(T700Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T500Grid(T500Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> Td850Grid(Td850Info->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> Td700Grid(Td700Info->Grid()->ToNewbaseGrid());

		size_t missingCount = 0;
		size_t count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *T850Info->Grid() &&
							*myTargetInfo->Grid() == *T700Info->Grid() &&
							*myTargetInfo->Grid() == *T500Info->Grid() &&
							*myTargetInfo->Grid() == *Td850Info->Grid() &&
							*myTargetInfo->Grid() == *Td700Info->Grid());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		string deviceType = "CPU";

		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{
			count++;

			double T850 = kFloatMissing;
			double T700 = kFloatMissing;
			double T500 = kFloatMissing;
			double Td850 = kFloatMissing;
			double Td700 = kFloatMissing;

			InterpolateToPoint(targetGrid, T850Grid, equalGrids, T850);
			InterpolateToPoint(targetGrid, T700Grid, equalGrids, T700);
			InterpolateToPoint(targetGrid, T500Grid, equalGrids, T500);
			InterpolateToPoint(targetGrid, Td850Grid, equalGrids, Td850);
			InterpolateToPoint(targetGrid, Td700Grid, equalGrids, Td700);


			if (T850 == kFloatMissing || T700 == kFloatMissing || T500 == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);  // No missing values
				continue;
			}

			double kIndex;
			double TBase = 273.15;

			T850 = T850 - TBase;
			T700 = T700 - TBase;
			T500 = T500 - TBase;
			Td850 = Td850 - TBase;
			Td700 = Td700 - TBase;

			kIndex = T850 - T500 + Td850 - (T700 - Td700);

			if (!myTargetInfo->Value(kIndex))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

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

		if (conf->FileWriteOption() != kSingleFile)
		{
			WriteToFile(conf, myTargetInfo);
		}
	}
}
