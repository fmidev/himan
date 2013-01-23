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
#include "writer.h"
#include "pcuda.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#undef HAVE_CUDA

#ifdef HAVE_CUDA
namespace himan
{
namespace plugin
{
namespace kindex_cuda
{
void doCuda(const float* Tin, float TBase, const float* Pin, float TScale, float* TPout, size_t N, float PConst, unsigned short index);
}
}
}
#endif

kindex::kindex() : itsUseCuda(false)
{
	itsClearTextFormula = "Kindex = T850 - T500 + TD850 - ( T700 - TD700 )";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("kindex"));

}

void kindex::Process(shared_ptr<configuration> theConfiguration)
{

	shared_ptr<plugin::pcuda> c = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

	if (c && c->HaveCuda())
	{
		string msg = "I possess the powers of CUDA ";

		if (!theConfiguration->UseCuda())
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

	unsigned short threadCount = ThreadCount(theConfiguration->ThreadCount());

	boost::thread_group g;

	/*
	 * The target information is parsed from the configuration file.
	 */

	shared_ptr<info> theTargetInfo = theConfiguration->Info();

	/*
	 * Get producer information from neons if whole_file_write is false.
	 */

	if (!theConfiguration->WholeFileWrite())
	{
		shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

		map<string,string> prodInfo = n->ProducerInfo(theTargetInfo->Producer().Id());

		if (!prodInfo.empty())
		{
			producer prod(theTargetInfo->Producer().Id());

			prod.Process(boost::lexical_cast<long> (prodInfo["process"]));
			prod.Centre(boost::lexical_cast<long> (prodInfo["centre"]));
			prod.Name(prodInfo["name"]);

			theTargetInfo->Producer(prod);
		}

	}

	/*
	 * Set target parameter to Kindex
	 * - name KINDEX-N
	 * - univ_id 80
	 * 
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from theConfiguration but why bother?)
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("KINDEX-N", 80);

	theRequestedParam.GribParameter(80);
        theRequestedParam.GribTableVersion(203);

	theParams.push_back(theRequestedParam);

	theTargetInfo->Params(theParams);

	/*
	 * Create data structures.
	 */

	theTargetInfo->Create();

	/*
	 * Initialize parent class functions for dimension handling
	 */

	Dimension(theConfiguration->LeadingDimension());
	FeederInfo(theTargetInfo->Clone());
	FeederInfo()->Param(theRequestedParam);

	/*
	 * Each thread will have a copy of the target info.
	 */

	vector<shared_ptr<info> > theTargetInfos;

	theTargetInfos.resize(threadCount);

	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		theTargetInfos[i] = theTargetInfo->Clone();

		boost::thread* t = new boost::thread(&kindex::Run,
								this,
								theTargetInfos[i],
								theConfiguration,
								i + 1);

		g.add_thread(t);

	}

	g.join_all();

	if (theConfiguration->WholeFileWrite())
	{

		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		theTargetInfo->FirstTime();

		string theOutputFile = "himan_" + theTargetInfo->Param().Name() + "_" + theTargetInfo->Time().OriginDateTime()->String("%Y%m%d%H");
		theWriter->ToFile(theTargetInfo, theConfiguration->OutputFileType(), false, theOutputFile);

	}
}

void kindex::Run(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex)
{
	while (AdjustLeadingDimension(myTargetInfo))
	{
		Calculate(myTargetInfo, theConfiguration, theThreadIndex);
	}
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void kindex::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex)
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

		myTargetInfo->Data()->Resize(theConfiguration->Ni(), theConfiguration->Nj());

		shared_ptr<info> T850Info;
		shared_ptr<info> T700Info;
                shared_ptr<info> T500Info;
		shared_ptr<info> Td850Info;
                shared_ptr<info> Td700Info;

		try
		{
			// Source info for T850
			T850Info = theFetcher->Fetch(theConfiguration,
								 myTargetInfo->Time(),
								 T850Level,
								 TParam);				
			// Source info for T700
			T700Info = theFetcher->Fetch(theConfiguration,
								 myTargetInfo->Time(),
								 T700Level,
								 TParam);
			// Source info for T500
			T500Info = theFetcher->Fetch(theConfiguration,
								 myTargetInfo->Time(),
								 T500Level,
								 TParam);
                        // Source info for Td850
			Td850Info = theFetcher->Fetch(theConfiguration,
								 myTargetInfo->Time(),
								 T850Level,
								 TdParam);
                        // Source info for Td700
			Td700Info = theFetcher->Fetch(theConfiguration,
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
				continue;
				break;

			default:
				throw runtime_error(ClassName() + ": Unable to proceed");
				break;
			}
		}

		shared_ptr<NFmiGrid> targetGrid = myTargetInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> T850Grid = T850Info->ToNewbaseGrid();
		shared_ptr<NFmiGrid> T700Grid = T700Info->ToNewbaseGrid();
		shared_ptr<NFmiGrid> T500Grid = T500Info->ToNewbaseGrid();
                shared_ptr<NFmiGrid> Td850Grid = Td850Info->ToNewbaseGrid();
		shared_ptr<NFmiGrid> Td700Grid = Td700Info->ToNewbaseGrid();

		int missingCount = 0;
		int count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (myTargetInfo->GridAndAreaEquals(T850Info) &&
							myTargetInfo->GridAndAreaEquals(T700Info) &&
							myTargetInfo->GridAndAreaEquals(T500Info));

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

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
                        
                        kIndex = T850 - T500 + Td850 - (T700 - Td700);

			if (!myTargetInfo->Value(kIndex))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (!theConfiguration->WholeFileWrite())
		{
			shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

			theWriter->ToFile(myTargetInfo->Clone(), theConfiguration->OutputFileType(), true);
		}
	}
}
