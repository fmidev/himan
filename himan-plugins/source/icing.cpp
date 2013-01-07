/**
 * @file icing.cpp
 *
 *  Created on: Jan 03, 2013
 *  @author aaltom
 */

#include "icing.h"
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
namespace tpot_cuda
{
void doCuda(const float* Tin, float TBase, const float* Pin, float TScale, float* TPout, size_t N, float PConst, unsigned short index);
}
}
}
#endif

const unsigned int MAX_THREADS = 1; // Max number of threads we allow

icing::icing() : itsUseCuda(false)
{
	itsClearTextFormula = "Icing = FF * ( -.35 -T2m ) / ( 1 + .3 * ( T0 + .35 ))";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("icing"));

}

void icing::Process(shared_ptr<configuration> theConfiguration)
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

	unsigned int theCoreCount = boost::thread::hardware_concurrency(); // Number of cores

	unsigned int theThreadCount = theCoreCount > MAX_THREADS ? MAX_THREADS : theCoreCount;

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

		if (prodInfo.size())
		{
			producer prod(theTargetInfo->Producer().Id());

			prod.Process(boost::lexical_cast<long> (prodInfo["process"]));
			prod.Centre(boost::lexical_cast<long> (prodInfo["centre"]));
			prod.Name(prodInfo["name"]);

			theTargetInfo->Producer(prod);
		}

	}

	/*
	 * Set target parameter to icing
	 * - name ICING-N
	 * - univ_id 480
	 * - grib2 descriptor 0'00'002
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from theConfiguration but why bother?)
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("ICING-N", 480);

	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(0);
	theRequestedParam.GribParameter(2);

	theParams.push_back(theRequestedParam);

	theTargetInfo->Params(theParams);

	/*
	 * Create data structures.
	 */

	theTargetInfo->Create();

	/*
	 * FeederInfo is used to feed the running threads.
	 */

	itsThreadManager = shared_ptr<util::thread_manager> (new util::thread_manager());

	itsThreadManager->Dimension(theConfiguration->LeadingDimension());
	itsThreadManager->FeederInfo(theTargetInfo->Clone());
	itsThreadManager->FeederInfo()->Param(theRequestedParam);

	/*
	 * Each thread will have a copy of the target info.
	 */

	vector<shared_ptr<info> > theTargetInfos;

	theTargetInfos.resize(theThreadCount);

	for (size_t i = 0; i < theThreadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		theTargetInfos[i] = theTargetInfo->Clone();

		boost::thread* t = new boost::thread(&icing::Run,
		                                     this,
		                                     theTargetInfos[i],
		                                     theConfiguration,
		                                     i + 1);

		g.add_thread(t);

	}

	g.join_all();

	itsLogger->Info("Calculation done");

	if (theConfiguration->WholeFileWrite())
	{

		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		theTargetInfo->FirstTime();

		string theOutputFile = "himan_" + theTargetInfo->Param().Name() + "_" + theTargetInfo->Time().OriginDateTime()->String("%Y%m%d%H");
		theWriter->ToFile(theTargetInfo, theConfiguration->OutputFileType(), false, theOutputFile);

	}
}

void icing::Run(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex)
{
	while (itsThreadManager->AdjustLeadingDimension(myTargetInfo))
	{
		Calculate(myTargetInfo, theConfiguration, theThreadIndex);
	}

}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void icing::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-C");
	param TgParam("TG-C");
    param FfParam("FF10-MS");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("icingThread #" + boost::lexical_cast<string> (theThreadIndex)));

	myTargetInfo->ResetLevel();
	myTargetInfo->FirstParam();

	while (itsThreadManager->AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
		                        " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		myTargetInfo->Data()->Resize(theConfiguration->Ni(), theConfiguration->Nj());

		double TBase = 0;

		// Source info for T
		shared_ptr<info> TInfo = theFetcher->Fetch(theConfiguration,
		                         myTargetInfo->Time(),
		                         myTargetInfo->Level(),
		                         TParam);
                
                // Source info for Tg
		shared_ptr<info> TgInfo = theFetcher->Fetch(theConfiguration,
		                         myTargetInfo->Time(),
		                         myTargetInfo->Level(),
		                         TgParam);

		// Source info for FF
		shared_ptr<info> FfInfo = theFetcher->Fetch(theConfiguration,
		                         myTargetInfo->Time(),
		                         myTargetInfo->Level(),
		                         FfParam);
                

		if (TInfo->Param().Unit() == kC)
		{
			TBase = 273.15;
		}

		shared_ptr<NFmiGrid> targetGrid = myTargetInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> TGrid = TInfo->ToNewbaseGrid();
        shared_ptr<NFmiGrid> TgGrid = TgInfo->ToNewbaseGrid();
        shared_ptr<NFmiGrid> FfGrid = FfInfo->ToNewbaseGrid();

		int missingCount = 0;
		int count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{
			count++;

			NFmiPoint thePoint = targetGrid->LatLon();

			double T = kFloatMissing;
			double Tg = kFloatMissing;
                        double Ff = kFloatMissing;

			TGrid->InterpolateToLatLonPoint(thePoint, T);
                        TgGrid->InterpolateToLatLonPoint(thePoint, Tg);
                        FfGrid->InterpolateToLatLonPoint(thePoint, Ff);

			if (T == kFloatMissing || Tg == kFloatMissing || Ff == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(-10);  // No missing values
                                continue;
			}
                        double Icing;
                        
                        if (Tg > -2 ) {
                          Icing = -10;  
                        }
                        else {
			  Icing = Ff * ( -.35 -T ) / ( 1 + .3 * ( Tg + 0.35 ));
                        }
                        
                        if (Icing > 100) {
                          Icing = 100;
                        }
                        
			if (!myTargetInfo->Value(Icing))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (!theConfiguration->WholeFileWrite())
		{
			shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

			theWriter->ToFile(myTargetInfo->Clone(), theConfiguration->OutputFileType(), true);
		}
	}
}
