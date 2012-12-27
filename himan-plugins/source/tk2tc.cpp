/*
 * tk2tc.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "tk2tc.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "util.h"
#include "writer.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const unsigned int MAX_THREADS = 2; // Max number of threads we allow

tk2tc::tk2tc()
{
	itsClearTextFormula = "Tc = Tk - 273.15";

	itsLogger = logger_factory::Instance()->GetLog("tk2tc");
}

void tk2tc::Process(std::shared_ptr<configuration> theConfiguration)
{

	// Get number of threads to use

	unsigned int theCoreCount = boost::thread::hardware_concurrency(); // Number of cores

	unsigned int theThreadCount = theCoreCount > MAX_THREADS ? MAX_THREADS : theCoreCount;

	boost::thread_group g;

	/*
	 * The target information is parsed from the configuration file.
	 */

	shared_ptr<info> theTargetInfo = theConfiguration->Info();

	/*
	 * Set target parameter to potential temperature
	 * - name T-C
	 * - univ_id 4
	 * - grib2 descriptor 0'00'000
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<shared_ptr<param>> theParams;

	shared_ptr<param> theRequestedParam = std::shared_ptr<param> (new param("T-C", 4));

	theRequestedParam->GribDiscipline(0);
	theRequestedParam->GribCategory(0);
	theRequestedParam->GribParameter(0);

	theParams.push_back(theRequestedParam);

	theTargetInfo->Params(theParams);

	/*
	 * Create data structures.
	 */

	theTargetInfo->Create();

	/*
	 * FeederInfo is used to feed the running threads.
	 */

	itsFeederInfo = theTargetInfo->Clone();

	itsFeederInfo->Reset();

	itsFeederInfo->Param(theRequestedParam);

	/*
	 * Each thread will have a copy of the target info.
	 */

	vector<shared_ptr<info> > theTargetInfos;

	theTargetInfos.resize(theThreadCount);

	for (size_t i = 0; i < theThreadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		//theTargetInfos[i] = std::shared_ptr<info> (new info(*theTargetInfo)); //theTargetInfo->Clone();
		theTargetInfos[i] = theTargetInfo->Clone();

		boost::thread* t = new boost::thread(&tk2tc::Run,
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
		string theOutputFile = "himan_" + theTargetInfo->Time()->OriginDateTime()->String("%Y%m%d%H");
		theWriter->ToFile(theTargetInfo, theOutputFile, theConfiguration->OutputFileType(), false);

	}

}

void tk2tc::Run(shared_ptr<info> myTargetInfo,
                shared_ptr<const configuration> theConfiguration,
                unsigned short theThreadIndex)
{

	while (AdjustParams(myTargetInfo))
	{
		Calculate(myTargetInfo, theConfiguration, theThreadIndex);
	}
}

bool tk2tc::AdjustParams(shared_ptr<info> myTargetInfo)
{

	boost::mutex::scoped_lock lock(itsAdjustParamMutex);

	// This function has access to the original target info

	// Leading dimension can be: time or level
	// Location cannot be, the calculations on cuda are spread on location
	// Param cannot be, since calculated params are only one

	if (1)   // say , leading_dimension == time
	{

		if (itsFeederInfo->NextTime())
		{
			myTargetInfo->Time(itsFeederInfo->Time());
		}
		else
		{
			return false;
		}
	}

	return true;
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void tk2tc::Calculate(shared_ptr<info> myTargetInfo,
                      shared_ptr<const configuration> theConfiguration,
                      unsigned short theThreadIndex)
{


	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	shared_ptr<param> TParam (new param("T-K"));

	unique_ptr<logger> myThreadedLogger = logger_factory::Instance()->GetLog("tpotThread #" + boost::lexical_cast<string> (theThreadIndex));

	myTargetInfo->ResetLevel();
	myTargetInfo->FirstParam();

	while (myTargetInfo->NextLevel())
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time()->ValidDateTime()->String("%Y%m%d%H") +
		                        " level " + boost::lexical_cast<string> (myTargetInfo->Level()->Value()));

		myTargetInfo->Data()->Resize(theConfiguration->Ni(), theConfiguration->Nj());

		// Source info for T
		shared_ptr<info> theTInfo = theFetcher->Fetch(theConfiguration,
		                            myTargetInfo->Time(),
		                            myTargetInfo->Level(),
		                            TParam);

		assert(theTInfo->Param()->Unit() == kK);

		shared_ptr<NFmiGrid> targetGrid = myTargetInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> TGrid = theTInfo->ToNewbaseGrid();

		int missingCount = 0;
		int count = 0;

		myTargetInfo->ResetLocation();

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		targetGrid->Reset();

		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{

			count++;

			NFmiPoint thePoint = targetGrid->LatLon();

			double T = kFloatMissing;

			TGrid->InterpolateToLatLonPoint(thePoint, T);

			if (T == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);
				continue;
			}

			double TC = T + 273.15;

			if (!myTargetInfo->Value(TC))
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
			shared_ptr<util> theUtil = dynamic_pointer_cast <util> (plugin_factory::Instance()->Plugin("util"));

			string outputFile = theUtil->MakeNeonsFileName(*myTargetInfo);

			theWriter->ToFile(myTargetInfo->Clone(), outputFile, theConfiguration->OutputFileType(), true);
		}
	}
}
