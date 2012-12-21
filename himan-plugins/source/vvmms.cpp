/*
 * vvmms.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "vvmms.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>

#define HILPEE_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "writer.h"
#include "util.h"

#undef HILPEE_AUXILIARY_INCLUDE

using namespace std;
using namespace hilpee::plugin;

const int MAX_THREADS = 1; // Max number of threads we allow

vvmms::vvmms()
{
	itsClearTextFormula = "w = 1000 * -(ver) * 287 * T * (9.81*p)";

	itsLogger = logger_factory::Instance()->GetLog("vvmms");

}

void vvmms::Process(shared_ptr<configuration> theConfiguration)
{

	// Get number of threads to use

	int theCoreCount = boost::thread::hardware_concurrency(); // Number of cores

	unsigned short theThreadCount = theCoreCount > MAX_THREADS ? MAX_THREADS : theCoreCount;

	boost::thread_group g;

	/*
	 * The target information is parsed from the configuration file.
	 */

	shared_ptr<info> theTargetInfo = theConfiguration->Info();

	/*
	 * Set target parameter to potential temperature
	 * - name TP-K
	 * - univ_id 8
	 * - grib2 descriptor 0'00'002
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from theConfiguration but why bother?)
	 *
	 */

	vector<shared_ptr<param>> theParams;

	shared_ptr<param> theRequestedParam = std::shared_ptr<param> (new param("TP-K", 8));

	theRequestedParam->GribDiscipline(0);
	theRequestedParam->GribCategory(0);
	theRequestedParam->GribParameter(2);

	theParams.push_back(theRequestedParam);

	theTargetInfo->Params(theParams);

	/*
	 * Create data structures.
	 */

	theTargetInfo->Create();

	/*
	 * MetaTargetInfo is used to feed the running threads.
	 */

	//itsMetaTargetInfo = shared_ptr<info> (new info (*theTargetInfo));
	itsMetaTargetInfo = theTargetInfo->Clone();

	itsMetaTargetInfo->Reset();

	itsMetaTargetInfo->Param(theRequestedParam);

	/*
	 * Each thread will have a copy of the target info.
	 */

	vector<shared_ptr<info> > theTargetInfos;

	theTargetInfos.resize(theThreadCount);

	for (size_t i = 0; i < theThreadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		//theTargetInfos[i] = std::shared_ptr<info> (new info(*theTargetInfo));
		theTargetInfos[i] = theTargetInfo->Clone();

		boost::thread* t = new boost::thread(&vvmms::Run,
		                                     this,
		                                     theTargetInfos[i],
		                                     *theConfiguration,
		                                     i + 1);

		g.add_thread(t);

	}

	g.join_all();

	itsLogger->Info("Calculation done");

	if (theConfiguration->WholeFileWrite())
	{

		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		theTargetInfo->FirstTime();

		string theOutputFile = "Hilpee_" + theTargetInfo->Param()->Name() + "_" + theTargetInfo->Time()->OriginDateTime()->String("%Y%m%d%H");
		theWriter->ToFile(theTargetInfo, theOutputFile, theConfiguration->OutputFileType(), false);

	}
}

void vvmms::Run(shared_ptr<info> myTargetInfo,
               const configuration& theConfiguration,
               unsigned short theThreadIndex)
{

	while (AdjustParams(myTargetInfo))
	{
		Calculate(myTargetInfo, theConfiguration, theThreadIndex);

	}

}

bool vvmms::AdjustParams(shared_ptr<info> myTargetInfo)
{

	boost::mutex::scoped_lock lock(itsMetaMutex);

	// This function has access to the original target info

	// Leading dimension can be: time or level
	// Location cannot be, the calculations on cuda are spread on location
	// Param cannot be, since calculated params are only one

	if (1)   // say , leading_dimension == time
	{

		if (itsMetaTargetInfo->NextTime())
		{
			myTargetInfo->Time(itsMetaTargetInfo->Time());
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

void vvmms::Calculate(shared_ptr<info> myTargetInfo,
                     const configuration& theConfiguration,
                     unsigned short theThreadIndex)
{


	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param theT ("T-K");
	param theP ("P-HPA");
	param theVV ("VV-PAS");

	unique_ptr<logger> myThreadedLogger = logger_factory::Instance()->GetLog("vvmmsThread #" + boost::lexical_cast<string> (theThreadIndex));

	myTargetInfo->ResetLevel();
	myTargetInfo->FirstParam();

	while (myTargetInfo->NextLevel())
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time()->ValidDateTime()->String("%Y%m%d%H") +
		                        " level " + boost::lexical_cast<string> (myTargetInfo->Level()->Value()));

		myTargetInfo->Data()->Resize(theConfiguration.Ni(), theConfiguration.Nj());

		double PScale = 1;
		double TBase = 0;

		// Source info for T
		shared_ptr<info> TInfo = theFetcher->Fetch(theConfiguration,
		                            *myTargetInfo->Time(),
		                            *myTargetInfo->Level(),
		                            theT);

		if (TInfo->Param()->Unit() == kC)
		{
			TBase = 273.15;
		}

		/*
		 * If vvmms is calculated for pressure levels, the P value
		 * equals to level value. Otherwise we have to fetch P
		 * separately.
		 */

		shared_ptr<info> PInfo;
		shared_ptr<NFmiGrid> PGrid;

		bool isPressureLevel = myTargetInfo->Level()->Type() == kPressure;

		if (!isPressureLevel)
		{
			// Source info for P
			PInfo = theFetcher->Fetch(theConfiguration,
		                          *myTargetInfo->Time(),
		                          *myTargetInfo->Level(),
		                          theP);

			if (PInfo->Param()->Unit() == kPa)
			{
				PScale = 0.01;
			}

			shared_ptr<NFmiGrid> PGrid = PInfo->ToNewbaseGrid();
		}

		// Source info for Vertical Velocity
		shared_ptr<info> VVInfo = theFetcher->Fetch(theConfiguration,
				                            *myTargetInfo->Time(),
				                            *myTargetInfo->Level(),
				                            theVV);

		shared_ptr<NFmiGrid> targetGrid = myTargetInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> TGrid = TInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> VVGrid = VVInfo->ToNewbaseGrid();

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
			double P = kFloatMissing;
			double VV = kFloatMissing;

			TGrid->InterpolateToLatLonPoint(thePoint, T);

			if (isPressureLevel)
			{
				P = myTargetInfo->Level()->Value();
			}
			else
			{
				PGrid->InterpolateToLatLonPoint(thePoint, P);
			}

			VVGrid->InterpolateToLatLonPoint(thePoint, VV);

			if (T == kFloatMissing || P == kFloatMissing || VV == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);
				continue;
			}

			//double Tp = (T + TBase) * powf((1000 / (P * PScale)), 0.286);
			double VVmms = 287 * -VV * (T+TBase) / (9.81 * (P*PScale));

			if (!myTargetInfo->Value(VVmms))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

		}

		/*
		 * Now we are done for this level
		 *
		 * If output file type is GRIB, we can write individual time/level combination
		 * to file.
		 *
		 * TODO: Should we clone myTargetInfo to prevent writer modifying our version of info ?
		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (!theConfiguration.WholeFileWrite())
		{

			shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));
			shared_ptr<util> theUtil = dynamic_pointer_cast <util> (plugin_factory::Instance()->Plugin("util"));

			string outputFile = theUtil->MakeNeonsFileName(*myTargetInfo);

			theWriter->ToFile(myTargetInfo, outputFile, theConfiguration.OutputFileType(), true);
		}
	}
}
