/*
 * vvms.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "vvms.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "writer.h"
#include "util.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const unsigned int MAX_THREADS = 2; // Max number of threads we allow

vvms::vvms()
{
	itsClearTextFormula = "w = -(ver) * 287 * T * (9.81*p)";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("vvms"));

}

void vvms::Process(shared_ptr<configuration> theConfiguration)
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
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from theConfiguration but why bother?)
	 *
	 */

	vector<shared_ptr<param>> theParams;

	shared_ptr<param> theRequestedParam = std::shared_ptr<param> (new param("VV-MS", 143));

	theRequestedParam->GribDiscipline(0);
	theRequestedParam->GribCategory(2);
	theRequestedParam->GribParameter(9);

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

		theTargetInfos[i] = theTargetInfo->Clone();

		boost::thread* t = new boost::thread(&vvms::Run,
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

		string theOutputFile = "himan_" + theTargetInfo->Param()->Name() + "_" + theTargetInfo->Time()->OriginDateTime()->String("%Y%m%d%H");
		theWriter->ToFile(theTargetInfo, theConfiguration->OutputFileType(), false, theOutputFile);

	}
}

void vvms::Run(shared_ptr<info> myTargetInfo,
               shared_ptr<const configuration> theConfiguration,
               unsigned short theThreadIndex)
{

	while (AdjustParams(myTargetInfo))
	{
		Calculate(myTargetInfo, theConfiguration, theThreadIndex);
	}

}

bool vvms::AdjustParams(shared_ptr<info> myTargetInfo)
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

void vvms::Calculate(shared_ptr<info> myTargetInfo,
                     shared_ptr<const configuration> theConfiguration,
                     unsigned short theThreadIndex)
{


	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	shared_ptr<param> TParam (new param("T-K"));
	shared_ptr<param> PParam (new param("P-HPA"));
	shared_ptr<param> VVParam (new param("VV-PAS"));

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("vvmsThread #" + boost::lexical_cast<string> (theThreadIndex)));

	myTargetInfo->ResetLevel();
	myTargetInfo->FirstParam();

	while (myTargetInfo->NextLevel())
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time()->ValidDateTime()->String("%Y%m%d%H") +
		                        " level " + boost::lexical_cast<string> (myTargetInfo->Level()->Value()));

		myTargetInfo->Data()->Resize(theConfiguration->Ni(), theConfiguration->Nj());

		double PScale = 1;
		double TBase = 0;

		// Source info for T
		shared_ptr<info> TInfo = theFetcher->Fetch(theConfiguration,
		                         myTargetInfo->Time(),
		                         myTargetInfo->Level(),
		                         TParam);

		if (TInfo->Param()->Unit() == kC)
		{
			TBase = 273.15;
		}

		/*
		 * If vvms is calculated for pressure levels, the P value
		 * equals to level value. Otherwise we have to fetch P
		 * separately.
		 */

		shared_ptr<info> PInfo;
		shared_ptr<NFmiGrid> PGrid;

		bool isPressureLevel = (myTargetInfo->Level()->Type() == kPressure);

		if (!isPressureLevel)
		{
			// Source info for P
			PInfo = theFetcher->Fetch(theConfiguration,
			                          myTargetInfo->Time(),
			                          myTargetInfo->Level(),
			                          PParam);

			if (PInfo->Param()->Unit() == kHPa)
			{
				PScale = 100;
			}

			PGrid = PInfo->ToNewbaseGrid();
		}

		// Source info for Vertical Velocity
		shared_ptr<info> VVInfo = theFetcher->Fetch(theConfiguration,
		                          myTargetInfo->Time(),
		                          myTargetInfo->Level(),
		                          VVParam);

		shared_ptr<NFmiGrid> targetGrid = myTargetInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> TGrid = TInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> VVGrid = VVInfo->ToNewbaseGrid();

		int missingCount = 0;
		int count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		// TODO: data is not always +x+y

		// check if source area and grid == target area and grid --> no interpolation required

		bool haveCUDA = false;

		if (haveCUDA)
		{

			if (myTargetInfo->GridAndAreaEquals(TInfo) && myTargetInfo->GridAndAreaEquals(VVInfo) &&
			        (isPressureLevel || myTargetInfo->GridAndAreaEquals(PInfo)))
			{
				// doCUDA();
				// continue;
			}
			else
			{
				// itsLogger->Info("Grid definition not suitable for CUDA calculation");
			}

		}

		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{
			count++;

			double T = kFloatMissing;
			double P = kFloatMissing;
			double VV = kFloatMissing;

#ifdef NO_INTERPOLATION_WHEN_GRIDS_ARE_EQUAL

			if (gridsAreEqual)
			{
				T = TGrid->FloatValue();
				VV = VVGrid->FloatValue();

				if (isPressureLevel)
				{
					P = myTargetInfo->Level()->Value();
				}
				else
				{
					PGrid->FloatValue();
				}

			}
			else
			{
#endif
				NFmiPoint thePoint = targetGrid->LatLon();

				TGrid->InterpolateToLatLonPoint(thePoint, T);
				VVGrid->InterpolateToLatLonPoint(thePoint, VV);

				if (isPressureLevel)
				{
					P = 100 * myTargetInfo->Level()->Value();
				}
				else
				{
					PGrid->InterpolateToLatLonPoint(thePoint, P);
				}

#ifdef NO_INTERPOLATION_WHEN_GRIDS_ARE_EQUAL
			}

#endif

			if (T == kFloatMissing || P == kFloatMissing || VV == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);
				continue;
			}

			double VVms = 287 * -VV * (T + TBase) / (9.81 * (P * PScale));

			if (!myTargetInfo->Value(VVms))
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
