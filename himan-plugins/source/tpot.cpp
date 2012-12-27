/*
 * tpot.cpp
 *
 *  Created on: Nov 20, 2012
 *      Author: partio
 */

#include "tpot.h"
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

#undef CUDA

#ifdef CUDA

void tpot_cuda(double* Tin, double* Pin, double* TPout, int N);

#endif

const unsigned int MAX_THREADS = 1; // Max number of threads we allow

tpot::tpot()
{
	itsClearTextFormula = "Tp = Tk * powf((1000/P), 0.286)"; // Poissons equation

	itsLogger = logger_factory::Instance()->GetLog("tpot");

}

void tpot::Process(shared_ptr<configuration> theConfiguration)
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
	 * - name TP-K
	 * - univ_id 8
	 * - grib2 descriptor 0'00'002
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from theConfiguration but why bother?)
	 *
	 */

	vector<shared_ptr<param> > theParams;

	shared_ptr<param> theRequestedParam = shared_ptr<param> (new param("TP-K", 8));

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

		boost::thread* t = new boost::thread(&tpot::Run,
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
		theWriter->ToFile(theTargetInfo, theOutputFile, theConfiguration->OutputFileType(), false);

	}
}

void tpot::Run(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex)
{
	while (AdjustParams(myTargetInfo))
	{
		Calculate(myTargetInfo, theConfiguration, theThreadIndex);
	}

}

bool tpot::AdjustParams(shared_ptr<info> myTargetInfo)
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

void tpot::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	std::shared_ptr<param> TParam (new param("T-K"));
	std::shared_ptr<param> PParam (new param("P-HPA"));

	unique_ptr<logger> myThreadedLogger = logger_factory::Instance()->GetLog("tpotThread #" + boost::lexical_cast<string> (theThreadIndex));

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

		// Source info for P
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

			if (PInfo->Param()->Unit() == kPa)
			{
				PScale = 0.01;
			}

			PGrid = PInfo->ToNewbaseGrid();
		}

		if (TInfo->Param()->Unit() == kC)
		{
			TBase = 273.15;
		}

		shared_ptr<NFmiGrid> targetGrid = myTargetInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> TGrid = TInfo->ToNewbaseGrid();

#ifdef CUDA

		double* t = 0;
		double* p = 0;
		double* tp = 0;

		tpot_cuda(t, p, tp, 0);

#else

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
			double P = kFloatMissing;

			TGrid->InterpolateToLatLonPoint(thePoint, T);

			if (isPressureLevel)
			{
				P = myTargetInfo->Level()->Value();
			}
			else
			{
				PGrid->InterpolateToLatLonPoint(thePoint, P);
			}

			if (T == kFloatMissing || P == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);
				continue;
			}

			double Tp = (T + TBase) * pow((1000 / (P * PScale)), 0.286);

			if (!myTargetInfo->Value(Tp))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

		}

#endif

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 */

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
