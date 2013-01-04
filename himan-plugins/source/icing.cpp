/*
 * icing.cpp
 *
 *  Created on: Jan 03, 2013
 *      Author: aaltom
 */

#include "icing.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "writer.h"
#include "util.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#undef CUDA

#ifdef CUDA

void icing_cuda(const float* Tin, const float* Pin, float* TPout, int N);

#endif

const unsigned int MAX_THREADS = 1; // Max number of threads we allow

icing::icing()
{
	itsClearTextFormula = "Icing = FF * ( -.35 -T2m ) / ( 1 + .3 * ( T0 + .35 ))";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("icing"));

}

void icing::Process(shared_ptr<configuration> theConfiguration)
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

	vector<shared_ptr<param> > theParams;

	shared_ptr<param> theRequestedParam = shared_ptr<param> (new param("ICING-N", 480));

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

		string theOutputFile = "himan_" + theTargetInfo->Param()->Name() + "_" + theTargetInfo->Time()->OriginDateTime()->String("%Y%m%d%H");
		theWriter->ToFile(theTargetInfo, theConfiguration->OutputFileType(), false, theOutputFile);

	}
}

void icing::Run(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex)
{
	while (AdjustParams(myTargetInfo))
	{
		Calculate(myTargetInfo, theConfiguration, theThreadIndex);
	}

}

bool icing::AdjustParams(shared_ptr<info> myTargetInfo)
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

void icing::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const configuration> theConfiguration, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	std::shared_ptr<param> TParam (new param("T-C"));
	std::shared_ptr<param> TgParam (new param("TG-C"));
        std::shared_ptr<param> FfParam (new param("FF10-MS"));

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("icingThread #" + boost::lexical_cast<string> (theThreadIndex)));

	myTargetInfo->ResetLevel();
	myTargetInfo->FirstParam();

	while (myTargetInfo->NextLevel())
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time()->ValidDateTime()->String("%Y%m%d%H") +
		                        " level " + boost::lexical_cast<string> (myTargetInfo->Level()->Value()));

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
                

		if (TInfo->Param()->Unit() == kC)
		{
			TBase = 273.15;
		}

		shared_ptr<NFmiGrid> targetGrid = myTargetInfo->ToNewbaseGrid();
		shared_ptr<NFmiGrid> TGrid = TInfo->ToNewbaseGrid();
                shared_ptr<NFmiGrid> TgGrid = TgInfo->ToNewbaseGrid(); 
                shared_ptr<NFmiGrid> FfGrid = FfInfo->ToNewbaseGrid(); 

		int missingCount = 0;
		int count = 0;

#ifdef CUDA

		size_t size = targetGrid->Size();

		// cuda works on float

		const float* t = TGrid->DataPool()->Data();
		const float* p = new float[size]; // Assume pressure level data
		float* tp = new float[size];

		icing_cuda(t, p, tp, size);

		double *data = new double[size];

		for (size_t i = 0; i < size; i++)
		{
			data[i] = static_cast<float> (tp[i]);

			if (data[i] == kFloatMissing)
			{
				missingCount++;
			}

			count++;
		}

		myTargetInfo->Data()->Data(data, size);

		delete [] tp;
		delete [] p;
		delete [] data;

#else

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

#endif

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
