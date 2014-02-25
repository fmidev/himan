/**
 * @file hybrid_pressure.cpp
 *
 *  @date: Mar 23, 2013
 *  @author aaltom
 */

#include "hybrid_pressure.h"
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

hybrid_pressure::hybrid_pressure()
{
	itsClearTextFormula = "P = Vertkoord_A + P0 * Vertkoord_B";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("hybrid_pressure"));

}

void hybrid_pressure::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to P-HPA
	 * - name P-HPA
	 * - univ_id 1
	 * 
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("P-HPA", 1);

	// GRIB 2
	
	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(3);
	theRequestedParam.GribParameter(0);

	theParams.push_back(theRequestedParam);

	SetParams(theParams);

	Start();

}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void hybrid_pressure::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param PParam("P-PA");
	param QParam("Q-KGKG");
	level PLevel(himan::kHeight, 0, "HEIGHT");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("hybrid_pressureThread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));


		shared_ptr<info> PInfo;
		shared_ptr<info> QInfo;
		
		try
		{
			// Source info for P
			PInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 PLevel,
								 PParam);
			// Source info for Q
			QInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 QParam);

		
		}
		catch (HPExceptionType& e)
		{

			switch (e)
			{
			case kFileDataNotFound:
				itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
				myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value

				if (itsConfiguration->StatisticsEnabled())
				{
						itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
						itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
				}

				continue;
				break;

			default:
				throw runtime_error(ClassName() + ": Unable to proceed");
				break;
			}
		}

		unique_ptr<timer> processTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());

		if (itsConfiguration->StatisticsEnabled())
		{
			processTimer->Start();
		}

		SetAB(myTargetInfo, QInfo);

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> PGrid(PInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> QGrid(QInfo->Grid()->ToNewbaseGrid());

		size_t missingCount = 0;
		size_t count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *PInfo->Grid() && *myTargetInfo->Grid() == *QInfo->Grid());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();

		// Vertical coordinates for hybrid levels
		std::vector<double> ab = QInfo->Grid()->AB();

	    double A = ab[0];
	    double B = ab[1];
 
		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{
			count++;

			double P = kFloatMissing;
		
			InterpolateToPoint(targetGrid, PGrid, equalGrids, P);

			if (P == kFloatMissing)
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);  // No missing values
				continue;
			}

			double hybrid_pressure;

			hybrid_pressure = 0.01 * (A + P * B);

			if (!myTargetInfo->Value(hybrid_pressure))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

		}

		/*
		 * Newbase normalizes scanning mode to bottom left -- if that's not what
		 * the target scanning mode is, we have to swap the data back.
		 */

		SwapTo(myTargetInfo, kBottomLeft);

		if (itsConfiguration->StatisticsEnabled())
		{
			processTimer->Stop();
			itsConfiguration->Statistics()->AddToProcessingTime(processTimer->GetTime());
			itsConfiguration->Statistics()->AddToMissingCount(missingCount);
			itsConfiguration->Statistics()->AddToValueCount(count);

		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}
