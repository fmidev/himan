/**
 * @file roughness.cpp
 *
 * Template for calculation of surface roughness from HIRLAM data.
 *
 * @date Mar 27, 2014
 * @author Tack
 */

#include "roughness.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("roughness");

roughness::roughness()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void roughness::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	vector<param> theParams;
	param theRequestedParam("SR-M", 283);

	//param theRequestedParam(PARM_ NAME, UNIV_ID);

	// GRIB 2

	theRequestedParam.GribDiscipline(2);
	theRequestedParam.GribCategory(0);
	theRequestedParam.GribParameter(1);

	// GRIB 1

	/*
	 * GRIB 1 parameters go here
	 *
	 */

	theParams.push_back(theRequestedParam);

	SetParams(theParams);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void roughness::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	/*
	 * eg. param PParam("P-Pa"); for pressure in pascals
	 *
	 */

	param RoughTParam("SR-M"); // Surface roughness terrain contribution
	param RoughVParam("SRMOM-M"); // Surface roughness vegetation contribution
	// ----	


	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> RoughTInfo;
		shared_ptr<info> RoughVInfo;
		try
		{

			// Source info for RoughTParam and RoughVParam
			RoughTInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 RoughTParam);

			RoughVInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 RoughVParam);
			// ----

		}
		catch (HPExceptionType e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing);

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
		
		SetAB(myTargetInfo, RoughTInfo);

		size_t missingCount = 0;
		size_t count = 0;

		/*
		 * Converting original grid-data to newbase grid
		 *
		 */

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RoughTGrid(RoughTInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RoughVGrid(RoughVInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = (*myTargetInfo->Grid() == *RoughTInfo->Grid() && *myTargetInfo->Grid() == *RoughVInfo->Grid());


		string deviceType;

		{

			deviceType = "CPU";

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{

				count++;

				/*
				 * interpolation happens here
				 *
				 */
				double RoughT = kFloatMissing;
				double RoughV = kFloatMissing;

				InterpolateToPoint(targetGrid, RoughTGrid, equalGrids, RoughT);

				InterpolateToPoint(targetGrid, RoughVGrid, equalGrids, RoughV);

				if (RoughT == kFloatMissing || RoughV == kFloatMissing )
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				/*
				 * Calculations go here
				 *
				 */
				RoughT+=RoughV;

				if (!myTargetInfo->Value(RoughT))
				{
					throw runtime_error(ClassName() + ": Failed to set value to matrix");
				}
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

#ifdef DEBUG
			itsLogger->Debug("Calculation took " + boost::lexical_cast<string> (processTimer->GetTime()) + " microseconds on "  + deviceType);
#endif

			itsConfiguration->Statistics()->AddToMissingCount(missingCount);
			itsConfiguration->Statistics()->AddToValueCount(count);

		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places
		 * */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}
