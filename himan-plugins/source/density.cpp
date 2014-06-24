/**
 * @file density.cpp
 *
 * Computes desity for dry air from pressure and temperature using the ideal gas law.
 *
 * @date Mar 11, 2014
 * @author Tack
 */

#include "density.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include "NFmiGrid.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("density");

density::density()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void density::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to ???
	 * - name PARM_NAME
	 * - univ_id UNIV_ID
	 * - grib2 descriptor X'Y'Z
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("RHO-KGM3", 63);

	// GRIB 2

	
	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(3);
	theRequestedParam.GribParameter(10);
	
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

void density::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	/*
	 * eg. param PParam("P-Pa"); for pressure in pascals
	 *
	 */

	params PParam = { param("P-PA"), param("P-HPA") };
	param TParam("T-K");
	// ----	


	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();
	
	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
		double PScale = 1;

		shared_ptr<info> PInfo;
		shared_ptr<info> TInfo;

		shared_ptr<NFmiGrid> PGrid;
		bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

		try
		{
			if(!isPressureLevel)
			{
			// Source info for PParam and TParam
				PInfo = theFetcher->Fetch(itsConfiguration,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								PParam);

				if (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA")
				{
					PScale = 100;
				}
		
				PGrid = shared_ptr<NFmiGrid> (PInfo->Grid()->ToNewbaseGrid());
			}
			TInfo = theFetcher->Fetch(itsConfiguration,
							myTargetInfo->Time(),
							myTargetInfo->Level(),
							TParam);
			
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

		SetAB(myTargetInfo, TInfo);
		
		size_t missingCount = 0;
		size_t count = 0;

		/*
		 * Converting original grid-data to newbase grid
		 *
		 */
		SetAB(myTargetInfo, TInfo);

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = ((isPressureLevel || *myTargetInfo->Grid() == *PInfo->Grid()) && *myTargetInfo->Grid() == *TInfo->Grid());


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
				double P = kFloatMissing;
				double T = kFloatMissing;

				if (isPressureLevel)
				{
					P = 100 * myTargetInfo->Level().Value();
				}
				else
				{
				 	InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
				}

				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);

				if (P == kFloatMissing || T == kFloatMissing )
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}
				
				// actual calculation of the density using the ideal gas law
				double rho;
				
				rho = P * PScale / (constants::kRd * T);

				if (!myTargetInfo->Value(rho))
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
