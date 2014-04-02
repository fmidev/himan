/**
 * @file absolute_humidity.cpp
 *
 * Plug-in to calculate total humidity
 *
 * @date Mar 27, 2014
 * @author Tack
 */

#include "absolute_humidity.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("absolute_humidity");

absolute_humidity::absolute_humidity()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void absolute_humidity::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	vector<param> theParams;

	// First parameter - absolute humidity
	param ABSH("ABSH-KGM3", 1192);

	// GRIB 2

	ABSH.GribDiscipline(0);
	ABSH.GribCategory(1);
	ABSH.GribParameter(18);
	
	theParams.push_back(ABSH);

	SetParams(theParams);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void absolute_humidity::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters (Density from plug-in density; rain, snow and graupel from Harmonie model output)

	param RhoParam("RHO-KGM3");	// Density in kg/m3
	param RainParam("RRI-KGM2");	// Large Scale precipitation in kg/m2
	param SnowParam("SNRI-KGM2");	// Large scale snow accumulation in kg/m2
	param GraupelParam("GRI-KGM2");	// Graupel precipitation in kg/m2
	// ----	


	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> RhoInfo;
		shared_ptr<info> RainInfo;
		shared_ptr<info> SnowInfo;
		shared_ptr<info> GraupelInfo;
		try
		{

			// Source info for RhoParam
			RhoInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 RhoParam);
			
			// Source info for RainParam
			RainInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 RainParam);
			
			// Source info for SnowParam
			SnowInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 SnowParam);
			
			// Source info for GraupelParam
			GraupelInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 GraupelParam);
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

		SetAB(myTargetInfo, RhoInfo);
		
		int missingCount = 0;
		int count = 0;

		/*
		 * Converting original grid-data to newbase grid
		 *
		 */

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RhoGrid(RhoInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RainGrid(RainInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> SnowGrid(SnowInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> GraupelGrid(GraupelInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = (*myTargetInfo->Grid() == *RhoInfo->Grid() && *myTargetInfo->Grid() == *RainInfo->Grid() &&
					 *myTargetInfo->Grid() == *SnowInfo->Grid() &&*myTargetInfo->Grid() == *GraupelInfo->Grid());

		string deviceType;

		// Calculate on CPU
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
			double Rho = kFloatMissing;
			double Rain = kFloatMissing;
			double Snow = kFloatMissing;
			double Graupel = kFloatMissing;

			InterpolateToPoint(targetGrid, RhoGrid, equalGrids, Rho);
			InterpolateToPoint(targetGrid, RainGrid, equalGrids, Rain);
			InterpolateToPoint(targetGrid, SnowGrid, equalGrids, Snow);
			InterpolateToPoint(targetGrid, GraupelGrid, equalGrids, Graupel);

			// Check if mixing ratio for rain is not missing
			if (Rho == kFloatMissing || Rain == kFloatMissing || Snow == kFloatMissing || Graupel == kFloatMissing)
			{
				missingCount++;
				myTargetInfo->Value(kFloatMissing);

				continue;

			} 
		
			// Calculate absolute humidity if mixing ratio is not missing. If mixing ratio is negative use 0.0 kg/kg instead.
			double absolute_humidity;
			absolute_humidity = Rho * fmax((Rain + Snow + Graupel), 0.0);

			if (!myTargetInfo->Value(absolute_humidity))
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
