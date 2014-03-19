/**
 * @file precipitation_rate.cpp
 *
 * Template for future plugins.
 *
 * @date Mar 14, 2014
 * @author Tack
 */

#include "precipitation_rate.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("precipitation_rate");

precipitation_rate::precipitation_rate()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void precipitation_rate::Process(std::shared_ptr<const plugin_configuration> conf)
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

	// First parameter - rain
	param RPRATE("RRR-KGM2", 49);

	// GRIB 2

	RPRATE.GribDiscipline(0);
	RPRATE.GribCategory(1);
	RPRATE.GribParameter(65);
	
	theParams.push_back(RPRATE);

	// Second parameter - snow/solid precipitation
	param SPRATE("RRRS-KGM2", 200);

	// GRIB 2

	SPRATE.GribDiscipline(0);
	SPRATE.GribCategory(1);
	SPRATE.GribParameter(66);
	
	theParams.push_back(SPRATE);

	SetParams(theParams);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void precipitation_rate::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	// define quotients in formulas for rain rate and solid precipitation rate as constants
	const double rain_rate_factor = 1000.0/0.072;
	const double rain_rate_exponent = 1.0/0.880;
	const double snow_rate_factor = 1000.0/0.200;
	const double snow_rate_exponent = 1.0/0.900;

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

			if (Rho == kFloatMissing || Rain == kFloatMissing)
			{
				missingCount++;
				myTargetInfo->ParamIndex(0);
				myTargetInfo->Value(kFloatMissing);
				continue;
			}

			if (Rho == kFloatMissing || Rain == kFloatMissing || Snow == kFloatMissing || Graupel == kFloatMissing)
			{
				missingCount++;
				myTargetInfo->ParamIndex(1);
				myTargetInfo->Value(kFloatMissing);
				continue;
			}

			/*
			 * Calculations go here
			 *
			 */
			double rain_rate;
			double sprec_rate;
			
			// Calculate rain rate
			rain_rate = pow(Rho * Rain * rain_rate_factor, rain_rate_exponent);

			assert(rain_rate == rain_rate);  // Checking NaN (note: assert() is defined only in debug builds)
			
			myTargetInfo->ParamIndex(0);

			if (!myTargetInfo->Value(rain_rate))
			{
				throw runtime_error(ClassName() + ": Failed to set value to matrix");
			}

			// Calculate solid precipitation rate 
			sprec_rate = pow(Rho * (Snow + Graupel) * snow_rate_factor, snow_rate_exponent);

			assert(sprec_rate == sprec_rate); // Checking NaN (note: assert() is defined only in debug builds)

			myTargetInfo->ParamIndex(1);

			if (!myTargetInfo->Value(sprec_rate))
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
