/**
 * @file fmi_weather_symbol_1.cpp
 *
 *  @date: May, 2014
 *  @author Andreas Tack
 */

#include "fmi_weather_symbol_1.h"
#include <iostream>
#include <map>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"
#include "metutil.h"
#include "NFmiGrid.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

fmi_weather_symbol_1::fmi_weather_symbol_1()
{
	itsClearTextFormula = "fmi_weather_symbol_1 = ";
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("fmi_weather_symbol_1"));
}

void fmi_weather_symbol_1::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	vector<param> theParams;

	param theRequestedParam("ILSAA1-N", 350);

	theParams.push_back(theRequestedParam);

	SetParams(theParams);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void fmi_weather_symbol_1::Calculate(shared_ptr<info> myTargetInfo, unsigned short theThreadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters
	// new parameters used...
	param PrecformParam("PRECFORM-N");
	param TotalPrecParam("RRR-KGM2");
	param TotalCloudCoverParam("N-0TO1");
	param LowCloudCoverParam("NL-PRCNT");
	param MedCloudCoverParam("NM-PRCNT");
	param HighCloudCoverParam("NH-PRCNT");
	param FogParam("FOGSYM-N");
	param CloudParam("CLDSYM-N");

	// parameters to check convection
	const param TParam("T-K");
	const param KParam("KINDEX-N");

	level HLevel(himan::kHeight, 0, "HEIGHT");
	
	// levels to check convection
	level T0mLevel(himan::kHeight, 0, "HEIGHT");
	level RH850Level(himan::kPressure, 850, "PRESSURE");


	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("fmi_weather_symbol_1Thread #" + boost::lexical_cast<string> (theThreadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		//myTargetInfo->Data()->Resize(conf->Ni(), conf->Nj());

		shared_ptr<info> CloudInfo;
		shared_ptr<info> PrecformInfo;
		shared_ptr<info> TotalPrecInfo;
		shared_ptr<info> TotalCloudCoverInfo;
		shared_ptr<info> LowCloudCoverInfo;
		shared_ptr<info> MedCloudCoverInfo;
		shared_ptr<info> HighCloudCoverInfo;
		shared_ptr<info> FogInfo;

		// convection Infos
		shared_ptr<info> T0mInfo;
		shared_ptr<info> T850Info;
		shared_ptr<info> KInfo;

		try
		{
			// Source infos
			PrecformInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 PrecformParam);
			TotalPrecInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 TotalPrecParam);
			TotalCloudCoverInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 TotalCloudCoverParam);
			LowCloudCoverInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 LowCloudCoverParam);
			MedCloudCoverInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 MedCloudCoverParam);
			HighCloudCoverInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 HighCloudCoverParam);
			FogInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 FogParam);				


			// convection infos
			// Source info for T0m
			T0mInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 T0mLevel,
								 TParam);
			// Source info for T850
			T850Info = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 RH850Level,
								 TParam);
			// Source info for kIndex
			KInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 KParam);
			// for thunder
			CloudInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 CloudParam);
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

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> PrecformGrid(PrecformInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TotalPrecGrid(TotalPrecInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TotalCloudCoverGrid(TotalCloudCoverInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> LowCloudCoverGrid(LowCloudCoverInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> MedCloudCoverGrid(MedCloudCoverInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> HighCloudCoverGrid(HighCloudCoverInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> FogGrid(FogInfo->Grid()->ToNewbaseGrid());

		// convection info to grid
		shared_ptr<NFmiGrid> T0mGrid(T0mInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> KGrid(KInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> T850Grid(T850Info->Grid()->ToNewbaseGrid());
		// for thunder
		shared_ptr<NFmiGrid> CloudGrid(CloudInfo->Grid()->ToNewbaseGrid());

		size_t missingCount = 0;
		size_t count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *PrecformInfo->Grid() &&
				   *myTargetInfo->Grid() == *CloudInfo->Grid() &&
				   *myTargetInfo->Grid() == *TotalPrecInfo->Grid() &&
				   *myTargetInfo->Grid() == *TotalCloudCoverInfo->Grid() &&
				   *myTargetInfo->Grid() == *LowCloudCoverInfo->Grid() &&
				   *myTargetInfo->Grid() == *MedCloudCoverInfo->Grid() &&
				   *myTargetInfo->Grid() == *FogInfo->Grid());

		myTargetInfo->ResetLocation();

		targetGrid->Reset();


		string deviceType = "CPU";

		while (myTargetInfo->NextLocation() && targetGrid->Next())
		{
			count++;

			double precForm = kFloatMissing;
			double totalPrec = kFloatMissing;
			double totalCC = kFloatMissing;
			double lowCC = kFloatMissing;
			double medCC = kFloatMissing;
			double highCC = kFloatMissing;
			double fog = kFloatMissing;

			// convection input parameters
			double T0m = kFloatMissing;
			double kIndex = kFloatMissing;
			double T850 = kFloatMissing;

			// for thunder
			double cloud = kFloatMissing;

			// derived parameters
			double precType, fogIntensity, thunderProb;

			// output parameter
			double weather_symbol;
			
			InterpolateToPoint(targetGrid, PrecformGrid, equalGrids, precForm);
			InterpolateToPoint(targetGrid, TotalPrecGrid, equalGrids, totalPrec);
			InterpolateToPoint(targetGrid, TotalCloudCoverGrid, equalGrids, totalCC);
			InterpolateToPoint(targetGrid, LowCloudCoverGrid, equalGrids, lowCC);
			InterpolateToPoint(targetGrid, MedCloudCoverGrid, equalGrids, medCC);
			InterpolateToPoint(targetGrid, HighCloudCoverGrid, equalGrids, highCC);
			InterpolateToPoint(targetGrid, FogGrid, equalGrids, fog);

			// interpolate convection parameters
			InterpolateToPoint(targetGrid, T0mGrid, equalGrids, T0m);
			InterpolateToPoint(targetGrid, KGrid, equalGrids, kIndex);
			InterpolateToPoint(targetGrid, T850Grid, equalGrids, T850);

			// for thunder
			InterpolateToPoint(targetGrid, CloudGrid, equalGrids, cloud);

			if (cloud == kFloatMissing || precForm == kFloatMissing || totalPrec == kFloatMissing || totalCC == kFloatMissing || lowCC == kFloatMissing || medCC == kFloatMissing || highCC == kFloatMissing || fog == kFloatMissing )
			{
				missingCount++;

				myTargetInfo->Value(kFloatMissing);  // No missing values
				continue;
			}


			weather_symbol = 0;
			precType = rain_type(kIndex, T0m, T850); // calculate rain type index
			fogIntensity = ( fog == 607 ? 1 : 0 ); // calculate fog intesity index
			thunderProb = thunder_prob(kIndex,cloud); //calculate thunder probability

			// weather algorithm goes here

			// precipitation below 0.02 mm/h is assumed dry weather
			if ( totalPrec <= 0.02 )
			{
				// !!!Logical error, weather_symbol = 10 will always be overwritten in the else clause (in smartmet script, fixed here by else if clause)!!!
				if      ( fogIntensity == 1 ) weather_symbol = 10; // mist: two horizontal lines
				else if ( fogIntensity == 2 ) weather_symbol = 40; // fog: three horizontal lines	
				else
				{
					// Total cloud cover less than 11%
					if (totalCC <= 0.1) weather_symbol = 0; // clear: sun (moon and/or stars)
					// !!!Attention, N-0TO1 is not percent. It's between 0 and 1!!!
					else if ((lowCC + medCC) <= 10 && totalCC > 0.10) weather_symbol = 1; // thin high clouds: the sun and transparent cloud
					else if ((lowCC + medCC) > 10 && totalCC > 0.10 && totalCC <= 0.30) weather_symbol = 2; // almost clear: the sun and the (small) light cloud
					else if ((lowCC + medCC) > 10 && totalCC > 0.30 && totalCC <= 0.60) weather_symbol = 3; // Partly cloudy: the sun and the light cloud
					else if ((lowCC + medCC) > 10 && totalCC > 0.60 && totalCC <= 0.80) weather_symbol = 4; // almost overcast: the sun and the (large) light cloud
					else if ((lowCC + medCC) > 10 && totalCC > 0.80) weather_symbol = 5; // Cloudy: light cloud
				}
			}
			// rainfall of > 0.02 mm/h
			else if ( precForm == 0 ) // drizzling rain
			{
				if ( totalPrec > 0.02 && totalPrec <= 0.03) weather_symbol = 51; // weak drizzle: for example a dark cloud, and one rain drop
				else if ( totalPrec > 0.03 ) weather_symbol = 53; // drizzle: for example a dark cloud, and two rain drops
			}
			else if ( precForm == 1 ) // rainfall
			{
				// logical error for prectype == 2 will always give thunderstorm
				if ( totalPrec <= 0.4 && precType == 1) weather_symbol = 61; // slight rain: a dark cloud and one rain drop
				else if ( totalPrec > 0.4 && totalPrec <= 2.0 && precType == 1) weather_symbol = 63; // moderate rain: a dark cloud, and two rain drops
				else if ( totalPrec > 2.0 && precType == 1) weather_symbol = 65; // heavy rain: a dark cloud, and three rain drops
				else if ( totalPrec <= 0.4 && precType == 2) weather_symbol = 80; // Poor water: deaf Sun and half dark cloud droplets and lines
				else if ( totalPrec > 0.4 && totalPrec <= 7.0 && precType == 2) weather_symbol = 81; // moderate to hard water deaf: a dark cloud droplets and lines +
				else if ( totalPrec > 7.0 && precType == 2) weather_symbol = 82; // very heavy rain showers: CB cloud droplets and lines +
				/*
				 * Thunder so far determined by using the probability of thunder
				 * weak thunder when tn < 20%
				 * moderate thunder when the probability is 20-50%
				 * loud thunder when t > 50%
				 */
				if ( precType == 2 && thunderProb > 20 && thunderProb <= 50 ) weather_symbol = 95; // slight to moderate thunderstorms: the sun and the cb-cloud + lightning
				else if ( precType == 2 && thunderProb > 50 ) weather_symbol = 97; // heavy thunderstorms: cb-cloud + two lightning
			}
			else if ( precForm == 2 ) // sleet
			{
				if ( totalPrec <= 0.4 && precType == 1 ) weather_symbol = 68; // weak rain and snow: a dark cloud and flake + rain drop
				else if ( totalPrec > 0.4 && precType == 1 ) weather_symbol = 69; // moderate or heavy rain and snow: a dark cloud and two flakes + drops
				else if ( totalPrec <= 0.4 && precType == 2 ) weather_symbol = 83; // weak sleet showers: the sun and the dark side of the cloud and flake + drop
				else if ( totalPrec > 0.4 && precType == 2 ) weather_symbol = 84; // moderate or severe sleet showers: a dark cloud and 2x flake + drop
				/*
				 * Thunder so far determined by using the probability of thunder
				 * weak thunder when tn < 20%
				 * moderate thunder when the probability is 20-50%
				 * loud thunder when t > 50%
				 */
				if ( precType == 2 && thunderProb > 20 && thunderProb <= 50 ) weather_symbol = 96; // slight to moderate thunderstorms: the sun and the cb-cloud + lightning
				else if ( precType == 2 && thunderProb > 50 ) weather_symbol = 99; // heavy thunderstorms: cb-cloud + two lightning
			}
			else if ( precForm == 3 ) // snowfall
			{
				if ( totalPrec <= 0.4 && precType == 1 ) weather_symbol = 71; // light snow: a dark cloud and snowflakes
				else if ( totalPrec > 0.4 && totalPrec <= 1.5 && precType == 1 ) weather_symbol = 73; // moderate snow showers: a dark cloud, and 2 flakes
				else if ( totalPrec > 1.5 && precType == 1 ) weather_symbol = 75; // dense Snow: a dark cloud, and 3 flakes
				else if ( totalPrec <= 0.4 && precType == 2 ) weather_symbol = 85; // weak snow showers: the sun and the dark side of the cloud and lines + flakes
				else if ( totalPrec > 0.4 && precType == 2 ) weather_symbol = 73; // moderate or severe snow showers: a dark cloud lines and flakes
				/*
				 * Thunder so far determined by using the probability of thunder
				 * weak thunder when tn < 20%
				 * moderate thunder when the probability is 20-50%
				 * loud thunder when t > 50%
				 */
				if ( precType == 2 && thunderProb > 20 && thunderProb <= 50 ) weather_symbol = 96; // slight to moderate thunderstorms: the sun and the cb-cloud + lightning
				else if ( precType == 2 && thunderProb > 50 ) weather_symbol = 99; // heavy thunderstorms: cb-cloud + two lightning

			}
			else if ( precForm == 4 ) // freezing drizzle rain
			{
				if ( totalPrec > 0.02 && totalPrec <= 0.03 ) weather_symbol = 56; // Light Freezing Drizzle: for example, a dark cloud, and the dotted curved line
				else if ( totalPrec > 0.03 ) weather_symbol = 57; // freezing drizzle: for example, a dark cloud, and two curved line in a comma
			}
			else if ( precForm == 5 ) // sleet
			{
				if ( totalPrec <= 0.4 ) weather_symbol = 66; // Light Freezing Rain: a dark cloud and a drop of the arc line
				else if ( totalPrec > 0.4 ) weather_symbol = 67; // moderate or heavy freezing rain: a dark cloud, and two drops of the arc line
				/*
				 * Thunder so far determined by using the probability of thunder
				 * weak thunder when tn < 20%
				 * moderate thunder when the probability is 20-50%
				 * loud thunder when t > 50%
				 */
				if ( precType == 2 && thunderProb > 20 && thunderProb <= 50 ) weather_symbol = 96; // slight to moderate thunderstorms: the sun and the cb-cloud + lightning
				else if ( precType == 2 && thunderProb > 50 ) weather_symbol = 99; // heavy thunderstorms: cb-cloud + two lightning

			}
			else if ( precForm == 6 || precForm == 7 ) // hail
			{
				/*
				 * Thunder so far determined by using the probability of thunder
				 * weak thunder when tn < 20%
				 * moderate thunder when the probability is 20-50%
				 * loud thunder when t > 50%
				 */
				if ( precType == 2 && thunderProb > 20 && thunderProb <= 50 ) weather_symbol = 96; // slight to moderate thunderstorms: the sun and the cb-cloud + lightning
				else if ( precType == 2 && thunderProb > 50 ) weather_symbol = 95; // heavy thunderstorms: cb-cloud + two lightning

			}

			if (!myTargetInfo->Value(weather_symbol))
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
			itsLogger->Debug("Calculation took " + boost::lexical_cast<string> (processTimer->GetTime()) + " microseconds on " + deviceType);
#endif
			itsConfiguration->Statistics()->AddToMissingCount(missingCount);
			itsConfiguration->Statistics()->AddToValueCount(count);
		}
		
		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places		 
		 */

		myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}

double fmi_weather_symbol_1::rain_type(double kIndex, double T0m, double T850) {

	double rain_type;
	if (kIndex > 15) rain_type = 2; // check for convection
	else if (metutil::LowConvection_(T0m, T850) == 0) rain_type = 1; // check for shallow convection
	else rain_type = 2;

    return rain_type;
}

double fmi_weather_symbol_1::thunder_prob(double kIndex, double cloud) {

	double thunder_prob = 0; // initialize thunder probability to 0%
			if ( cloud == 3309 || cloud == 2303 || cloud == 2302 || cloud == 1309 || cloud == 1303 || cloud == 1302 )
			{
				if (kIndex >= 37) thunder_prob = 60;  // heavy thunder, set thunder probability to a value over 50% (to be replaced by a more scientific way to determine thunder probability in the future)
				else if (kIndex >= 27) thunder_prob = 40; // thunder, set thunder probability to a value between 30% and 50% (to be replaced by a more scientific way to determine thunder probability in the future)
			}

	return thunder_prob;
}
