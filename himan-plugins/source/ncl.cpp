/**
 * @file ncl.cpp
 *
 * Template for future plugins.
 *
 * @date Apr 10, 2013
 * @author peramaki
 */

#include "ncl.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const string itsName("ncl");

ncl::ncl()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void ncl::Process(std::shared_ptr<const plugin_configuration> conf)
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
	
	// GRIB 2
	
	param theRequestedParam;
	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(3);
	theRequestedParam.GribParameter(6);

	shared_ptr<neons> theNeons = dynamic_pointer_cast <neons> (plugin_factory::Instance()->Plugin("neons"));

	itsBottomLevel = boost::lexical_cast<int> (theNeons->ProducerMetaData(230, "last hybrid level number"));


	if (itsConfiguration->Exists("temp") && itsConfiguration->GetValue("temp") == "-20" )
	{
    	theRequestedParam.Name("HM20C-M");
    	theRequestedParam.UnivId(28);
    	targetTemperature = -20;
    }

    if (itsConfiguration->Exists("temp") && itsConfiguration->GetValue("temp") == "0" )
	{
    	theRequestedParam.Name("H0C-M");
    	theRequestedParam.UnivId(270);
    	targetTemperature = 0;
    	
    }

	if (Dimension() != kTimeDimension)
	{
		itsLogger->Info("Changing leading_dimension to time");
		Dimension(kTimeDimension);
	}

	theParams.push_back(theRequestedParam);

	SetParams(theParams);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void ncl::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	param HParam("HL-M");
	param TParam("T-K");

	int levelNumber = itsBottomLevel;

	level HLevel(himan::kHybrid, static_cast<float> (levelNumber), "HYBRID");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> HInfo;
		shared_ptr<info> TInfo;
		shared_ptr<info> prevHInfo;
		shared_ptr<info> prevTInfo;

		try
		{

			HInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 HParam);
			
			TInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 HLevel,
								 TParam);
			
		}
		catch (HPExceptionType& e)
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
		
		size_t missingCount = 0;
		size_t count = 0;

		bool firstLevel = true;

		myTargetInfo->Data()->Fill(-1);		
		
		HInfo->ResetLocation();
		TInfo->ResetLocation();

		level curLevel = HLevel;
		level prevLevel;
		
		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

		string deviceType;
		deviceType = "CPU";


		while (--levelNumber > 0)
		{

			targetGrid->Reset();		
			myTargetInfo->ResetLocation();	

			shared_ptr<NFmiGrid> HGrid(HInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> prevHGrid;
			shared_ptr<NFmiGrid> prevTGrid;

			if (!firstLevel)
			{
				prevHGrid = shared_ptr<NFmiGrid>(prevHInfo->Grid()->ToNewbaseGrid());
				prevTGrid = shared_ptr<NFmiGrid>(prevTInfo->Grid()->ToNewbaseGrid());
			}


			bool equalGrids = 	(	
									*myTargetInfo->Grid() == *HInfo->Grid() && 
									*myTargetInfo->Grid() == *TInfo->Grid() && 	
									( 	
										firstLevel ||
										( 	*myTargetInfo->Grid() == *prevHInfo->Grid() && 
											*myTargetInfo->Grid() == *prevTInfo->Grid() 
										)
							   		) 
								);

			while 	(myTargetInfo->NextLocation() && targetGrid->Next() )
			{
				
				double height(kFloatMissing);
				double temp(kFloatMissing);
				double prevHeight(kFloatMissing);
				double prevTemp(kFloatMissing);

				double targetHeight = myTargetInfo->Value();

				assert(targetGrid->Size() == myTargetInfo->Data()->Size());

				InterpolateToPoint(targetGrid, TGrid, equalGrids, temp);
				InterpolateToPoint(targetGrid, HGrid, equalGrids, height);

				if (!firstLevel)
				{
					InterpolateToPoint(targetGrid, prevHGrid, equalGrids, prevHeight);
					InterpolateToPoint(targetGrid, prevTGrid, equalGrids, prevTemp);
				}

				if 	(
						height == kFloatMissing || 
						temp == kFloatMissing || 
						(
							!firstLevel && 
							( 
								prevHeight == kFloatMissing || 
								prevTemp == kFloatMissing
							)
						)
					)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				temp -= himan::constants::kKelvin;
				prevTemp -= himan::constants::kKelvin;

				if (targetHeight != -1)
				{

					if (temp >= targetTemperature) // && levelNumber >= (itsBottomLevel - 5))
					{
						/*
						 * Lowest 5 model levels and "inversion fix".
						 *
						 * Especially in winter time surface inversion is common: temperature
						 * close to ground surfaceis colder than the temperature above it.
						 * This inversion is usually very limited in height, maybe 10 .. 200
						 * meters. When calculating height of zero level, we can take into
						 * account this inversion so we test if the temperature at lowest level
						 * is already below target temperature (0 or -20), and if that is the
						 * situation we do not directly set height to kFloatMissing but
						 * move onwards and only if the temperature stays below target for
						 * the first 5 levels (in Hirlam lowest hybrid levels are spaced ~30m
						 * apart) we can say that surface inversion does not exist and the
						 * air temperature is simply too cold for this target temperature.
						 *
						 * In order to imitate hilake however the inversion fix limitation for
						 * bottom 5 has been disabled, since what hilake does is it applies
						 * the fix as long as the process is running, at least up to hybrid
						 * level 29 which on Hirlam is on average 3700m above ground.
						 *
						 * This seems like a non-optimal approach but this is how hilake
						 * does it and until we have confirmation that the inversion
						 * height should be limited, use the hilake way also in himan.
						 *
						 * Inversion: http://blogi.foreca.fi/2012/01/pakkanen-ja-inversio/
						 */

						myTargetInfo->Value(-1);

					}
					
					// No inversion fix, value already found for this gridpoint
					// (real value or missing value)

					continue;
				}

				if ((firstLevel && temp < targetTemperature) || (prevTemp < targetTemperature && temp < targetTemperature))
				{
					/*
					 * Height is below ground (first model level) or its so cold
					 * in the lower atmosphere that we cannot find the requested
					 * temperature, set height to to missing value.
					 */
					
					targetHeight = kFloatMissing;
				}				
				else if (prevTemp > targetTemperature && temp < targetTemperature)
				{
					// Found value, interpolate
					
					double p_rel = (targetTemperature - temp) / (prevTemp - temp);
					targetHeight = height + (prevHeight - height) * p_rel;
				}
				else
				{
					// Too warm
					continue;
				}

				count++;
				
				if (!myTargetInfo->Value(targetHeight))
				{
					throw runtime_error(ClassName() + ": Failed to set value to matrix");
				}

			}

			prevLevel = curLevel;
			curLevel = level(himan::kHybrid, static_cast<float> (levelNumber), "HYBRID");
			
			HInfo = FetchPrevious(myTargetInfo->Time(), curLevel, HParam);
			TInfo = FetchPrevious(myTargetInfo->Time(), curLevel, TParam);
			
			prevHInfo = FetchPrevious(myTargetInfo->Time(), prevLevel, HParam);
			prevTInfo = FetchPrevious(myTargetInfo->Time(), prevLevel, TParam);


			firstLevel = false;

			if (CountValues(myTargetInfo))
			{
				break;
			}
		} 

		/*
		 * Replaces all unset values
		 */

		myTargetInfo->ResetLocation();
		while(myTargetInfo->NextLocation())
		{
			if ( myTargetInfo->Value() == -1)
			{
				myTargetInfo->Value(kFloatMissing);
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

		myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}
shared_ptr<himan::info> ncl::FetchPrevious(const forecast_time& wantedTime, const level& wantedLevel, const param& wantedParam)
{
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	try
	{
		return f->Fetch(itsConfiguration,
						wantedTime,
						wantedLevel,
						wantedParam);
   	}
	catch (HPExceptionType& e)
	{
		throw;
	}

}
bool ncl::CountValues(const shared_ptr<himan::info> values)
{
	size_t s = values->Data()->Size();
	for (size_t j = 0; j < s; j++)
	{
		if (values->Data()->At(j) == -1)
			return false;
	}
	return true;
}
