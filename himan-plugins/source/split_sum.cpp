/**
 * @file split_sum.cpp
 *
 * @date Jan 28, 2012
 * @author partio
 */

#include "split_sum.h"
#include <iostream>
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include <map>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

/*
 * When calculating rate, we calculate the average of the time period.
 * So for example if time step is 30 (minutes), we fetch data for time step
 * 15 and time step 45, calculate the difference and divide by 30.
 *
 * The other option would be just to get data for step 30 and data for step 15
 * and divide by 15.
 *
 * For the first case we cannot calculate the last time step since for that we
 * would have to have data for the time step after last time step. Extrapolating
 * the data is really not an option since the parameters in question (precipitation and
 * to some degree radiation) are simply not suitable for extrapolation.
 * 
 * For the latter case we cannot calculate data for the first time step since in 
 * that case we would need to have data for the time step previous to first time
 * step. Again, extrapolation is not really an option.
 *
 * Bu default calculate rate using the average style, I'm not sure if this is how
 * it's supposed to be done but at least this is how it's done in hilake. Maybe
 * this could be a configuration file option?
 */

// Default behavior changed 27.12.2013 (HIMAN-26) // partio

const bool CALCULATE_AVERAGE_RATE = false;

map<string,params> sourceParameters;

split_sum::split_sum()
{
	itsClearTextFormula = "Hourly_sum = SUM_cur - SUM_prev; Rate = (SUM_cur - SUM_prev) / step";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("split_sum"));

	// Define source parameters for each output parameter
	
	// General precipitation (liquid + solid)
	sourceParameters["RR-1-MM"] = { param("RR-KGM2") };
	sourceParameters["RR-3-MM"] = { param("RR-KGM2") };
	sourceParameters["RR-6-MM"] = { param("RR-KGM2") };
	sourceParameters["RR-12-MM"] = { param("RR-KGM2") };	
	sourceParameters["RRR-KGM2"] = { param("RR-KGM2") }; // So-called HHSade
	sourceParameters["RRRC-KGM2"] = { param("RRC-KGM2") };
	sourceParameters["RRRL-KGM2"] = { param("RRL-KGM2") };

	// Snow
	sourceParameters["SNR-KGM2"] = { param("SNACC-KGM2") };
	sourceParameters["SNRC-KGM2"] = { param("SNC-KGM2") };
	sourceParameters["SNRL-KGM2"] = { param("SNL-KGM2") };

	// Graupel
	sourceParameters["GRR-MMH"] = { param("GR-KGM2") };

	// Solid (snow + graupel + hail)
	sourceParameters["RRRS-KGM2"] = { param("RRS-KGM2") };

	// Radiation
	sourceParameters["RADGLO-WM2"] = { param("RADGLOA-JM2"), param("RADGLO-WM2") };
	sourceParameters["RADLW-WM2"] = { param("RADLWA-JM2"), param("RADLW-WM2") };
	sourceParameters["RTOPLW-WM2"] = { param("RTOPLWA-JM2"), param("RTOPLW-WM2") };

}

void split_sum::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to split_sum.
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> params;

	if (itsConfiguration->Exists("rr1h") && itsConfiguration->GetValue("rr1h") == "true")
	{
		param parm;
		parm.Name("RR-1-MM");
		parm.UnivId(353);
		parm.Aggregation().TimeResolutionValue(1);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(8);

		parm.Aggregation().Type(kAccumulation);
		parm.Aggregation().TimeResolution(kHourResolution);

		params.push_back(parm);

	}

	if (itsConfiguration->Exists("rr3h") && itsConfiguration->GetValue("rr3h") == "true")
	{
		param parm;
		parm.Name("RR-3-MM");
		parm.UnivId(354);
		parm.Aggregation().TimeResolutionValue(3);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(8);

		parm.Aggregation().Type(kAccumulation);
		parm.Aggregation().TimeResolution(kHourResolution);

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("rr6h") && itsConfiguration->GetValue("rr6h") == "true")
	{
		param parm;
		parm.Name("RR-6-MM");
		parm.UnivId(355);
		parm.Aggregation().TimeResolutionValue(6);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(8);

		parm.Aggregation().Type(kAccumulation);
		parm.Aggregation().TimeResolution(kHourResolution);

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("rr12h") && itsConfiguration->GetValue("rr12h") == "true")
	{
		param parm;
		parm.Name("RR-12-MM");
		parm.UnivId(356);
		parm.Aggregation().TimeResolutionValue(12);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(8);

		parm.Aggregation().Type(kAccumulation);
		parm.Aggregation().TimeResolution(kHourResolution);

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("rrr") && itsConfiguration->GetValue("rrr") == "true")
	{
		param parm;
		parm.Name("RRR-KGM2");
		parm.UnivId(49);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(52);
		
		params.push_back(parm);
	}

	if (itsConfiguration->Exists("rrrc") && itsConfiguration->GetValue("rrc") == "true")
	{
		param parm;
		parm.Name("RRRC-KGM2");
		parm.UnivId(201);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(196);

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("rrrl") && itsConfiguration->GetValue("rrl") == "true")
	{
		param parm;
		parm.Name("RRRL-KGM2");
		parm.UnivId(200);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(54);
		
		params.push_back(parm);
	}

	// Graupel

	if (itsConfiguration->Exists("grr") && itsConfiguration->GetValue("grr") == "true")
	{
		param parm;
		parm.Name("GRR-MMH");
		parm.UnivId(200);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(54);

		params.push_back(parm);
	}

	// Solid

	if (itsConfiguration->Exists("rrrs") && itsConfiguration->GetValue("rrrs") == "true")
	{
		param parm;
		parm.Name("RRRS-KGM2");
		parm.UnivId(200);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(54);

		params.push_back(parm);
	}

	// Snow
	
	if (itsConfiguration->Exists("snr") && itsConfiguration->GetValue("snr") == "true")
	{
		param parm;
		parm.Name("SNR-KGM2");
		parm.UnivId(264);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(53);

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("snrc") && itsConfiguration->GetValue("snrc") == "true")
	{
		param parm;
		parm.Name("SNRC-KGM2");
		parm.UnivId(269);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(55);
		
		params.push_back(parm);
	}

	if (itsConfiguration->Exists("snrl") && itsConfiguration->GetValue("snrl") == "true")
	{
		param parm;
		parm.Name("SNRL-KGM2");
		parm.UnivId(268);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(56);

		params.push_back(parm);
	}

	// Radiation
	// These are RATES not SUMS

	if (itsConfiguration->Exists("glob") && itsConfiguration->GetValue("glob") == "true")
	{
		param parm;
		parm.Name("RADGLO-WM2");
		parm.UnivId(317);

		parm.GribDiscipline(0);
		parm.GribCategory(4);
		parm.GribParameter(3);

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("lw") && itsConfiguration->GetValue("lw") == "true")
	{
		param parm;
		parm.Name("RADLW-WM2");
		parm.UnivId(315);

		parm.GribDiscipline(0);
		parm.GribCategory(4);
		parm.GribParameter(5);

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("toplw") && itsConfiguration->GetValue("toplw") == "true")
	{
		param parm;
		parm.Name("RTOPLW-WM2");
		parm.UnivId(314);

		// Same grib2 parameter definition as with RADLW-WM2, this is just on
		// another surface
		
		parm.GribDiscipline(0);
		parm.GribCategory(4);
		parm.GribParameter(5);

		params.push_back(parm);
	}


	if (params.empty())
	{
		itsLogger->Trace("No parameter definition given, defaulting to rr6h");
		
		param parm;
		parm.Name("RR-6-MM");
		parm.UnivId(355);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(8);

		parm.Aggregation().Type(kAccumulation);
		parm.Aggregation().TimeResolution(kHourResolution);
		parm.Aggregation().TimeResolutionValue(6);

		params.push_back(parm);
	}

	SetParams(params);

	Start();
	
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void split_sum::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("split_sumThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
		{

			myThreadedLogger->Info("Calculating parameter " + myTargetInfo->Param().Name() + " time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
												  " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

			string parmName = myTargetInfo->Param().Name();

			bool isRadiationCalculation = (	parmName == "RADGLO-WM2" ||
											parmName == "RADLW-WM2" ||
											parmName == "RTOPLW-WM2");
			
			bool isRateCalculation = (isRadiationCalculation || 
										parmName == "RRR-KGM2" ||
										parmName == "RRRL-KGM2" ||
										parmName == "RRRC-KGM2" ||
										parmName == "SNR-KGM2" ||
										parmName == "SNRC-KGM2" ||
										parmName == "SNRL-KGM2" ||
										parmName == "GRR-MMH" ||
										parmName == "RRRS-KGM2");

			// Have to re-fetch infos each time since we might have to change element
			// from liquid to snow to radiation so we need also different source parameters

			shared_ptr<himan::info> curSumInfo;
			shared_ptr<himan::info> prevSumInfo;
			
			/*
			 * Two modes of operation:
			 *
			 * 1) When calculating split_sum sums, always get the previous
			 * step value from the current step and get both values (current and
			 * previous). If either can't be found, skip time step.
			 *
			 * 2) When calculating split_sum rate, get the first data that's
			 * earlier or same than current time step and the next data that's
			 * later or same than the current time step. Then calculate rate
			 * based on those values.
			 */

			if (isRateCalculation)
			{

				// Length of time step determined at configuration file

				if (myTargetInfo->Time().Step() == 0)
				{
					// This is the first time step, calculation can not be done

					myThreadedLogger->Info("This is the first time step -- not calculating " + myTargetInfo->Param().Name() + " for step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()));

					continue;
				}

				size_t timeIndex = myTargetInfo->TimeIndex();

				forecast_time previousTime;
				
				if (timeIndex == 0)
				{
					/*
					 * This is first time requested, but maybe not first time in the
					 * source data. We have to extrapolate the step to previous data
					 * from the step to next data.
					 *
					 * Of course we need to *have* next time step in order to do this.
					 * In order to work around the problem, we guess that we could
					 * have data in some time step and let the Fetch() function
					 * deal with it.
					 */

					previousTime = myTargetInfo->Time();
					
					if (myTargetInfo->SizeTimes() == 1)
					{
						if (previousTime.StepResolution() == kHourResolution)
						{
							previousTime.ValidDateTime()->Adjust(kHourResolution, -1);
						}
						else
						{
							previousTime.ValidDateTime()->Adjust(kMinuteResolution, -15);
						}						
					}
					else
					{

					
						forecast_time nextTime = myTargetInfo->PeekTime(timeIndex+1);

						int diff = nextTime.Step() - myTargetInfo->Time().Step();

						previousTime.ValidDateTime()->Adjust(myTargetInfo->Time().StepResolution(), -diff);
					}
				}
				else
				{
					previousTime = myTargetInfo->PeekTime(timeIndex-1);
				}
				
				assert(myTargetInfo->Time().StepResolution() == previousTime.StepResolution());

				int targetStep = myTargetInfo->Time().Step() - previousTime.Step();

				// Calculating RATE

				// Previous VALID data
				myThreadedLogger->Debug("Searching for previous time step data");
				prevSumInfo = GetSourceDataForRate(myTargetInfo, false, targetStep);
	
				// Next VALID data
				if (prevSumInfo)
				{
					myThreadedLogger->Debug("Searching for next time step data");
					curSumInfo = GetSourceDataForRate(myTargetInfo, true, targetStep);
				}

			}
			else
			{
				// Calculating SUM

				// Fetch data for previous step
				
				int paramStep = myTargetInfo->Param().Aggregation().TimeResolutionValue();

				if (myTargetInfo->Time().StepResolution() != kHourResolution)
				{
					assert(myTargetInfo->Time().StepResolution() == kMinuteResolution);

					paramStep *= 60;
				}

				// Skip early steps if necessary

				if (myTargetInfo->Time().Step() >= paramStep)
				{
					// Data from previous time step
					forecast_time prevTimeStep = myTargetInfo->Time();

					prevTimeStep.ValidDateTime()->Adjust(prevTimeStep.StepResolution(), -paramStep);

					prevSumInfo = FetchSourceData(myTargetInfo, prevTimeStep);

				}
					
				// Data from current time step, but only if we have data for previous
				// step

				if (prevSumInfo)
				{
					curSumInfo = FetchSourceData(myTargetInfo, myTargetInfo->Time());
				}
			}

			if (!prevSumInfo || !curSumInfo)
			{
				// Data was not found

				myThreadedLogger->Info("Data not found: not calculating " + myTargetInfo->Param().Name() + " for step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()));

				myTargetInfo->Data()->Fill(kFloatMissing);

				if (itsConfiguration->StatisticsEnabled())
				{
					itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
					itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
				}

				// Will write empty file
				
				myTargetInfo->Data()->Fill(kFloatMissing);

				continue;
			}

			myThreadedLogger->Debug("Previous data step is " + boost::lexical_cast<string> (prevSumInfo->Time().Step()));
			myThreadedLogger->Debug("Current/next data step is " + boost::lexical_cast<string> (curSumInfo->Time().Step()));

			shared_ptr<NFmiGrid> currentGrid(curSumInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> prevGrid(prevSumInfo->Grid()->ToNewbaseGrid());

			shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

			size_t missingCount = 0;
			size_t count = 0;

			bool equalGrids = (*myTargetInfo->Grid() == *curSumInfo->Grid() && *curSumInfo->Grid() == *prevSumInfo->Grid());

			double scaleFactor = 1.;

			// EC gives precipitation in meters, we are calculating millimeters

			if (curSumInfo->Param().Unit() == kM)
			{
				scaleFactor = 1000.;
			}

			string deviceType = "CPU";

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();
			currentGrid->Reset();
			prevGrid->Reset();

			double step = static_cast<double> (curSumInfo->Time().Step() - prevSumInfo->Time().Step());

			if (isRadiationCalculation)
			{

				/*
				 * Radiation unit is W/m^2 which is J/s/m^2, so we need to convert
				 * time to seconds.
				 *
				 * Here we do the same minute <--> hour switch than we do with
				 * precipitation rates: because smartmet views harmonie data
				 * only at one hour steps, we cannot calculate the radiation power
				 * every 15 minutes. We have to calculate the value from the past
				 * hour, not past 15 minutes. So in the case of Harmonie we
				 * divide the cumulative value with 60*60 seconds instead of
				 * 15*60 seconds.
				 */

				if (myTargetInfo->Time().StepResolution() == kMinuteResolution)
				{
					step = 3600;
				}
				else if (myTargetInfo->Time().StepResolution() == kHourResolution)
				{
					step *= 3600;
				}
				else
				{
					itsLogger->Error("Unknown time resolution: " + string(HPTimeResolutionToString.at(myTargetInfo->Time().StepResolution())));
					continue;
				}

			}

			else if (myTargetInfo->Time().StepResolution() == kMinuteResolution)
			{
				/*
				 * For precipitation:
				 *
				 * If calculating for Harmonie, use hour as base time unit!
				 * This has been agreed with AK.
				 *
				 * This is how its *should* be done, but it's not:
				 *
				 * --- THIS IS HOW IT SHOULD BE DONE BUT ITS NOT ---
				 *
				 * If target time resolution is hour, the basic time unit
				 * is one hour.
				 * If target time resolution is minute, the basic time unit
				 * is 15 minutes.
				 *
				 * For example, ECMWF:
				 *
				 * (DATA_STEP_240 - DATA_STEP_234) / (BASIC_TIME_UNIT * STEP)
				 *
				 * --> (DATA_STEP_240 - DATA_STEP_234) / (6 * 1 hour)
				 *
				 * For example, Harmonie:
				 *
				 * (DATA_STEP_120 - DATA_STEP_105) / (STEP / BASIC_TIME_UNIT * STEP)
				 *
				 * --> (DATA_STEP_120 - DATA_STEP_105) / 1
				 *
				 * --- THIS IS HOW IT SHOULD BE DONE BUT ITS NOT ---
				 */

				step = 1;
			}
			
			while (myTargetInfo->NextLocation() && targetGrid->Next() && currentGrid->Next() && prevGrid->Next())
			{
				count++;

				double currentSum = kFloatMissing;
				double previousSum = kFloatMissing;

				InterpolateToPoint(targetGrid, currentGrid, equalGrids, currentSum);
				InterpolateToPoint(targetGrid, prevGrid, equalGrids, previousSum);

				if (currentSum == kFloatMissing || previousSum == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				double sum = currentSum - previousSum;

				if (sum > 0 && isRateCalculation && step != 1)
				{
					sum /= step;
				}

				if (!isRadiationCalculation && sum < 0)
				{
					sum = 0;
				}

				sum *= scaleFactor;

				if (!myTargetInfo->Value(sum))
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
				itsConfiguration->Statistics()->AddToMissingCount(missingCount);
				itsConfiguration->Statistics()->AddToValueCount(count);
			}

			/*
			 * Now we are done for this param for this level
			 *
			 * Clone info-instance to writer since it might change our descriptor places
			 */

			myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));
		}

		// Write all parameters to disk
			
		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}

shared_ptr<himan::info> split_sum::GetSourceDataForRate(shared_ptr<const info> myTargetInfo, bool forward, int targetStep)
{
	shared_ptr<info> SumInfo;

	/*
	 * When calculating rate there are three different kinds if situations with
	 * regards to time:
	 *
	 * 1) The target step is smaller than data step.
	 *
	 * This is for example when calculating hourly rate for ECMWF data where the data
	 * step is 3 or 6 hours. So we have
	 *
	 * DATA_STEP  TARGET_STEP
	 * 0          0
	 *            1
	 *            2
	 * 3          3
	 *            4
	 *            5
	 * 6          6
	 *
	 * So for target steps 1-3, we calculate (data_step 3 - data_step 0) / 3
	 *
	 * 2) The target step is the same as data step.
	 *
	 * This is for example when calculating hourly rate for Hirlam. This is the
	 * most simple case.
	 *
	 * DATA_STEP  TARGET_STEP
	 * 0          0
	 * 1          1
	 * 2          2
	 * 3          3
	 *
	 * So for target step 1 we calculate (data_step 1 - data_step 0) / 1
	 *
	 * 3) The target step is larger than the data step.
	 * 
	 * This is for example when calculating hourly rate for Harmonie.
	 *
	 * DATA_STEP  TARGET_STEP
	 * 0          0
	 * 15
	 * 30
	 * 45
	 * 60         60
	 *
	 * So for target step 60 we calculate (data_step 60 - data_step 0) / 1
	 *
	 * This differs from point 2) since we cannot simply take the first time step
	 * that precedes step 60; we have to rollback all the way to step 0 and also
	 * be sure to use 1 as divisor, not 4 or 60.
	 */

	HPTimeResolution timeResolution = myTargetInfo->Time().StepResolution();

	int steps = 6; // by default look for 6 timesteps forward or backward
	int step = 1; // by default the difference between time steps is one (ie. one hour))

	if (timeResolution == kMinuteResolution)
	{
		step = 60;	// Forecast step is 15 (ie Harmonie), but it has been agreed
					// with AKS that we'll use one hour since editor displays
					// only hourly data.
	}
	else if (timeResolution != kHourResolution)
	{
		throw runtime_error(ClassName() + ": Invalid time resolution value: " + HPTimeResolutionToString.at(timeResolution));
	}

	int i = 0;

	/*
	 * Parameter RRR-KGM2 is always calculated with 'normal' mode (no averaging).
	 */
	
	if (!forward || (CALCULATE_AVERAGE_RATE && myTargetInfo->Param().Name() != "RRR-KGM2"))
	{ 
		//i = targetStep;
		i = step;
	}

	for (; !SumInfo && i <= steps*step; i += step)
	{
		int curstep = i;

		forecast_time wantedTimeStep = myTargetInfo->Time();

		if (!forward)
		{
			curstep *= -1;
		}

		wantedTimeStep.ValidDateTime()->Adjust(timeResolution, curstep);

		if (wantedTimeStep.Step() < 0)
		{
			continue;
		}

		SumInfo = FetchSourceData(myTargetInfo,wantedTimeStep);

	}

	return SumInfo;
}

shared_ptr<himan::info> split_sum::FetchSourceData(shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime)
{
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	level wantedLevel(kHeight, 0 ,"HEIGHT");

	// Transform ground level based on only the first source parameter

	auto params = sourceParameters[myTargetInfo->Param().Name()];

	// Must have source parameter for target parameter defined in map sourceParameters

	assert(!params.empty());

	if (myTargetInfo->Param().Name() == "RTOPLW-WM2")
	{
		wantedLevel = level(kTopOfAtmosphere, 0, "TOP");
	}

	shared_ptr<info> SumInfo;
	
	try
	{
		SumInfo = f->Fetch(itsConfiguration,
						wantedTime,
						wantedLevel,
						params);
	}
	catch (HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw;
		}
	}

	// If model does not provide data for timestep 0, emulate it
	// by providing a zero-grid

	if (!SumInfo && wantedTime.Step() == 0)
	{
		SumInfo = make_shared<info> (*myTargetInfo);
		vector<forecast_time> times = { wantedTime };
		vector<level> levels = { wantedLevel };
		vector<param> params = { sourceParameters[myTargetInfo->Param().Name()][0] };

		SumInfo->Params(params);
		SumInfo->Levels(levels);
		SumInfo->Times(times);

		SumInfo->Create(myTargetInfo->Grid());
		SumInfo->First();
		SumInfo->Grid()->Data()->Fill(0);


	}
	
	return SumInfo;

}