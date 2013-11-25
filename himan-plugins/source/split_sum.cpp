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

bool CALCULATE_AVERAGE_RATE = true;

map<string,params> sourceParameters;

split_sum::split_sum()
{
	itsClearTextFormula = "Hourly_sum = SUM_cur - SUM_prev; Rate = (SUM_next - SUM_prev) / (step * 2)";

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

	// Radiation
	sourceParameters["RADGLO-WM2"] = { param("RADGLOA-JM2") }; //, param("RADGLO-WM2") };
	sourceParameters["RADLW-WM2"] = { param("RADLWA-JM2") }; // , param("RADLW-WM2") };
	sourceParameters["RTOPLW-WM2"] = { param("RTOPLWA-JM2") }; //, param("RTOPLW-WM2") };

}

void split_sum::Process(std::shared_ptr<const plugin_configuration> conf)
{

	unique_ptr<timer> aTimer;

	// Get number of threads to use

	short threadCount = ThreadCount(conf->ThreadCount());

	if (conf->StatisticsEnabled())
	{
		aTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
		aTimer->Start();
		conf->Statistics()->UsedThreadCount(threadCount);
		conf->Statistics()->UsedGPUCount(conf->CudaDeviceCount());
	}

	boost::thread_group g;

	shared_ptr<info> targetInfo = conf->Info();

	/*
	 * Set target parameter to split_sum.
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> params;

	if (conf->Exists("rr1h") && conf->GetValue("rr1h") == "true")
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

	if (conf->Exists("rr3h") && conf->GetValue("rr3h") == "true")
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

	if (conf->Exists("rr6h") && conf->GetValue("rr6h") == "true")
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

	if (conf->Exists("rr12h") && conf->GetValue("rr12h") == "true")
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

	if (conf->Exists("rrr") && conf->GetValue("rrr") == "true")
	{
		param parm;
		parm.Name("RRR-KGM2");
		parm.UnivId(49);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(52);
		
		params.push_back(parm);
	}

	if (conf->Exists("rrrc") && conf->GetValue("rrc") == "true")
	{
		param parm;
		parm.Name("RRRC-KGM2");
		parm.UnivId(201);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(196);

		params.push_back(parm);
	}

	if (conf->Exists("rrrl") && conf->GetValue("rrl") == "true")
	{
		param parm;
		parm.Name("RRRL-KGM2");
		parm.UnivId(200);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(54);
		
		params.push_back(parm);
	}

	// Snow
	
	if (conf->Exists("snr") && conf->GetValue("snr") == "true")
	{
		param parm;
		parm.Name("SNR-KGM2");
		parm.UnivId(264);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(53);

		params.push_back(parm);
	}

	if (conf->Exists("snrc") && conf->GetValue("snrc") == "true")
	{
		param parm;
		parm.Name("SNRC-KGM2");
		parm.UnivId(269);

		parm.GribDiscipline(0);
		parm.GribCategory(1);
		parm.GribParameter(55);
		
		params.push_back(parm);
	}

	if (conf->Exists("snrl") && conf->GetValue("snrl") == "true")
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

	if (conf->Exists("glob") && conf->GetValue("glob") == "true")
	{
		param parm;
		parm.Name("RADGLO-WM2");
		parm.UnivId(317);

		parm.GribDiscipline(0);
		parm.GribCategory(4);
		parm.GribParameter(3);

		params.push_back(parm);
	}

	if (conf->Exists("lw") && conf->GetValue("lw") == "true")
	{
		param parm;
		parm.Name("RADLW-WM2");
		parm.UnivId(315);

		parm.GribDiscipline(0);
		parm.GribCategory(4);
		parm.GribParameter(5);

		params.push_back(parm);
	}

	if (conf->Exists("toplw") && conf->GetValue("toplw") == "true")
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

	// GRIB 1

	if (conf->OutputFileType() == kGRIB1)
	{
		StoreGrib1ParameterDefinitions(params, targetInfo->Producer().TableVersion());
	}

	targetInfo->Params(params);

	/*
	 * Create data structures.
	 */

	targetInfo->Create();

	/*
	 * Initialize parent class functions for dimension handling
	 */

	Dimension(conf->LeadingDimension());
	FeederInfo(shared_ptr<info> (new info(*targetInfo)));
	FeederInfo()->ParamIndex(0);

	if (conf->StatisticsEnabled())
	{
		aTimer->Stop();
		conf->Statistics()->AddToInitTime(aTimer->GetTime());
		aTimer->Start();
	}

	/*
	 * Each thread will have a copy of the target info.
	 */

	for (short i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		boost::thread* t = new boost::thread(&split_sum::Run,
											 this,
											 shared_ptr<info> (new info(*targetInfo)),
											 conf,
											 i + 1);

		g.add_thread(t);

	}

	g.join_all();

	if (conf->StatisticsEnabled())
	{
		aTimer->Stop();
		conf->Statistics()->AddToProcessingTime(aTimer->GetTime());
	}

	if (conf->FileWriteOption() == kSingleFile)
	{
		WriteToFile(conf, targetInfo);
	}
}

void split_sum::Run(shared_ptr<info> myTargetInfo,
				shared_ptr<const plugin_configuration> conf,
				unsigned short threadIndex)
{
	while (AdjustLeadingDimension(myTargetInfo))
	{
		Calculate(myTargetInfo, conf, threadIndex);
	}

}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void split_sum::Calculate(shared_ptr<info> myTargetInfo,
					 shared_ptr<const plugin_configuration> conf,
					 unsigned short threadIndex)
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
										parmName == "SNRL-KGM2");

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
					 * Of course we need to *have* next time step in order to do this
					 * (meaning that if this calculation is done for only a single
					 * time step, we will fail).
					 */

					if (myTargetInfo->SizeTimes() == 1)
					{
						itsLogger->Error("Rate cannot be calculate to a single time step");
						return;
					}
					
					previousTime = myTargetInfo->Time();
					
					forecast_time nextTime = myTargetInfo->PeekTime(timeIndex+1);

					int diff = nextTime.Step() - myTargetInfo->Time().Step();

					previousTime.ValidDateTime()->Adjust(myTargetInfo->Time().StepResolution(), -diff);
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
				prevSumInfo = GetSourceDataForRate(conf, myTargetInfo, false, targetStep);
	
				// Next VALID data
				if (prevSumInfo)
				{
					myThreadedLogger->Debug("Searching for next time step data");
					curSumInfo = GetSourceDataForRate(conf, myTargetInfo, true, targetStep);
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

					prevSumInfo = FetchSourceData(conf, myTargetInfo, prevTimeStep);

				}
					
				// Data from current time step, but only if we have data for previous
				// step

				if (prevSumInfo)
				{
					curSumInfo = FetchSourceData(conf, myTargetInfo, myTargetInfo->Time());
				}
			}

			if (!prevSumInfo || !curSumInfo)
			{
				// Data was not found

				myThreadedLogger->Info("Data not found: not calculating " + myTargetInfo->Param().Name() + " for step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()));

				myTargetInfo->Data()->Fill(kFloatMissing);

				if (conf->StatisticsEnabled())
				{
					conf->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
					conf->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
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

				// Radiation unit is W/m^2 which is J/s/m^2, so we need to
				// convert time to seconds

				if (myTargetInfo->Time().StepResolution() == kMinuteResolution)
				{
					step *= 60;
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

			if (myTargetInfo->Time().StepResolution() == kMinuteResolution)
			{
				/*
				 * If target time resolution is minute, we probably still
				 * want to calculate HOURLY rates.
				 *
				 * When resolution is hour, the basic unit of calculation
				 * is one hour. When it is minute, the basic unit is one
				 * minute. As it is not very useful to calculate rates
				 * per minute, we have adjust it here and try to be
				 * smarter than the user calling us.
				 *
				 * So, when time resolution is minute and larger than
				 * 15 minutes, divide the results with 1.
				 *
				 * The result then is, that if target step is 15,
				 * we calculate (for step 120)
				 *
				 * (DATA_STEP 120 - DATA_STEP 105) / 1
				 *
				 * which gives the correct result.
				 *
				 * If the target step is 60, we have (for step 120)
				 *
				 * (DATA_STEP 120 - DATA_STEP 60) / 1
				 *
				 * which gives the correct result.
				 *
				 */

				//assert(myTargetInfo->Time().Step() >= 15);

				//step = 1;
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

				if (isRateCalculation)
				{
					if (step != 1)
					{
						sum /= step;
					}
				}

				if (sum < 0)
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

			if (conf->StatisticsEnabled())
			{
				conf->Statistics()->AddToMissingCount(missingCount);
				conf->Statistics()->AddToValueCount(count);
			}

			/*
			 * Now we are done for this param for this level
			 *
			 * Clone info-instance to writer since it might change our descriptor places
			 */

			myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));
		}

		// Write all parameters to disk
			
		if (conf->FileWriteOption() != kSingleFile)
		{
			WriteToFile(conf, myTargetInfo);
		}
	}
}

shared_ptr<himan::info> split_sum::GetSourceDataForRate(shared_ptr<const plugin_configuration> conf, shared_ptr<const info> myTargetInfo, bool forward, int targetStep)
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
		step = 15; // Time step is 15 minutes (ie harmonie)
	}
	else if (timeResolution != kHourResolution)
	{
		throw runtime_error(ClassName() + ": Invalid time resolution value: " + HPTimeResolutionToString.at(timeResolution));
	}

	int i = 0;

	if (!forward || (CALCULATE_AVERAGE_RATE && myTargetInfo->Param().Name() != "RRR-KGM2"))
	{ 
		i = targetStep;
	}
	
	for (; !SumInfo && i <= steps + targetStep; i += step)
	{
		int curstep = i;

		//*= step;
		
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

		SumInfo = FetchSourceData(conf,myTargetInfo,wantedTimeStep);

	}

	return SumInfo;
}

shared_ptr<himan::info> split_sum::FetchSourceData(shared_ptr<const plugin_configuration> conf, shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime)
{
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	level groundLevel(kHeight, 0 ,"HEIGHT");

	// Transform ground level based on only the first source parameter

	auto params = sourceParameters[myTargetInfo->Param().Name()];

	// Must have source parameter for target parameter defined in map sourceParameters

	assert(!params.empty());

	groundLevel = LevelTransform(conf->SourceProducer(), params[0], groundLevel);

	shared_ptr<info> SumInfo;
	
	try
	{
		SumInfo = f->Fetch(conf,
						wantedTime,
						groundLevel,
						sourceParameters[myTargetInfo->Param().Name()]);
	}
	catch (HPExceptionType e)
	{
		if (e != kFileDataNotFound)
		{
			throw e;
		}
	}

	// If model does not provide data for timestep 0, emulate it
	// by providing a zero-grid

	if (!SumInfo && wantedTime.Step() == 0)
	{

		SumInfo = make_shared<info> (*myTargetInfo);
		vector<forecast_time> times = { wantedTime };
		vector<level> levels = { groundLevel };
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