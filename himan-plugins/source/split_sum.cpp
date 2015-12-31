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
#include "level.h"
#include "forecast_time.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "writer.h"

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

map<string,himan::params> sourceParameters;

split_sum::split_sum()
{
	itsClearTextFormula = "Hourly_sum = SUM_cur - SUM_prev; Rate = (SUM_cur - SUM_prev) / step";

	itsLogger = logger_factory::Instance()->GetLog("split_sum");

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
	sourceParameters["RNETLW-WM2"] = { param("RNETLWA-JM2"), param("RNETLW-WM2") };

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
		param parm("RR-1-MM", 353, 0, 1, 8);
		parm.Unit(kMm);

		parm.Aggregation(aggregation(kAccumulation, kHourResolution, 1));

		params.push_back(parm);

	}

	if (itsConfiguration->Exists("rr3h") && itsConfiguration->GetValue("rr3h") == "true")
	{
		param parm("RR-3-MM", 354, 0, 1, 8);
		parm.Unit(kMm);

		parm.Aggregation(aggregation(kAccumulation, kHourResolution, 3));
		params.push_back(parm);
	}

	if (itsConfiguration->Exists("rr6h") && itsConfiguration->GetValue("rr6h") == "true")
	{
		param parm("RR-6-MM", 355, 0, 1, 8);
		parm.Unit(kMm);

		parm.Aggregation(aggregation(kAccumulation, kHourResolution, 6));

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("rr12h") && itsConfiguration->GetValue("rr12h") == "true")
	{
		param parm("RR-12-MM", 356, 0, 1, 8);
		parm.Unit(kMm);

		parm.Aggregation(aggregation(kAccumulation, kHourResolution, 12));

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("rrr") && itsConfiguration->GetValue("rrr") == "true")
	{
		param parm("RRR-KGM2", 49, 0, 1, 52);
		parm.Unit(kKgm2);

		parm.Aggregation(aggregation(kAccumulation, kUnknownTimeResolution, kHPMissingInt));

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("rrrc") && itsConfiguration->GetValue("rrrc") == "true")
	{
		param parm("RRRC-KGM2", 201, 0, 1, 196);
		parm.Unit(kKgm2);

		parm.Aggregation(aggregation(kAccumulation, kUnknownTimeResolution, kHPMissingInt));

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("rrrl") && itsConfiguration->GetValue("rrrl") == "true")
	{
		param parm("RRRL-KGM2", 200, 0, 1, 54);
		parm.Unit(kKgm2);

		parm.Aggregation(aggregation(kAccumulation, kUnknownTimeResolution, kHPMissingInt));

		params.push_back(parm);
	}

	// Graupel

	if (itsConfiguration->Exists("grr") && itsConfiguration->GetValue("grr") == "true")
	{
		param parm("GRR-MMH", 1168);
		parm.Unit(kKgm2);

		parm.Aggregation(aggregation(kAccumulation, kUnknownTimeResolution, kHPMissingInt));

		params.push_back(parm);
	}

	// Solid

	if (itsConfiguration->Exists("rrrs") && itsConfiguration->GetValue("rrrs") == "true")
	{
		param parm("RRRS-KGM2", 1170);
		parm.Unit(kKgm2);

		parm.Aggregation(aggregation(kAccumulation, kUnknownTimeResolution, kHPMissingInt));

		params.push_back(parm);
	}

	// Snow
	
	if (itsConfiguration->Exists("snr") && itsConfiguration->GetValue("snr") == "true")
	{
		param parm("SNR-KGM2", 264, 0, 1, 53);
		parm.Unit(kKgm2);

		parm.Aggregation(aggregation(kAccumulation, kUnknownTimeResolution, kHPMissingInt));

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("snrc") && itsConfiguration->GetValue("snrc") == "true")
	{
		param parm("SNRC-KGM2", 269, 0, 1, 55);
		parm.Unit(kKgm2);

		parm.Aggregation(aggregation(kAccumulation, kUnknownTimeResolution, kHPMissingInt));

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("snrl") && itsConfiguration->GetValue("snrl") == "true")
	{
		param parm("SNRL-KGM2", 268, 0, 1, 56);
		parm.Unit(kKgm2);

		parm.Aggregation(aggregation(kAccumulation, kUnknownTimeResolution, kHPMissingInt));

		params.push_back(parm);
	}

	// Radiation
	// These are RATES not SUMS

	if (itsConfiguration->Exists("glob") && itsConfiguration->GetValue("glob") == "true")
	{
		param parm("RADGLO-WM2", 317, 0, 4, 3);

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("lw") && itsConfiguration->GetValue("lw") == "true")
	{
		param parm("RADLW-WM2", 315, 0, 5, 192);

		params.push_back(parm);
	}

	if (itsConfiguration->Exists("toplw") && itsConfiguration->GetValue("toplw") == "true")
	{
		// Same grib2 parameter definition as with RADLW-WM2, this is just on
		// another surface

		param parm("RTOPLW-WM2", 314, 0, 5, 192);

		params.push_back(parm);
	}
	
	if (itsConfiguration->Exists("netlw") && itsConfiguration->GetValue("netlw") == "true")
	{

		param parm("RNETLW-WM2", 312, 0, 5, 5);

		params.push_back(parm);
	}


	if (params.empty())
	{
		itsLogger->Trace("No parameter definition given, defaulting to rr6h");
		
		param parm("RR-6-MM", 355, 0, 1, 8);

		parm.Aggregation(aggregation(kAccumulation, kHourResolution, 6));
		
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

	auto myThreadedLogger = logger_factory::Instance()->GetLog("split_sumThread #" + boost::lexical_cast<string> (threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
	{

		myThreadedLogger->Info("Calculating parameter " + myTargetInfo->Param().Name() + " time " + static_cast<string>(forecastTime.ValidDateTime()) +
												  " level " + static_cast<string> (forecastLevel));

		string parmName = myTargetInfo->Param().Name();

		bool isRadiationCalculation = (	parmName == "RADGLO-WM2" ||
										parmName == "RADLW-WM2" ||
										parmName == "RTOPLW-WM2" ||
										parmName == "RNETLW-WM2"
		);
			
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

		info_t curSumInfo;
		info_t prevSumInfo;
		
		if (myTargetInfo->Time().Step() == 0)
		{
			// This is the first time step, calculation can not be done

			 myThreadedLogger->Info("This is the first time step -- not calculating " + myTargetInfo->Param().Name() + " for step " + boost::lexical_cast<string> (forecastTime.Step()));
			 continue;
		}

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


		int step = itsConfiguration->ForecastStep();

		if (isRateCalculation)
		{

			// Calculating RATE

			auto infos = GetSourceDataForRate(myTargetInfo, step);
			
			prevSumInfo = infos.first;
			curSumInfo = infos.second;
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

				prevTimeStep.ValidDateTime().Adjust(prevTimeStep.StepResolution(), -paramStep);

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

			myThreadedLogger->Warning("Data not found: not calculating " + myTargetInfo->Param().Name() + " for step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()));

			continue;
		}

		myThreadedLogger->Debug("Previous data step is " + boost::lexical_cast<string> (prevSumInfo->Time().Step()));
		myThreadedLogger->Debug("Current/next data step is " + boost::lexical_cast<string> (curSumInfo->Time().Step()));

		double scaleFactor = 1.;

		// EC gives precipitation in meters, we are calculating millimeters

		if (curSumInfo->Param().Unit() == kM
			 || (myTargetInfo->Producer().Id() == 240 && !isRadiationCalculation)) // HIMAN-98
		{
			scaleFactor = 1000.;
		}

		string deviceType = "CPU";

		step = static_cast<int> (curSumInfo->Time().Step() - prevSumInfo->Time().Step());
		
		if (isRadiationCalculation)
		{

			/*
			 * Radiation unit is W/m^2 which is J/m^2/s, so we need to convert
			 * time to seconds.
			 *
			 * Step is always one hour or more, even in the case of Harmonie.
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
			 * This has been agreed with AKS.
			 *
			 */

			step = 1;
		}

		double invstep = 1./step;

		LOCKSTEP(myTargetInfo, curSumInfo, prevSumInfo)
		{

			double currentSum = curSumInfo->Value();
			double previousSum = prevSumInfo->Value();

			if (currentSum == kFloatMissing || previousSum == kFloatMissing)
			{
				continue;
			}

			double sum = currentSum - previousSum;

			if (isRateCalculation && step != 1)
			{
				sum *= invstep;
			}

			if (sum < 0 && parmName != "RTOPLW-WM2")
			{
				sum = 0;
			}

			sum *= scaleFactor;

			myTargetInfo->Value(sum);

		}

		myThreadedLogger->Info("[" + deviceType + "] Parameter " + parmName + " missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));
	}
}

pair<shared_ptr<himan::info>,shared_ptr<himan::info>> split_sum::GetSourceDataForRate(shared_ptr<const info> myTargetInfo, int step)
{
	shared_ptr<info> prevInfo;
	shared_ptr<info> curInfo;

	HPTimeResolution timeResolution = myTargetInfo->Time().StepResolution();

	// 1. Assuming we *know* what the step is, fetch previous and current
	// based on that step.

	if (step != kHPMissingInt)
	{
		if (myTargetInfo->Producer().Id() == 210)
		{
			step = 60;	// Forecast step is 15 (Harmonie), but it has been agreed
						// with AKS that we'll use one hour since editor displays
						// only hourly data.
		}

		forecast_time wantedTimeStep(myTargetInfo->Time());
		wantedTimeStep.ValidDateTime().Adjust(timeResolution, -step);

		if (wantedTimeStep.Step() >= 0)
		{
			prevInfo = FetchSourceData(myTargetInfo,wantedTimeStep);
		}
	}
	else
	{
		itsLogger->Debug("Configuration file does not have key 'step': trying to guess correct step");
	}

	curInfo = FetchSourceData(myTargetInfo,myTargetInfo->Time());

	if (curInfo && prevInfo)
	{
		itsLogger->Debug("Found previous and current data");
		return make_pair(prevInfo,curInfo);
	}

	// 2. Data was not found on the requested steps. Now we have to scan the database
	// for data which is slow.
	
	itsLogger->Debug("Scanning database for source data");
	
	int maxSteps = 6; // by default look for 6 hours forward or backward
	step = 1; // by default the difference between time steps is one (ie. one hour))

	if (myTargetInfo->Producer().Id() == 210)
	{
		step = 60;	// see comment on line 512
	}
	else if (timeResolution != kHourResolution)
	{
		throw runtime_error(ClassName() + ": Invalid time resolution value: " + HPTimeResolutionToString.at(timeResolution));
	}

	itsLogger->Trace("Target time is " + static_cast<string> (myTargetInfo->Time().ValidDateTime()));

	if (!prevInfo)
	{
		itsLogger->Debug("Searching for previous data");

		// start going backwards in time and search for the
		// first data that exists

		forecast_time wantedTimeStep(myTargetInfo->Time());

		for (int i = 0; !prevInfo && i <= maxSteps*step; i++)
		{
			wantedTimeStep.ValidDateTime().Adjust(timeResolution, -step);

			if (wantedTimeStep.Step() < 0)
			{
				continue;
			}

			itsLogger->Debug("Trying time " + static_cast<string> (wantedTimeStep.ValidDateTime()));
			prevInfo = FetchSourceData(myTargetInfo,wantedTimeStep);

			if (prevInfo)
			{
				itsLogger->Trace("Found previous data");
			}
		}
	}

	if (!curInfo)
	{
		itsLogger->Debug("Searching for next data");

		// start going forwards in time and search for the
		// first data that exists

		forecast_time wantedTimeStep(myTargetInfo->Time());

		for (int i = 0; !curInfo && i <= maxSteps*step; i++)
		{
			wantedTimeStep.ValidDateTime().Adjust(timeResolution, step);

			itsLogger->Debug("Trying time " + static_cast<string> (wantedTimeStep.ValidDateTime()));
			curInfo = FetchSourceData(myTargetInfo,wantedTimeStep);

			if (curInfo)
			{
				itsLogger->Trace("Found current data");
			}
		}
	}
	
	return make_pair(prevInfo,curInfo);
}

shared_ptr<himan::info> split_sum::FetchSourceData(shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime)
{
	level wantedLevel(kHeight, 0 ,"HEIGHT");

	// Transform ground level based on only the first source parameter

	auto params = sourceParameters[myTargetInfo->Param().Name()];

	// Must have source parameter for target parameter defined in map sourceParameters

	assert(!params.empty());

	if (myTargetInfo->Param().Name() == "RTOPLW-WM2")
	{
		wantedLevel = level(kTopOfAtmosphere, 0, "TOP");
	}

	shared_ptr<info> SumInfo = Fetch(wantedTime, wantedLevel, params, myTargetInfo->ForecastType());

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
		SumInfo->Data().Fill(0);

	}
	
	return SumInfo;

}

void split_sum::WriteToFile(const info& targetInfo, const write_options& opts) const
{
	auto aWriter = GET_PLUGIN(writer);
	aWriter->WriteOptions(opts);

	// writing might modify iterator positions --> create a copy

	auto tempInfo = targetInfo;

	if (itsConfiguration->FileWriteOption() == kDatabase || itsConfiguration->FileWriteOption() == kMultipleFiles)
	{
		// If info holds multiple parameters, we must loop over them all
		// Note! We only loop over the parameters, not over the times or levels!

		tempInfo.ResetParam();

		while (tempInfo.NextParam())
		{
			if (itsConfiguration->FileWriteOption() == kDatabase && tempInfo.Data().Size() == tempInfo.Data().MissingCount())
			{
				itsLogger->Info("All data missing for " + tempInfo.Param().Name() + " step " + boost::lexical_cast<string> (tempInfo.Time().Step()) + ", not writing to disk");
				continue;
			}

			aWriter->ToFile(tempInfo, itsConfiguration);
		}
	}
	else if (itsConfiguration->FileWriteOption() == kSingleFile)
	{
		aWriter->ToFile(tempInfo, itsConfiguration, itsConfiguration->ConfigurationFile());
	}
}
