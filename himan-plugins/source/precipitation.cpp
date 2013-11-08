/**
 * @file precipitation.cpp
 *
 * @date Jan 28, 2012
 * @author partio
 */

#include "precipitation.h"
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

map<string,string> sourceParameters;

precipitation::precipitation()
{
	itsClearTextFormula = "RRsum = RR_cur - RR_prev; RR = (RR_cur - RR_prev) / step";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("precipitation"));
	
	sourceParameters["rr1h"] = "RR-KGM2";
	sourceParameters["rr3h"] = "RR-KGM2";
	sourceParameters["rr6h"] = "RR-KGM2";
	sourceParameters["rr12h"] = "RR-KGM2";
	sourceParameters["rr"] = "RR-KGM2";
	sourceParameters["snr"] = "SNACC-KGM2";

}

void precipitation::Process(std::shared_ptr<const plugin_configuration> conf)
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
	 * Set target parameter to precipitation.
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> params;

	param baseParam;
	baseParam.GribDiscipline(0);
	baseParam.GribCategory(1);
	baseParam.GribParameter(8);
	baseParam.Aggregation().Type(kAccumulation);
	baseParam.Aggregation().TimeResolution(kHourResolution);

	if (conf->Exists("rr1h") && conf->GetValue("rr1h") == "true")
	{
		baseParam.Name("RR-1-MM");
		baseParam.UnivId(353);
		baseParam.Aggregation().TimeResolutionValue(1);

		params.push_back(baseParam);

	}

	if (conf->Exists("rr3h") && conf->GetValue("rr3h") == "true")
	{
		baseParam.Name("RR-3-MM");
		baseParam.UnivId(354);
		baseParam.Aggregation().TimeResolutionValue(3);

		params.push_back(baseParam);
	}

	if (conf->Exists("rr6h") && conf->GetValue("rr6h") == "true")
	{
		baseParam.Name("RR-6-MM");
		baseParam.UnivId(355);
		baseParam.Aggregation().TimeResolutionValue(6);

		params.push_back(baseParam);
	}

	if (conf->Exists("rr12h") && conf->GetValue("rr12h") == "true")
	{
		baseParam.Name("RR-12-MM");
		baseParam.UnivId(356);
		baseParam.Aggregation().TimeResolutionValue(12);

		params.push_back(baseParam);
	}

	if (conf->Exists("rr") && conf->GetValue("rr") == "true")
	{
		baseParam.Name("RRR-KGM2");
		baseParam.UnivId(49);
		baseParam.Aggregation().TimeResolutionValue(1);

		params.push_back(baseParam);
	}

	if (conf->Exists("snr") && conf->GetValue("snr") == "true")
	{
		baseParam.Name("SNR-KGM2");
		baseParam.UnivId(264);
		baseParam.Aggregation().TimeResolutionValue(1);

		params.push_back(baseParam);
	}

	if (params.empty())
	{
		itsLogger->Trace("No parameter definition given, defaulting to rr6h");
		
		baseParam.Name("RR-6-MM");
		baseParam.UnivId(355);
		baseParam.Aggregation().TimeResolutionValue(6);

		params.push_back(baseParam);
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

		boost::thread* t = new boost::thread(&precipitation::Run,
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

void precipitation::Run(shared_ptr<info> myTargetInfo,
				shared_ptr<const plugin_configuration> conf,
				unsigned short threadIndex)
{
	// Thread scope variables for source data; useful when calculating rate
	// when step > 1 and using only one thread

	// g++ supports C++11-style thread-local variables only at version 4.8

	shared_ptr<himan::info> curRRInfo;
	shared_ptr<himan::info> prevRRInfo;

	while (AdjustLeadingDimension(myTargetInfo))
	{
		Calculate(myTargetInfo, conf, threadIndex, curRRInfo, prevRRInfo);
	}

}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void precipitation::Calculate(shared_ptr<info> myTargetInfo,
					 shared_ptr<const plugin_configuration> conf,
					 unsigned short threadIndex,
					shared_ptr<info> curRRInfo,
					shared_ptr<info> prevRRInfo)
{

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("precipitationThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	bool dataFoundFromRRParam = true; // Assume that we have parameter RR-KGM2 present

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
		{
			
			bool isRRCalculation = (myTargetInfo->Param().Name() == "RRR-KGM2" || myTargetInfo->Param().Name() == "SNR-KGM2");

			/*
			 * Move current data to previous data and fetch new current data IF
			 *
			 * - calculating SUM
			 * - calculating RATE and current step is larger than current source data step
			 */
			
			if (isRRCalculation && (curRRInfo && myTargetInfo->Time().Step() > curRRInfo->Time().Step()))
			{
				prevRRInfo = curRRInfo;
				curRRInfo.reset();
			}
			else if (!isRRCalculation)
			{
				prevRRInfo.reset();
				curRRInfo.reset();
			}

			/*
			 * Two modes of operation:
			 *
			 * 1) When calculating precipitation sums, always get the previous
			 * step value from the current step and get both values (current and
			 * previous). If either can't be found, skip time step.
			 *
			 * 2) When calculating precipitation rate, get the first data that's
			 * earlier or same than current time step and the next data that's
			 * later or same than the current time step. Then calculate rate
			 * based on those values.
			 */
			
			if (isRRCalculation)
			{
				// Calculating RATE

				// Previous VALID data
				if (!prevRRInfo)
				{
					myThreadedLogger->Debug("Searching for previous time step data");
					prevRRInfo = GetSourceDataForRate(conf, myTargetInfo, dataFoundFromRRParam, false);
				}
	
				// Next VALID data
				if (!curRRInfo)
				{
					myThreadedLogger->Debug("Searching for next time step data");
					curRRInfo = GetSourceDataForRate(conf, myTargetInfo, dataFoundFromRRParam, true);
				}

			}
			else
			{
				// Calculating SUM

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
					if (!prevRRInfo)
					{
						forecast_time prevTimeStep = myTargetInfo->Time();

						prevTimeStep.ValidDateTime()->Adjust(prevTimeStep.StepResolution(), -paramStep);

						prevRRInfo = GetSourceDataForSum(conf, myTargetInfo, prevTimeStep, dataFoundFromRRParam);

#ifdef EMULATE_ZERO_TIMESTEP_DATA

						If model does not provide data for timestep 0 and we want to emulate
						that by providing a zero-grid, enable this pre-process block

						if (myTargetInfo->Time().Step().Step() == paramStep)
						{
							/*
							 * If current timestep equals to target param aggregation time value,
							 * use constant 0 field as previous data.
							 *
							 * For example for param RR-3-H if step is 3.
							 */

							prevRRInfo = make_shared<info> (*myTargetInfo);
							vector<param> temp = { param(sourceParameters[myTargetInfo->Param().Name()]) };
							prevRRInfo->Params(temp);
							prevRRInfo->Create();
							prevRRInfo->First();
							prevRRInfo->Grid()->Data()->Fill(0);

						}
#endif
					}
					
					// Data from current time step
					if (!curRRInfo)
					{
						curRRInfo = GetSourceDataForSum(conf, myTargetInfo, myTargetInfo->Time(), dataFoundFromRRParam);
					}
				}
			}

			if (!prevRRInfo || !curRRInfo)
			{
				// Data was not found

				myThreadedLogger->Info("Not calculating " + myTargetInfo->Param().Name() + " for step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()));

				myTargetInfo->Data()->Fill(kFloatMissing);

				if (conf->StatisticsEnabled())
				{
					conf->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
					conf->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
				}

#ifdef WRITE_EMPTY_FILES

				/*
				 * If empty files need to be written to disk, enable the following code
				 */

				myTargetInfo->Data()->Fill(kFloatMissing);

				if (conf->FileWriteOption() == kNeons || conf->FileWriteOption() == kMultipleFiles)
				{
					shared_ptr<writer> w = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

					w->ToFile(shared_ptr<info> (new info(*myTargetInfo)), conf);
				}

#endif
				continue;
			}

			myThreadedLogger->Trace("Previous data step is " + boost::lexical_cast<string> (prevRRInfo->Time().Step()));
			myThreadedLogger->Trace("Current/next data step is " + boost::lexical_cast<string> (curRRInfo->Time().Step()));

			myThreadedLogger->Info("Calculating parameter " + myTargetInfo->Param().Name() + " time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
												  " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

			shared_ptr<NFmiGrid> RRGrid(curRRInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> prevGrid(prevRRInfo->Grid()->ToNewbaseGrid());

			shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

			size_t missingCount = 0;
			size_t count = 0;

			bool equalGrids = (*myTargetInfo->Grid() == *curRRInfo->Grid());

			double scaleFactor = 1.;

			// EC gives precipitation in meters, we are calculating millimeters

			if (curRRInfo->Param().Unit() == kM)
			{
				scaleFactor = 1000.;
			}

			string deviceType = "CPU";

			{
				assert(targetGrid->Size() == myTargetInfo->Data()->Size());

				myTargetInfo->ResetLocation();

				targetGrid->Reset();
				prevGrid->Reset();

				while (myTargetInfo->NextLocation() && targetGrid->Next() && prevGrid->Next())
				{
					count++;

					double RRcur = kFloatMissing;
					double RRprev = kFloatMissing;

					InterpolateToPoint(targetGrid, RRGrid, equalGrids, RRcur);
					InterpolateToPoint(targetGrid, prevGrid, equalGrids, RRprev);

					if (RRcur == kFloatMissing || RRprev == kFloatMissing)
					{
						missingCount++;

						myTargetInfo->Value(kFloatMissing);
						continue;
					}

					double RR = RRcur - RRprev;

					if (myTargetInfo->Param().Name() == "RRR-KGM2")
					{
						long step = curRRInfo->Time().Step() - prevRRInfo->Time().Step();

						RR /= static_cast<double> (step);
					}

					if (RR < 0)
					{
						RR = 0;
					}

					RR *= scaleFactor;

					if (!myTargetInfo->Value(RR))
					{
						throw runtime_error(ClassName() + ": Failed to set value to matrix");
					}

				}

				/*
				 * Newbase normalizes scanning mode to bottom left -- if that's not what
				 * the target scanning mode is, we have to swap the data back.
				 */

				SwapTo(myTargetInfo, kBottomLeft);

			}

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

			myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

			if (conf->FileWriteOption() != kSingleFile)
			{
				WriteToFile(conf, myTargetInfo);
			}
		}
	}
}

shared_ptr<himan::info> precipitation::GetSourceDataForRate(shared_ptr<const plugin_configuration> conf, shared_ptr<const info> myTargetInfo, bool& dataFoundFromRRParam, bool forward)
{
	shared_ptr<info> RRInfo;

	/*
	 * With RR we don't set param step
	 */
	
	int paramStep = 6; // by default look for 12 hours (time steps) forward or backward

	int i = 0;

	if (!forward)
	{
		/*
		 * When searching backwards, we always take the next time step, the current
		 * one won't do!
		 *
		 * When searching forwards it's ok to use the current time step if
		 * that time step has data.
		 */
		
		i++;
	}

	while (!RRInfo && i <= paramStep)
	{
		int step = i++;

		try
		{

			forecast_time wantedTimeStep = myTargetInfo->Time();

			if (!forward)
			{
				step *= -1;
			}
			
			wantedTimeStep.ValidDateTime()->Adjust(kHourResolution, step);

			RRInfo = FetchSourcePrecipitation(conf,myTargetInfo,wantedTimeStep,dataFoundFromRRParam);

		}
		catch (HPExceptionType e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					break;

				default:
					throw runtime_error(ClassName() + ": Unable to proceed");
					break;
			}
		}

	}

	return RRInfo;
}

shared_ptr<himan::info> precipitation::GetSourceDataForSum(shared_ptr<const plugin_configuration> conf, shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime, bool& dataFoundFromRRParam)
{
	shared_ptr<info> RRInfo;
	
	try
	{
		RRInfo = FetchSourcePrecipitation(conf,myTargetInfo,wantedTime,dataFoundFromRRParam);
	}
	catch (HPExceptionType e)
	{
		switch (e)
		{
			case kFileDataNotFound:
				break;

			default:
				throw runtime_error(ClassName() + ": Unable to proceed");
				break;
		}
	}

	return RRInfo;
}

shared_ptr<himan::info> precipitation::FetchSourcePrecipitation(shared_ptr<const plugin_configuration> conf, shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime, bool& dataFoundFromRRParam)
{

	level groundLevel(kHeight, 0 ,"HEIGHT");
	param sourceParam(sourceParameters[myTargetInfo->Param().Name()]);
	
	groundLevel = LevelTransform(conf->SourceProducer(), sourceParam, groundLevel);

	try
	{
		if (dataFoundFromRRParam)
		{

			return FetchSourceRR(conf,myTargetInfo,wantedTime,groundLevel);

		}
		else
		{
			return FetchSourceConvectiveAndLSRR(conf,myTargetInfo,wantedTime,groundLevel);
		}
	}
	catch (HPExceptionType e)
	{

		if (e == kFileDataNotFound && dataFoundFromRRParam)
		{
			try
			{


				shared_ptr<info> i = FetchSourceConvectiveAndLSRR(conf,myTargetInfo,wantedTime,groundLevel);

				dataFoundFromRRParam = false;

				return i;

			}
			catch (HPExceptionType e)
			{
				throw e;
			}

		}
		else
		{
			throw e;
		}
	}

	throw runtime_error(ClassName() + ": We should never be here but compiler forces this statement");
}


shared_ptr<himan::info> precipitation::FetchSourceRR(shared_ptr<const plugin_configuration> conf, shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime, const level& wantedLevel)
{
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	try
	{
		return f->Fetch(conf,
						wantedTime,
						wantedLevel,
						param("RR-KGM2"));
		}
	catch (HPExceptionType e)
	{
		throw e;
	}

}

shared_ptr<himan::info> precipitation::FetchSourceConvectiveAndLSRR(shared_ptr<const plugin_configuration> conf, shared_ptr<const info> myTargetInfo, const forecast_time& wantedTime, const level& wantedLevel)
{
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	shared_ptr<himan::info> ConvectiveInfo;
	shared_ptr<himan::info> LSInfo;
	shared_ptr<himan::info> RRInfo;

	try
	{

		ConvectiveInfo = f->Fetch(conf,
						wantedTime,
						wantedLevel,
						param("RRC-KGM2"));

		LSInfo = f->Fetch(conf,
						wantedTime,
						wantedLevel,
						param("RRR-KGM2"));
	}
	catch (HPExceptionType e)
	{
		throw e;
	}

	ConvectiveInfo->ResetLocation();
	LSInfo->ResetLocation();

	assert(*ConvectiveInfo->Grid() == *LSInfo->Grid());

	RRInfo->Data()->Resize(ConvectiveInfo->Data()->SizeX(),LSInfo->Data()->SizeY());

	while (ConvectiveInfo->NextLocation() && LSInfo->NextLocation() && RRInfo->NextLocation())
	{
		double C = ConvectiveInfo->Value();
		double LS = LSInfo->Value();

		RRInfo->Value(C+LS);
	}

	return RRInfo;
}
