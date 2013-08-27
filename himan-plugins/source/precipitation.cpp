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

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

precipitation::precipitation()
{
	itsClearTextFormula = "RR = RR_cur - RR_prev";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("precipitation"));

}

void precipitation::Process(std::shared_ptr<const plugin_configuration> conf)
{

	unique_ptr<timer> aTimer;

	// Get number of threads to use

	unsigned short threadCount = ThreadCount(conf->ThreadCount());

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
	baseParam.Aggregation().AggregationType(kAccumulation);
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

	for (size_t i = 0; i < threadCount; i++)
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

	WriteToFile(conf, targetInfo);
}

void precipitation::Run(shared_ptr<info> myTargetInfo,
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

void precipitation::Calculate(shared_ptr<info> myTargetInfo,
					 shared_ptr<const plugin_configuration> conf,
					 unsigned short threadIndex)
{

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("precipitationThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	bool dataFoundFromRRParam = true; // Assume that we have parameter RR-KGM2 present

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
		{
			shared_ptr<info> RRInfo;

			int paramStep;

			if (myTargetInfo->Param().Name() == "RR-1-MM")
			{
				paramStep = 1;
			}
			else if (myTargetInfo->Param().Name() == "RR-3-MM")
			{
				paramStep = 3;
			}
			else if (myTargetInfo->Param().Name() == "RR-6-MM")
			{
				paramStep = 6;
			}
			else if (myTargetInfo->Param().Name() == "RR-12-MM")
			{
				paramStep = 12;
			}
			else
			{
				throw runtime_error(ClassName() + ": Unsupported parameter: " + myTargetInfo->Param().Name());
			}

			if (myTargetInfo->Time().StepResolution() != kHourResolution)
			{
				assert(myTargetInfo->Time().StepResolution() == kMinuteResolution);

				paramStep *= 60;
			}

			shared_ptr<info> prevInfo;

			try
			{

				forecast_time prevTimeStep = myTargetInfo->Time();

				if (prevTimeStep.Step() >= paramStep && prevTimeStep.Step() - paramStep >= 0)
				{
					prevTimeStep.ValidDateTime()->Adjust(myTargetInfo->Time().StepResolution(), -paramStep);

					prevInfo = FetchSourcePrecipitation(conf,prevTimeStep,myTargetInfo->Level(),dataFoundFromRRParam);
				}
				else
				{
					// Unable to get data for this step, as target time is smaller than step time
					// (f.ex fcst_per = 2, and target parameter is rr3h
			
					myThreadedLogger->Info("Not calculating " + myTargetInfo->Param().Name() + " for step " + boost::lexical_cast<string> (prevTimeStep.Step()));

					if (conf->StatisticsEnabled())
					{
						conf->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
						conf->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
					}

					/*
					 * If we want to write empty files to disk, comment out the following code
					 */

					/*

					myTargetInfo->Data()->Fill(kFloatMissing);

					if (conf->FileWriteOption() == kNeons || conf->FileWriteOption() == kMultipleFiles)
					{
						shared_ptr<writer> w = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

						w->ToFile(shared_ptr<info> (new info(*myTargetInfo)), conf);
					}
					*/

					continue;
				}

				// Get data for current step

				RRInfo = FetchSourcePrecipitation(conf,myTargetInfo->Time(),myTargetInfo->Level(),dataFoundFromRRParam);

			}
			catch (HPExceptionType e)
			{
				switch (e)
				{
					case kFileDataNotFound:
						myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
						myTargetInfo->Data()->Fill(kFloatMissing);

						if (conf->StatisticsEnabled())
						{
							conf->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
							conf->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
						}

						continue;
						break;

					default:
						throw runtime_error(ClassName() + ": Unable to proceed");
						break;
				}
			}

			myThreadedLogger->Info("Calculating parameter " + myTargetInfo->Param().Name() + " time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
												  " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

			shared_ptr<NFmiGrid> RRGrid(RRInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> prevGrid(prevInfo->Grid()->ToNewbaseGrid());

			shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

			int missingCount = 0;
			int count = 0;

			bool equalGrids = (*myTargetInfo->Grid() == *RRInfo->Grid());

			double scaleFactor = 1.;

			// EC gives precipitation in meters, we are calculating millimeters

			if (RRInfo->Param().Unit() == kM)
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

shared_ptr<himan::info> precipitation::FetchSourcePrecipitation(shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime, const level& wantedLevel, bool& dataFoundFromRRParam)
{

	try
	{
		if (dataFoundFromRRParam)
		{

			return FetchSourceRR(conf,
								wantedTime,
								wantedLevel);

		}
		else
		{
			return FetchSourceConvectiveAndLSRR(conf,
												wantedTime,
												wantedLevel);
		}
	}
	catch (HPExceptionType e)
	{

		if (e == kFileDataNotFound && dataFoundFromRRParam)
		{
			try
			{


				shared_ptr<info> i = FetchSourceConvectiveAndLSRR(conf,
						wantedTime,
						wantedLevel);

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


shared_ptr<himan::info> precipitation::FetchSourceRR(shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime, const level& wantedLevel)
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

shared_ptr<himan::info> precipitation::FetchSourceConvectiveAndLSRR(shared_ptr<const plugin_configuration> conf, const forecast_time& wantedTime, const level& wantedLevel)
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
