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
#include "writer.h"
#include "neons.h"
#include "pcuda.h"

#undef HIMAN_AUXILIARY_INCLUDE

#ifdef DEBUG
#include "timer_factory.h"
#endif

using namespace std;
using namespace himan::plugin;

precipitation::precipitation() : itsUseCuda(false)
{
	itsClearTextFormula = "RR = RR_cur - RR_prev";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("precipitation"));

}

void precipitation::Process(std::shared_ptr<const plugin_configuration> conf)
{

	unique_ptr<timer> initTimer;

	if (conf->StatisticsEnabled())
	{
		initTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());
		initTimer->Start();
	}
	
	shared_ptr<plugin::pcuda> c = dynamic_pointer_cast<plugin::pcuda> (plugin_factory::Instance()->Plugin("pcuda"));

	if (c->HaveCuda())
	{
		string msg = "I possess the powers of CUDA";

		if (!conf->UseCuda())
		{
			msg += ", but I won't use them";
		}
		else
		{
			msg += ", and I'm not afraid to use them";
			itsUseCuda = true;
		}

		itsLogger->Info(msg);

	}

	// Get number of threads to use

	unsigned short threadCount = ThreadCount(conf->ThreadCount());

	if (conf->StatisticsEnabled())
	{
		conf->Statistics()->UsedThreadCount(threadCount);
		conf->Statistics()->UsedGPUCount(0);
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

	param requestedParam ("RR-6-MM", 355);

	if (conf->Exists("rr1h") && conf->GetValue("rr1h") == "true")
	{
		params.push_back(param("RR-1-MM", 353));
	}

	if (conf->Exists("rr3h") && conf->GetValue("rr3h") == "true")
	{
		params.push_back(param("RR-3-MM", 354));
	}

	if (conf->Exists("rr6h") && conf->GetValue("rr6h") == "true")
	{
		params.push_back(param("RR-6-MM", 355));
	}

	if (conf->Exists("rr12h") && conf->GetValue("rr12h") == "true")
	{
		params.push_back(param("RR-12-MM", 356));
	}

	if (params.empty())
	{
		itsLogger->Trace("No parameter definition given, defaulting to rr6h");
		params.push_back(param("RR-6-MM", 355));
	}

	// GRIB 1

	if (conf->OutputFileType() == kGRIB1)
	{
		shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		for (size_t i = 0; i < params.size(); i++)
		{
			param p = params[i];
			
			long parm_id = n->NeonsDB().GetGridParameterId(targetInfo->Producer().TableVersion(), p.Name());
			p.GribIndicatorOfParameter(parm_id);
			p.GribTableVersion(targetInfo->Producer().TableVersion());

			params[i] = p;
		}
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
	FeederInfo()->Param(requestedParam);

	if (conf->StatisticsEnabled())
	{
		initTimer->Stop();
		conf->Statistics()->AddToInitTime(initTimer->GetTime());
	}

	/*
	 * Each thread will have a copy of the target info.
	 */

	unique_ptr<timer> processTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());

	if (conf->StatisticsEnabled())
	{
		processTimer->Start();
	}

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
		processTimer->Stop();
		conf->Statistics()->AddToProcessingTime(processTimer->GetTime());
	}

	if (conf->FileWriteOption() == kSingleFile)
	{

		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		string theOutputFile = conf->ConfigurationFile();

		theWriter->ToFile(targetInfo, conf, theOutputFile);

	}
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
		shared_ptr<info> prevInfo;

		assert(myTargetInfo->Time().StepResolution() == kHourResolution);

		for (myTargetInfo->ResetParam(); myTargetInfo->NextParam(); )
		{
			shared_ptr<info> RRInfo;

			long paramStep;

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

			myThreadedLogger->Warning("Calculating parameter " + myTargetInfo->Param().Name() + " time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
												  " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
			try
			{
				if (!prevInfo || myTargetInfo->Time().Step() - prevInfo->Time().Step() != paramStep)
				{

					/*
					 * If this is the first time this loop executed, or if the previous data is not suitable
					 * for calculating against current time step data, fetch source data.
					 */

					forecast_time prevTimeStep = myTargetInfo->Time();

					if (prevTimeStep.Step() >= paramStep)
					{
						prevTimeStep.ValidDateTime()->Adjust(kHourResolution, -paramStep);

						prevInfo = FetchSourcePrecipitation(conf,prevTimeStep,myTargetInfo->Level(),dataFoundFromRRParam);
					}
					else
					{
						continue;
					}
				}

				// Get data for current step

				RRInfo = FetchSourcePrecipitation(conf,myTargetInfo->Time(),myTargetInfo->Level(),dataFoundFromRRParam);

			}
			catch (HPExceptionType e)
			{
				switch (e)
				{
					case kFileDataNotFound:
						itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
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

			myThreadedLogger->Debug("Calculating parameter " + myTargetInfo->Param().Name() + " time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
												  " level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

			shared_ptr<NFmiGrid> RRGrid(RRInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> prevGrid(prevInfo->Grid()->ToNewbaseGrid());

			shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

			int missingCount = 0;
			int count = 0;

			bool equalGrids = (*myTargetInfo->Grid() == *RRInfo->Grid());

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

					if (!myTargetInfo->Value(RR))
					{
						throw runtime_error(ClassName() + ": Failed to set value to matrix");
					}

				}

				prevInfo = RRInfo;

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

		   if (conf->FileWriteOption() == kNeons || conf->FileWriteOption() == kMultipleFiles)
		   {
			   shared_ptr<writer> w = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

			   shared_ptr<info> tempInfo(new info(*myTargetInfo));

			   for (tempInfo->ResetParam(); tempInfo->NextParam(); )
			   {
				   w->ToFile(tempInfo, conf);
			   }
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
