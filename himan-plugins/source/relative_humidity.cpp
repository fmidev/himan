/**
 * @file relative_humidity.cpp
 *
 * @date Jan 21, 2012
 * @author partio
 */

#include "relative_humidity.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

const double b = 17.27;
const double c = 237.3;
const double d = 1.8;
const double e = 0.622; // ratio molecular weight of water vapor /dry air

relative_humidity::relative_humidity()
{
	itsClearTextFormula = "RH = 100 *  (P * Q / 0.622 / es) * (P - es) / (P - Q * P / 0.622)";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("relative_humidity"));

}

void relative_humidity::Process(shared_ptr<const plugin_configuration> conf)
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
	 * Set target parameter to potential temperature
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> params;

	param requestedParam ("RH-PRCNT", 13);

	// GRIB 2

	requestedParam.GribDiscipline(0);
	requestedParam.GribCategory(1);
	requestedParam.GribParameter(1);

	params.push_back(requestedParam);

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
	FeederInfo()->Param(requestedParam);

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

		boost::thread* t = new boost::thread(&relative_humidity::Run,
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

void relative_humidity::Run(shared_ptr<info> myTargetInfo,
			   const shared_ptr<const plugin_configuration> conf,
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

void relative_humidity::Calculate(shared_ptr<info> myTargetInfo,
					 const shared_ptr<const plugin_configuration> conf,
					 unsigned short threadIndex)
{

	
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	param PParam("P-HPA");
	param QParam("Q-KGKG");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("relative_humidityThread #" + boost::lexical_cast<string> (threadIndex)));
	
	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	bool useCudaInThisThread = false;

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		double TBase = 0;
		//double TDBase = 0;

		shared_ptr<info> TInfo;
		shared_ptr<info> PInfo;
		shared_ptr<info> QInfo;

		try
		{
			TInfo = f->Fetch(conf,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								TParam,
								conf->UseCudaForPacking() && useCudaInThisThread);

			PInfo = f->Fetch(conf,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								PParam,
								conf->UseCudaForPacking() && useCudaInThisThread);

			QInfo = f->Fetch(conf,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								QParam,
								conf->UseCudaForPacking() && useCudaInThisThread);

		}
		catch (HPExceptionType e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value

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

		assert(TInfo->Grid()->AB() == PInfo->Grid()->AB());
		
		SetAB(myTargetInfo, TInfo);

		if (TInfo->Param().Unit() == kK)
		{
			TBase = -273.15;
		}

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> PGrid(PInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> QGrid(QInfo->Grid()->ToNewbaseGrid());

		size_t missingCount = 0;
		size_t count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() && *myTargetInfo->Grid() == *QInfo->Grid() && *myTargetInfo->Grid() == *PInfo->Grid());

		string deviceType;

		{
			deviceType = "CPU";

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{
				count++;

				double T = kFloatMissing;
				double P = kFloatMissing;
				double Q = kFloatMissing;

				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
				InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
				InterpolateToPoint(targetGrid, QGrid, equalGrids, Q);

				if (T == kFloatMissing || P == kFloatMissing || Q == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				T += TBase;
				
				double es = util::Es(T) ;

				double RH = (P * Q / e / es) * (P - es) / (P - Q * P / e);

				if (RH > 1.0)
				{
					RH = 1.0;
				}
				else if (RH < 0.0)
				{
					RH = 0.0;
				}

				RH *= 100;

				if (!myTargetInfo->Value(RH))
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
		 * Now we are done for this level
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
