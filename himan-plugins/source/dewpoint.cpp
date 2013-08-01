/**
 * @file dewpoint.cpp
 *
 * @date Jan 21, 2012
 * @author partio
 */

#include "dewpoint.h"
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

#include "dewpoint_cuda.h"
#include "cuda_helper.h"

using namespace std;
using namespace himan::plugin;

const double RW = 461.5; // Vesihoyryn kaasuvakio (J / K kg)
const double L = 2.5e6; // Veden hoyrystymislampo (J / kg)
const double RW_div_L = RW / L;

dewpoint::dewpoint() : itsUseCuda(false)
{
	itsClearTextFormula = "Td = T / (1 - (T * ln(RH)*(Rw/L)))";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("dewpoint"));

}

void dewpoint::Process(shared_ptr<const plugin_configuration> conf)
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
		
		itsCudaDeviceCount = c->DeviceCount();

	}

	// Get number of threads to use

	unsigned short threadCount = ThreadCount(conf->ThreadCount());

	if (conf->StatisticsEnabled())
	{
		conf->Statistics()->UsedThreadCount(threadCount);
		conf->Statistics()->UsedGPUCount(itsCudaDeviceCount);
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

	param requestedParam ("TD-C", 10);

	// GRIB 2

	requestedParam.GribDiscipline(0);
	requestedParam.GribCategory(0);
	requestedParam.GribParameter(6);

	// GRIB 1

	if (conf->OutputFileType() == kGRIB1)
	{
		shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		long parm_id = n->NeonsDB().GetGridParameterId(targetInfo->Producer().TableVersion(), requestedParam.Name());
		requestedParam.GribIndicatorOfParameter(parm_id);
		requestedParam.GribTableVersion(targetInfo->Producer().TableVersion());

	}

	params.push_back(requestedParam);

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

		boost::thread* t = new boost::thread(&dewpoint::Run,
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
	
	itsLogger->Info("Threads finished");

	if (conf->FileWriteOption() == kSingleFile)
	{
		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		string theOutputFile = conf->ConfigurationFile();

		theWriter->ToFile(targetInfo, conf, theOutputFile);

	}
}

void dewpoint::Run(shared_ptr<info> myTargetInfo,
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

void dewpoint::Calculate(shared_ptr<info> myTargetInfo,
					 const shared_ptr<const plugin_configuration> conf,
					 unsigned short threadIndex)
{

	
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	param RHParam("RH-PRCNT");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("dewpointThread #" + boost::lexical_cast<string> (threadIndex)));
	
	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	bool useCudaInThisThread = itsUseCuda && threadIndex <= itsCudaDeviceCount;

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		double TBase = 0;

		shared_ptr<info> TInfo;
		shared_ptr<info> RHInfo;

		try
		{
			TInfo = f->Fetch(conf,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								TParam,
								conf->UseCudaForPacking() && useCudaInThisThread);

			RHInfo = f->Fetch(conf,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								RHParam,
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

		assert(TInfo->Grid()->AB() == RHInfo->Grid()->AB());
		
		SetAB(myTargetInfo, TInfo);

		assert(RHInfo->Param().Unit() == kPrcnt);

		if (TInfo->Param().Unit() == kC)
		{
			TBase = 273.15;
		}

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RHGrid(RHInfo->Grid()->ToNewbaseGrid());

		int missingCount = 0;
		int count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() && *myTargetInfo->Grid() == *RHInfo->Grid());

		string deviceType;

#ifdef HAVE_CUDA
		
		if (itsUseCuda && equalGrids && threadIndex <= itsCudaDeviceCount)
		{

			deviceType = "GPU";

			dewpoint_cuda::dewpoint_cuda_options opts;
			dewpoint_cuda::dewpoint_cuda_data datas;

			opts.N = TInfo->Data()->Size();
			opts.cudaDeviceIndex = threadIndex-1;

			opts.TBase = TBase;

			CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.TD), opts.N * sizeof(double), cudaHostAllocMapped));
			
			if (TInfo->Grid()->DataIsPacked())
			{
				assert(TInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> t = dynamic_pointer_cast<simple_packed> (TInfo->Grid()->PackedData());

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.T), opts.N * sizeof(double), cudaHostAllocMapped));

				datas.pT = *(t);

				opts.pT = true;
			}
			else
			{
				datas.T = const_cast<double*> (TInfo->Grid()->Data()->Values());
			}

			if (RHInfo->Grid()->DataIsPacked())
			{
				assert(RHInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> rh = dynamic_pointer_cast<simple_packed> (RHInfo->Grid()->PackedData());

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.RH), opts.N * sizeof(double), cudaHostAllocMapped));

				datas.pRH = *(rh);

				opts.pRH = true;
			}
			else
			{
				datas.RH = const_cast<double*> (RHInfo->Grid()->Data()->Values());
			}

			dewpoint_cuda::DoCuda(opts, datas);

			myTargetInfo->Data()->Set(datas.TD, opts.N);
			assert(TInfo->Grid()->ScanningMode() == RHInfo->Grid()->ScanningMode());

			missingCount = opts.missingValuesCount;
			count = opts.N;
			
			CUDA_CHECK(cudaFreeHost(datas.TD));

			if (TInfo->Grid()->DataIsPacked())
			{
				TInfo->Data()->Set(datas.T, opts.N);
				TInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.T));
			}

			if (RHInfo->Grid()->DataIsPacked())
			{
				RHInfo->Data()->Set(datas.RH, opts.N);
				RHInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.RH));
			}

			SwapTo(myTargetInfo, TInfo->Grid()->ScanningMode());

		}
		else
#endif
		{
			deviceType = "CPU";

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{
				count++;

				double T = kFloatMissing;
				double RH = kFloatMissing;

				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
				InterpolateToPoint(targetGrid, RHGrid, equalGrids, RH);

				if (T == kFloatMissing || RH == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				double TD = ((T+TBase) / (1 - ((T+TBase) * log(RH) * (RW_div_L)))) - 273.15 + TBase;

				if (!myTargetInfo->Value(TD))
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

		if (conf->FileWriteOption() == kMultipleFiles || conf->FileWriteOption() == kNeons)
		{
			shared_ptr<writer> w = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

			w->ToFile(shared_ptr<info>(new info(*myTargetInfo)), conf);
		}
	}
}
