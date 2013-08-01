/**
 * @file fog.cpp
 *
 * @date Jul 3, 2013
 * @author peramaki
 */

#include "fog.h"
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

#include "fog_cuda.h"
#include "cuda_helper.h"

using namespace std;
using namespace himan::plugin;

const string itsName("fog");

fog::fog() : itsUseCuda(false)
{
	itsClearTextFormula = "FOG = (DT2M-TGround> -0.3 && FF10M < 5) ? 607 : 0";
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void fog::Process(std::shared_ptr<const plugin_configuration> conf)
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
	 * - name PARM_NAME
	 * - univ_id UNIV_ID
	 * - grib2 descriptor 0'6'8
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param theRequestedParam("FOGSYM-N", 334);

	// GRIB 2

	//temp parameters
	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(6);
	theRequestedParam.GribParameter(8);
	
	// GRIB 1

	/*
	 * GRIB 1 parameters go here
	 *
	 */
 	if (conf->OutputFileType() == kGRIB1)
	{
		shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		long parm_id = n->NeonsDB().GetGridParameterId(targetInfo->Producer().TableVersion(), theRequestedParam.Name());
		theRequestedParam.GribIndicatorOfParameter(parm_id);
		theRequestedParam.GribTableVersion(targetInfo->Producer().TableVersion());

	}

	// ----

	theParams.push_back(theRequestedParam);

	targetInfo->Params(theParams);

	/*
	 * Create data structures.
	 */

	targetInfo->Create();


	/*
	 * Initialize parent class functions for dimension handling
	 */

	Dimension(conf->LeadingDimension());
	FeederInfo(shared_ptr<info> (new info(*targetInfo)));
	FeederInfo()->Param(theRequestedParam);

	if (conf->StatisticsEnabled())
	{
		initTimer->Stop();
		conf->Statistics()->AddToInitTime(initTimer->GetTime());
	}

	/*
	 * Each thread will have a copy of the target info.
	 */

	vector<shared_ptr<info> > targetInfos;

	targetInfos.resize(threadCount);

	for (size_t i = 0; i < threadCount; i++)
	{

		itsLogger->Info("Thread " + boost::lexical_cast<string> (i + 1) + " starting");

		targetInfos[i] = shared_ptr<info> (new info(*targetInfo));

		boost::thread* t = new boost::thread(&fog::Run,
								this,
								targetInfos[i],
								conf,
								i + 1);

		g.add_thread(t);

	}

	g.join_all();

	if (conf->FileWriteOption() == kSingleFile)
	{

		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		string theOutputFile = conf->ConfigurationFile();

		theWriter->ToFile(targetInfo, conf, theOutputFile);

	}
}

void fog::Run(shared_ptr<info> myTargetInfo,
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

void fog::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	//2m kastepiste
	//10m tuulen nopeus
	//alustan lämpötila

	param groundParam("T-K");
	param dewParam("TD-C");
	param windParam("FF-MS");
	
	level ground(himan::kHeight, 0, "HEIGHT");
	level h2m(himan::kHeight, 2, "HEIGHT");
	level h10m(himan::kHeight, 10, "HEIGHT");



	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName + "Thread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{
		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		shared_ptr<info> groundInfo;
		shared_ptr<info> dewInfo;
		shared_ptr<info> windInfo;
		try
		{

			groundInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 ground,
								 groundParam);
			
			dewInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 h2m,
								 dewParam);

			windInfo = theFetcher->Fetch(conf,
								 myTargetInfo->Time(),
								 h10m,
								 windParam);

		}
		catch (HPExceptionType e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
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

		unique_ptr<timer> processTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());

		if (conf->StatisticsEnabled())
		{
			processTimer->Start();
		}
		
		int missingCount = 0;
		int count = 0;

		/*
		 * Converting original grid-data to newbase grid
		 *
		 */

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> groundGrid(groundInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> dewGrid(dewInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> windGrid(windInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = (*myTargetInfo->Grid() == *groundInfo->Grid() 
						&& *myTargetInfo->Grid() == *dewInfo->Grid() 
						&& *myTargetInfo->Grid() == *windInfo->Grid() );


		string deviceType;

#ifdef HAVE_CUDA

		if (itsUseCuda && equalGrids && threadIndex <= itsCudaDeviceCount)
		{

			deviceType = "GPU";

			fog_cuda::fog_cuda_options opts;
			fog_cuda::fog_cuda_data datas;

			opts.N = dewInfo->Data()->Size();
			opts.cudaDeviceIndex = threadIndex-1;

			CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.F), opts.N * sizeof(double), cudaHostAllocMapped));
			
			if (dewInfo->Grid()->DataIsPacked())
			{
				assert(dewInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> dtc2m = dynamic_pointer_cast<simple_packed> (dewInfo->Grid()->PackedData());

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.DTC2M), opts.N * sizeof(double), cudaHostAllocMapped));

				datas.pDTC2M = *(dtc2m);

				opts.pDTC2M = true;
			}
			else
			{
				datas.DTC2M = const_cast<double*> (dewInfo->Grid()->Data()->Values());
			}

			if (groundInfo->Grid()->DataIsPacked())
			{
				assert(groundInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> tkground = dynamic_pointer_cast<simple_packed> (groundInfo->Grid()->PackedData());

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.TKGround), opts.N * sizeof(double), cudaHostAllocMapped));

				datas.pTKGround = *(tkground);

				opts.pTKGround = true;
			}
			else
			{
				datas.TKGround = const_cast<double*> (groundInfo->Grid()->Data()->Values());
			}

			if (windInfo->Grid()->DataIsPacked())
			{
				assert(windInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> ff10m = dynamic_pointer_cast<simple_packed> (windInfo->Grid()->PackedData());

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.FF10M), opts.N * sizeof(double), cudaHostAllocMapped));

				datas.pFF10M = *(ff10m);

				opts.pFF10M = true;
			}
			else
			{
				datas.FF10M = const_cast<double*> (windInfo->Grid()->Data()->Values());
			}

			fog_cuda::DoCuda(opts, datas);

			myTargetInfo->Data()->Set(datas.F, opts.N);
			assert(dewInfo->Grid()->ScanningMode() == groundInfo->Grid()->ScanningMode());
			assert(dewInfo->Grid()->ScanningMode() == windInfo->Grid()->ScanningMode());

			missingCount = opts.missingValuesCount;
			count = opts.N;
			
			CUDA_CHECK(cudaFreeHost(datas.F));

			if (dewInfo->Grid()->DataIsPacked())
			{
				dewInfo->Data()->Set(datas.DTC2M, opts.N);
				dewInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.DTC2M));
			}

			if (groundInfo->Grid()->DataIsPacked())
			{
				groundInfo->Data()->Set(datas.TKGround, opts.N);
				groundInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.TKGround));
			}

			if (windInfo->Grid()->DataIsPacked())
			{
				windInfo->Data()->Set(datas.FF10M, opts.N);
				windInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.FF10M));
			}

			SwapTo(myTargetInfo, dewInfo->Grid()->ScanningMode());

		}

		else 
#endif
		{
			deviceType = "CPU";

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			dewGrid->Reset();
			groundGrid->Reset();
			windGrid->Reset();

			while (myTargetInfo->NextLocation() 
				&& targetGrid->Next() 
				&& groundGrid->Next()
				&& dewGrid->Next() 
				&& windGrid->Next() )
			{

				count++;

				double dt2m = kFloatMissing;
				double wind10m = kFloatMissing;
				double tGround = kFloatMissing;

				InterpolateToPoint(targetGrid, groundGrid, equalGrids, tGround);
				InterpolateToPoint(targetGrid, dewGrid, equalGrids, dt2m);
				InterpolateToPoint(targetGrid, windGrid, equalGrids, wind10m);

				if (tGround == kFloatMissing || dt2m == kFloatMissing || wind10m == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}


				//double TBase = 273.15;
				double fog = 0;

				if (dt2m-tGround > -0.3 && wind10m < 5 )
					fog = 607;
				//else
				//	fog = 0;

				if (!myTargetInfo->Value(fog))
				{
					throw runtime_error(ClassName() + ": Failed to set value to matrix");
				}
			}
		}

		if (conf->StatisticsEnabled())
		{
			processTimer->Stop();
			conf->Statistics()->AddToProcessingTime(processTimer->GetTime());

#ifdef DEBUG
			itsLogger->Debug("Calculation took " + boost::lexical_cast<string> (processTimer->GetTime()) + " microseconds on "  + deviceType);
#endif

			conf->Statistics()->AddToMissingCount(missingCount);
			conf->Statistics()->AddToValueCount(count);

		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places
		 * */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (conf->FileWriteOption() == kNeons || conf->FileWriteOption() == kMultipleFiles)
		{
			shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

			theWriter->ToFile(shared_ptr<info> (new info(*myTargetInfo)), conf);
		}
	}
}