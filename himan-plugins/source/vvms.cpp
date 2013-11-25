/**
 * @file vvms.cpp
 *
 * @date Nov 20, 2012
 * @author partio
 */

#include "vvms.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include "timer_factory.h"
#include <boost/lexical_cast.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#include "vvms_cuda.h"
#include "cuda_helper.h"

vvms::vvms()
{
	itsClearTextFormula = "w = -(ver) * 287 * T * (9.81*p)";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("vvms"));

}

void vvms::Process(std::shared_ptr<const plugin_configuration> conf)
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
	 * (todo: we could check from conf but why bother?)
	 *
	 */

	vector<param> theParams;

	param theRequestedParam ("VV-MS", 143);

	// GRIB 2

	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(2);
	theRequestedParam.GribParameter(9);

	theParams.push_back(theRequestedParam);

	// GRIB 1

	if (conf->OutputFileType() == kGRIB1)
	{
		StoreGrib1ParameterDefinitions(theParams, targetInfo->Producer().TableVersion());
	}

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

		boost::thread* t = new boost::thread(&vvms::Run,
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

void vvms::Run(shared_ptr<info> myTargetInfo,
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

void vvms::Calculate(shared_ptr<info> myTargetInfo,
					 shared_ptr<const plugin_configuration> conf,
					 unsigned short threadIndex)
{


	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	params PParam = { param("P-PA"), param("P-HPA") };
	param VVParam("VV-PAS");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("vvmsThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(conf, threadIndex);

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Info("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		double PScale = 1;
		double TBase = 0;

		/*
		 * If vvms is calculated for pressure levels, the P value
		 * equals to level value. Otherwise we have to fetch P
		 * separately.
		 */

		shared_ptr<info> PInfo;
		shared_ptr<info> VVInfo;
		shared_ptr<info> TInfo;

		shared_ptr<NFmiGrid> PGrid;

		bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

		try
		{
			VVInfo = theFetcher->Fetch(conf,
						  myTargetInfo->Time(),
						  myTargetInfo->Level(),
						  VVParam,
						  conf->UseCudaForPacking() && useCudaInThisThread);

			TInfo = theFetcher->Fetch(conf,
						myTargetInfo->Time(),
						myTargetInfo->Level(),
						TParam,
						conf->UseCudaForPacking() && useCudaInThisThread);

			if (!isPressureLevel)
			{
				// Source info for P
				PInfo = theFetcher->Fetch(conf,
							myTargetInfo->Time(),
							myTargetInfo->Level(),
							PParam,
							conf->UseCudaForPacking() && useCudaInThisThread);

				if (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA")
				{
					PScale = 100;
				}

				PGrid = shared_ptr<NFmiGrid> (PInfo->Grid()->ToNewbaseGrid());
			}
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

		assert(TInfo->Grid()->AB() == VVInfo->Grid()->AB() && (isPressureLevel || PInfo->Grid()->AB() == TInfo->Grid()->AB()));

		SetAB(myTargetInfo, TInfo);
		
		if (TInfo->Param().Unit() == kC)
		{
			TBase = 273.15;
		}

		size_t missingCount = 0;
		size_t count = 0;

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() &&
							*myTargetInfo->Grid() == *VVInfo->Grid() &&
						   (isPressureLevel || *myTargetInfo->Grid() == *PInfo->Grid()));

		string deviceType;

#ifdef HAVE_CUDA
		if (useCudaInThisThread && equalGrids)
		{
			deviceType = "GPU";
			
			vvms_cuda::vvms_cuda_options opts;
			vvms_cuda::vvms_cuda_data datas;

			opts.isConstantPressure = isPressureLevel;
			opts.TBase = TBase;
			opts.PScale = PScale;
			opts.cudaDeviceIndex = static_cast<unsigned short> (threadIndex-1);

			opts.N = TInfo->Grid()->Size();

			cudaMallocHost(reinterpret_cast<void**> (&datas.VVMS), opts.N * sizeof(double));

			if (TInfo->Grid()->DataIsPacked())
			{
				assert(TInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> t = dynamic_pointer_cast<simple_packed> (TInfo->Grid()->PackedData());

				datas.pT = t.get();

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.T), opts.N * sizeof(double), cudaHostAllocMapped));

				opts.pT = true;

			}
			else
			{
				datas.T = const_cast<double*> (TInfo->Grid()->Data()->Values());
			}

			if (VVInfo->Grid()->DataIsPacked())
			{
				assert(VVInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> vv = dynamic_pointer_cast<simple_packed> (VVInfo->Grid()->PackedData());

				datas.pVV = vv.get();

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.VV), opts.N * sizeof(double), cudaHostAllocMapped));

				opts.pVV = true;

			}
			else
			{
				datas.VV = const_cast<double*> (VVInfo->Grid()->Data()->Values());
			}

			if (!isPressureLevel)
			{
				if (PInfo->Grid()->DataIsPacked())
				{
					assert(PInfo->Grid()->PackedData()->ClassName() == "simple_packed");

					shared_ptr<simple_packed> p = dynamic_pointer_cast<simple_packed> (PInfo->Grid()->PackedData());
					
					datas.pP = p.get();

					CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.P), opts.N * sizeof(double), cudaHostAllocMapped));

					opts.pP = true;
				}
				else
				{
					datas.P = const_cast<double*> (PInfo->Grid()->Data()->Values());
				}

			}
			else
			{
				opts.PConst = myTargetInfo->Level().Value() * 100; // Pa
			}

			CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.VVMS), opts.N * sizeof(double), cudaHostAllocMapped));

			vvms_cuda::DoCuda(opts, datas);
				
			myTargetInfo->Data()->Set(datas.VVMS, opts.N);

			assert(TInfo->Grid()->ScanningMode() == VVInfo->Grid()->ScanningMode() && (isPressureLevel || VVInfo->Grid()->ScanningMode() == PInfo->Grid()->ScanningMode()));

			missingCount = opts.missingValuesCount;
			count = opts.N;

			CUDA_CHECK(cudaFreeHost(datas.VVMS));

			if (TInfo->Grid()->DataIsPacked())
			{
				TInfo->Data()->Set(datas.T, opts.N);
				TInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.T));
			}

			if (VVInfo->Grid()->DataIsPacked())
			{
				VVInfo->Data()->Set(datas.VV, opts.N);
				VVInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.VV));
			}

			if (!opts.isConstantPressure && PInfo->Grid()->DataIsPacked())
			{
				PInfo->Data()->Set(datas.P, opts.N);
				PInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.P));
			}

			SwapTo(myTargetInfo, TInfo->Grid()->ScanningMode());

		}
		else
#endif
		{
			deviceType = "CPU";

			shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
			shared_ptr<NFmiGrid> VVGrid(VVInfo->Grid()->ToNewbaseGrid());

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{
				count++;

				double T = kFloatMissing;
				double P = kFloatMissing;
				double VV = kFloatMissing;

				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);
				InterpolateToPoint(targetGrid, VVGrid, equalGrids, VV);

				if (isPressureLevel)
				{
					P = 100 * myTargetInfo->Level().Value();
				}
				else
				{
				 	InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
				}

				if (T == kFloatMissing || P == kFloatMissing || VV == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				double VVms = 287 * -VV * (T + TBase) / (9.81 * P * PScale);

				if (!myTargetInfo->Value(VVms))
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
