/**
 * @file tk2tc.cpp
 *
 * @dateNov 20, 2012
 * @author partio
 */

#include "tk2tc.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#include "cuda_helper.h"


tk2tc::tk2tc() : itsBase(-273.15), itsScale(1)
{
	itsClearTextFormula = "Tc = Tk - 273.15";
	itsCudaEnabledCalculation = true;

	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("tk2tc"));
}

void tk2tc::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to T
	 * - name T-C
	 * - univ_id 4
	 * - grib2 descriptor 0'00'000
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> theParams;

	param requestedParam("T-C", 4);

	// GRIB 2
	
	requestedParam.GribDiscipline(0);
	requestedParam.GribCategory(0);
	requestedParam.GribParameter(0);

	// GRIB 1

	theParams.push_back(requestedParam);

	SetParams(theParams);

	Start();
	
}


/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void tk2tc::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	shared_ptr<fetcher> aFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("tk2tcThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(itsConfiguration, threadIndex);

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		// Source info for T

		shared_ptr<info> sourceInfo;

		try
		{
			sourceInfo = aFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 myTargetInfo->Level(),
								 TParam,
								 itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

			assert(sourceInfo->Param().Unit() == kK);

		}
		catch (HPExceptionType e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing);

					if (itsConfiguration->StatisticsEnabled())
					{
						itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Grid()->Size());
						itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Grid()->Size());
					}
					
					continue;
					break;

				default:
					throw runtime_error(ClassName() + ": Unable to proceed");
					break;
			}
		}

		SetAB(myTargetInfo, sourceInfo);

		size_t missingCount = 0;
		size_t count = 0;

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());

		bool equalGrids = (*myTargetInfo->Grid() == *sourceInfo->Grid());

		string deviceType;

#ifdef HAVE_CUDA

		// If we read packed data but grids are not equal we cannot use cuda
		// for calculations (our cuda routines do not know how to interpolate)

		if (!equalGrids && sourceInfo->Grid()->IsPackedData())
		{
			myThreadedLogger->Debug("Unpacking for CPU calculation");

			Unpack({sourceInfo});
		}

		if (useCudaInThisThread && equalGrids)
		{
	
			deviceType = "GPU";

			auto opts = CudaPrepare(sourceInfo);

			tk2tc_cuda::Process(*opts);

			missingCount = opts->missing;
			count = opts->N;

			CudaFinish(move(opts), myTargetInfo, sourceInfo);

		}
		else
#endif
		{

			deviceType = "CPU";

			shared_ptr<NFmiGrid> sourceGrid(sourceInfo->Grid()->ToNewbaseGrid());

			assert(targetGrid->Size() == myTargetInfo->Data()->Size());

			myTargetInfo->ResetLocation();

			targetGrid->Reset();

			while (myTargetInfo->NextLocation() && targetGrid->Next())
			{

				count++;

				double value = kFloatMissing;

				InterpolateToPoint(targetGrid, sourceGrid, equalGrids, value);

				if (value == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				double newValue = value - constants::kKelvin;

				if (!myTargetInfo->Value(newValue))
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

		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(missingCount);
			itsConfiguration->Statistics()->AddToValueCount(count);
		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places
		 */

		myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}

	}
}

#ifdef HAVE_CUDA

unique_ptr<tk2tc_cuda::options> tk2tc::CudaPrepare(shared_ptr<info> sourceInfo)
{
	unique_ptr<tk2tc_cuda::options> opts(new tk2tc_cuda::options);

	opts->N = sourceInfo->Data()->Size();

	opts->base = itsBase;
	opts->scale = itsScale;

	CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&opts->dest), opts->N * sizeof(double)));

	if (sourceInfo->Grid()->IsPackedData())
	{
		assert(sourceInfo->Grid()->PackedData()->ClassName() == "simple_packed");

		shared_ptr<simple_packed> t = dynamic_pointer_cast<simple_packed> (sourceInfo->Grid()->PackedData());

		CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**> (&opts->source), opts->N * sizeof(double)));

		opts->p = t.get();
	}
	else
	{
		opts->source = const_cast<double*> (sourceInfo->Grid()->Data()->ValuesAsPOD());
	}

	return opts;
}

void tk2tc::CudaFinish(unique_ptr<tk2tc_cuda::options> opts, shared_ptr<info> myTargetInfo, shared_ptr<info> sourceInfo)
{
	// Copy data back to infos

	myTargetInfo->Data()->Set(opts->dest, opts->N);
	CUDA_CHECK(cudaFreeHost(opts->dest));

	if (sourceInfo->Grid()->IsPackedData())
	{
		sourceInfo->Data()->Set(opts->source, opts->N);
		sourceInfo->Grid()->PackedData()->Clear();
		CUDA_CHECK(cudaFreeHost(opts->source));
	}

	SwapTo(myTargetInfo, sourceInfo->Grid()->ScanningMode());

}

#endif