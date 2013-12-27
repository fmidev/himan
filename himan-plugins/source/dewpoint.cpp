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

#undef HIMAN_AUXILIARY_INCLUDE

#include "dewpoint_cuda.h"
#include "cuda_helper.h"

using namespace std;
using namespace himan::plugin;

const double Rw_div_L = himan::constants::kRw / himan::constants::kL;

dewpoint::dewpoint()
{
	itsClearTextFormula = "Td = T / (1 - (T * ln(RH)*(Rw/L)))";

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("dewpoint"));

}

void dewpoint::Process(shared_ptr<const plugin_configuration> conf)
{

	Init(conf);
	
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

	params.push_back(requestedParam);

	SetParams(params);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void dewpoint::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	param RHParam("RH-PRCNT");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("dewpointThread #" + boost::lexical_cast<string> (threadIndex)));
	
	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(itsConfiguration, threadIndex);

	// Force use of CPU since cuda does not handle RHScale yet!
	useCudaInThisThread = false;
	
	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		double TBase = 0;
		double RHScale = 1;
		
		shared_ptr<info> TInfo;
		shared_ptr<info> RHInfo;

		try
		{
			TInfo = f->Fetch(itsConfiguration,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								TParam,
								itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

			RHInfo = f->Fetch(itsConfiguration,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								RHParam,
								itsConfiguration->UseCudaForPacking() && useCudaInThisThread);


		}
		catch (HPExceptionType e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					itsLogger->Info("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
					myTargetInfo->Data()->Fill(kFloatMissing); // Fill data with missing value

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

		assert(TInfo->Grid()->AB() == RHInfo->Grid()->AB());
		
		SetAB(myTargetInfo, TInfo);

		if (RHInfo->Param().Unit() != kPrcnt)
		{
			// If unit cannot be detected, assume the values are from 0 .. 1
			RHScale = 100;
			myThreadedLogger->Warning("Unable to determine unit for relative humidity -- assuming values are from 0 to 1");
		}

		if (TInfo->Param().Unit() == kC)
		{
			TBase = 273.15;
		}

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> RHGrid(RHInfo->Grid()->ToNewbaseGrid());

		size_t missingCount = 0;
		size_t count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() && *myTargetInfo->Grid() == *RHInfo->Grid());

		string deviceType;

#ifdef HAVE_CUDA
		if (useCudaInThisThread && equalGrids)
		{

			deviceType = "GPU";

			dewpoint_cuda::dewpoint_cuda_options opts;
			dewpoint_cuda::dewpoint_cuda_data datas;

			opts.N = TInfo->Data()->Size();
			opts.cudaDeviceIndex = static_cast<unsigned short> (threadIndex-1);

			opts.TBase = TBase;

			CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.TD), opts.N * sizeof(double), cudaHostAllocMapped));
			
			if (TInfo->Grid()->DataIsPacked())
			{
				assert(TInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> t = dynamic_pointer_cast<simple_packed> (TInfo->Grid()->PackedData());

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.T), opts.N * sizeof(double), cudaHostAllocMapped));

				datas.pT = t.get();

				opts.pT = true;
			}
			else
			{
				datas.T = const_cast<double*> (TInfo->Grid()->Data()->ValuesAsPOD());
			}

			if (RHInfo->Grid()->DataIsPacked())
			{
				assert(RHInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> rh = dynamic_pointer_cast<simple_packed> (RHInfo->Grid()->PackedData());

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.RH), opts.N * sizeof(double), cudaHostAllocMapped));

				datas.pRH = rh.get();

				opts.pRH = true;
			}
			else
			{
				datas.RH = const_cast<double*> (RHInfo->Grid()->Data()->ValuesAsPOD());
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

				double TD = ((T+TBase) / (1 - ((T+TBase) * log(RHScale * RH) * (Rw_div_L)))) - 273.15 + TBase;

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

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}
