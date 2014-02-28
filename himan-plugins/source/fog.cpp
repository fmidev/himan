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

#undef HIMAN_AUXILIARY_INCLUDE

#include "cuda_helper.h"

using namespace std;
using namespace himan::plugin;

const string itsName("fog");

fog::fog()
{
	itsClearTextFormula = "FOG = (DT2M-TGround> -0.3 && FF10M < 5) ? 607 : 0";
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog(itsName));

}

void fog::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to fog
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

	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(6);
	theRequestedParam.GribParameter(8);

	theParams.push_back(theRequestedParam);

	SetParams(theParams);

	Start();

}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void fog::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
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

			groundInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 ground,
								 groundParam);
			
			dewInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 h2m,
								 dewParam);

			windInfo = theFetcher->Fetch(itsConfiguration,
								 myTargetInfo->Time(),
								 h10m,
								 windParam);

		}
		catch (HPExceptionType& e)
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

		unique_ptr<timer> processTimer = unique_ptr<timer> (timer_factory::Instance()->GetTimer());

		if (itsConfiguration->StatisticsEnabled())
		{
			processTimer->Start();
		}
		
		size_t missingCount = 0;
		size_t count = 0;

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

#if 0

		if (itsConfiguration->UseCuda() && equalGrids && threadIndex <= itsConfiguration->CudaDeviceCount())
		{

			deviceType = "GPU";

			fog_cuda::fog_cuda_options opts;
			fog_cuda::fog_cuda_data datas;

			opts.N = dewInfo->Data()->Size();
			opts.cudaDeviceIndex = static_cast<unsigned short> (threadIndex-1);

			CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.F), opts.N * sizeof(double), cudaHostAllocMapped));
			
			if (dewInfo->Grid()->IsPackedData())
			{
				assert(dewInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> dtc2m = dynamic_pointer_cast<simple_packed> (dewInfo->Grid()->PackedData());

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.DTC2M), opts.N * sizeof(double), cudaHostAllocMapped));

				datas.pDTC2M = dtc2m.get();

				opts.pDTC2M = true;
			}
			else
			{
				datas.DTC2M = const_cast<double*> (dewInfo->Grid()->Data()->ValuesAsPOD());
			}

			if (groundInfo->Grid()->IsPackedData())
			{
				assert(groundInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> tkground = dynamic_pointer_cast<simple_packed> (groundInfo->Grid()->PackedData());

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.TKGround), opts.N * sizeof(double), cudaHostAllocMapped));

				datas.pTKGround = tkground.get();

				opts.pTKGround = true;
			}
			else
			{
				datas.TKGround = const_cast<double*> (groundInfo->Grid()->Data()->ValuesAsPOD());
			}

			if (windInfo->Grid()->IsPackedData())
			{
				assert(windInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> ff10m = dynamic_pointer_cast<simple_packed> (windInfo->Grid()->PackedData());

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.FF10M), opts.N * sizeof(double), cudaHostAllocMapped));

				datas.pFF10M = ff10m.get();

				opts.pFF10M = true;
			}
			else
			{
				datas.FF10M = const_cast<double*> (windInfo->Grid()->Data()->ValuesAsPOD());
			}

			fog_cuda::DoCuda(opts, datas);

			myTargetInfo->Data()->Set(datas.F, opts.N);
			assert(dewInfo->Grid()->ScanningMode() == groundInfo->Grid()->ScanningMode());
			assert(dewInfo->Grid()->ScanningMode() == windInfo->Grid()->ScanningMode());

			missingCount = opts.missingValuesCount;
			count = opts.N;
			
			CUDA_CHECK(cudaFreeHost(datas.F));

			if (dewInfo->Grid()->IsPackedData())
			{
				dewInfo->Data()->Set(datas.DTC2M, opts.N);
				dewInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.DTC2M));
			}

			if (groundInfo->Grid()->IsPackedData())
			{
				groundInfo->Data()->Set(datas.TKGround, opts.N);
				groundInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.TKGround));
			}

			if (windInfo->Grid()->IsPackedData())
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

		/*
		 * Newbase normalizes scanning mode to bottom left -- if that's not what
		 * the target scanning mode is, we have to swap the data back.
		 */

		SwapTo(myTargetInfo, kBottomLeft);

		if (itsConfiguration->StatisticsEnabled())
		{
			processTimer->Stop();
			itsConfiguration->Statistics()->AddToProcessingTime(processTimer->GetTime());

#ifdef DEBUG
			itsLogger->Debug("Calculation took " + boost::lexical_cast<string> (processTimer->GetTime()) + " microseconds on "  + deviceType);
#endif

			itsConfiguration->Statistics()->AddToMissingCount(missingCount);
			itsConfiguration->Statistics()->AddToValueCount(count);

		}

		/*
		 * Now we are done for this level
		 *
		 * Clone info-instance to writer since it might change our descriptor places
		 * */

		myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}
	}
}