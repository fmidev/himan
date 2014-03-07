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

vvms::vvms() : itsScale(1)
{
	itsClearTextFormula = "w = -(ver) * 287 * T * (9.81*p)";
	itsCudaEnabledCalculation = true;

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("vvms"));

}

void vvms::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to vertical velocity
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from conf but why bother?)
	 *
	 */

	vector<param> theParams;

	param theRequestedParam ("VV-MS", 143);

	if (itsConfiguration->Exists("millimeters") && itsConfiguration->GetValue("millimeters") == "true")
	{
		theRequestedParam = param("VV-MMS", 43);
		itsScale = 1000;
	}
	
	// GRIB 2

	theRequestedParam.GribDiscipline(0);
	theRequestedParam.GribCategory(2);
	theRequestedParam.GribParameter(9);

	theParams.push_back(theRequestedParam);

	SetParams(theParams);

	Start();

}


/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void vvms::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{


	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	param TParam("T-K");
	params PParam = { param("P-PA"), param("P-HPA") };
	param VVParam("VV-PAS");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("vvmsThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(itsConfiguration, threadIndex);

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
			VVInfo = theFetcher->Fetch(itsConfiguration,
						  myTargetInfo->Time(),
						  myTargetInfo->Level(),
						  VVParam,
						  itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

			TInfo = theFetcher->Fetch(itsConfiguration,
						myTargetInfo->Time(),
						myTargetInfo->Level(),
						TParam,
						itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

			if (!isPressureLevel)
			{
				// Source info for P
				PInfo = theFetcher->Fetch(itsConfiguration,
							myTargetInfo->Time(),
							myTargetInfo->Level(),
							PParam,
							itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

				if (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA")
				{
					PScale = 100;
				}

				PGrid = shared_ptr<NFmiGrid> (PInfo->Grid()->ToNewbaseGrid());
			}
		}
		catch (HPExceptionType& e)
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

		assert(TInfo->Grid()->AB() == VVInfo->Grid()->AB() && (isPressureLevel || PInfo->Grid()->AB() == TInfo->Grid()->AB()));

		SetAB(myTargetInfo, TInfo);
		
		if (TInfo->Param().Unit() == kC)
		{
			TBase = himan::constants::kKelvin;
		}

		size_t missingCount = 0;
		size_t count = 0;

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() &&
							*myTargetInfo->Grid() == *VVInfo->Grid() &&
						   (isPressureLevel || *myTargetInfo->Grid() == *PInfo->Grid()));

		string deviceType;

#ifdef HAVE_CUDA

		// If we read packed data but grids are not equal we cannot use cuda
		// for calculations (our cuda routines do not know how to interpolate)

		if (!equalGrids && (TInfo->Grid()->IsPackedData() || VVInfo->Grid()->IsPackedData() || (PInfo && PInfo->Grid()->IsPackedData())))
		{
			myThreadedLogger->Debug("Unpacking for CPU calculation");

			Unpack({TInfo, VVInfo});

			if (PInfo)
			{
				Unpack({PInfo});
			}	
		}

		if (useCudaInThisThread && equalGrids)
		{
			deviceType = "GPU";
			
			auto opts = CudaPrepare(myTargetInfo, TInfo, VVInfo, PInfo);

			vvms_cuda::Process(*opts);

			count = opts->N;
			missingCount = opts->missing;

			CudaFinish(move(opts), myTargetInfo, TInfo, VVInfo, PInfo);

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

				double w = itsScale * (287 * -VV * (T + TBase) / (himan::constants::kG * P * PScale));

				if (!myTargetInfo->Value(w))
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

unique_ptr<vvms_cuda::options> vvms::CudaPrepare(shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> VVInfo, shared_ptr<info> PInfo)
{
	unique_ptr<vvms_cuda::options> opts(new vvms_cuda::options);

	opts->is_constant_pressure = (myTargetInfo->Level().Type() == kPressure);

	opts->t = TInfo->ToSimple();
	opts->vv = VVInfo->ToSimple();
	opts->vv_ms= myTargetInfo->ToSimple();

	if (!opts->is_constant_pressure)
	{
		opts->p = PInfo->ToSimple();

		if (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA")
		{
			opts->p_scale = 100;
		}
	}
	else
	{
		opts->p_const = myTargetInfo->Level().Value() * 100; // Pa
	}

	if (TInfo->Param().Unit() == kC)
	{
		opts->t_base = himan::constants::kKelvin;
	}

	opts->N = TInfo->Grid()->Size();

	opts->vv_ms_scale = itsScale;

	return opts;
}

void vvms::CudaFinish(unique_ptr<vvms_cuda::options> opts, shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> VVInfo, shared_ptr<info> PInfo)
{
	// Copy data back to infos

	CopyDataFromSimpleInfo(myTargetInfo, opts->vv_ms, false);

	assert(TInfo->Grid()->ScanningMode() == VVInfo->Grid()->ScanningMode());

	SwapTo(myTargetInfo, TInfo->Grid()->ScanningMode());
	
	// Copy unpacked data to source info in case
	// some other thread/plugin calls for this same data.
	// Clear packed data now that it's been unpacked

	if (TInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(TInfo, opts->t, true);
	}

	if (VVInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(VVInfo, opts->vv, true);
	}

	if (PInfo && PInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(PInfo, opts->p, true);
	}

	// opts is destroyed after leaving this function
}

#endif