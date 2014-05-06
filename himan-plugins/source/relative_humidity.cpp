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

relative_humidity::relative_humidity()
{
	itsClearTextFormula = "RH = 100 *  (P * Q / 0.622 / es) * (P - es) / (P - Q * P / 0.622)";
	itsCudaEnabledCalculation = true;

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("relative_humidity"));

}

void relative_humidity::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to relative humidity
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> params;

	param requestedParam ("RH-PRCNT", 13);
	requestedParam.Unit(kPrcnt);

	// GRIB 2

	requestedParam.GribDiscipline(0);
	requestedParam.GribCategory(1);
	requestedParam.GribParameter(1);

	params.push_back(requestedParam);

	SetParams(params);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void relative_humidity::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{	
	shared_ptr<fetcher> f = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	// Required source parameters

	const param TParam("T-K");
	const params PParams = { param("P-HPA"), param("P-PA") };
	const param QParam("Q-KGKG");
	const param TDParam("TD-C");

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("relative_humidityThread #" + boost::lexical_cast<string> (threadIndex)));
	
	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	// Disable cuda until we are 100% confident the tests are working
	//bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(threadIndex);
	bool useCudaInThisThread = false;

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H%M") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		double TBase = 0;
		double TDBase = 0;
		double PScale = 1;
		bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);
		
		shared_ptr<info> TInfo;
		shared_ptr<info> TDInfo;
		shared_ptr<info> PInfo;
		shared_ptr<info> QInfo;

		// Temperature is always needed

		try
		{
			TInfo = f->Fetch(itsConfiguration,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								TParam,
								itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

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

		// First try to calculate using Q and P

		bool calculateWithTD = false;
		
		try
		{
			QInfo = f->Fetch(itsConfiguration,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								QParam,
								itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
			if (!isPressureLevel)
			{
				PInfo = f->Fetch(itsConfiguration,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								PParams,
								itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
			}
		}
		catch (HPExceptionType& e)
		{
			switch (e)
			{
				case kFileDataNotFound:
					myThreadedLogger->Debug("Q or P not found, trying calculation with TD");
					calculateWithTD = true;
					break;

				default:
					throw runtime_error(ClassName() + ": Unable to proceed");
					break;
			}
		}

		if (calculateWithTD)
		{
			try
			{

				TDInfo = f->Fetch(itsConfiguration,
								myTargetInfo->Time(),
								myTargetInfo->Level(),
								TDParam,
								itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

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
		}

		assert(!PInfo || TInfo->Grid()->AB() == PInfo->Grid()->AB());
		assert(!TDInfo || TInfo->Grid()->AB() == TDInfo->Grid()->AB());
		assert(!QInfo || TInfo->Grid()->AB() == QInfo->Grid()->AB());

		
		SetAB(myTargetInfo, TInfo);

		if (TInfo->Param().Unit() == kK)
		{
			TBase = -constants::kKelvin;
		}

		if (TDInfo && TDInfo->Param().Unit() == kK)
		{
			TDBase = -constants::kKelvin;
		}

		if (!calculateWithTD && !isPressureLevel && (PInfo->Param().Name() == "P-PA" || PInfo->Param().Unit() == kPa))
		{
			PScale = 0.01;
		}

		shared_ptr<NFmiGrid> PGrid;
		shared_ptr<NFmiGrid> QGrid;
		shared_ptr<NFmiGrid> TDGrid;

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());

		if (calculateWithTD)
		{
			TDGrid = shared_ptr<NFmiGrid> (TDInfo->Grid()->ToNewbaseGrid());
		}
		else
		{
			QGrid = shared_ptr<NFmiGrid> (QInfo->Grid()->ToNewbaseGrid());
			
			if (!isPressureLevel)
			{
				PGrid = shared_ptr<NFmiGrid> (PInfo->Grid()->ToNewbaseGrid());
			}
		}
	
		size_t missingCount = 0;
		size_t count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() && 
							(!QInfo || (*myTargetInfo->Grid() == *QInfo->Grid())) &&
							(!PInfo || (*myTargetInfo->Grid() == *PInfo->Grid())) &&
							(!TDInfo || (*myTargetInfo->Grid() == *TDInfo->Grid())));

		string deviceType;

#ifdef HAVE_CUDA

		// If we read packed data but grids are not equal we cannot use cuda
		// for calculations (our cuda routines do not know how to interpolate)

		if (!equalGrids)
		{
			myThreadedLogger->Debug("Unpacking for CPU calculation");
		
			if (TInfo && TInfo->Grid()->IsPackedData())
			{
				Unpack({TInfo});
			}

			if (TDInfo && TDInfo->Grid()->IsPackedData())
			{
				Unpack({TDInfo});
			}

			if (QInfo && QInfo->Grid()->IsPackedData())
			{
				Unpack({QInfo});
			}

			if (PInfo && PInfo->Grid()->IsPackedData())
			{
				Unpack({PInfo});
			}
		}

		if (useCudaInThisThread && equalGrids)
		{
	
			deviceType = "GPU";

			if (calculateWithTD)
			{
				auto opts = CudaPrepare(myTargetInfo, TInfo, TDInfo, TDBase, TBase);

				relative_humidity_cuda::Process(*opts);

				missingCount = opts->missing;
				count = opts->N;

				CudaFinish(move(opts), myTargetInfo, TInfo, TDInfo);
			}
			else if (isPressureLevel)
			{
				auto opts = CudaPrepare(myTargetInfo, TInfo, QInfo, myTargetInfo->Level().Value(), TBase);

				relative_humidity_cuda::Process(*opts);

				missingCount = opts->missing;
				count = opts->N;

				CudaFinish(move(opts), myTargetInfo, TInfo, QInfo);
			}
			else
			{
				auto opts = CudaPrepare(myTargetInfo, TInfo, QInfo, PInfo, PScale, TBase);

				relative_humidity_cuda::Process(*opts);

				missingCount = opts->missing;
				count = opts->N;

				CudaFinish(move(opts), myTargetInfo, TInfo, QInfo, PInfo);
			}

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
				double TD = kFloatMissing;
				double P = kFloatMissing;
				double Q = kFloatMissing;
				double RH = kFloatMissing;

				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);

				if (T == kFloatMissing)
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}
				
				T += TBase;

				if (calculateWithTD)
				{
					InterpolateToPoint(targetGrid, TDGrid, equalGrids, TD);

					if (TD == kFloatMissing)
					{
						missingCount++;

						myTargetInfo->Value(kFloatMissing);
						continue;
					}

					TD += TDBase;
					RH = WithTD(T, TD);
				}
				else
				{
					if (isPressureLevel)
					{
						P = myTargetInfo->Level().Value();
					}
					else
					{
						InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
					}

					InterpolateToPoint(targetGrid, QGrid, equalGrids, Q);

					if (P == kFloatMissing || Q == kFloatMissing)
					{
						missingCount++;

						myTargetInfo->Value(kFloatMissing);
						continue;
					}

					P *= PScale;
					RH = WithQ(T, Q, P);
				}

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

inline
double relative_humidity::WithQ(double T, double Q, double P)
{
	// Pressure needs to be hPa and temperature C

	double es = util::Es(T + constants::kKelvin) * 0.01;
	
	return (P * Q / himan::constants::kEp / es) * (P - es) / (P - Q * P / himan::constants::kEp);
}

inline
double relative_humidity::WithTD(double T, double TD)
{
	const double b = 17.27;
	const double c = 237.3;
	const double d = 1.8;

	return exp(d + b * (TD / (TD + c))) / exp(d + b * (T / (T + c)));
}
#ifdef HAVE_CUDA
// Case where RH is calculated from T and TD
unique_ptr<relative_humidity_cuda::options> relative_humidity::CudaPrepare( shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> TDInfo, double TDBase, double TBase)
{
	unique_ptr<relative_humidity_cuda::options> opts(new relative_humidity_cuda::options);

	opts->N = TInfo->Data()->Size();
	opts->select_case = 0;

	opts->TDBase = TDBase;
	opts->TBase = TBase;

	opts->T = TInfo->ToSimple();
	opts->TD = TDInfo->ToSimple();
	opts->RH = myTargetInfo->ToSimple();

	return opts;
}
// Case where RH is calculated from T, Q and P
unique_ptr<relative_humidity_cuda::options> relative_humidity::CudaPrepare( shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> QInfo, shared_ptr<info> PInfo, double PScale, double TBase)
{
	unique_ptr<relative_humidity_cuda::options> opts(new relative_humidity_cuda::options);

	opts->N = TInfo->Data()->Size();

	opts->select_case = 1;

	opts->kEp = constants::kEp;
	opts->PScale = PScale;
	opts->TBase = TBase;

	opts->T = TInfo->ToSimple();
	opts->P = PInfo->ToSimple();
	opts->Q = QInfo->ToSimple();
	opts->RH = myTargetInfo->ToSimple();

	return opts;
}
// Case where RH is calculated for pressure levels from T and Q
unique_ptr<relative_humidity_cuda::options> relative_humidity::CudaPrepare( shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> QInfo, double P_level, double TBase)
{
	unique_ptr<relative_humidity_cuda::options> opts(new relative_humidity_cuda::options);

	opts->N = TInfo->Data()->Size();

	opts->select_case = 2;

	opts->TBase = TBase;

	opts->P_level = P_level;
	opts->T = TInfo->ToSimple();
	opts->Q = QInfo->ToSimple();
	opts->RH = myTargetInfo->ToSimple();

	return opts;
}
// Copy data back to infos
// Case where RH is calculated from (T and TD) or from (T and Q)
void relative_humidity::CudaFinish(unique_ptr<relative_humidity_cuda::options> opts, shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> TD_Q_Info)
{
	CopyDataFromSimpleInfo(myTargetInfo, opts->RH, false);

	if (TInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(TInfo, opts->T, true);
	}

	switch(opts->select_case)
	{
	case(0):
	{
		if (TD_Q_Info->Grid()->IsPackedData())
		{
			CopyDataFromSimpleInfo(TD_Q_Info, opts->TD, true);
		}
		break;
	}
	case(2):
	{
		if (TD_Q_Info->Grid()->IsPackedData())
		{
			CopyDataFromSimpleInfo(TD_Q_Info, opts->Q, true);
		}
		break;
	}
	}	
	SwapTo(myTargetInfo, TInfo->Grid()->ScanningMode());

}
// Case where RH is calculated from T, Q and P
void relative_humidity::CudaFinish(unique_ptr<relative_humidity_cuda::options> opts, shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> QInfo, shared_ptr<info> PInfo)
{
	CopyDataFromSimpleInfo(myTargetInfo, opts->RH, false);

	if (TInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(TInfo, opts->T, true);
	}

	if (QInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(QInfo, opts->Q, true);
	}

	if (PInfo->Grid()->IsPackedData())
	{
		CopyDataFromSimpleInfo(PInfo, opts->P, true);
	}

	SwapTo(myTargetInfo, TInfo->Grid()->ScanningMode());

}
#endif
