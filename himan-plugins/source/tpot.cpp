/**
 * @file tpot.cpp
 *
 * @brief Plugin to calculate potential temperature, pseudo-adiabatic
 * potential temperature or equivalent potential temperature.
 *
 * Created on: Nov 20, 2012
 * @author partio
 */

#include "tpot.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include "timer_factory.h"
#include <boost/lexical_cast.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

#include "tpot_cuda.h"
#include "cuda_helper.h"
#include "util.h"

tpot::tpot()
: itsThetaCalculation(false)
, itsThetaWCalculation(false)
, itsThetaECalculation(false)
{
	itsClearTextFormula = "TP = Tk * pow((1000/P), 0.286) ; TPW calculated with LCL ; TPE = X"; 

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("tpot"));

}

void tpot::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to potential temperature
	 * - name TP-K
	 * - univ_id 8
	 * - grib2 descriptor 0'00'002
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 * (todo: we could check from conf but why bother?)
	 *
	 */

	vector<param> theParams;

	if (itsConfiguration->Exists("theta") && itsConfiguration->GetValue("theta") == "true")
	{
		itsThetaCalculation = true;

		itsLogger->Trace("Theta calculation requested");

		param p("TP-K", 8);

		p.GribDiscipline(0);
		p.GribCategory(0);
		p.GribParameter(2);		
		
		theParams.push_back(p);
	}

	if (itsConfiguration->Exists("thetaw") && itsConfiguration->GetValue("thetaw") == "true")
	{
		itsThetaWCalculation = true;

		itsLogger->Trace("ThetaW calculation requested");

		param p("TPW-K", 9);

		// Sharing number with thetae!

		p.GribDiscipline(0);
		p.GribCategory(0);
		p.GribParameter(3);

		theParams.push_back(p);
	}

	if (itsConfiguration->Exists("thetae") && itsConfiguration->GetValue("thetae") == "true")
	{
		itsThetaECalculation = true;

		itsLogger->Trace("ThetaE calculation requested");

		param p("TPE-K", 99999);

		// Sharing number with thetaw!

		p.GribDiscipline(0);
		p.GribCategory(0);
		p.GribParameter(3);

		theParams.push_back(p);

	}

	if (theParams.size() == 0)
	{
		// By default assume we'll calculate theta

		itsThetaCalculation = true;

		param p("TP-K", 8);

		p.GribDiscipline(0);
		p.GribCategory(0);
		p.GribParameter(2);

		theParams.push_back(p);
	}

	SetParams(theParams);

	Start();

}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void tpot::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("tpotThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	params PParam = { param("P-PA"), param("P-HPA") };
	
	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		double PScale = 1;
		double TBase = 273.15;
		double TDBase  = 273.15;
		
		// Source infos
		shared_ptr<info> TInfo;
		shared_ptr<info> PInfo;
		shared_ptr<info> TDInfo;

		shared_ptr<NFmiGrid> PGrid;
		shared_ptr<NFmiGrid> TDGrid;

		bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

		try
		{

			TInfo = theFetcher->Fetch(itsConfiguration,
										myTargetInfo->Time(),
										myTargetInfo->Level(),
										param("T-K"));

			if (!isPressureLevel)
			{
				// Source info for P
				PInfo = theFetcher->Fetch(itsConfiguration,
											myTargetInfo->Time(),
											myTargetInfo->Level(),
											PParam);

				if (PInfo->Param().Unit() == kPa)
				{
					PScale = 0.01;
				}

				PGrid = shared_ptr<NFmiGrid> (PInfo->Grid()->ToNewbaseGrid());
			}

			if (itsThetaWCalculation || itsThetaECalculation)
			{
				TDInfo = theFetcher->Fetch(itsConfiguration,
										myTargetInfo->Time(),
										myTargetInfo->Level(),
										param("TD-C"));

				TDGrid = shared_ptr<NFmiGrid> (TDInfo->Grid()->ToNewbaseGrid());

			}
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

		assert(isPressureLevel || ((PInfo->Grid()->AB() == TInfo->Grid()->AB() && PInfo->Grid()->AB() == TDInfo->Grid()->AB())));

		SetAB(myTargetInfo, TInfo);

		if (TInfo->Param().Unit() == kC)
		{
			TBase = 273.15;
		}

		if (TDInfo && TDInfo->Param().Unit() == kC)
		{
			TBase = 273.15;
		}

		shared_ptr<NFmiGrid> targetGrid(myTargetInfo->Grid()->ToNewbaseGrid());
		shared_ptr<NFmiGrid> TGrid(TInfo->Grid()->ToNewbaseGrid());

		size_t missingCount = 0;
		size_t count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() && (isPressureLevel || *myTargetInfo->Grid() == *PInfo->Grid()));

		string deviceType;

#ifdef HAVE_CUDA

		if (itsConfiguration->UseCuda() && equalGrids && threadIndex <= itsConfiguration->CudaDeviceCount())
		{
			itsLogger->Warning("tpot@cuda not supported for now");
		}

		// Force CPU
		
		if (false)
		{
			deviceType = "GPU";

			tpot_cuda::tpot_cuda_options opts;
			tpot_cuda::tpot_cuda_data datas;

			opts.N = TGrid->Size();

			opts.isConstantPressure = isPressureLevel;
			opts.TBase = TBase;
			opts.PScale = PScale;
			opts.cudaDeviceIndex = static_cast<unsigned short> (threadIndex-1);
						
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
				datas.T = const_cast<double*> (TInfo->Grid()->Data()->ValuesAsPOD());
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
					datas.P = const_cast<double*> (PInfo->Grid()->Data()->ValuesAsPOD());
				}

			}
			else
			{
				opts.PConst = myTargetInfo->Level().Value() * 100; // Pa
			}

			CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.Tp), opts.N * sizeof(double), cudaHostAllocMapped));

			tpot_cuda::DoCuda(opts, datas);

			myTargetInfo->Data()->Set(datas.Tp, opts.N);

			CUDA_CHECK(cudaFreeHost(datas.Tp));

			assert(TInfo->Grid()->ScanningMode() && (isPressureLevel || PInfo->Grid()->ScanningMode() == TInfo->Grid()->ScanningMode()));

			missingCount = opts.missingValuesCount;
			count = opts.N;

			SwapTo(myTargetInfo, TInfo->Grid()->ScanningMode());

			if (TInfo->Grid()->DataIsPacked())
			{
				TInfo->Data()->Set(datas.T, opts.N);
				TInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.T));
			}

			if (!opts.isConstantPressure && PInfo->Grid()->DataIsPacked())
			{
				PInfo->Data()->Set(datas.P, opts.N);
				PInfo->Grid()->PackedData()->Clear();
				CUDA_CHECK(cudaFreeHost(datas.P));
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
				double P = kFloatMissing;
				double TD = kFloatMissing;

				InterpolateToPoint(targetGrid, TGrid, equalGrids, T);

				if (isPressureLevel)
				{
					P = myTargetInfo->Level().Value();
				}
				else
				{
					InterpolateToPoint(targetGrid, PGrid, equalGrids, P);
				}

				if (itsThetaWCalculation || itsThetaECalculation)
				{
					InterpolateToPoint(targetGrid, TDGrid, equalGrids, TD);
				}

				if (T == kFloatMissing || P == kFloatMissing || ((itsThetaECalculation || itsThetaWCalculation) && TD == kFloatMissing))
				{
					missingCount++;

					myTargetInfo->Value(kFloatMissing);
					continue;
				}

				T -= TBase; // to Celsius
				TD -= TDBase; // to Celsius
				P /= PScale; // to hPa

				double value = kFloatMissing;
				double theta = kFloatMissing;

				if (itsThetaCalculation)
				{
					theta = Theta(P, T);

					myTargetInfo->Param(param("TP-K"));

					if (!myTargetInfo->Value(theta))
					{
						throw runtime_error(ClassName() + ": Failed to set value to matrix");
					}
				}
				
				if (itsThetaWCalculation)
				{
					value = ThetaW(P, T, TD);

					myTargetInfo->Param(param("TPW-K"));

					if (!myTargetInfo->Value(value))
					{
						throw runtime_error(ClassName() + ": Failed to set value to matrix");
					}
				}
				
				if (itsThetaECalculation)
				{
					value = ThetaE(P, T, TD, theta);
					
					myTargetInfo->Param(param("TPE-K"));

					if (!myTargetInfo->Value(value))
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
		 *
		 */

		myThreadedLogger->Info("Missing values: " + boost::lexical_cast<string> (missingCount) + "/" + boost::lexical_cast<string> (count));

		if (itsConfiguration->FileWriteOption() != kSingleFile)
		{
			WriteToFile(myTargetInfo);
		}

	}
}

double tpot::Theta(double P, double T)
{
	double value = (T+273.15) * pow((1000 / P), 0.28586) - 273.15;
	return value;
}

double tpot::ThetaW(double P, double T, double TD)
{
   const double Pstep = 500;
   double value = kFloatMissing;

   // Search LCL level

   vector<double> LCL = util::LCL(P, T, TD);

   double Tint = LCL[1];
   double Pint = LCL[0];

   if (Tint == kFloatMissing || Pint == kFloatMissing)
   {
	   value = kFloatMissing;
   }
   else
   {
       Pint *= 100; // back to Pa

	   double T0 = Tint;

	   int i = 0;

	   double Z = kFloatMissing;

	   while (++i < 500)
	   {
		   double TA = Tint;

		   if (i <= 2)
		   {
			   Z = i * Pstep/2;
		   }
		   else
		   {
			   Z = 2 * Pstep;
		   }

		   // Gammas() takes hPa
		   Tint = T0 + util::Gammas(Pint/100, Tint) * Z;

		   if (i > 2)
		   {
			   T0 = TA;
		   }

		   Pint += Pstep;

		   if (Pint >= 1e5)
		   {
			   value = Tint;
			   break;
		   }
	   }
   }

   return value;
}

double tpot::ThetaE(double P, double T, double TD, double theta)
{
	vector<double> LCL = util::LCL(P, T, TD);

	double TLCL = LCL[1];

	if (theta == kFloatMissing)
	{
		// theta was not calculated in this plugin session :(

		theta = Theta(P, T);
	}

	double ZEs = util::Es(TLCL);
	double ZQs = himan::constants::kEp * (ZEs / (P - ZEs));

	double value = theta * exp(himan::constants::kL * ZQs / himan::constants::kCp / (TLCL + 273.15));

	return value;

}
