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
#include <iostream>
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

using namespace std;
using namespace himan::plugin;

#include "tpot_cuda.h"
#include "cuda_helper.h"
#include "util.h"

const double Cp = 1004; // specific heat at constant pressure
const double ZLc = 2.5e6; // latent heat of vaporization J/kg (=Rd)
const double e = 0.622; // Rd/Rv

tpot::tpot()
: itsThetaCalculation(false)
, itsThetaWCalculation(false)
, itsThetaECalculation(false)
, itsUseCuda(false)
, itsCudaDeviceCount(0)
{
	itsClearTextFormula = "TP = Tk * pow((1000/P), 0.286) ; TPW calculated with LCL ; TPE = X"; 

	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("tpot"));

}

void tpot::Process(std::shared_ptr<const plugin_configuration> conf)
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

	if (conf->Exists("theta") && conf->GetValue("theta") == "true")
	{
		itsThetaCalculation = true;

		param p("TP-K", 8);

		p.GribDiscipline(0);
		p.GribCategory(0);
		p.GribParameter(2);		
		
		theParams.push_back(p);
	}

	if (conf->Exists("thetaw") && conf->GetValue("thetaw") == "true")
	{
		itsThetaWCalculation = true;

		param p("TPW-K", 9);

		// Sharing number with thetae!

		p.GribDiscipline(0);
		p.GribCategory(0);
		p.GribParameter(3);

		theParams.push_back(p);
	}

	if (conf->Exists("thetae") && conf->GetValue("thetae") == "true")
	{
		itsThetaECalculation = true;

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
	
	// GRIB 1

	if (conf->OutputFileType() == kGRIB1)
	{
		shared_ptr<neons> n = dynamic_pointer_cast<neons> (plugin_factory::Instance()->Plugin("neons"));

		for (unsigned int i = 0; i < theParams.size(); i++)
		{
			long parm_id = n->NeonsDB().GetGridParameterId(targetInfo->Producer().TableVersion(), theParams[i].Name());
			theParams[i].GribIndicatorOfParameter(parm_id);
			theParams[i].GribTableVersion(targetInfo->Producer().TableVersion());
		}
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
	FeederInfo()->ParamIndex(0); // Set index to first param (it doesn't matter which one, as long as its set

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

		boost::thread* t = new boost::thread(&tpot::Run,
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
	
	if (conf->FileWriteOption() == kSingleFile)
	{

		shared_ptr<writer> theWriter = dynamic_pointer_cast <writer> (plugin_factory::Instance()->Plugin("writer"));

		string theOutputFile = conf->ConfigurationFile();

		theWriter->ToFile(targetInfo, conf, theOutputFile);

	}
}

void tpot::Run(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
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

void tpot::Calculate(shared_ptr<info> myTargetInfo, shared_ptr<const plugin_configuration> conf, unsigned short threadIndex)
{

	shared_ptr<fetcher> theFetcher = dynamic_pointer_cast <fetcher> (plugin_factory::Instance()->Plugin("fetcher"));

	unique_ptr<logger> myThreadedLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("tpotThread #" + boost::lexical_cast<string> (threadIndex)));

	ResetNonLeadingDimension(myTargetInfo);

	myTargetInfo->FirstParam();

	while (AdjustNonLeadingDimension(myTargetInfo))
	{

		myThreadedLogger->Debug("Calculating time " + myTargetInfo->Time().ValidDateTime()->String("%Y%m%d%H") +
								" level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));

		double PScale = 1;
		double TBase = 273.15;

		// Source infos
		shared_ptr<info> TInfo;
		shared_ptr<info> PInfo;
		shared_ptr<info> TDInfo;

		shared_ptr<NFmiGrid> PGrid;
		shared_ptr<NFmiGrid> TDGrid;

		bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

		try
		{

			TInfo = theFetcher->Fetch(conf,
										myTargetInfo->Time(),
										myTargetInfo->Level(),
										param("T-K"));

			if (!isPressureLevel)
			{
				// Source info for P
				PInfo = theFetcher->Fetch(conf,
											myTargetInfo->Time(),
											myTargetInfo->Level(),
											param("P-PA"));

				if (PInfo->Param().Unit() == kPa)
				{
					PScale = 0.01;
				}

				PGrid = shared_ptr<NFmiGrid> (PInfo->Grid()->ToNewbaseGrid());
			}

			if (itsThetaWCalculation || itsThetaECalculation)
			{
				TDInfo = theFetcher->Fetch(conf,
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

		int missingCount = 0;
		int count = 0;

		assert(targetGrid->Size() == myTargetInfo->Data()->Size());

		bool equalGrids = (*myTargetInfo->Grid() == *TInfo->Grid() && (isPressureLevel || *myTargetInfo->Grid() == *PInfo->Grid()));

		string deviceType;

#ifdef HAVE_CUDA
		
		if (itsUseCuda && equalGrids && threadIndex <= itsCudaDeviceCount)
		{
			deviceType = "GPU";

			tpot_cuda::tpot_cuda_options opts;
			tpot_cuda::tpot_cuda_data datas;

			opts.N = TGrid->Size();

			opts.isConstantPressure = isPressureLevel;
			opts.TBase = TBase;
			opts.PScale = PScale;
			opts.cudaDeviceIndex = threadIndex-1;
						
			if (TInfo->Grid()->DataIsPacked())
			{
				assert(TInfo->Grid()->PackedData()->ClassName() == "simple_packed");

				shared_ptr<simple_packed> t = dynamic_pointer_cast<simple_packed> (TInfo->Grid()->PackedData());

				datas.pT = *(t);

				CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**> (&datas.T), opts.N * sizeof(double), cudaHostAllocMapped));

				opts.pT = true;

			}
			else
			{
				datas.T = const_cast<double*> (TInfo->Grid()->Data()->Values());
			}

			if (!isPressureLevel)
			{
				if (PInfo->Grid()->DataIsPacked())
				{
					assert(PInfo->Grid()->PackedData()->ClassName() == "simple_packed");

					shared_ptr<simple_packed> p = dynamic_pointer_cast<simple_packed> (PInfo->Grid()->PackedData());
					
					datas.pP = *(p);

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
				TD -= TBase; // to Celsius
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

		if (conf->StatisticsEnabled())
		{
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

			shared_ptr<info> tempInfo(new info(*myTargetInfo));

			for (tempInfo->ResetParam(); tempInfo->NextParam(); )
			{
				theWriter->ToFile(tempInfo, conf);
			}
		}
	}
}

double tpot::Theta(double P, double T)
{
	// Calculating theta with poissons equation

	double value = T * pow((1000 / P), 0.286);

	return value;
}

double tpot::ThetaW(double P, double T, double TD)
{
	/*
	* Calculating pseudo-adiabatic theta.
	*
	* Method: numerical integration from LCL to 1000mb level
	* along wet adiabatic line.
	*
	* Numerical integration method used: leapfrog starting with euler
	*
	* Originally author AK Sarkanen / May 1985
	*/

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
	/*
	 * Calculate equivalent potential temperature
	 *
	 * Method:
	 *
	 * The approximation given by Holton: Introduction to Dyn. Met.
	 * page 331 is used. If the air is not saturated, it is
	 * taken adiabatically to LCL.
	 *
	 * Original author K Eerola.
	 */

	vector<double> LCL = util::LCL(P, T, TD);

	double TLCL = LCL[1];

	if (theta == kFloatMissing)
	{
		// theta was not calculated in this plugin session :(

		theta = Theta(P, T);
	}

	double ZEs = util::Es(TLCL);
	double ZQs = e * (ZEs / (P - ZEs));

	double value = theta * exp(ZLc * ZQs / Cp / (TLCL + 273.15));

	return value;

}