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
#include <boost/lexical_cast.hpp>
#include "level.h"
#include "forecast_time.h"

using namespace std;
using namespace himan::plugin;

#include "tpot_cuda.h"
#include "cuda_helper.h"
#include "metutil.h"

tpot::tpot()
: itsThetaCalculation(false)
, itsThetaWCalculation(false)
, itsThetaECalculation(false)
{
	itsClearTextFormula = "TP = Tk * pow((1000/P), 0.286) ; TPW calculated with LCL ; TPE = X"; 
	itsCudaEnabledCalculation = true;

	itsLogger = logger_factory::Instance()->GetLog("tpot");

}

void tpot::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to potential temperature
	 */

	vector<param> theParams;

	if (itsConfiguration->Exists("theta") && itsConfiguration->GetValue("theta") == "true")
	{
		itsThetaCalculation = true;

		itsLogger->Trace("Theta calculation requested");

		param p("TP-K", 8, 0, 0, 2);

		theParams.push_back(p);
	}

	if (itsConfiguration->Exists("thetaw") && itsConfiguration->GetValue("thetaw") == "true")
	{
		itsThetaWCalculation = true;

		itsLogger->Trace("ThetaW calculation requested");

		// Sharing GRIB2 number with thetae!

		param p("TPW-K", 9, 0, 0, 3);

		theParams.push_back(p);
	}

	if (itsConfiguration->Exists("thetae") && itsConfiguration->GetValue("thetae") == "true")
	{
		itsThetaECalculation = true;

		itsLogger->Trace("ThetaE calculation requested");

		param p("TPE-K", 129, 0, 0, 3);

		// Sharing GRIB2 number with thetaw!

		theParams.push_back(p);
	}

	if (theParams.size() == 0)
	{
		// By default assume we'll calculate theta

		itsThetaCalculation = true;

		param p("TP-K", 8, 0, 0, 2);

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

	auto myThreadedLogger = logger_factory::Instance()->GetLog("tpotThread #" + boost::lexical_cast<string> (threadIndex));

	const param TParam("T-K");
	const params PParam = { param("P-PA"), param("P-HPA") };
	const params TDParam = { param("TD-C"), param("TD-K") };
	
	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " +static_cast<string> (forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	double PScale = 1;
	double TBase = 0;
	double TDBase  = 0;
		
	info_t TInfo = Fetch(forecastTime, forecastLevel, param("T-K"), myTargetInfo->ForecastType(), itsConfiguration->UseCudaForPacking());
	info_t TDInfo, PInfo;

	bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

	if (!isPressureLevel)
	{
		PInfo = Fetch(forecastTime, forecastLevel, PParam, myTargetInfo->ForecastType(), itsConfiguration->UseCudaForPacking());

		if (PInfo && (PInfo->Param().Unit() == kHPa || PInfo->Param().Name() == "P-HPA"))
		{
			PScale = 100;
		}
	}

	if (itsThetaWCalculation || itsThetaECalculation)
	{
		TDInfo = Fetch(forecastTime, forecastLevel, TDParam, myTargetInfo->ForecastType(), itsConfiguration->UseCudaForPacking());
	}

	if (!TInfo || (!isPressureLevel && !PInfo) || ((itsThetaWCalculation || itsThetaECalculation) && !TDInfo))
	{
		myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
		return;
	}

	assert(isPressureLevel || ((PInfo->Grid()->AB() == TInfo->Grid()->AB()) && (!TDInfo || (PInfo->Grid()->AB() == TDInfo->Grid()->AB()))));

	SetAB(myTargetInfo, TInfo);

	if (TInfo->Param().Unit() == kC)
	{
		TBase = -himan::constants::kKelvin;
	}

	if (TDInfo && TDInfo->Param().Unit() == kC)
	{
		TDBase = -himan::constants::kKelvin;
	}

	string deviceType;

#ifdef HAVE_CUDA

	if (useCudaInThisThread)
	{
		deviceType = "GPU";

		auto opts = CudaPrepare(myTargetInfo, TInfo, PInfo, TDInfo);

		tpot_cuda::Process(*opts);

	}
	else
#endif
	{
		deviceType = "CPU";

		if (PInfo)
		{
			PInfo->ResetLocation();
		}

		if (TDInfo)
		{
			TDInfo->ResetLocation();
		}

		LOCKSTEP(myTargetInfo, TInfo)
		{
		
			double T = TInfo->Value() + TBase; // to Kelvin
			
			double P = kFloatMissing, TD = kFloatMissing;

			if (isPressureLevel)
			{
				P = myTargetInfo->Level().Value() * 100;
			}
			else
			{
				PInfo->NextLocation();
				P = PInfo->Value() * PScale; // to Pa
			}

			if (itsThetaWCalculation || itsThetaECalculation)
			{
				TDInfo->NextLocation();
				TD = TDInfo->Value() + TDBase; // to Kelvin
			}

			if (T == kFloatMissing || P == kFloatMissing || ((itsThetaECalculation || itsThetaWCalculation) && TD == kFloatMissing))
			{
				continue;
			}

			double theta = kFloatMissing;

			if (itsThetaCalculation)
			{
				theta = Theta(P, T);

				myTargetInfo->Param(param("TP-K"));

				myTargetInfo->Value(theta);
			}

			if (itsThetaWCalculation)
			{
				double value = ThetaW(P, T, TD);

				myTargetInfo->Param(param("TPW-K"));

				myTargetInfo->Value(value);
			}

			if (itsThetaECalculation)
			{
				double value = ThetaE(P, T, TD, theta);

				myTargetInfo->Param(param("TPE-K"));

				!myTargetInfo->Value(value);
			}
		}
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}

double tpot::Theta(double P, double T)
{
	double value = T * pow((1000 / (P*0.01)), 0.28586);

	return value;
}

double tpot::ThetaW(double P, double T, double TD)
{
	
	double value = kFloatMissing;

	// Search LCL level
	lcl_t LCL = metutil::LCL_(P, T, TD);

	double Pint = LCL.P; // Pa
	double Tint = LCL.T; // K

	if (Tint == kFloatMissing || Pint == kFloatMissing)
	{
		return kFloatMissing;
	}
	else
	{

		const double Pstep = 500; // Pa
		int i = 0;
		
		/*
		 * Units: Temperature in Kelvins, Pressure in Pascals
		 */

		double T0 = Tint;

		double Z = kFloatMissing;

		while (++i < 500) // usually we don't reach this value
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

			// Gammas() takes Pa
			Tint = T0 + metutil::Gammas_(Pint, Tint) * Z;

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

	assert(value == value); // check NaN

	return value;
}

double tpot::ThetaE(double P, double T, double TD, double theta)
{
	lcl_t LCL = metutil::LCL_(P, T, TD);

	if (LCL.T == kFloatMissing)
	{
		return kFloatMissing;
	}
	else if (theta == kFloatMissing)
	{
		// theta was not calculated in this plugin session :(

		theta = Theta(P, T);

		if (theta == kFloatMissing)
		{
			return theta;
		}
	}

	theta -= constants::kKelvin;
	
	double Es = metutil::Es_(LCL.T) * 0.01;
	double ZQs = himan::constants::kEp * (Es / (P*0.01 - Es));

	double value = theta * exp(himan::constants::kL * ZQs / himan::constants::kCp / (LCL.T));

	return value + constants::kKelvin;

}

#ifdef HAVE_CUDA

unique_ptr<tpot_cuda::options> tpot::CudaPrepare(shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> PInfo, shared_ptr<info> TDInfo)
{
	unique_ptr<tpot_cuda::options> opts(new tpot_cuda::options);

	opts->is_constant_pressure = (myTargetInfo->Level().Type() == kPressure);

	opts->t = TInfo->ToSimple();
	opts->theta = itsThetaCalculation;

	if (opts->theta)
	{
		myTargetInfo->Param(param("TP-K"));
		opts->tp = myTargetInfo->ToSimple();
	}

	opts->thetaw = itsThetaWCalculation;

	if (opts->thetaw)
	{
		myTargetInfo->Param(param("TPW-K"));
		opts->tpw = myTargetInfo->ToSimple();
	}

	opts->thetae = itsThetaECalculation;

	if (opts->thetae)
	{
		myTargetInfo->Param(param("TPE-K")); 
		opts->tpe = myTargetInfo->ToSimple();
	}

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

	if (TDInfo)
	{
		opts->td = TDInfo->ToSimple();

		if (TDInfo->Param().Unit() == kC)
		{
			opts->td_base = himan::constants::kKelvin;
		}
	}

	opts->N = opts->t->size_x * opts->t->size_y;

	if (TInfo->Param().Unit() == kC)
	{
		opts->t_base = himan::constants::kKelvin;
	}

	return opts;
}

#endif