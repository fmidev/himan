/**
 * @file relative_humidity.cpp
 *
 * @date Jan 21, 2012
 * @author partio
 */

#include "relative_humidity.h"
#include "logger_factory.h"
#include <boost/lexical_cast.hpp>
#include "metutil.h"
#include "level.h"
#include "forecast_time.h"

using namespace std;
using namespace himan::plugin;

const double b = 17.27;
const double c = 237.3;
const double d = 1.8;

relative_humidity::relative_humidity()
{
	itsClearTextFormula = "RH = 100 *  (P * Q / 0.622 / es) * (P - es) / (P - Q * P / 0.622)";
	itsCudaEnabledCalculation = true;

	itsLogger = logger_factory::Instance()->GetLog("relative_humidity");

}

void relative_humidity::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({param("RH-PRCNT", 13, 0, 1, 1)});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void relative_humidity::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{	

	const param TParam("T-K");
	const params PParams = { param("P-HPA"), param("P-PA") };
	const param QParam("Q-KGKG");
	const param TDParam("TD-C");

	auto myThreadedLogger = logger_factory::Instance()->GetLog("relative_humidityThread #" + boost::lexical_cast<string> (threadIndex));
	
	bool useCudaInThisThread = compiled_plugin_base::GetAndSetCuda(threadIndex);

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Info("Calculating time " + static_cast<string>(*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	double TBase = 0;
	double TDBase = 0;
	double PScale = 1;
	bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

	// Temperature is always needed

	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, itsConfiguration->UseCudaForPacking() && useCudaInThisThread);

	if (!TInfo)
	{
		itsLogger->Warning("Skipping step " + boost::lexical_cast<string> (myTargetInfo->Time().Step()) + ", level " + boost::lexical_cast<string> (myTargetInfo->Level().Value()));
		return;
	}

	// First try to calculate using Q and P

	bool calculateWithTD = false;

	info_t QInfo = Fetch(forecastTime, forecastLevel, QParam, itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
	info_t PInfo;
	
	if (!isPressureLevel)
	{
		PInfo = Fetch(forecastTime, forecastLevel, PParams, itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
	}

	if (!QInfo || (!isPressureLevel && !PInfo))
	{
		myThreadedLogger->Debug("Q or P not found, trying calculation with TD");
		calculateWithTD = true;
	}

	info_t TDInfo;
	
	if (calculateWithTD)
	{
		TDInfo = Fetch(forecastTime, forecastLevel, TDParam, itsConfiguration->UseCudaForPacking() && useCudaInThisThread);
		
		if (!TDInfo)
		{
			myThreadedLogger->Warning("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
			return;
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

	string deviceType;

#ifdef HAVE_CUDA

	if (useCudaInThisThread)
	{

		deviceType = "GPU";

		if (calculateWithTD)
		{
			auto opts = CudaPrepareTTD(myTargetInfo, TInfo, TDInfo, TDBase, TBase);

			relative_humidity_cuda::Process(*opts);

			CudaFinish(move(opts), myTargetInfo, TInfo, TDInfo);
		}
		else if (isPressureLevel)
		{
			auto opts = CudaPrepareTQ(myTargetInfo, TInfo, QInfo, myTargetInfo->Level().Value(), TBase);

			relative_humidity_cuda::Process(*opts);

			CudaFinish(move(opts), myTargetInfo, TInfo, QInfo);
		}
		else
		{
			auto opts = CudaPrepareTQP(myTargetInfo, TInfo, QInfo, PInfo, PScale, TBase);

			relative_humidity_cuda::Process(*opts);

			CudaFinish(move(opts), myTargetInfo, TInfo, QInfo, PInfo);
		}
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

		if (QInfo)
		{
			QInfo->ResetLocation();
		}

		LOCKSTEP(myTargetInfo, TInfo)
		{

			double T = TInfo->Value();

			if (T == kFloatMissing)
			{
				continue;
			}

			double TD = kFloatMissing;
			double P = kFloatMissing;
			double Q = kFloatMissing;
			double RH = kFloatMissing;

			T += TBase;

			if (calculateWithTD)
			{
				TDInfo->NextLocation();
				TD = TDInfo->Value();

				if (TD == kFloatMissing)
				{
					continue;
				}

				RH = WithTD(T, TD + TDBase);
			}
			else
			{
				if (isPressureLevel)
				{
					P = myTargetInfo->Level().Value(); // Pressure is needed as hPa, no scaling
				}
				else
				{
					PInfo->NextLocation();
					P = PInfo->Value();
				}

				QInfo->NextLocation();
				Q = QInfo->Value();

				if (P == kFloatMissing || Q == kFloatMissing)
				{
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

			myTargetInfo->Value(RH);
		}

	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data()->MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data()->Size()));

}

inline
double relative_humidity::WithQ(double T, double Q, double P)
{
	// Pressure needs to be hPa and temperature C

	double es = metutil::Es_(T + constants::kKelvin) * 0.01;
	
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
unique_ptr<relative_humidity_cuda::options> relative_humidity::CudaPrepareTTD( shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> TDInfo, double TDBase, double TBase)
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
unique_ptr<relative_humidity_cuda::options> relative_humidity::CudaPrepareTQP( shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> QInfo, shared_ptr<info> PInfo, double PScale, double TBase)
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
unique_ptr<relative_humidity_cuda::options> relative_humidity::CudaPrepareTQ( shared_ptr<info> myTargetInfo, shared_ptr<info> TInfo, shared_ptr<info> QInfo, double P_level, double TBase)
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

}
#endif
