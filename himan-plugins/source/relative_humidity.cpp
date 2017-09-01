/**
 * @file relative_humidity.cpp
 *
 */

#include "relative_humidity.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"

using namespace std;
using namespace himan::plugin;

void WithTD(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t TDInfo, double TDBase, double TBase);
void WithQ(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t QInfo, himan::info_t PInfo, double PScale,
           double TBase);
void WithQ(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t QInfo, double P, double TBase);

relative_humidity::relative_humidity()
{
	itsCudaEnabledCalculation = true;

	itsLogger = logger("relative_humidity");
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
	const params PParams = {param("P-HPA"), param("P-PA")};
	const param QParam("Q-KGKG");
	const param TDParam("TD-K");

	auto myThreadedLogger = logger("relative_humidityThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
						  static_cast<string>(forecastLevel));

	double TBase = 0;
	double TDBase = 0;
	double PScale = 1;
	bool isPressureLevel = (myTargetInfo->Level().Type() == kPressure);

	// Temperature is always needed

	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, itsConfiguration->UseCudaForPacking());

	if (!TInfo)
	{
		itsLogger.Warning("Skipping step " + to_string(myTargetInfo->Time().Step()) + ", level " +
		                  to_string(myTargetInfo->Level().Value()));
		return;
	}

	// First try to calculate using Q and P

	bool calculateWithTD = false;

	info_t QInfo = Fetch(forecastTime, forecastLevel, QParam, forecastType, itsConfiguration->UseCudaForPacking());
	info_t PInfo;

	if (!isPressureLevel)
	{
		PInfo = Fetch(forecastTime, forecastLevel, PParams, forecastType, itsConfiguration->UseCudaForPacking());
	}

	if (!QInfo || (!isPressureLevel && !PInfo))
	{
		myThreadedLogger.Debug("Q or P not found, trying calculation with TD");
		calculateWithTD = true;
	}

	info_t TDInfo;

	if (calculateWithTD)
	{
		TDInfo = Fetch(forecastTime, forecastLevel, TDParam, forecastType, itsConfiguration->UseCudaForPacking());

		if (!TDInfo)
		{
			myThreadedLogger.Warning("Skipping step " + to_string(forecastTime.Step()) + ", level " +
			                         static_cast<string>(forecastLevel));
			return;
		}
	}

	assert(!PInfo || TInfo->Grid()->AB() == PInfo->Grid()->AB());
	assert(!TDInfo || TInfo->Grid()->AB() == TDInfo->Grid()->AB());
	assert(!QInfo || TInfo->Grid()->AB() == QInfo->Grid()->AB());

	SetAB(myTargetInfo, TInfo);

	if (TInfo->Param().Unit() == kC)
	{
		TBase = constants::kKelvin;
	}

	if (TDInfo && TDInfo->Param().Unit() == kC)
	{
		TDBase = constants::kKelvin;
	}

	if (!calculateWithTD && !isPressureLevel && PInfo->Param().Name() == "P-PA")
	{
		PScale = 0.01;
	}

	string deviceType;

#ifdef HAVE_CUDA

	if (itsConfiguration->UseCuda())
	{
		deviceType = "GPU";

		if (calculateWithTD)
		{
			auto opts = CudaPrepareTTD(myTargetInfo, TInfo, TDInfo, TDBase, TBase);

			relative_humidity_cuda::Process(*opts);
		}
		else if (isPressureLevel)
		{
			auto opts = CudaPrepareTQ(myTargetInfo, TInfo, QInfo, myTargetInfo->Level().Value(), TBase);

			relative_humidity_cuda::Process(*opts);
		}
		else
		{
			auto opts = CudaPrepareTQP(myTargetInfo, TInfo, QInfo, PInfo, PScale, TBase);

			relative_humidity_cuda::Process(*opts);
		}
	}
	else
#endif
	{
		deviceType = "CPU";

		if (calculateWithTD)
		{
			WithTD(myTargetInfo, TInfo, TDInfo, TDBase, TBase);
		}
		else
		{
			if (isPressureLevel)
			{
				WithQ(myTargetInfo, TInfo, QInfo, myTargetInfo->Level().Value(),
				      TBase);  // Pressure is needed as hPa, no scaling
			}
			else
			{
				WithQ(myTargetInfo, TInfo, QInfo, PInfo, PScale, TBase);
			}
		}
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}

void WithQ(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t QInfo, double P, double TBase)
{
	// Pressure needs to be hPa and temperature C

	for (auto&& tup : zip_range(VEC(myTargetInfo), VEC(TInfo), VEC(QInfo)))
	{
		double& result = tup.get<0>();
		double T = tup.get<1>() + TBase;
		double Q = tup.get<2>();

		double es = himan::metutil::Es_(T) * 0.01;

		result = (P * Q / himan::constants::kEp / es) * (P - es) / (P - Q * P / himan::constants::kEp);

		result = fmin(fmax(0.0, result), 1.0) * 100;  // scale to range 0 .. 100
	}
}

void WithQ(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t QInfo, himan::info_t PInfo, double PScale,
           double TBase)
{
	// Pressure needs to be hPa and temperature C

	for (auto&& tup : zip_range(VEC(myTargetInfo), VEC(TInfo), VEC(QInfo), VEC(PInfo)))
	{
		double& result = tup.get<0>();
		double T = tup.get<1>() + TBase;
		double Q = tup.get<2>();
		double P = tup.get<3>() * PScale;

		double es = himan::metutil::Es_(T) * 0.01;

		result = (P * Q / himan::constants::kEp / es) * (P - es) / (P - Q * P / himan::constants::kEp);

		result = fmin(fmax(0.0, result), 1.0) * 100;  // scale to range 0 .. 100
	}
}

void WithTD(himan::info_t myTargetInfo, himan::info_t TInfo, himan::info_t TDInfo, double TDBase, double TBase)
{
	const double b = 17.27;
	const double c = 237.3;
	const double d = 1.8;

	for (auto&& tup : zip_range(VEC(myTargetInfo), VEC(TInfo), VEC(TDInfo)))
	{
		double& result = tup.get<0>();
		double T = tup.get<1>() + TBase - himan::constants::kKelvin;
		double TD = tup.get<2>() + TDBase - himan::constants::kKelvin;

		result = exp(d + b * (TD / (TD + c))) / exp(d + b * (T / (T + c)));

		if (result > 1.0)
		{
			result = 1.0;
		}
		else if (result < 0.0)
		{
			result = 0.0;
		}

		result *= 100;
	}
}

#ifdef HAVE_CUDA
// Case where RH is calculated from T and TD
unique_ptr<relative_humidity_cuda::options> relative_humidity::CudaPrepareTTD(shared_ptr<info> myTargetInfo,
                                                                              shared_ptr<info> TInfo,
                                                                              shared_ptr<info> TDInfo, double TDBase,
                                                                              double TBase)
{
	unique_ptr<relative_humidity_cuda::options> opts(new relative_humidity_cuda::options);

	opts->N = TInfo->Data().Size();
	opts->select_case = 0;

	opts->TDBase = TDBase;
	opts->TBase = TBase;

	opts->T = TInfo->ToSimple();
	opts->TD = TDInfo->ToSimple();
	opts->RH = myTargetInfo->ToSimple();

	return opts;
}
// Case where RH is calculated from T, Q and P
unique_ptr<relative_humidity_cuda::options> relative_humidity::CudaPrepareTQP(shared_ptr<info> myTargetInfo,
                                                                              shared_ptr<info> TInfo,
                                                                              shared_ptr<info> QInfo,
                                                                              shared_ptr<info> PInfo, double PScale,
                                                                              double TBase)
{
	unique_ptr<relative_humidity_cuda::options> opts(new relative_humidity_cuda::options);

	opts->N = TInfo->Data().Size();

	opts->select_case = 1;

	opts->PScale = PScale;
	opts->TBase = TBase;

	opts->T = TInfo->ToSimple();
	opts->P = PInfo->ToSimple();
	opts->Q = QInfo->ToSimple();
	opts->RH = myTargetInfo->ToSimple();

	return opts;
}
// Case where RH is calculated for pressure levels from T and Q
unique_ptr<relative_humidity_cuda::options> relative_humidity::CudaPrepareTQ(shared_ptr<info> myTargetInfo,
                                                                             shared_ptr<info> TInfo,
                                                                             shared_ptr<info> QInfo, double P_level,
                                                                             double TBase)
{
	unique_ptr<relative_humidity_cuda::options> opts(new relative_humidity_cuda::options);

	opts->N = TInfo->Data().Size();

	opts->select_case = 2;

	opts->TBase = TBase;

	opts->P_level = P_level;
	opts->T = TInfo->ToSimple();
	opts->Q = QInfo->ToSimple();
	opts->RH = myTargetInfo->ToSimple();

	return opts;
}

#endif
