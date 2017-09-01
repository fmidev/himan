/**
 * @file monin_obukhov.cpp
 *
 * Claculates the inverse of the Monin-Obukhov length.
 *
 */

#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "metutil.h"
#include "monin_obukhov.h"

using namespace std;
using namespace himan::plugin;

monin_obukhov::monin_obukhov()
{
	itsLogger = logger("monin_obukhov");
}

void monin_obukhov::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param theRequestedParam("MOL-M", 1204);

	SetParams({theRequestedParam});

	Start();
}

void monin_obukhov::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	const param TParam("T-K");          // ground Temperature
	const param SHFParam("FLSEN-JM2");  // accumulated surface sensible heat flux
	const param LHFParam("FLLAT-JM2");  // accumulated surface latent heat flux
	const param U_SParam("FRVEL-MS");   // friction velocity
	const param PParam("P-PA");

	auto myThreadedLogger = logger("monin_obukhov Thread #" + std::to_string(threadIndex));

	// Prev/current time and level

	int paramStep = 1;
	HPTimeResolution timeResolution = myTargetInfo->Time().StepResolution();

	forecast_time forecastTime = myTargetInfo->Time();
	forecast_time forecastTimePrev = myTargetInfo->Time();
	forecastTimePrev.ValidDateTime().Adjust(timeResolution, -paramStep);
	forecast_type forecastType = myTargetInfo->ForecastType();

	level forecastLevel = level(himan::kHeight, 0, "Height");
	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
						  static_cast<string>(forecastLevel));

	info_t TInfo = Fetch(forecastTime, forecastLevel, TParam, forecastType, false);
	info_t SHFInfo = Fetch(forecastTime, forecastLevel, SHFParam, forecastType, false);
	info_t PrevSHFInfo = Fetch(forecastTimePrev, forecastLevel, SHFParam, forecastType, false);
	info_t LHFInfo = Fetch(forecastTime, forecastLevel, LHFParam, forecastType, false);
	info_t PrevLHFInfo = Fetch(forecastTimePrev, forecastLevel, LHFParam, forecastType, false);
	info_t U_SInfo = Fetch(forecastTime, forecastLevel, U_SParam, forecastType, false);
	info_t PInfo = Fetch(forecastTime, forecastLevel, PParam, forecastType, false);

	// determine length of forecast step to calculate surface heat flux in W/m2
	double forecastStepSize;

	if (itsConfiguration->SourceProducer().Id() != 199)
	{
		forecastStepSize = itsConfiguration->ForecastStep() * 3600;  // step size in seconds
	}
	else
	{
		forecastStepSize = itsConfiguration->ForecastStep() * 60;  // step size in seconds
	}

	if (!TInfo || !SHFInfo || !U_SInfo || !PInfo || !PrevSHFInfo || !LHFInfo || !PrevLHFInfo)
	{
		myThreadedLogger.Info("Skipping step " + std::to_string(forecastTime.Step()) + ", level " +
							  static_cast<string>(forecastLevel));
		return;
	}

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, TInfo, SHFInfo, PrevSHFInfo, LHFInfo, PrevLHFInfo, U_SInfo, PInfo)
	{
		double T = TInfo->Value();
		double SHF = SHFInfo->Value() - PrevSHFInfo->Value();
		double LHF = LHFInfo->Value() - PrevLHFInfo->Value();
		double U_S = U_SInfo->Value();
		double P = PInfo->Value();

		double T_C = T - constants::kKelvin;  // Convert Temperature to Celvins
		double mol = MissingDouble();

		SHF /= forecastStepSize;  // divide by time step to obtain Watts/m2
		LHF /= forecastStepSize;  // divide by time step to obtain Watts/m2

		// Calculation of the inverse of Monin-Obukhov length to avoid division by 0

		if (U_S != 0.0)
		{
			double rho = P / (constants::kRd * T);  // Calculate density

			double cp = 1.0056e3 + 0.017766 * T_C + 4.0501e-4 * pow(T_C, 2) - 1.017e-6 * pow(T_C, 3) +
			            1.4715e-8 * pow(T_C, 4) - 7.4022e-11 * pow(T_C, 5) +
			            1.2521e-13 * pow(T_C, 6);  // Calculate specific heat capacity
			double L = 2500800.0 - 2360.0 * T_C + 1.6 * pow(T_C, 2) -
			           0.06 * pow(T_C, 3);  // Calculate specific latent heat of condensation of water

			mol = -constants::kG * constants::kK * (SHF + 0.61 * cp * T / L * LHF) /
			      (rho * cp * U_S * U_S * U_S * himan::metutil::VirtualTemperature_(T, P));
		}
		myTargetInfo->Value(mol);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " +
						  std::to_string(myTargetInfo->Data().MissingCount()) + "/" +
						  std::to_string(myTargetInfo->Data().Size()));
}
