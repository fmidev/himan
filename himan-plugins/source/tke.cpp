/**
 * @file tke.cpp
 *
 */
#include <math.h>

#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "tke.h"
#include "util.h"

using namespace std;
using namespace himan::plugin;

tke::tke() : itsTopLevel()
{
	itsLogger = logger("tke");
}

void tke::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter properties
	 * - name PARM_NAME, this name is found from neons. For example: T-K
	 * - univ_id UNIV_ID, newbase-id, ie code table 204
	 * - grib1 id must be in database
	 * - grib2 descriptor X'Y'Z, http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-2.shtml
	 *
	 */

	// param theRequestedParam(PARM_NAME, UNIV_ID, GRIB2DISCIPLINE, GRIB2CATEGORY, GRIB2NUMBER);
	param TKE("TKEN-JKG", 301, 0, 19, 11);
	// If this param is also used as a source param for other calculations
	// (like for example dewpoint, relative humidity), unit should also be
	// specified

	// TKE.Unit(77);

	SetParams({TKE});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void tke::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	/*
	 * Required source parameters
	 */

	const param FrvelParam("FRVEL-MS");
	const param MoninObukhovParam("MOL-M");
	const param MixHgtParam("MIXHGT-M");
	const param TGParam("TG-K");
	const param PGParam("P-PA");
	const param SHFParam("FLSEN-JM2");  // accumulated surface sensible heat flux
	const param LHFParam("FLLAT-JM2");  // accumulated surface latent heat flux
	const param ZParam("HL-M");         // model level height
	// ----

	// Current time and level as given to this thread
	int paramStep = 1;  // myTargetInfo->Param().Aggregation().TimeResolutionValue();
	HPTimeResolution timeResolution = myTargetInfo->Time().StepResolution();

	forecast_time forecastTime = myTargetInfo->Time();
	forecast_time forecastTimePrev = myTargetInfo->Time();
	forecastTimePrev.ValidDateTime().Adjust(timeResolution, -paramStep);

	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	level groundLevel = level(himan::kHeight, 0, "HEIGHT");

	auto myThreadedLogger = logger("tke_pluginThread #" + to_string(threadIndex));

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	info_t FrvelInfo = Fetch(forecastTime, groundLevel, FrvelParam, forecastType, false);
	info_t MoninObukhovInfo = Fetch(forecastTime, groundLevel, MoninObukhovParam, forecastType, false);
	info_t MixHgtInfo = Fetch(forecastTime, groundLevel, MixHgtParam, forecastType, false);
	info_t TGInfo = Fetch(forecastTime, level(himan::kGroundDepth, 0, 7), TGParam, forecastType, false);
	info_t PGInfo = Fetch(forecastTime, groundLevel, PGParam, forecastType, false);
	info_t SHFInfo = Fetch(forecastTime, groundLevel, SHFParam, forecastType, false);
	info_t SHFPrevInfo = Fetch(forecastTimePrev, groundLevel, SHFParam, forecastType, false);
	info_t LHFInfo = Fetch(forecastTime, groundLevel, LHFParam, forecastType, false);
	info_t LHFPrevInfo = Fetch(forecastTimePrev, groundLevel, LHFParam, forecastType, false);
	info_t ZInfo = Fetch(forecastTime, forecastLevel, ZParam, forecastType, false);

	if (!(FrvelInfo && MoninObukhovInfo && MixHgtInfo && TGInfo && PGInfo && SHFInfo && SHFPrevInfo && LHFInfo &&
	      LHFPrevInfo && ZInfo))
	{
		myThreadedLogger.Info("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                      static_cast<string>(forecastLevel));
		return;
	}

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

	SetAB(myTargetInfo, ZInfo);

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, MoninObukhovInfo, MixHgtInfo, TGInfo, PGInfo, SHFInfo, SHFPrevInfo, LHFInfo, LHFPrevInfo,
	         ZInfo, FrvelInfo)
	{
		double TKE = 0;
		double Frvel = FrvelInfo->Value();
		double MoninObukhov = MoninObukhovInfo->Value();
		double MixHgt = MixHgtInfo->Value();
		double TG = TGInfo->Value();
		double PG = PGInfo->Value();
		double SHF = SHFInfo->Value() - SHFPrevInfo->Value();
		double LHF = LHFInfo->Value() - LHFPrevInfo->Value();
		double Z = ZInfo->Value();

		SHF /= forecastStepSize;  // divide by time step to obtain Watts/m2
		LHF /= forecastStepSize;  // divide by time step to obtain Watts/m2

		double T_C = TG - constants::kKelvin;     // Convert Temperature to Celvins
		double cp = 1.0056e3 + 0.017766 * T_C;    // Calculate specific heat capacity, linear approximation
		double rho = PG / (constants::kRd * TG);  // Calculate density

		if (Z > MixHgt)
		{
			TKE = MissingDouble();
		}
		else if (Z * MoninObukhov >= 0)
		{
			if (Z <= 0.1 * MixHgt)
			{
				TKE = 6 * Frvel * Frvel;
			}
			else
			{
				TKE = 6 * Frvel * Frvel * pow(1 - Z / MixHgt, 1.75);
			}
		}
		else
		{
			if (abs(Z * MoninObukhov) > 0.5)
			{
				if (Z <= 0.1 * MixHgt)
				{
					TKE = 0.36 * pow(constants::kG / TG * (SHF / (rho * cp) + 0.61 * TG * LHF * MoninObukhov / cp) *
					                     MixHgt,
					                 2 / 3) +
					      0.85 * Frvel * Frvel * pow(1 - 3 * Z * MoninObukhov, 2 / 3);
				}
				else
				{
					TKE = 0.54 *
					      pow(constants::kG / TG * (SHF / (rho * cp) + 0.61 * TG * LHF * MoninObukhov / cp) * MixHgt,
					          2 / 3);
				}
			}
			else if (0.02 < abs(Z * MoninObukhov) && abs(Z * MoninObukhov) <= 0.5)
			{
				TKE = 0.54 * pow(constants::kG / TG * (SHF / (rho * cp) + 0.61 * TG * LHF * MoninObukhov / cp) * MixHgt,
				                 2 / 3);
			}
			else
			{
				TKE = 6 * Frvel * Frvel * pow(1 - Z / MixHgt, 1.75);
			}
		}

		// return result
		myTargetInfo->Value(TKE);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
