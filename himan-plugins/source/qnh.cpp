/**
 * @file qnh.cpp
 *
 * Template for future plugins.
 *
 */

#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "qnh.h"

using namespace std;
using namespace himan::plugin;

// Simo Neiglick's text to the original SmartMet macro:
// Local air pressure reduced to mean sea level according to ICAO standard atmosphere
// = QNH. Note: Model pressure reduced to sea level (=QFF) is done using constant
// (current station) temperature, which can differ from QNH by several hPa (when the
// conditions differ significantly from ISA).
//
// ICAO ISA:
//   sea level standard atmospheric pressure p0 = 1013.25 hPa
//   sea level standard temperature T0 = 288.15 K
//   Earth-surface gravitational acceleration g = 9.80665 m/s2.
//   temperature lapse rate L = 0.0065 K/m
//   universal gas constant R = 8.31447 J/(mol K)
//   molar mass of dry air M = 0.0289644 kg/mol
//      p(h) = p0 * (1-L*h/T0)^(g*M/R/L)
// => h(p) = [T0-T0*(p/p0)^(R*L/g/M)] / L
//
// QFE -> QNH:
//   1. calculate ICAO ISA altitude z corresponding to pressure at station (QFE) [m]
//   2. calculate MSL (at station) = z - topo (topography = height of aerodrome) [m]
//   3. calculate p at level MSL in ISA = QNH

qnh::qnh()
{
	itsLogger = logger("qnh");
}

void qnh::Process(std::shared_ptr<const plugin_configuration> conf)
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

	param theRequestedParam("QNH-HPA", 1207, 0, 3, 1);

	// If this param is also used as a source param for other calculations
	// (like for example dewpoint, relative humidity), unit should also be
	// specified

	theRequestedParam.Unit(kPa);

	SetParams({theRequestedParam});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void qnh::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{
	/*
	 * Required source parameters
	 *
	 * eg. param PParam("P-Pa"); for pressure in pascals
	 *
	 */

	const double p0 = 1013.25;
	const double inv_p0 = 1. / 1013.25;
	const double T0 = 288.15;
	const double inv_T0 = 1. / 288.15;
	const double R = 8.31447;
	const double M = 0.0289644;
	const double L = 0.0065;
	const double inv_L = 1. / 0.0065;
	const double qnh_exponent = himan::constants::kG * M / R / L;
	const double alt_exponent = R * L / himan::constants::kG / M;

	const param topoParam("Z-M2S2");
	const params PParams({param("PGR-PA"), param("P-PA")});  // ground level pressure for EC is PGR-PA, for Hirlam P-PA

	level groundLevel(kHeight, 0);

	if (itsConfiguration->SourceProducer().Id() == 131 || itsConfiguration->SourceProducer().Id() == 134)  // EC
	{
		groundLevel = level(himan::kGround, 0);
	}

	auto myThreadedLogger = logger("qnhThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
						   static_cast<string>(forecastLevel));

	// Current time and level

	info_t topoInfo = Fetch(forecastTime, groundLevel, topoParam,
	                        forecastType);  // surface elevation (as geopotential [m2/s2]) from database
	info_t pressureInfo =
	    Fetch(forecastTime, groundLevel, PParams, forecastType);  // ground level (= 0 m) pressure [Pa] from database

	if (!topoInfo || !pressureInfo)
	{
		myThreadedLogger.Info("Skipping step " + to_string(forecastTime.Step()) + ", level " +
		                      static_cast<string>(forecastLevel));

		return;
	}

	// If calculating for hybrid levels, A/B vertical coordinates must be set
	// (copied from source)

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, topoInfo, pressureInfo)
	{
		double topo = topoInfo->Value();
		double pressure = pressureInfo->Value();

		topo = topo * himan::constants::kIg;  // [m2/s2] -> [m]
		if (topo < 0)
		{
			topo = 0;
		}

		pressure = pressure * 0.01;

		/* Calculations go here */

		double altitude = (T0 - T0 * (pow(pressure * inv_p0, alt_exponent))) * inv_L;

		double msl = altitude - topo;

		double qnh = p0 * pow(1 - L * msl * inv_T0, qnh_exponent);

		myTargetInfo->Value(qnh);
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
