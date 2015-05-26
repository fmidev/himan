/**
 * @file pot.cpp
 *
 *
 * @date May 25, 2015
 * @author Tack
 */

#include <boost/lexical_cast.hpp>
#include <math.h>

#include "pot.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include "level.h"
#include "forecast_time.h"
#include "regular_grid.h"
#include "util.h"

using namespace std;
using namespace himan::plugin;

pot::pot()
{
    itsClearTextFormula = "complex formula";

    itsLogger = logger_factory::Instance()->GetLog("pot");
}

void pot::Process(std::shared_ptr<const plugin_configuration> conf)
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
    param POT("POT-N", 9999, 0, 192, 15);
    // If this param is also used as a source param for other calculations
    // (like for example dewpoint, relative humidity), unit should also be
    // specified

    POT.Unit(kUnknownUnit);

    SetParams({POT});

    Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void pot::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{

    /*
     * Required source parameters
     */

    const param CapeParam("CAPE-JKG");
    const param RainParam("RRR-KGM2");
    // ----

    // Current time and level as given to this thread

    forecast_time forecastTime = myTargetInfo->Time();
    level forecastLevel = myTargetInfo->Level();
    forecast_type forecastType = myTargetInfo->ForecastType();

    auto myThreadedLogger = logger_factory::Instance()->GetLog("pot_pluginThread #" + boost::lexical_cast<string> (threadIndex));

    myThreadedLogger->Debug("Calculating time " + static_cast<string> (forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

    info_t CAPEInfo, RRInfo;

    CAPEInfo = Fetch(forecastTime, forecastLevel, CapeParam, forecastType, false);
    RRInfo = Fetch(forecastTime, forecastLevel, RainParam, forecastType, false);

    if (!(CAPEInfo && RRInfo))
    {
        myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
        return;
    }

    string deviceType = "CPU";

    LOCKSTEP(myTargetInfo, CAPEInfo, RRInfo)
    {
	double POT;
        double CAPE = CAPEInfo->Value();
        double RR = RRInfo->Value();
        double LAT = myTargetInfo->LatLon().X();

	double PoLift = 0;
	double PoThermoDyn = 0;


        if (CAPE == kFloatMissing || RR == kFloatMissing)
        {
            continue;
        }

	if (CAPE >= 100 && CAPE <= 1000)           
	{
		PoThermoDyn = 0.434294482 * log(CAPE) - 2;    //salamoinnin todennäköisyyden on oikeastikin havaittu kasvavan logaritmisesti CAPE-arvojen kasvaessa.
	}

	if (CAPE > 1000)
	{
		PoThermoDyn = 1;
	}

	// Tropiikki käsitellään vähän eri tavalla, koska siellä CAPE <---> salamointi -relaatio on erilainen

	if (LAT <20 && LAT > -20)
	{
 
		if (CAPE >= 500 && CAPE <= 2000)           
		{
			PoThermoDyn = 0.72134752 * log(CAPE) - 4.482892142;
		}

		if (CAPE > 2000)
		{
			PoThermoDyn = 1;
		}
	}

	// Laukaisevan tekijän (Lift) todennäköisyys

	//sade = MAX(rr_ec -1 -1 1 1)   //käytetään sadeparametrina mallin sateesta kurkkausmenetelmällä lähi-hilapisteistä poimittuja maksimiarvoja

	if (RR >= 0.05 && RR <= 5)           
   	{
		PoLift  = 0.217147241 * log(RR) + 0.650514998;	// Funktio kasvaa nopeasti logaritmisesti nollasta ykköseen, kun sateen intensiteetti saa arvoja 0.05 ---> 5 mm/h
	}

	if (RR > 5)
   	{
		PoLift = 1;
	}

	//POT on ainesosiensa todennäköisyyksien tulo
 
	POT = PoLift * PoThermoDyn * 100;

        //return result
        myTargetInfo->Value(POT);
    }

    myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}
