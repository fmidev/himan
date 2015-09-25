/**
 * @file tke.cpp
 *
 *
 * @date Jun 15, 2015
 * @author Tack
 */

#include <boost/lexical_cast.hpp>
#include <math.h>

#include "tke.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include "level.h"
#include "forecast_time.h"
#include "regular_grid.h"
#include "util.h"
#include "numerical_functions.h"

using namespace std;
using namespace himan::plugin;

tke::tke()
{
    itsClearTextFormula = "complex formula";

    itsLogger = logger_factory::Instance()->GetLog("tke");
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

    //TKE.Unit(77);

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
    const param PGParam("PGR-PA");
    const param QParam("FLSEN-JM2"); // accumulated surface heat flux
    const param QPrevParam("FLSEN-JM2"); // accumulated surface heat flux
    const param ZParam("HL-M"); // model level height 
    // ----

    // Current time and level as given to this thread
    int paramStep = 1; // myTargetInfo->Param().Aggregation().TimeResolutionValue();
    HPTimeResolution timeResolution = myTargetInfo->Time().StepResolution();


    forecast_time forecastTime = myTargetInfo->Time();
    forecast_time forecastTimePrev = myTargetInfo->Time();
    forecastTimePrev.ValidDateTime().Adjust(timeResolution, -paramStep);

    level forecastLevel = myTargetInfo->Level();
    forecast_type forecastType = myTargetInfo->ForecastType();

    level groundLevel;

    // this will come back to us
    if ( itsConfiguration->SourceProducer().Id() == 131)
    {
            groundLevel = level(himan::kGndLayer, 0, "GNDLAYER");
    }
    else
    {
            groundLevel = level(himan::kHeight, 0, "HEIGHT");
    }

    auto myThreadedLogger = logger_factory::Instance()->GetLog("tke_pluginThread #" + boost::lexical_cast<string> (threadIndex));

    myThreadedLogger->Debug("Calculating time " + static_cast<string> (forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

    info_t FrvelInfo, MoninObukhovInfo, MixHgtInfo, TGInfo, QInfo, QPrevInfo, ZInfo, PGInfo;

    FrvelInfo = Fetch(forecastTime, groundLevel, FrvelParam, forecastType, false);
    MoninObukhovInfo = Fetch(forecastTime, groundLevel, MoninObukhovParam, forecastType, false);
    MixHgtInfo = Fetch(forecastTime, groundLevel, MixHgtParam, forecastType, false);
    TGInfo = Fetch(forecastTime, groundLevel, TGParam, forecastType, false);
    PGInfo = Fetch(forecastTime, groundLevel, PGParam, forecastType, false);
    QInfo = Fetch(forecastTime, groundLevel, QParam, forecastType, false);
    QPrevInfo  = Fetch(forecastTimePrev, groundLevel, QParam, forecastType, false);
    ZInfo = Fetch(forecastTime, forecastLevel, TGParam, forecastType, false);

    /*if (!())
    {
        myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
        return;
    }
    */

    // determine length of forecast step to calculate surface heat flux in W/m2
    double forecastStepSize;

    if ( itsConfiguration->SourceProducer().Id() != 199)
    {
            forecastStepSize = itsConfiguration->ForecastStep()*3600; //step size in seconds
    }
    else
    {
            forecastStepSize = itsConfiguration->ForecastStep()*60; //step size in seconds
    }


    string deviceType = "CPU";

    LOCKSTEP(myTargetInfo)
    {
        double TKE;
        double Frvel = FrvelInfo->Value();
        double MoninObukhov = MoninObukhovInfo->Value();
        double MixHgt = MixHgtInfo->Value();
        double TG = TGInfo->Value();
        double PG = PGInfo->Value();
        double Q = QPrevInfo->Value()-QInfo->Value();
        double Z = ZInfo->Value();

        Q /= forecastStepSize; // divide by time step to obtain Watts/m2
        double T_C = TG - constants::kKelvin; // Convert Temperature to Celvins
        double cp = 1.0056e3 + 0.017766 * T_C; // Calculate specific heat capacity, linear approximation
        double rho =  PG / (constants::kRd * TG); // Calculate density

        if (Z > MixHgt)
        {
            TKE = kFloatMissing;
        }
        else if (Z*MoninObukhov >= 0)
        {
            if (Z <= 0.1*MixHgt) 
            {
                TKE = 6*Frvel*Frvel;
            }
            else
            {
                TKE = 6*Frvel*Frvel*pow(1-Z/MixHgt,1.75);
            }
        }
        else
        {
            if (abs(Z*MoninObukhov) > 0.5)
            {
                if (Z <= 0.1*MixHgt)
                {
                    TKE = 0.36*pow(constants::kG/TG*Q/(rho*cp)*MixHgt,2/3)+0.85*Frvel*Frvel*pow(1-3*Z*MoninObukhov,2/3);
                }
                else
                {
                    TKE = 0.54*pow(constants::kG/TG*Q/(rho*cp)*MixHgt,2/3);
                }
            }
            else if (0.02 < abs(Z*MoninObukhov) && abs(Z*MoninObukhov) <= 0.5)
            {
                TKE = 0.54*pow(constants::kG/TG*Q/(rho*cp)*MixHgt,2/3);
            }
            else
            {
                TKE = 6*Frvel*Frvel*pow(1-Z/MixHgt,1.75);
            }
        }

        //return result
        myTargetInfo->Value(TKE);
    }
    

    myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}
