/**
 * @file turbulence.cpp
 *
 * @date Jan 7, 2015
 * @author Tack
 */

#include <boost/lexical_cast.hpp>

#include "turbulence.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include "level.h"
#include "forecast_time.h"
#include "regular_grid.h"
#include "util.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan::plugin;

turbulence::turbulence()
{
    itsClearTextFormula = "complex formula";

    itsLogger = logger_factory::Instance()->GetLog("turbulence");
}

void turbulence::Process(std::shared_ptr<const plugin_configuration> conf)
{
    Init(conf);

    itsInfo->LevelOrder(kTopToBottom);

    //shared_ptr<neons> theNeons = dynamic_pointer_cast <neons> (plugin_factory::Instance()->Plugin("neons"));

    //itsTopLevel = boost::lexical_cast<int> (theNeons->ProducerMetaData(itsConfiguration->SourceProducer().Id(), "first hybrid level number"));

    if (Dimension() != kTimeDimension)
    {
        itsLogger->Info("Changing leading_dimension to time");
        Dimension(kTimeDimension);
    }

    /*
     * Set target parameter properties
     * - name PARM_NAME, this name is found from neons. For example: T-K
     * - univ_id UNIV_ID, newbase-id, ie code table 204
     * - grib1 id must be in database
     * - grib2 descriptor X'Y'Z, http://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_table4-2.shtml
     *
     */

    // param theRequestedParam(PARM_NAME, UNIV_ID, GRIB2DISCIPLINE, GRIB2CATEGORY, GRIB2NUMBER);
    param TI("TI-S2", 1208);
    param TI2("TI2-S2", 1209);
    // If this param is also used as a source param for other calculations
    // (like for example dewpoint, relative humidity), unit should also be
    // specified

    TI.Unit(kS2);
    TI2.Unit(kS2);

    SetParams({TI,TI2});

    Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void turbulence::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{

    /*
     * Required source parameters
     */

    const param UParam("U-MS");
    const param VParam("V-MS");
    const param HParam("HL-M");
    // ----

    // Current time and level as given to this thread

    forecast_time forecastTime = myTargetInfo->Time();
    level forecastLevel = myTargetInfo->Level();

    level prevLevel;

    prevLevel = level(myTargetInfo->Level());
    prevLevel.Value(myTargetInfo->Level().Value() - 1);
    prevLevel.Index(prevLevel.Index() - 1);


    auto myThreadedLogger = logger_factory::Instance()->GetLog("turbulence_pluginThread #" + boost::lexical_cast<string> (threadIndex));

    myThreadedLogger->Debug("Calculating time " + static_cast<string> (forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

    /*bool firstLevel = false;

    if (myTargetInfo->Level().Value() == itsTopLevel)
    {
        firstLevel = true;
    }

    if (firstLevel)
    {
    	myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", first hybrid level " + static_cast<string> (forecastLevel));
    	return;
    }*/

    info_t UInfo, VInfo, HInfo, prevUInfo, prevVInfo, prevHInfo;

    prevHInfo = Fetch(forecastTime, prevLevel, HParam, false);
    prevUInfo = Fetch(forecastTime, prevLevel, UParam, false);
    prevVInfo = Fetch(forecastTime, prevLevel, VParam, false);

    HInfo = Fetch(forecastTime, forecastLevel, HParam, false);
    UInfo = Fetch(forecastTime, forecastLevel, UParam, false);
    VInfo = Fetch(forecastTime, forecastLevel, VParam, false);

    if (!(prevHInfo && prevUInfo && prevVInfo && HInfo && UInfo && VInfo))
    {
        myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));
        return;
    }

    // If calculating for hybrid levels, A/B vertical coordinates must be set
    // (copied from source)
    myTargetInfo->ParamIndex(0);
    SetAB(myTargetInfo, HInfo);

    myTargetInfo->ParamIndex(1);
    SetAB(myTargetInfo, HInfo);

    string deviceType = "CPU";

    double Di = dynamic_cast<regular_grid*> (myTargetInfo->Grid())->Di();
    double Dj = dynamic_cast<regular_grid*> (myTargetInfo->Grid())->Dj();
    point firstPoint = dynamic_cast<regular_grid*> (myTargetInfo->Grid())->FirstGridPoint();

    size_t Ni = dynamic_cast<regular_grid*> (myTargetInfo->Grid())->Ni();
    size_t Nj = dynamic_cast<regular_grid*> (myTargetInfo->Grid())->Nj();

    vector<double> dx (Nj, kFloatMissing);
    vector<double> dy (Ni, kFloatMissing);

    for (size_t i=0; i < Ni; ++i)
    {
        dy[i] = util::LatitudeLength(0) * Dj / 360;
    }

    for (size_t j=0; j < Nj; ++j)
    {
        dx[j] = util::LatitudeLength(firstPoint.Y() + double(j) * Dj) * Di / 360;
    }

    pair<matrix<double>,matrix<double>> gradU = util::CentralDifference(UInfo->Data(),dx,dy);
    pair<matrix<double>,matrix<double>> gradV = util::CentralDifference(VInfo->Data(),dx,dy);

    LOCKSTEP(myTargetInfo, UInfo, VInfo, HInfo, prevUInfo, prevVInfo, prevHInfo)
    {
        size_t index = myTargetInfo->LocationIndex();
        double U = UInfo->Value();
        double V = VInfo->Value();
        double H = HInfo->Value();
        double prevU = prevUInfo->Value();
        double prevV = prevVInfo->Value();
        double prevH = prevHInfo->Value();

        if (U == kFloatMissing || V == kFloatMissing || H == kFloatMissing || prevU == kFloatMissing || prevV == kFloatMissing || prevH == kFloatMissing)
        {
            continue;
        }

        //Precalculation of wind shear, deformation and convergence
        double VWS = sqrt(pow((prevU-U)/(prevH-H),2) + pow((prevV-V)/(prevH-H),2));
        double DEF = sqrt(pow(get<0>(gradU).At(index)-get<1>(gradV).At(index),2) + pow(get<0>(gradV).At(index) + get<1>(gradU).At(index),2));
        double CVG = -get<0>(gradU).At(index)-get<1>(gradV).At(index);

        //Calculation of turbulence indices
        double TI = VWS*DEF;
        double TI2 = VWS*(DEF+CVG);

        //return result
        myTargetInfo->ParamIndex(0);
        myTargetInfo->Value(TI);

        myTargetInfo->ParamIndex(1);
        myTargetInfo->Value(TI2);

    }

    myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}
