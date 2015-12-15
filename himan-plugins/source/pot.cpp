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
#include "matrix.h"


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
    param POT("POT-PRCNT", 12100, 0, 19, 2);
    // If this param is also used as a source param for other calculations
    // (like for example dewpoint, relative humidity), unit should also be
    // specified

    POT.Unit(kPrcnt);

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

    // käytetään sadeparametrina mallin sateen alueellista keskiarvoa, jotta diskreettejä sadeolioita saadaan vähän levitettyä ympäristöön, tässä toimisi paremmin esim. 30 km säde.
    // Filter RR
    himan::matrix<double> filter_kernel(3,3,1,kFloatMissing);
    filter_kernel.Fill(1.0/9.0);
    himan::matrix<double> filtered_RR = util::Filter2D(RRInfo->Data(), filter_kernel);
    RRInfo->Grid()->Data(filtered_RR);
    
    string deviceType = "CPU";

    LOCKSTEP(myTargetInfo, CAPEInfo, RRInfo)
    {
	double POT;
        double CAPE_ec = CAPEInfo->Value();
        double RR = RRInfo->Value();
        double LAT = myTargetInfo->LatLon().Y();

	double PoLift_ec = 0;
	double PoThermoDyn_ec = 0;

	// Määritetään salamointi vs. CAPE riippuvuutta varten tarvitaan CAPEn ala- ja ylärajoille yhtälöt. Ala- ja ylärajat muuttuvat leveyspiirin funktiona.
	double lat_abs = abs(LAT);
	double cape_low = -25*lat_abs + 1225;
	double cape_high = -40*lat_abs + 3250;

	// Kiinnitetään cape_low ja high levespiirien 25...45  ulkopuolella vakioarvoihin.
	if (lat_abs <25)
	{
		cape_low = 600;
		cape_high = 2000;
	}

	if (lat_abs > 45)
	{
		cape_low = 100;
		cape_high = 1000;
	}

	// CAPE-arvot skaalataan arvoihin 1....10. Ala- ja ylärajat muuttuvat leveyspiirin funktiona.
	double k =  9/(cape_high - cape_low);
	double scaled_cape = 1;
        if (CAPE_ec >= cape_low) scaled_cape = k*CAPE_ec + (1- k*cape_low);

	assert( scaled_cape > 0);

	// Leikataan skaalatun CAPEN arvot, jotka menevät yli 10
	if (scaled_cape >10) scaled_cape = 10;

	//Ukkosta suosivan termodynamiikan todennäköisyys
	PoThermoDyn_ec = 0.4343 * log(scaled_cape);      //salamoinnin todennäköisyys kasvaa logaritmisesti CAPE-arvojen kasvaessa.

	// Laukaisevan tekijän (Lift) todennäköisyys kasvaa ennustetun sateen intensiteetin funktiona.
	// Perustelu: Konvektioparametrisointi (eli malli saa nollasta poikkeavaa sademäärää) malleissa käynnistyy useimmiten alueilla, missä mallissa on konvergenssia tai liftiä tarjolla

	if (RR >= 0.05 && RR <= 5)           
	{
		PoLift_ec  = 0.217147241 * log(RR) + 0.650514998;  // Funktio kasvaa nopeasti logaritmisesti nollasta ykköseen
	}

	if (RR > 5)
	{
		PoLift_ec  = 1;
	}

	//POT on ainesosiensa todennäköisyyksien tulo
	POT = PoLift_ec * PoThermoDyn_ec * 100;

        //return result
        myTargetInfo->Value(POT);
    }

    myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}
