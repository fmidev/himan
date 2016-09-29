/**
 * @file pot.cpp
 *
 *
 * @date May 25, 2015
 * @author Tack
 */

#include <boost/lexical_cast.hpp>
#include <math.h>

#include "forecast_time.h"
#include "level.h"
#include "logger_factory.h"
#include "matrix.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "pot.h"

using namespace std;
using namespace himan::plugin;

pot::pot() : itsTopLevel()
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

	const param CapeParamEC("CAPE-JKG");
	const param CapeParamHiman("CAPE1040-JKG");
	const level CapeLevelHiman(kMaximumThetaE, 0);
	const param RainParam("RRR-KGM2");
	// ----

	// switch indicating which version of cape is used
	bool cape1040 = true;

	// Step from previous leadtime, taken from configuration file
	int step = itsConfiguration->ForecastStep();

	if (step == kHPMissingInt)
	{
		// himan was mabe started with configuration option "hours"
		// so step is not readily available

		if (myTargetInfo->SizeTimes() > 1)
		{
			// More than one time is calculated - check the difference to previous
			// or next time

			int leadtime = myTargetInfo->Time().Step();
			int otherLeadtime;

			if (myTargetInfo->PreviousTime())
			{
				otherLeadtime = myTargetInfo->Time().Step();
				myTargetInfo->NextTime();  // return
			}
			else
			{
				myTargetInfo->NextTime();
				otherLeadtime = myTargetInfo->Time().Step();
				myTargetInfo->PreviousTime();  // return
			}

			step = abs(otherLeadtime - leadtime);
		}
		else
		{
			// default
			step = 1;
		}
	}

	HPTimeResolution timeResolution = myTargetInfo->Time().StepResolution();

	forecast_time forecastTime = myTargetInfo->Time();
	forecast_time forecastTimePrev = myTargetInfo->Time();
	forecastTimePrev.ValidDateTime().Adjust(timeResolution, -step);
	forecast_time forecastTimeNext = myTargetInfo->Time();
	forecastTimeNext.ValidDateTime().Adjust(timeResolution, +step);
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	auto myThreadedLogger =
	    logger_factory::Instance()->GetLog("pot_pluginThread #" + boost::lexical_cast<string>(threadIndex));

	myThreadedLogger->Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                        static_cast<string>(forecastLevel));

	info_t CAPEInfo, RRInfo, RRPrevInfo, RRNextInfo;

	CAPEInfo = Fetch(forecastTime, CapeLevelHiman, CapeParamHiman, forecastType, false);
	if (!CAPEInfo)
	{
		CAPEInfo = Fetch(forecastTime, forecastLevel, CapeParamEC, forecastType, false);
		cape1040 = false;
	}
	RRInfo = Fetch(forecastTime, forecastLevel, RainParam, forecastType, false);
	RRPrevInfo = Fetch(forecastTimePrev, forecastLevel, RainParam, forecastType, false);
	RRNextInfo = Fetch(forecastTimeNext, forecastLevel, RainParam, forecastType, false);
	if (!(CAPEInfo && RRInfo))
	{
		myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string>(forecastTime.Step()) + ", level " +
		                       static_cast<string>(forecastLevel));
		return;
	}

	// käytetään sadeparametrina mallin sateen alueellista keskiarvoa, jotta diskreettejä sadeolioita saadaan vähän
	// levitettyä ympäristöön, tässä toimisi paremmin esim. 30 km säde.
	// Filter RR
	/* This will be requested back soon
	himan::matrix<double> filter_kernel(3,3,1,kFloatMissing);
	filter_kernel.Fill(1.0/9.0);
	himan::matrix<double> filtered_RR = numerical_functions::Filter2D(RRInfo->Data(), filter_kernel);
	RRInfo->Grid()->Data(filtered_RR);
	*/

	himan::matrix<double> filter_kernel(5, 5, 1, kFloatMissing);
	filter_kernel.Fill(1.0 / 25.0);
	himan::matrix<double> filtered_CAPE = numerical_functions::Filter2D(CAPEInfo->Data(), filter_kernel);

	CAPEInfo->Grid()->Data(filtered_CAPE);

	if (!CAPEInfo || !RRInfo)
	{
		myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string>(forecastTime.Step()) + ", level " +
		                       static_cast<string>(forecastLevel));
		return;
	}

	string deviceType = "CPU";

	if (RRPrevInfo)
	{
		RRPrevInfo->ResetLocation();
	}

	if (RRNextInfo)
	{
		RRNextInfo->ResetLocation();
	}

	LOCKSTEP(myTargetInfo, CAPEInfo, RRInfo)
	{
		double RR = RRInfo->Value();
		double RRPrev = RR;
		double RRNext = RR;

		if (RRPrevInfo)
		{
			RRPrevInfo->NextLocation();
			RRPrev = RRPrevInfo->Value();
		}

		if (RRNextInfo)
		{
			RRNextInfo->NextLocation();
			RRNext = RRNextInfo->Value();
		}

		double POT;
		double CAPE_ec = CAPEInfo->Value();

		double LAT = myTargetInfo->LatLon().Y();

		double PoLift_ec = 0;
		double PoThermoDyn_ec = 0;

		// Määritetään salamointi vs. CAPE riippuvuutta varten tarvitaan CAPEn ala- ja ylärajoille yhtälöt. Ala- ja
		// ylärajat muuttuvat leveyspiirin funktiona.
		double lat_abs = abs(LAT);
		double cape_low0 = 50;
		double cape_low1 = 400;
		double cape_high0 = 600;
		double cape_high1 = 1100;

		double cape_low = kFloatMissing;
		double cape_high = kFloatMissing;

		if (!cape1040)
		{
			cape_low0 = 100;
			cape_low1 = 800;
			cape_high0 = 1200;
			cape_high1 = 2200;
		}

		if (CAPE_ec == kFloatMissing || RRPrev == kFloatMissing || RR == kFloatMissing || RRNext == kFloatMissing)
		{
			continue;
		}

		// time average rain parameter over three timesteps
		RR = (RRPrev + RR + RRNext) / 3.0;

		// Kiinnitetään cape_low ja high levespiirien 25...45  ulkopuolella vakioarvoihin.
		if (lat_abs > 30)
		{
			cape_low = cape_low0;
			cape_high = cape_high0;
		}

		if (lat_abs < 10)
		{
			cape_low = cape_low1;
			cape_high = cape_high1;
		}

		if (lat_abs <= 30 && lat_abs >= 10)
		{
			double lowk = (cape_low1 - cape_low0) / (10 - 30);
			double highk = (cape_high1 - cape_high0) / (10 - 30);
			cape_low = lowk * lat_abs + (cape_low0 - lowk * 30);
			cape_high = highk * lat_abs + (cape_high0 - highk * 30);
		}

		// CAPE-arvot skaalataan arvoihin 1....10. Ala- ja ylärajat muuttuvat leveyspiirin funktiona.
		double k = 9 / (cape_high - cape_low);
		double scaled_cape = 1;

		if (CAPE_ec >= cape_low) scaled_cape = k * CAPE_ec + (1 - k * cape_low);

		assert(scaled_cape > 0);

		// Leikataan skaalatun CAPEN arvot, jotka menevät yli 10
		if (scaled_cape > 10) scaled_cape = 10;

		// Ukkosta suosivan termodynamiikan todennäköisyys
		PoThermoDyn_ec =
		    0.4343 * log(scaled_cape);  // salamoinnin todennäköisyys kasvaa logaritmisesti CAPE-arvojen kasvaessa.

		// Laukaisevan tekijän (Lift) todennäköisyys kasvaa ennustetun sateen intensiteetin funktiona.
		// Perustelu: Konvektioparametrisointi (eli malli saa nollasta poikkeavaa sademäärää) malleissa käynnistyy
		// useimmiten alueilla, missä mallissa on konvergenssia tai liftiä tarjolla

		if (RR >= 0.05 && RR <= 5)
		{
			PoLift_ec =
			    0.217147241 * log(RR) + 0.650514998;  // Funktio kasvaa nopeasti logaritmisesti nollasta ykköseen
		}

		if (RR > 5)
		{
			PoLift_ec = 1;
		}

		// POT on ainesosiensa todennäköisyyksien tulo
		POT = PoLift_ec * PoThermoDyn_ec * 100;

		// return result
		myTargetInfo->Value(POT);
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " +
	                       boost::lexical_cast<string>(myTargetInfo->Data().MissingCount()) + "/" +
	                       boost::lexical_cast<string>(myTargetInfo->Data().Size()));
}
