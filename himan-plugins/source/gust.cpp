/**
 * @file gust.cpp
 *
 * Computes wind gusts
 *
 * @date Jul 8, 2014
 * @author Tack
 */

#include <boost/lexical_cast.hpp>

#include "gust.h"
#include "util.h"
#include "plugin_factory.h"
#include "logger_factory.h"
#include "level.h"
#include "forecast_time.h"
#include "logger_factory.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "hitool.h"
#include "fetcher.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;
using namespace himan::plugin;

gust::gust()
{
	itsLogger = logger_factory::Instance()->GetLog("gust");
}

void gust::Process(std::shared_ptr<const plugin_configuration> conf)
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

	param theRequestedParam("FFG-MS", 417, 0, 2, 22);

	// If this param is also used as a source param for other calculations
	// (like for example dewpoint, relative humidity), unit should also be
	// specified

	theRequestedParam.Unit(kMs);
	
	SetParams({theRequestedParam});

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void gust::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	/*
	 * Required source parameters
	 *
	 * eg. param PParam("P-Pa"); for pressure in pascals
	 *
	 */
	const param TPotParam("TP-K");
	const param WSParam("FF-MS");
	const param GustParam("FFG-MS");
	const param TParam("T-K");
	const param TGParam("TG-K");
	const param TopoParam("Z-M2S2");

	shared_ptr<plugin::fetcher> f = dynamic_pointer_cast <plugin::fetcher> (plugin_factory::Instance()->Plugin("fetcher"));
	auto puuskaInfo = f->Fetch(itsConfiguration,myTargetInfo->Time(),myTargetInfo->Level(),GustParam);

        level H2 = level(himan::kHeight, 2, "HEIGHT");
        level H0 = level(himan::kHeight, 0, "HEIGHT");

	auto TGInfo = f->Fetch(itsConfiguration,myTargetInfo->Time(),H2,TGParam);
	auto TopoInfo = f->Fetch(itsConfiguration,myTargetInfo->Time(),H0,TopoParam);
	// maybe need adjusting
        auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

        h->Configuration(itsConfiguration);
        h->Time(myTargetInfo->Time());

	// Potential temperature differences
	vector<double> potero0  = h->VerticalValue(TPotParam,  10);
	vector<double> potero1  = h->VerticalValue(TPotParam, 100);
	vector<double> potero2  = h->VerticalValue(TPotParam, 200);
	vector<double> potero3  = h->VerticalValue(TPotParam, 300);
	vector<double> potero4  = h->VerticalValue(TPotParam, 400);
	vector<double> potero5  = h->VerticalValue(TPotParam, 500);
	vector<double> potero6  = h->VerticalValue(TPotParam, 600);
	vector<double> potero7  = h->VerticalValue(TPotParam, 700);
	vector<double> potero8  = h->VerticalValue(TPotParam, 800);
	vector<double> potero9  = h->VerticalValue(TPotParam, 900);
	vector<double> potero10 = h->VerticalValue(TPotParam,1000);
	vector<double> potero11 = h->VerticalValue(TPotParam,1100);

	// Wind speeds
	vector<double> ws_10   = h->VerticalValue(WSParam,  10);
	vector<double> ws_100  = h->VerticalValue(WSParam, 100);
	vector<double> ws_200  = h->VerticalValue(WSParam, 200);
	vector<double> ws_300  = h->VerticalValue(WSParam, 300);
	vector<double> ws_400  = h->VerticalValue(WSParam, 400);
	vector<double> ws_500  = h->VerticalValue(WSParam, 500);
	vector<double> ws_600  = h->VerticalValue(WSParam, 600);
	vector<double> ws_700  = h->VerticalValue(WSParam, 700);
	vector<double> ws_800  = h->VerticalValue(WSParam, 800);
	vector<double> ws_900  = h->VerticalValue(WSParam, 900);
	vector<double> ws_1000 = h->VerticalValue(WSParam,1000);
	vector<double> ws_1100 = h->VerticalValue(WSParam,1100);

	// maximum windspeed 0-1200m
	vector<double> myrsky = h->VerticalMaximum(WSParam,0,1200);

	// maximum temperature 0-200m
	vector<double> maxt = h->VerticalMaximum(TParam,0,200);

	// base wind speed
	vector<double> pohja = h->VerticalAverage(WSParam,10,200);

	// average wind speed 0-60m
	vector<double> pohja_0_60 = h->VerticalAverage(WSParam,0,60);	

	// ----	

	auto myThreadedLogger = logger_factory::Instance()->GetLog("gust_pluginThread #" + boost::lexical_cast<string> (threadIndex));

	// Current time and level
	
	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	myThreadedLogger->Debug("Calculating time " + static_cast<string> (*forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	// info_t TPotInfo = Fetch(forecastTime, forecastLevel, TPotParam);

	/*
	if (!exampleInfo)
	{
		myThreadedLogger->Info("Skipping step " + boost::lexical_cast<string> (forecastTime.Step()) + ", level " + static_cast<string> (forecastLevel));

		if (itsConfiguration->StatisticsEnabled())
		{
			// When time or level is skipped, all values are missing
			itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo->Data()->Size());
			itsConfiguration->Statistics()->AddToValueCount(myTargetInfo->Data()->Size());
		}

		return;

	}
	*/

	// If calculating for hybrid levels, A/B vertical coordinates must be set
	// (copied from source)
	
	// SetAB(myTargetInfo, exampleInfo);

	vector<double> x(itsConfiguration->Info()->Grid()->Size(),-0.15);

	for (size_t i = 0; i < itsConfiguration->Info()->Grid()->Size(); ++i)
	{
		if (myrsky[i] >= 20)
		{
			x[i] = -0.20;
		}
		else continue;

		if (myrsky[i] >= 25)
		{
			x[i] = -0.25;
		}
		else continue;

		if (myrsky[i] >= 30)
		{
			x[i] = -0.30;
		}
	}
	
	vector<double> lowerHeight(itsConfiguration->Info()->Grid()->Size(),0);
	vector<double> upperHeight(itsConfiguration->Info()->Grid()->Size(),0);

	for (size_t i = 0; i < itsConfiguration->Info()->Grid()->Size(); ++i)
	{
		if (potero1[i] - potero2[i] > x[i] && ws_200[i] > ws_100[i])
		{
			lowerHeight[i] = 50;
			upperHeight[i] = 200;
		}
		else continue;

		if (potero2[i] - potero3[i] > x[i] && ws_300[i] > ws_200[i])
		{
			lowerHeight[i] = 100;
			upperHeight[i] = 300;
		}
		else continue;

		if (potero3[i] - potero4[i] > x[i] && ws_400[i] > ws_300[i])
		{
			lowerHeight[i] = 150;
			upperHeight[i] = 400;
		}
		else continue;

		if (potero4[i] - potero5[i] > x[i] && ws_500[i] > ws_400[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 500;
		}
		else continue;
	
		if (potero5[i] - potero6[i] > x[i] && ws_600[i] > ws_500[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 600;
		}
		else continue;
		
		if (potero6[i] - potero7[i] > x[i] && ws_700[i] > ws_600[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 700;
		}
		else continue;

		if (potero7[i] - potero8[i] > x[i] && ws_800[i] > ws_700[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 800;
		}
		else continue;

		if (potero8[i] - potero9[i] > x[i] && ws_900[i] > ws_800[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 900;
		}
		else continue;

		if (potero9[i] - potero10[i] > x[i] && ws_1000[i] > ws_900[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 1000;
		}
		else continue;

		if (potero10[i] - potero11[i] > x[i] && ws_1100[i] > ws_1000[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 1100;
		}
	}

	vector<double> gust = h->VerticalAverage(WSParam,lowerHeight,upperHeight);	

	for (size_t i = 0; i < itsConfiguration->Info()->Grid()->Size(); ++i)
	{
		if (potero0[i] - potero1[i] <= x[i] && ws_100[i] <= ws_10[i])
		{
			 gust[i] = 0;
		}
	}

	double a = 0.5;

	for (size_t i = 0; i < itsConfiguration->Info()->Grid()->Size(); ++i)
	{
		if (potero1[i] - potero2[i] > a && ws_200[i] > ws_100[i])
		{
			lowerHeight[i] = 50;
			upperHeight[i] = 200;
		}
		else continue;

		if (potero2[i] - potero3[i] > a && ws_300[i] > ws_200[i])
		{
			lowerHeight[i] = 100;
			upperHeight[i] = 300;
		}
		else continue;

		if (potero3[i] - potero4[i] > a && ws_400[i] > ws_300[i])
		{
			lowerHeight[i] = 150;
			upperHeight[i] = 400;
		}
		else continue;

		if (potero4[i] - potero5[i] > a && ws_500[i] > ws_400[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 500;
		}
		else continue;
	
		if (potero5[i] - potero6[i] > a && ws_600[i] > ws_500[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 600;
		}
		else continue;
		
		if (potero6[i] - potero7[i] > a && ws_700[i] > ws_600[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 700;
		}
		else continue;

		if (potero7[i] - potero8[i] > a && ws_800[i] > ws_700[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 800;
		}
		else continue;

		if (potero8[i] - potero9[i] > a && ws_900[i] > ws_800[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 900;
		}
		else continue;

		if (potero9[i] - potero10[i] > a && ws_1000[i] > ws_900[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 1000;
		}
		else continue;

		if (potero10[i] - potero11[i] > a && ws_1100[i] > ws_1000[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 1100;
		}
	}


	vector<double> maxgust = h->VerticalAverage(WSParam,lowerHeight,upperHeight);	

	for (size_t i = 0; i < itsConfiguration->Info()->Grid()->Size(); ++i)
	{
		if (potero0[i] - potero1[i] <= a && ws_100[i] <= ws_10[i])
		{
			 maxgust[i] = 0;
		}
	}

	vector<double> par466(itsConfiguration->Info()->Grid()->Size(),0);

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, puuskaInfo, TGInfo, TopoInfo)
	{
		size_t i = myTargetInfo->ParamIndex();

		double par467 = puuskaInfo->Value();
		double t_ground = TGInfo->Value();
		double topo = TopoInfo->Value()*himan::constants::kIg;
		double puuska = pohja[i];

		if (par467 == kFloatMissing)
		{
			continue;
		}

		/* Calculations go here */

		if (gust[i] > pohja[i])
		{
			puuska = gust[i];
		}

		if ((maxgust[i] - gust[i]) > 8)
		{
			puuska = (maxgust[i] + pohja[i])/2;
		}

		if ((maxt[i] - t_ground) > 1 && topo > 15)
		{
			puuska = pohja_0_60[i];
			par466[i] = ws_10[i];
			
			if (maxt[i] - t_ground > 2)
			{
				puuska = ws_10[i];
				par466[i] = ws_10[i] * 0.7;
				
				if (maxt[i] - t_ground > 4)
				{
					puuska = ws_10[i] * 0.7;
					par466[i] = ws_10[i] * 0.4;
				}
			}
		}

		myTargetInfo->Value(puuska);

	}

	himan::matrix<double> filter_kernel(3,3,1);
	filter_kernel.Fill(1.0/9.0);
	himan::matrix<double> puuska_filtered = util::Filter2D(*myTargetInfo->Data(), filter_kernel);

	auto puuska_filtered_ptr = make_shared<himan::matrix<double>> (puuska_filtered);
	myTargetInfo->Grid()->Data(puuska_filtered_ptr);

        LOCKSTEP(myTargetInfo)
        {
		size_t i = myTargetInfo->ParamIndex();
		if( par466[i]*1.12 > myTargetInfo->Value())
		{
			myTargetInfo->Value(par466[i]*1.15);
		}
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data()->MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data()->Size()));

}
