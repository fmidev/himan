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
#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;
using namespace himan::plugin;

struct potero
{
	vector<double> potero0;
	vector<double> potero1;
	vector<double> potero2;
	vector<double> potero3;
	vector<double> potero4;
	vector<double> potero5;
	vector<double> potero6;
	vector<double> potero7;
	vector<double> potero8;
	vector<double> potero9;
	vector<double> potero10;
	vector<double> potero11;

	potero() {}
};

void Potero(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, potero& poterot, bool& succeeded);

gust::gust()
{
	itsLogger = logger_factory::Instance()->GetLog("gust");
}

void gust::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param theRequestedParam("FFG2-MS", 417, 0, 2, 22);

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

	auto myThreadedLogger = logger_factory::Instance()->GetLog("gust_pluginThread #" + boost::lexical_cast<string> (threadIndex));

	/*
	 * Required source parameters
	 *
	 */

	const param WSParam("FF-MS");
	const param GustParam("FFG-MS");
	const param TParam("T-K");
	const param T_LowestLevelParam("T-K");
	const param TopoParam("Z-M2S2");

	level H0, H10, Ground;

	producer prod = itsConfiguration->SourceProducer(0);

	auto n = dynamic_pointer_cast <plugin::neons> (plugin_factory::Instance()->Plugin("neons"));
	long lowestHybridLevelNumber = boost::lexical_cast<long> (n->ProducerMetaData(prod.Id(), "last hybrid level number"));
	level lowestHybridLevel(kHybrid,lowestHybridLevelNumber);

	if (myTargetInfo->Producer().Id() == 240)
	{
		H0 = level(kGround, 0);
		H10 = H0;
		Ground = H0;
	}
	else if (myTargetInfo->Producer().Id() == 210)
	{
		H0 = level(kHeight, 0);
		H10 = level(kHeight, 10);
		Ground = H0;
	}
	else
	{
		Ground = level(kGround, 0);
		H0 = level(kHeight, 0);
		H10 = level(kHeight, 10);
	}
	
	info_t puuskaInfo, T_LowestLevelInfo, TopoInfo;

	// Current time and level

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	puuskaInfo = Fetch(forecastTime,H10,GustParam,false);
	T_LowestLevelInfo = Fetch(forecastTime,lowestHybridLevel,T_LowestLevelParam,false);
	TopoInfo = Fetch(forecastTime,H0,TopoParam,false);

	if (!puuskaInfo || !T_LowestLevelInfo || !TopoInfo)
	{
		itsLogger->Error("Unable to find all source data");
		return;
	}
	
	// maybe need adjusting
	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));

	h->Configuration(itsConfiguration);
	h->Time(forecastTime);

	potero poterot;

	bool succeeded = false;

	boost::thread t(&Potero, itsConfiguration, forecastTime, boost::ref(poterot), boost::ref(succeeded));

	vector<double> ws_10, ws_100, ws_200, ws_300, ws_400, ws_500, ws_600, ws_700, ws_800, ws_900, ws_1000, ws_1100;

	try
	{
		// Wind speeds
		ws_10   = h->VerticalValue(WSParam,  10);
		ws_100  = h->VerticalValue(WSParam, 100);
		ws_200  = h->VerticalValue(WSParam, 200);
		ws_300  = h->VerticalValue(WSParam, 300);
		ws_400  = h->VerticalValue(WSParam, 400);
		ws_500  = h->VerticalValue(WSParam, 500);
		ws_600  = h->VerticalValue(WSParam, 600);
		ws_700  = h->VerticalValue(WSParam, 700);
		ws_800  = h->VerticalValue(WSParam, 800);
		ws_900  = h->VerticalValue(WSParam, 900);
		ws_1000 = h->VerticalValue(WSParam,1000);
		ws_1100 = h->VerticalValue(WSParam,1100);
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("WindSpeed caught exception " + boost::lexical_cast<string> (e));
		}
		else
		{
			myThreadedLogger->Error("hitool calculation for windspeed failed, unable to proceed");
			return;
		}
	}

	t.join();

	if (!succeeded)
	{
		myThreadedLogger->Error("hitool calculation for potero failed, unable to proceed");
		return;
	}

	vector<double> myrsky, maxt, pohja, pohja_0_60;
	try
	{
		// maximum windspeed 0-1200m
		myrsky = h->VerticalMaximum(WSParam,0,1200);

		// maximum temperature 0-200m
		maxt = h->VerticalMaximum(TParam,0,200);

		// base wind speed
		pohja = h->VerticalAverage(WSParam,10,200);

		// average wind speed 0-60m
		pohja_0_60 = h->VerticalAverage(WSParam,0,60);	
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Caught exception " + boost::lexical_cast<string> (e));
		}
		else
		{
			myThreadedLogger->Error("hitool was unable to find data");
			return;
		}
	}

	myThreadedLogger->Debug("Calculating time " + static_cast<string> (forecastTime.ValidDateTime()) + " level " + static_cast<string> (forecastLevel));

	vector<double> x(itsConfiguration->Info()->Grid()->Size(),-0.15);

	for (size_t i = 0; i < itsConfiguration->Info()->Grid()->Size(); ++i)
	{
		double val = myrsky[i];
		
		if (val == kFloatMissing)
		{
			continue;
		}

		if (val >= 20)
		{
			x[i] = -0.20;
		}
		else continue;

		if (val >= 25)
		{
			x[i] = -0.25;
		}
		else continue;

		if (val >= 30)
		{
			x[i] = -0.30;
		}
	}
	
	vector<double> lowerHeight(itsConfiguration->Info()->Grid()->Size(),0);
	vector<double> upperHeight(itsConfiguration->Info()->Grid()->Size(),0);

	for (size_t i = 0; i < itsConfiguration->Info()->Grid()->Size(); ++i)
	{
		if (x[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero1[i] == kFloatMissing || poterot.potero2[i] == kFloatMissing || ws_100[i] == kFloatMissing || ws_200[i] == kFloatMissing)
		{
			continue;
		}
					
		if (poterot.potero1[i] - poterot.potero2[i] > x[i] && ws_200[i] > ws_100[i])
		{
			lowerHeight[i] = 50;
			upperHeight[i] = 200;
		}
		else continue;

		if (poterot.potero2[i] == kFloatMissing || poterot.potero3[i] == kFloatMissing || ws_200[i] == kFloatMissing || ws_300[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero2[i] - poterot.potero3[i] > x[i] && ws_300[i] > ws_200[i])
		{
			lowerHeight[i] = 100;
			upperHeight[i] = 300;
		}
		else continue;

		if (poterot.potero3[i] == kFloatMissing || poterot.potero4[i] == kFloatMissing || ws_300[i] == kFloatMissing || ws_400[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero3[i] - poterot.potero4[i] > x[i] && ws_400[i] > ws_300[i])
		{
			lowerHeight[i] = 150;
			upperHeight[i] = 400;
		}
		else continue;

		if (poterot.potero4[i] == kFloatMissing || poterot.potero5[i] == kFloatMissing || ws_400[i] == kFloatMissing || ws_500[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero4[i] - poterot.potero5[i] > x[i] && ws_500[i] > ws_400[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 500;
		}
		else continue;
	
		if (poterot.potero5[i] == kFloatMissing || poterot.potero6[i] == kFloatMissing || ws_500[i] == kFloatMissing || ws_600[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero5[i] - poterot.potero6[i] > x[i] && ws_600[i] > ws_500[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 600;
		}
		else continue;
		
		if (poterot.potero6[i] == kFloatMissing || poterot.potero7[i] == kFloatMissing || ws_600[i] == kFloatMissing || ws_700[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero6[i] - poterot.potero7[i] > x[i] && ws_700[i] > ws_600[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 700;
		}
		else continue;

		if (poterot.potero7[i] == kFloatMissing || poterot.potero8[i] == kFloatMissing || ws_700[i] == kFloatMissing || ws_800[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero7[i] - poterot.potero8[i] > x[i] && ws_800[i] > ws_700[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 800;
		}
		else continue;

		if (poterot.potero8[i] == kFloatMissing || poterot.potero9[i] == kFloatMissing || ws_800[i] == kFloatMissing || ws_900[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero8[i] - poterot.potero9[i] > x[i] && ws_900[i] > ws_800[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 900;
		}
		else continue;

		if (poterot.potero9[i] == kFloatMissing || poterot.potero10[i] == kFloatMissing || ws_900[i] == kFloatMissing || ws_1000[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero9[i] - poterot.potero10[i] > x[i] && ws_1000[i] > ws_900[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 1000;
		}
		else continue;

		if (poterot.potero10[i] == kFloatMissing || poterot.potero11[i] == kFloatMissing || ws_1000[i] == kFloatMissing || ws_1100[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero10[i] - poterot.potero11[i] > x[i] && ws_1100[i] > ws_1000[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 1100;
		}
	}

	vector<double> gust;
	
	try
	{
		gust = h->VerticalAverage(WSParam,lowerHeight,upperHeight);	
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Caught exception " + boost::lexical_cast<string> (e));
		}
		else
		{
			myThreadedLogger->Error("hitool was unable to find data");
			return;
		}
	}
	
	for (size_t i = 0; i < itsConfiguration->Info()->Grid()->Size(); ++i)
	{
		if (poterot.potero0[i] == kFloatMissing || poterot.potero1[i] == kFloatMissing || x[i] == kFloatMissing || ws_100[i] == kFloatMissing || ws_10[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero0[i] - poterot.potero1[i] <= x[i] && ws_100[i] <= ws_10[i])
		{
			 gust[i] = 0;
		}
	}

	const double a = 0.5;

	for (size_t i = 0; i < itsConfiguration->Info()->Grid()->Size(); ++i)
	{
		if (poterot.potero1[i] == kFloatMissing || poterot.potero2[i] == kFloatMissing || ws_100[i] == kFloatMissing || ws_200[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero1[i] - poterot.potero2[i] > a && ws_200[i] > ws_100[i])
		{
			lowerHeight[i] = 50;
			upperHeight[i] = 200;
		}
		else continue;

		if (poterot.potero2[i] == kFloatMissing || poterot.potero3[i] == kFloatMissing || ws_200[i] == kFloatMissing || ws_300[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero2[i] - poterot.potero3[i] > a && ws_300[i] > ws_200[i])
		{
			lowerHeight[i] = 100;
			upperHeight[i] = 300;
		}
		else continue;

		if (poterot.potero3[i] == kFloatMissing || poterot.potero4[i] == kFloatMissing || ws_300[i] == kFloatMissing || ws_400[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero3[i] - poterot.potero4[i] > a && ws_400[i] > ws_300[i])
		{
			lowerHeight[i] = 150;
			upperHeight[i] = 400;
		}
		else continue;

		if (poterot.potero4[i] == kFloatMissing || poterot.potero5[i] == kFloatMissing || ws_400[i] == kFloatMissing || ws_500[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero4[i] - poterot.potero5[i] > a && ws_500[i] > ws_400[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 500;
		}
		else continue;
	
		if (poterot.potero5[i] == kFloatMissing || poterot.potero6[i] == kFloatMissing || ws_500[i] == kFloatMissing || ws_600[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero5[i] - poterot.potero6[i] > a && ws_600[i] > ws_500[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 600;
		}
		else continue;
		
		if (poterot.potero6[i] == kFloatMissing || poterot.potero7[i] == kFloatMissing || ws_600[i] == kFloatMissing || ws_700[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero6[i] - poterot.potero7[i] > a && ws_700[i] > ws_600[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 700;
		}
		else continue;

		if (poterot.potero7[i] == kFloatMissing || poterot.potero8[i] == kFloatMissing || ws_700[i] == kFloatMissing || ws_800[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero7[i] - poterot.potero8[i] > a && ws_800[i] > ws_700[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 800;
		}
		else continue;

		if (poterot.potero8[i] == kFloatMissing || poterot.potero9[i] == kFloatMissing || ws_800[i] == kFloatMissing || ws_900[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero8[i] - poterot.potero9[i] > a && ws_900[i] > ws_800[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 900;
		}
		else continue;

		if (poterot.potero9[i] == kFloatMissing || poterot.potero10[i] == kFloatMissing || ws_900[i] == kFloatMissing || ws_1000[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero9[i] - poterot.potero10[i] > a && ws_1000[i] > ws_900[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 1000;
		}
		else continue;

		if (poterot.potero10[i] == kFloatMissing || poterot.potero11[i] == kFloatMissing || ws_1000[i] == kFloatMissing || ws_1100[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero10[i] - poterot.potero11[i] > a && ws_1100[i] > ws_1000[i])
		{
			lowerHeight[i] = 200;
			upperHeight[i] = 1100;
		}
	}

	vector<double> maxgust;
	
	try
	{
		maxgust = h->VerticalAverage(WSParam,lowerHeight,upperHeight);	
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Caught exception " + boost::lexical_cast<string> (e));
		}
		else
		{
			myThreadedLogger->Error("hitool was unable to find data");
			return;
		}
	}
	
	for (size_t i = 0; i < itsConfiguration->Info()->Grid()->Size(); ++i)
	{
		if (poterot.potero0[i] == kFloatMissing || poterot.potero1[i] == kFloatMissing || ws_100[i] == kFloatMissing || ws_10[i] == kFloatMissing)
		{
			continue;
		}
		
		if (poterot.potero0[i] - poterot.potero1[i] <= a && ws_100[i] <= ws_10[i])
		{
			 maxgust[i] = 0;
		}
	}

	vector<double> par466(itsConfiguration->Info()->Grid()->Size(),0);

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, puuskaInfo, T_LowestLevelInfo, TopoInfo)
	{
		size_t i = myTargetInfo->LocationIndex();

		double par467 = puuskaInfo->Value();
		double t_LowestLevel = T_LowestLevelInfo->Value();
		double topo = TopoInfo->Value();
		double puuska = pohja[i];
		double gustval = gust[i];
		double maxgustval = maxgust[i];
		double ws10 = ws_10[i];

		if (par467 == kFloatMissing 
			|| t_LowestLevel == kFloatMissing 
			|| topo == kFloatMissing 
			|| puuska == kFloatMissing 
			|| gustval == kFloatMissing 
			|| maxgustval == kFloatMissing
			|| ws10 == kFloatMissing 
			|| pohja_0_60[i] == kFloatMissing)
		{
			continue;
		}

		topo *= himan::constants::kIg;
		
		/* Calculations go here */

		puuska = fmax(gustval, puuska);
		
		if ((maxgustval - gustval) > 8)
		{
			puuska = (maxgustval + pohja[i])/2;
		}

		if ((maxt[i] - t_LowestLevel) > 1 && topo > 15)
		{
			puuska = pohja_0_60[i];
			par466[i] = ws10;
			
			if (maxt[i] - t_LowestLevel > 2)
			{
				puuska = ws10;
				par466[i] = ws10 * 0.7;
				
				if (maxt[i] - t_LowestLevel > 4)
				{
					puuska = ws10 * 0.7;
					par466[i] = ws10 * 0.4;
				}
			}
		}

		myTargetInfo->Value(puuska);

	}

	himan::matrix<double> filter_kernel(3,3,1);
	filter_kernel.Fill(1.0/9.0);
	himan::matrix<double> puuska_filtered = util::Filter2D(myTargetInfo->Data(), filter_kernel);

	
	//auto puuska_filtered_ptr = make_shared<himan::matrix<double>> (puuska_filtered);
	puuska_filtered.MissingValue(kFloatMissing);
	
	myTargetInfo->Grid()->Data(puuska_filtered);
	
	LOCKSTEP(myTargetInfo)
	{
		size_t i = myTargetInfo->LocationIndex();
		
		if (par466[i] == kFloatMissing || myTargetInfo->Value() == kFloatMissing)
		{
			continue;
		}
		
		if( par466[i]*1.12 > myTargetInfo->Value())
		{
			myTargetInfo->Value(par466[i]*1.15);
		}
	}

	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}

void Potero(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, potero& poterot, bool& succeeded)
{
	auto h = dynamic_pointer_cast <hitool> (plugin_factory::Instance()->Plugin("hitool"));
	
	h->Configuration(conf);
	h->Time(ftime);

	const param TPotParam("TP-K");

	try
	{
		// Potential temperature differences
		poterot.potero0  = h->VerticalValue(TPotParam,  10);
		poterot.potero1  = h->VerticalValue(TPotParam, 100);
		poterot.potero2  = h->VerticalValue(TPotParam, 200);
		poterot.potero3  = h->VerticalValue(TPotParam, 300);
		poterot.potero4  = h->VerticalValue(TPotParam, 400);
		poterot.potero5  = h->VerticalValue(TPotParam, 500);
		poterot.potero6  = h->VerticalValue(TPotParam, 600);
		poterot.potero7  = h->VerticalValue(TPotParam, 700);
		poterot.potero8  = h->VerticalValue(TPotParam, 800);
		poterot.potero9  = h->VerticalValue(TPotParam, 900);
		poterot.potero10 = h->VerticalValue(TPotParam,1000);
		poterot.potero11 = h->VerticalValue(TPotParam,1100);
		succeeded = true;
	}
	catch (const HPExceptionType& e)
	{
		if (e != kFileDataNotFound)
		{
			throw runtime_error("Potero() caught exception " + boost::lexical_cast<string> (e));
		}

		succeeded = false;
	}

}
