/**
 * @file gust.cpp
 *
 * Computes wind gusts
 *
 * @date Feb 15, 2016
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
#include <boost/thread.hpp>

#define HIMAN_AUXILIARY_INCLUDE

#include "hitool.h"
#include "neons.h"
#include "radon.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace std;
using namespace himan;
using namespace himan::plugin;

struct deltaT
{
	vector<double> deltaT_100;
	vector<double> deltaT_200;
	vector<double> deltaT_300;
	vector<double> deltaT_400;
	vector<double> deltaT_500;
	vector<double> deltaT_600;
	vector<double> deltaT_700;

	deltaT() {}
};

struct deltaTot
{
        vector<double> deltaTot_100;
        vector<double> deltaTot_200;
        vector<double> deltaTot_300;
        vector<double> deltaTot_400;
        vector<double> deltaTot_500;
        vector<double> deltaTot_600;
        vector<double> deltaTot_700;

	deltaTot() {}
};

struct intT
{
	vector<double> intT_100;
        vector<double> intT_200;
        vector<double> intT_300;
        vector<double> intT_400;
        vector<double> intT_500;
        vector<double> intT_600;
        vector<double> intT_700;

	intT() {}
};

struct intTot
{
        vector<double> intTot_100;
        vector<double> intTot_200;
        vector<double> intTot_300;
        vector<double> intTot_400;
        vector<double> intTot_500;
        vector<double> intTot_600;
        vector<double> intTot_700;

	intTot() {}
};

void DeltaT(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, deltaT& dT, bool& succeeded);
void DeltaTot(deltaTot& dTot, info_t T_lowestLevel, size_t gridSize);
void IntT(intT& iT, const deltaT& dT, size_t gridSize);
void IntTot(intTot& iTot, const deltaTot& dTot, size_t gridSize);

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
 
 * This function does the actual calculation.
 */

void gust::Calculate(shared_ptr<info> myTargetInfo, unsigned short threadIndex)
{

	auto myThreadedLogger = logger_factory::Instance()->GetLog("gust_pluginThread #" + boost::lexical_cast<string> (threadIndex));

	/*
	 * Required source parameters
	 *
	 */
        const size_t gridSize = myTargetInfo->Grid()->Size();

	const param BLHParam("MIXHGT-M");         // boundary layer height
	const param WSParam("FF-MS");             // wind speed
	const param GustParam("FFG-MS");          // wind gust
	const param TParam("T-K");                // temperature
	const param T_LowestLevelParam("T-K");    // temperature at lowest level
        const param TopoParam("Z-M2S2");          // geopotential height

	level H0, H10, Ground;

	producer prod = itsConfiguration->SourceProducer(0);

	HPDatabaseType dbtype = itsConfiguration->DatabaseType();
		
	long lowestHybridLevelNumber = kHPMissingInt;
	
	if (dbtype == kNeons || dbtype == kNeonsAndRadon)
	{
		auto n = GET_PLUGIN(neons);

		lowestHybridLevelNumber = boost::lexical_cast<long> (n->ProducerMetaData(prod.Id(), "last hybrid level number"));
	}

	if ((dbtype == kRadon || dbtype == kNeonsAndRadon) && lowestHybridLevelNumber == kHPMissingInt)
	{
		auto r = GET_PLUGIN(radon);

		lowestHybridLevelNumber = boost::lexical_cast<long> (r->ProducerMetaData(prod.Id(), "last hybrid level number"));
	}

	assert(lowestHybridLevelNumber != kHPMissingInt);
	
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
	
	info_t GustInfo, T_LowestLevelInfo, BLHInfo, TopoInfo;

	// Current time and level

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	GustInfo = Fetch(forecastTime,H10,GustParam,forecastType,false);
	T_LowestLevelInfo = Fetch(forecastTime,lowestHybridLevel,T_LowestLevelParam,forecastType,false);
        BLHInfo = Fetch(forecastTime,H0,BLHParam,forecastType,false);
	TopoInfo = Fetch(forecastTime,H0,TopoParam,forecastType,false);

	if (!GustInfo || !T_LowestLevelInfo)
	{
		itsLogger->Error("Unable to find all source data");
		return;
	}

        deltaT dT;

        bool succeeded = false;

        boost::thread t(&DeltaT, itsConfiguration, forecastTime, boost::ref(dT), boost::ref(succeeded));

        // calc boundary layer height
        vector<double> z_boundaryl(gridSize,0);
        vector<double> z_half_boundaryl(gridSize,0);
        vector<double> z_zero (gridSize,0);
        for(size_t i=0; i<gridSize; ++i)	
        {
                if (BLHInfo->Data()[i] >=200)
                {
                        z_boundaryl[i] = 0.625 * BLHInfo->Data()[i] + 75;

                        if (BLHInfo->Data()[i] >1000)
                        {
                                z_boundaryl[i] = 700;
                        }

                        z_half_boundaryl[i] = z_boundaryl[i]/2;
                }
        }

        // maybe need adjusting
        auto h = GET_PLUGIN(hitool);
        
        h->Configuration(itsConfiguration);
        h->Time(forecastTime);
        
        vector<double> maxWind;
        try
        {
                maxWind = h->VerticalMaximum(WSParam, z_zero, z_boundaryl);
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
			t.join();
                        return;
                }
        }

        try
	{
		z_boundaryl = h->VerticalHeight(WSParam, z_zero, z_boundaryl, maxWind);
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
			t.join();
                        return;
                }
        }


        vector<double> BLtop_ws;
        try
        {
                BLtop_ws = h->VerticalAverage(WSParam, z_half_boundaryl, z_boundaryl);
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
			t.join();
                        return;
                }
        }

        vector<double> baseGust;
        try
        {
                baseGust = h->VerticalAverage(WSParam, 10, 170);
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
			t.join();
                        return;
                }
        }

        vector<double> t_diff;
        try
        {
                t_diff = h->VerticalAverage(TParam, 0, 200);
                for(size_t i=0; i<gridSize; ++i)
                {
                        t_diff[i] = t_diff[i] - T_LowestLevelInfo->Data()[i];
                }
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
			t.join();
                        return;
                }
        }

        vector<double> BLbottom_ws;
        try
        {
                BLbottom_ws = h->VerticalAverage(WSParam, z_zero, z_half_boundaryl);
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
			t.join();
                        return;
                }
        }

        t.join();

        deltaTot dTot;
        intT iT;
        intTot iTot;

        DeltaTot(dTot, T_LowestLevelInfo, gridSize);
        IntT(iT, dT, gridSize);
        IntTot(iTot, dTot, gridSize);

        string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, GustInfo, T_LowestLevelInfo, TopoInfo, BLHInfo)
	{
		size_t i = myTargetInfo->LocationIndex();

                double topo = TopoInfo->Value();
                double esto = kFloatMissing;
                double esto_tot = kFloatMissing;
                double gust = kFloatMissing;

		topo *= himan::constants::kIg;
		
		/* Calculations go here */

                if (z_boundaryl[i] >= 200)
                {
                        esto = iT.intT_100[i] + iT.intT_200[i];
                        esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i];
                }
                else if (z_boundaryl[i] >= 300)
                {
                        esto = iT.intT_100[i] + iT.intT_200[i] + iT.intT_300[i];
                        esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i] + iTot.intTot_300[i];
                }
                else if(z_boundaryl[i] >= 400)
                {
                        esto = iT.intT_100[i] + iT.intT_200[i] + iT.intT_300[i] + iT.intT_400[i];
                        esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i] + iTot.intTot_300[i] + iTot.intTot_400[i];
                }
                else if(z_boundaryl[i] >= 500)
                {
                        esto = iT.intT_100[i] + iT.intT_200[i] + iT.intT_300[i] + iT.intT_400[i] + iT.intT_500[i];
                        esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i] + iTot.intTot_300[i] + iTot.intTot_400[i] + iTot.intTot_500[i];
                }
                else if(z_boundaryl[i] >= 600)
                {
                        esto = iT.intT_100[i] + iT.intT_200[i] + iT.intT_300[i] + iT.intT_400[i] + iT.intT_500[i] + iT.intT_600[i];
                        esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i] + iTot.intTot_300[i] + iTot.intTot_400[i] + iTot.intTot_500[i] + iTot.intTot_600[i];
                }
                else if(z_boundaryl[i] == 700)
                {
                        esto = iT.intT_100[i] + iT.intT_200[i] + iT.intT_300[i] + iT.intT_400[i] + iT.intT_500[i] + iT.intT_600[i] + iT.intT_700[i];
                        esto_tot = iTot.intTot_100[i] + iTot.intTot_200[i] + iTot.intTot_300[i] + iTot.intTot_400[i] + iTot.intTot_500[i] + iTot.intTot_600[i] + iTot.intTot_700[i];
                }


                if(z_boundaryl[i] >= 200 && esto >= 0 && esto <= esto_tot)
                {
                        gust = ((baseGust[i] - BLtop_ws[i])/esto_tot)*esto + BLtop_ws[i];
                }

                if(z_boundaryl[i] >= 200 && esto <= 0 && esto >= -150)
                {
                        gust = ((BLtop_ws[i] - maxWind[i])/150)*esto + BLtop_ws[i];
                }

                if(z_boundaryl[i] >= 200 && esto < -150)
                {
                        gust = maxWind[i];
                }

                if(z_boundaryl[i] >= 200 && esto > esto_tot)
                {
                        gust = baseGust[i];
                }
 
                if(z_boundaryl[i] < 200)
                {
                        gust = BLbottom_ws[i];
                }

                if(gust == kFloatMissing || gust < 1)
                {
                        gust = 1;
                }

                if(topo > 15 && t_diff[i] > 0 && t_diff[i] <=4 && baseGust[i] < 8)
                {
                        gust = ((1-baseGust[i])/4) * t_diff[i] + baseGust[i];
                }

                if(topo > 15 && t_diff[i] > 4 && baseGust[i] < 8)
                {
                        gust = 1;
                }

                if(topo > 400 || T_LowestLevelInfo->Value() == kFloatMissing || BLHInfo->Value() == kFloatMissing)
                {
                        gust = GustInfo->Value()*0.95;
                }

		myTargetInfo->Value(gust);

	}

	himan::matrix<double> filter_kernel(3,3,1,kFloatMissing);
	filter_kernel.Fill(1.0/9.0);
	himan::matrix<double> gust_filtered = util::Filter2D(myTargetInfo->Data(), filter_kernel);

	myTargetInfo->Grid()->Data(gust_filtered);
	
	myThreadedLogger->Info("[" + deviceType + "] Missing values: " + boost::lexical_cast<string> (myTargetInfo->Data().MissingCount()) + "/" + boost::lexical_cast<string> (myTargetInfo->Data().Size()));

}

void DeltaT(shared_ptr<const plugin_configuration> conf, const forecast_time& ftime, deltaT& dT, bool& succeeded)
{
        auto h = GET_PLUGIN(hitool);

        h->Configuration(conf);
        h->Time(ftime);

        const param TParam("T-K");

        try
        {
                // Potential temperature differences
                dT.deltaT_100  = h->VerticalValue(TParam, 100);
                dT.deltaT_200  = h->VerticalValue(TParam, 200);
                dT.deltaT_300  = h->VerticalValue(TParam, 300);
                dT.deltaT_400  = h->VerticalValue(TParam, 400);
                dT.deltaT_500  = h->VerticalValue(TParam, 500);
                dT.deltaT_600  = h->VerticalValue(TParam, 600);
                dT.deltaT_700  = h->VerticalValue(TParam, 700);
                succeeded = true;
        }
        catch (const HPExceptionType& e)
        {
                if (e != kFileDataNotFound)
                {
                        throw runtime_error("DeltaT() caught exception " + boost::lexical_cast<string> (e));
                }

                succeeded = false;
        }
	
}

void DeltaTot(deltaTot& dTot, info_t T_lowestLevel, size_t gridSize)
{
        dTot.deltaTot_100 = vector<double> (gridSize,0);
        dTot.deltaTot_200 = vector<double> (gridSize,0);
        dTot.deltaTot_300 = vector<double> (gridSize,0);
        dTot.deltaTot_400 = vector<double> (gridSize,0);
        dTot.deltaTot_500 = vector<double> (gridSize,0);
        dTot.deltaTot_600 = vector<double> (gridSize,0);
        dTot.deltaTot_700 = vector<double> (gridSize,0);

        for (size_t i=0; i<gridSize; ++i)
        {
                dTot.deltaTot_100[i]  = (T_lowestLevel->Data()[i] + 5*(0.010 - (100/1000))) - (T_lowestLevel->Data()[i] + 9.8*(0.010 - (100/1000)));
                dTot.deltaTot_200[i]  = (T_lowestLevel->Data()[i] + 5*(0.010 - (200/1000))) - (T_lowestLevel->Data()[i] + 9.8*(0.010 - (200/1000)));
                dTot.deltaTot_300[i]  = (T_lowestLevel->Data()[i] + 5*(0.010 - (300/1000))) - (T_lowestLevel->Data()[i] + 9.8*(0.010 - (300/1000)));
                dTot.deltaTot_400[i]  = (T_lowestLevel->Data()[i] + 5*(0.010 - (400/1000))) - (T_lowestLevel->Data()[i] + 9.8*(0.010 - (400/1000)));
                dTot.deltaTot_500[i]  = (T_lowestLevel->Data()[i] + 5*(0.010 - (500/1000))) - (T_lowestLevel->Data()[i] + 9.8*(0.010 - (500/1000)));
                dTot.deltaTot_600[i]  = (T_lowestLevel->Data()[i] + 5*(0.010 - (600/1000))) - (T_lowestLevel->Data()[i] + 9.8*(0.010 - (600/1000)));
                dTot.deltaTot_700[i]  = (T_lowestLevel->Data()[i] + 5*(0.010 - (700/1000))) - (T_lowestLevel->Data()[i] + 9.8*(0.010 - (700/1000)));
        }
}

void IntT(intT& iT, const deltaT& dT, size_t gridSize)
{
        iT.intT_100 = vector<double> (gridSize,0);
        iT.intT_200 = vector<double> (gridSize,0);
        iT.intT_300 = vector<double> (gridSize,0);
        iT.intT_400 = vector<double> (gridSize,0);
        iT.intT_500 = vector<double> (gridSize,0);
        iT.intT_600 = vector<double> (gridSize,0);
        iT.intT_700 = vector<double> (gridSize,0);


        for (size_t i=0; i<gridSize; ++i)
      	{
                iT.intT_100[i] = 0.5*dT.deltaT_100[i] * 100;
                iT.intT_200[i] = 0.5*(dT.deltaT_200[i] + dT.deltaT_100[i])*100;
                iT.intT_300[i] = 0.5*(dT.deltaT_300[i] + dT.deltaT_200[i])*100;
                iT.intT_400[i] = 0.5*(dT.deltaT_400[i] + dT.deltaT_300[i])*100;
                iT.intT_500[i] = 0.5*(dT.deltaT_500[i] + dT.deltaT_400[i])*100;
                iT.intT_600[i] = 0.5*(dT.deltaT_600[i] + dT.deltaT_500[i])*100;
                iT.intT_700[i] = 0.5*(dT.deltaT_700[i] + dT.deltaT_600[i])*100;
        }
}

void IntTot(intTot& iTot, const deltaTot& dTot, size_t gridSize)
{
        iTot.intTot_100 = vector<double> (gridSize,0);
        iTot.intTot_200 = vector<double> (gridSize,0);
        iTot.intTot_300 = vector<double> (gridSize,0);
        iTot.intTot_400 = vector<double> (gridSize,0);
        iTot.intTot_500 = vector<double> (gridSize,0);
        iTot.intTot_600 = vector<double> (gridSize,0);
        iTot.intTot_700 = vector<double> (gridSize,0);

        for (size_t i=0; i<gridSize; ++i)
        {
                iTot.intTot_100[i] = 0.5*dTot.deltaTot_100[i] * 100;
                iTot.intTot_200[i] = 0.5*(dTot.deltaTot_200[i] + dTot.deltaTot_100[i])*100;
                iTot.intTot_300[i] = 0.5*(dTot.deltaTot_300[i] + dTot.deltaTot_200[i])*100;
                iTot.intTot_400[i] = 0.5*(dTot.deltaTot_400[i] + dTot.deltaTot_300[i])*100;
                iTot.intTot_500[i] = 0.5*(dTot.deltaTot_500[i] + dTot.deltaTot_400[i])*100;
                iTot.intTot_600[i] = 0.5*(dTot.deltaTot_600[i] + dTot.deltaTot_500[i])*100;
                iTot.intTot_700[i] = 0.5*(dTot.deltaTot_700[i] + dTot.deltaTot_600[i])*100;
        }
}

