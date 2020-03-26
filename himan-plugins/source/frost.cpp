#include "frost.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "util.h"
#include "plugin_factory.h"
#include "fetcher.h"
#include <math.h>
#include <NFmiLocation.h>
#include <NFmiMetTime.h>

using namespace std;
using namespace himan;
using namespace himan::plugin;

const string itsName("frost");

const param FParam("FROST-PROB");

frost::frost()
{
	itsLogger = logger("itsName");
}

void frost::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams({FParam});

	Start();
}

void frost::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	NFmiMetTime theTime(static_cast<short>(stoi(myTargetInfo->Time().ValidDateTime().String("%Y"))),
	                    static_cast<short>(stoi(myTargetInfo->Time().ValidDateTime().String("%m"))),
	                    static_cast<short>(stoi(myTargetInfo->Time().ValidDateTime().String("%d"))),
	                    static_cast<short>(stoi(myTargetInfo->Time().ValidDateTime().String("%H"))),
	                    static_cast<short>(stoi(myTargetInfo->Time().ValidDateTime().String("%M"))));

	const param TParam("T-K");
	const param TDParam("TD-K");
	const param TGParam("TG-K");
	const param WGParam("FFG-MS");
	const param T0Param("PROB-TC-0");
	const param NParam("N-0TO1");
	const param RADParam("RADGLO-WM2");
	//const param LCParam("LC-0TO1");
	//const param ICNParam("ICNCT-PRCNT");

	auto myThreadedLogger = logger("frostThread #" + to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();
	forecast_type forecastType = myTargetInfo->ForecastType();

	myThreadedLogger.Info("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + ", level " +
			static_cast<string>(forecastLevel));

	auto cnf = make_shared<plugin_configuration>(*itsConfiguration);
	auto f = GET_PLUGIN(fetcher);

	cnf->SourceProducers({producer(131, 98, 150, "ECG")});

        const std::vector<std::string>& names = { "ECEUR0100", "ECGLO0100", "ECEUR0200" };
	cnf->SourceGeomNames(names);

	info_t TInfo = Fetch(forecastTime, level(kGround, 0), TParam, forecastType, false);
	info_t TDInfo = Fetch(forecastTime, level(kGround, 0), TDParam, forecastType, false);
	info_t TGInfo = Fetch(forecastTime, level(kGroundDepth, 0, 7), TGParam, forecastType, false);
	info_t WGInfo = Fetch(forecastTime, level(kGround, 0), WGParam, forecastType, false);
	info_t NInfo = Fetch(forecastTime, level(kGround, 0), NParam, forecastType, false);
        //info_t LCInfo = Fetch(forecastTime, level(kGround, 0), LCParam, forecastType, false);

        info_t RADInfo;

        try
        {
                cnf->SourceProducers({producer(240, 86, 240, "ECGMTA")});
                RADInfo = f->Fetch(cnf, forecastTime, level(kHeight, 0), RADParam, forecastType, false);
        }
        catch (HPExceptionType& e)
        {
                if  (e == kFileDataNotFound)
                {
                        myThreadedLogger.Error("No data found.");
                }
                return;
        }

	info_t T0ECInfo;

	forecast_type stat_type = forecast_type(kStatisticalProcessing);

	try
	{
		cnf->SourceProducers({producer(242, 86, 242, "ECM_PROB")});
		cnf->SourceGeomNames({"ECEUR0200"});
		T0ECInfo = f->Fetch(cnf, forecastTime, level(kGround, 0), T0Param, stat_type, false);
	}
	catch (HPExceptionType& e)
	{
		if  (e == kFileDataNotFound)
		{
			myThreadedLogger.Error("No data found.");
		}
		return;
	}

	info_t T0MEPSInfo;

	try
	{
		cnf->SourceProducers({producer(260, 86, 204, "MEPSMTA")});
		cnf->SourceGeomNames({"MEPS2500D"});
		T0MEPSInfo = f->Fetch(cnf, forecastTime, level(kHeight, 2), T0Param, stat_type, false);
	}
	catch (HPExceptionType& e)
	{
                if  (e == kFileDataNotFound)
                {
                        myThreadedLogger.Error("No data found.");
                }
                return;
	}

	/*info_t ICNInfo;

	try
	{
		cnf->SourceProducers({producer(150, 86, 150, "HBM_EC")});
		cnf->SourceGeomNames({"HBM"});
		ICNInfo = f->Fetch(cnf, forecastTime, level(kHeight, 0), ICNParam, forecastType, false);
	}
	        catch (HPExceptionType& e)
        {
                if  (e == kFileDataNotFound)
                {
                        myThreadedLogger.Error("No data found.");
                }
                return;
	}*/

	if (!TInfo || !TDInfo || !TGInfo || !WGInfo || !NInfo || !RADInfo || !T0ECInfo || !T0MEPSInfo) // LCINfo, ICNInfo removed.
	{
		myThreadedLogger.Warning("Skipping step " + static_cast<string>(forecastTime.Step()) + ", level " +
			static_cast<string>(forecastLevel));
		return;
	} 

	string deviceType = "CPU";

	LOCKSTEP(myTargetInfo, TInfo, TDInfo, TGInfo, WGInfo, NInfo, RADInfo, T0ECInfo, T0MEPSInfo) // LCInfo, ICNInfo, removed.

	{
		double T = TInfo->Value() - himan::constants::kKelvin;
		double TD = TDInfo->Value() - himan::constants::kKelvin;
		double TG = TGInfo->Value() - himan::constants::kKelvin;
		double WG = WGInfo->Value();
		double N = NInfo->Value();
		double RAD = RADInfo->Value();
		double T0EC = T0ECInfo->Value();
		double T0MEPS = T0MEPSInfo->Value();
		//double LC = TInfo->Value();
		//double ICN = TInfo->Value();

		if (IsMissingValue({T, TD, TG, WG, N, RAD, T0EC, T0MEPS})) // LC, ICN
		{
			continue;
		}

		// Calculating indexes and coefficients.

		// dewIndex

		double dewIndex = kHPMissingValue;

		if (TD < -5.0)
		{
			dewIndex = 1.0;
		}

		else if (TD > 5.0)
		{
			dewIndex = 0.0;
		}

		else
		{
			dewIndex = fabs((TD - 5.0) / 10.0);  // TD -5...5
		}

		// nIndex

		double nIndex = kHPMissingValue;

		nIndex = 1.0 - N;

		// tIndexHigh

		double tIndexHigh = kHPMissingValue;

		if (T < 2.5)
		{
			tIndexHigh = 1;
		}

		else if (T > 15.0)
		{
			tIndexHigh = 0;
		}

		else
		{
			tIndexHigh = (fabs((T - 15.0) / 12.5)) * (fabs((T - 15.0) / 12.5));
		}

		// wgWind

		double wgWind = kHPMissingValue;

		if (WG < 1.0)
		{
			wgWind = 1.0;
		}

		else if (WG > 6.0)
		{
			wgWind = 0;
		}

		else
		{
			wgWind = fabs((WG - 6.0) / 5.0);
		}

		double lowWindCoef = 1.0;
		double nCoef = 1.0;
		double weight = 4.0;
		double stabCoef = 1.5 * wgWind + 1.0;
		
		// Adjusting coefficients when T above and near zero.

		if (T >= 0 && T < 2.5)
		{
			lowWindCoef = fabs((T - 2.5) / 2.5) * weight + 1.0;
			nCoef = 1.0 / lowWindCoef;
		}

		// Calculating frost probability.

		double frost_prob = kHPMissingValue;

		if (T < -3.0)
		{
			frost_prob = 100.0;
		}

		else if (T < 0)
		{
			frost_prob = 90.0;
		}

		else
		{
			frost_prob = ((lowWindCoef * dewIndex) + stabCoef + (nCoef * nIndex)) /
				(lowWindCoef + stabCoef + (1.0 / lowWindCoef)) * 100 * tIndexHigh;
		}

		// Raising the frost probability due to ground temperature TG.

		double tgModel = kHPMissingValue;

		if (T < 5)
		{

			if (TG < -6.0)
			{
				tgModel = 100.0;
			}

			else if (TG > 5.0)
			{
				tgModel = 0;
			}
		
			else
			{
				tgModel = sqrt(fabs((TG - 5.0) / 11.0)) * 100;
			}

		}

		if (frost_prob < tgModel)
		{
			frost_prob = tgModel;
		}

		// Raising the frost probability due to probability of T<0 and T. Both EC and MEPS cases.

		if (T0EC > 60.0 && T < 5.0 && frost_prob < T0EC)
		{
			frost_prob = T0EC;
		}

		if (frost_prob < T0MEPS && T0MEPS > 40.0)
		{
			frost_prob = (frost_prob * 2.0 + T0MEPS) / 3.0;
		}

		// No frost when radiation is high enough. Valid in  1.4.-15.9.

		int month = stoi(forecastTime.ValidDateTime().String("%m"));
		int day = stoi(forecastTime.ValidDateTime().String("%d"));	

		if ((month >= 4 && month <= 8) || (month == 9 && day <= 15))
		{
			if (RAD > 175)
			{
				frost_prob = 0;
			}

		}

		// Lowering frost probability due to sun's elevation angle. Valid in 1.4.-15.9.  

		if ((month >= 4 && month <= 8) || (month == 9 && day <= 15))
		{
			NFmiLocation theLocation(myTargetInfo->LatLon().X(), myTargetInfo->LatLon().Y());

                	double elevationAngle = theLocation.ElevationAngle(theTime);
                	double angleCoef = kHPMissingValue;

			if (elevationAngle < -1.0)
                	{
                        	angleCoef = 1;
                	}

                	else if (elevationAngle > 20.0)
                	{
                        	angleCoef = 0;
                	}

                	else
                	{
                        	angleCoef = fabs((elevationAngle - 20.0) / 21.0);
                	}

			frost_prob = angleCoef * frost_prob;
		}

		// No frost probability on sea when there is no ice.

		/*if (ICN == 0 &&  LC == 0)
		{
			frost_prob = 0;
		}*/

		// Lowering frost probability when forecasted T is high enough.

		if (T > 6)
		{
			if (T > 15.0)
			{
				frost_prob = 0;
			}
			else
			{
				frost_prob = frost_prob * fabs((T - 15.0) / 9.0);
			}
		}

		myTargetInfo->Value(frost_prob);

	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
