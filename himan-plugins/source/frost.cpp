#include "frost.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "fetcher.h"
#include <math.h>
#include <NFmiLocation.h>
#include <NFmiMetTime.h>

using namespace std;
using namespace himan;
using namespace himan::plugin;

// rampDown function returns a value between 1 and 0,
// depending where valueInBetween is in the interval between start and end.

double rampDown(const double& start, const double& end, const double& valueInBetween)
{
	if (valueInBetween <= start)
		return 1.0;

	if (valueInBetween >= end)
		return 0.0;

	return fabs((valueInBetween - end) / (end - start));
}

const param FParam("PROB-FROST");

frost::frost()
{
	itsLogger = logger("frost");
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
                //return;
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

		dewIndex = rampDown(-5, 5, TD); // TD -5...5

		// nIndex

		double nIndex = kHPMissingValue;

		nIndex = 1.0 - N;

		// tIndexHigh

		double tIndexHigh = kHPMissingValue;

		tIndexHigh = rampDown(2.5, 15, T) * rampDown(2.5, 15, T);

		// wgWind

		double wgWind = kHPMissingValue;

		wgWind = rampDown(1, 6, WG);

		double lowWindCoef = 1.0;
		double nCoef = 1.0;
		double weight = 4.0;
		double stabCoef = 1.5 * wgWind + 1.0;
		
		// Adjusting coefficients when T above and near zero.

		if (T >= 0 && T < 2.5)
		{
			lowWindCoef = rampDown(0, 2.5, T) * weight + 1.0;
			nCoef = 1.0 / lowWindCoef;
		}

		// Calculating frost probability.

		double frost_prob = kHPMissingValue;

		if (T < -3.0)
		{
			frost_prob = 1.0;
		}

		else if (T < 0)
		{
			frost_prob = 0.9;
		}

		else
		{
			frost_prob = ((lowWindCoef * dewIndex) + stabCoef + (nCoef * nIndex)) /
				(lowWindCoef + stabCoef + (1.0 / lowWindCoef)) * tIndexHigh;
		}

		// Raising the frost probability due to ground temperature TG.

		double tgModel = kHPMissingValue;

		if (T < 5)
		{
			tgModel = sqrt(rampDown(-6, 5, TG));
		}

		if (frost_prob < tgModel)
		{
			frost_prob = tgModel;
		}

		// Raising the frost probability due to probability of T<0 and T. Both EC and MEPS cases.

		if (T0EC > 0.6 && T < 5.0 && frost_prob < T0EC)
		{
			frost_prob = T0EC;
		}

		if (frost_prob < T0MEPS && T0MEPS > 0.4)
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
				frost_prob = 0.0;
			}

		}

		// Lowering frost probability due to sun's elevation angle. Valid in 1.4.-15.9.  

		if ((month >= 4 && month <= 8) || (month == 9 && day <= 15))
		{
			NFmiLocation theLocation(myTargetInfo->LatLon().X(), myTargetInfo->LatLon().Y());

                	double elevationAngle = theLocation.ElevationAngle(theTime);
                	double angleCoef = kHPMissingValue;

			angleCoef = rampDown(-1, 20, elevationAngle);

			frost_prob = angleCoef * frost_prob;
		}

		// No frost probability on sea when there is no ice.

		/*if (ICN == 0 &&  LC == 0)
		{
			frost_prob = 0;
		}*/

		// Lowering frost probability when forecasted T is high enough.

		if (T > 6.0)
		{
			frost_prob = frost_prob * rampDown(6, 15, T);
		}

		myTargetInfo->Value(frost_prob);

	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
