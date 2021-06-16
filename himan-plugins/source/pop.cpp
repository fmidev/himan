/**
 * @file pop.cpp
 *
 */
#include "pop.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "util.h"

#include "fetcher.h"

using namespace std;
using namespace himan::plugin;

himan::raw_time GetOriginTime(const himan::raw_time& start, int producerId)
{
	ASSERT(producerId == 242 || producerId == 260);

	// Provide latest analysis time for either MEPS or ECEPS
	himan::raw_time ret = start;

	const int hour = stoi(start.String("%H"));
	const int ostep = (producerId == 242) ? 12 : 3; // origin time "step"
	const int adjust = (hour % ostep == 0) ? ostep : hour % ostep;
	ret.Adjust(himan::kHourResolution, -adjust);

	return ret;
}

pop::pop() : itsECEPSGeom("ECGLO0200"), itsMEPSGeom("MEPS2500D")
{
	itsLogger = logger("pop");
}

void pop::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param theRequestedParam("POP-PRCNT", 259);

	theRequestedParam.Unit(kPrcnt);

	SetParams({theRequestedParam});

	if (itsConfiguration->Exists("eceps_geom"))
	{
		itsECEPSGeom = itsConfiguration->GetValue("eceps_geom");
	}

	if (itsConfiguration->Exists("meps_geom"))
	{
		itsMEPSGeom = itsConfiguration->GetValue("meps_geom");
	}

	Start();
}

void pop::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	// 2021 POP
	// versio 18.5.2021
	// Käytetään pääasiallisesti parviennusteiden sateen todennäköisyyksiä:
	// <= 45h ennusteet: Prob MEPS(1h) RR>=0.025mm
	// 46h - 130h ennusteet: Prob EC(3h) RR>=0.075mm
	// >131h ennusteet: Prob EC(6h) RR>=0.15mm
	// lisäksi huomioidaan että jos editoitu sadetta, myös POP arvoja
	// lisäksi kaukaisia sateita heikennetään
	// lopuksi tehdään aluetasoitus

	// Current time and level as given to this thread

	const forecast_type forecastType = myTargetInfo->ForecastType();
	forecast_time forecastTime = myTargetInfo->Time();
	const level forecastLevel = myTargetInfo->Level();

	auto myThreadedLogger = logger(fmt::format("popThread #{}", threadIndex));

	myThreadedLogger.Debug(fmt::format("Calculating time {} level {}",
	                                   static_cast<string>(forecastTime.ValidDateTime()),
	                                   static_cast<string>(forecastLevel)));

	shared_ptr<info<double>> MEPS, EC, SmartMet;

	const long step = forecastTime.Step().Hours();

	if (step <= 46)
	{
		myThreadedLogger.Info("Fetching MEPS");

		int tryNo = 0;

		forecast_time curTime = forecastTime;
		const param p("PROB-RR-1");  // MEPS(1h) RR>=0.025mm

		do
		{
			curTime = forecast_time(GetOriginTime(curTime.OriginDateTime(), 260), curTime.ValidDateTime());

			MEPS = Fetch(curTime, forecastLevel, p, forecast_type(kStatisticalProcessing), {itsMEPSGeom},
			             producer(260, "MEPSMTA"), false);
			tryNo++;
		} while (tryNo < 3 && !MEPS);

		if (!MEPS)
		{
			return;
		}
	}

	if (step >= 43)
	{
		myThreadedLogger.Info("Fetching ECMWF");

		param p("PROB-RR3-1");  // EC(3h) RR>0.075mm

		int tryNo = 0;

		forecast_time curTime = forecastTime;

		do
		{
			curTime = forecast_time(GetOriginTime(curTime.OriginDateTime(), 242), curTime.ValidDateTime());

			if (curTime.Step().Hours() > 131)
			{
				p = param("PROB-RR-1");  // EC(6h) RR>0.15mm
			}

			EC = Fetch(curTime, forecastLevel, p, forecast_type(kStatisticalProcessing), {itsECEPSGeom},
			           producer(242, "ECGEPSMTA"), false);
			tryNo++;
		} while (tryNo < 2 && !EC);

		if (!EC)
		{
			return;
		}
	}

	myThreadedLogger.Info("Fetching SmartMet");

	SmartMet = Fetch(forecastTime, forecastLevel, param("RRR-KGM2"), forecastType);

	if (!SmartMet)
	{
		return;
	}

	vector<double> result(myTargetInfo->Data().Size(), MissingDouble());

	const auto& MEPSdata = (MEPS) ? VEC(MEPS) : result;
	const auto& ECdata = (EC) ? VEC(EC) : result;
	const auto& SmartMetdata = VEC(SmartMet);

	auto& resultdata = VEC(myTargetInfo);

	// maxprob and minprob are used to limit the allowed range
	// of values

	double maxprob = 1.0;
	double minprob = 0.4;

	if (step > 130)
	{
		maxprob = 0.7;
		minprob = 0.1;
	}
	else if (step > 45)
	{
		maxprob = 0.8;
		minprob = 0.2;
	}
	else if (step > 24)
	{
		maxprob = 0.9;
	}

	for (auto&& tup : zip_range(resultdata, MEPSdata, ECdata, SmartMetdata))
	{
		double& r = tup.get<0>();
		const double meps = tup.get<1>();
		const double ec = tup.get<2>();
		const double rr = tup.get<3>();

		if (step < 43)
		{
			r = meps;
		}
		else if (step >= 43 && step <= 46)
		{
			r = (meps + ec) * 0.5;
		}
		else
		{
			r = ec;
		}

		if (rr > 0.0)
		{
			r = max(r, minprob);
		}

		r = min(r, maxprob);
	}

	// Produce area average over the neighboring grid points
	// x x x
	// x o x
	// x x x

	const himan::matrix<double> filter_kernel(3, 3, 1, MissingDouble(), 1 / 9.);

	const auto smoothened =
	    numerical_functions::Filter2D<double>(myTargetInfo->Data(), filter_kernel, itsConfiguration->UseCuda());

	myTargetInfo->Base()->data = move(smoothened);

	myThreadedLogger.Info(
	    fmt::format("[CPU] Missing values: {}/{}", myTargetInfo->Data().MissingCount(), myTargetInfo->Data().Size()));
}
