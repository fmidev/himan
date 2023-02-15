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
	const int ostep = (producerId == 242) ? 12 : 3;  // origin time "step"
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

	param theRequestedParam("POP-0TO1", aggregation(), processing_type(kProbability));

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
	// version 6.9.2021
	// Use ensemble forecasts precipitation probabilies as source data
	// <= 45h forecasts: Prob MEPS(1h) RR>=0.025mm
	// 46h - 130h forecasts: Prob EC(3h) RR>=0.2mm
	// >131h forecasts: Prob EC(6h) RR>=0.4mm
	// also: if smartemt data contains rain, POP should have values
	// also: weaken probabilites of temporally distant rains
	// finish with spatial smoothing

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
		const param p("PROB-RR-7", aggregation(),
		              processing_type(kProbabilityGreaterThan, 0.025));  // MEPS(1h) RR>0.025mm
		const producer MEPSprod(260, 86, 204, "MEPSMTA");
		do
		{
			raw_time now;
			curTime = forecast_time(GetOriginTime(curTime.OriginDateTime(), 260), curTime.ValidDateTime());

			MEPS =
			    Fetch(curTime, forecastLevel, p, forecast_type(kStatisticalProcessing), {itsMEPSGeom}, MEPSprod, false);
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

		param p("PROB-RR3-6", aggregation(), processing_type(kProbabilityGreaterThan, 0.4));  // EC(3h) RR>0.2mm

		int tryNo = 0;

		forecast_time curTime = forecastTime;
		const producer ECprod(242, 86, 242, "ECGEPSMTA");

		do
		{
			curTime = forecast_time(GetOriginTime(curTime.OriginDateTime(), 242), curTime.ValidDateTime());

			if (curTime.Step().Hours() > 131)
			{
				p = param("PROB-RR-4", aggregation(),
				          processing_type(kProbabilityGreaterThan, 0.4));  // EC(6h) RR>0.4mm
			}

			EC = Fetch(curTime, forecastLevel, p, forecast_type(kStatisticalProcessing), {itsECEPSGeom}, ECprod, false);
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

	// shortest forecast has these:
	double maxprob = 1.0;
	double minprob = 0.8;

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
		minprob = 0.4;
	}
	else if (step > 6)
	{
		minprob = 0.6;
	}

	// Minimum precipitation value to be accepted, mm/h
	const double RRlimit = 0.01;

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

		if (rr > RRlimit)
		{
			r = max(r, minprob);
		}

		r = min(r, maxprob);
	}

	// Produce area average over the neighboring grid points
	// x x x x x
	// x x x x x
	// x x o x x
	// x x x x x
	// x x x x x

	const himan::matrix<double> filter_kernel(5, 5, 1, MissingDouble(), 1 / 25.);

	const auto smoothened =
	    numerical_functions::Filter2D<double>(myTargetInfo->Data(), filter_kernel, itsConfiguration->UseCuda());

	myTargetInfo->Base()->data = move(smoothened);

	myThreadedLogger.Info(
	    fmt::format("[CPU] Missing values: {}/{}", myTargetInfo->Data().MissingCount(), myTargetInfo->Data().Size()));
}
