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
using namespace himan;
using namespace himan::plugin;

const static param p1_e("PROB-RR1-1", aggregation(kAccumulation, ONE_HOUR),
                        processing_type(kProbabilityGreaterThan, 0.14));  // ECMWF(1h) RR>0.14mm
const static param p1_m("PROB-RR-7", aggregation(kAccumulation, ONE_HOUR),
                        processing_type(kProbabilityGreaterThan, 0.025));  // MEPS(1h) RR>0.025mm
const static param p3("PROB-RR3-6", aggregation(kAccumulation, THREE_HOURS),
                      processing_type(kProbabilityGreaterThan, 0.2));  // EC(3h) RR>0.2mm
const static param p6("PROB-RR-4", aggregation(kAccumulation, SIX_HOURS),
                      processing_type(kProbabilityGreaterThan, 0.4));  // EC(6h) RR>0.4mm

const static producer MEPSprod(260, 86, 204, "MEPSMTA");
const static producer ECprod(242, 86, 242, "ECGEPSMTA");

raw_time RoundOriginTime(const raw_time& start, int producerId)
{
	raw_time ret = start;
	const int hour = stoi(start.String("%H"));
	const int ostep = (producerId == 242) ? 12 : 3;  // origin time "step"
	const int adjust = hour % ostep;

	ret.Adjust(kHourResolution, -adjust);

	return ret;
}

raw_time GetOriginTime(const raw_time& start, int producerId)
{
	raw_time ret = start;
	const int adjust = (producerId == 242) ? 12 : 3;  // origin time "step"

	ret.Adjust(kHourResolution, -adjust);
	return ret;
}

raw_time GetValidTime(const raw_time& start)
{
	raw_time ret = start;
	ret -= ONE_HOUR;

	return ret;
}

pop::pop() : itsECEPSGeom("ECGLO0100"), itsMEPSGeom("MEPS2500D"), itsUseMEPS(true)
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

	if (itsConfiguration->Exists("disable_meps"))
	{
		itsUseMEPS = !util::ParseBoolean(itsConfiguration->GetValue("disable_meps"));

		if (itsUseMEPS == false)
		{
			itsLogger.Info("MEPS usage disabled");
		}
	}

	Start();
}

shared_ptr<info<double>> pop::GetShortProbabilityData(const forecast_time& forecastTime, const level& forecastLevel,
                                                      logger& logr)
{
	shared_ptr<info<double>> ret = nullptr;

	forecast_type ft(kStatisticalProcessing);
	if (itsUseMEPS)
	{
		logr.Info("Fetching MEPS");

		int tryNo = 0;

		forecast_time curTime =
		    forecast_time(RoundOriginTime(forecastTime.OriginDateTime(), 260), forecastTime.ValidDateTime());

		do
		{
			ret = Fetch(curTime, forecastLevel, p1_m, ft, {itsMEPSGeom}, MEPSprod, false);
			curTime = forecast_time(GetOriginTime(curTime.OriginDateTime(), 260), curTime.ValidDateTime());

		} while (++tryNo < 3 && !ret);
	}
	else
	{
		logr.Info("Fetching ECMWF");

		int tryNo = 0;

		forecast_time curTime =
		    forecast_time(RoundOriginTime(forecastTime.OriginDateTime(), 242), forecastTime.ValidDateTime());

		do
		{
			ret = Fetch(curTime, forecastLevel, p1_e, ft, {itsECEPSGeom}, ECprod, false);
			curTime = forecast_time(GetOriginTime(curTime.OriginDateTime(), 242), curTime.ValidDateTime());

		} while (++tryNo < 2 && !ret);

		// Backup solution: try to fetch 3h probability data if 1h data is not available.
		// This is far from optimal, but better than nothing.

		logr.Info("Fetching 3h probability data as backup solution");
		tryNo = 0;
		curTime = forecast_time(RoundOriginTime(forecastTime.OriginDateTime(), 242), forecastTime.ValidDateTime());
		do
		{
			ret = Fetch(curTime, forecastLevel, p3, ft, {itsECEPSGeom}, ECprod, false);
			curTime.ValidDateTime() -=
			    ONE_HOUR;

		} while (++tryNo < 3 && !ret);

		// Did not find using the latest forecast time, try the previous one to be sure
		tryNo = 0;

		curTime = forecast_time(GetOriginTime(curTime.OriginDateTime(), 242), forecastTime.ValidDateTime());

		do
		{
			ret = Fetch(curTime, forecastLevel, p3, ft, {itsECEPSGeom}, ECprod, false);
			curTime.ValidDateTime() -=
			    ONE_HOUR;

		} while (++tryNo < 3 && !ret);
	}

	return ret;
}

std::shared_ptr<info<double>> pop::GetLongProbabilityData(const forecast_time& forecastTime, const level& forecastLevel,
                                                          logger& logr)

{
	// Fetch either 3h or 6h probability data, depending on the leadtime in question.
	// The decision is actually based on both the leadtime of the smartmet forecast and
	// the analysis time of the latest ENS forecast.
	//
	// If for some reason the ENS production is delayed, we might end up trying to fetch
	// 3h probability for leadtimes those are not available (> 144h). In this case we
	// should continue with 6h probability.

	shared_ptr<info<double>> ret = nullptr;

	// origin time counter
	int otryNo = 0;

	forecast_time curTime =
	    forecast_time(RoundOriginTime(forecastTime.OriginDateTime(), 242), forecastTime.ValidDateTime());

	const forecast_type ftype(kStatisticalProcessing);

	auto f = GET_PLUGIN(fetcher);

	auto cnf = make_shared<plugin_configuration>(*itsConfiguration);
	cnf->SourceProducers({ECprod});
	cnf->SourceGeomNames({itsECEPSGeom});

	do
	{
		const long step = curTime.Step().Hours();
		logr.Info(fmt::format(
		    "Trying to fetch long probability data from ECMWF analysis time {} step {}, time interpolation enabled",
		    curTime.OriginDateTime().String(), step));

		const param& p = (step <= 144) ? p3 : p6;
		ret = f->Fetch<double>(cnf, curTime, forecastLevel, p, ftype, false, true, false, true);

		curTime = forecast_time(GetOriginTime(curTime.OriginDateTime(), 242), forecastTime.ValidDateTime());

	} while (++otryNo < 2 && !ret);

	return ret;
}

void pop::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	// 2021 POP
	// version 6.9.2021
	// Use ensemble forecasts precipitation probabilies as source data
	// <= 45h forecasts: Prob MEPS(1h) RR>=0.025mm OR Prob Ec(1h) RR>=0.14
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

	shared_ptr<info<double>> ShortProb, LongProb, Precipitation;

	const long step = forecastTime.Step().Hours();

	if (step <= 46)
	{
		ShortProb = GetShortProbabilityData(forecastTime, forecastLevel, myThreadedLogger);
		if (!ShortProb)
		{
			return;
		}
	}

	if (step >= 43)
	{
		LongProb = GetLongProbabilityData(forecastTime, forecastLevel, myThreadedLogger);

		if (!LongProb)
		{
			return;
		}
	}

	myThreadedLogger.Info("Fetching Precipitation");

	Precipitation = Fetch(forecastTime, forecastLevel,
	                      param("RRR-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type()), forecastType);

	if (!Precipitation)
	{
		return;
	}

	vector<double> result(myTargetInfo->Data().Size(), MissingDouble());

	const auto& ShortProbData = (ShortProb) ? VEC(ShortProb) : result;
	const auto& LongProbData = (LongProb) ? VEC(LongProb) : result;
	const auto& PrecipitationData = VEC(Precipitation);

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

	for (auto&& tup : zip_range(resultdata, ShortProbData, LongProbData, PrecipitationData))
	{
		double& r = tup.get<0>();
		const double shrt = tup.get<1>();
		const double lng = tup.get<2>();
		const double rr = tup.get<3>();

		if (step < 43)
		{
			r = shrt;
		}
		else if (step >= 43 && step <= 46)
		{
			r = (shrt + lng) * 0.5;
		}
		else
		{
			r = lng;
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

	const matrix<double> filter_kernel(5, 5, 1, MissingDouble(), 1 / 25.);

	const auto smoothened =
	    numerical_functions::Filter2D<double>(myTargetInfo->Data(), filter_kernel, itsConfiguration->UseCuda());

	myTargetInfo->Base()->data = move(smoothened);

	myThreadedLogger.Info(
	    fmt::format("[CPU] Missing values: {}/{}", myTargetInfo->Data().MissingCount(), myTargetInfo->Data().Size()));
}
