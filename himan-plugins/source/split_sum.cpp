#include "split_sum.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <iostream>
#include <map>
#include <thread>

#include "radon.h"
#include "writer.h"

using namespace himan;
using namespace himan::plugin;

const int SUB_THREAD_COUNT = 5;

std::map<std::string, params> sourceParameters;

struct target_paramdef
{
	param targetParam;
	bool isRate;  // define if this param is a rate or an accumulation (rates are divided by the time period)
	double truncateSmallerValues;     // truncate smaller values than this to zero
	double lowerLimit;                // define lowest possible value
	double scale;                     // scale values by this
	HPTimeResolution rateResolution;  // base time resolution when calculating rates

	target_paramdef(const param& p, bool ir = false, double tsv = MissingDouble(), double ll = 0.0, double s = 1.0,
	                HPTimeResolution rr = kHourResolution)
	    : targetParam(p), isRate(ir), truncateSmallerValues(tsv), lowerLimit(ll), scale(s), rateResolution(rr)
	{
	}
};

std::vector<target_paramdef> targetParameters;

split_sum::split_sum()
{
	itsLogger = logger("split_sum");

	// Define source parameters for each output parameter

	targetParameters.clear();

	// General precipitation (liquid + solid)
	sourceParameters["RR-1-MM"] = {param("RR-KGM2")};
	sourceParameters["RR-3-MM"] = {param("RR-KGM2")};
	sourceParameters["RR-6-MM"] = {param("RR-KGM2")};
	sourceParameters["RR-12-MM"] = {param("RR-KGM2")};
	sourceParameters["RR-24-MM"] = {param("RR-KGM2")};

	sourceParameters["RRC-3-MM"] = {param("RRC-KGM2")};

	sourceParameters["RRR-KGM2"] = {param("RR-KGM2")};
	sourceParameters["RRRC-KGM2"] = {param("RRC-KGM2")};
	sourceParameters["RRRL-KGM2"] = {param("RRL-KGM2")};

	// Snow
	sourceParameters["SN-3-MM"] = {param("SNACC-KGM2")};
	sourceParameters["SN-6-MM"] = {param("SNACC-KGM2")};
	sourceParameters["SN-24-MM"] = {param("SNACC-KGM2")};
	sourceParameters["SN-120-MM"] = {param("SNACC-KGM2")};

	sourceParameters["SNR-KGM2"] = {param("SNACC-KGM2")};
	sourceParameters["SNRC-KGM2"] = {param("SNC-KGM2")};
	sourceParameters["SNRL-KGM2"] = {param("SNL-KGM2")};

	// Graupel
	sourceParameters["GRR-MMH"] = {param("GR-KGM2")};

	// Solid (snow + graupel + hail)
	sourceParameters["RRRS-KGM2"] = {param("RRS-KGM2")};
	sourceParameters["RRS-3-MM"] = {param("RRS-KGM2")};
	sourceParameters["RRS-24-MM"] = {param("RRS-KGM2")};

	// Radiation
	sourceParameters["RADGLO-WM2"] = {param("RADGLOA-JM2")};
	sourceParameters["RADLW-WM2"] = {param("RADLWA-JM2")};
	sourceParameters["RTOPLW-WM2"] = {param("RTOPLWA-JM2")};
	sourceParameters["RNETLW-WM2"] = {param("RNETLWA-JM2")};
	sourceParameters["RADSW-WM2"] = {param("RADSWA-JM2")};
	sourceParameters["RNETSW-WM2"] = {param("RNETSWA-JM2")};
	sourceParameters["RADGLOC-WM2"] = {param("RADGLOCA-JM2")};
	sourceParameters["RADLWC-WM2"] = {param("RADLWCA-JM2")};
}

void SetupParameters(std::shared_ptr<const plugin_configuration> conf)
{
	/*
	 * Set target parameter to split_sum.
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	const long producerId = conf->TargetProducer().Id();
	const bool isMetcoop = producerId == 260 || producerId == 261 || (producerId >= 270 && producerId <= 272);
	const bool isECMWF = producerId == 240 || producerId == 241 || producerId == 243;

	if (conf->Exists("rr1h") && conf->GetValue("rr1h") == "true")
	{
		targetParameters.emplace_back(param("RR-1-MM", aggregation(kAccumulation, ONE_HOUR), processing_type()), false,
		                              isMetcoop ? 0.01 : MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("rr3h") && conf->GetValue("rr3h") == "true")
	{
		targetParameters.emplace_back(param("RR-3-MM", aggregation(kAccumulation, THREE_HOURS), processing_type()),
		                              false, isMetcoop ? 0.01 : MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("rr6h") && conf->GetValue("rr6h") == "true")
	{
		targetParameters.emplace_back(param("RR-6-MM", aggregation(kAccumulation, SIX_HOURS), processing_type()), false,
		                              isMetcoop ? 0.01 : MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("rr12h") && conf->GetValue("rr12h") == "true")
	{
		targetParameters.emplace_back(param("RR-12-MM", aggregation(kAccumulation, TWELVE_HOURS), processing_type()),
		                              false, isMetcoop ? 0.01 : MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("rr24h") && conf->GetValue("rr24h") == "true")
	{
		targetParameters.emplace_back(
		    param("RR-24-MM", aggregation(kAccumulation, time_duration("24:00")), processing_type()), false,
		    isMetcoop ? 0.01 : MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("sn3h") && conf->GetValue("sn3h") == "true")
	{
		targetParameters.emplace_back(param("SN-3-MM", aggregation(kAccumulation, THREE_HOURS), processing_type()),
		                              false, MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("sn6h") && conf->GetValue("sn6h") == "true")
	{
		targetParameters.emplace_back(param("SN-6-MM", aggregation(kAccumulation, SIX_HOURS), processing_type()), false,
		                              MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("sn24h") && conf->GetValue("sn24h") == "true")
	{
		targetParameters.emplace_back(
		    param("SN-24-MM", aggregation(kAccumulation, time_duration("24:00")), processing_type()), false,
		    MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("sn120h") && conf->GetValue("sn120h") == "true")
	{
		targetParameters.emplace_back(
		    param("SN-120-MM", aggregation(kAccumulation, time_duration("120:00")), processing_type()), false,
		    MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("rrc3h") && conf->GetValue("rrc3h") == "true")
	{
		targetParameters.emplace_back(param("RRC-3-MM", aggregation(kAccumulation, THREE_HOURS), processing_type()),
		                              false, isMetcoop ? 0.01 : MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("rrr") && conf->GetValue("rrr") == "true")
	{
		targetParameters.emplace_back(param("RRR-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type()), true,
		                              isMetcoop ? 0.01 : MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("rrrc") && conf->GetValue("rrrc") == "true")
	{
		targetParameters.emplace_back(param("RRRC-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type()), true,
		                              isMetcoop ? 0.01 : MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("rrrl") && conf->GetValue("rrrl") == "true")
	{
		targetParameters.emplace_back(param("RRRL-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type()), true,
		                              isMetcoop ? 0.01 : MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	// Graupel

	if (conf->Exists("grr") && conf->GetValue("grr") == "true")
	{
		targetParameters.emplace_back(param("GRR-MMH", aggregation(kAccumulation, ONE_HOUR), processing_type()), true);
	}

	// Solid

	if (conf->Exists("rrrs") && conf->GetValue("rrrs") == "true")
	{
		targetParameters.emplace_back(param("RRRS-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type()),
		                              true);
	}

	if (conf->Exists("rrs3h") && conf->GetValue("rrs3h") == "true")
	{
		targetParameters.emplace_back(param("RRS-3-MM", aggregation(kAccumulation, THREE_HOURS), processing_type()));
	}

	if (conf->Exists("rrs24h") && conf->GetValue("rrs24h") == "true")
	{
		targetParameters.emplace_back(
		    param("RRS-24-MM", aggregation(kAccumulation, time_duration("24:00")), processing_type()));
	}

	// Snow

	if (conf->Exists("snr") && conf->GetValue("snr") == "true")
	{
		targetParameters.emplace_back(param("SNR-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type()), true,
		                              MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("snrc") && conf->GetValue("snrc") == "true")
	{
		targetParameters.emplace_back(param("SNRC-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type()), true,
		                              MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	if (conf->Exists("snrl") && conf->GetValue("snrl") == "true")
	{
		targetParameters.emplace_back(param("SNRL-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type()), true,
		                              MissingDouble(), 0.0, isECMWF ? 1000. : 1.);
	}

	// Radiation

	if (conf->Exists("glob") && conf->GetValue("glob") == "true")
	{
		targetParameters.emplace_back(param("RADGLO-WM2", aggregation(kAverage), processing_type()), true,
		                              MissingDouble(), 0.0, 1., kSecondResolution);
	}

	if (conf->Exists("globc") && conf->GetValue("globc") == "true")
	{
		targetParameters.emplace_back(param("RADGLOC-WM2", aggregation(kAverage), processing_type()), true,
		                              MissingDouble(), 0.0, 1., kSecondResolution);
	}

	if (conf->Exists("lw") && conf->GetValue("lw") == "true")
	{
		targetParameters.emplace_back(param("RADLW-WM2", aggregation(kAverage), processing_type()), true,
		                              MissingDouble(), 0.0, 1., kSecondResolution);
	}

	if (conf->Exists("lwc") && conf->GetValue("lwc") == "true")
	{
		targetParameters.emplace_back(param("RADLWC-WM2", aggregation(kAverage), processing_type()), true,
		                              MissingDouble(), 0.0, 1., kSecondResolution);
	}

	if (conf->Exists("toplw") && conf->GetValue("toplw") == "true")
	{
		// Same grib2 parameter definition as with RADLW-WM2, this is just on
		// another surface
		targetParameters.emplace_back(param("RTOPLW-WM2", aggregation(kAverage), processing_type()), true,
		                              MissingDouble(), MissingDouble(), 1., kSecondResolution);
	}

	if (conf->Exists("netlw") && conf->GetValue("netlw") == "true")
	{
		targetParameters.emplace_back(param("RNETLW-WM2", aggregation(kAverage), processing_type()), true,
		                              MissingDouble(), MissingDouble(), 1., kSecondResolution);
	}

	if (conf->Exists("sw") && conf->GetValue("sw") == "true")
	{
		targetParameters.emplace_back(param("RADSW-WM2", aggregation(kAverage), processing_type()), true,
		                              MissingDouble(), 0.0, 1., kSecondResolution);
	}

	if (conf->Exists("netsw") && conf->GetValue("netsw") == "true")
	{
		targetParameters.emplace_back(param("RNETSW-WM2", aggregation(kAverage), processing_type()), true,
		                              MissingDouble(), MissingDouble(), 1., kSecondResolution);
	}

	auto ParseParamFromString =
	    [](const std::string& name, const std::string& aname, const std::string& adur, const std::string& pname)
	{
		param p(name);

		if (aname.empty() == false)
		{
			time_duration d = adur.empty() ? time_duration() : time_duration(adur);

			aggregation a(HPStringToAggregationType.at(aname), d);
			p.Aggregation(a);
		}

		return p;
	};

	if (conf->Exists("source_param") && conf->Exists("target_param") && conf->Exists("target_param_aggregation"))
	{
		const auto sourceParam =
		    ParseParamFromString(conf->GetValue("source_param"), conf->GetValue("source_param_aggregation"),
		                         conf->GetValue("source_param_aggregation_period"), "");
		const auto targetParam =
		    ParseParamFromString(conf->GetValue("target_param"), conf->GetValue("target_param_aggregation"),
		                         conf->GetValue("target_param_aggregation_period"), "");

		sourceParameters[targetParam.Name()] = {sourceParam};
		targetParameters.emplace_back(targetParam);

		if (conf->Exists("is_rate"))
		{
			targetParameters.back().isRate = util::ParseBoolean(conf->GetValue("is_rate"));
		}

		if (conf->Exists("truncate_smaller_values"))
		{
			const std::string str = conf->GetValue("truncate_smaller_values");
			if (str == "MISSING")
			{
				targetParameters.back().truncateSmallerValues = MissingDouble();
			}
			else
			{
				targetParameters.back().truncateSmallerValues = stod(str);
			}
		}

		if (conf->Exists("lower_limit"))
		{
			const std::string str = conf->GetValue("lower_limit");
			if (str == "MISSING")
			{
				targetParameters.back().lowerLimit = MissingDouble();
			}
			else
			{
				targetParameters.back().lowerLimit = stod(str);
			}
		}

		if (conf->Exists("scale"))
		{
			const std::string str = conf->GetValue("scale");
			targetParameters.back().scale = stod(str);
		}

		if (conf->Exists("rate_resolution"))
		{
			const std::string str = conf->GetValue("rate_resolution");
			targetParameters.back().rateResolution = HPStringToTimeResolution.at(str);
		}
	}
}

void split_sum::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetupParameters(conf);

	if (targetParameters.empty())
	{
		itsLogger.Error("No parameter definition given, abort");
		return;
	}

	params pp;

	for (const auto& p : targetParameters)
	{
		pp.push_back(p.targetParam);
	}

	SetParams(pp);

	// Default max workers (db connections) is 16.
	// Later in this plugin we launch sub threads so that the total number of
	// threads will be over 16. Connection pool is not working 100% correctly
	// because at that point it will hang.
	// To prevent this, make the pool larger.

	auto r = GET_PLUGIN(radon);
	r->PoolMaxWorkers(SUB_THREAD_COUNT * 12);

	Start();
}

/*
 * Calculate()
 *
 * This function does the actual calculation.
 */

void split_sum::Calculate(std::shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	std::vector<std::thread*> threads;
	std::vector<std::shared_ptr<info<double>>> infos;

	int subThreadIndex = 0;

	for (myTargetInfo->Reset<param>(); myTargetInfo->Next<param>(); ++subThreadIndex)
	{
		auto newInfo = std::make_shared<info<double>>(*myTargetInfo);

		infos.push_back(newInfo);  // extend lifetime over this loop

		// Pass param by reference to sub-thread: aggregation duration is only available there
		// when data is actually fetched, and we need that information when writing data to disk

		threads.push_back(new std::thread(&split_sum::DoParam, this, newInfo, std::ref(myTargetInfo->Param()),
		                                  fmt::format("{}_{}", threadIndex, subThreadIndex)));

		if (subThreadIndex % SUB_THREAD_COUNT == 0)
		{
			for (auto& thread : threads)
			{
				if (thread->joinable())
					thread->join();
			}

			infos.clear();
		}
	}

	for (auto& thread : threads)
	{
		if (thread->joinable())
			thread->join();
	}
}

void split_sum::DoParam(std::shared_ptr<info<double>> myTargetInfo, param& par, std::string subThreadIndex) const
{
	ASSERT(myTargetInfo);

	target_paramdef pd(param("XX"));

	for (const auto& p : targetParameters)
	{
		if (p.targetParam.Name() == par.Name())
		{
			pd = p;
			break;
		}
	}

	if (pd.targetParam.Name() == "XX")
	{
		itsLogger.Error(fmt::format("Given parameter {} aggregation {} not found from configuration", par.Name(),
		                            static_cast<std::string>(par.Aggregation())));
		return;
	}

	const std::string myParamName = myTargetInfo->Param().Name();

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	auto myThreadedLogger = logger("splitSumSubThread#" + subThreadIndex);

	myThreadedLogger.Info(fmt::format("Calculating parameter {} time {} level {}", myParamName,
	                                  static_cast<std::string>(myTargetInfo->Time().ValidDateTime()),
	                                  static_cast<std::string>(forecastLevel)));

	// Have to re-fetch infos each time since we might have to change element
	// from liquid to snow to radiation so we need also different source parameters

	std::shared_ptr<info<double>> curSumInfo;
	std::shared_ptr<info<double>> prevSumInfo;

	if (myTargetInfo->Time().Step().Minutes() == 0)
	{
		// This is the first time step, calculation can not be done

		myThreadedLogger.Info(fmt::format("This is the first time step -- not calculating {} for step {}", myParamName,
		                                  static_cast<std::string>(forecastTime.Step())));
		return;
	}

	/*
	 * Two modes of operation:
	 *
	 * 1) When calculating split_sum sums, always get the previous
	 * step value from the current step and get both values (current and
	 * previous). If either can't be found, skip time step. The time step
	 * is determined from the parameter (for example RR-3-H --> time step
	 * is three hours).
	 *
	 * 2) When calculating split_sum rate, get the first data that's
	 * earlier or same than current time step and the next data that's
	 * later or same than the current time step. Then calculate rate
	 * based on those values.
	 */

	time_duration paramStep = myTargetInfo->Param().Aggregation().TimeDuration();
	int step = static_cast<int>(itsConfiguration->ForecastStep().Hours());

	if (pd.isRate)
	{
		auto infos =
		    GetSourceDataForRate(myTargetInfo, (paramStep.Empty() ? step : static_cast<int>(paramStep.Hours())));

		prevSumInfo = infos.first;
		curSumInfo = infos.second;

		// Set aggregation period, now that we know what it is
		// (for radiation that is)
		auto& agg = par.Aggregation();

		if (prevSumInfo && curSumInfo && agg.TimeDuration().Empty() && agg.Type() != kUnknownAggregationType)
		{
			const auto td = curSumInfo->Time().ValidDateTime() - prevSumInfo->Time().ValidDateTime();
			agg.TimeDuration(td);
		}
	}
	else
	{
		// Fetch data for previous step
		// time_duration paramStep = myTargetInfo->Param().Aggregation().TimeDuration();

		if (paramStep.Empty())
		{
			myThreadedLogger.Error(fmt::format(
			    "Parameter {} is an accumulation with no time period set, unable to continue", myParamName));
			return;
		}

		forecast_time prevTimeStep = myTargetInfo->Time();
		prevTimeStep.ValidDateTime() -= paramStep;
		prevSumInfo = FetchSourceData(myTargetInfo, prevTimeStep);

		curSumInfo = FetchSourceData(myTargetInfo, myTargetInfo->Time());
	}

	if (!prevSumInfo || !curSumInfo)
	{
		// Data was not found

		myThreadedLogger.Warning(fmt::format("Data not found: not calculating {} for step {}", myParamName,
		                                     static_cast<std::string>(myTargetInfo->Time().Step())));
		return;
	}

	myThreadedLogger.Trace("Previous data step is " + static_cast<std::string>(prevSumInfo->Time().Step()));
	myThreadedLogger.Trace("Current/next data step is " + static_cast<std::string>(curSumInfo->Time().Step()));

	// EC gives precipitation in meters, we are calculating millimeters

	if (curSumInfo->Param().Unit() == kM)
	{
		// HIMAN-98
		pd.scale = 1000.;
	}

	std::string deviceType = "CPU";

	// By default values are scaled (divided) over the time period in question (in hours)
	step = static_cast<int>(curSumInfo->Time().Step().Hours() - prevSumInfo->Time().Step().Hours());

	step = 1;  // NO rate

	if (pd.isRate)
	{
		const time_duration td = curSumInfo->Time().Step() - prevSumInfo->Time().Step();
		switch (pd.rateResolution)
		{
			case kHourResolution:
				step = static_cast<int>(td.Hours());
				break;
			case kMinuteResolution:
				step = static_cast<int>(td.Minutes());
				break;
			case kSecondResolution:
				step = static_cast<int>(td.Seconds());
				break;
			default:
				myThreadedLogger.Fatal(
				    fmt::format("Unsupported rate resolution: {}", HPTimeResolutionToString.at(pd.rateResolution)));
				Abort();
		}
	}

	const double invstep = 1. / step;

	auto& resultVec = VEC(myTargetInfo);

	for (auto&& tup : zip_range(resultVec, VEC(curSumInfo), VEC(prevSumInfo)))
	{
		double& result = tup.get<0>();
		const double currentSum = tup.get<1>();
		const double previousSum = tup.get<2>();

		result = fmax(pd.lowerLimit, ((currentSum - previousSum) * invstep * pd.scale));

		// STU-13786: remove precipitations smaller than 0.01mm/h for MEPS/MEPS_preop/MNWC/MNWC_preop
		if (result < pd.truncateSmallerValues)
		{
			result = 0.0;
		}
	}

	myThreadedLogger.Info(fmt::format("[{}] Parameter: {} missing values: {}/{}", deviceType, myParamName,
	                                  myTargetInfo->Data().MissingCount(), myTargetInfo->Data().Size()));
}

std::pair<std::shared_ptr<info<double>>, std::shared_ptr<info<double>>> split_sum::GetSourceDataForRate(
    std::shared_ptr<info<double>> myTargetInfo, int step) const
{
	std::shared_ptr<info<double>> prevInfo, curInfo;

	// 1. Assuming we *know* what the step is, fetch previous and current
	// based on that step.

	if (step != kHPMissingInt)
	{
		if (myTargetInfo->Producer().Id() == 210 || myTargetInfo->Producer().Id() == 270)
		{
			step = 1;  // Forecast step is 15 (Harmonie), but it has been agreed
			           // with AKS that we'll use one hour since editor displays
			           // only hourly data.
		}

		forecast_time wantedTimeStep(myTargetInfo->Time());
		wantedTimeStep.ValidDateTime().Adjust(kHourResolution, -step);

		if (wantedTimeStep.Step().Hours() >= 0)
		{
			prevInfo = FetchSourceData(myTargetInfo, wantedTimeStep);
		}
	}

	curInfo = FetchSourceData(myTargetInfo, myTargetInfo->Time());

	if (curInfo && prevInfo)
	{
		return make_pair(prevInfo, curInfo);
	}

	// 2. Data was not found on the requested steps. Now we have to scan the database
	// for data which is slow.

	int maxSteps = 6;  // by default look for 6 hours forward or backward
	step = 1;          // by default the difference between time steps is one (ie. one hour))

	itsLogger.Trace("Target time is " + static_cast<std::string>(myTargetInfo->Time().ValidDateTime()));

	if (!prevInfo)
	{
		itsLogger.Trace("Searching for previous data");

		// start going backwards in time and search for the
		// first data that exists

		forecast_time wantedTimeStep(myTargetInfo->Time());

		for (int i = 0; !prevInfo && i <= maxSteps * step; i++)
		{
			wantedTimeStep.ValidDateTime().Adjust(kHourResolution, -step);

			if (wantedTimeStep.Step().Minutes() < 0)
			{
				continue;
			}

			itsLogger.Trace("Trying time " + static_cast<std::string>(wantedTimeStep.ValidDateTime()));
			prevInfo = FetchSourceData(myTargetInfo, wantedTimeStep);

			if (prevInfo)
			{
				itsLogger.Debug("Found previous data");
			}
		}

		if (!prevInfo)
		{
			itsLogger.Error("Previous data not found");
			return make_pair(prevInfo, curInfo);
		}
	}

	if (!curInfo)
	{
		itsLogger.Trace("Searching for next data");

		// start going forwards in time and search for the
		// first data that exists

		forecast_time wantedTimeStep(myTargetInfo->Time());

		for (int i = 0; !curInfo && i <= maxSteps * step; i++)
		{
			wantedTimeStep.ValidDateTime().Adjust(kHourResolution, step);

			itsLogger.Trace("Trying time " + static_cast<std::string>(wantedTimeStep.ValidDateTime()));
			curInfo = FetchSourceData(myTargetInfo, wantedTimeStep);

			if (curInfo)
			{
				itsLogger.Debug("Found current data");
			}
		}
	}

	return make_pair(prevInfo, curInfo);
}

std::shared_ptr<info<double>> split_sum::FetchSourceData(std::shared_ptr<info<double>> myTargetInfo,
                                                         const forecast_time& wantedTime) const
{
	level wantedLevel = myTargetInfo->Level();

	auto params = sourceParameters[myTargetInfo->Param().Name()];

	if (params.empty())
	{
		itsLogger.Fatal("Source parameter for " + myTargetInfo->Param().Name() + " not found");
		Abort();
	}

	if (myTargetInfo->Param().Name() == "RTOPLW-WM2")
	{
		wantedLevel = level(kTopOfAtmosphere, 0, "TOP");
	}

	for (auto& p : params)
	{
		auto& a = p.Aggregation();
		if (a.Type() != kUnknownAggregationType && a.TimeDuration() == time_duration())
		{
			a.TimeDuration(wantedTime.Step());
		}
	}

	auto SumInfo = Fetch(wantedTime, wantedLevel, params, myTargetInfo->ForecastType());

	// If model does not provide data for timestep 0, emulate it
	// by providing a zero-grid

	if (!SumInfo && wantedTime.Step().Minutes() == 0)
	{
		SumInfo = std::make_shared<info<double>>(*myTargetInfo);
		std::vector<forecast_time> times = {wantedTime};
		std::vector<level> levels = {wantedLevel};
		params = {sourceParameters[myTargetInfo->Param().Name()][0]};

		SumInfo->Set<param>(params);
		SumInfo->Set<level>(levels);
		SumInfo->Set<forecast_time>(times);

		SumInfo->Create(myTargetInfo->Base());
		SumInfo->Data().Fill(0);
	}

	return SumInfo;
}
