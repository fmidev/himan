#include "split_sum.h"
#include "forecast_time.h"
#include "level.h"
#include "logger.h"
#include "plugin_factory.h"
#include <iostream>
#include <map>
#include <thread>

#include "radon.h"
#include "writer.h"

using namespace std;
using namespace himan::plugin;

const int SUB_THREAD_COUNT = 5;

map<string, himan::params> sourceParameters;

split_sum::split_sum()
{
	itsLogger = logger("split_sum");

	// Define source parameters for each output parameter

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
	sourceParameters["RADGLO-WM2"] = {param("RADGLOA-JM2"), param("RADGLO-WM2")};
	sourceParameters["RADLW-WM2"] = {param("RADLWA-JM2"), param("RADLW-WM2")};
	sourceParameters["RTOPLW-WM2"] = {param("RTOPLWA-JM2"), param("RTOPLW-WM2")};
	sourceParameters["RNETLW-WM2"] = {param("RNETLWA-JM2"), param("RNETLW-WM2")};
	sourceParameters["RADSW-WM2"] = {param("RADSWA-JM2")};
	sourceParameters["RNETSW-WM2"] = {param("RNETSWA-JM2"), param("RNETSW-WM2")};
	sourceParameters["RADGLOC-WM2"] = {param("RADGLOCA-JM2")};
	sourceParameters["RADLWC-WM2"] = {param("RADLWCA-JM2")};
}

void split_sum::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	/*
	 * Set target parameter to split_sum.
	 *
	 * We need to specify grib and querydata parameter information
	 * since we don't know which one will be the output format.
	 *
	 */

	vector<param> params;

	if (itsConfiguration->Exists("rr1h") && itsConfiguration->GetValue("rr1h") == "true")
	{
		params.emplace_back("RR-1-MM", aggregation(kAccumulation, ONE_HOUR), processing_type());
	}

	if (itsConfiguration->Exists("rr3h") && itsConfiguration->GetValue("rr3h") == "true")
	{
		params.emplace_back("RR-3-MM", aggregation(kAccumulation, THREE_HOURS), processing_type());
	}

	if (itsConfiguration->Exists("rr6h") && itsConfiguration->GetValue("rr6h") == "true")
	{
		params.emplace_back("RR-6-MM", aggregation(kAccumulation, SIX_HOURS), processing_type());
	}

	if (itsConfiguration->Exists("rr12h") && itsConfiguration->GetValue("rr12h") == "true")
	{
		params.emplace_back("RR-12-MM", aggregation(kAccumulation, TWELVE_HOURS), processing_type());
	}

	if (itsConfiguration->Exists("rr24h") && itsConfiguration->GetValue("rr24h") == "true")
	{
		params.emplace_back("RR-24-MM", aggregation(kAccumulation, time_duration("24:00")), processing_type());
	}

	if (itsConfiguration->Exists("sn3h") && itsConfiguration->GetValue("sn3h") == "true")
	{
		params.emplace_back("SN-3-MM", aggregation(kAccumulation, THREE_HOURS), processing_type());
	}

	if (itsConfiguration->Exists("sn6h") && itsConfiguration->GetValue("sn6h") == "true")
	{
		params.emplace_back("SN-6-MM", aggregation(kAccumulation, SIX_HOURS), processing_type());
	}

	if (itsConfiguration->Exists("sn24h") && itsConfiguration->GetValue("sn24h") == "true")
	{
		params.emplace_back("SN-24-MM", aggregation(kAccumulation, time_duration("24:00")), processing_type());
	}

	if (itsConfiguration->Exists("sn120h") && itsConfiguration->GetValue("sn120h") == "true")
	{
		params.emplace_back("SN-120-MM", aggregation(kAccumulation, time_duration("120:00")), processing_type());
	}

	if (itsConfiguration->Exists("rrc3h") && itsConfiguration->GetValue("rrc3h") == "true")
	{
		params.emplace_back("RRC-3-MM", aggregation(kAccumulation, THREE_HOURS), processing_type());
	}

	if (itsConfiguration->Exists("rrr") && itsConfiguration->GetValue("rrr") == "true")
	{
		params.emplace_back("RRR-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type());
	}

	if (itsConfiguration->Exists("rrrc") && itsConfiguration->GetValue("rrrc") == "true")
	{
		params.emplace_back("RRRC-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type());
	}

	if (itsConfiguration->Exists("rrrl") && itsConfiguration->GetValue("rrrl") == "true")
	{
		params.emplace_back("RRRL-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type());
	}

	// Graupel

	if (itsConfiguration->Exists("grr") && itsConfiguration->GetValue("grr") == "true")
	{
		params.emplace_back("GRR-MMH", aggregation(kAccumulation, ONE_HOUR), processing_type());
	}

	// Solid

	if (itsConfiguration->Exists("rrrs") && itsConfiguration->GetValue("rrrs") == "true")
	{
		params.emplace_back("RRRS-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type());
	}

	if (itsConfiguration->Exists("rrs3h") && itsConfiguration->GetValue("rrs3h") == "true")
	{
		params.emplace_back("RRS-3-MM", aggregation(kAccumulation, THREE_HOURS), processing_type());
	}

	if (itsConfiguration->Exists("rrs24h") && itsConfiguration->GetValue("rrs24h") == "true")
	{
		params.emplace_back("RRS-24-MM", aggregation(kAccumulation, time_duration("24:00")), processing_type());
	}

	// Snow

	if (itsConfiguration->Exists("snr") && itsConfiguration->GetValue("snr") == "true")
	{
		params.emplace_back("SNR-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type());
	}

	if (itsConfiguration->Exists("snrc") && itsConfiguration->GetValue("snrc") == "true")
	{
		params.emplace_back("SNRC-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type());
	}

	if (itsConfiguration->Exists("snrl") && itsConfiguration->GetValue("snrl") == "true")
	{
		params.emplace_back("SNRL-KGM2", aggregation(kAccumulation, ONE_HOUR), processing_type());
	}

	// Radiation
	// These are RATES not SUMS

	if (itsConfiguration->Exists("glob") && itsConfiguration->GetValue("glob") == "true")
	{
		params.emplace_back("RADGLO-WM2", aggregation(kAverage), processing_type());
	}

	if (itsConfiguration->Exists("globc") && itsConfiguration->GetValue("globc") == "true")
	{
		params.emplace_back("RADGLOC-WM2", aggregation(kAverage), processing_type());
	}

	if (itsConfiguration->Exists("lw") && itsConfiguration->GetValue("lw") == "true")
	{
		params.emplace_back("RADLW-WM2", aggregation(kAverage), processing_type());
	}

	if (itsConfiguration->Exists("lwc") && itsConfiguration->GetValue("lwc") == "true")
	{
		params.emplace_back("RADLWC-WM2", aggregation(kAverage), processing_type());
	}

	if (itsConfiguration->Exists("toplw") && itsConfiguration->GetValue("toplw") == "true")
	{
		// Same grib2 parameter definition as with RADLW-WM2, this is just on
		// another surface
		params.emplace_back("RTOPLW-WM2", aggregation(kAverage), processing_type());
	}

	if (itsConfiguration->Exists("netlw") && itsConfiguration->GetValue("netlw") == "true")
	{
		params.emplace_back("RNETLW-WM2", aggregation(kAverage), processing_type());
	}

	if (itsConfiguration->Exists("sw") && itsConfiguration->GetValue("sw") == "true")
	{
		params.emplace_back("RADSW-WM2", aggregation(kAverage), processing_type());
	}

	if (itsConfiguration->Exists("netsw") && itsConfiguration->GetValue("netsw") == "true")
	{
		params.emplace_back("RNETSW-WM2", aggregation(kAverage), processing_type());
	}

	if (params.empty())
	{
		itsLogger.Error("No parameter definition given, abort");
		return;
	}

	SetParams(params);

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

void split_sum::Calculate(shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	vector<thread*> threads;
	vector<shared_ptr<info<double>>> infos;

	int subThreadIndex = 0;

	for (myTargetInfo->Reset<param>(); myTargetInfo->Next<param>(); ++subThreadIndex)
	{
		auto newInfo = make_shared<info<double>>(*myTargetInfo);

		infos.push_back(newInfo);  // extend lifetime over this loop

		// Pass param by reference to sub-thread: aggregation duration is only available there
		// when data is actually fetched, and we need that information when writing data to disk

		threads.push_back(new thread(&split_sum::DoParam, this, newInfo, std::ref(myTargetInfo->Param()),
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

void split_sum::DoParam(shared_ptr<info<double>> myTargetInfo, param& par, string subThreadIndex) const
{
	ASSERT(myTargetInfo);

	const std::string myParamName = myTargetInfo->Param().Name();

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	auto myThreadedLogger = logger("splitSumSubThread#" + subThreadIndex);

	myThreadedLogger.Info("Calculating parameter " + myParamName + " time " +
	                      static_cast<string>(myTargetInfo->Time().ValidDateTime()) + " level " +
	                      static_cast<string>(forecastLevel));

	const bool isRadiationCalculation =
	    (myParamName == "RADGLO-WM2" || myParamName == "RADGLOC-WM2" || myParamName == "RADLW-WM2" ||
	     myParamName == "RADLWC-WM2" || myParamName == "RTOPLW-WM2" || myParamName == "RNETLW-WM2" ||
	     myParamName == "RADSW-WM2" || myParamName == "RNETSW-WM2");

	const bool isRateCalculation =
	    (isRadiationCalculation || myParamName == "RRR-KGM2" || myParamName == "RRRL-KGM2" ||
	     myParamName == "RRRC-KGM2" || myParamName == "SNR-KGM2" || myParamName == "SNRC-KGM2" ||
	     myParamName == "SNRL-KGM2" || myParamName == "GRR-MMH" || myParamName == "RRRS-KGM2" ||
	     myParamName == "RRC-KGM2");
	const bool isPrecipitationCalculation =
	    myParamName == "RR-1-MM" || myParamName == "RR-3-MM" || myParamName == "RR-6-MM" || myParamName == "RR-12-MM" ||
	    myParamName == "RR-24-MM" || myParamName == "RRC-3-MM" || myParamName == "RRR-KGM2" ||
	    myParamName == "RRRC-KGM2" || myParamName == "RRRL-KGM2";
	const long producerId = myTargetInfo->Producer().Id();

	// Have to re-fetch infos each time since we might have to change element
	// from liquid to snow to radiation so we need also different source parameters

	shared_ptr<info<double>> curSumInfo;
	shared_ptr<info<double>> prevSumInfo;

	if (myTargetInfo->Time().Step().Minutes() == 0)
	{
		// This is the first time step, calculation can not be done

		myThreadedLogger.Info(fmt::format("This is the first time step -- not calculating {} for step {}", myParamName,
		                                  static_cast<string>(forecastTime.Step())));
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

	int step = static_cast<int>(itsConfiguration->ForecastStep().Hours());

	if (isRateCalculation)
	{
		auto infos = GetSourceDataForRate(myTargetInfo, step);

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
		time_duration paramStep = myTargetInfo->Param().Aggregation().TimeDuration();

		if (paramStep.Empty())
		{
			myThreadedLogger.Error(
			    fmt::format("Parameter {} has no aggregation duration set, unable to continue", myParamName));
			return;
		}
		// Skip early steps if necessary

		if (myTargetInfo->Time().Step() >= paramStep)
		{
			forecast_time prevTimeStep = myTargetInfo->Time();

			prevTimeStep.ValidDateTime() -= paramStep;

			prevSumInfo = FetchSourceData(myTargetInfo, prevTimeStep);
		}

		// Data from current time step, but only if we have data for previous
		// step

		if (prevSumInfo)
		{
			curSumInfo = FetchSourceData(myTargetInfo, myTargetInfo->Time());
		}
	}

	if (!prevSumInfo || !curSumInfo)
	{
		// Data was not found

		myThreadedLogger.Warning(fmt::format("Data not found: not calculating {} for step {}", myParamName,
		                                     static_cast<string>(myTargetInfo->Time().Step())));
		return;
	}

	myThreadedLogger.Trace("Previous data step is " + static_cast<string>(prevSumInfo->Time().Step()));
	myThreadedLogger.Trace("Current/next data step is " + static_cast<string>(curSumInfo->Time().Step()));

	double scaleFactor = 1.;

	// EC gives precipitation in meters, we are calculating millimeters

	if (curSumInfo->Param().Unit() == kM ||
	    ((producerId == 240 || producerId == 241 || producerId == 243) && !isRadiationCalculation))  // HIMAN-98
	{
		scaleFactor = 1000.;
	}

	string deviceType = "CPU";

	step = static_cast<int>(curSumInfo->Time().Step().Hours() - prevSumInfo->Time().Step().Hours());

	if (isRadiationCalculation)
	{
		/*
		 * Radiation unit is W/m^2 which is J/m^2/s, so we need to convert
		 * time to seconds.
		 *
		 * Step is always one hour or more, even in the case of Harmonie.
		 */

		step *= 3600;
	}
	else if (!isRateCalculation)
	{
		/*
		 * If calculating for Harmonie, use hour as base time unit, or disable
		 * it if sum is calculated.
		 */

		step = 1;
	}

	double lowLimit = 0.0;  // value can't be lower than 0

	if (myParamName == "RTOPLW-WM2" || myParamName == "RNETLW-WM2" || myParamName == "RNETSW-WM2")
	{
		lowLimit = himan::MissingDouble();  // no low limit
	}

	const double invstep = 1. / step;

	auto& resultVec = VEC(myTargetInfo);

	for (auto&& tup : zip_range(resultVec, VEC(curSumInfo), VEC(prevSumInfo)))
	{
		double& result = tup.get<0>();
		const double currentSum = tup.get<1>();
		const double previousSum = tup.get<2>();

		result = fmax(lowLimit, ((currentSum - previousSum) * invstep * scaleFactor));

		// STU-13786: remove precipitations smaller than 0.01mm/h for MEPS/MEPS_preop/MNWC/MNWC_preop
		if (isPrecipitationCalculation &&
		    (producerId == 260 || producerId == 261 || (producerId >= 270 && producerId <= 272)) && result < 0.01)
		{
			result = 0.0;
		}
		ASSERT(isRadiationCalculation || result >= 0 || IsMissing(result));
	}

	myThreadedLogger.Info(fmt::format("[{}] Parameter: {} missing values: {}/{}", deviceType, myParamName,
	                                  myTargetInfo->Data().MissingCount(), myTargetInfo->Data().Size()));
}

pair<shared_ptr<himan::info<double>>, shared_ptr<himan::info<double>>> split_sum::GetSourceDataForRate(
    shared_ptr<info<double>> myTargetInfo, int step) const
{
	shared_ptr<info<double>> prevInfo, curInfo;

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

	itsLogger.Trace("Target time is " + static_cast<string>(myTargetInfo->Time().ValidDateTime()));

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

			itsLogger.Trace("Trying time " + static_cast<string>(wantedTimeStep.ValidDateTime()));
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

			itsLogger.Trace("Trying time " + static_cast<string>(wantedTimeStep.ValidDateTime()));
			curInfo = FetchSourceData(myTargetInfo, wantedTimeStep);

			if (curInfo)
			{
				itsLogger.Debug("Found current data");
			}
		}
	}

	return make_pair(prevInfo, curInfo);
}

shared_ptr<himan::info<double>> split_sum::FetchSourceData(shared_ptr<info<double>> myTargetInfo,
                                                           const forecast_time& wantedTime) const
{
	level wantedLevel = myTargetInfo->Level();

	auto params = sourceParameters[myTargetInfo->Param().Name()];

	if (params.empty())
	{
		itsLogger.Fatal("Source parameter for " + myTargetInfo->Param().Name() + " not found");
		himan::Abort();
	}

	if (myTargetInfo->Param().Name() == "RTOPLW-WM2")
	{
		wantedLevel = level(kTopOfAtmosphere, 0, "TOP");
	}

	shared_ptr<info<double>> SumInfo = Fetch(wantedTime, wantedLevel, params, myTargetInfo->ForecastType());

	// If model does not provide data for timestep 0, emulate it
	// by providing a zero-grid

	if (!SumInfo && wantedTime.Step().Minutes() == 0)
	{
		SumInfo = make_shared<info<double>>(*myTargetInfo);
		vector<forecast_time> times = {wantedTime};
		vector<level> levels = {wantedLevel};
		params = {sourceParameters[myTargetInfo->Param().Name()][0]};

		SumInfo->Set<param>(params);
		SumInfo->Set<level>(levels);
		SumInfo->Set<forecast_time>(times);

		SumInfo->Create(myTargetInfo->Base());
		SumInfo->Data().Fill(0);
	}

	return SumInfo;
}
