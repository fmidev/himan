#include <math.h>

#include "forecast_time.h"
#include "info.h"
#include "level.h"
#include "logger.h"
#include "matrix.h"
#include "numerical_functions.h"
#include "plugin_factory.h"
#include "pot_gfs.h"

#include "fetcher.h"

using namespace std;
using namespace himan::plugin;

/*
 *
 * class definitions for time_series
 *
 * */

time_series::time_series(param theParam, size_t expectedSize) : itsParam(theParam)
{
	itsInfos.reserve(expectedSize);
}
void time_series::Fetch(std::shared_ptr<const plugin_configuration> config, forecast_time startTime,
                        const HPTimeResolution& timeSpan, int stepSize, int numSteps, const level& forecastLevel,
                        const forecast_type& requestedType = forecast_type(kDeterministic), bool readPackedData = false)
{
	auto f = GET_PLUGIN(fetcher);

	itsInfos.clear();

	for (int i = 0; i < numSteps; ++i)
	{
		try
		{
			auto info = f->Fetch(config, startTime, forecastLevel, itsParam, requestedType);

			startTime.ValidDateTime().Adjust(timeSpan, stepSize);

			itsInfos.push_back(info);
		}
		catch (HPExceptionType& e)
		{
			if (e != kFileDataNotFound)
			{
				himan::Abort();
			}
			else
			{
				startTime.ValidDateTime().Adjust(timeSpan, stepSize);
			}
		}
	}
}

/*
 *
 * function definitions for "modifier" functions
 *
 * These functions are contained in the POT plugin preliminarily until time_series/generator functionality
 * is implemented to himan-lib. Thus these functions are now written in a more generic way then required
 * for this particular case. Functions expect input_iterators as arguments with iterators pointing to the
 * half-closed interval [begin, end), i.e. end is not included in the interval.
 *
 * */

template <class InputIt>
std::shared_ptr<himan::info<double>> Max(InputIt begin, InputIt end)
{
	// Empty series
	if (begin == end)
		return nullptr;

	// Find first field that contains data
	while (*begin == nullptr)
	{
		++begin;
		if (begin == end)
			return nullptr;
	}

	// Set first field as first set of maximum values
	auto maxInfo = make_shared<himan::info<double>>(**begin);
	maxInfo->Base(make_shared<himan::base<double>>(std::shared_ptr<himan::grid>(maxInfo->Grid()->Clone()),
	                                               himan::matrix<double>(maxInfo->Data())));

	++begin;

	for (; begin != end; ++begin)
	{
		// Empty info instance, skip
		if (*begin == nullptr)
			continue;

		// An explicit way to write the zip_range, avoiding the tuples
		auto input = VEC((*begin)).begin();
		auto maximum = VEC(maxInfo).begin();

		auto inputEnd = VEC((*begin)).end();
		auto maximumEnd = VEC(maxInfo).end();

		for (; input != inputEnd, maximum != maximumEnd; ++input, ++maximum)
		{
			*maximum = std::max(*input, *maximum);
		}
	}

	return maxInfo;
}

template <class InputIt>
std::shared_ptr<himan::info<double>> Mean(InputIt begin, InputIt end)
{
	if (begin == end)
		return nullptr;

	// Find first field that contains data
	while (*begin == nullptr)
	{
		++begin;
		if (begin == end)
			return nullptr;
	}

	// Set first field as first set of mean values
	auto meanInfo = make_shared<himan::info<double>>(**begin);
	meanInfo->Base(make_shared<himan::base<double>>(std::shared_ptr<himan::grid>(meanInfo->Grid()->Clone()),
	                                                himan::matrix<double>(meanInfo->Data())));

	++begin;

	size_t count = 1;

	for (; begin != end; ++begin)
	{
		// Empty info instance, skip
		if (*begin == nullptr)
			continue;

		// An explicit way to write the zip_range, avoiding the tuples
		auto input = VEC((*begin)).begin();
		auto sum = VEC(meanInfo).begin();

		auto inputEnd = VEC((*begin)).end();
		auto sumEnd = VEC(meanInfo).end();

		for (; input != inputEnd, sum != sumEnd; ++input, ++sum)
		{
			*sum += *input;
		}
		++count;
	}

	// Calculate actual mean values
	double countInv = 1 / static_cast<double>(count);

	for (auto&& val : VEC(meanInfo))
	{
		val *= countInv;
	}

	return meanInfo;
}

/*
 *  plug-in definitions
 *
 * */

pot_gfs::pot_gfs() : itsStrictMode(false)
{
	itsLogger = logger("pot_gfs");
}
void pot_gfs::Process(std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	param POT("POT-PRCNT", aggregation(), processing_type(kProbability));

	POT.Unit(kPrcnt);

	if (itsConfiguration->GetValue("strict") == "true")
	{
		itsStrictMode = true;
	}

	SetParams({POT});

	Start();
}

void pot_gfs::Calculate(std::shared_ptr<info<double>> myTargetInfo, unsigned short threadIndex)
{
	/*
	 * Required source parameters
	 */

	const param CapeParamEC("CAPE-JKG");
	const param CapeParamHiman("CAPE1040-JKG");
	const level CapeLevelHiman(kMaximumThetaE, 0);
	const param RainParam("RRR-KGM2");

	// Step from previous leadtime, taken from configuration file
	int step = static_cast<int>(itsConfiguration->ForecastStep().Hours());

	if (step == kHPMissingInt)
	{
		// himan was mabe started with configuration option "hours"
		// so step is not readily available

		if (myTargetInfo->Size<forecast_time>() > 1)
		{
			// More than one time is calculated - check the difference to previous
			// or next time

			int leadtime = static_cast<int>(myTargetInfo->Time().Step().Hours());
			int otherLeadtime;

			if (myTargetInfo->Previous<forecast_time>())
			{
				otherLeadtime = static_cast<int>(myTargetInfo->Time().Step().Hours());
				myTargetInfo->Next<forecast_time>();  // return
			}
			else
			{
				myTargetInfo->Next<forecast_time>();
				otherLeadtime = static_cast<int>(myTargetInfo->Time().Step().Hours());
				myTargetInfo->Previous<forecast_time>();  // return
			}

			step = abs(otherLeadtime - leadtime);
		}
		else
		{
			// default
			step = 1;
		}
	}

	forecast_type forecastType = myTargetInfo->ForecastType();
	forecast_time forecastTime = myTargetInfo->Time();
	forecast_time forecastTimeNext = myTargetInfo->Time();
	forecastTimeNext.ValidDateTime().Adjust(kHourResolution, +step);
	level forecastLevel = myTargetInfo->Level();

	auto myThreadedLogger = logger("pot_gfs_pluginThread #" + to_string(threadIndex));

	myThreadedLogger.Debug("Calculating time " + static_cast<string>(forecastTime.ValidDateTime()) + " level " +
	                       static_cast<string>(forecastLevel));

	time_series CAPEts(CapeParamHiman, 4), RRts(RainParam, 3);

	forecast_time startTime = myTargetInfo->Time();
	startTime.ValidDateTime().Adjust(kHourResolution, -2);

	// create time series CAPE
	CAPEts.Fetch(itsConfiguration, startTime, kHourResolution, 1, 4, CapeLevelHiman, forecastType);

	if (CAPEts.Size() == 0)
	{
		if (itsStrictMode)
		{
			myThreadedLogger.Info("Cold cape not found, skipping step " + static_cast<string>(forecastTime.Step()));
			return;
		}

		CAPEts.Param(CapeParamEC);
		CAPEts.Fetch(itsConfiguration, startTime, kHourResolution, 1, 4, forecastLevel, forecastType);
	}

	// find max
	std::shared_ptr<info<double>> CAPEMaxInfo;
	CAPEMaxInfo = Max(CAPEts.begin(), CAPEts.end());

	// create time series RR
	startTime.ValidDateTime().Adjust(kHourResolution, 1);
	RRts.Fetch(itsConfiguration, startTime, kHourResolution, 1, 3, forecastLevel, forecastType);

	// fnd mean
	std::shared_ptr<info<double>> RRMeanInfo;
	RRMeanInfo = Mean(RRts.begin(), RRts.end());

	if (!CAPEMaxInfo || !RRMeanInfo)
	{
		myThreadedLogger.Info("Missing source data. Skipping step " + static_cast<string>(forecastTime.Step()));
		return;
	}

	// filter CAPE
	himan::matrix<double> filter_kernel(3, 3, 1, MissingDouble(), 1.0 / 9.0);
	himan::matrix<double> filtered_CAPE =
	    numerical_functions::Filter2D<double>(CAPEMaxInfo->Data(), filter_kernel, itsConfiguration->UseCuda());

	auto b = CAPEMaxInfo->Base();
	b->data = move(filtered_CAPE);

	// filter RR
	himan::matrix<double> filtered_RR =
	    numerical_functions::Filter2D<double>(RRMeanInfo->Data(), filter_kernel, itsConfiguration->UseCuda());
	b = RRMeanInfo->Base();
	b->data = move(filtered_RR);

	string deviceType = "CPU";

	// starting point of the algorithm POT v2.1
	for (auto&& tup : zip_range(VEC(myTargetInfo), VEC(CAPEMaxInfo), VEC(RRMeanInfo)))
	{
		double& POT = tup.get<0>();
		double CAPE = tup.get<1>();
		double RR = tup.get<2>();

		double PoLift = 0;       // Probability of Lift (-->initiation)
		double PoThermoDyn = 0;  // Probability of ThermoDynamics

		double verticalVelocity = sqrt(2 * CAPE);

		PoThermoDyn = 0.03333 * verticalVelocity -
		              0.16667;  // Probability grows 0->1, if vertical velocity "sqrt(2*)" grows 5->35 m/s
		PoThermoDyn = max(0.0, min(PoThermoDyn, 1.0));

		if (RR >= 0.05 && RR <= 5)
		{
			PoLift = 0.217147241 * log(RR) + 0.650514998;  // Function grows fast logarithmically from zero to one
		}
		else if (RR > 5)
		{
			PoLift = 1;
		}

		POT = PoLift * PoThermoDyn * 100;
	}

	myThreadedLogger.Info("[" + deviceType + "] Missing values: " + to_string(myTargetInfo->Data().MissingCount()) +
	                      "/" + to_string(myTargetInfo->Data().Size()));
}
