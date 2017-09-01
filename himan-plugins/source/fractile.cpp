/**
 * @file fractile.cpp
 *
 **/
#include "fractile.h"

#include <algorithm>
#include <boost/algorithm/string.hpp>

#include "logger.h"
#include "plugin_factory.h"

#include "ensemble.h"
#include "lagged_ensemble.h"
#include "time_ensemble.h"

#include "fetcher.h"
#include "radon.h"

#include "util.h"

namespace himan
{
namespace plugin
{
fractile::fractile()
    : itsEnsembleSize(0),
      itsEnsembleType(kPerturbedEnsemble),
      itsFractiles({0., 10., 25., 50., 75., 90., 100.}),
      itsLag(0),
      itsLaggedSteps(0),
      itsMaximumMissingForecasts(0)
{
	itsCudaEnabledCalculation = false;
	itsLogger = logger("fractile");
}

fractile::~fractile() {}
void fractile::Process(const std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	if (!itsConfiguration->GetValue("param").empty())
	{
		itsParamName = itsConfiguration->GetValue("param");
	}
	else
	{
		itsLogger.Error("Param not specified");
		return;
	}

	auto ensType = itsConfiguration->GetValue("ensemble_type");

	if (!ensType.empty())
	{
		itsEnsembleType = HPStringToEnsembleType.at(ensType);
	}

	auto ensSize = itsConfiguration->GetValue("ensemble_size");

	if (!ensSize.empty())
	{
		itsEnsembleSize = boost::lexical_cast<int>(ensSize);
	}

	auto maximumMissing = itsConfiguration->GetValue("max_missing_forecasts");

	if (!maximumMissing.empty())
	{
		itsMaximumMissingForecasts = boost::lexical_cast<int>(maximumMissing);
	}

	if (itsEnsembleType == kLaggedEnsemble)
	{
		if (itsConfiguration->Exists("lag"))
		{
			int lag = std::stoi(itsConfiguration->GetValue("lag"));
			if (lag == 0)
			{
				throw std::runtime_error(ClassName() + ": specify lag < 0");
			}
			else if (lag > 0)
			{
				itsLogger.Warning("negating lag value " + std::to_string(-lag));
				lag = -lag;
			}

			itsLag = lag;
		}
		else
		{
			throw std::runtime_error(ClassName() + ": specify lag value for lagged_ensemble");
		}

		// How many lagged steps to include in the calculation
		if (itsConfiguration->Exists("lagged_steps"))
		{
			const int steps = std::stoi(itsConfiguration->GetValue("lagged_steps"));
			if (steps <= 0)
			{
				throw std::runtime_error(ClassName() + ": invalid lagged_steps value. Allowed range >= 0");
			}
			itsLaggedSteps = steps + 1;
		}
		else
		{
			throw std::runtime_error(ClassName() + ": specify lagged_steps when using time lagging ('lag')");
		}
	}

	if (itsEnsembleSize == 0 && (itsEnsembleType == kPerturbedEnsemble || itsEnsembleType == kLaggedEnsemble))
	{
		// Regular ensemble size is static, get it from database if user
		// hasn't specified any size

		auto r = GET_PLUGIN(radon);

		std::string ensembleSizeStr =
		    r->RadonDB().GetProducerMetaData(itsConfiguration->SourceProducer(0).Id(), "ensemble size");

		if (ensembleSizeStr.empty())
		{
			itsLogger.Error("Unable to find ensemble size from database");
			return;
		}

		itsEnsembleSize = boost::lexical_cast<int>(ensembleSizeStr);
	}

	auto fractiles = itsConfiguration->GetValue("fractiles");

	if (!fractiles.empty())
	{
		itsFractiles.clear();

		auto list = util::Split(fractiles, ",", false);

		for (std::string& val : list)
		{
			boost::trim(val);
			try
			{
				itsFractiles.push_back(boost::lexical_cast<double>(val));
			}
			catch (const boost::bad_lexical_cast& e)
			{
				itsLogger.Fatal("Invalid fractile value: '" + val + "'");
				abort();
			}
		}
	}

	params calculatedParams;

	for (double fractile : itsFractiles)
	{
		auto name = "F" + boost::lexical_cast<std::string>(fractile) + "-" + itsParamName;
		calculatedParams.push_back(param(name));
	}

	auto name = util::Split(itsParamName, "-", false);
	calculatedParams.push_back(param(name[0] + "-MEAN-" + name[1]));    // mean
	calculatedParams.push_back(param(name[0] + "-STDDEV-" + name[1]));  // standard deviation

	SetParams(calculatedParams);

	Start();
}

void fractile::Calculate(std::shared_ptr<info> myTargetInfo, uint16_t threadIndex)
{
	const std::string deviceType = "CPU";

	auto threadedLogger = logger("fractileThread # " + boost::lexical_cast<std::string>(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	threadedLogger.Info("Calculating time " + static_cast<std::string>(forecastTime.ValidDateTime()) + " level " +
	                    static_cast<std::string>(forecastLevel));

	std::unique_ptr<ensemble> ens;

	switch (itsEnsembleType)
	{
		case kPerturbedEnsemble:
			ens = std::unique_ptr<ensemble>(new ensemble(param(itsParamName), itsEnsembleSize));
			break;
		case kTimeEnsemble:
			ens = std::unique_ptr<time_ensemble>(new time_ensemble(param(itsParamName), itsEnsembleSize));
			break;
		case kLaggedEnsemble:
			ens = std::unique_ptr<lagged_ensemble>(
			    new lagged_ensemble(param(itsParamName), itsEnsembleSize, kHourResolution, itsLag, itsLaggedSteps));
			break;
		default:
			itsLogger.Fatal("Unknown ensemble type: " + HPEnsembleTypeToString.at(itsEnsembleType));
			abort();
	}

	ens->MaximumMissingForecasts(itsMaximumMissingForecasts);

	try
	{
		ens->Fetch(itsConfiguration, forecastTime, forecastLevel);
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			itsLogger.Error("Failed to find ensemble data");
			return;
		}
	}

	myTargetInfo->ResetLocation();
	ens->ResetLocation();

	while (myTargetInfo->NextLocation() && ens->NextLocation())
	{
		auto sortedValues = ens->SortedValues();
		const size_t ensembleSize = sortedValues.size();

		// Skip this step if we didn't get any valid fields
		if (ensembleSize == 0)
		{
			continue;
		}
		// sortedValues needs to have one element at the back for correct array indexing
		// NOTE: `ensembleSize` stays the same
		else
		{
			sortedValues.push_back(0.0);
		}

		assert(!itsFractiles.empty());

		size_t targetInfoIndex = 0;
		for (auto P : itsFractiles)
		{
			// use the linear interpolation between closest ranks method recommended by NIST
			// http://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm
			double x;

			// check lower corner case p E [0,1/(N+1)]
			if (P / 100.0 <= 1.0 / static_cast<double>(ensembleSize + 1))
			{
				x = 1;
			}
			// check upper corner case p E [N/(N+1),1]
			else if (P / 100.0 >= static_cast<double>(ensembleSize) / static_cast<double>(ensembleSize + 1))
			{
				x = static_cast<double>(ensembleSize);
			}
			// everything that happens on the interval between
			else
			{
				x = P / 100.0 * static_cast<double>(ensembleSize + 1);
			}
			// floor x explicitly before casting to int
			int i = static_cast<int>(std::floor(x));

			myTargetInfo->ParamIndex(targetInfoIndex);

			myTargetInfo->Value(sortedValues[i - 1] + std::fmod(x, 1.0) * (sortedValues[i] - sortedValues[i - 1]));
			++targetInfoIndex;
		}

		double mean = ens->Mean();
		if (!std::isfinite(mean))
		{
			mean = MissingDouble();
		}

		double var = std::sqrt(ens->Variance());
		if (!std::isfinite(var))
		{
			var = MissingDouble();
		}

		myTargetInfo->ParamIndex(targetInfoIndex);
		myTargetInfo->Value(mean);
		++targetInfoIndex;
		myTargetInfo->ParamIndex(targetInfoIndex);
		myTargetInfo->Value(var);
	}

	threadedLogger.Info("[" + deviceType +
	                    "] Missing values: " + boost::lexical_cast<std::string>(myTargetInfo->Data().MissingCount()) +
	                    "/" + boost::lexical_cast<std::string>(myTargetInfo->Data().Size()));
}

}  // plugin

}  // namespace
