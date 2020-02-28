#include "fractile.h"

#include <algorithm>
#include <boost/algorithm/string.hpp>

#include "logger.h"
#include "plugin_factory.h"

#include "lagged_ensemble.h"
#include "time_ensemble.h"

#include "fetcher.h"
#include "radon.h"

#include "numerical_functions.h"
#include "util.h"

namespace himan
{
namespace plugin
{
fractile::fractile() : itsFractiles({0., 10., 25., 50., 75., 90., 100.})
{
	itsCudaEnabledCalculation = false;
	itsLogger = logger("fractile");
}

std::unique_ptr<ensemble> CreateEnsemble(const std::shared_ptr<const plugin_configuration> conf)
{
	logger log("fractile");

	std::string paramName = conf->GetValue("param");

	auto ensTypestr = conf->GetValue("ensemble_type");
	HPEnsembleType ensType = kPerturbedEnsemble;

	if (!ensTypestr.empty())
	{
		ensType = HPStringToEnsembleType.at(ensTypestr);
	}

	size_t ensSize = 0;

	if (conf->Exists("ensemble_size"))
	{
		ensSize = std::stoi(conf->GetValue("ensemble_size"));
	}
	else if (ensType == kPerturbedEnsemble || ensType == kLaggedEnsemble)
	{
		// Regular ensemble size is static, get it from database if user
		// hasn't specified any size

		auto r = GET_PLUGIN(radon);

		std::string ensembleSizeStr = r->RadonDB().GetProducerMetaData(conf->SourceProducer(0).Id(), "ensemble size");

		if (ensembleSizeStr.empty())
		{
			log.Error("Unable to find ensemble size from database");
			himan::Abort();
		}

		ensSize = std::stoi(ensembleSizeStr);
	}

	std::unique_ptr<ensemble> ens = nullptr;

	switch (ensType)
	{
		case kPerturbedEnsemble:
			ens = std::unique_ptr<ensemble>(new ensemble(param(paramName), ensSize));
			break;
		case kTimeEnsemble:
		{
			int secondaryLen = 0, secondaryStep = 1;
			HPTimeResolution secondarySpan = kHourResolution;

			if (conf->Exists(("secondary_time_len")))
			{
				secondaryLen = std::stoi(conf->GetValue("secondary_time_len"));
			}
			if (conf->Exists(("secondary_time_step")))
			{
				secondaryStep = std::stoi(conf->GetValue("secondary_time_step"));
			}
			if (conf->Exists(("secondary_time_span")))
			{
				secondarySpan = HPStringToTimeResolution.at(conf->GetValue("secondary_time_span"));
			}

			ens = std::unique_ptr<time_ensemble>(new time_ensemble(param(paramName), ensSize, kYearResolution,
			                                                       secondaryLen, secondaryStep, secondarySpan));
		}
		break;
		case kLaggedEnsemble:
		{
			auto name = conf->GetValue("named_ensemble");

			if (name.empty() == false)
			{
				ens = std::unique_ptr<lagged_ensemble>(new lagged_ensemble(param(paramName), name));
			}

			else
			{
				auto lagstr = conf->GetValue("lag");
				if (lagstr.empty())
				{
					log.Fatal("specify lag value for lagged_ensemble");
					himan::Abort();
				}

				int lag = std::stoi(conf->GetValue("lag"));

				if (lag == 0)
				{
					log.Fatal("lag value needs to be negative integer");
					himan::Abort();
				}
				else if (lag > 0)
				{
					log.Warning("negating lag value " + std::to_string(-lag));
					lag = -lag;
				}

				auto stepsstr = conf->GetValue("lagged_steps");

				if (stepsstr.empty())
				{
					log.Fatal("specify lagged_steps value for lagged_ensemble");
					himan::Abort();
				}

				int steps = std::stoi(conf->GetValue("lagged_steps"));

				if (steps <= 0)
				{
					log.Fatal("invalid lagged_steps value. Allowed range >= 0");
					himan::Abort();
				}

				steps++;

				ens = std::unique_ptr<lagged_ensemble>(
				    new lagged_ensemble(param(paramName), ensSize, time_duration(kHourResolution, lag), steps));
			}
		}
		break;
		default:
			log.Fatal("Unknown ensemble type: " + ensType);
			himan::Abort();
	}

	if (conf->Exists("max_missing_forecasts"))
	{
		ens->MaximumMissingForecasts(std::stoi(conf->GetValue("max_missing_forecasts")));
	}

	return std::move(ens);
}

void fractile::SetParams()
{
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
				itsFractiles.push_back(std::stof(val));
			}
			catch (const std::invalid_argument& e)
			{
				itsLogger.Fatal("Invalid fractile value: '" + val + "'");
				himan::Abort();
			}
		}
	}

	params calculatedParams;
	std::string paramName = itsConfiguration->GetValue("param");

	if (paramName.empty())
	{
		itsLogger.Fatal("param not specified");
		himan::Abort();
	}

	for (float frac : itsFractiles)
	{
		auto name = "F" + boost::lexical_cast<std::string>(frac) + "-" + paramName;
		param p(name);
		p.ProcessingType(processing_type(kFractile, frac, kHPMissingValue, kHPMissingInt));
		calculatedParams.push_back(p);
	}

	auto name = util::Split(paramName, "-", false);

	param mean(name[0] + "-MEAN-" + name[1]);
	mean.ProcessingType(processing_type(kEnsembleMean, kHPMissingInt, kHPMissingInt, kHPMissingInt));

	calculatedParams.push_back(mean);

	param stdev(name[0] + "-STDDEV-" + name[1]);
	stdev.ProcessingType(processing_type(kStandardDeviation, kHPMissingInt, kHPMissingInt, kHPMissingInt));

	calculatedParams.push_back(stdev);

	compiled_plugin_base::SetParams(calculatedParams);
}

void fractile::SetForecastType()
{
	if (itsForecastTypeIterator.Size() > 1)
	{
		itsLogger.Warning("More than one forecast type defined - fractile can only produce 'statistical processing'");
	}

	itsForecastTypeIterator = forecast_type_iter({forecast_type(kStatisticalProcessing)});
}

void fractile::Process(const std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	SetParams();
	SetForecastType();
	Start<float>();
}

void fractile::Calculate(std::shared_ptr<info<float>> myTargetInfo, uint16_t threadIndex)
{
	const std::string deviceType = "CPU";
	auto ens = CreateEnsemble(itsConfiguration);

	auto threadedLogger = logger("fractileThread # " + std::to_string(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	threadedLogger.Info("Calculating time " + static_cast<std::string>(forecastTime.ValidDateTime()) + " level " +
	                    static_cast<std::string>(forecastLevel));

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

		// process mean&var before hanging size of sortedValues
		float mean = numerical_functions::Mean<float>(sortedValues);
		if (!std::isfinite(mean))
		{
			mean = MissingFloat();
		}

		float var = std::sqrt(numerical_functions::Variance<float>(sortedValues));
		if (!std::isfinite(var))
		{
			var = MissingFloat();
		}

		// sortedValues needs to have one element at the back for correct array indexing
		// NOTE: `ensembleSize` stays the same

		sortedValues.push_back(0.0);

		ASSERT(!itsFractiles.empty());

		size_t targetInfoIndex = 0;
		for (auto P : itsFractiles)
		{
			// use the linear interpolation between closest ranks method recommended by NIST
			// http://www.itl.nist.gov/div898/handbook/prc/section2/prc262.htm
			float x;

			// check lower corner case p E [0,1/(N+1)]
			if (P / 100.0 <= 1.0 / static_cast<float>(ensembleSize + 1))
			{
				x = 1;
			}
			// check upper corner case p E [N/(N+1),1]
			else if (P / 100.0f >= static_cast<float>(ensembleSize) / static_cast<float>(ensembleSize + 1))
			{
				x = static_cast<float>(ensembleSize);
			}
			// everything that happens on the interval between
			else
			{
				x = P / 100.0f * static_cast<float>(ensembleSize + 1);
			}
			// floor x explicitly before casting to int
			int i = static_cast<int>(std::floor(x));

			myTargetInfo->Index<param>(targetInfoIndex);

			myTargetInfo->Value(sortedValues[i - 1] + std::fmod(x, 1.0f) * (sortedValues[i] - sortedValues[i - 1]));
			++targetInfoIndex;
		}

		myTargetInfo->Index<param>(targetInfoIndex);
		myTargetInfo->Value(mean);
		++targetInfoIndex;
		myTargetInfo->Index<param>(targetInfoIndex);
		myTargetInfo->Value(var);
	}

	threadedLogger.Info("[" + deviceType + "] Missing values: " + std::to_string(myTargetInfo->Data().MissingCount()) +
	                    "/" + std::to_string(myTargetInfo->Data().Size()));
}

}  // plugin
}  // namespace
