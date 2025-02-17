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

void fractile::SetParams()
{
	auto fractiles = itsConfiguration->GetValue("fractiles");

	if (!fractiles.empty())
	{
		itsFractiles.clear();

		auto list = util::Split(fractiles, ",");

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

	const aggregation agg(itsConfiguration->GetValue("aggregation"));

	for (float frac : itsFractiles)
	{
		const auto name = fmt::format("F{}-{}", frac, paramName);
		const auto pt = processing_type(kFractile, frac);
		calculatedParams.emplace_back(name, agg, pt);
	}

	auto name = util::Split(paramName, "-");

	param mean(name[0] + "-MEAN-" + name[1]);
	mean.ProcessingType(processing_type(kMean));

	calculatedParams.push_back(mean);

	param stdev(name[0] + "-STDDEV-" + name[1]);
	stdev.ProcessingType(processing_type(kStandardDeviation));

	calculatedParams.push_back(stdev);

	compiled_plugin_base::SetParams(calculatedParams);
}

void fractile::SetForecastType()
{
	if (itsForecastTypeIterator.Size() > 1)
	{
		itsLogger.Warning("More than one forecast type defined - fractile can only produce 'statistical processing'");
	}

	const std::vector<forecast_type> type({forecast_type(kStatisticalProcessing)});
	itsForecastTypeIterator = forecast_type_iter(type);
	std::const_pointer_cast<himan::plugin_configuration>(itsConfiguration)->ForecastTypes(type);
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
	auto ens = util::CreateEnsembleFromConfiguration(itsConfiguration);

	HPMissingValueTreatment treatMissing = kRemove;
	if (itsConfiguration->GetValue("missing_value_treatment") != "")
	{
		try
                {
                        treatMissing = HPStringToMissingValueTreatment.at(itsConfiguration->GetValue("missing_value_treatment"));
                }
                catch (const std::out_of_range& e)
                {
                        itsLogger.Fatal("Invalid configuration value of missing_value_treatment.");
			himan::Abort();
                }
	}

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
		auto sortedValues = ens->SortedValues(treatMissing);
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

}  // namespace plugin
}  // namespace himan
