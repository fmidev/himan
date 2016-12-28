/**
 * @file fractile.cpp
 *
 **/
#include "fractile.h"

#include <algorithm>
#include <iostream>
#include <string>

#include <boost/algorithm/string.hpp>

#include "logger_factory.h"
#include "plugin_factory.h"

#include "ensemble.h"
#include "time_ensemble.h"

#include "fetcher.h"
#include "json_parser.h"
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
      itsMaximumMissingForecasts(0)
{
	itsClearTextFormula = "%";
	itsCudaEnabledCalculation = false;
	itsLogger = logger_factory::Instance()->GetLog("fractile");
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
		itsLogger->Error("Param not specified");
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

	if (itsEnsembleSize == 0 && itsEnsembleType == kPerturbedEnsemble)
	{
		// Regular ensemble size is static, get it from database if user
		// hasn't specified any size

		auto r = GET_PLUGIN(radon);

		std::string ensembleSizeStr =
		    r->RadonDB().GetProducerMetaData(itsConfiguration->SourceProducer(0).Id(), "ensemble size");

		if (ensembleSizeStr.empty())
		{
			itsLogger->Error("Unable to find ensemble size from database");
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
				itsLogger->Fatal("Invalid fractile value: '" + val + "'");
				exit(1);
			}
		}
	}

	params calculatedParams;

	for (double fractile : itsFractiles)
	{
		auto name = "F" + boost::lexical_cast<std::string>(fractile) + "-" + itsParamName;
		calculatedParams.push_back(param(name));
	}

	calculatedParams.push_back(param(itsParamName));  // mean

	auto name = util::Split(itsParamName, "-", false);
	calculatedParams.push_back(param(name[0] + "DEV-" + name[1]));  // standard deviation

	SetParams(calculatedParams);

	Start();
}

void fractile::Calculate(std::shared_ptr<info> myTargetInfo, uint16_t threadIndex)
{
	const std::string deviceType = "CPU";

	auto threadedLogger =
	    logger_factory::Instance()->GetLog("fractileThread # " + boost::lexical_cast<std::string>(threadIndex));

	forecast_time forecastTime = myTargetInfo->Time();
	level forecastLevel = myTargetInfo->Level();

	threadedLogger->Info("Calculating time " + static_cast<std::string>(forecastTime.ValidDateTime()) + " level " +
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
		default:
			itsLogger->Fatal("Unknown ensemble type: " + HPEnsembleTypeToString.at(itsEnsembleType));
			exit(1);
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
			throw std::runtime_error(ClassName() + " failed to find ensemble data");
		}
	}

	myTargetInfo->ResetLocation();
	ens->ResetLocation();

	while (myTargetInfo->NextLocation() && ens->NextLocation())
	{
		auto sortedValues = ens->SortedValues();
		const size_t ensembleSize = sortedValues.size();

		// Skip this step if we didn't get any valid fields
		if (sortedValues.empty())
		{
			continue;
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

		// write mean value and stddev to last target info indices
		myTargetInfo->ParamIndex(targetInfoIndex);
		myTargetInfo->Value(ens->Mean());

		++targetInfoIndex;
		myTargetInfo->ParamIndex(targetInfoIndex);
		myTargetInfo->Value(std::sqrt(ens->Variance()));
	}

	threadedLogger->Info("[" + deviceType + "] Missing values: " +
	                     boost::lexical_cast<std::string>(myTargetInfo->Data().MissingCount()) + "/" +
	                     boost::lexical_cast<std::string>(myTargetInfo->Data().Size()));
}

}  // plugin

}  // namespace
