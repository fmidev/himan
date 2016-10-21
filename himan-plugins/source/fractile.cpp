/**
 * @file fractile.cpp
 *
 **/

#include "fractile.h"

#include <algorithm>
#include <iostream>
#include <string>

#include <boost/thread.hpp>

#include "logger_factory.h"
#include "plugin_factory.h"

#include "ensemble.h"
#include "time_ensemble.h"

#include "fetcher.h"
#include "json_parser.h"
#include "radon.h"

namespace himan
{
namespace plugin
{
fractile::fractile() : itsEnsembleSize(0), itsEnsembleType(kPerturbedEnsemble)
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

	if (!itsConfiguration->GetValue("ensemble_type").empty())
	{
		auto ensType = itsConfiguration->GetValue("ensemble_type");
		assert(ensType == "time_ensemble");

		auto ensSize = itsConfiguration->GetValue("ensemble_size");

		if (!ensSize.empty())
		{
			itsEnsembleSize = boost::lexical_cast<int>(ensSize);
		}
		else
		{
			itsLogger->Error("Time ensemble size not specified");
			return;
		}

		itsEnsembleType = kTimeEnsemble;
	}
	else
	{
		itsLogger->Error("Param not specified");
		return;
	}

	params calculatedParams;
	std::vector<std::string> fractiles = {"F0-", "F10-", "F25-", "F50-", "F75-", "F90-", "F100-", ""};

	for (const std::string& fractile : fractiles)
	{
		calculatedParams.push_back(param(fractile + itsParamName));
	}

	auto r = GET_PLUGIN(radon);

	if (itsEnsembleType == kPerturbedEnsemble)
	{
		std::string ensembleSizeStr =
		    r->RadonDB().GetProducerMetaData(itsConfiguration->SourceProducer(0).Id(), "ensemble size");

		if (ensembleSizeStr.empty())
		{
			itsLogger->Error("Unable to find ensemble size from database");
			return;
		}

		itsEnsembleSize = boost::lexical_cast<int>(ensembleSizeStr);
	}

	SetParams(calculatedParams);

	Start();
}

void fractile::Calculate(std::shared_ptr<info> myTargetInfo, uint16_t threadIndex)
{
	std::vector<int> fractile = {0, 10, 25, 50, 75, 90, 100};

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
		default:
		case kPerturbedEnsemble:
			ens = std::unique_ptr<ensemble>(new ensemble(param(itsParamName), itsEnsembleSize));
			break;
		case kTimeEnsemble:
			ens = std::unique_ptr<time_ensemble>(new time_ensemble(param(itsParamName), itsEnsembleSize));
			break;
	}

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

		size_t targetInfoIndex = 0;
		for (auto i : fractile)
		{
			myTargetInfo->ParamIndex(targetInfoIndex);
			myTargetInfo->Value(sortedValues[i * (itsEnsembleSize - 1) / 100]);
			++targetInfoIndex;
		}
		// write mean value to last target info index
		myTargetInfo->ParamIndex(targetInfoIndex);
		myTargetInfo->Value(ens->Mean());
	}

	threadedLogger->Info("[" + deviceType + "] Missing values: " +
	                     boost::lexical_cast<std::string>(myTargetInfo->Data().MissingCount()) + "/" +
	                     boost::lexical_cast<std::string>(myTargetInfo->Data().Size()));
}

}  // plugin

}  // namespace
