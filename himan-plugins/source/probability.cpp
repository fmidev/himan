#include "probability.h"
#include "ensemble.h"
#include "lagged_ensemble.h"
#include "plugin_factory.h"
#include "probability_impl.h"
#include <thread>

#include <algorithm>
#include <exception>

#include "point_list.h"
#include "radon.h"

using namespace PROB;
static std::mutex getMutex;
static int getIndex;

namespace himan
{
namespace plugin
{
probability::probability()
{
	itsCudaEnabledCalculation = false;
	itsLogger = logger("probability");
}

probability::~probability()
{
}
param GetParamFromDatabase(const std::string& paramName, const std::shared_ptr<const plugin_configuration> conf)
{
	param p;
	auto r = GET_PLUGIN(radon);

	auto paraminfo = r->RadonDB().GetParameterFromDatabaseName(conf->TargetProducer().Id(), paramName);

	if (paraminfo.empty())
	{
		throw std::runtime_error("probability: key 'input_param' not specified");
	}

	return param(paraminfo);
}

/// @brief Configuration reading
/// @param outParamConfig is modified to have information about the threshold value and input parameters
/// @returns param to be pushed in the calculatedParams vector in Process()

static void GetConfigurationParameter(const std::string& name, const std::shared_ptr<const plugin_configuration> conf,
                                      partial_param_configuration* outParamConfig)
{
	himan::logger log("probability");

	if (!conf->ParameterExists(name))
	{
		log.Fatal(
		    fmt::format("configuration error: requested parameter doesn't exist in the configuration file '{}'", name));
		himan::Abort();
	}

	const auto paramOpts = conf->GetParameterOptions(name);

	param param, output = GetParamFromDatabase(name, conf);

	// NOTE These are plugin dependent
	for (auto&& p : paramOpts)
	{
		if (p.first == "threshold")
		{
			output.ProcessingType().Value(stod(p.second));
			outParamConfig->thresholds.push_back(p.second);
		}
		else if (p.first == "input_param1" || p.first == "input_param")
		{
			// input_param1 is for backwards compatibility
			param = GetParamFromDatabase(p.second, conf);
		}
		else if (p.first == "input_param2")
		{
			throw std::runtime_error("input_param2 is not supported anymore");
		}
		else if (p.first == "comparison")
		{
			if (p.second == "<")
			{
				output.ProcessingType().Type(kProbabilityLessThan);
			}
			else if (p.second == "<=")
			{
				output.ProcessingType().Type(kProbabilityLessThanOrEqual);
			}
			else if (p.second == ">")
			{
				output.ProcessingType().Type(kProbabilityGreaterThan);
			}
			else if (p.second == ">=")
			{
				output.ProcessingType().Type(kProbabilityGreaterThanOrEqual);
			}
			else if (p.second == "=")
			{
				output.ProcessingType().Type(kProbabilityEquals);
			}
			else if (p.second == "=[]")
			{
				output.ProcessingType().Type(kProbabilityEqualsIn);
			}
			else if (p.second == "!=" || p.second == "<>")
			{
				output.ProcessingType().Type(kProbabilityNotEquals);
			}
			else if (p.second == "[]")
			{
				output.ProcessingType(processing_type(kProbabilityBetween));
			}
			else
			{
				log.Fatal(fmt::format("configuration error: invalid comparison operator '{}'", p.second));
				himan::Abort();
			}
		}
		else if (p.first == "aggregation")
		{
			output.Aggregation(aggregation(p.second));
		}
		else
		{
			// station-wise threshold
			auto elems = util::Split(p.first, "_");
			if (elems.size() == 2 && elems[0] == "threshold")
			{
				// Note: station-wise limits only support single-value thresholds
				outParamConfig->stationThresholds[std::stoi(elems[1])] = p.second;
			}
		}
	}

	if (output.ProcessingType().Type() == kUnknownProcessingType)
	{
		output.ProcessingType().Type(kProbabilityGreaterThanOrEqual);
	}

	const auto iname = param.Name();

	if (iname == "XX-X")
	{
		log.Fatal(fmt::format("configuration error:: input parameter not specified for '{}'", name));
		himan::Abort();
	}

	const bool spread = (iname == "T-K" || iname == "T-C" || iname == "WATLEV-CM" || iname == "TD-K" ||
	                     iname == "P-PA" || iname == "P-HPA" || iname == "WATLEV-N2000-CM") &&
	                    (output.ProcessingType().Type() == kProbabilityLessThan ||
	                     output.ProcessingType().Type() == kProbabilityLessThanOrEqual ||
	                     output.ProcessingType().Type() == kProbabilityGreaterThan ||
	                     output.ProcessingType().Type() == kProbabilityGreaterThanOrEqual);

	outParamConfig->useGaussianSpread = spread;
	outParamConfig->parameter = param;
	outParamConfig->output = output;
}

static void FetchRemainingLimitsForStations(const grid* targetGrid,
                                            std::vector<PROB::partial_param_configuration>& paramConfigurations,
                                            logger& log)
{
	// Make sure that limit exists for all stations (if source data is stations)

	if (targetGrid->Type() == kPointList)
	{
		for (auto& pc : paramConfigurations)
		{
			if (pc.thresholds.size() == 0)
			{
				pc.thresholds.resize(targetGrid->Size());
			}
		}

		const auto stations = dynamic_cast<const point_list*>(targetGrid)->Stations();

		// Find limit for each station defined.
		// If value is defined in configuration file, use that. Otherwise
		// check from database.

		auto r = GET_PLUGIN(radon);

		int i = -1;
		for (const auto& st : stations)
		{
			i++;
			for (auto& pc : paramConfigurations)
			{
				auto it = pc.stationThresholds.find(st.Id());

				if (it == pc.stationThresholds.end())
				{
					double limit = r->RadonDB().GetProbabilityLimitForStation(st.Id(), pc.output.Name());

					if (IsMissing(limit))
					{
						log.Fatal(
						    fmt::format("Threshold not found for param {}, station {}", pc.output.Name(), st.Id()));
						himan::Abort();
					}

					log.Trace(
					    fmt::format("Threshold for param {}, station {} is {}", pc.output.Name(), st.Id(), limit));

					pc.thresholds[i] = std::to_string(limit);
				}
				else
				{
					log.Trace(
					    fmt::format("Threshold for param {}, station {} is {}", pc.output.Name(), st.Id(), it->second));

					pc.thresholds[i] = it->second;
				}
			}
		}
	}
#ifdef DEBUG
	for (const auto& pc : paramConfigurations)
	{
		for (const auto& limit : pc.thresholds)
		{
			ASSERT(!limit.empty());
		}
	}
#endif
}

void probability::Process(const std::shared_ptr<const plugin_configuration> conf)
{
	Init(conf);

	params calculatedParams;

	const auto names = conf->GetParameterNames();

	for (const std::string& name : names)
	{
		partial_param_configuration config;

		config.useGaussianSpread = false;

		GetConfigurationParameter(name, conf, &config);

		itsParamConfigurations.push_back(config);
		calculatedParams.push_back(config.output);
	}

	SetParams(calculatedParams);

	if (itsForecastTypeIterator.Size() > 1)
	{
		itsLogger.Warning(
		    "More than one forecast type defined - probability can only produce 'statistical processing'");
	}

	const std::vector<forecast_type> type = {forecast_type(kStatisticalProcessing)};
	itsForecastTypeIterator = forecast_type_iter(type);
	std::const_pointer_cast<himan::plugin_configuration>(itsConfiguration)->ForecastTypes(type);

	FetchRemainingLimitsForStations(conf->BaseGrid(), itsParamConfigurations, itsLogger);

	Start<float>();
}

std::unique_ptr<ensemble> Mogrify(const ensemble* baseEns, const himan::param& par)
{
	if (baseEns->ClassName() == "himan::ensemble")
	{
		return std::move(std::make_unique<ensemble>(par, baseEns->ExpectedSize(), baseEns->MaximumMissingForecasts()));
	}
	else if (baseEns->ClassName() == "himan::lagged_ensemble")
	{
		return std::move(
		    std::make_unique<lagged_ensemble>(par, dynamic_cast<const lagged_ensemble*>(baseEns)->DesiredForecasts(),
		                                      baseEns->MaximumMissingForecasts()));
	}
	else if (baseEns->ClassName() == "himan::time_ensemble")
	{
		const auto d = dynamic_cast<const time_ensemble*>(baseEns);
		return std::move(std::make_unique<time_ensemble>(par, baseEns->ExpectedSize(), d->PrimaryTimeSpan(),
		                                                 d->SecondaryTimeMaskLen(), d->SecondaryTimeMaskStep(),
		                                                 d->SecondaryTimeSpan(), baseEns->MaximumMissingForecasts()));
	}
	return nullptr;
}

PROB::partial_param_configuration probability::GetTarget()
{
	std::lock_guard<std::mutex> lock(getMutex);

	auto pc = itsParamConfigurations.at(getIndex);
	getIndex++;

	return pc;
}

void ProcessParameter(std::shared_ptr<const plugin_configuration>& conf, std::shared_ptr<info<float>>& myTargetInfo,
                      const PROB::partial_param_configuration& pc, const ensemble* baseEns, const logger& logr)
{
	auto ens = Mogrify(baseEns, pc.parameter);

	if (ens == nullptr)
	{
		return;
	}

	// combine ensemble configuration with the source parameter name for this loop
	// iteration

	logr.Info(fmt::format("Calculating {} time {}", pc.output.Name(),
	                      static_cast<std::string>(myTargetInfo->Time().ValidDateTime())));

	try
	{
		ens->Fetch(conf, myTargetInfo->Time(), myTargetInfo->Level());
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			return;
		}
		else
		{
			logr.Fatal("Received error code " + std::to_string(e));
			himan::Abort();
		}
	}

	ASSERT(myTargetInfo->Data().Size() > 0);

	if (pc.useGaussianSpread)
	{
		logr.Debug("Gaussian spread is enabled");
		ProbabilityWithGaussianSpread<float>(myTargetInfo, ToParamConfiguration<float>(pc), ens);
	}
	else
	{
		logr.Trace("Gaussian spread is disabled");
		switch (pc.output.ProcessingType().Type())
		{
			case kProbabilityLessThan:
				Probability<float>(myTargetInfo, ToParamConfiguration<float>(pc), ens, std::less<float>());
				break;
			case kProbabilityLessThanOrEqual:
				Probability<float>(myTargetInfo, ToParamConfiguration<float>(pc), ens, std::less_equal<float>());
				break;
			case kProbabilityGreaterThan:
				Probability<float>(myTargetInfo, ToParamConfiguration<float>(pc), ens, std::greater<float>());
				break;
			case kProbabilityGreaterThanOrEqual:
				Probability<float>(myTargetInfo, ToParamConfiguration<float>(pc), ens, std::greater_equal<float>());
				break;
			case kProbabilityEquals:
				Probability<float>(myTargetInfo, ToParamConfiguration<float>(pc), ens, std::equal_to<float>());
				break;
			case kProbabilityNotEquals:
				Probability<float>(myTargetInfo, ToParamConfiguration<float>(pc), ens, std::not_equal_to<float>());
				break;
			case kProbabilityEqualsIn:
				Probability<std::vector<float>>(myTargetInfo, ToParamConfiguration<std::vector<float>>(pc), ens,
				                                EQINCompare());
				break;
			case kProbabilityBetween:
				Probability<std::vector<float>>(myTargetInfo, ToParamConfiguration<std::vector<float>>(pc), ens,
				                                BTWNCompare());
				break;
			default:
				logr.Error(fmt::format("Unsupported comparison operator: {}", pc.output.ProcessingType().Type()));
				break;
		}
	}
}

void probability::Worker(std::shared_ptr<info<float>> myTargetInfo, short threadIndex)
{
	auto threadedLogger = logger("probabilityThread # " + std::to_string(threadIndex));
	auto baseEns = util::CreateEnsembleFromConfiguration(itsConfiguration);

	threadedLogger.Info("Starting");

	while (true)
	{
		try
		{
			auto pc = GetTarget();
			ProcessParameter(itsConfiguration, myTargetInfo, pc, baseEns.get(), threadedLogger);
		}
		catch (...)
		{
			threadedLogger.Info("Stopping");
			return;
		}
	}
}

void probability::Calculate(std::shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
{
	std::vector<std::thread> threads;

	short realThreadCount = itsThreadCount;

	// If multithreading doesn't happen on level/time basis (=only processing a single level and time),
	// we can do it per probability parameter basis (=process all probability parameters for this level
	// and time in parallel)

	if (realThreadCount == 1)
	{
		const auto cnfCount = itsConfiguration->ThreadCount();
		realThreadCount = (cnfCount == -1)
		                      ? static_cast<short>(std::min(12, static_cast<int>(itsParamConfigurations.size())))
		                      : cnfCount;

		{
			std::lock_guard<std::mutex> lock(getMutex);
			getIndex = 0;
		}
		for (short i = 0; i < realThreadCount; i++)
		{
			threads.emplace_back(&probability::Worker, this, std::make_shared<info<float>>(*myTargetInfo), i + 1);
		}

		for (auto& t : threads)
		{
			t.join();
		}
	}
	else
	{
		auto baseEns = util::CreateEnsembleFromConfiguration(itsConfiguration);

		for (const auto& pc : itsParamConfigurations)
		{
			ProcessParameter(itsConfiguration, myTargetInfo, pc, baseEns.get(), itsLogger);
		}
	}
	itsLogger.Info(
	    fmt::format("[CPU] Missing values: {}/{}", myTargetInfo->Data().MissingCount(), myTargetInfo->Data().Size()));
}

}  // namespace plugin

}  // namespace himan
