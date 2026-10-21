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

static HPProcessingType ParseComparison(const std::string& op, const std::string& paramName)
{
	if (op == "<")
	{
		return kProbabilityLessThan;
	}
	else if (op == "<=")
	{
		return kProbabilityLessThanOrEqual;
	}
	else if (op == ">")
	{
		return kProbabilityGreaterThan;
	}
	else if (op == ">=")
	{
		return kProbabilityGreaterThanOrEqual;
	}
	else if (op == "=")
	{
		return kProbabilityEquals;
	}
	else if (op == "=[]")
	{
		return kProbabilityEqualsIn;
	}
	else if (op == "!=" || op == "<>")
	{
		return kProbabilityNotEquals;
	}
	else if (op == "[)")
	{
		return kProbabilityBetween;
	}
	else if (op == "[]")
	{
		// The range is bounded by [lower, upper): the upper endpoint is not included. The operator was
		// previously documented as "[)" but accepted as "[]"; the old name is kept for compatibility.
		himan::logger log("probability");
		log.Warning(fmt::format("configuration warning: '{}': comparison operator '[]' is deprecated, use '[)' instead",
		                        paramName));
		return kProbabilityBetween;
	}

	himan::logger log("probability");
	log.Fatal(fmt::format("configuration error: invalid comparison operator '{}' for '{}'", op, paramName));
	himan::Abort();
}

/// @brief Reading of the additional conditions
/// @param options contains the flattened 'conditions' options, ie. 'conditions[0].input_param', 'conditions[0].value'
/// etc. See json_parser.cpp.
/// @param name is the name of the output parameter, used for error messages

static std::vector<PROB::partial_condition> GetConditions(
    const std::map<size_t, std::map<std::string, std::string>>& options, const std::string& name,
    const std::shared_ptr<const plugin_configuration> conf)
{
	himan::logger log("probability");

	std::vector<PROB::partial_condition> conditions;

	for (const auto& cond : options)
	{
		const auto& opts = cond.second;

		for (const auto& opt : opts)
		{
			if (opt.first == "threshold")
			{
				log.Fatal(fmt::format("configuration error: '{}': conditions use key 'value' instead of 'threshold'",
				                      name));
				himan::Abort();
			}
			else if (opt.first != "input_param" && opt.first != "comparison" && opt.first != "value" &&
			         opt.first != "aggregation")
			{
				log.Fatal(fmt::format("configuration error: '{}': unknown condition option '{}'", name, opt.first));
				himan::Abort();
			}
		}

		for (const auto& required : {"input_param", "comparison", "value"})
		{
			if (opts.find(required) == opts.end())
			{
				log.Fatal(fmt::format("configuration error: '{}': condition {} is missing '{}'", name, cond.first,
				                      required));
				himan::Abort();
			}
		}

		PROB::partial_condition pc;

		pc.parameter = util::InitializeParameter(conf->TargetProducer(), himan::param(opts.at("input_param")),
		                                         level(kUnknownLevel, 0));
		pc.comparison = ParseComparison(opts.at("comparison"), name);
		pc.value = opts.at("value");

		if (opts.find("aggregation") != opts.end())
		{
			pc.parameter.Aggregation(aggregation(opts.at("aggregation")));
		}

		conditions.push_back(pc);
	}

	return conditions;
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

	param param,
	    output = util::InitializeParameter(conf->TargetProducer(), himan::param(name), level(kUnknownLevel, 0));

	std::map<size_t, std::map<std::string, std::string>> conditionOptions;

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
			param = util::InitializeParameter(conf->TargetProducer(), himan::param(p.second), level(kUnknownLevel, 0));
		}
		else if (p.first == "input_param2")
		{
			throw std::runtime_error("input_param2 is not supported anymore");
		}
		else if (p.first == "comparison")
		{
			output.ProcessingType().Type(ParseComparison(p.second, name));
		}
		else if (p.first == "aggregation")
		{
			param.Aggregation(aggregation(p.second));
			output.Aggregation(aggregation(p.second));
		}
		else if (p.first.compare(0, 11, "conditions[") == 0)
		{
			// flattened conditions array: conditions[<index>].<option>
			const auto end = p.first.find("].");

			if (end == std::string::npos)
			{
				log.Fatal(fmt::format("configuration error: '{}': invalid conditions option '{}'", name, p.first));
				himan::Abort();
			}

			const size_t index = std::stoul(p.first.substr(11, end - 11));
			conditionOptions[index][p.first.substr(end + 2)] = p.second;
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

	outParamConfig->conditions = GetConditions(conditionOptions, name, conf);

	if (spread && !outParamConfig->conditions.empty())
	{
		log.Fatal(fmt::format(
		    "configuration error: '{}': conditions cannot be combined with gaussian spread (input parameter {})", name,
		    iname));
		himan::Abort();
	}

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
	std::unique_ptr<ensemble> ens = nullptr;

	if (baseEns->ClassName() == "himan::ensemble")
	{
		ens = std::make_unique<ensemble>(*baseEns);
	}
	else if (baseEns->ClassName() == "himan::lagged_ensemble")
	{
		ens = std::make_unique<lagged_ensemble>(*dynamic_cast<const lagged_ensemble*>(baseEns));
	}
	else if (baseEns->ClassName() == "himan::time_ensemble")
	{
		ens = std::make_unique<time_ensemble>(*dynamic_cast<const time_ensemble*>(baseEns));
	}
	else
	{
		himan::logger logr("probability");
		logr.Fatal(fmt::format("Unknown ensemble class: {}", baseEns->ClassName()));
		himan::Abort();
	}

	ens->Param(par);
	return ens;
}

PROB::partial_param_configuration probability::GetTarget()
{
	std::lock_guard<std::mutex> lock(getMutex);

	auto pc = itsParamConfigurations.at(getIndex);
	getIndex++;

	return pc;
}

/// @brief Fetch ensemble data
/// @returns false if data was not found

static bool FetchEnsemble(ensemble& ens, std::shared_ptr<const plugin_configuration>& conf,
                          std::shared_ptr<info<float>>& myTargetInfo, const logger& logr)
{
	try
	{
		ens.Fetch(conf, myTargetInfo->Time(), myTargetInfo->Level());
	}
	catch (const HPExceptionType& e)
	{
		if (e == kFileDataNotFound)
		{
			return false;
		}
		else
		{
			logr.Fatal("Received error code " + std::to_string(e));
			himan::Abort();
		}
	}

	return true;
}

template <typename T>
static void CalculateProbability(std::shared_ptr<info<float>>& myTargetInfo, const PROB::partial_param_configuration& pc,
                      std::unique_ptr<ensemble>& ens, std::vector<std::unique_ptr<ensemble>>& conditionEns,
                      std::function<bool(float, T)> comp_op)
{
	if (pc.conditions.empty())
	{
		Probability<T>(myTargetInfo, ToParamConfiguration<T>(pc), ens, comp_op);
	}
	else
	{
		ProbabilityWithConditions<T>(myTargetInfo, ToParamConfiguration<T>(pc), ens, conditionEns, comp_op);
	}
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

	if (FetchEnsemble(*ens, conf, myTargetInfo, logr) == false)
	{
		return;
	}

	// Ensembles of the additional conditions. Ensemble members are paired by their position, so all
	// ensembles must be of equal size. If data is not found for any of the conditions, the probability
	// cannot be calculated at all.

	std::vector<std::unique_ptr<ensemble>> conditionEns;

	for (const auto& c : pc.conditions)
	{
		auto cens = Mogrify(baseEns, c.parameter);

		if (FetchEnsemble(*cens, conf, myTargetInfo, logr) == false)
		{
			return;
		}

		if (cens->Size() != ens->Size())
		{
			logr.Fatal(fmt::format("Size of ensemble {} ({}) differs from the size of ensemble {} ({})",
			                       c.parameter.Name(), cens->Size(), pc.parameter.Name(), ens->Size()));
			himan::Abort();
		}

		conditionEns.push_back(std::move(cens));
	}

	ASSERT(myTargetInfo->Data().Size() > 0);

	if (pc.useGaussianSpread)
	{
		logr.Trace("Gaussian spread is enabled");
		ProbabilityWithGaussianSpread<float>(myTargetInfo, ToParamConfiguration<float>(pc), ens);
	}
	else
	{
		logr.Trace("Gaussian spread is disabled");
		switch (pc.output.ProcessingType().Type())
		{
			case kProbabilityLessThan:
				CalculateProbability<float>(myTargetInfo, pc, ens, conditionEns, std::less<float>());
				break;
			case kProbabilityLessThanOrEqual:
				CalculateProbability<float>(myTargetInfo, pc, ens, conditionEns, std::less_equal<float>());
				break;
			case kProbabilityGreaterThan:
				CalculateProbability<float>(myTargetInfo, pc, ens, conditionEns, std::greater<float>());
				break;
			case kProbabilityGreaterThanOrEqual:
				CalculateProbability<float>(myTargetInfo, pc, ens, conditionEns, std::greater_equal<float>());
				break;
			case kProbabilityEquals:
				CalculateProbability<float>(myTargetInfo, pc, ens, conditionEns, std::equal_to<float>());
				break;
			case kProbabilityNotEquals:
				CalculateProbability<float>(myTargetInfo, pc, ens, conditionEns, std::not_equal_to<float>());
				break;
			case kProbabilityEqualsIn:
				CalculateProbability<std::vector<float>>(myTargetInfo, pc, ens, conditionEns, EQINCompare());
				break;
			case kProbabilityBetween:
				CalculateProbability<std::vector<float>>(myTargetInfo, pc, ens, conditionEns, BTWNCompare());
				break;
			default:
				logr.Error(fmt::format("Unsupported comparison operator: {}",
				                       fmt::underlying(pc.output.ProcessingType().Type())));
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
