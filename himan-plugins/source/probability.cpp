#include "probability.h"
#include "plugin_factory.h"
#include "probability_impl.h"

#include "ensemble.h"
#include "lagged_ensemble.h"

#include <algorithm>
#include <exception>

#include "point_list.h"
#include "radon.h"

using namespace PROB;

namespace himan
{
namespace plugin
{
static const std::string kClassName = "himan::plugin::probability";

probability::probability()
    : itsEnsembleSize(0), itsMaximumMissingForecasts(0), itsUseLaggedEnsemble(0), itsLag(), itsLagStep()
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
		log.Fatal("configuration error: requested parameter doesn't exist in the configuration file '" + name + "'");
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
			if (p.second == "<=")
			{
				output.ProcessingType().Type(kProbabilityLessThan);
			}
			else if (p.second == ">=")
			{
				output.ProcessingType().Type(kProbabilityGreaterThan);
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
				log.Fatal("configuration error: invalid comparison operator '" + p.second + "'");
				himan::Abort();
			}
		}
		else
		{
			// station-wise threshold
			auto elems = util::Split(p.first, "_", false);
			if (elems.size() == 2 && elems[0] == "threshold")
			{
				// Note: station-wise limits only support single-value thresholds
				outParamConfig->stationThresholds[std::stoi(elems[1])] = p.second;
			}
		}
	}

	if (output.ProcessingType().Type() == kUnknownProcessingType)
	{
		output.ProcessingType().Type(kProbabilityGreaterThan);
	}

	const auto iname = param.Name();

	if (iname == "XX-X")
	{
		log.Fatal("configuration error:: input parameter not specified for '" + name + "'");
		himan::Abort();
	}

	const bool spread = (iname == "T-K" || iname == "T-C" || iname == "WATLEV-CM" || iname == "TD-K" ||
	                     iname == "P-PA" || iname == "P-HPA") &&
	                    (output.ProcessingType().Type() == kProbabilityLessThan ||
	                     output.ProcessingType().Type() == kProbabilityGreaterThan);

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
						log.Fatal("Threshold not found for param " + pc.output.Name() + ", station " +
						          std::to_string(st.Id()));
						himan::Abort();
					}

					log.Trace("Threshold for param " + pc.output.Name() + ", station " + std::to_string(st.Id()) +
					          " is " + std::to_string(limit));

					pc.thresholds[i] = std::to_string(limit);
				}
				else
				{
					log.Trace("Threshold for param " + pc.output.Name() + ", station " + std::to_string(st.Id()) +
					          " is " + it->second);

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

	//
	// 1. Parse json configuration specific to this plugin
	//

	// Most of the plugin operates by the configuration in the json file.
	// Here we collect all the parameters we want to calculate, and the parameters that are used for the calculation.
	// If the configuration is invalid we will bail out asap!

	// Get the number of forecasts this ensemble has from plugin configuration
	if (itsConfiguration->Exists("ensemble_size"))
	{
		const int ensembleSize = std::stoi(itsConfiguration->GetValue("ensemble_size"));
		if (ensembleSize <= 0)
		{
			throw std::runtime_error(ClassName() + " invalid ensemble_size in plugin configuration");
		}
		itsEnsembleSize = ensembleSize;
	}
	else
	{
		auto r = GET_PLUGIN(radon);
		auto ensSize = r->RadonDB().GetProducerMetaData(conf->TargetProducer().Id(), "ensemble size");

		if (ensSize.empty())
		{
			throw std::runtime_error(
			    ClassName() + " ensemble_size not specified in plugin configuration and not found from database");
		}

		itsEnsembleSize = std::stoi(ensSize);
	}

	// Maximum number of missing forecasts for an ensemble
	if (itsConfiguration->Exists("max_missing_forecasts"))
	{
		const int maxMissingForecasts = std::stoi(itsConfiguration->GetValue("max_missing_forecasts"));
		if (maxMissingForecasts < 0)
		{
			throw std::runtime_error(ClassName() +
			                         " invalid max_missing_forecasts value specified in plugin configuration");
		}
		itsMaximumMissingForecasts = maxMissingForecasts;
	}

	// Are we using lagged ensemble?
	// NOTE 'lag' needs to be specified first
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

		itsLag = time_duration(kHourResolution, lag);

		// How many lagged steps to include in the calculation

		if (itsConfiguration->Exists("lagged_steps"))
		{
			const std::string lagsteps = itsConfiguration->GetValue("lagged_steps");

			if (lagsteps.find(":") == std::string::npos)
			{
				const int steps = std::stoi(lagsteps);
				if (steps <= 0)
				{
					throw std::runtime_error(ClassName() + ": invalid lagged_steps value. Allowed range >= 0");
				}
				itsLagStep = itsLag * -1;
				itsLag *= steps;
			}
			else
			{
				itsLagStep = time_duration(lagsteps);
			}
		}
		else
		{
			throw std::runtime_error(ClassName() + ": specify lagged_steps when using time lagging ('lag')");
		}

		itsUseLaggedEnsemble = true;
	}

	if (itsConfiguration->Exists("named_ensemble"))
	{
		itsUseLaggedEnsemble = true;
		itsNamedEnsemble = itsConfiguration->GetValue("named_ensemble");
	}

	//
	// 2. Setup input and output parameters from the json configuration.
	//    `calculatedParams' will hold the output parameter, it's inputs,
	//    and the 'threshold' value.
	//

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

	itsForecastTypeIterator = forecast_type_iter({forecast_type(kStatisticalProcessing)});

	FetchRemainingLimitsForStations(conf->BaseGrid(), itsParamConfigurations, itsLogger);

	Start<float>();
}

void probability::Calculate(std::shared_ptr<info<float>> myTargetInfo, unsigned short threadIndex)
{
	auto threadedLogger = logger("probabilityThread # " + std::to_string(threadIndex));

	for (const auto& pc : itsParamConfigurations)
	{
		std::unique_ptr<ensemble> ens;

		if (itsUseLaggedEnsemble)
		{
			threadedLogger.Info("Using lagged ensemble");
			if (itsNamedEnsemble.empty() == false)
			{
				ens = std::unique_ptr<ensemble>(new lagged_ensemble(pc.parameter, itsNamedEnsemble));
			}
			else
			{
				ens = std::unique_ptr<ensemble>(new lagged_ensemble(pc.parameter, itsEnsembleSize, itsLag, itsLagStep));
			}
		}
		else
		{
			ens = std::unique_ptr<ensemble>(new ensemble(pc.parameter, itsEnsembleSize));
		}

		ens->MaximumMissingForecasts(itsMaximumMissingForecasts);

		threadedLogger.Info("Calculating " + pc.output.Name() + " time " +
		                    static_cast<std::string>(myTargetInfo->Time().ValidDateTime()));

		try
		{
			ens->Fetch(itsConfiguration, myTargetInfo->Time(), myTargetInfo->Level());
		}
		catch (const HPExceptionType& e)
		{
			if (e == kFileDataNotFound)
			{
				continue;
			}
			else
			{
				itsLogger.Fatal("Received error code " + std::to_string(e));
				himan::Abort();
			}
		}

		ASSERT(myTargetInfo->Data().Size() > 0);

		if (pc.useGaussianSpread)
		{
			threadedLogger.Debug("Gaussian spread is enabled");
			ProbabilityWithGaussianSpread<float>(myTargetInfo, ToParamConfiguration<float>(pc), ens);
		}
		else
		{
			threadedLogger.Trace("Gaussian spread is disabled");
			switch (pc.output.ProcessingType().Type())
			{
				case kProbabilityLessThan:
					Probability<float>(myTargetInfo, ToParamConfiguration<float>(pc), ens, std::less_equal<float>());
					break;
				case kProbabilityGreaterThan:
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
					threadedLogger.Error("Unsupported comparison operator: " +
					                     std::to_string(pc.output.ProcessingType().Type()));
					break;
			}
		}
	}

	threadedLogger.Info("[CPU] Missing values: " + std::to_string(myTargetInfo->Data().MissingCount()) + "/" +
	                    std::to_string(myTargetInfo->Data().Size()));
}

}  // plugin

}  // namespace
