#include "probability.h"
#include "plugin_factory.h"
#include "probability_impl.h"

#include "ensemble.h"
#include "lagged_ensemble.h"

#include <algorithm>
#include <exception>

#include "radon.h"

using namespace PROB;

namespace himan
{
namespace plugin
{
static const std::string kClassName = "himan::plugin::probability";

probability::probability()
{
	itsCudaEnabledCalculation = false;
	itsLogger = logger("probability");

	itsEnsembleSize = 0;
	itsMaximumMissingForecasts = 0;
	itsUseNormalizedResult = false;
	itsUseLaggedEnsemble = false;
	itsLag = 0;
	itsLaggedSteps = 0;
}

probability::~probability()
{
}
param GetParamFromDatabase(const std::string& paramName, const std::shared_ptr<const plugin_configuration> conf)
{
	param p;
	auto r = GET_PLUGIN(radon);

	auto paraminfo = r->RadonDB().GetParameterFromDatabaseName(conf->Info()->Producer().Id(), paramName);

	if (paraminfo.empty())
	{
		p.Name(paramName);
		return p;
	}

	return param(paraminfo);
}

/// @brief Configuration reading
/// @param outParamConfig is modified to have information about the threshold value and input parameters
/// @returns param to be pushed in the calculatedParams vector in Process()

static param GetConfigurationParameter(const std::string& name, const std::shared_ptr<const plugin_configuration> conf,
                                       partial_param_configuration* outParamConfig)
{
	if (!conf->ParameterExists(name))
	{
		throw std::runtime_error(
		    "probability : configuration error:: requested parameter doesn't exist in the configuration file '" + name +
		    "'");
	}

	const auto paramOpts = conf->GetParameterOptions(name);

	param param;

	// NOTE These are plugin dependent
	for (auto&& p : paramOpts)
	{
		if (p.first == "threshold")
		{
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
				outParamConfig->comparison = comparison_op::LTEQ;
			}
			else if (p.second == ">=")
			{
				outParamConfig->comparison = comparison_op::GTEQ;
			}
			else if (p.second == "=")
			{
				outParamConfig->comparison = comparison_op::EQ;
			}
			else if (p.second == "=[]")
			{
				outParamConfig->comparison = comparison_op::EQIN;
			}
			else if (p.second == "!=" || p.second == "<>")
			{
				outParamConfig->comparison = comparison_op::NEQ;
			}
			else if (p.second == "[]")
			{
				outParamConfig->comparison = comparison_op::BTWN;
			}
			else
			{
				throw std::runtime_error("probability : configuration error:: invalid comparison operator '" +
				                         p.second + "'");
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

	if (param.Name() == "XX-X")
	{
		throw std::runtime_error("probability : configuration error:: input parameter not specified for '" + name +
		                         "'");
	}
	outParamConfig->parameter = param;

	return himan::param(name);
}

static void FetchRemainingLimitsForStations(info_t tempInfo,
                                            std::vector<PROB::partial_param_configuration>& paramConfigurations,
                                            logger& log)
{
	// Make sure that limit exists for all stations (if source data is stations)
	tempInfo->First();

	if (tempInfo->Grid()->Type() == kPointList)
	{
		for (auto& pc : paramConfigurations)
		{
			if (pc.thresholds.size() == 0)
			{
				pc.thresholds.resize(tempInfo->SizeLocations());
			}
		}

		// Find limit for each station defined.
		// If value is defined in configuration file, use that. Otherwise
		// check from database.

		auto r = GET_PLUGIN(radon);
		for (tempInfo->ResetLocation(); tempInfo->NextLocation();)
		{
			const auto st = tempInfo->Station();

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

					pc.thresholds[tempInfo->LocationIndex()] = std::to_string(limit);
				}
				else
				{
					log.Trace("Threshold for param " + pc.output.Name() + ", station " + std::to_string(st.Id()) +
					          " is " + it->second);

					pc.thresholds[tempInfo->LocationIndex()] = it->second;
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
		auto ensSize = r->RadonDB().GetProducerMetaData(conf->Info()->Producer().Id(), "ensemble size");

		if (ensSize.empty())
		{
			throw std::runtime_error(
			    ClassName() + " ensemble_size not specified in plugin configuration and not found from database");
		}

		itsEnsembleSize = std::stoi(ensSize);
	}

	// Find out whether we want probabilities in [0,1] range or [0,100] range
	if (itsConfiguration->Exists("normalized_results"))
	{
		const std::string useNormalized = itsConfiguration->GetValue("normalized_results");

		itsUseNormalizedResult = (useNormalized == "true") ? true : false;
	}
	else
	{
		// default to [0,100] for compatibility
		itsUseNormalizedResult = false;
		itsLogger.Info(
		    "'normalized_results' not found from the configuration, results will be written in [0,100] range");
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

		itsLag = lag;
		itsUseLaggedEnsemble = true;
	}
	else
	{
		itsUseLaggedEnsemble = false;
	}

	// How many lagged steps to include in the calculation
	if (itsUseLaggedEnsemble)
	{
		if (itsConfiguration->Exists("lagged_steps"))
		{
			const int steps = std::stoi(itsConfiguration->GetValue("lagged_steps"));
			if (steps <= 0)
			{
				throw std::runtime_error(ClassName() + ": invalid lagged_steps value. Allowed range >= 0");
			}
			itsLaggedSteps = steps;
		}
		else
		{
			throw std::runtime_error(ClassName() + ": specify lagged_steps when using time lagging ('lag')");
		}
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

		config.output.Name(name);
		config.comparison = comparison_op::GTEQ;

		param p = GetConfigurationParameter(name, conf, &config);

		if (p.Name() == "")
		{
			throw std::runtime_error(ClassName() + "Misconfigured parameter definition in JSON");
		}

		itsParamConfigurations.push_back(config);
		calculatedParams.push_back(p);
	}

	SetParams(calculatedParams);

	FetchRemainingLimitsForStations(std::make_shared<himan::info>(*conf->Info()), itsParamConfigurations, itsLogger);

	Start();
}

void probability::Calculate(info_t myTargetInfo, unsigned short threadIndex)
{
	auto threadedLogger = logger("probabilityThread # " + std::to_string(threadIndex));

	for (const auto& pc : itsParamConfigurations)
	{
		const int ensembleSize = itsEnsembleSize;
		const bool normalized = itsUseNormalizedResult;

		std::unique_ptr<ensemble> ens;

		if (itsUseLaggedEnsemble)
		{
			threadedLogger.Info("Using lagged ensemble");
			ens = std::unique_ptr<ensemble>(
			    new lagged_ensemble(pc.parameter, ensembleSize, kHourResolution, itsLag, itsLaggedSteps + 1));
		}
		else
		{
			ens = std::unique_ptr<ensemble>(new ensemble(pc.parameter, ensembleSize));
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

		switch (pc.comparison)
		{
			case comparison_op::LTEQ:
				Probability<double>(myTargetInfo, ToParamConfiguration<double>(pc), normalized, ens,
				                    std::less_equal<double>());
				break;
			case comparison_op::GTEQ:
				Probability<double>(myTargetInfo, ToParamConfiguration<double>(pc), normalized, ens,
				                    std::greater_equal<double>());
				break;
			case comparison_op::EQ:
				Probability<double>(myTargetInfo, ToParamConfiguration<double>(pc), normalized, ens,
				                    std::equal_to<double>());
				break;
			case comparison_op::NEQ:
				Probability<double>(myTargetInfo, ToParamConfiguration<double>(pc), normalized, ens,
				                    std::not_equal_to<double>());
				break;
			case comparison_op::EQIN:
				Probability<std::vector<double>>(myTargetInfo, ToParamConfiguration<std::vector<double>>(pc),
				                                 normalized, ens, EQINCompare());
				break;
			case comparison_op::BTWN:
				Probability<std::vector<double>>(myTargetInfo, ToParamConfiguration<std::vector<double>>(pc),
				                                 normalized, ens, BTWNCompare());
				break;
			default:
				threadedLogger.Error("Unsupported comparison operator");
				break;
		}
	}

	threadedLogger.Info("[CPU] Missing values: " + std::to_string(myTargetInfo->Data().MissingCount()) + "/" +
	                    std::to_string(myTargetInfo->Data().Size()));
}

}  // plugin

}  // namespace
