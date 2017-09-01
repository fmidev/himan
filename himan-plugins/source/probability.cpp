#include "probability.h"

#include "plugin_factory.h"

#include "logger.h"
#include "fetcher.h"
#include "writer.h"

#include "ensemble.h"
#include "lagged_ensemble.h"
#include "util.h"

#include <boost/thread.hpp>

#include <algorithm>
#include <exception>
#include <iostream>

#include "radon.h"
#include <math.h>

namespace himan
{
namespace plugin
{
static std::mutex singleFileWriteMutex;

static const std::string kClassName = "himan::plugin::probability";

/// @brief Used for calculating wind vector magnitude
static inline double Magnitude(double u, double v) { return sqrt(u * u + v * v); }
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

probability::~probability() {}
/// @brief Configuration reading
/// @param outParamConfig is modified to have information about the threshold value and input parameters
/// @returns param to be pushed in the calculatedParams vector in Process()
static param GetConfigurationParameter(const std::string& name, const std::shared_ptr<const plugin_configuration> conf,
                                       param_configuration* outParamConfig)
{
	if (conf->ParameterExists(name))
	{
		const auto paramOpts = conf->GetParameterOptions(name);

		param param1;
		param param2;

		// NOTE These are plugin dependent
		for (auto&& p : paramOpts)
		{
			if (p.first == "threshold")
			{
				outParamConfig->gridThreshold = std::stod(p.second);
			}
			else if (p.first == "input_param1")
			{
				param1.Name(p.second);
			}
			else if (p.first == "input_param2")
			{
				param2.Name(p.second);
			}
			else
			{
				auto elems = util::Split(p.first, "_", false);
				if (elems.size() == 2 && elems[0] == "threshold")
				{
					outParamConfig->stationThreshold[std::stoi(elems[1])] = std::stod(p.second);
				}
			}
		}

		if (param1.Name() == "XX-X")
		{
			throw std::runtime_error("probability : configuration error:: input parameter not specified for '" + name +
			                         "'");
		}
		outParamConfig->parameter = param1;

		// NOTE param2 is used only with wind calculation at the moment
		if (param2.Name() != "XX-X")
		{
			outParamConfig->parameter2 = param2;
		}
	}
	else
	{
		throw std::runtime_error(
		    "probability : configuration error:: requested parameter doesn't exist in the configuration file '" + name +
		    "'");
	}

	return param(name);
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
		throw std::runtime_error(ClassName() + " ensemble_size not specified in plugin configuration");
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

	int targetInfoIndex = 0;
	const auto& names = conf->GetParameterNames();
	for (const std::string& name : names)
	{
		param_configuration config;

		config.targetInfoIndex = targetInfoIndex;
		config.output.Name(name);

		param p = GetConfigurationParameter(name, conf, &config);

		if (p.Name() == "")
		{
			throw std::runtime_error(ClassName() + "Misconfigured parameter definition in JSON");
		}

		itsParamConfigurations.push_back(config);
		calculatedParams.push_back(p);
		targetInfoIndex++;
	}

	SetParams(calculatedParams);

	// Make sure that limit exists for all stations (if source data is stations)
	auto tempInfo = std::make_shared<himan::info>(*conf->Info());
	tempInfo->First();

	if (tempInfo->Grid()->Type() == kPointList)
	{
		auto r = GET_PLUGIN(radon);
		for (tempInfo->ResetLocation(); tempInfo->NextLocation();)
		{
			const auto st = tempInfo->Station();

			for (auto& pc : itsParamConfigurations)
			{
				auto it = pc.stationThreshold.find(st.Id());

				if (it == pc.stationThreshold.end())
				{
					itsLogger.Trace("Fetching threshold for param " + pc.output.Name() + ", station " +
					                std::to_string(st.Id()) + " from radon");
					double limit = r->RadonDB().GetProbabilityLimitForStation(st.Id(), pc.output.Name());

					if (IsMissing(limit))
					{
						itsLogger.Fatal("Threshold not found for param " + pc.output.Name() + ", station " +
						                std::to_string(st.Id()));
						abort();
					}

					pc.stationThreshold[st.Id()] = limit;
				}
			}
		}
	}

	// NOTE
	// NOTE We circumvent the usual Himan Start => Run => RunAll / RunTimeDimension - control flow structure
	// NOTE

	//
	// 3. Process parameters in LIFO order by spawning threads for each parameter.
	//	  (We don't divide timesteps between threads)
	//
	boost::thread_group g;
	auto paramConfigurations = itsParamConfigurations;

	// Set iterators at this stage to avoid invalid indexing when loading from auxiliary files
	itsInfo->First();

	int threadIdx = 0;

	for (const auto& pc : paramConfigurations)
	{
		g.add_thread(new boost::thread(&himan::plugin::probability::Calculate, this, threadIdx, boost::ref(pc)));

		if (threadIdx == itsThreadCount)
		{
			g.join_all();
			threadIdx = 0;
		}

		threadIdx++;
	}

	g.join_all();

	Finish();
}

static void CalculateNormal(std::shared_ptr<info> targetInfo, uint16_t threadIndex,
                            const param_configuration& paramConf, int infoIndex, bool normalized,
                            std::unique_ptr<ensemble>& ens);

static void CalculateNegative(std::shared_ptr<info> targetInfo, uint16_t threadIndex,
                              const param_configuration& paramConf, const int infoIndex, const bool normalized,
                              std::unique_ptr<ensemble>& ens);

static void CalculateWind(const logger& log, std::shared_ptr<info> targetInfo, uint16_t threadIndex,
                          const param_configuration& paramConf, int infoIndex, bool normalized,
                          std::unique_ptr<ensemble>& ens1, std::unique_ptr<ensemble>& ens2);

void probability::Calculate(uint16_t threadIndex, const param_configuration& pc)
{
	info myTargetInfo = *itsInfo;

	auto threadedLogger = logger("probabilityThread # " + std::to_string(threadIndex));
	const std::string deviceType = "CPU";

	const double threshold = pc.gridThreshold;
	const int infoIndex = pc.targetInfoIndex;
	const int ensembleSize = itsEnsembleSize;
	const bool normalized = itsUseNormalizedResult;

	myTargetInfo.First();

	std::unique_ptr<ensemble> ens1;
	std::unique_ptr<ensemble> ens2;  // used with wind calculation

	if (itsUseLaggedEnsemble)
	{
		threadedLogger.Info("Using lagged ensemble for ensemble #1");
		ens1 = std::unique_ptr<ensemble>(
		    new lagged_ensemble(pc.parameter, ensembleSize, kHourResolution, itsLag, itsLaggedSteps + 1));
	}
	else
	{
		ens1 = std::unique_ptr<ensemble>(new ensemble(pc.parameter, ensembleSize));
	}
	ens1->MaximumMissingForecasts(itsMaximumMissingForecasts);

	if (pc.parameter.Name() == "U-MS" || pc.parameter.Name() == "V-MS")
	{
		// Wind
		if (itsUseLaggedEnsemble)
		{
			threadedLogger.Info("Using lagged ensemble for ensemble #2");
			ens2 = std::unique_ptr<ensemble>(
			    new lagged_ensemble(pc.parameter2, ensembleSize, kHourResolution, itsLag, itsLaggedSteps + 1));
		}
		else
		{
			ens2 = std::unique_ptr<ensemble>(new ensemble(pc.parameter2, ensembleSize));
		}
		ens2->MaximumMissingForecasts(itsMaximumMissingForecasts);
	}

	// NOTE we only loop through the time steps here
	do
	{
		threadedLogger.Info("Calculating " + pc.output.Name() + " time " +
		                    static_cast<std::string>(myTargetInfo.Time().ValidDateTime()) + " threshold '" +
		                    std::to_string(threshold) + "' infoIndex " + std::to_string(infoIndex));

		//
		// Setup input data, data fetching
		//

		try
		{
			ens1->Fetch(itsConfiguration, myTargetInfo.Time(), myTargetInfo.Level());

			if (pc.parameter.Name() == "U-MS" || pc.parameter.Name() == "V-MS")
			{
				ens2->Fetch(itsConfiguration, myTargetInfo.Time(), myTargetInfo.Level());
			}
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
				abort();
			}
		}

		// Output memory
		if (itsConfiguration->UseDynamicMemoryAllocation())
		{
			AllocateMemory(myTargetInfo);
		}
		assert(myTargetInfo.Data().Size() > 0);

		//
		// Choose the correct calculation function for this parameter and do the actual calculation
		//
		// Unfortunately we use both the input parameter name and output parameter name for doing this.
		//
		if (pc.parameter.Name() == "U-MS" || pc.parameter.Name() == "V-MS")
		{
			CalculateWind(threadedLogger, std::make_shared<info>(myTargetInfo), threadIndex, pc, infoIndex, normalized,
			              ens1, ens2);
		}
		else if (pc.output.Name() == "PROB-TC-0" || pc.output.Name() == "PROB-TC-1" ||
		         pc.output.Name() == "PROB-TC-2" || pc.output.Name() == "PROB-TC-3" ||
		         pc.output.Name() == "PROB-TC-4" || pc.output.Name() == "PROB-WATLEV-LOW-1")
		{
			CalculateNegative(std::make_shared<info>(myTargetInfo), threadIndex, pc, infoIndex, normalized, ens1);
		}
		else
		{
			CalculateNormal(std::make_shared<info>(myTargetInfo), threadIndex, pc, infoIndex, normalized, ens1);
		}

		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo.Data().MissingCount());
			itsConfiguration->Statistics()->AddToValueCount(myTargetInfo.Data().Size());
		}

		// Finally write this parameter out
		WriteToFile(myTargetInfo, pc.targetInfoIndex);

	} while (myTargetInfo.NextTime());

	threadedLogger.Info("[" + deviceType + "] Missing values: " + std::to_string(myTargetInfo.Data().MissingCount()) +
	                    "/" + std::to_string(myTargetInfo.Data().Size()));
}

// Usually himan writes all the parameters out on a call to WriteToFile, but probability calculates
// each parameter separately in separate threads so this makes no sense (writing all out if we've only
// calculated one parameter)
void probability::WriteToFile(const info& targetInfo, size_t targetInfoIndex, write_options opts)
{
	auto writer = GET_PLUGIN(writer);

	writer->WriteOptions(opts);

	auto info = targetInfo;

	info.ResetParam();
	info.ParamIndex(targetInfoIndex);

	if (itsConfiguration->FileWriteOption() == kDatabase || itsConfiguration->FileWriteOption() == kMultipleFiles)
	{
		writer->ToFile(info, itsConfiguration);
	}
	else
	{
		std::lock_guard<std::mutex> lock(singleFileWriteMutex);

		writer->ToFile(info, itsConfiguration, itsConfiguration->ConfigurationFile());
	}

	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		DeallocateMemory(info);
	}
}

double GetThreshold(std::shared_ptr<info>& targetInfo, const param_configuration& paramConf, bool isGrid)
{
	if (isGrid)
	{
		return paramConf.gridThreshold;
	}
	else
	{
		const int stationId = targetInfo->Station().Id();
		const auto iter = paramConf.stationThreshold.find(stationId);

		if (iter == paramConf.stationThreshold.end())
		{
			throw std::runtime_error("Threshold for station " + std::to_string(stationId) + " not found");
		}

		return iter->second;
	}
}

void CalculateWind(const logger& log, std::shared_ptr<info> targetInfo, uint16_t threadIndex,
                   const param_configuration& paramConf, int infoIndex, bool normalized,
                   std::unique_ptr<ensemble>& ens1, std::unique_ptr<ensemble>& ens2)
{
	targetInfo->ParamIndex(infoIndex);
	targetInfo->ResetLocation();
	ens1->ResetLocation();
	ens2->ResetLocation();

	const size_t ensembleSize = ens1->Size();
	if (ensembleSize != ens2->Size())
	{
		log.Fatal(" CalculateWind(): U and V ensembles are of different size, aborting");
		abort();
	}

	const double invN =
	    normalized ? 1.0 / static_cast<double>(ensembleSize) : 100.0 / static_cast<double>(ensembleSize);

	const bool isGrid = (targetInfo->Grid()->Type() != kPointList);

	while (targetInfo->NextLocation() && ens1->NextLocation() && ens2->NextLocation())
	{
		double probability = 0.0;
		const double threshold = GetThreshold(targetInfo, paramConf, isGrid);

		for (size_t i = 0; i < ensembleSize; i++)
		{
			const auto u = ens1->Value(i);
			const auto v = ens2->Value(i);

			if (IsMissing(u) || IsMissing(v))
			{
				continue;
			}

			if (Magnitude(u, v) >= threshold)
			{
				probability += invN;
			}
		}
		targetInfo->Value(probability);
	}
}

void CalculateNegative(std::shared_ptr<info> targetInfo, uint16_t threadIndex, const param_configuration& paramConf,
                       int infoIndex, bool normalized, std::unique_ptr<ensemble>& ens)
{
	targetInfo->ParamIndex(infoIndex);
	targetInfo->ResetLocation();
	ens->ResetLocation();

	const size_t ensembleSize = ens->Size();
	const double invN =
	    normalized ? 1.0 / static_cast<double>(ensembleSize) : 100.0 / static_cast<double>(ensembleSize);

	const bool isGrid = (targetInfo->Grid()->Type() != kPointList);

	while (targetInfo->NextLocation() && ens->NextLocation())
	{
		double probability = 0.0;
		const double threshold = GetThreshold(targetInfo, paramConf, isGrid);

		for (size_t i = 0; i < ensembleSize; i++)
		{
			const auto x = ens->Value(i);
			if (!IsMissing(x) && (x <= threshold))
			{
				probability += invN;
			}
		}
		targetInfo->Value(probability);
	}
}

void CalculateNormal(std::shared_ptr<info> targetInfo, uint16_t threadIndex, const param_configuration& paramConf,
                     int infoIndex, bool normalized, std::unique_ptr<ensemble>& ens)
{
	targetInfo->ParamIndex(infoIndex);
	targetInfo->ResetLocation();
	ens->ResetLocation();

	const size_t ensembleSize = ens->Size();
	const double invN =
	    normalized ? 1.0 / static_cast<double>(ensembleSize) : 100.0 / static_cast<double>(ensembleSize);

	const bool isGrid = (targetInfo->Grid()->Type() != kPointList);

	while (targetInfo->NextLocation() && ens->NextLocation())
	{
		double probability = 0.0;
		const double threshold = GetThreshold(targetInfo, paramConf, isGrid);

		for (size_t i = 0; i < ensembleSize; i++)
		{
			const auto x = ens->Value(i);
			if (!IsMissing(x) && (x >= threshold))
			{
				probability += invN;
			}
		}
		targetInfo->Value(probability);
	}
}

}  // plugin

}  // namespace
