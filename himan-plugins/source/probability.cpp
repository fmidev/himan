// vim: set ai shiftwidth=4 softtabstop=4 tabstop=4
#include "probability.h"

#include <iostream>
#include <algorithm>
#include <exception>

#include <math.h>

#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>

#include "logger_factory.h"
#include "plugin_factory.h"

#include "fetcher.h"
#include "writer.h"

#include "util.h"
#include "ensemble.h"

namespace himan
{
namespace plugin
{

static const std::string kClassName = "himan::plugin::probability";

/// @brief Used for calculating wind vector magnitude
static inline double Magnitude(double u, double v) { return sqrt(u * u + v * v); }

probability::probability()
{
	itsClearTextFormula = "???";
	itsCudaEnabledCalculation = false;
	itsLogger = logger_factory::Instance()->GetLog("probability");

	itsEnsembleSize = 0;
	itsUseNormalizedResult = false;
}

probability::~probability() {}

/// @brief Gnarrrly configuration reading 
/// @param outParamConfig is modified to have information about the threshold value and input parameters
/// @returns param to be pushed in the calculatedParams vector in Process()
static param GetConfigurationParameter(const std::string& name, const std::shared_ptr<const plugin_configuration> conf,
                                param_configuration* outParamConfig)
{
	long univId = 0;

	if (conf->ParameterExists(name))
	{
		const auto paramOpts = conf->GetParameterOptions(name);

		param param1;
		param param2;

		// NOTE These are plugin dependent
		for (auto&& p : paramOpts)
		{
			if (p.first == "univ_id")
			{
				univId = boost::lexical_cast<long>(p.second);
			}
			else if (p.first == "threshold")
			{
				outParamConfig->threshold = boost::lexical_cast<double>(p.second);
			}
			else if (p.first == "input_param1")
			{
				param1.Name(p.second);
			}
			else if (p.first == "input_param2")
			{
				param2.Name(p.second);
			}
			else if (p.first == "input_id1")
			{
				param1.UnivId(boost::lexical_cast<long>(p.second));
			}
			else if (p.first == "input_id2")
			{
				param2.UnivId(boost::lexical_cast<long>(p.second));
			}
			else
			{
				std::cout << "?? " << p.first << " " << p.second << "\n";
			}
		}

		if (param1.Name() == "XX-X" || param1.UnivId() == kHPMissingInt)
		{
			throw std::runtime_error("probability : configuration error:: input parameter not specified for '" + name +
			                         "'");
		}
		outParamConfig->parameter = param1;

		// NOTE param2 is used only with wind calculation at the moment
		if (param2.Name() != "XX-X" && param2.UnivId() != kHPMissingInt)
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

	assert(univId != 0);

	return param(name, univId);
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
	if (itsConfiguration->Exists("ensembleSize"))
	{
		const int ensembleSize = boost::lexical_cast<int>(itsConfiguration->GetValue("ensembleSize"));
		if (ensembleSize <= 0)
		{
			throw std::runtime_error(ClassName() + " invalid ensembleSize in plugin configuration");
		}
		itsEnsembleSize = ensembleSize;
	}
	else
	{
		throw std::runtime_error(ClassName() + " ensembleSize not specified in plugin configuration");
	}

	// Find out whether we want probabilities in [0,1] range or [0,100] range
	if (itsConfiguration->Exists("normalizedResults"))
	{
		const std::string useNormalized = itsConfiguration->GetValue("normalizedResults");

		itsUseNormalizedResult = (useNormalized == "true") ? true : false;
	}
	else
	{
		// default to [0,100] for compatibility
		itsUseNormalizedResult = false;
		itsLogger->Info("'useNormalized' not found from the configuration, results will be written in [0,100] range");
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

		if (p.Name() == "" || p.UnivId() == 0)
		{
			throw std::runtime_error(ClassName() + "Misconfigured parameter definition in JSON");
		}

		itsParamConfigurations.push_back(config);
		calculatedParams.push_back(p);
		targetInfoIndex++;
	}

	SetParams(calculatedParams);

	// NOTE
	// NOTE We circumvent the usual Himan Start => Run => RunAll / RunTimeDimension - control flow structure
	// NOTE

	//
	// 3. Process parameters in LIFO order by spawning threads for each parameter.
	//	  (We don't divide timesteps between threads)
	//
	boost::thread_group g;
	auto paramConfigurations = itsParamConfigurations;

	// I only get 3 threads even if I specify -j [6,12]? 
	// itsThreadCount = 6;

	while (!paramConfigurations.empty())
	{
		for (int threadIdx = 0; threadIdx < itsThreadCount; threadIdx++)
		{
			if (paramConfigurations.empty())
			{
				break;
			}

			const auto& pc = paramConfigurations.back();
			paramConfigurations.pop_back();

			g.add_thread(new boost::thread(&himan::plugin::probability::Calculate, this, threadIdx, pc));
		}

		g.join_all();
	}

	Finish();
}

static void CalculateNormal(std::shared_ptr<info> targetInfo, uint16_t threadIndex, const double threshold,
                            const int infoIndex, const int ensembleSize, const bool normalized, ensemble& ens);

static void CalculateNegative(std::shared_ptr<info> targetInfo, uint16_t threadIndex, const double threshold,
                              const int infoIndex, const int ensembleSize, const bool normalized, ensemble& ens);

static void CalculateWind(std::shared_ptr<info> targetInfo, uint16_t threadIndex, const double threshold,
                          const int infoIndex, const int ensembleSize, const bool normalized, ensemble& ens1,
                          ensemble& ens2);

void probability::Calculate(uint16_t threadIndex, const param_configuration& pc)
{
	auto myTargetInfo = *itsInfo;

	auto threadedLogger =
	    logger_factory::Instance()->GetLog("probabilityThread # " + boost::lexical_cast<std::string>(threadIndex));
	const std::string deviceType = "CPU";

	const double threshold = pc.threshold;
	const int infoIndex = pc.targetInfoIndex;
	const int ensembleSize = itsEnsembleSize;
	const bool normalized = itsUseNormalizedResult;

	// used with all calculations
	ensemble ens1(pc.parameter, ensembleSize);
	// used with wind calculation
	ensemble ens2;

	if (pc.parameter.Name() == "U-MS" || pc.parameter.Name() == "V-MS")
	{
		// Wind
		ens2 = ensemble(pc.parameter2, ensembleSize);
	}

	myTargetInfo.ResetTime();
	myTargetInfo.First();

	// NOTE we only loop through the time steps here
	do
	{
		threadedLogger->Info("Calculating " + pc.output.Name() + " time " +
		                     static_cast<std::string>(myTargetInfo.Time().ValidDateTime()) + " threshold '" +
		                     boost::lexical_cast<std::string>(threshold) + "' infoIndex " +
		                     boost::lexical_cast<std::string>(infoIndex));

		//
		// Setup input data, data fetching
		//
		ens1.Fetch(itsConfiguration, myTargetInfo.Time(), myTargetInfo.Level());

		if (pc.parameter.Name() == "U-MS" || pc.parameter.Name() == "V-MS")
		{
			ens2.Fetch(itsConfiguration, myTargetInfo.Time(), myTargetInfo.Level());
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
			CalculateWind(std::make_shared<info>(myTargetInfo), threadIndex, threshold, infoIndex, ensembleSize,
			              normalized, ens1, ens2);
		}
		else if (pc.output.Name() == "PROB-TC-0" || pc.output.Name() == "PROB-TC-1" ||
		         pc.output.Name() == "PROB-TC-2" || pc.output.Name() == "PROB-TC-3" || pc.output.Name() == "PROB-TC-4")
		{
			CalculateNegative(std::make_shared<info>(myTargetInfo), threadIndex, threshold, infoIndex, ensembleSize,
			                  normalized, ens1);
		}
		else
		{
			CalculateNormal(std::make_shared<info>(myTargetInfo), threadIndex, threshold, infoIndex, ensembleSize,
			                normalized, ens1);
		}


		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(myTargetInfo.Data().MissingCount());
			itsConfiguration->Statistics()->AddToValueCount(myTargetInfo.Data().Size());
		}

		// Finally write this parameter out
		WriteToFile(myTargetInfo, pc.targetInfoIndex);

	} while (myTargetInfo.NextTime());
}

// Usually himan writes all the parameters out on a call to WriteToFile, but probability calculates
// each parameter separately in separate threads so this makes no sense (writing all out if we've only
// calculated one parameter)
void probability::WriteToFile(const info& targetInfo, const size_t targetInfoIndex, write_options opts)
{
	auto writer = GET_PLUGIN(writer);

	writer->WriteOptions(opts);

	auto info = targetInfo;

	info.ResetParam();
	info.ParamIndex(targetInfoIndex);

	writer->ToFile(info, itsConfiguration);

	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		DeallocateMemory(info);
	}
}

void CalculateWind(std::shared_ptr<info> targetInfo, uint16_t threadIndex, const double threshold, const int infoIndex,
                   const int ensembleSize, const bool normalized, ensemble& ens1, ensemble& ens2)
{
	targetInfo->ParamIndex(infoIndex);
	targetInfo->ResetLocation();
	ens1.ResetLocation();
	ens2.ResetLocation();

	const double invN =
	    normalized ? 1.0 / static_cast<double>(ensembleSize) : 100.0 / static_cast<double>(ensembleSize);

	while (targetInfo->NextLocation() && ens1.NextLocation() && ens2.NextLocation())
	{
		double probability = 0.0;

		for (size_t i = 0; i < ens1.Size(); i++)
		{
			if (Magnitude(ens1.Value(i), ens2.Value(i)) >= threshold)
			{
				probability += invN;
			}
		}
		targetInfo->Value(probability);
	}
}

void CalculateNegative(std::shared_ptr<info> targetInfo, uint16_t threadIndex, const double threshold,
                       const int infoIndex, const int ensembleSize, const bool normalized, ensemble& ens)
{
	targetInfo->ParamIndex(infoIndex);
	targetInfo->ResetLocation();
	ens.ResetLocation();

	const double invN =
	    normalized ? 1.0 / static_cast<double>(ensembleSize) : 100.0 / static_cast<double>(ensembleSize);

	while (targetInfo->NextLocation() && ens.NextLocation())
	{
		double probability = 0.0;

		for (size_t i = 0; i < ens.Size(); i++)
		{
			if (ens.Value(i) <= threshold)
			{
				probability += invN;
			}
		}
		targetInfo->Value(probability);
	}
}

void CalculateNormal(std::shared_ptr<info> targetInfo, uint16_t threadIndex, const double threshold,
                     const int infoIndex, const int ensembleSize, const bool normalized, ensemble& ens)
{
	targetInfo->ParamIndex(infoIndex);
	targetInfo->ResetLocation();
	ens.ResetLocation();

	const double invN =
	    normalized ? 1.0 / static_cast<double>(ensembleSize) : 100.0 / static_cast<double>(ensembleSize);

	while (targetInfo->NextLocation() && ens.NextLocation())
	{
		double probability = 0.0;

		for (size_t i = 0; i < ens.Size(); i++)
		{
			if (ens.Value(i) >= threshold)
			{
				probability += invN;
			}
		}
		targetInfo->Value(probability);
	}
}

}  // plugin

}  // namespace
