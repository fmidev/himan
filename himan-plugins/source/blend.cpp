#include "blend.h"
#include "fetcher.h"
#include "plugin_factory.h"
#include "radon.h"
#include "util.h"
#include "writer.h"

#include <future>
#include <mutex>
#include <thread>

//
// Forecast blender
//

namespace himan
{
namespace plugin
{
using namespace std;

static mutex singleFileWriteMutex;

static const string kClassName = "himan::plugin::blend";

blend::blend() { itsLogger = logger("blend"); }

blend::~blend() {}

static info_t FetchWithProperties(shared_ptr<plugin_configuration> cnf, const forecast_time& forecastTime,
                                  HPTimeResolution stepResolution, const level& lvl, const param& parm,
                                  const forecast_type& type, const string& geom, const producer& prod)
{
	auto f = GET_PLUGIN(fetcher);

	forecast_time ftime = forecastTime;
	ftime.StepResolution(stepResolution);

	cnf->SourceGeomNames({geom});
	cnf->SourceProducers({prod});

	try
	{
		info_t ret = f->Fetch(cnf, ftime, lvl, parm, type, false);
		return ret;
	}
	catch (HPExceptionType& e)
	{
		throw;
	}
}

static vector<meta> ParseProducerOptions(shared_ptr<const plugin_configuration> conf);
static vector<double> OpenAndParseWeightsFile(const std::string& wf);

void blend::Process(shared_ptr<const plugin_configuration> conf)
{
	Init(conf);
	vector<param> params;

	// Each parameter has to be processed with a separate process queue invocation.
	const string p = conf->Exists("param") ? conf->GetValue("param") : "";
	if (p.empty())
	{
		throw std::runtime_error(ClassName() + ": parameter not specified");
	}
	params.push_back(param(p));

	itsMetaOpts = ParseProducerOptions(conf);

	// Weights are stored in a separate file and read after parsing the JSON configuration.
	// We'll check that the number of weights matches the number of forecasts.
	const string wf = conf->Exists("weights_file") ? conf->GetValue("weights_file") : "";
	if (wf.empty())
	{
		throw std::runtime_error(ClassName() + ": weights file not specified");
	}
	itsWeights = OpenAndParseWeightsFile(wf);

	if (itsWeights.size() != itsMetaOpts.size())
	{
		throw std::runtime_error(ClassName() +
		                         ": weights file and configuration doesn't have the"
		                         " same number of elements '" +
		                         to_string(itsWeights.size()) + "' and '" + to_string(itsMetaOpts.size()) +
		                         "', respectively");
	}

	PrimaryDimension(kTimeDimension);
	SetupOutputForecastTypes(itsInfo);
	SetParams(params);
	Start();
}

void blend::Calculate(shared_ptr<info> targetInfo, unsigned short threadIndex)
{
	auto log = logger("blendThread#" + to_string(threadIndex));
	const string deviceType = "CPU";

	forecast_time currentTime = targetInfo->Time();
	HPTimeResolution currentResolution = currentTime.StepResolution();
	const level currentLevel = targetInfo->Level();
	const param currentParam = targetInfo->Param();

	auto f = GET_PLUGIN(fetcher);

	log.Info("Blending " + currentParam.Name() + " " + static_cast<string>(currentTime.ValidDateTime()));

	// Construct a bunch of jobs for parallel fetching.
	vector<future<info_t>> futures;

	for (const auto& m : itsMetaOpts)
	{
		futures.push_back(async(launch::async, [=]() {
			auto cnf = make_shared<plugin_configuration>(*itsConfiguration);
			return FetchWithProperties(cnf, currentTime, currentResolution, m.lvl, currentParam, m.type, m.geom,
			                           m.prod);
		}));
	}

	// Wait for all fetches to reach completion. Currently we don't allow
	// fetches to fail (missing data).
	bool allDone = true;
	do
	{
		allDone = true;
		for (auto& f : futures)
		{
			std::future_status status = f.wait_for(std::chrono::microseconds(1));
			if (status != future_status::ready)
			{
				allDone = false;
			}
		}
	} while (!allDone);

	vector<info_t> forecasts;
	forecasts.reserve(itsMetaOpts.size());

	targetInfo->FirstForecastType();
	size_t findex = 0;
	for (auto& f : futures)
	{
		info_t Info = f.get();
		forecasts.push_back(Info);
		findex++;

		// Copy the underlying data :(
		targetInfo->Data() = Info->Data();

		if (!targetInfo->NextForecastType())
		{
			log.Info("Not enough forecast types defined for target info, breaking");
			break;
		}
	}

	if (!targetInfo->ForecastType(forecast_type(kEpsControl, 0.0)))
	{
		throw runtime_error(ClassName() + ": unable to select the control forecast for the target info");
	}
	else
	{
		//log.Info("creating 'control forecast' grid");
		//targetInfo->Grid(shared_ptr<grid>(itsInfo->Grid()->Clone()));
	}

	// First reset all the locations for the upcoming loop.
	for (const auto& f : forecasts)
	{
		f->ResetLocation();
	}

	// Used to accumulate values (in the case of missing values).
	vector<double> values;
	values.reserve(forecasts.size());

	// To store information about missing forecast values.
	vector<bool> valid;
	valid.reserve(forecasts.size());

	// Create a copy of the weights for possible rebalancing.
	vector<double> currentWeights = itsWeights;

	targetInfo->ResetLocation();
	while (targetInfo->NextLocation())
	{
		double F = 0.0;

		size_t numMissing = 0;
		size_t findex = 0;
		for (const auto& f : forecasts)
		{
			if (!f->NextLocation())
			{
				// XXX All the infos should have the same grid by now.
				log.Warning("unable to advance location iterator position");
				continue;
			}

			const double v = f->Value();
			if (v == kFloatMissing)
			{
				numMissing++;
				valid[findex] = false;
			}
			else
			{
				valid[findex] = true;
			}

			values.push_back(v);
			findex++;
		}

		if (numMissing == forecasts.size())
		{
			F = kFloatMissing;
		}
		else
		{
			// Rebalance the weights in the case of 'missing values'.
			size_t i = 0;
			for (const double& v : values)
			{
				if (v == kFloatMissing)
				{
					const double w = currentWeights[i] / static_cast<double>(values.size() - numMissing);

					for (size_t wi = 0; wi < currentWeights.size(); wi++)
					{
						if (valid[wi])
						{
							currentWeights[wi] += w;
						}
						else
						{
							if (wi <= i)
							{
								currentWeights[wi] = 0.0;
							}
						}
					}
				}
				i++;
			}

			// Apply the weights to the forecast values. Missing values are not considered here
			// since this will result in a multiplication by 0.
			for (const auto&& tup : zip_range(currentWeights, values))
			{
				const double w = tup.get<0>();
				const double v = tup.get<1>();
				F += w * v;
			}
		}

		targetInfo->Value(F);

		values.clear();
	}

	log.Info("[" + deviceType + "] Missing values: " + to_string(targetInfo->Data().MissingCount()) + "/" +
	         to_string(targetInfo->Data().Size()));
}

void blend::WriteToFile(const info& targetInfo, write_options writeOptions)
{
	auto aWriter = GET_PLUGIN(writer);

	aWriter->WriteOptions(writeOptions);
	auto tempInfo = targetInfo;

	tempInfo.ResetForecastType();

	while (tempInfo.NextForecastType())
	{
		if (itsConfiguration->FileWriteOption() == kDatabase || itsConfiguration->FileWriteOption() == kMultipleFiles)
		{
			aWriter->ToFile(tempInfo, itsConfiguration);
		}
		else
		{
			lock_guard<mutex> lock(singleFileWriteMutex);

			aWriter->ToFile(tempInfo, itsConfiguration, itsConfiguration->ConfigurationFile());
		}
	}

	if (itsConfiguration->UseDynamicMemoryAllocation())
	{
		DeallocateMemory(targetInfo);
	}
}

// Output original forecasts as perturbations and the generated 'blend' as the control forecast.
void blend::SetupOutputForecastTypes(shared_ptr<info> Info)
{
	vector<forecast_type> ftypes;

	for (size_t index = 1; index <= itsMetaOpts.size(); index++)
	{
		ftypes.push_back(forecast_type(kEpsPerturbation, static_cast<double>(index)));
	}

	ftypes.push_back(forecast_type(kEpsControl, 0.0));
	Info->ForecastTypes(ftypes);
}

vector<meta> ParseProducerOptions(shared_ptr<const plugin_configuration> conf)
{
	vector<meta> metaOpts;

	auto R = GET_PLUGIN(radon);

	auto producers = conf->GetParameterNames();
	for (const auto& p : producers)
	{
		const string producerName = p;
		string geom;
		string levelType;
		int levelValue = -1;
		vector<forecast_type> ftypes;

		auto options = conf->GetParameterOptions(p);
		for (const auto& option : options)
		{
			if (option.first == "forecast_type")
			{
				vector<string> types = himan::util::Split(option.second, ",", false);
				for (const string& type : types)
				{
					HPForecastType ty;

					if (type.find("pf") != string::npos)
					{
						ty = kEpsPerturbation;
						string list = "";
						for (size_t i = 2; i < type.size(); i++)
						{
							list += type[i];
						}

						vector<string> range = himan::util::Split(list, "-", false);
						if (range.size() == 1)
						{
							ftypes.push_back(forecast_type(ty, stoi(range[0])));
						}
						else
						{
							assert(range.size() == 2);
							int current = stoi(range[0]);
							const int stop = stoi(range[1]);

							while (current <= stop)
							{
								ftypes.push_back(forecast_type(ty, current));
								current++;
							}
						}
					}
					else
					{
						if (type == "cf")
						{
							ftypes.push_back(forecast_type(kEpsControl, 0));
						}
						else if (type == "deterministic")
						{
							ftypes.push_back(forecast_type(kDeterministic));
						}
						else
						{
							throw runtime_error("invalid forecast_type: " + type);
						}
					}
				}
			}
			else if (option.first == "geom")
			{
				geom = option.second;
			}
			else if (option.first == "leveltype")
			{
				levelType = option.second;
			}
			else if (option.first == "level")
			{
				levelValue = stoi(option.second);
			}
			else
			{
				throw runtime_error(kClassName + ": invalid configuration " + option.first + " : " + option.second);
			}
		}

		// Fetch producer information from database by producer name
		// Problems:
		// - easy to configure wrong geometry for a producer (this is not checked, and only seen when we don't find
		//   data)
		auto producerDefinition = R->RadonDB().GetProducerDefinition(producerName);
		if (producerDefinition.empty())
		{
			throw runtime_error(kClassName + ": producer information not found for '" + producerName + "'");
		}

		const long prodId = stol(producerDefinition["producer_id"]);
		const string name = producerDefinition["ref_prod"];
		const long centre = stol(producerDefinition["ident_id"]);
		const long ident = stol(producerDefinition["model_id"]);
		const producer prod(prodId, centre, ident, name);

		if (levelValue != -1 && !levelType.empty() && !geom.empty() && !ftypes.empty())
		{
			const level lvl = level(HPStringToLevelType.at(levelType), levelValue);
			for (const auto& f : ftypes)
			{
				meta m;
				m.prod = prod;
				m.geom = geom;
				m.type = f;
				m.lvl = lvl;
				metaOpts.push_back(m);
			}
		}
		else
		{
			throw runtime_error(kClassName + ": missing producer information");
		}
	}
	return metaOpts;
}

std::vector<double> OpenAndParseWeightsFile(const std::string& wf)
{
	// Do the reading after reading the configuration. CHECK that the number of weights matches the number of
	// members we're going to be reading!
	std::vector<double> values;

	std::ifstream in(wf);
	std::string line;

	while (std::getline(in, line))
	{
		std::istringstream stream(line);

		double value;
		while (stream >> value)
		{
			values.push_back(value);
		}
	}

	return values;
}

}  // namespace plugin
}  // namespace himan
