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
		if (e != kFileDataNotFound)
		{
			himan::Abort();
		}
		else
		{
			return nullptr;
		}
	}
}

static vector<meta> ParseProducerOptions(shared_ptr<const plugin_configuration> conf);

static mutex dimensionMutex;

// We overload Start & Run functions since we don't want to step through the forecast types defined in targetInfo.
// Presumably we don't want to iterate over levels either. (Not sure how that would be configured.)
void blend::Run(unsigned short threadIndex)
{
	// Iterate through timesteps
	auto targetInfo = make_shared<info>(*itsInfo);

	while (itsDimensionsRemaining)
	{
		{
			lock_guard<mutex> lock(dimensionMutex);

			if (itsInfo->NextTime())
			{
				if (!targetInfo->Time(itsInfo->Time()))
				{
					throw std::runtime_error("invalid target time");
				}
			}
			else
			{
				itsDimensionsRemaining = false;
				break;
			}
		}

		Calculate(targetInfo, threadIndex);

		if (itsConfiguration->StatisticsEnabled())
		{
			itsConfiguration->Statistics()->AddToMissingCount(targetInfo->Data().MissingCount());
			itsConfiguration->Statistics()->AddToValueCount(targetInfo->Data().Size());
		}
		WriteToFile(*targetInfo);
	}
}

void blend::Start()
{
	itsThreadCount = static_cast<short>(itsInfo->SizeTimes());

	vector<thread> threads;
	threads.reserve(itsThreadCount);

	itsInfo->Reset();
	itsInfo->FirstForecastType();
	itsInfo->FirstLevel();
	itsInfo->FirstParam();

	for (unsigned short i = 0; i < itsThreadCount; i++)
	{
		threads.push_back(move(thread(&blend::Run, this, i)));
	}

	for (auto&& t : threads)
	{
		t.join();
	}

	Finish();
}

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

	PrimaryDimension(kTimeDimension);
	SetupOutputForecastTypes(itsInfo);
	SetParams(params);
	Start();
}

void blend::Calculate(shared_ptr<info> targetInfo, unsigned short threadIndex)
{
	auto f = GET_PLUGIN(fetcher);
	auto log = logger("blendThread#" + to_string(threadIndex));
	const string deviceType = "CPU";

	forecast_time currentTime = targetInfo->Time();
	HPTimeResolution currentResolution = currentTime.StepResolution();
	const level currentLevel = targetInfo->Level();
	const param currentParam = targetInfo->Param();

	log.Info("Blending " + currentParam.Name() + " " + static_cast<string>(currentTime.ValidDateTime()));

	// Construct a bunch of jobs for parallel fetching and interpolation.
	vector<future<info_t>> futures;

	for (const auto& m : itsMetaOpts)
	{
		futures.push_back(async(launch::async, [=]() {
			auto cnf = make_shared<plugin_configuration>(*itsConfiguration);
			return FetchWithProperties(cnf, currentTime, currentResolution, m.lvl, currentParam, m.type, m.geom,
			                           m.prod);
		}));
	}

	// Copy the fetched forecasts into the target info so that we can simply write them out after calculation.
	// TODO: Don't copy the data? Should just point to the already existing data?
	vector<info_t> forecasts;
	forecasts.reserve(itsMetaOpts.size());

	targetInfo->FirstForecastType();
	size_t findex = 0;
	for (auto& f : futures)
	{
		info_t Info = f.get();

		if (!Info)
		{
			continue;
		}

		forecasts.push_back(Info);
		findex++;

		targetInfo->Data() = Info->Data();
		if (!targetInfo->NextForecastType())
		{
			log.Warning("Not enough forecast types defined for target info, breaking");
			break;
		}
	}

	if (forecasts.empty())
	{
		log.Error("Failed to acquire any source data");
		himan::Abort();
	}

	if (!targetInfo->ForecastType(forecast_type(kEpsControl, 0.0)))
	{
		throw runtime_error(ClassName() + ": unable to select the control forecast for the target info");
	}

	// First reset all the locations for the upcoming loop.
	for (const auto& f : forecasts)
	{
		f->ResetLocation();
	}

	// Used to accumulate values in the case of missing values.
	vector<double> values;
	values.reserve(forecasts.size());

	// Pick weights from the current set into this vector
	vector<info_t> weights;
	weights.reserve(forecasts.size());

	vector<double> currentWeights;
	currentWeights.reserve(forecasts.size());

	// Create unit weight grids based on the target info. This is in preparation for the precalculated
	// weight grids stored on disk.
	for (size_t i = 0; i < forecasts.size(); i++)
	{
		auto w = std::make_shared<info>(*targetInfo);
		w->Create(targetInfo->Grid(), true);

		const double unitWeight = 1.0 / static_cast<double>(forecasts.size());
		w->Data().Fill(unitWeight);
		w->ResetLocation();

		weights.push_back(w);
	}

	targetInfo->ResetLocation();
	while (targetInfo->NextLocation())
	{
		double F = 0.0;
		size_t numMissing = 0;

		for (const auto&& tup : zip_range(forecasts, weights))
		{
			info_t f = tup.get<0>();
			info_t w = tup.get<1>();

			if (!f->NextLocation())
			{
				log.Warning("unable to advance forecast location iterator position");
				continue;
			}

			if (!w->NextLocation())
			{
				log.Warning("unable to advance weight location iterator position");
				continue;
			}

			const double v = f->Value();
			const double wv = w->Value();
			if (IsMissing(v) || IsMissing(wv))
			{
				numMissing++;
				continue;
			}

			values.push_back(v);
			currentWeights.push_back(wv);
		}

		if (numMissing == forecasts.size())
		{
			F = MissingDouble();
		}
		else
		{
			double sw = 0.0;
			for (const auto& w : currentWeights)
			{
				sw += w;
			}

			if (sw <= 0.0)
			{
				continue;
			}

			// Finally apply the weights to the forecast values.
			for (const auto&& tup : zip_range(currentWeights, values))
			{
				const double w = tup.get<0>();
				const double v = tup.get<1>();
				F += w * v;
			}

			F /= sw;
		}

		targetInfo->Value(F);

		currentWeights.clear();
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
							ASSERT(range.size() == 2);
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

}  // namespace plugin
}  // namespace himan
