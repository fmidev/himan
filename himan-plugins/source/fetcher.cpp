#include "fetcher.h"
#include "interpolate.h"
#include "logger.h"
#include "plugin_factory.h"
#include "statistics.h"
#include "util.h"
#include <boost/filesystem/operations.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <fstream>
#include <future>

#include "cache.h"
#include "csv.h"
#include "geotiff.h"
#include "grib.h"
#include "param.h"
#include "querydata.h"
#include "radon.h"

using namespace himan;
using namespace himan::plugin;
using namespace std;

// A flag to guard that we read auxiliary files only once

static once_flag oflag;

string GetOtherVectorComponentName(const string& name);

// If multiple plugins are executed using auxiliary files, prevent
// each plugin execution from re-reading the aux files to memory.

static std::atomic<bool> auxiliaryFilesRead(false);

namespace
{
// Sticky param cache will store the producer that provided data.
// With this information we can skip the regular producer loop cycle
// (try prod 1, data not found, try prod 2) which will improve fetching
// times when reading from database.

std::string UniqueName(const himan::producer& prod, const himan::param& par, const himan::level& lev)
{
	return to_string(prod.Id()) + "_" + par.Name() + "_" + himan::HPLevelTypeToString.at(lev.Type());
}

void AmendParamWithAggregationAndProcessingType(param& p)
{
	p.Aggregation(util::GetAggregationFromParamName(p.Name()));
	p.ProcessingType(util::GetProcessingTypeFromParamName(p.Name()));
}
}
static vector<string> stickyParamCache;
static mutex stickyMutex;

// Container to store shared mutexes for data fetch for a single grid.
// The idea is that if multiple threads fetch the *same* data, they
// will be synchronized so that one thread fetches the data and updates
// cache while the other threads will wait for the completion of that task.

static mutex singleFetcherMutex;
map<string, boost::shared_mutex> singleFetcherMap;

string CreateNotFoundString(const vector<producer>& prods, const forecast_type& ftype, const forecast_time& time,
                            const level& lev, const vector<param>& params)
{
	stringstream str;

	str << "No valid data found for producer(s): ";

	for (const auto& prod : prods)
	{
		str << prod.Id() << ",";
	}

	str.seekp(-1, ios_base::end);

	str << " origintime: " << time.OriginDateTime().String() << ", step: " << static_cast<string>(time.Step())
	    << " param(s): ";

	for (const auto& par : params)
	{
		str << par.Name() << ",";
	}

	str.seekp(-1, ios_base::end);

	str << " level: " << static_cast<string>(lev) << " forecast type: " << static_cast<string>(ftype);

	return str.str();
}

namespace
{
template <typename T>
shared_ptr<info<T>> ConvertTo(shared_ptr<info<double>>&);

template <>
shared_ptr<info<double>> ConvertTo(shared_ptr<info<double>>& anInfo)
{
	return anInfo;
}

template <>
shared_ptr<info<float>> ConvertTo(shared_ptr<info<double>>& anInfo)
{
	return make_shared<info<float>>(*anInfo);
}
}  // namespace

fetcher::fetcher()
    : itsDoLevelTransform(true),
      itsDoInterpolation(true),
      itsDoVectorComponentRotation(false),
      itsUseCache(true),
      itsApplyLandSeaMask(false),
      itsLandSeaMaskThreshold(0.5)
{
	itsLogger = logger("fetcher");
}

shared_ptr<info<double>> fetcher::Fetch(shared_ptr<const plugin_configuration> config, forecast_time requestedTime,
                                        level requestedLevel, const params& requestedParams,
                                        forecast_type requestedType, bool readPackedData)
{
	return Fetch<double>(config, requestedTime, requestedLevel, requestedParams, requestedType, readPackedData);
}

template <typename T>
shared_ptr<info<T>> fetcher::Fetch(shared_ptr<const plugin_configuration> config, forecast_time requestedTime,
                                   level requestedLevel, const params& requestedParams, forecast_type requestedType,
                                   bool readPackedData)
{
	shared_ptr<info<T>> ret;

	for (size_t i = 0; i < requestedParams.size(); i++)
	{
		try
		{
			return Fetch<T>(config, requestedTime, requestedLevel, requestedParams[i], requestedType, readPackedData,
			                true);
		}
		catch (const HPExceptionType& e)
		{
			if (e != kFileDataNotFound)
			{
				throw;
			}
		}
	}

	itsLogger.Warning(
	    CreateNotFoundString(config->SourceProducers(), requestedType, requestedTime, requestedLevel, requestedParams));

	throw kFileDataNotFound;
}

template shared_ptr<info<double>> fetcher::Fetch<double>(shared_ptr<const plugin_configuration>, forecast_time, level,
                                                         const params&, forecast_type, bool);
template shared_ptr<info<float>> fetcher::Fetch<float>(shared_ptr<const plugin_configuration>, forecast_time, level,
                                                       const params&, forecast_type, bool);

shared_ptr<info<double>> fetcher::Fetch(shared_ptr<const plugin_configuration> config, forecast_time requestedTime,
                                        level requestedLevel, param requestedParam, forecast_type requestedType,
                                        bool readPackedData, bool suppressLogging)
{
	return Fetch<double>(config, requestedTime, requestedLevel, requestedParam, requestedType, readPackedData,
	                     suppressLogging);
}

template <typename T>
shared_ptr<info<T>> fetcher::Fetch(shared_ptr<const plugin_configuration> config, forecast_time requestedTime,
                                   level requestedLevel, param requestedParam, forecast_type requestedType,
                                   bool readPackedData, bool suppressLogging)
{
	timer t(true);

	AmendParamWithAggregationAndProcessingType(requestedParam);

	// Check sticky param cache first

	shared_ptr<info<T>> ret;

	for (const auto& prod : config->SourceProducers())
	{
		bool found = false;

		{
			const auto uName = UniqueName(prod, requestedParam, requestedLevel);
			lock_guard<mutex> lock(stickyMutex);

			// Linear search, size of stickyParamCache should be relatively small
			if (find(stickyParamCache.begin(), stickyParamCache.end(), uName) != stickyParamCache.end())
			{
				// oh,goody
				found = true;
			}
		}

		if (found)
		{
			search_options opts(requestedTime, requestedParam, requestedLevel, prod, requestedType, config);

			ret = FetchFromProducer<T>(opts, readPackedData, suppressLogging);

			if (ret)
			{
				break;
			}
		}
	}

	if (!ret)
	{
		// first time fetch (not in sticky cache)
		for (const auto& prod : config->SourceProducers())
		{
			search_options opts(requestedTime, requestedParam, requestedLevel, prod, requestedType, config);

			ret = FetchFromProducer<T>(opts, readPackedData, suppressLogging);

			if (ret)
			{
				const auto uName = UniqueName(prod, requestedParam, requestedLevel);

				lock_guard<mutex> lock(stickyMutex);
				if (find(stickyParamCache.begin(), stickyParamCache.end(), uName) == stickyParamCache.end())
				{
					itsLogger.Trace("Updating sticky param cache: " + UniqueName(opts.prod, opts.param, opts.level));
					stickyParamCache.push_back(uName);
				}

				break;
			}
		}
	}

	if (config->StatisticsEnabled())
	{
		t.Stop();

		config->Statistics()->AddToFetchingTime(t.GetTime());
	}

	if (!ret)
	{
		if (!suppressLogging)
		{
			itsLogger.Warning(CreateNotFoundString(config->SourceProducers(), requestedType, requestedTime,
			                                       requestedLevel, {requestedParam}));
		}

		throw kFileDataNotFound;
	}

	return ret;
}

template shared_ptr<info<double>> fetcher::Fetch<double>(shared_ptr<const plugin_configuration>, forecast_time, level,
                                                         param, forecast_type, bool, bool);
template shared_ptr<info<float>> fetcher::Fetch<float>(shared_ptr<const plugin_configuration>, forecast_time, level,
                                                       param, forecast_type, bool, bool);

template <typename T>
shared_ptr<info<T>> fetcher::FetchFromProducerSingle(search_options& opts, bool readPackedData, bool suppressLogging)
{
	level newLevel = opts.level;

	if (itsDoLevelTransform && opts.configuration->DatabaseType() != kNoDatabase &&
	    (opts.level.Type() != kHybrid && opts.level.Type() != kPressure && opts.level.Type() != kHeightLayer))
	{
		newLevel = LevelTransform(opts.configuration, opts.prod, opts.param, opts.level);

		if (newLevel != opts.level || newLevel.Value() != opts.level.Value())
		{
			itsLogger.Trace("Transform level " + static_cast<string>(opts.level) + " to " +
			                static_cast<string>(newLevel) + " for producer " + to_string(opts.prod.Id()) +
			                ", parameter " + opts.param.Name());

			opts.level = newLevel;
		}
	}

	auto ret = FetchFromAllSources<T>(opts, readPackedData);

	auto theInfos = ret.second;

	if (theInfos.empty())
	{
		return shared_ptr<info<T>>();
	}
	else if (ret.first == HPDataFoundFrom::kCache)
	{
		return theInfos[0];
	}

	RotateVectorComponents<T>(theInfos, opts.configuration->BaseGrid(), opts.configuration, opts.prod);

	if (itsDoInterpolation)
	{
		if (!interpolate::Interpolate(opts.configuration->BaseGrid(), theInfos))
		{
			// interpolation failed
			throw kFileDataNotFound;
		}
	}
	else
	{
		itsLogger.Trace("Interpolation disabled");
	}

	if (itsApplyLandSeaMask)
	{
		if (!ApplyLandSeaMask<T>(opts.configuration, theInfos[0], opts.time, opts.ftype))
		{
			itsLogger.Warning("Land sea mask apply failed");
		}
	}

	/*
	 * Insert interpolated data to cache if
	 * 1. Cache is not disabled locally (itsUseCache) AND
	 * 2. Cache is not disabled globally (config->UseCache()) AND
	 * 3. Data is not packed
	 */

	if (ret.first != HPDataFoundFrom::kCache && itsUseCache && opts.configuration->UseCacheForReads() &&
	    !theInfos[0]->PackedData()->HasData())
	{
		auto c = GET_PLUGIN(cache);
		c->Insert<T>(theInfos[0]);
	}

	ASSERT((theInfos[0]->Time()) == opts.time);
	ASSERT((theInfos[0]->Param()) == opts.param);

	return theInfos[0];
}

template shared_ptr<info<double>> fetcher::FetchFromProducerSingle<double>(search_options&, bool, bool);
template shared_ptr<info<float>> fetcher::FetchFromProducerSingle<float>(search_options&, bool, bool);

template <typename T>
shared_ptr<info<T>> fetcher::FetchFromProducer(search_options& opts, bool readPackedData, bool suppressLogging)
{
	// When reading packed data, data is not pushed to cache because it's only unpacked
	// later. Therefore there is no reason to synchronize thread access.
	// TODO: *should* data be unpacked and pushed to cache (it's done so anyway later)?
	if (readPackedData)
	{
		return FetchFromProducerSingle<T>(opts, readPackedData, suppressLogging);
	}

	const auto uname = util::UniqueName(opts);
	pair<map<string, boost::shared_mutex>::iterator, bool> muret;

	// First acquire mutex to (possibly) modify map
	unique_lock<mutex> sflock(singleFetcherMutex);

	muret = singleFetcherMap.emplace(piecewise_construct, forward_as_tuple(uname), forward_as_tuple());

	if (muret.second == true)
	{
		// first time this data is being fetched: take exclusive lock to prevent
		// other threads from fetching the same data at the same time
		boost::unique_lock<boost::shared_mutex> uniqueLock(muret.first->second);
		sflock.unlock();

		return FetchFromProducerSingle<T>(opts, readPackedData, suppressLogging);
	}

	sflock.unlock();

	// this data is being fetched right now by other thread, or it has been fetched
	// earlier
	boost::shared_lock<boost::shared_mutex> lock(muret.first->second);

	return FetchFromProducerSingle<T>(opts, readPackedData, suppressLogging);
}

template shared_ptr<info<double>> fetcher::FetchFromProducer<double>(search_options&, bool, bool);
template shared_ptr<info<float>> fetcher::FetchFromProducer<float>(search_options&, bool, bool);

template <typename T>
vector<shared_ptr<info<T>>> fetcher::FromFile(const vector<file_information>& files, search_options& options,
                                              bool readPackedData, bool readIfNotMatching)
{
	vector<shared_ptr<info<T>>> allInfos;

	for (const auto& inputFile : files)
	{
		if (inputFile.storage_type == HPFileStorageType::kLocalFileSystem &&
		    !boost::filesystem::exists(inputFile.file_location))
		{
			itsLogger.Error("Input file '" + inputFile.file_location + "' does not exist");
			continue;
		}

		vector<shared_ptr<info<T>>> curInfos;

		switch (inputFile.file_type)
		{
			case kGRIB:
			case kGRIB1:
			case kGRIB2:
			{
				auto g = GET_PLUGIN(grib);
				curInfos = g->FromFile<T>(inputFile, options, readPackedData, readIfNotMatching);
				break;
			}

			case kQueryData:
				throw runtime_error("QueryData as input is not supported");

			case kNetCDF:
				throw runtime_error("NetCDF as input is not supported");

			case kCSV:
			{
				auto c = GET_PLUGIN(csv);
				auto anInfo = c->FromFile<T>(inputFile.file_location, options, readIfNotMatching);
				curInfos.push_back(anInfo);
				break;
			}

			case kGeoTIFF:
			{
				auto g = GET_PLUGIN(geotiff);
				curInfos = g->FromFile<T>(inputFile, options);
				break;
			}

			default:
				// Unknown file type, cannot proceed
				throw runtime_error("Input file is neither GRIB, NetCDF, QueryData nor CSV");
				break;
		}

		allInfos.insert(allInfos.end(), curInfos.begin(), curInfos.end());
	}

	return allInfos;
}

template <typename T>
vector<shared_ptr<info<T>>> fetcher::FromCache(search_options& options)
{
	auto c = GET_PLUGIN(cache);

	return c->GetInfo<T>(options);
}

himan::level fetcher::LevelTransform(const shared_ptr<const configuration>& conf, const producer& sourceProducer,
                                     const param& targetParam, const level& targetLevel) const
{
	level ret = targetLevel;

	HPDatabaseType dbtype = conf->DatabaseType();

	if (ret == targetLevel && dbtype == kRadon)
	{
		auto r = GET_PLUGIN(radon);

		auto levelInfo =
		    r->RadonDB().GetLevelFromDatabaseName(boost::to_upper_copy(HPLevelTypeToString.at(targetLevel.Type())));

		if (levelInfo.empty())
		{
			itsLogger.Warning("Level type '" + HPLevelTypeToString.at(targetLevel.Type()) + "' not found from radon");
			return ret;
		}

		auto paramInfo = r->RadonDB().GetParameterFromDatabaseName(sourceProducer.Id(), targetParam.Name(),
		                                                           stoi(levelInfo["id"]), targetLevel.Value());

		if (paramInfo.empty())
		{
			itsLogger.Trace("Level transform failed: no parameter information for param " + targetParam.Name());
			return ret;
		}

		auto levelXrefInfo = r->RadonDB().GetLevelTransform(sourceProducer.Id(), stoi(paramInfo["id"]),
		                                                    stoi(levelInfo["id"]), targetLevel.Value());

		if (!levelXrefInfo.empty())
		{
			double lvlValue;

			try
			{
				lvlValue = stod(levelXrefInfo["value"]);
			}
			catch (const invalid_argument& e)
			{
				lvlValue = targetLevel.Value();
			}

			HPLevelType lvlType = HPStringToLevelType.at(boost::to_lower_copy(levelXrefInfo["name"]));

			if (lvlType == kGround)
			{
				lvlValue = 0;
			}

			ret = level(lvlType, lvlValue);
		}
	}

	return ret;
}

void fetcher::DoLevelTransform(bool theDoLevelTransform)
{
	itsDoLevelTransform = theDoLevelTransform;
}
bool fetcher::DoLevelTransform() const
{
	return itsDoLevelTransform;
}
void fetcher::DoInterpolation(bool theDoInterpolation)
{
	itsDoInterpolation = theDoInterpolation;
}
bool fetcher::DoInterpolation() const
{
	return itsDoInterpolation;
}
void fetcher::UseCache(bool theUseCache)
{
	itsUseCache = theUseCache;
}
bool fetcher::UseCache() const
{
	return itsUseCache;
}

template <typename T>
pair<HPDataFoundFrom, vector<shared_ptr<info<T>>>> fetcher::FetchFromAllSources(search_options& opts,
                                                                                bool readPackedData)
{
	auto ret = FetchFromCache<T>(opts);

	if (!ret.empty())
	{
		return make_pair(HPDataFoundFrom::kCache, ret);
	}

	if (!auxiliaryFilesRead)
	{
		// second ret, different from first
		auto _ret = FetchFromAuxiliaryFiles(opts, readPackedData);

		if (!_ret.second.empty())
		{
			return make_pair(_ret.first, vector<shared_ptr<info<T>>>({ConvertTo<T>(_ret.second[0])}));
		}
	}

	return make_pair(HPDataFoundFrom::kDatabase, FetchFromDatabase<T>(opts, readPackedData));
}

template pair<HPDataFoundFrom, vector<shared_ptr<info<double>>>> fetcher::FetchFromAllSources<double>(search_options&,
                                                                                                      bool);
template pair<HPDataFoundFrom, vector<shared_ptr<info<float>>>> fetcher::FetchFromAllSources<float>(search_options&,
                                                                                                    bool);

template <typename T>
vector<shared_ptr<info<T>>> fetcher::FetchFromCache(search_options& opts)
{
	vector<shared_ptr<info<T>>> ret;

	if (itsUseCache && opts.configuration->UseCacheForReads())
	{
		// 1. Fetch data from cache
		ret = FromCache<T>(opts);

		if (ret.size())
		{
			if (dynamic_pointer_cast<const plugin_configuration>(opts.configuration)->StatisticsEnabled())
			{
				dynamic_pointer_cast<const plugin_configuration>(opts.configuration)
				    ->Statistics()
				    ->AddToCacheHitCount(1);
			}
		}
	}

	return ret;
}

template vector<shared_ptr<info<double>>> fetcher::FetchFromCache<double>(search_options&);
template vector<shared_ptr<info<float>>> fetcher::FetchFromCache<float>(search_options&);

template <typename T>
vector<shared_ptr<info<T>>> fetcher::FetchFromDatabase(search_options& opts, bool readPackedData)
{
	vector<shared_ptr<info<T>>> ret;

	HPDatabaseType dbtype = opts.configuration->DatabaseType();

	if (!opts.configuration->ReadFromDatabase() || dbtype == kNoDatabase)
	{
		return ret;
	}

	if (opts.prod.Class() == kGridClass)
	{
		vector<file_information> files;

		if (dbtype == kRadon)
		{
			auto r = GET_PLUGIN(radon);

			files = r->Files(opts);
		}

		if (files.size() == 0)
		{
			const string ref_prod = opts.prod.Name();
			const string analtime = opts.time.OriginDateTime().String("%Y%m%d%H%M%S");
			const vector<string> sourceGeoms = opts.configuration->SourceGeomNames();
			itsLogger.Trace("No geometries found for producer " + ref_prod + ", analysistime " + analtime +
			                ", source geom name(s) '" + util::Join(sourceGeoms, ",") + "', param " + opts.param.Name());
		}
		else
		{
			ret = FromFile<T>(files, opts, readPackedData, false);

			if (dynamic_pointer_cast<const plugin_configuration>(opts.configuration)->StatisticsEnabled())
			{
				dynamic_pointer_cast<const plugin_configuration>(opts.configuration)
				    ->Statistics()
				    ->AddToCacheMissCount(1);
			}

			return ret;
		}
	}
	else if (opts.prod.Class() == kPreviClass)
	{
		auto r = GET_PLUGIN(radon);

		auto csv_forecasts = r->CSV(opts);
		auto _ret = util::CSVToInfo<T>(csv_forecasts);

		if (_ret)
		{
			ret.push_back(_ret);
		}
	}

	return ret;
}

template vector<shared_ptr<info<double>>> fetcher::FetchFromDatabase<double>(search_options&, bool);
template vector<shared_ptr<info<float>>> fetcher::FetchFromDatabase<float>(search_options&, bool);

pair<HPDataFoundFrom, vector<shared_ptr<info<double>>>> fetcher::FetchFromAuxiliaryFiles(search_options& opts,
                                                                                         bool readPackedData)
{
	vector<info_t> ret;
	HPDataFoundFrom source = HPDataFoundFrom::kAuxFile;

	if (!opts.configuration->AuxiliaryFiles().empty())
	{
		vector<file_information> files;
		files.reserve(opts.configuration->AuxiliaryFiles().size());

		for (const auto& file : opts.configuration->AuxiliaryFiles())
		{
			file_information f;
			f.file_location = file;
			f.file_type = util::FileType(file);
			f.offset = boost::none;
			f.length = boost::none;
			f.message_no = boost::none;
			f.storage_type = HPFileStorageType::kLocalFileSystem;

			files.push_back(f);
		}

		if (itsUseCache && opts.configuration->UseCacheForReads() && opts.configuration->ReadAllAuxiliaryFilesToCache())
		{
			if (itsApplyLandSeaMask)
			{
				itsLogger.Fatal("Land sea mask cannot be applied when reading all auxiliary files to cache");
				itsLogger.Fatal("Restart himan with command line option --no-auxiliary-file-full-cache-read");
				himan::Abort();
			}

			auto c = GET_PLUGIN(cache);

			call_once(oflag, [&]() {

				itsLogger.Debug("Start full auxiliary files read");

				timer t(true);

				ret = FromFile<double>(files, opts, readPackedData, true);

				AuxiliaryFilesRotateAndInterpolate(opts, ret);

#ifdef HAVE_CUDA
				util::Unpack<double>(ret, false);
#endif

				for (const auto& info : ret)
				{
					info->First();
					info->Reset<param>();

					while (info->Next())
					{
						c->Insert(info);
					}
				}

				t.Stop();
				itsLogger.Debug("Auxiliary files read finished in " + to_string(t.GetTime()) +
				                "ms, cache size: " + to_string(c->Size()));
			});

			auxiliaryFilesRead = true;
			source = HPDataFoundFrom::kCache;

			ret = FromCache<double>(opts);
		}
		else
		{
			ret = FromFile<double>(files, opts, readPackedData, false);
		}

		if (!ret.empty())
		{
			itsLogger.Trace("Data found from auxiliary file(s)");

			if (dynamic_pointer_cast<const plugin_configuration>(opts.configuration)->StatisticsEnabled())
			{
				dynamic_pointer_cast<const plugin_configuration>(opts.configuration)
				    ->Statistics()
				    ->AddToCacheMissCount(1);
			}
		}
		else
		{
			itsLogger.Trace("Data not found from auxiliary file(s)");
		}
	}

	return make_pair(source, ret);
}

void fetcher::AuxiliaryFilesRotateAndInterpolate(const search_options& opts, vector<info_t>& infos)
{
	// Step 1. Rotate if needed

	const grid* baseGrid = opts.configuration->BaseGrid();

	auto eq = [](const info_t& a, const info_t& b) {
		return a->Param() == b->Param() && a->Level() == b->Level() && a->Time() == b->Time() &&
		       a->ForecastType() == b->ForecastType();
	};

	vector<info_t> skip;

	for (const auto& component : infos)
	{
		const HPGridType to = baseGrid->Type();
		const HPGridType from = component->Grid()->Type();
		const auto name = component->Param().Name();

		if (interpolate::IsVectorComponent(name) &&
		    count_if(skip.begin(), skip.end(), [&](const info_t& info) { return eq(info, component); }) == 0 &&
		    interpolate::IsSupportedGridForRotation(from) && component->Grid()->UVRelativeToGrid() && to != from)
		{
			auto otherName = GetOtherVectorComponentName(name);

			info_t u, v, other;

			for (const auto temp : infos)
			{
				if (temp->Param().Name() == otherName && temp->Level() == component->Level() &&
				    temp->Time() == component->Time() && temp->ForecastType() == component->ForecastType())
				{
					other = temp;
					break;
				}
			}

			if (!other)
			{
				// Other component not found
				continue;
			}

			if (name.find("U-MS") != std::string::npos)
			{
				u = component;
				v = other;
			}
			else if (name.find("V-MS") != std::string::npos)
			{
				u = other;
				v = component;
			}
			else
			{
				itsLogger.Fatal("Unrecognized vector component parameter: " + name);
				himan::Abort();
			}

#ifdef HAVE_CUDA
			util::Unpack<double>({u, v}, false);
#endif
			interpolate::RotateVectorComponents(component->Grid().get(), baseGrid, *u, *v,
			                                    opts.configuration->UseCuda());

			auto c = GET_PLUGIN(cache);
			c->Replace<double>(u);
			c->Replace<double>(v);

			// RotateVectorComponent modifies both components, so make sure we don't re-rotate the other
			// component.
			skip.push_back(other);
		}
	}

	// Step 2. Interpolate if needed

	if (itsDoInterpolation)
	{
		if (!interpolate::Interpolate(opts.configuration->BaseGrid(), infos))
		{
			itsLogger.Fatal("Interpolation failed");
			himan::Abort();
		}
	}
}

template <typename T>
bool fetcher::ApplyLandSeaMask(std::shared_ptr<const plugin_configuration> config, shared_ptr<info<T>> theInfo,
                               const forecast_time& requestedTime, const forecast_type& requestedType)
{
	raw_time originTime = requestedTime.OriginDateTime();
	forecast_time firstTime(originTime, originTime);

	itsLogger.Trace("Applying land-sea mask with threshold " + to_string(itsLandSeaMaskThreshold));

	try
	{
		itsApplyLandSeaMask = false;

		auto lsmInfo = Fetch<T>(config, firstTime, level(kHeight, 0), param("LC-0TO1"), requestedType, false);

		itsApplyLandSeaMask = true;

		lsmInfo->First();

		ASSERT(*lsmInfo->Grid() == *theInfo->Grid());

		ASSERT(itsLandSeaMaskThreshold >= -1 && itsLandSeaMaskThreshold <= 1);
		ASSERT(itsLandSeaMaskThreshold != 0);

#ifdef HAVE_CUDA
		if (theInfo->PackedData()->HasData())
		{
			// We need to unpack
			util::Unpack<T>({theInfo}, false);
		}
#endif

		ASSERT(theInfo->PackedData()->HasData() == false);

		double multiplier = (itsLandSeaMaskThreshold > 0) ? 1. : -1.;

		for (lsmInfo->ResetLocation(), theInfo->ResetLocation(); lsmInfo->NextLocation() && theInfo->NextLocation();)
		{
			T lsm = lsmInfo->Value();

			if (multiplier * lsm <= itsLandSeaMaskThreshold)
			{
				theInfo->Value(MissingValue<T>());
			}
		}
	}
	catch (HPExceptionType& e)
	{
		itsApplyLandSeaMask = true;
		return false;
	}

	return true;
}

template bool fetcher::ApplyLandSeaMask<double>(std::shared_ptr<const plugin_configuration>, shared_ptr<info<double>>,
                                                const forecast_time&, const forecast_type&);
template bool fetcher::ApplyLandSeaMask<float>(std::shared_ptr<const plugin_configuration>, shared_ptr<info<float>>,
                                               const forecast_time&, const forecast_type&);

bool fetcher::ApplyLandSeaMask() const
{
	return itsApplyLandSeaMask;
}
void fetcher::ApplyLandSeaMask(bool theApplyLandSeaMask)
{
	itsApplyLandSeaMask = theApplyLandSeaMask;
}
double fetcher::LandSeaMaskThreshold() const
{
	return itsLandSeaMaskThreshold;
}
void fetcher::LandSeaMaskThreshold(double theLandSeaMaskThreshold)
{
	if (theLandSeaMaskThreshold < -1 || theLandSeaMaskThreshold > 1)
	{
		itsLogger.Fatal("Invalid value for land sea mask threshold: " + to_string(theLandSeaMaskThreshold));
		himan::Abort();
	}

	itsLandSeaMaskThreshold = theLandSeaMaskThreshold;
}

bool fetcher::DoVectorComponentRotation() const
{
	return itsDoVectorComponentRotation;
}
void fetcher::DoVectorComponentRotation(bool theDoVectorComponentRotation)
{
	itsDoVectorComponentRotation = theDoVectorComponentRotation;
}

string GetOtherVectorComponentName(const string& name)
{
	// wind
	if (name == "U-MS")
		return "V-MS";
	else if (name == "V-MS")
		return "U-MS";
	// wind gust
	else if (name == "WGU-MS")
		return "WGV-MS";
	else if (name == "WGV-MS")
		return "WGU-MS";
	// ice
	else if (name == "IVELU-MS")
		return "IVELV-MS";
	else if (name == "IVELV-MS")
		return "IVELU-MS";
	// sea
	else if (name == "WVELU-MS")
		return "WVELV-MS";
	else if (name == "WVELV-MS")
		return "WVELU-MS";

	throw runtime_error("Unable to find component pair for " + name);
}

template <typename T>
void fetcher::RotateVectorComponents(vector<shared_ptr<info<T>>>& components, const grid* target,
                                     shared_ptr<const plugin_configuration> config, const producer& sourceProd)
{
	for (auto& component : components)
	{
		const HPGridType from = component->Grid()->Type();
		const HPGridType to = target->Type();
		const auto name = component->Param().Name();

		if (interpolate::IsVectorComponent(name) && itsDoVectorComponentRotation && to != from &&
		    interpolate::IsSupportedGridForRotation(from))
		{
			auto otherName = GetOtherVectorComponentName(name);

			search_options opts(component->Time(), param(otherName), component->Level(), sourceProd,
			                    component->ForecastType(), config);

			itsLogger.Trace("Fetching " + otherName + " for U/V rotation");

			auto ret = FetchFromAllSources<T>(opts, component->PackedData()->HasData());

			auto otherVec = ret.second;

			if (otherVec.empty())
			{
				// Other component not found
				continue;
			}

			shared_ptr<info<T>> u, v, other = otherVec[0];

			if (name.find("U-MS") != std::string::npos)
			{
				u = component;
				v = other;
			}
			else if (name.find("V-MS") != std::string::npos)
			{
				u = other;
				v = component;
			}
			else
			{
				throw runtime_error("Unrecognized vector component parameter: " + name);
			}

			interpolate::RotateVectorComponents(component->Grid().get(), target, *u, *v, config->UseCuda());

			// Most likely both U&V are requested, so interpolate the other one now
			// and put it to cache.

			std::vector<shared_ptr<info<T>>> list({other});
			if (itsDoInterpolation && interpolate::Interpolate(target, list))
			{
				if (itsUseCache && config->UseCacheForReads() && !other->PackedData()->HasData())
				{
					auto c = GET_PLUGIN(cache);
					c->Insert<T>(other);
				}
			}
		}
	}
}

template void fetcher::RotateVectorComponents<double>(vector<shared_ptr<info<double>>>&, const grid*,
                                                      shared_ptr<const plugin_configuration>, const producer&);
template void fetcher::RotateVectorComponents<float>(vector<shared_ptr<info<float>>>&, const grid*,
                                                     shared_ptr<const plugin_configuration>, const producer&);
