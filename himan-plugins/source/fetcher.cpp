#include "fetcher.h"
#include "cache.h"
#include "csv.h"
#include "geotiff.h"
#include "grib.h"
#include "interpolate.h"
#include "logger.h"
#include "numerical_functions.h"
#include "param.h"
#include "plugin_factory.h"
#include "querydata.h"
#include "radon.h"
#include "spiller.h"
#include "statistics.h"
#include "util.h"
#include <filesystem>
#include <fmt/ranges.h>
#include <fstream>
#include <future>
#include <shared_mutex>

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
	return fmt::format("{}_{}_{}", prod.Id(), par.Name(), fmt::underlying(lev.Type()));
}

void AmendParamWithAggregationAndProcessingType(param& p, const forecast_time& ftime)
{
	if (p.ProcessingType().Type() == kUnknownProcessingType)
	{
		p.ProcessingType(util::GetProcessingTypeFromParamName(p.Name()));
	}
	if (p.Aggregation().Type() == kUnknownAggregationType && p.ProcessingType().Type() != kMean)
	{
		p.Aggregation(util::GetAggregationFromParamName(p.Name(), ftime));
	}
}
}  // namespace
static vector<string> stickyParamCache;
static mutex stickyMutex;

// Container to store shared mutexes for data fetch for a single grid.
// The idea is that if multiple threads fetch the *same* data, they
// will be synchronized so that one thread fetches the data and updates
// cache while the other threads will wait for the completion of that task.

static mutex singleFetcherMutex;
map<string, std::shared_mutex> singleFetcherMap;

string CreateNotFoundString(const vector<producer>& prods, const forecast_type& ftype, const forecast_time& time,
                            const level& lev, const vector<param>& params)
{
	vector<long> prodIds;
	for_each(prods.begin(), prods.end(), [&prodIds](const producer& prod) { prodIds.push_back(prod.Id()); });
	vector<string> paramNames;
	for_each(params.begin(), params.end(), [&paramNames](const param& par) { paramNames.push_back(par.Name()); });

	return fmt::format(
	    "No valid data found for producer(s): {}, origintime: {}, step: {}, param(s): {}, level: {}, forecast_type: {}",
	    fmt::join(prodIds, ","), time.OriginDateTime().ToSQLTime(), static_cast<string>(time.Step()),
	    fmt::join(paramNames, ","), static_cast<string>(lev), static_cast<string>(ftype));
}

namespace
{
template <typename T, typename U>
shared_ptr<info<U>> ConvertTo(shared_ptr<info<T>> in)
{
	return make_shared<info<U>>(*in);
}
template <>
shared_ptr<info<double>> ConvertTo(shared_ptr<info<double>> in)
{
	return in;
}

/*
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
*/
}  // namespace

fetcher::fetcher()
    : itsDoLevelTransform(true),
      itsDoInterpolation(true),
      itsDoVectorComponentRotation(false),
      itsUseCache(true),
      itsApplyLandSeaMask(false),
      itsLandSeaMaskThreshold(0.5),
      itsDoTimeInterpolation(false),
      itsTimeInterpolationSearchStep(ONE_HOUR)
{
	itsLogger = logger("fetcher");
}

shared_ptr<info<double>> fetcher::Fetch(shared_ptr<const plugin_configuration> config, forecast_time requestedTime,
                                        level requestedLevel, const params& requestedParams,
                                        forecast_type requestedType, bool readPackedData,
                                        bool readFromPreviousForecastIfNotFound, bool doTimeInterpolation)
{
	return Fetch<double>(config, requestedTime, requestedLevel, requestedParams, requestedType, readPackedData,
	                     doTimeInterpolation);
}

template <typename T>
shared_ptr<info<T>> fetcher::Fetch(shared_ptr<const plugin_configuration> config, forecast_time requestedTime,
                                   level requestedLevel, const params& requestedParams, forecast_type requestedType,
                                   bool readPackedData, bool readFromPreviousForecastIfNotFound,
                                   bool doTimeInterpolation)
{
	if (doTimeInterpolation)
	{
		if (readFromPreviousForecastIfNotFound)
		{
			itsLogger.Warning("Not reading from previous forecast as time interpolation is enabled");
		}

		return InterpolateTime<T>(config, requestedTime, requestedLevel, requestedParams, requestedType, false);
	}

	shared_ptr<info<T>> ret;

	for (size_t i = 0; i < requestedParams.size(); i++)
	{
		try
		{
			return Fetch<T>(config, requestedTime, requestedLevel, requestedParams[i], requestedType, readPackedData,
			                true, false);
		}
		catch (const HPExceptionType& e)
		{
			if (e != kFileDataNotFound)
			{
				throw;
			}
		}
	}

	if (readFromPreviousForecastIfNotFound)
	{
		auto r = GET_PLUGIN(radon);
		const auto prev = r->RadonDB().GetLatestTime(static_cast<int>(config->SourceProducers()[0].Id()), "", 1);
		if (prev.empty() == false)
		{
			itsLogger.Info(fmt::format("Trying to read from previous forecast with analysis time {}", prev));

			raw_time prevatime(prev);
			time_duration diff = requestedTime.OriginDateTime() - prevatime;
			raw_time validtime = requestedTime.ValidDateTime() + diff;
			return Fetch<T>(config, forecast_time(prevatime, validtime), requestedLevel, requestedParams, requestedType,
			                readPackedData, false);
		}
	}

	itsLogger.Warning(
	    CreateNotFoundString(config->SourceProducers(), requestedType, requestedTime, requestedLevel, requestedParams));

	throw kFileDataNotFound;
}

template shared_ptr<info<double>> fetcher::Fetch<double>(shared_ptr<const plugin_configuration>, forecast_time, level,
                                                         const params&, forecast_type, bool, bool, bool);
template shared_ptr<info<float>> fetcher::Fetch<float>(shared_ptr<const plugin_configuration>, forecast_time, level,
                                                       const params&, forecast_type, bool, bool, bool);
template shared_ptr<info<short>> fetcher::Fetch<short>(shared_ptr<const plugin_configuration>, forecast_time, level,
                                                       const params&, forecast_type, bool, bool, bool);
template shared_ptr<info<unsigned char>> fetcher::Fetch<unsigned char>(shared_ptr<const plugin_configuration>,
                                                                       forecast_time, level, const params&,
                                                                       forecast_type, bool, bool, bool);

shared_ptr<info<double>> fetcher::Fetch(shared_ptr<const plugin_configuration> config, forecast_time requestedTime,
                                        level requestedLevel, param requestedParam, forecast_type requestedType,
                                        bool readPackedData, bool suppressLogging,
                                        bool readFromPreviousForecastIfNotFound, bool doTimeInterpolation)
{
	return Fetch<double>(config, requestedTime, requestedLevel, requestedParam, requestedType, readPackedData,
	                     suppressLogging, readFromPreviousForecastIfNotFound, doTimeInterpolation);
}

template <typename T>
shared_ptr<info<T>> fetcher::Fetch(shared_ptr<const plugin_configuration> config, forecast_time requestedTime,
                                   level requestedLevel, param requestedParam, forecast_type requestedType,
                                   bool readPackedData, bool suppressLogging, bool readFromPreviousForecastIfNotFound,
                                   bool doTimeInterpolation)
{
	timer t(true);

	AmendParamWithAggregationAndProcessingType(requestedParam, requestedTime);

	if (doTimeInterpolation)
	{
		if (readFromPreviousForecastIfNotFound)
		{
			itsLogger.Warning("Not reading from previous forecast as time interpolation is enabled");
		}

		return InterpolateTime<T>(config, requestedTime, requestedLevel, {requestedParam}, requestedType,
		                          suppressLogging);
	}
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
				if (config->UseCacheForReads() &&
				    find(stickyParamCache.begin(), stickyParamCache.end(), uName) == stickyParamCache.end())
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
		if (readFromPreviousForecastIfNotFound)
		{
			auto r = GET_PLUGIN(radon);
			const auto prev = r->RadonDB().GetLatestTime(static_cast<int>(config->SourceProducers()[0].Id()), "", 1);
			if (prev.empty() == false)
			{
				itsLogger.Info(
				    fmt::format("Data not found, trying to read from previous forecast with analysis time {}", prev));

				raw_time prevatime(prev);
				time_duration diff = requestedTime.OriginDateTime() - prevatime;
				raw_time validtime = requestedTime.ValidDateTime() + diff;
				return Fetch<T>(config, forecast_time(prevatime, validtime), requestedLevel, requestedParam,
				                requestedType, readPackedData, suppressLogging, false);
			}
			else
			{
				itsLogger.Warning("Previous forecast not found");
			}
		}

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
                                                         param, forecast_type, bool, bool, bool, bool);
template shared_ptr<info<float>> fetcher::Fetch<float>(shared_ptr<const plugin_configuration>, forecast_time, level,
                                                       param, forecast_type, bool, bool, bool, bool);
template shared_ptr<info<short>> fetcher::Fetch<short>(shared_ptr<const plugin_configuration>, forecast_time, level,
                                                       param, forecast_type, bool, bool, bool, bool);
template shared_ptr<info<unsigned char>> fetcher::Fetch<unsigned char>(shared_ptr<const plugin_configuration>,
                                                                       forecast_time, level, param, forecast_type, bool,
                                                                       bool, bool, bool);

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
			itsLogger.Trace(fmt::format("Transform level {} to {} for producer {}, param {}",
			                            static_cast<string>(opts.level), static_cast<string>(newLevel), opts.prod.Id(),
			                            opts.param.Name()));

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
	ASSERT(opts.configuration->ValidateMetadata() == false || theInfos[0]->Param() == opts.param);

	return theInfos[0];
}

template shared_ptr<info<double>> fetcher::FetchFromProducerSingle<double>(search_options&, bool, bool);
template shared_ptr<info<float>> fetcher::FetchFromProducerSingle<float>(search_options&, bool, bool);
template shared_ptr<info<short>> fetcher::FetchFromProducerSingle<short>(search_options&, bool, bool);
template shared_ptr<info<unsigned char>> fetcher::FetchFromProducerSingle<unsigned char>(search_options&, bool, bool);

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
	pair<map<string, std::shared_mutex>::iterator, bool> muret;

	// First acquire mutex to (possibly) modify map
	unique_lock<mutex> sflock(singleFetcherMutex);

	muret = singleFetcherMap.emplace(piecewise_construct, forward_as_tuple(uname), forward_as_tuple());

	if (muret.second == true)
	{
		// first time this data is being fetched: take exclusive lock to prevent
		// other threads from fetching the same data at the same time
		std::unique_lock<std::shared_mutex> uniqueLock(muret.first->second);
		sflock.unlock();

		return FetchFromProducerSingle<T>(opts, readPackedData, suppressLogging);
	}

	sflock.unlock();

	// this data is being fetched right now by other thread, or it has been fetched
	// earlier
	std::shared_lock<std::shared_mutex> lock(muret.first->second);

	return FetchFromProducerSingle<T>(opts, readPackedData, suppressLogging);
}

template shared_ptr<info<double>> fetcher::FetchFromProducer<double>(search_options&, bool, bool);
template shared_ptr<info<float>> fetcher::FetchFromProducer<float>(search_options&, bool, bool);
template shared_ptr<info<short>> fetcher::FetchFromProducer<short>(search_options&, bool, bool);
template shared_ptr<info<unsigned char>> fetcher::FetchFromProducer<unsigned char>(search_options&, bool, bool);

template <typename T>
vector<shared_ptr<info<T>>> fetcher::FromFile(const vector<file_information>& files, search_options& options,
                                              bool readPackedData, bool forceCaching)
{
	vector<shared_ptr<info<T>>> allInfos;

	for (const auto& inputFile : files)
	{
		if (inputFile.storage_type == HPFileStorageType::kLocalFileSystem &&
		    !filesystem::exists(inputFile.file_location))
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
				curInfos = g->FromFile<T>(inputFile, options, readPackedData, forceCaching);
				break;
			}

			case kQueryData:
				throw runtime_error("QueryData as input is not supported");

			case kNetCDF:
			case kNetCDFv4:
				throw runtime_error("NetCDF as input is not supported");

			case kCSV:
			{
				auto c = GET_PLUGIN(csv);
				auto anInfo = c->FromFile<T>(inputFile.file_location, options, forceCaching);
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
time_duration fetcher::TimeInterpolationSearchStep() const
{
	return itsTimeInterpolationSearchStep;
}
void fetcher::TimeInterpolationSearchStep(const time_duration& step)
{
	itsTimeInterpolationSearchStep = step;
}

template <typename T>
pair<HPDataFoundFrom, vector<shared_ptr<info<T>>>> fetcher::FetchFromAllSources(search_options& opts,
                                                                                bool readPackedData)
{
	auto fromCache = FetchFromCache<T>(opts);

	if (!fromCache.empty())
	{
		return make_pair(HPDataFoundFrom::kCache, fromCache);
	}

#ifdef HAVE_CEREAL
	if (spiller::Enabled())
	{
		auto fromSpill = spiller::ReadFromUniqueName<T>(util::UniqueName(opts));
		if (fromSpill)
		{
			itsLogger.Warning(
			    fmt::format("Spill files accessed: increase cache_limit to speed up processing (currently {} bytes)",
			                opts.configuration->CacheLimit()));

			return make_pair(HPDataFoundFrom::kSpillFile, vector<shared_ptr<info<T>>>({fromSpill}));
		}
	}
#endif
	if (!auxiliaryFilesRead)
	{
		auto fromAux = FetchFromAuxiliaryFiles(opts, readPackedData);

		if (!fromAux.second.empty())
		{
			return make_pair(fromAux.first, vector<shared_ptr<info<T>>>({ConvertTo<double, T>(fromAux.second[0])}));
		}
	}

	return make_pair(HPDataFoundFrom::kDatabase, FetchFromDatabase<T>(opts, readPackedData));
}

template pair<HPDataFoundFrom, vector<shared_ptr<info<double>>>> fetcher::FetchFromAllSources<double>(search_options&,
                                                                                                      bool);
template pair<HPDataFoundFrom, vector<shared_ptr<info<float>>>> fetcher::FetchFromAllSources<float>(search_options&,
                                                                                                    bool);
template pair<HPDataFoundFrom, vector<shared_ptr<info<short>>>> fetcher::FetchFromAllSources<short>(search_options&,
                                                                                                    bool);
template pair<HPDataFoundFrom, vector<shared_ptr<info<unsigned char>>>> fetcher::FetchFromAllSources<unsigned char>(
    search_options&, bool);

template <typename T>
vector<shared_ptr<info<T>>> fetcher::FetchFromCache(search_options& opts)
{
	vector<shared_ptr<info<T>>> ret;

	if (itsUseCache && opts.configuration->UseCacheForReads())
	{
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
template vector<shared_ptr<info<short>>> fetcher::FetchFromCache<short>(search_options&);
template vector<shared_ptr<info<unsigned char>>> fetcher::FetchFromCache<unsigned char>(search_options&);

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
			const string analtime = opts.time.OriginDateTime().String("%Y%m%d%H%M");
			const vector<string> sourceGeoms = opts.configuration->SourceGeomNames();
			itsLogger.Trace(
			    fmt::format("No geometries found for producer {}, analysistime {}, source geom name(s) {}, param {}",
			                ref_prod, analtime, fmt::join(sourceGeoms, ","), opts.param.Name()));
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

		auto c = GET_PLUGIN(csv);

		auto _ret = c->FromMemory<T>(csv_forecasts);

		if (_ret)
		{
			ret.push_back(_ret);
		}
	}

	return ret;
}

template vector<shared_ptr<info<double>>> fetcher::FetchFromDatabase<double>(search_options&, bool);
template vector<shared_ptr<info<float>>> fetcher::FetchFromDatabase<float>(search_options&, bool);
template vector<shared_ptr<info<short>>> fetcher::FetchFromDatabase<short>(search_options&, bool);
template vector<shared_ptr<info<unsigned char>>> fetcher::FetchFromDatabase<unsigned char>(search_options&, bool);

pair<HPDataFoundFrom, vector<shared_ptr<info<double>>>> fetcher::FetchFromAuxiliaryFiles(search_options& opts,
                                                                                         bool readPackedData)
{
	vector<shared_ptr<info<double>>> ret;
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
			f.offset = std::nullopt;
			f.length = std::nullopt;
			f.message_no = std::nullopt;
			f.storage_type = (file.find("s3://") != string::npos) ? HPFileStorageType::kS3ObjectStorageSystem
			                                                      : HPFileStorageType::kLocalFileSystem;

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

			call_once(
			    oflag,
			    [&]()
			    {
				    vector<string> filenames;
				    std::for_each(files.begin(), files.end(),
				                  [&filenames](const auto& a) { filenames.push_back(a.file_location); });
				    itsLogger.Debug(fmt::format("Start full auxiliary files read for: {}", fmt::join(filenames, ", ")));

				    timer t(true);

				    ret = FromFile<double>(files, opts, readPackedData, true);

				    AuxiliaryFilesRotateAndInterpolate(opts, ret);

#ifdef HAVE_CUDA
				    if (readPackedData && opts.configuration->UseCudaForUnpacking())
				    {
					    util::Unpack<double>(ret, false);
				    }
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
				    itsLogger.Debug(
				        fmt::format("Auxiliary files read finished in {} ms, cache size: {}", t.GetTime(), c->Size()));
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

void fetcher::AuxiliaryFilesRotateAndInterpolate(const search_options& opts, vector<shared_ptr<info<double>>>& infos)
{
	// Step 1. Rotate if needed

	const grid* baseGrid = opts.configuration->BaseGrid();

	auto eq = [](const shared_ptr<info<double>>& a, const shared_ptr<info<double>>& b)
	{
		return a->Param() == b->Param() && a->Level() == b->Level() && a->Time() == b->Time() &&
		       a->ForecastType() == b->ForecastType();
	};

	vector<shared_ptr<info<double>>> skip;

	for (const auto& component : infos)
	{
		const HPGridType to = baseGrid->Type();
		const HPGridType from = component->Grid()->Type();
		const auto name = component->Param().Name();

		if (interpolate::IsVectorComponent(name) &&
		    count_if(skip.begin(), skip.end(),
		             [&](const shared_ptr<info<double>>& info) { return eq(info, component); }) == 0 &&
		    interpolate::IsSupportedGridForRotation(from) && component->Grid()->UVRelativeToGrid() && to != from)
		{
			auto otherName = GetOtherVectorComponentName(name);

			shared_ptr<info<double>> u, v, other;

			for (const auto& temp : infos)
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
			if (opts.configuration->UseCudaForPacking())
			{
				util::Unpack<double>({u, v}, false);
			}
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
	else
	{
		itsLogger.Trace("Interpolation disabled");
	}
}

template <typename T>
bool fetcher::ApplyLandSeaMask(std::shared_ptr<const plugin_configuration> config, shared_ptr<info<T>> theInfo,
                               const forecast_time& requestedTime, const forecast_type& requestedType)
{
	raw_time originTime = requestedTime.OriginDateTime();
	forecast_time firstTime(originTime, originTime);

	itsLogger.Trace(fmt::format("Applying land-sea mask with threshold {}", itsLandSeaMaskThreshold));

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
template bool fetcher::ApplyLandSeaMask<short>(std::shared_ptr<const plugin_configuration>, shared_ptr<info<short>>,
                                               const forecast_time&, const forecast_type&);
template bool fetcher::ApplyLandSeaMask<unsigned char>(std::shared_ptr<const plugin_configuration>,
                                                       shared_ptr<info<unsigned char>>, const forecast_time&,
                                                       const forecast_type&);

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
		itsLogger.Fatal(fmt::format("Invalid value for land sea mask threshold: {}", theLandSeaMaskThreshold));
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
			if (itsDoInterpolation)
			{
				if (interpolate::Interpolate(target, list))
				{
					if (itsUseCache && config->UseCacheForReads() && !other->PackedData()->HasData())
					{
						auto c = GET_PLUGIN(cache);
						c->Insert<T>(other);
					}
				}
			}
			else
			{
				itsLogger.Trace("Interpolation disabled");
			}
		}
	}
}

template void fetcher::RotateVectorComponents<double>(vector<shared_ptr<info<double>>>&, const grid*,
                                                      shared_ptr<const plugin_configuration>, const producer&);
template void fetcher::RotateVectorComponents<float>(vector<shared_ptr<info<float>>>&, const grid*,
                                                     shared_ptr<const plugin_configuration>, const producer&);
template void fetcher::RotateVectorComponents<short>(vector<shared_ptr<info<short>>>&, const grid*,
                                                     shared_ptr<const plugin_configuration>, const producer&);
template void fetcher::RotateVectorComponents<unsigned char>(vector<shared_ptr<info<unsigned char>>>&, const grid*,
                                                             shared_ptr<const plugin_configuration>, const producer&);

template <typename T>
shared_ptr<himan::info<T>> fetcher::InterpolateTime(const shared_ptr<const plugin_configuration>& config,
                                                    const forecast_time& ftime, const level& lev, const params& pars,
                                                    const forecast_type& ftype, bool suppressLogging)
{
	itsLogger.Trace("Starting time interpolation");

	for (const auto& par : pars)
	{
		// fetch previous data, max 6 hours to past

		forecast_time curtime = ftime;

		shared_ptr<info<T>> prev = nullptr, next = nullptr;

		int count = 0, max = 6;
		do
		{
			try
			{
				prev = Fetch<T>(config, curtime, lev, par, ftype, false, suppressLogging, false, false);
				break;
			}
			catch (const HPExceptionType& e)
			{
				if (e != kFileDataNotFound)
				{
					count = max;
				}
			}
			curtime.ValidDateTime(curtime.ValidDateTime() - itsTimeInterpolationSearchStep);

		} while (++count < max && curtime.OriginDateTime() <= curtime.ValidDateTime());

		if (curtime == ftime)
		{
			itsLogger.Info("No time interpolation needed");
			return prev;
		}

		// fetch next data, max 6 hours to future

		curtime = ftime;
		count = 0;

		do
		{
			curtime.ValidDateTime(curtime.ValidDateTime() + itsTimeInterpolationSearchStep);

			try
			{
				next = Fetch<T>(config, curtime, lev, par, ftype, false, suppressLogging, false, false);
				break;
			}
			catch (const HPExceptionType& e)
			{
				if (e != kFileDataNotFound)
				{
					count = max;
				}
			}

		} while (++count < max);

		if (!prev || !next)
		{
			itsLogger.Error(
			    fmt::format("Time interpolation failed for {}: unable to find previous or next data", par.Name()));
			continue;
		}

		auto interpolated = make_shared<info<T>>(ftype, ftime, lev, par);
		interpolated->Producer(prev->Producer());
		interpolated->Create(prev->Base(), true);

		const auto& prevdata = VEC(prev);
		const auto& nextdata = VEC(next);
		auto& interp = VEC(interpolated);

		const T X = static_cast<T>(ftime.Step().Hours());
		const T X1 = static_cast<T>(prev->Time().Step().Hours());
		const T X2 = static_cast<T>(next->Time().Step().Hours());

		for (size_t i = 0; i < prevdata.size(); i++)
		{
			interp[i] = numerical_functions::interpolation::Linear<T>(X, X1, X2, prevdata[i], nextdata[i]);
		}

		return interpolated;
	}

	return nullptr;
}

template shared_ptr<himan::info<double>> fetcher::InterpolateTime<double>(const shared_ptr<const plugin_configuration>&,
                                                                          const forecast_time&, const level&,
                                                                          const params&, const forecast_type&, bool);
template shared_ptr<himan::info<float>> fetcher::InterpolateTime<float>(const shared_ptr<const plugin_configuration>&,
                                                                        const forecast_time&, const level&,
                                                                        const params&, const forecast_type&, bool);
