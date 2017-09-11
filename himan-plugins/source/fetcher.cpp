/**
 * @file fetcher.cpp
 *
 */

#include "fetcher.h"
#include "interpolate.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <boost/filesystem/operations.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <future>

#include "cache.h"
#include "csv.h"
#include "grib.h"
#include "neons.h"
#include "param.h"
#include "querydata.h"
#include "radon.h"

using namespace himan::plugin;
using namespace std;

static once_flag oflag;

// If multiple plugins are executed using auxiliary files, prevent
// each plugin execution from re-reading the aux files to memory.

static std::atomic<bool> auxiliaryFilesRead(false);

// Sticky param cache will store the producer that provided data.
// With this information we can skip the regular producer loop cycle
// (try prod 1, data not found, try prod 2) which will improve fetching
// times when reading from database.

std::string UniqueName(const himan::producer& prod, const himan::param& par, const himan::level& lev)
{
	return to_string(prod.Id()) + "_" + par.Name() + "_" + himan::HPLevelTypeToString.at(lev.Type());
}

static vector<string> stickyParamCache;
static mutex stickyMutex;

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

shared_ptr<himan::info> fetcher::Fetch(shared_ptr<const plugin_configuration> config, forecast_time requestedTime,
                                       level requestedLevel, const params& requestedParams, forecast_type requestedType,
                                       bool readPackedData)
{
	shared_ptr<info> ret;

	for (size_t i = 0; i < requestedParams.size(); i++)
	{
		try
		{
			return Fetch(config, requestedTime, requestedLevel, requestedParams[i], requestedType, readPackedData,
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

	string optsStr = "producer(s): ";

	for (size_t prodNum = 0; prodNum < config->SizeSourceProducers(); prodNum++)
	{
		optsStr += boost::lexical_cast<string>(config->SourceProducer(prodNum).Id()) + ",";
	}

	optsStr = optsStr.substr(0, optsStr.size() - 1);

	optsStr += " origintime: " + requestedTime.OriginDateTime().String() + ", step: " +
	           boost::lexical_cast<string>(requestedTime.Step());

	optsStr += " param(s): ";

	for (size_t i = 0; i < requestedParams.size(); i++)
	{
		optsStr += requestedParams[i].Name() + ",";
	}

	optsStr = optsStr.substr(0, optsStr.size() - 1) + " level: " + static_cast<std::string>(requestedLevel);

	if (static_cast<int>(requestedType.Type()) > 2)
	{
		optsStr += " forecast type: " + string(himan::HPForecastTypeToString.at(requestedType.Type())) + "/" +
		           boost::lexical_cast<string>(requestedType.Value());
	}

	itsLogger.Warning("No valid data found with given search options " + optsStr);

	throw kFileDataNotFound;
}

shared_ptr<himan::info> fetcher::FetchFromProducer(search_options& opts, bool readPackedData, bool suppressLogging)
{
	level newLevel = opts.level;

	if (itsDoLevelTransform && opts.configuration->DatabaseType() != kNoDatabase &&
	    (opts.level.Type() != kHybrid && opts.level.Type() != kPressure && opts.level.Type() != kHeightLayer))
	{
		newLevel = LevelTransform(opts.configuration, opts.prod, opts.param, opts.level);

		if (newLevel != opts.level)
		{
			itsLogger.Trace("Transform level " + static_cast<string>(opts.level) + " to " +
			                static_cast<string>(newLevel) + " for producer " + to_string(opts.prod.Id()) +
			                ", parameter " + opts.param.Name());

			opts.level = newLevel;
		}
	}

	auto ret = FetchFromAllSources(opts, readPackedData);

	auto theInfos = ret.second;

	if (theInfos.empty())
	{
		return shared_ptr<info>();
	}

	auto baseInfo = make_shared<info>(*opts.configuration->Info());
	assert(baseInfo->Dimensions().size());

	baseInfo->First();

	RotateVectorComponents(theInfos, baseInfo, opts.configuration, opts.prod);

	if (itsDoInterpolation)
	{
		if (!interpolate::Interpolate(*baseInfo, theInfos, opts.configuration->UseCudaForInterpolation()))

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
		itsLogger.Trace("Applying land-sea mask with threshold " +
		                boost::lexical_cast<string>(itsLandSeaMaskThreshold));

		itsApplyLandSeaMask = false;

		if (!ApplyLandSeaMask(opts.configuration, *theInfos[0], opts.time, opts.ftype))
		{
			itsLogger.Warning("Land sea mask apply failed");
		}

		itsApplyLandSeaMask = true;
	}

	/*
	 * Insert interpolated data to cache if
	 * 1. Cache is not disabled locally (itsUseCache) AND
	 * 2. Cache is not disabled globally (config->UseCache()) AND
	 * 3. Data is not packed
	 */

	if (ret.first != HPDataFoundFrom::kCache && itsUseCache && opts.configuration->UseCache() &&
	    !theInfos[0]->Grid()->IsPackedData())
	{
		auto c = GET_PLUGIN(cache);
		c->Insert(*theInfos[0]);
	}

	baseInfo.reset();

	assert((theInfos[0]->Level()) == opts.level);

	assert((theInfos[0]->Time()) == opts.time);

	assert((theInfos[0]->Param()) == opts.param);

	return theInfos[0];
}

shared_ptr<himan::info> fetcher::Fetch(shared_ptr<const plugin_configuration> config, forecast_time requestedTime,
                                       level requestedLevel, param requestedParam, forecast_type requestedType,
                                       bool readPackedData, bool suppressLogging)
{
	timer t;

	if (config->StatisticsEnabled())
	{
		t.Start();
	}

	// Check sticky param cache first

	shared_ptr<info> ret;

	for (size_t prodNum = 0; prodNum < config->SizeSourceProducers(); prodNum++)
	{
		bool found = false;

		{
			lock_guard<mutex> lock(stickyMutex);

			// Linear search, size of stickyParamCache should be relatively small
			if (find(stickyParamCache.begin(), stickyParamCache.end(),
			         UniqueName(config->SourceProducer(prodNum), requestedParam, requestedLevel)) !=
			    stickyParamCache.end())
			{
				// oh,goody
				found = true;
			}
		}

		if (found)
		{
			search_options opts(requestedTime, requestedParam, requestedLevel, config->SourceProducer(prodNum),
			                    requestedType, config);

			ret = FetchFromProducer(opts, readPackedData, suppressLogging);
			if (ret) break;

			itsLogger.Warning("Sticky cache failed, trying all producers just to be sure");
		}
	}

	for (size_t prodNum = 0; (ret == nullptr) && prodNum < config->SizeSourceProducers(); prodNum++)
	{
		search_options opts(requestedTime, requestedParam, requestedLevel, config->SourceProducer(prodNum),
		                    requestedType, config);

		ret = FetchFromProducer(opts, readPackedData, suppressLogging);
	}

	if (config->StatisticsEnabled())
	{
		t.Stop();

		config->Statistics()->AddToFetchingTime(t.GetTime());
	}

	/*
	 *  Safeguard; later in the code we do not check whether the data requested
	 *  was actually what was requested.
	 */

	if (!ret)
	{
		if (!suppressLogging)
		{
			string optsStr = "producer(s): ";

			for (size_t prodNum = 0; prodNum < config->SizeSourceProducers(); prodNum++)
			{
				optsStr += to_string(config->SourceProducer(prodNum).Id()) + ",";
			}

			optsStr = optsStr.substr(0, optsStr.size() - 1);

			optsStr += " origintime: " + requestedTime.OriginDateTime().String() +
			           ", step: " + to_string(requestedTime.Step());
			optsStr += " param: " + requestedParam.Name();
			optsStr += " level: " + static_cast<string>(requestedLevel);

			if (static_cast<int>(requestedType.Type()) > 2)
			{
				optsStr += " forecast type: " + string(himan::HPForecastTypeToString.at(requestedType.Type())) + "/" +
				           to_string(static_cast<int>(requestedType.Value()));
			}

			itsLogger.Warning("No valid data found with given search options " + optsStr);
		}

		throw kFileDataNotFound;
	}

	// assert(theConfiguration->SourceProducer() == theInfos[0]->Producer());

	return ret;
}

vector<shared_ptr<himan::info>> fetcher::FromFile(const vector<string>& files, search_options& options,
                                                  bool readContents, bool readPackedData, bool readIfNotMatching)
{
	vector<shared_ptr<himan::info>> allInfos;

	set<string> fileset(files.begin(), files.end());

	for (const string& inputFile : fileset)
	{
		if (!boost::filesystem::exists(inputFile))
		{
			itsLogger.Error("Input file '" + inputFile + "' does not exist");
			continue;
		}

		vector<shared_ptr<himan::info>> curInfos;

		switch (util::FileType(inputFile))
		{
			case kGRIB:
			case kGRIB1:
			case kGRIB2:
			{
				curInfos = FromGrib(inputFile, options, readContents, readPackedData, readIfNotMatching);
				break;
			}
			case kGRIBIndex:
			{
				curInfos = FromGribIndex(inputFile, options, readContents, readPackedData, readIfNotMatching);
				break;
			}

			case kQueryData:
			{
				curInfos = FromQueryData(inputFile, options, readContents);
				break;
			}

			case kNetCDF:
				itsLogger.Error("File is NetCDF");
				break;

			case kCSV:
			{
				curInfos = FromCSV(inputFile, options, readIfNotMatching);
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

vector<shared_ptr<himan::info>> fetcher::FromCache(search_options& options)
{
	auto c = GET_PLUGIN(cache);

	vector<shared_ptr<himan::info>> infos = c->GetInfo(options);

	return infos;
}

vector<shared_ptr<himan::info>> fetcher::FromGrib(const string& inputFile, search_options& options, bool readContents,
                                                  bool readPackedData, bool readIfNotMatching)
{
	auto g = GET_PLUGIN(grib);

	return g->FromFile(inputFile, options, readContents, readPackedData, readIfNotMatching);
}

vector<shared_ptr<himan::info>> fetcher::FromGribIndex(const string& inputFile, search_options& options,
                                                       bool readContents, bool readPackedData, bool readIfNotMatching)
{
	auto g = GET_PLUGIN(grib);

	return g->FromIndexFile(inputFile, options, readContents, readPackedData, readIfNotMatching);
}

vector<shared_ptr<himan::info>> fetcher::FromQueryData(const string& inputFile, search_options& options,
                                                       bool readContents)
{
	auto q = GET_PLUGIN(querydata);

	shared_ptr<info> i = q->FromFile(inputFile, options);

	vector<shared_ptr<info>> theInfos;

	theInfos.push_back(i);

	return theInfos;
}

vector<shared_ptr<himan::info>> fetcher::FromCSV(const string& inputFile, search_options& options,
                                                 bool readIfNotMatching)
{
	auto c = GET_PLUGIN(csv);

	auto info = c->FromFile(inputFile, options, readIfNotMatching);

	vector<info_t> infos;
	infos.push_back(info);

	return infos;
}

himan::level fetcher::LevelTransform(const shared_ptr<const configuration>& conf, const producer& sourceProducer,
                                     const param& targetParam, const level& targetLevel) const
{
	level ret = targetLevel;

	HPDatabaseType dbtype = conf->DatabaseType();

	if (dbtype == kNeons || dbtype == kNeonsAndRadon)
	{
		auto n = GET_PLUGIN(neons);

		string lvlName =
		    n->NeonsDB().GetGridLevelName(targetParam.Name(), targetLevel.Type(), 204, sourceProducer.TableVersion());

		if (!lvlName.empty())
		{
			double lvlValue = targetLevel.Value();

			HPLevelType lvlType = HPStringToLevelType.at(boost::to_lower_copy(lvlName));

			if (lvlType == kGround)
			{
				lvlValue = 0;
			}

			ret = level(lvlType, lvlValue, lvlName);
		}
	}

	if (ret == targetLevel && (dbtype == kRadon || dbtype == kNeonsAndRadon))
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

		auto levelXrefInfo =
		    r->RadonDB().GetLevelTransform(sourceProducer.Id(), boost::lexical_cast<int>(paramInfo["id"]),
		                                   boost::lexical_cast<int>(levelInfo["id"]), targetLevel.Value());

		if (!levelXrefInfo.empty())
		{
			double lvlValue = targetLevel.Value();

			HPLevelType lvlType = HPStringToLevelType.at(boost::to_lower_copy(levelXrefInfo["name"]));

			if (lvlType == kGround)
			{
				lvlValue = 0;
			}

			ret = level(lvlType, lvlValue);
		}
	}

	if (ret == targetLevel)
	{
		itsLogger.Trace("No level transformation found for param " + targetParam.Name() + " level " +
						static_cast<string>(targetLevel));
	}

	return ret;
}

void fetcher::DoLevelTransform(bool theDoLevelTransform) { itsDoLevelTransform = theDoLevelTransform; }
bool fetcher::DoLevelTransform() const { return itsDoLevelTransform; }
void fetcher::DoInterpolation(bool theDoInterpolation) { itsDoInterpolation = theDoInterpolation; }
bool fetcher::DoInterpolation() const { return itsDoInterpolation; }
void fetcher::UseCache(bool theUseCache) { itsUseCache = theUseCache; }
bool fetcher::UseCache() const { return itsUseCache; }
void fetcher::AuxiliaryFilesRotateAndInterpolate(const search_options& opts, vector<info_t>& infos)
{
	vector<future<void>> futures;

	const unsigned int maxFutureSize = 8;  // arbitrary number of parallel interpolation threads

	for (const auto& anInfo : infos)
	{
		vector<info_t> vec(1);
		vec[0] = anInfo;

		futures.push_back(async(
		    launch::async,
		    [&](vector<info_t> vec) {
			    auto baseInfo =
			        make_shared<info>(*dynamic_cast<const plugin_configuration*>(opts.configuration.get())->Info());
			    assert(baseInfo->Dimensions().size());

			    baseInfo->First();

			    if (itsDoInterpolation)
			    {
				    if (!interpolate::Interpolate(*baseInfo, vec, opts.configuration->UseCudaForInterpolation()))
				    {
					    throw runtime_error("Interpolation failed");
				    }
			    }

			    baseInfo.reset();

			},
		    vec));

		if (futures.size() == maxFutureSize)
		{
			for (auto& fut : futures)
			{
				fut.get();
			}

			futures.clear();

			itsLogger.Trace("Processed " + to_string(maxFutureSize) + " infos");
		}
	}

	// remainder of the loop
	for (auto& fut : futures)
	{
		fut.get();
	}

	futures.clear();
}

vector<shared_ptr<himan::info>> fetcher::FetchFromCache(search_options& opts)
{
	vector<shared_ptr<info>> ret;

	if (itsUseCache && opts.configuration->UseCache())
	{
		// 1. Fetch data from cache
		ret = FromCache(opts);

		if (ret.size())
		{
			itsLogger.Trace("Data found from cache");

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

pair<HPDataFoundFrom, vector<shared_ptr<himan::info>>> fetcher::FetchFromAuxiliaryFiles(search_options& opts,
                                                                                        bool readPackedData)
{
	vector<info_t> ret;
	HPDataFoundFrom source = HPDataFoundFrom::kAuxFile;

	if (!opts.configuration->AuxiliaryFiles().empty())
	{
		auto files = opts.configuration->AuxiliaryFiles();

		if (itsUseCache && opts.configuration->UseCache() && opts.configuration->ReadAllAuxiliaryFilesToCache())
		{
			if (itsApplyLandSeaMask)
			{
				itsLogger.Fatal("Land sea mask cannot be applied when reading all auxiliary files to cache");
				itsLogger.Fatal("Restart himan with command line option --no-auxiliary-file-full-cache-read");
				abort();
			}

			call_once(oflag, [&]() {

				itsLogger.Debug("Start full auxiliary files read");

				ret = FromFile(files, opts, true, readPackedData, true);

				AuxiliaryFilesRotateAndInterpolate(opts, ret);

				/*
				 * Insert interpolated data to cache if
				 * 1. Cache is not disabled locally (itsUseCache) AND
				 * 2. Cache is not disabled globally (config->UseCache()) AND
				 * 3. Data is not packed
				 */

				auto c = GET_PLUGIN(cache);

				for (const auto& anInfo : ret)
				{
#ifdef HAVE_CUDA
					if (anInfo->Grid()->IsPackedData())
					{
						util::Unpack({anInfo->Grid()});
					}
#endif
					// Insert each grid of an info to cache. Usually one info
					// has only one grid but in some cases this is not true.
					for (anInfo->First(), anInfo->ResetParam(); anInfo->Next();)
					{
						c->Insert(*anInfo);
					}
				}

				itsLogger.Debug("Auxiliary files read finished, cache size is now " + to_string(c->Size()));
			});

			auxiliaryFilesRead = true;
			source = HPDataFoundFrom::kCache;

			ret = FromCache(opts);
		}
		else
		{
			ret = FromFile(files, opts, true, readPackedData, false);
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

vector<shared_ptr<himan::info>> fetcher::FetchFromDatabase(search_options& opts, bool readPackedData)
{
	vector<info_t> ret;

	HPDatabaseType dbtype = opts.configuration->DatabaseType();

	if (!opts.configuration->ReadDataFromDatabase() || dbtype == kNoDatabase)
	{
		return ret;
	}

	if (opts.prod.Class() == kGridClass)
	{
		vector<string> files;

		if (dbtype == kNeons || dbtype == kNeonsAndRadon)
		{
			// try neons first
			auto n = GET_PLUGIN(neons);

			itsLogger.Trace("Accessing Neons database");

			files = n->Files(opts);
		}

		if ((dbtype == kRadon || dbtype == kNeonsAndRadon) && files.empty())
		{
			// try radon next

			auto r = GET_PLUGIN(radon);

			itsLogger.Trace("Accessing Radon database");

			files = r->Files(opts);
		}

		if (files.empty())
		{
			const string ref_prod = opts.prod.Name();
			const string analtime = opts.time.OriginDateTime().String("%Y%m%d%H%M%S");
			const vector<string> sourceGeoms = opts.configuration->SourceGeomNames();
			itsLogger.Trace("No geometries found for producer " + ref_prod + ", analysistime " + analtime +
			                ", source geom name(s) '" + util::Join(sourceGeoms, ",") + "', param " + opts.param.Name());
		}
		else
		{
			ret = FromFile(files, opts, true, readPackedData);

			if (dynamic_pointer_cast<const plugin_configuration>(opts.configuration)->StatisticsEnabled())
			{
				dynamic_pointer_cast<const plugin_configuration>(opts.configuration)
				    ->Statistics()
				    ->AddToCacheMissCount(1);
			}

			lock_guard<mutex> lock(stickyMutex);
			auto uName = UniqueName(opts.prod, opts.param, opts.level);
			if (find(stickyParamCache.begin(), stickyParamCache.end(), uName) == stickyParamCache.end())
			{
				itsLogger.Trace("Updating sticky param cache: " + UniqueName(opts.prod, opts.param, opts.level));
				stickyParamCache.push_back(uName);
			}
			return ret;
		}
	}
	else if (opts.prod.Class() == kPreviClass)
	{
		auto r = GET_PLUGIN(radon);

		itsLogger.Trace("Accessing Radon database for previ data");

		auto csv_forecasts = r->CSV(opts);
		auto _ret = util::CSVToInfo(csv_forecasts);

		if (_ret)
		{
			ret.push_back(_ret);
		}
	}

	return ret;
}

pair<HPDataFoundFrom, vector<shared_ptr<himan::info>>> fetcher::FetchFromAllSources(search_options& opts,
                                                                                    bool readPackedData)
{
	auto ret = FetchFromCache(opts);

	if (!ret.empty())
	{
		return make_pair(HPDataFoundFrom::kCache, ret);
	}

	if (!auxiliaryFilesRead)
	{
		// second ret, different from first
		auto ret = FetchFromAuxiliaryFiles(opts, readPackedData);

		if (!ret.second.empty())
		{
			return ret;
		}
	}

	return make_pair(HPDataFoundFrom::kDatabase, FetchFromDatabase(opts, readPackedData));
}

bool fetcher::ApplyLandSeaMask(std::shared_ptr<const plugin_configuration> config, info& theInfo,
                               const forecast_time& requestedTime, const forecast_type& requestedType)
{
	raw_time originTime = requestedTime.OriginDateTime();
	forecast_time firstTime(originTime, originTime);

	try
	{
		itsApplyLandSeaMask = false;

		auto lsmInfo = Fetch(config, firstTime, level(kHeight, 0), param("LC-0TO1"), requestedType, false);

		itsApplyLandSeaMask = true;

		lsmInfo->First();

		assert(*lsmInfo->Grid() == *theInfo.Grid());

		assert(itsLandSeaMaskThreshold >= -1 && itsLandSeaMaskThreshold <= 1);
		assert(itsLandSeaMaskThreshold != 0);

#ifdef HAVE_CUDA
		if (theInfo.Grid()->IsPackedData())
		{
			// We need to unpack
			util::Unpack({theInfo.Grid()});
		}
#endif

		assert(!theInfo.Grid()->IsPackedData());

		double multiplier = (itsLandSeaMaskThreshold > 0) ? 1. : -1.;

		for (lsmInfo->ResetLocation(), theInfo.ResetLocation(); lsmInfo->NextLocation() && theInfo.NextLocation();)
		{
			double lsm = lsmInfo->Value();

			if (multiplier * lsm <= itsLandSeaMaskThreshold)
			{
				theInfo.Value(MissingDouble());
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

bool fetcher::ApplyLandSeaMask() const { return itsApplyLandSeaMask; }
void fetcher::ApplyLandSeaMask(bool theApplyLandSeaMask) { itsApplyLandSeaMask = theApplyLandSeaMask; }
double fetcher::LandSeaMaskThreshold() const { return itsLandSeaMaskThreshold; }
void fetcher::LandSeaMaskThreshold(double theLandSeaMaskThreshold)
{
	if (theLandSeaMaskThreshold < -1 || theLandSeaMaskThreshold > 1)
	{
		itsLogger.Fatal("Invalid value for land sea mask threshold: " +
						boost::lexical_cast<string>(theLandSeaMaskThreshold));
		abort();
	}

	itsLandSeaMaskThreshold = theLandSeaMaskThreshold;
}

bool fetcher::DoVectorComponentRotation() const { return itsDoVectorComponentRotation; }
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

void fetcher::RotateVectorComponents(vector<info_t>& components, info_t target,
                                     shared_ptr<const plugin_configuration> config, const producer& sourceProd)
{
	for (auto& component : components)
	{
		HPGridType from = component->Grid()->Type();
		HPGridType to = target->Grid()->Type();
		const auto name = component->Param().Name();

		if (interpolate::IsVectorComponent(name) &&
		    ((itsDoVectorComponentRotation) ||
		     ((from == kRotatedLatitudeLongitude || from == kStereographic || from == kLambertConformalConic) &&
		      to != from)))
		{
			if (!itsDoVectorComponentRotation &&
			    (to == kRotatedLatitudeLongitude || to == kStereographic || to == kLambertConformalConic))
			{
				throw runtime_error("Rotating vector components to projected area is not supported (yet)");
			}

			auto otherName = GetOtherVectorComponentName(name);

			search_options opts(component->Time(), param(otherName), component->Level(), sourceProd,
			                    component->ForecastType(), config);

			itsLogger.Trace("Fetching " + otherName + " for U/V rotation");

			auto ret = FetchFromAllSources(opts, component->Grid()->IsPackedData());

			auto otherVec = ret.second;
			assert(!otherVec.empty());

			info_t u, v, other = otherVec[0];

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

			interpolate::RotateVectorComponents(*u, *v, config->UseCudaForInterpolation());

			// Most likely both U&V are requested, so interpolate the other one now
			// and put it to cache.

			std::vector<info_t> list({other});
			if (interpolate::Interpolate(*target, list, config->UseCudaForInterpolation()))
			{
				if (itsUseCache && config->UseCache() && !other->Grid()->IsPackedData())
				{
					auto c = GET_PLUGIN(cache);
					c->Insert(*other);
				}
			}
		}
	}
}
