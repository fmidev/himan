/**
 * @file json_parser.cpp
 *
 */
#include "json_parser.h"
#include "interpolate.h"
#include "lambert_conformal_grid.h"
#include "latitude_longitude_grid.h"
#include "plugin_factory.h"
#include "point.h"
#include "point_list.h"
#include "reduced_gaussian_grid.h"
#include "stereographic_grid.h"
#include "util.h"
#include <boost/algorithm/string/trim.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <map>
#include <ogr_spatialref.h>
#include <stdexcept>
#include <utility>

#define HIMAN_AUXILIARY_INCLUDE

#include "cache.h"
#include "radon.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace std;

unique_ptr<grid> ParseAreaAndGridFromPoints(const boost::property_tree::ptree& pt);
unique_ptr<grid> ParseAreaAndGridFromDatabase(configuration& conf, const boost::property_tree::ptree& pt);
vector<level> ParseLevels(const boost::property_tree::ptree& pt);
vector<producer> ParseSourceProducer(const shared_ptr<configuration>& conf, const boost::property_tree::ptree& pt);
producer ParseTargetProducer(const shared_ptr<configuration>& conf, const boost::property_tree::ptree& pt);
vector<forecast_type> ParseForecastTypes(const boost::property_tree::ptree& pt);
unique_ptr<grid> ParseAreaAndGrid(const std::shared_ptr<configuration>& conf, const boost::property_tree::ptree& pt);
vector<forecast_time> ParseTime(std::shared_ptr<configuration> conf, const boost::property_tree::ptree& pt);
tuple<HPWriteMode, bool, bool, string> ParseWriteMode(const shared_ptr<configuration>& conf,
                                                      const boost::property_tree::ptree& pt);

vector<level> LevelsFromString(const string& levelType, const string& levelValues);

static logger itsLogger;

/*
 * Parse()
 *
 * Read command line options and create info instance.
 *
 * Steps taken:
 *
 * 1) Read command line options. Options specified in command line will
 *	override those in the conf file.
 *
 * 2) Read configuration file (if specified).
 *
 * 3) Create configuration instance.
 *
 * Some of the required information is missing, this function will not
 * behave nicely and will throw an error.
 *
 */

json_parser::json_parser()
{
	itsLogger = logger("json_parser");
}
vector<shared_ptr<plugin_configuration>> json_parser::Parse(shared_ptr<configuration> conf)
{
	if (conf->ConfigurationFile().empty())
	{
		throw runtime_error("Configuration file not defined");
	}

	vector<shared_ptr<plugin_configuration>> plugins = ParseConfigurationFile(conf);

	if (plugins.size() == 0)
	{
		throw runtime_error("Empty processqueue");
	}

	return plugins;
}

vector<shared_ptr<plugin_configuration>> json_parser::ParseConfigurationFile(shared_ptr<configuration> conf)
{
	itsLogger.Trace("Parsing configuration file '" + conf->ConfigurationFile() + "'");

	boost::property_tree::ptree pt;

	try
	{
		boost::property_tree::json_parser::read_json(conf->ConfigurationFile(), pt);
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error reading configuration file: ") + e.what());
	}

	vector<shared_ptr<plugin_configuration>> pluginContainer;

	/* Check producers */

	conf->SourceProducers(ParseSourceProducer(conf, pt));
	conf->TargetProducer(ParseTargetProducer(conf, pt));

	/* Check area definitions */

	auto g_targetGrid = ParseAreaAndGrid(conf, pt);

	/* Check time definitions */

	const auto g_times = ParseTime(conf, pt);

	/* Check file_write */

	auto parsed = ParseWriteMode(conf, pt);

	if (get<0>(parsed) != kUnknown)
	{
		conf->WriteMode(get<0>(parsed));
		conf->WriteToDatabase(get<1>(parsed));
		conf->LegacyWriteMode(get<2>(parsed));
		conf->FilenameTemplate(get<3>(parsed));
	}

	/* Check file_compression */

	try
	{
		string theFileCompression = pt.get<string>("file_compression");

		if (theFileCompression == "gzip")
		{
			conf->FileCompression(kGZIP);
		}
		else if (theFileCompression == "bzip2")
		{
			conf->FileCompression(kBZIP2);
		}
		else
		{
			conf->FileCompression(kNoCompression);
		}

		if (conf->FileCompression() != kNoCompression && conf->WriteMode() == kAllGridsToAFile)
		{
			itsLogger.Warning("file_mode value 'all' conflicts with file_compression, using 'single' instead");
			conf->WriteMode(kSingleGridToAFile);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key file_write: ") + e.what());
	}

	/* Check read_data_from_database (legacy) */

	try
	{
		string theReadFromDatabase = pt.get<string>("read_data_from_database");

		if (!util::ParseBoolean(theReadFromDatabase) || conf->DatabaseType() == kNoDatabase)
		{
			conf->ReadFromDatabase(false);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key read_data_from_database: ") + e.what());
	}

	/* Check read_from_database */

	try
	{
		string theReadFromDatabase = pt.get<string>("read_from_database");

		if (!util::ParseBoolean(theReadFromDatabase) || conf->DatabaseType() == kNoDatabase)
		{
			conf->ReadFromDatabase(false);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key read_from_database: ") + e.what());
	}

	// Check global use_cache_for_writes option

	try
	{
		string theUseCacheForWrites = pt.get<string>("use_cache_for_writes");

		if (!util::ParseBoolean(theUseCacheForWrites))
		{
			conf->UseCacheForWrites(false);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key use_cache_for_writes: ") + e.what());
	}

	// Check global use_cache option

	// For backwards compatibility
	try
	{
		string theUseCache = pt.get<string>("use_cache");

		if (!util::ParseBoolean(theUseCache))
		{
			conf->UseCacheForReads(false);
		}
		itsLogger.Warning("Key 'use_cache' is deprecated. Rename it to 'use_cache_for_reads'");
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key use_cache: ") + e.what());
	}

	try
	{
		string theUseCache = pt.get<string>("use_cache_for_reads");

		if (!util::ParseBoolean(theUseCache))
		{
			conf->UseCacheForReads(false);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key use_cache_for_reads: ") + e.what());
	}

	// Check global cache_limit option

	try
	{
		int theCacheLimit = pt.get<int>("cache_limit");

		if (theCacheLimit < 1)
		{
			itsLogger.Warning("cache_limit must be larger than 0");
		}
		else
		{
			conf->CacheLimit(theCacheLimit);
			plugin::cache_pool::Instance()->CacheLimit(theCacheLimit);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key cache_limit: ") + e.what());
	}

	// Check global file_type option

	try
	{
		string theFileType = boost::to_upper_copy(pt.get<string>("file_type"));

		if (theFileType == "GRIB")
		{
			conf->itsOutputFileType = kGRIB;
		}
		else if (theFileType == "GRIB1")
		{
			conf->itsOutputFileType = kGRIB1;
		}
		else if (theFileType == "GRIB2")
		{
			conf->itsOutputFileType = kGRIB2;
		}
		else if (theFileType == "FQD" || theFileType == "QUERYDATA")
		{
			conf->itsOutputFileType = kQueryData;
		}
		else
		{
			throw runtime_error("Invalid option for 'file_type': " + theFileType);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key file_type: ") + e.what());
	}

	// Check global forecast_type option

	auto g_forecastTypes = ParseForecastTypes(pt);

	if (g_forecastTypes.empty())
	{
		// Default to deterministic
		g_forecastTypes.push_back(forecast_type(kDeterministic));
	}

	/* Check dynamic_memory_allocation */

	try
	{
		string theUseDynamicMemoryAllocation = pt.get<string>("dynamic_memory_allocation");

		if (util::ParseBoolean(theUseDynamicMemoryAllocation))
		{
			conf->UseDynamicMemoryAllocation(true);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key dynamic_memory_allocation: ") + e.what());
	}

	/* Check storage_type */

	try
	{
		string theStorageType = pt.get<string>("write_storage_type");

		conf->WriteStorageType(HPStringToFileStorageType.at(theStorageType));
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key storage_type: ") + e.what());
	}

	/* Check packing_type */

	try
	{
		const string thePackingType = pt.get<string>("file_packing_type");

		conf->PackingType(HPStringToPackingType.at(thePackingType));
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key storage_type: ") + e.what());
	}

	/*
	 * Check processqueue.
	 *
	 * Some configuration elements might be replicated here; if so they will overwrite
	 * those specified in the upper level.
	 */

	boost::property_tree::ptree& pq = pt.get_child("processqueue");

	if (pq.empty())
	{
		throw runtime_error(ClassName() + ": processqueue missing");
	}

	for (boost::property_tree::ptree::value_type& element : pq)
	{
		auto times = g_times;
		auto targetGrid = unique_ptr<grid>(g_targetGrid->Clone());

		try
		{
			times = ParseTime(conf, element.second);
		}
		catch (...)
		{
			// do nothing
		}

		try
		{
			targetGrid = ParseAreaAndGrid(conf, element.second);
		}
		catch (...)
		{
			// do nothing
		}

		vector<level> g_levels;

		try
		{
			g_levels = ParseLevels(element.second);
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing level information: ") + e.what());
		}

		// Check local use_cache_for_writes option

		bool delayedUseCacheForWrites = conf->UseCacheForWrites();

		try
		{
			string theUseCache = element.second.get<string>("use_cache_for_writes");

			delayedUseCacheForWrites = util::ParseBoolean(theUseCache);
		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			// Something was not found; do nothing
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing use_cache_for_writes key: ") + e.what());
		}

		// Check local use_cache option

		bool delayedUseCacheForReads = conf->UseCacheForReads();

		try
		{
			string theUseCache = element.second.get<string>("use_cache");

			delayedUseCacheForReads = util::ParseBoolean(theUseCache);

			itsLogger.Warning("Key 'use_cache' is deprecated. Rename it to 'use_cache_for_reads'");
		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			// Something was not found; do nothing
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing use_cache key: ") + e.what());
		}

		try
		{
			string theUseCache = element.second.get<string>("use_cache_for_reads");

			delayedUseCacheForReads = util::ParseBoolean(theUseCache);
		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			// Something was not found; do nothing
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing use_cache_for_reads key: ") + e.what());
		}

		// Check local file_type option

		HPFileType delayedFileType = conf->itsOutputFileType;

		// Check local async options

		try
		{
			string async = element.second.get<string>("async");
			if (async == "true")
			{
				conf->AsyncExecution(true);
			}
		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			// Something was not found; do nothing
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing async key: ") + e.what());
		}

		try
		{
			string theFileType = boost::to_upper_copy(element.second.get<string>("file_type"));

			if (theFileType == "GRIB")
			{
				delayedFileType = kGRIB;
			}
			else if (theFileType == "GRIB1")
			{
				delayedFileType = kGRIB1;
			}
			else if (theFileType == "GRIB2")
			{
				delayedFileType = kGRIB2;
			}
			else if (theFileType == "FQD" || theFileType == "QUERYDATA")
			{
				delayedFileType = kQueryData;
			}
			else
			{
				throw runtime_error("Invalid option for 'file_type': " + theFileType);
			}
		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			// Something was not found; do nothing
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing meta information: ") + e.what());
		}

		// Check local file_write option

		HPWriteMode delayedWriteMode = conf->WriteMode();
		bool delayedWriteToDatabase = conf->WriteToDatabase();
		bool delayedLegacyWriteMode = conf->LegacyWriteMode();
		string delayedFilenameTemplate = conf->FilenameTemplate();
		auto delayedParsed = ParseWriteMode(conf, element.second);
		auto delayedPackingType = conf->PackingType();

		if (get<0>(delayedParsed) != kUnknown)
		{
			delayedWriteMode = get<0>(delayedParsed);
			delayedWriteToDatabase = get<1>(delayedParsed);
			delayedLegacyWriteMode = get<2>(delayedParsed);
			delayedFilenameTemplate = get<3>(delayedParsed);
		}

		// Check local forecast_type option

		auto forecastTypes = ParseForecastTypes(element.second);

		if (forecastTypes.empty())
		{
			forecastTypes = g_forecastTypes;
		}

		boost::property_tree::ptree& plugins = element.second.get_child("plugins");

		// Check local producer option

		vector<producer> delayedSourceProducers = conf->SourceProducers();

		try
		{
			delayedSourceProducers = ParseSourceProducer(conf, element.second);
		}
		catch (...)
		{
		}

		producer delayedTargetProducer = conf->TargetProducer();

		try
		{
			delayedTargetProducer = ParseTargetProducer(conf, element.second);
		}
		catch (...)
		{
		}

		/* Check local packing_type option */

		try
		{
			const string thePackingType = pt.get<string>("file_packing_type");

			delayedPackingType = HPStringToPackingType.at(thePackingType);
		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			// Something was not found; do nothing
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing key storage_type: ") + e.what());
		}

		if (plugins.empty())
		{
			throw runtime_error(ClassName() + ": plugin definitions not found");
		}

		for (boost::property_tree::ptree::value_type& plugin : plugins)
		{
			shared_ptr<plugin_configuration> pc = make_shared<plugin_configuration>(*conf);

			pc->itsTimes = times;
			pc->itsForecastTypes = forecastTypes;
			pc->itsLevels = g_levels;

			pc->itsBaseGrid = unique_ptr<grid>(targetGrid->Clone());
			pc->UseCacheForReads(delayedUseCacheForReads);
			pc->UseCacheForWrites(delayedUseCacheForWrites);
			pc->itsOutputFileType = delayedFileType;
			pc->WriteMode(delayedWriteMode);
			pc->WriteToDatabase(delayedWriteToDatabase);
			pc->LegacyWriteMode(delayedLegacyWriteMode);
			pc->FilenameTemplate(delayedFilenameTemplate);
			pc->SourceProducers(delayedSourceProducers);
			pc->TargetProducer(delayedTargetProducer);
			pc->PackingType(delayedPackingType);

			if (plugin.second.empty())
			{
				throw runtime_error(ClassName() + ": plugin definition is empty");
			}

			for (boost::property_tree::ptree::value_type& kv : plugin.second)
			{
				string key = kv.first;
				string value;

				try
				{
					value = kv.second.get<string>("");
				}
				catch (...)
				{
					continue;
				}

				if (key == "name")
				{
					pc->Name(value);
				}
				else if (key == "param_list" || key == "options")
				{
					boost::property_tree::ptree params = plugin.second.get_child(key);

					if (params.empty())
					{
						throw runtime_error(ClassName() + ": param_list definition is empty");
					}

					for (boost::property_tree::ptree::value_type& param : params)
					{
						string name;
						std::vector<std::pair<std::string, std::string>> opts;

						for (boost::property_tree::ptree::value_type& paramOpt : param.second)
						{
							string paramOptName = paramOpt.first;
							string paramOptValue = paramOpt.second.get<string>("");

							if (paramOptName.empty())
							{
								throw runtime_error(ClassName() + ": param_list parameter option name is empty");
							}

							if (paramOptValue.empty())
							{
								throw runtime_error(ClassName() + ": param_list parameter option '" + paramOptName +
								                    "' value is empty");
							}

							if (paramOptName == "name" || paramOptName == "producer")
							{
								name = paramOptValue;
							}
							else
							{
								opts.push_back(std::make_pair(paramOptName, paramOptValue));
							}
						}
						pc->AddParameter(name, opts);
					}
				}
				else if (key == "async")
				{
					pc->AsyncExecution(util::ParseBoolean(value));
				}
				else
				{
					if (value.empty())
					{
						for (boost::property_tree::ptree::value_type& listval : kv.second)
						{
							// pc->AddOption(key, value);
							pc->AddOption(key, himan::util::Expand(listval.second.get<string>("")));
						}
					}
					else
					{
						pc->AddOption(key, himan::util::Expand(value));
					}
				}
			}

			if (pc->Name().empty())
			{
				throw runtime_error(ClassName() + ": plugin name not found from configuration");
			}

			ASSERT(pc.unique());

			pc->OrdinalNumber(static_cast<unsigned int>(pluginContainer.size()));
			pc->RelativeOrdinalNumber(static_cast<unsigned int>(
			    count_if(pluginContainer.begin(), pluginContainer.end(),
			             [pc](const shared_ptr<plugin_configuration>& c) { return c->Name() == pc->Name(); })));

			pluginContainer.push_back(pc);
		}

	}  // END for

	return pluginContainer;
}

raw_time GetLatestOriginDateTime(const shared_ptr<configuration> conf, const string& latest)
{
	using namespace himan;

	auto strlist = himan::util::Split(latest, "-", false);

	unsigned int offset = 0;

	if (strlist.size() == 2)
	{
		// will throw if origintime is not in the form "latest-X", where X : integer >= 0
		offset = static_cast<unsigned>(stoi(strlist[1]));
	}

	ASSERT(conf->SourceProducers().empty() == false);

	const HPDatabaseType dbtype = conf->DatabaseType();
	const producer sourceProducer = conf->SourceProducer(0);

	raw_time latestOriginDateTime;

	auto r = GET_PLUGIN(radon);

	auto latestFromDatabase = r->RadonDB().GetLatestTime(static_cast<int>(sourceProducer.Id()), "", offset);

	if (!latestFromDatabase.empty())
	{
		logger log("json_parser");
		log.Debug("Latest analysis time: " + latestFromDatabase);
		return raw_time(latestFromDatabase, "%Y-%m-%d %H:%M:%S");
	}

	throw runtime_error("Latest time not found from " + HPDatabaseTypeToString.at(dbtype) + " for producer " +
	                    to_string(sourceProducer.Id()));
}

vector<forecast_time> ParseSteps(shared_ptr<configuration>& conf, const boost::property_tree::ptree& pt,
                                 const vector<raw_time>& originDateTimes)
{
	auto GenerateList = [&originDateTimes](const time_duration& start, const time_duration& stop,
	                                       const time_duration& step) {
		vector<forecast_time> times;
		for (const auto& originDateTime : originDateTimes)
		{
			auto curtime = start;

			do
			{
				forecast_time theTime(originDateTime, curtime);
				times.push_back(theTime);

				curtime += step;

			} while (curtime <= stop);
		}
		return times;
	};

	/*
	 * Three DEPRECATED ways of providing information on steps:
	 * - hours
	 * - start_hour + stop_hour + step
	 * - start_minute + stop_minute + step
	 *
	 * The new ways of specifying time are:
	 * - times
	 * - start_time + stop_time + step
	 */

	try
	{
		vector<string> timesStr = himan::util::Split(pt.get<string>("times"), ",", true);
		vector<forecast_time> times;

		for (const auto& originDateTime : originDateTimes)
		{
			for (const auto& str : timesStr)
			{
				times.push_back(forecast_time(originDateTime, time_duration(str)));
			}
		}
		return times;
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing time information from 'times': ") + e.what());
	}

	try
	{
		auto start = time_duration(pt.get<string>("start_time"));
		auto stop = time_duration(pt.get<string>("stop_time"));
		auto step = time_duration(pt.get<string>("step"));

		conf->ForecastStep(step);

		return GenerateList(start, stop, step);
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing time information from 'start_hour': ") + e.what());
	}

	try
	{
		vector<string> timesStr = himan::util::Split(pt.get<string>("hours"), ",", true);
		vector<forecast_time> times;

		for (const auto& originDateTime : originDateTimes)
		{
			for (const auto& str : timesStr)
			{
				times.push_back(forecast_time(originDateTime, time_duration(str)));
			}
		}

		return times;
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing time information from 'hours': ") + e.what());
	}

	// hours was not specified
	// check if start/stop times are

	try
	{
		auto start = time_duration(pt.get<string>("start_hour") + ":00");
		auto stop = time_duration(pt.get<string>("stop_hour") + ":00");
		auto step = time_duration(pt.get<string>("step") + ":00");

		conf->ForecastStep(step);

		return GenerateList(start, stop, step);
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing time information from 'start_hour': ") + e.what());
	}

	try
	{
		// try start_minute/stop_minute

		auto start = time_duration("00:" + pt.get<string>("start_minute"));
		auto stop = time_duration("00:" + pt.get<string>("stop_minute"));
		auto step = time_duration("00:" + pt.get<string>("step"));

		conf->ForecastStep(step);

		return GenerateList(start, stop, step);
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing time information: ") + e.what());
	}
}

vector<forecast_time> ParseTime(shared_ptr<configuration> conf, const boost::property_tree::ptree& pt)
{
	vector<forecast_time> theTimes;

	/* Check origin time */
	const string mask = "%Y-%m-%d %H:%M:%S";

	std::vector<raw_time> originDateTimes;

	try
	{
		auto originDateTime = pt.get<string>("origintime");

		boost::algorithm::to_lower(originDateTime);

		if (originDateTime.find("latest") != string::npos)
		{
			if (conf->DatabaseType() == kNoDatabase)
			{
				throw std::invalid_argument("Unable to get latest time from database when no database mode is enabled");
			}
			originDateTimes.push_back(GetLatestOriginDateTime(conf, originDateTime));
		}
		else
		{
			originDateTimes.push_back(raw_time(originDateTime, mask));
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		try
		{
			auto datesList = himan::util::Split(pt.get<string>("origintimes"), ",", false);

			for (const auto& dateString : datesList)
			{
				originDateTimes.push_back(raw_time(dateString, mask));
			}
		}
		catch (boost::property_tree::ptree_bad_path& ee)
		{
			throw runtime_error("Origin datetime not found with keys 'origintime' or 'origindatetimes'");
		}
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing origin time information: ") + e.what());
	}

	ASSERT(!originDateTimes.empty());

	/* Check time steps */

	return ParseSteps(conf, pt, originDateTimes);
}

unique_ptr<grid> ParseAreaAndGridFromDatabase(configuration& conf, const boost::property_tree::ptree& pt)
{
	using himan::kBottomLeft;
	using himan::kTopLeft;

	unique_ptr<grid> g;

	try
	{
		string geom = pt.get<string>("target_geom_name");

		conf.TargetGeomName(geom);

		g = util::GridFromDatabase(geom);
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		itsLogger.Fatal(string("Error parsing area information: ") + e.what());
		himan::Abort();
	}

	return g;
}

unique_ptr<grid> ParseAreaAndGridFromPoints(const boost::property_tree::ptree& pt)
{
	unique_ptr<grid> g;

	// check points

	try
	{
		vector<string> stations = himan::util::Split(pt.get<string>("points"), ",", false);

		g = unique_ptr<point_list>(new point_list());

		vector<station> theStations;

		int i = 1;

		for (const string& line : stations)
		{
			vector<string> point = himan::util::Split(line, " ", false);

			if (point.size() != 2)
			{
				cout << "Error::json_parser Line " + line + " is invalid" << endl;
				continue;
			}

			string lon = point[0];
			string lat = point[1];

			boost::algorithm::trim(lon);
			boost::trim(lat);

			theStations.push_back(station(kHPMissingInt, "", stod(lon), stod(lat)));

			i++;
		}

		if (theStations.size())
		{
			dynamic_cast<point_list*>(g.get())->Stations(theStations);
			return g;
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (const exception& e)
	{
		throw runtime_error(string("Fatal::json_parser Error parsing points: ") + e.what());
	}

	// Check stations

	try
	{
		vector<string> stations = himan::util::Split(pt.get<string>("stations"), ",", false);

		g = unique_ptr<point_list>(new point_list);

		vector<station> theStations;

		auto r = GET_PLUGIN(radon);

		for (const string& str : stations)
		{
			unsigned long fmisid;

			try
			{
				fmisid = static_cast<unsigned long>(stol(str));
			}
			catch (const invalid_argument& e)
			{
				cout << "Error::json_parser Invalid fmisid: " << str << endl;
				continue;
			}

			auto stationinfo = r->RadonDB().GetStationDefinition(kFmiSIDNetwork, fmisid);

			if (stationinfo.empty())
			{
				cout << "Error::json_parser Station " << str << " not found from database" << endl;
				continue;
			}

			theStations.push_back(station(static_cast<int>(fmisid), stationinfo["station_name"],
			                              stod(stationinfo["longitude"]), stod(stationinfo["latitude"])));
		}

		dynamic_cast<point_list*>(g.get())->Stations(theStations);
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (const exception& e)
	{
		throw runtime_error(string("Fatal::json_parser Error parsing stations: ") + e.what());
	}

	if (g && dynamic_cast<point_list*>(g.get())->Stations().empty())
	{
		throw runtime_error("Fatal::json_parser No valid points or stations found");
	}

	return g;
}

unique_ptr<grid> ParseAreaAndGrid(const shared_ptr<configuration>& conf, const boost::property_tree::ptree& pt)
{
	/*
	 * Parse area and grid from different possible options.
	 * Order or parsing:
	 *
	 * 1. 'source_geom_name': this is used in fetching data, it's not used to create an area instance
	 * 2. radon style geom name: 'target_geom_name'
	 * 3. irregular grid: 'points' and 'stations'
	 * 4. bounding box: 'bbox'
	 * 5. manual definition:
	 * -> 'projection',
	 * -> 'bottom_left_longitude', 'bottom_left_latitude',
	 * -> 'top_right_longitude', 'top_right_latitude'
	 * -> 'orientation'
	 * -> 'south_pole_longitude', 'south_pole_latitude'
	 * -> 'ni', 'nj'
	 * -> 'scanning_mode'
	 *
	 */

	// 1. Check for source geom name

	try
	{
		conf->SourceGeomNames(himan::util::Split(pt.get<string>("source_geom_name"), ",", false));
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing area information: ") + e.what());
	}

	// 2. radon-style geom_name

	auto g = ParseAreaAndGridFromDatabase(*conf, pt);

	if (g)
	{
		return g;
	}

	// 3. Points

	auto ig = ParseAreaAndGridFromPoints(pt);

	if (ig)
	{
		return ig;
	}

	// 4. Target geometry is still not set, check for bbox

	try
	{
		const auto scmode = HPScanningModeFromString.at(pt.get<string>("scanning_mode"));

		if (scmode != kBottomLeft && scmode != kTopLeft)
		{
			throw runtime_error("Only bottom_left or top_left scanning mode is supported with bbox");
		}

		vector<string> coordinates = himan::util::Split(pt.get<string>("bbox"), ",", false);

		point fp, lp;
		if (scmode == kTopLeft)
		{
			// 'bbox' is always bl,tr -- have to juggle a bit here
			fp = point(stod(coordinates[0]), stod(coordinates[3]));
			lp = point(stod(coordinates[2]), stod(coordinates[1]));
		}
		else
		{
			fp = point(stod(coordinates[0]), stod(coordinates[1]));
			lp = point(stod(coordinates[2]), stod(coordinates[3]));
		}

		// clang-format off
		return unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid(
		    scmode,
		    fp,
		    lp,
		    pt.get<size_t>("ni"),
		    pt.get<size_t>("nj"),
		    earth_shape<double>(6371220.)
		));
		// clang-format on
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing bbox: ") + e.what());
	}

	// 5. Check for manual definition of area

	try
	{
		HPScanningMode mode = HPScanningModeFromString.at(pt.get<string>("scanning_mode"));

		if (mode != kTopLeft && mode != kBottomLeft)
		{
			throw runtime_error("scanning mode " + HPScanningModeToString.at(mode) + " not supported (ever)");
		}

		string projection = pt.get<string>("projection");

		if (projection == "latlon")
		{
			// clang-format off
			return unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid(
			    mode,
			    point(pt.get<double>("bottom_left_longitude"), pt.get<double>("bottom_left_latitude")),
			    point(pt.get<double>("top_right_longitude"), pt.get<double>("top_right_latitude")),
			    pt.get<size_t>("ni"),
			    pt.get<size_t>("nj"),
			    earth_shape<double>(6371220.)));
			// clang-format on
		}
		else if (projection == "rotated_latlon")
		{
			// clang-format off
			return unique_ptr<rotated_latitude_longitude_grid>(new rotated_latitude_longitude_grid(
			    mode,
			    point(pt.get<double>("bottom_left_longitude"), pt.get<double>("bottom_left_latitude")),
			    point(pt.get<double>("top_right_longitude"), pt.get<double>("top_right_latitude")),
			    pt.get<size_t>("ni"),
			    pt.get<size_t>("nj"),
			    earth_shape<double>(6371220.),
			    point(pt.get<double>("south_pole_longitude"), pt.get<double>("south_pole_latitude"))));
			// clang-format on
		}
		else if (projection == "stereographic")
		{
			// clang-format off
			return unique_ptr<stereographic_grid>(new stereographic_grid(
			    mode,
			    point(pt.get<double>("first_point_longitude"), pt.get<double>("first_point_latitude")),
			    pt.get<size_t>("ni"),
			    pt.get<size_t>("nj"),
			    pt.get<double>("di"),
			    pt.get<double>("dj"),
			    pt.get<double>("orientation"),
			    earth_shape<double>(6371220.),
			    false
			));
			// clang-format on
		}
		else
		{
			throw runtime_error("Unknown type: " + projection);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		throw runtime_error(e.what());
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing area: ") + e.what());
	}
}

std::vector<producer> ParseSourceProducer(const shared_ptr<configuration>& conf, const boost::property_tree::ptree& pt)
{
	std::vector<producer> sourceProducers;
	vector<string> sourceProducersStr = himan::util::Split(pt.get<string>("source_producer"), ",", false);

	const HPDatabaseType dbtype = conf->DatabaseType();

	if (dbtype == kRadon)
	{
		auto r = GET_PLUGIN(radon);

		for (const auto& prodstr : sourceProducersStr)
		{
			long pid = stol(prodstr);

			producer prod(pid);

			map<string, string> prodInfo = r->RadonDB().GetProducerDefinition(static_cast<unsigned long>(pid));

			if (!prodInfo.empty())
			{
				prod.Name(prodInfo["ref_prod"]);

				if (!prodInfo["ident_id"].empty())
				{
					prod.Centre(stol(prodInfo["ident_id"]));
					prod.Process(stol(prodInfo["model_id"]));
				}

				prod.Class(static_cast<HPProducerClass>(stoi(prodInfo["producer_class"])));

				sourceProducers.push_back(prod);
			}
			else
			{
				itsLogger.Warning("Failed to find source producer from Radon: " + prodstr);
			}
		}
	}
	else if (dbtype != kNoDatabase && sourceProducers.size() == 0)
	{
		itsLogger.Fatal("Source producer information was not found from database");
		himan::Abort();
	}
	else if (dbtype == kNoDatabase)
	{
		for (const auto& prodstr : sourceProducersStr)
		{
			sourceProducers.push_back(producer(stoi(prodstr)));
		}
	}

	return sourceProducers;
}

producer ParseTargetProducer(const shared_ptr<configuration>& conf, const boost::property_tree::ptree& pt)
{
	const HPDatabaseType dbtype = conf->DatabaseType();

	/*
	 * Target producer is also set to target info; source infos (and producers) are created
	 * as data is fetched from files.
	 */

	long pid = stol(pt.get<string>("target_producer"));
	producer prod(pid);

	auto r = GET_PLUGIN(radon);
	auto prodInfo = r->RadonDB().GetProducerDefinition(static_cast<unsigned long>(pid));

	if (!prodInfo.empty())
	{
		if (prodInfo["ident_id"].empty() || prodInfo["model_id"].empty())
		{
			itsLogger.Warning("Centre or ident information not found for producer " + prodInfo["ref_prod"]);
		}
		else
		{
			prod.Centre(stol(prodInfo["ident_id"]));
			prod.Process(stol(prodInfo["model_id"]));
		}

		prod.Name(prodInfo["ref_prod"]);

		if (prodInfo["producer_class"].empty())
		{
			prod.Class(kGridClass);
		}
		else
		{
			prod.Class(static_cast<HPProducerClass>(stoi(prodInfo["producer_class"])));
		}
	}
	else if (dbtype != kNoDatabase)
	{
		itsLogger.Warning("Unknown target producer: " + pt.get<string>("target_producer"));
	}

	return prod;
}

vector<level> ParseLevels(const boost::property_tree::ptree& pt)
{
	try
	{
		string levelTypeStr = pt.get<string>("leveltype");
		string levelValuesStr = pt.get<string>("levels");

		return LevelsFromString(levelTypeStr, levelValuesStr);
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		throw runtime_error(e.what());
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(e.what());
	}
}

vector<level> LevelsFromString(const string& levelType, const string& levelValues)
{
	HPLevelType theLevelType = HPStringToLevelType.at(boost::to_lower_copy(levelType));
	vector<level> levels;

	if (theLevelType == kHeightLayer || theLevelType == kGroundDepth || theLevelType == kPressureDelta)
	{
		const vector<string> levelsStr = himan::util::Split(levelValues, ",", false);
		for (size_t i = 0; i < levelsStr.size(); i++)
		{
			const vector<string> levelIntervals = himan::util::Split(levelsStr[i], "_", false);

			if (levelIntervals.size() != 2)
			{
				throw runtime_error(
				    "height_layer, ground_depth and pressure delta requires two level values per definition (lx1_ly1, "
				    "lx2_ly2, ..., "
				    "lxN_lyN)");
			}

			levels.push_back(level(theLevelType, stof(levelIntervals[0]), stof(levelIntervals[1])));
		}
	}
	else
	{
		const vector<string> levelsStr = himan::util::Split(levelValues, ",", true);
		for (size_t i = 0; i < levelsStr.size(); i++)
		{
			levels.push_back(level(theLevelType, stof(levelsStr[i]), levelType));
		}
	}

	ASSERT(!levels.empty());

	return levels;
}

vector<forecast_type> ParseForecastTypes(const boost::property_tree::ptree& pt)
{
	vector<forecast_type> forecastTypes;

	try
	{
		vector<string> types = himan::util::Split(pt.get<string>("forecast_type"), ",", false);

		for (string& type : types)
		{
			boost::algorithm::to_lower(type);
			HPForecastType forecastType;

			if (type.find("pf") != string::npos)
			{
				forecastType = kEpsPerturbation;
				string list = "";
				for (size_t i = 2; i < type.size(); i++)
					list += type[i];

				vector<string> range = himan::util::Split(list, "-", false);

				if (range.size() == 1)
				{
					forecastTypes.push_back(forecast_type(forecastType, stod(range[0])));
				}
				else
				{
					ASSERT(range.size() == 2);

					int start = stoi(range[0]);
					int stop = stoi(range[1]);

					while (start <= stop)
					{
						forecastTypes.push_back(forecast_type(forecastType, start));
						start++;
					}
				}
			}
			else
			{
				if (type == "cf")
				{
					forecastTypes.push_back(forecast_type(kEpsControl, 0));
				}
				else if (type == "det" || type == "deterministic")
				{
					forecastTypes.push_back(forecast_type(kDeterministic));
				}
				else if (type == "an" || type == "analysis")
				{
					forecastTypes.push_back(forecast_type(kAnalysis));
				}
				else if (type == "sp" || type == "statistical")
				{
					forecastTypes.push_back(forecast_type(kStatisticalProcessing));
				}
				else
				{
					throw runtime_error("Invalid forecast_type: " + type);
				}
			}
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (const boost::exception& e)
	{
		throw runtime_error(string("Invalid forecast_type value"));
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key forecast_type: ") + e.what());
	}

	return forecastTypes;
}

tuple<HPWriteMode, bool, bool, string> ParseWriteMode(const shared_ptr<configuration>& conf,
                                                      const boost::property_tree::ptree& pt)
{
	HPWriteMode writeMode = kUnknown;
	bool writeToDatabase = false;
	bool legacyWriteMode = false;

	// legacy way of defining things

	try
	{
		string theFileWriteOption = pt.get<string>("file_write");

		if (theFileWriteOption == "database")
		{
			writeMode = kSingleGridToAFile;
			writeToDatabase = true;
		}
		else if (theFileWriteOption == "single")
		{
			writeMode = kAllGridsToAFile;
			writeToDatabase = false;
		}
		else if (theFileWriteOption == "multiple")
		{
			writeMode = kSingleGridToAFile;
			writeToDatabase = false;
		}
		else if (theFileWriteOption == "cache only")
		{
			writeMode = kNoFileWrite;
			writeToDatabase = false;
		}
		else
		{
			throw runtime_error("Invalid value for file_write: " + theFileWriteOption);
		}

		legacyWriteMode = true;
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing meta information: ") + e.what());
	}

	// new way, will overwrite legacy if both are defined

	try
	{
		string theWriteMode = pt.get<string>("write_mode");

		if (theWriteMode == "all")
		{
			writeMode = kAllGridsToAFile;
		}
		else if (theWriteMode == "few")
		{
			writeMode = kFewGridsToAFile;
		}
		else if (theWriteMode == "single")
		{
			writeMode = kSingleGridToAFile;
		}
		else if (theWriteMode == "no")
		{
			writeMode = kNoFileWrite;
		}
		else
		{
			throw runtime_error("Invalid value for file_mode: " + theWriteMode);
		}

		writeToDatabase = util::ParseBoolean(pt.get<string>("write_to_database"));

		legacyWriteMode = false;
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing meta information: ") + e.what());
	}

	// filename template

	string filenameTemplate("");
	try
	{
		filenameTemplate = pt.get<string>("filename_template");
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing meta information: ") + e.what());
	}

	return make_tuple(writeMode, writeToDatabase, legacyWriteMode, filenameTemplate);
}
