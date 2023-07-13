/**
 * @file json_parser.cpp
 *
 */
#include "json_parser.h"
#include "interpolate.h"
#include "lambert_conformal_grid.h"
#include "lambert_equal_area_grid.h"
#include "latitude_longitude_grid.h"
#include "plugin_factory.h"
#include "point.h"
#include "point_list.h"
#include "reduced_gaussian_grid.h"
#include "stereographic_grid.h"
#include "transverse_mercator_grid.h"
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

void AreaAndGrid(const boost::property_tree::ptree& pt, const shared_ptr<configuration>& conf);
void SourceProducer(const boost::property_tree::ptree& pt, const shared_ptr<configuration>& conf);
void TargetProducer(const boost::property_tree::ptree& pt, const shared_ptr<configuration>& conf);
vector<forecast_type> ParseForecastTypes(const boost::property_tree::ptree& pt);
void Steps(const boost::property_tree::ptree& pt, shared_ptr<configuration>& conf);
raw_time GetLatestOriginDateTime(const shared_ptr<configuration> conf, const string& latest);

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
	if (conf->ConfigurationFileName().empty())
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

template <typename T>
boost::optional<T> ReadElement(const boost::property_tree::ptree& pt, const std::string& name)
{
	try
	{
		return pt.get<T>(name);
	}
	catch (const boost::property_tree::ptree_bad_path& e)
	{
		return boost::none;
	}
	catch (const exception& e)
	{
		throw runtime_error(string("Error parsing key " + name + ": ") + e.what());
	}
}

void WriteMode(const shared_ptr<configuration>& conf, const boost::property_tree::ptree& pt)
{
	// legacy way of defining things

	if (auto fw = ReadElement<string>(pt, "file_write"))
	{
		itsLogger.Warning("Configuration option 'file_write' is deprecated - use 'write_mode' instead");

		string theFileWriteOption = fw.get();

		if (theFileWriteOption == "database")
		{
			conf->WriteMode(kSingleGridToAFile);
			conf->WriteToDatabase(true);
		}
		else if (theFileWriteOption == "single")
		{
			conf->WriteMode(kAllGridsToAFile);
		}
		else if (theFileWriteOption == "multiple")
		{
			conf->WriteMode(kSingleGridToAFile);
		}
		else if (theFileWriteOption == "cache only")
		{
			conf->WriteMode(kNoFileWrite);
		}
		else
		{
			throw runtime_error("Invalid value for file_write: " + theFileWriteOption);
		}

		conf->LegacyWriteMode(true);
	}

	// new way, will overwrite legacy if both are defined

	if (auto fw = ReadElement<string>(pt, "write_mode"))
	{
		string theWriteMode = fw.get();

		if (theWriteMode == "all")
		{
			conf->WriteMode(kAllGridsToAFile);
		}
		else if (theWriteMode == "few")
		{
			conf->WriteMode(kFewGridsToAFile);
		}
		else if (theWriteMode == "single")
		{
			conf->WriteMode(kSingleGridToAFile);
		}
		else if (theWriteMode == "no")
		{
			conf->WriteMode(kNoFileWrite);
		}
		else
		{
			throw runtime_error("Invalid value for file_mode: " + theWriteMode);
		}

		conf->LegacyWriteMode(false);
	}

	if (auto wtd = ReadElement<bool>(pt, "write_to_database"))
	{
		conf->WriteToDatabase(wtd.get());
	}

	// filename template

	if (auto ft = ReadElement<string>(pt, "filename_template"))
	{
		conf->FilenameTemplate(ft.get());
	}
}
void FileCompression(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto fileCompression = ReadElement<string>(pt, "file_compression"))
	{
		if (fileCompression.get() == "gzip")
		{
			conf->FileCompression(kGZIP);
		}
		else if (fileCompression.get() == "bzip2")
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
}

void ReadDataFromDatabase(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto readFromDatabase = ReadElement<bool>(pt, "read_data_from_database"))
	{
		if (readFromDatabase.get() == false || conf->DatabaseType() == kNoDatabase)
		{
			conf->ReadFromDatabase(false);
		}
	}
}

void ReadFromDatabase(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto readFromDatabase = ReadElement<bool>(pt, "read_from_database"))
	{
		if (readFromDatabase.get() == false || conf->DatabaseType() == kNoDatabase)
		{
			conf->ReadFromDatabase(false);
		}
	}
}

void UseCacheForWrites(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto useCacheForWrites = ReadElement<bool>(pt, "use_cache_for_writes"))
	{
		conf->UseCacheForWrites(useCacheForWrites.get());
	}
}

void UseCacheForReads(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto useCacheForReads = ReadElement<bool>(pt, "use_cache_for_reads"))
	{
		conf->UseCacheForReads(useCacheForReads.get());
	}
}

void UseCache(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto useCache = ReadElement<bool>(pt, "use_cache"))
	{
		conf->UseCacheForReads(useCache.get());
		itsLogger.Warning("Key 'use_cache' is deprecated. Rename it to 'use_cache_for_reads'");
	}
}

void CacheLimit(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto cacheLimit = ReadElement<int>(pt, "cache_limit"))
	{
		if (cacheLimit.get() < 1)
		{
			itsLogger.Warning("cache_limit must be larger than 0");
		}
		else
		{
			conf->CacheLimit(cacheLimit.get());
			plugin::cache_pool::Instance()->CacheLimit(cacheLimit.get());
		}
	}
}

void FileType(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto fileType = ReadElement<std::string>(pt, "file_type"))
	{
		auto ft = boost::to_upper_copy(fileType.get());

		if (ft == "GRIB")
		{
			conf->OutputFileType(kGRIB);
		}
		else if (ft == "GRIB1")
		{
			conf->OutputFileType(kGRIB1);
		}
		else if (ft == "GRIB2")
		{
			conf->OutputFileType(kGRIB2);
		}
		else if (ft == "FQD" || ft == "QUERYDATA")
		{
			conf->OutputFileType(kQueryData);
		}
		else if (ft == "CSV")
		{
			conf->OutputFileType(kCSV);
		}
		else if (ft == "GEOTIFF")
		{
			conf->OutputFileType(kGeoTIFF);
		}
		else
		{
			throw runtime_error("Invalid option for 'file_type': " + ft);
		}
	}
}

void AsyncExecution(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto async = ReadElement<bool>(pt, "async"))
	{
		conf->AsyncExecution(async.get());
	}
}

void DynamicMemoryAllocation(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto dma = ReadElement<bool>(pt, "dynamic_memory_allocation"))
	{
		conf->UseDynamicMemoryAllocation(dma.get());
	}
}

void WriteStorageType(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto storageType = ReadElement<std::string>(pt, "write_storage_type"))
	{
		conf->WriteStorageType(HPStringToFileStorageType.at(storageType.get()));
	}
}

void SSStateTableName(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto tableName = ReadElement<std::string>(pt, "ss_state_table_name"))
	{
		conf->SSStateTableName(tableName.get());
	}
}

void FilePackingType(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto packingType = ReadElement<std::string>(pt, "file_packing_type"))
	{
		conf->PackingType(HPStringToPackingType.at(packingType.get()));
	}
}

void AllowedMissingValues(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf, size_t N)
{
	if (auto allowedMissingValues = ReadElement<std::string>(pt, "allowed_missing_values"))
	{
		std::string allowed = allowedMissingValues.get();
		if (allowed.back() == '%')
		{
			allowed.pop_back();
			conf->AllowedMissingValues(static_cast<size_t>(static_cast<double>(N) * 0.01 * stod(allowed)));
		}
		else
		{
			conf->AllowedMissingValues(stol(allowed));
		}
	}
}

void Levels(const boost::property_tree::ptree& pt, shared_ptr<configuration>& conf)
{
	auto levelType = ReadElement<string>(pt, "leveltype");
	auto levelValues = ReadElement<string>(pt, "levels");

	if (levelType && levelValues)
	{
		conf->Levels(LevelsFromString(levelType.get(), levelValues.get()));
	}
}

void Producers(const boost::property_tree::ptree& pt, shared_ptr<configuration>& conf)
{
	SourceProducer(pt, conf);
	TargetProducer(pt, conf);
}

void ForecastTypes(const boost::property_tree::ptree& pt, std::shared_ptr<configuration>& conf)
{
	if (auto ftypes = ReadElement<std::string>(pt, "forecast_type"))
	{
		conf->ForecastTypes(util::ForecastTypesFromString(ftypes.get()));
	}
}

std::vector<raw_time> OriginTime(const boost::property_tree::ptree& pt, shared_ptr<configuration>& conf)
{
	const string mask = "%Y-%m-%d %H:%M:%S";

	std::vector<raw_time> originDateTimes;

	if (auto time = ReadElement<string>(pt, "origintime"))
	{
		auto originDateTime = boost::algorithm::to_lower_copy(time.get());

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
	else if (auto times = ReadElement<string>(pt, "origintimes"))
	{
		auto datesList = himan::util::Split(times.get(), ",");

		for (const auto& dateString : datesList)
		{
			originDateTimes.push_back(raw_time(dateString, mask));
		}
	}

	return originDateTimes;
}

void WriteToObjectStorageBetweenPluginCalls(const boost::property_tree::ptree& pt, shared_ptr<configuration>& conf)
{
	if (auto writeBetweenCalls = ReadElement<bool>(pt, "write_to_object_storage_between_plugin_calls"))
	{
		conf->WriteToObjectStorageBetweenPluginCalls(writeBetweenCalls.get());
	}
}

void ValidateMetadata(const boost::property_tree::ptree& pt, shared_ptr<configuration>& conf)
{
	if (auto validateMetadata = ReadElement<bool>(pt, "validate_metadata"))
	{
		conf->ValidateMetadata(validateMetadata.get());
	}
}

void CheckConsistency(shared_ptr<configuration>& conf)
{
	logger logr("json_parser");
	if (conf->WriteStorageType() != kS3ObjectStorageSystem && conf->WriteToObjectStorageBetweenPluginCalls())
	{
		logr.Warning("unable to set 'write_to_s3_between_plugin_calls=true' when not writing to s3");
		conf->WriteToObjectStorageBetweenPluginCalls(false);
	}
	if (conf->UseCacheForWrites() == false && conf->WriteStorageType() == kS3ObjectStorageSystem &&
	    conf->WriteToObjectStorageBetweenPluginCalls() == false)
	{
		logr.Warning(
		    "Unable to set 'use_cache_for_writes=false' when writing to S3 and write_to_s3_between_plugin_calls=false");
	}
}

void CheckCommonOptions(const boost::property_tree::ptree& pt, shared_ptr<configuration>& conf)
{
	Producers(pt, conf);
	AreaAndGrid(pt, conf);
	Steps(pt, conf);
	FileCompression(pt, conf);
	ReadDataFromDatabase(pt, conf);
	ReadFromDatabase(pt, conf);
	UseCacheForWrites(pt, conf);
	UseCacheForReads(pt, conf);
	UseCache(pt, conf);
	FileType(pt, conf);
	WriteStorageType(pt, conf);
	FilePackingType(pt, conf);
	AllowedMissingValues(pt, conf, conf->BaseGrid()->Size());
	ForecastTypes(pt, conf);
	WriteMode(conf, pt);
	SSStateTableName(pt, conf);
	ValidateMetadata(pt, conf);

	CheckConsistency(conf);
}

vector<shared_ptr<plugin_configuration>> json_parser::ParseConfigurationFile(shared_ptr<configuration> conf)
{
	itsLogger.Trace("Parsing configuration file '" + conf->ConfigurationFileName() + "'");

	boost::property_tree::ptree pt;

	try
	{
		std::stringstream ss;
		ss << conf->ConfigurationFileContent();
		boost::property_tree::json_parser::read_json(ss, pt);
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error reading configuration file: ") + e.what());
	}

	vector<shared_ptr<plugin_configuration>> pluginContainer;

	WriteToObjectStorageBetweenPluginCalls(pt, conf);

	CheckCommonOptions(pt, conf);

	// Only global scope
	CacheLimit(pt, conf);
	DynamicMemoryAllocation(pt, conf);

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
		const auto& pqOpts = element.second;

		boost::property_tree::ptree& plugins = element.second.get_child("plugins");

		if (plugins.empty())
		{
			throw runtime_error(ClassName() + ": plugin definitions not found");
		}

		for (boost::property_tree::ptree::value_type& plugin : plugins)
		{
			const auto& pluginOpts = plugin.second;

			if (pluginOpts.empty())
			{
				throw runtime_error(ClassName() + ": plugin definition is empty");
			}

			shared_ptr<plugin_configuration> pc = make_shared<plugin_configuration>(*conf);
			auto pc_as_conf = std::dynamic_pointer_cast<configuration>(pc);

			// First check options from "processqueue" scope
			// {
			//   "leveltype" : "ground",
			//   "levels" : "0",
			//   "plugins" : [ ... ]
			// }

			CheckCommonOptions(pqOpts, pc_as_conf);
			AsyncExecution(pqOpts, pc_as_conf);
			Levels(pqOpts, pc_as_conf);

			// Then check options from inside one plugin
			// plugins : [
			//   {
			//     "name" : "windvector",
			//     "file_type" : "grib2",
			//     "some_local_option_specific_to_this_plugin" : ...
			//   }
			// ]

			CheckCommonOptions(pluginOpts, pc_as_conf);
			AsyncExecution(pluginOpts, pc_as_conf);

			// Loop through those "local options"
			for (const boost::property_tree::ptree::value_type& kv : pluginOpts)
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
						for (const boost::property_tree::ptree::value_type& listval : kv.second)
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

			pc->OrdinalNumber(static_cast<unsigned int>(pluginContainer.size()));
			pc->RelativeOrdinalNumber(static_cast<unsigned int>(count_if(pluginContainer.begin(), pluginContainer.end(),
			                                                             [pc](const shared_ptr<plugin_configuration>& c)
			                                                             { return c->Name() == pc->Name(); })));

			pluginContainer.push_back(pc);
		}

	}  // END for

	return pluginContainer;
}

raw_time GetLatestOriginDateTime(const shared_ptr<configuration> conf, const string& latest)
{
	using namespace himan;

	auto strlist = himan::util::Split(latest, "-");

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

void Steps(const boost::property_tree::ptree& pt, shared_ptr<configuration>& conf)
{
	auto originDateTimes = OriginTime(pt, conf);

	if (originDateTimes.empty())
	{
		const auto oldTimes = conf->Times();
		if (oldTimes.empty())
		{
			return;
		}

		set<raw_time> uniqTimes;
		for_each(oldTimes.begin(), oldTimes.end(),
		         [&uniqTimes](const forecast_time& ftime) { uniqTimes.insert(ftime.OriginDateTime()); });

		copy(uniqTimes.begin(), uniqTimes.end(), std::back_inserter(originDateTimes));
	}

	ASSERT(originDateTimes.empty() == false);

	auto GenerateList =
	    [&originDateTimes](const time_duration& start, const time_duration& stop, const time_duration& step)
	{
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
		vector<string> timesStr = himan::util::Split(pt.get<string>("times"), ",");
		vector<forecast_time> times;

		for (const auto& originDateTime : originDateTimes)
		{
			for (const auto& str : timesStr)
			{
				times.push_back(forecast_time(originDateTime, time_duration(str)));
			}
		}
		conf->Times(times);
		return;
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

		conf->Times(GenerateList(start, stop, step));
		return;
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
		vector<int> timeValues = util::ExpandString(pt.get<string>("hours"));

		vector<forecast_time> times;

		for (const auto& originDateTime : originDateTimes)
		{
			for (const auto& timeValue : timeValues)
			{
				times.push_back(forecast_time(originDateTime, time_duration(ONE_HOUR * timeValue)));
			}
		}

		conf->Times(times);
		return;
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
		conf->Times(GenerateList(start, stop, step));
		return;
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
		conf->Times(GenerateList(start, stop, step));
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing time information from 'start_minute': ") + e.what());
	}
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
		itsLogger.Fatal(string("Error parsing target area information found from database: ") + e.what());
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
		vector<string> stations = himan::util::Split(pt.get<string>("points"), ",");

		g = unique_ptr<point_list>(new point_list());

		vector<station> theStations;

		int i = 1;

		for (const string& line : stations)
		{
			vector<string> point = himan::util::Split(line, " ");

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
		vector<string> stations = himan::util::Split(pt.get<string>("stations"), ",");

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

unique_ptr<grid> ParseAreaAndGridFromProj4String(const boost::property_tree::ptree& pt)
{
	unique_ptr<grid> g;

	try
	{
		const auto sm = HPScanningModeFromString.at(pt.get<string>("scanning_mode"));

		if (sm != kBottomLeft && sm != kTopLeft)
		{
			throw runtime_error("Only bottom_left or top_left scanning mode is supported");
		}

		const string proj4 = pt.get<string>("proj4");
		const point fp(pt.get<double>("first_point_longitude"), pt.get<double>("first_point_latitude"));
		const size_t ni = pt.get<size_t>("ni");
		const size_t nj = pt.get<size_t>("nj");
		const double di = pt.get<double>("di");
		const double dj = pt.get<double>("dj");

		OGRSpatialReference sp;
		sp.importFromProj4(proj4.c_str());

		const char* projptr = sp.GetAttrValue("PROJECTION");
		himan::logger log("json_parser");

		if (projptr != nullptr)
		{
			const std::string projection = sp.GetAttrValue("PROJECTION");

			if (projection == SRS_PT_LAMBERT_AZIMUTHAL_EQUAL_AREA)
			{
				return std::unique_ptr<lambert_equal_area_grid>(new lambert_equal_area_grid(
				    sm, fp, ni, nj, di, dj, std::unique_ptr<OGRSpatialReference>(sp.Clone()), false));
			}
			else if (projection == SRS_PT_TRANSVERSE_MERCATOR)
			{
				return std::unique_ptr<transverse_mercator_grid>(new transverse_mercator_grid(
				    sm, fp, ni, nj, di, dj, std::unique_ptr<OGRSpatialReference>(sp.Clone()), false));
			}
			else if (projection == SRS_PT_LAMBERT_CONFORMAL_CONIC_2SP)
			{
				return std::unique_ptr<lambert_conformal_grid>(new lambert_conformal_grid(
				    sm, fp, ni, nj, di, dj, std::unique_ptr<OGRSpatialReference>(sp.Clone()), false));
			}

			log.Error("Unsupported projection: " + projection);
		}
		else if (sp.IsGeographic())
		{
			// No projection -- latlon with some datum

			OGRErr erra = 0, errb = 0;
			const double A = sp.GetSemiMajor(&erra);
			const double B = sp.GetSemiMinor(&errb);

			earth_shape<double> es;

			if (erra != OGRERR_NONE || errb != OGRERR_NONE)
			{
				log.Error("Unable to extract datum information from file");
			}
			else
			{
				es = earth_shape<double>(A, B);
			}

			return std::unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid(sm, fp, ni, nj, di, dj, es));
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing proj4: ") + e.what());
	}

	return nullptr;
}

unique_ptr<grid> ParseAreaAndGridFromBbox(const boost::property_tree::ptree& pt)
{
	unique_ptr<grid> g;

	try
	{
		const auto scmode = HPScanningModeFromString.at(pt.get<string>("scanning_mode"));

		if (scmode != kBottomLeft && scmode != kTopLeft)
		{
			throw runtime_error("Only bottom_left or top_left scanning mode is supported with bbox");
		}

		const vector<string> coordinates = himan::util::Split(pt.get<string>("bbox"), ",");

		if (coordinates.size() != 4)
		{
			throw runtime_error("Invalid coordinate format: correct format is x1,y1,x2,y2");
		}
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

	return nullptr;
}

unique_ptr<grid> ParseAreaAndGridFromManualDefinition(const boost::property_tree::ptree& pt)
{
	unique_ptr<grid> g;

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
			g = unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid(
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
			g = unique_ptr<rotated_latitude_longitude_grid>(new rotated_latitude_longitude_grid(
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
			g = unique_ptr<stereographic_grid>(new stereographic_grid(
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
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing area: ") + e.what());
	}

	return std::move(g);
}

void AreaAndGrid(const boost::property_tree::ptree& pt, const shared_ptr<configuration>& conf)
{
	/*
	 * Parse area and grid from different possible options.
	 * Order or parsing:
	 *
	 * 1. 'source_geom_name': this is used in fetching data, it's not used to create an area instance
	 * 2. radon style geom name: 'target_geom_name'
	 * 3. irregular grid: 'points' and 'stations'
	 * 4. bounding box: 'bbox'
	 * 5. proj4 string
	 * 6. manual definition:
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

	if (auto sourceGeom = ReadElement<string>(pt, "source_geom_name"))
	{
		conf->SourceGeomNames(himan::util::Split(sourceGeom.get(), ","));
	}

	// 2. radon-style geom_name

	std::unique_ptr<grid> g;

	if (g = ParseAreaAndGridFromDatabase(*conf, pt))
	{
		conf->BaseGrid(std::move(g));
		return;
	}

	// 3. Points

	if (g = ParseAreaAndGridFromPoints(pt))
	{
		conf->BaseGrid(std::move(g));
		return;
	}

	// 4. Target geometry is still not set, check for bbox

	if (g = ParseAreaAndGridFromBbox(pt))
	{
		conf->BaseGrid(std::move(g));
		return;
	}

	// 5. Check for proj4 string

	if (g = ParseAreaAndGridFromProj4String(pt))
	{
		conf->BaseGrid(std::move(g));
		return;
	}

	// 6. Check for manual definition of area

	if (g = ParseAreaAndGridFromManualDefinition(pt))
	{
		conf->BaseGrid(std::move(g));
		return;
	}
}

void SourceProducer(const boost::property_tree::ptree& pt, const shared_ptr<configuration>& conf)
{
	if (auto sp = ReadElement<string>(pt, "source_producer"))
	{
		std::vector<producer> sourceProducers;
		vector<string> sourceProducersStr = himan::util::Split(sp.get(), ",");

		const HPDatabaseType dbtype = conf->DatabaseType();

		if (dbtype == kRadon)
		{
			auto r = GET_PLUGIN(radon);

			for (const auto& prodstr : sourceProducersStr)
			{
				long pid = stol(prodstr);

				producer prod(pid);

				map<string, string> prodInfo = r->RadonDB().GetProducerDefinition(static_cast<unsigned long>(pid));

				if (prodInfo.empty())
				{
					itsLogger.Fatal("Failed to find source producer from Radon: " + prodstr);
					himan::Abort();
				}

				prod.Name(prodInfo["ref_prod"]);

				if (!prodInfo["ident_id"].empty())
				{
					prod.Centre(stol(prodInfo["ident_id"]));
					prod.Process(stol(prodInfo["model_id"]));
				}

				prod.Class(static_cast<HPProducerClass>(stoi(prodInfo["producer_class"])));

				sourceProducers.push_back(prod);
			}
		}
		else if (dbtype != kNoDatabase && sourceProducers.size() == 0)
		{
			itsLogger.Fatal("Source producer information invalid");
			himan::Abort();
		}
		else if (dbtype == kNoDatabase)
		{
			for (const auto& prodstr : sourceProducersStr)
			{
				sourceProducers.push_back(producer(stoi(prodstr)));
			}
		}

		conf->SourceProducers(sourceProducers);
	}
}

void TargetProducer(const boost::property_tree::ptree& pt, const shared_ptr<configuration>& conf)
{
	if (auto tp = ReadElement<string>(pt, "target_producer"))
	{
		long pid = stol(tp.get());
		producer prod(pid);

		if (conf->DatabaseType() != kNoDatabase)
		{
			auto r = GET_PLUGIN(radon);
			auto prodInfo = r->RadonDB().GetProducerDefinition(static_cast<unsigned long>(pid));

			if (!prodInfo.empty())
			{
				if (!prodInfo["ident_id"].empty() && !prodInfo["model_id"].empty())
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
			else if (conf->DatabaseType() != kNoDatabase)
			{
				itsLogger.Warning("Unknown target producer: " + pt.get<string>("target_producer"));
			}
		}
		conf->TargetProducer(prod);
	}
}

vector<level> LevelsFromString(const string& levelType, const string& levelValuesStr)
{
	HPLevelType theLevelType = HPStringToLevelType.at(boost::to_lower_copy(levelType));
	vector<level> levels;

	if (theLevelType == kHeightLayer || theLevelType == kGroundDepth || theLevelType == kPressureDelta ||
	    theLevelType == kGeneralizedVerticalLayer)
	{
		const vector<string> levelsStr = util::Split(levelValuesStr, ",");
		for (size_t i = 0; i < levelsStr.size(); i++)
		{
			const vector<string> levelIntervals = himan::util::Split(levelsStr[i], "_");

			if (levelIntervals.size() == 2)
			{
				levels.emplace_back(theLevelType, stof(levelIntervals[0]), stof(levelIntervals[1]));
			}
			else if (theLevelType == kGeneralizedVerticalLayer)
			{
				levels.emplace_back(theLevelType, stof(levelIntervals[0]));
			}
			else
			{
				throw runtime_error(
				    "height_layer, ground_depth and pressure delta requires two level values per definition (lx1_ly1, "
				    "lx2_ly2, ..., "
				    "lxN_lyN)");
			}
		}
	}
	else
	{
		const vector<int> levelValues = util::ExpandString(levelValuesStr);
		levels.reserve(levelValues.size());

		std::transform(levelValues.begin(), levelValues.end(), std::back_inserter(levels),
		               [&](int levelValue) { return level(theLevelType, static_cast<float>(levelValue), levelType); });
	}

	ASSERT(!levels.empty());

	return levels;
}
