/**
 * @file json_parser.cpp
 *
 */

#include "json_parser.h"
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
#include <boost/regex.hpp>
#include <map>
#include <stdexcept>
#include <utility>

#define HIMAN_AUXILIARY_INCLUDE

#include "cache.h"
#include "neons.h"
#include "radon.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace std;

unique_ptr<grid> ParseAreaAndGridFromPoints(configuration& conf, const boost::property_tree::ptree& pt);
unique_ptr<grid> ParseAreaAndGridFromDatabase(configuration& conf, const boost::property_tree::ptree& pt);
void ParseLevels(shared_ptr<info> anInfo, const boost::property_tree::ptree& pt);
void ParseProducers(shared_ptr<configuration> conf, shared_ptr<info> anInfo, const boost::property_tree::ptree& pt);
vector<forecast_type> ParseForecastTypes(const boost::property_tree::ptree& pt);

bool ParseBoolean(string booleanValue);
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

json_parser::json_parser() { itsLogger = logger("json_parser"); }
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
	/* Create our base info */

	auto baseInfo = make_shared<info>();

	/* Check producers */

	ParseProducers(conf, baseInfo, pt);

	/* Check area definitions */

	auto g = ParseAreaAndGrid(conf, pt);

	baseInfo->itsBaseGrid = move(g);

	/* Check time definitions */

	conf->FirstSourceProducer();
	ParseTime(conf, baseInfo, pt);

	/* Check levels */

	// ParseLevels(baseInfo, pt);

	/* Check file_write */

	try
	{
		string theFileWriteOption = pt.get<string>("file_write");

		if (theFileWriteOption == "database")
		{
			conf->FileWriteOption(kDatabase);
		}
		else if (theFileWriteOption == "single")
		{
			conf->FileWriteOption(kSingleFile);
		}
		else if (theFileWriteOption == "multiple")
		{
			conf->FileWriteOption(kMultipleFiles);
		}
		else if (theFileWriteOption == "cache only")
		{
			conf->FileWriteOption(kCacheOnly);
		}
		else
		{
			throw runtime_error("Invalid value for file_write: " + theFileWriteOption);
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

		if (conf->FileCompression() != kNoCompression && conf->FileWriteOption() == kSingleFile)
		{
			itsLogger.Warning(
			    "file_write_option value 'single' conflicts with file_compression, using 'multiple' instead");
			conf->FileWriteOption(kMultipleFiles);
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

	/* Check read_data_from_database */

	try
	{
		string theReadDataFromDatabase = pt.get<string>("read_data_from_database");

		if (!ParseBoolean(theReadDataFromDatabase) || conf->DatabaseType() == kNoDatabase)
		{
			conf->ReadDataFromDatabase(false);
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

	// Check global use_cache option

	try
	{
		string theUseCache = pt.get<string>("use_cache");

		if (!ParseBoolean(theUseCache))
		{
			conf->UseCache(false);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing key use_cache: ") + e.what());
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
		throw runtime_error(string("Error parsing key use_cache: ") + e.what());
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

	auto forecastTypes = ParseForecastTypes(pt);

	if (forecastTypes.empty())
	{
		// Default to deterministic
		forecastTypes.push_back(forecast_type(kDeterministic));
	}

	baseInfo->ForecastTypes(forecastTypes);

	/* Check dynamic_memory_allocation */

	try
	{
		string theUseDynamicMemoryAllocation = pt.get<string>("dynamic_memory_allocation");

		if (ParseBoolean(theUseDynamicMemoryAllocation))
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
		throw runtime_error(string("Error parsing key file_write: ") + e.what());
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
		auto anInfo = make_shared<info>(*baseInfo);

		try
		{
			ParseTime(conf, anInfo, element.second);
		}
		catch (...)
		{
			// do nothing
		}

		try
		{
			auto g = ParseAreaAndGrid(conf, element.second);

			anInfo->itsBaseGrid = move(g);
		}
		catch (...)
		{
			// do nothing
		}

		try
		{
			ParseLevels(anInfo, element.second);
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing level information: ") + e.what());
		}

		// Check local use_cache option

		bool delayedUseCache = conf->UseCache();

		try
		{
			string theUseCache = element.second.get<string>("use_cache");

			delayedUseCache = ParseBoolean(theUseCache);
		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			// Something was not found; do nothing
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing use_cache key: ") + e.what());
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

		HPFileWriteOption delayedFileWrite = conf->FileWriteOption();

		try
		{
			string theFileWriteOption = element.second.get<string>("file_write");

			if (theFileWriteOption == "database")
			{
				delayedFileWrite = kDatabase;
			}
			else if (theFileWriteOption == "single")
			{
				delayedFileWrite = kSingleFile;
			}
			else if (theFileWriteOption == "multiple")
			{
				delayedFileWrite = kMultipleFiles;
			}
			else if (theFileWriteOption == "cache only")
			{
				delayedFileWrite = kCacheOnly;
			}
			else
			{
				throw runtime_error("Invalid value for file_write: " + theFileWriteOption);
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

		// Check local forecast_type option

		forecastTypes = ParseForecastTypes(element.second);

		if (!forecastTypes.empty())
		{
			anInfo->ForecastTypes(forecastTypes);
		}

		boost::property_tree::ptree& plugins = element.second.get_child("plugins");

		if (plugins.empty())
		{
			throw runtime_error(ClassName() + ": plugin definitions not found");
		}

		for (boost::property_tree::ptree::value_type& plugin : plugins)
		{
			shared_ptr<plugin_configuration> pc = make_shared<plugin_configuration>(*conf);

			pc->UseCache(delayedUseCache);
			pc->itsOutputFileType = delayedFileType;
			pc->FileWriteOption(delayedFileWrite);

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
				else if (key == "param_list")
				{
					boost::property_tree::ptree params = plugin.second.get_child("param_list");

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

							if (paramOptName == "name")
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
					pc->AsyncExecution(ParseBoolean(value));
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

			pc->Info(make_shared<info>(*anInfo));  // We have to have a copy for all configs.
			                                       // Each plugin will later on create a data backend.

			assert(pc.unique());

			pluginContainer.push_back(pc);
		}

	}  // END for

	return pluginContainer;
}

raw_time GetLatestOriginDateTime(const shared_ptr<configuration> conf, const string& latest)
{
	using namespace himan;

	auto strlist = himan::util::Split(latest, "-", false);

	int offset = 0;

	if (strlist.size() == 2)
	{
		// will throw if origintime is not in the form "latest-X", where X : integer >= 0
		offset = boost::lexical_cast<unsigned int>(strlist[1]);
	}

	HPDatabaseType dbtype = conf->DatabaseType();
	producer sourceProducer = conf->SourceProducer();

	map<string, string> prod;

	raw_time latestOriginDateTime;

	if (dbtype == kNeons || dbtype == kNeonsAndRadon)
	{
		auto n = GET_PLUGIN(neons);

		prod = n->NeonsDB().GetProducerDefinition(static_cast<unsigned long>(sourceProducer.Id()));

		if (!prod.empty())
		{
			auto latestFromDatabase = n->NeonsDB().GetLatestTime(prod["ref_prod"], "", offset);

			if (!latestFromDatabase.empty())
			{
				latestOriginDateTime = raw_time(latestFromDatabase, "%Y%m%d%H%M");
			}
		}
	}
	if (latestOriginDateTime.Empty() && (dbtype == kRadon || dbtype == kNeonsAndRadon))
	{
		auto r = GET_PLUGIN(radon);

		auto latestFromDatabase = r->RadonDB().GetLatestTime(sourceProducer.Id(), "", offset);

		if (!latestFromDatabase.empty())
		{
			latestOriginDateTime = raw_time(latestFromDatabase, "%Y-%m-%d %H:%M:%S");
		}
	}

	if (latestOriginDateTime.Empty())
	{
		throw runtime_error("Latest time not found from " + HPDatabaseTypeToString.at(dbtype) + " for producer " +
		                    boost::lexical_cast<string>(sourceProducer.Id()));
	}

	return latestOriginDateTime;
}

void json_parser::ParseTime(shared_ptr<configuration> conf, std::shared_ptr<info> anInfo,
                            const boost::property_tree::ptree& pt)
{
	/* Check origin time */
	const string mask = "%Y-%m-%d %H:%M:%S";

	std::vector<raw_time> originDateTimes;

	try
	{
		auto originDateTime = pt.get<string>("origintime");

		boost::algorithm::to_lower(originDateTime);

		if (boost::regex_search(originDateTime, boost::regex("latest")))
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
		catch (boost::property_tree::ptree_bad_path& e)
		{
			throw runtime_error("Origin datetime not found with keys 'origintime' or 'origindatetimes'");
		}
	}
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing origin time information: ") + e.what());
	}

	assert(!originDateTimes.empty());

	/* Check time steps */

	/*
	 * Three ways of providing information on steps:
	 * - hours
	 * - start_hour + stop_hour + step
	 * - start_minute + stop_minute + step
	 */

	try
	{
		string hours = pt.get<string>("hours");
		vector<string> timesStr = himan::util::Split(hours, ",", true);

		vector<int> times;

		for (size_t i = 0; i < timesStr.size(); i++)
		{
			times.push_back(boost::lexical_cast<int>(timesStr[i]));
		}

		sort(times.begin(), times.end());

		vector<forecast_time> theTimes;

		// Create forecast_time with both times origintime, then adjust the validtime

		for (const auto& originDateTime : originDateTimes)
		{
			for (int hour : times)
			{
				forecast_time theTime(originDateTime, originDateTime);

				theTime.ValidDateTime().Adjust(kHourResolution, hour);

				theTimes.push_back(theTime);
			}
		}

		anInfo->Times(theTimes);

		return;
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing time information from 'times': ") + e.what());
	}

	// hours was not specified
	// check if start/stop times are

	// First check step_unit which is deprecated and issue warning
	try
	{
		string stepUnit = pt.get<string>("step_unit");

		if (!stepUnit.empty())
		{
			itsLogger.Warning("Key 'step_unit' is deprecated");
		}
	}
	catch (exception& e)
	{
	}

	try
	{
		int start = pt.get<int>("start_hour");
		int stop = pt.get<int>("stop_hour");
		int step = pt.get<int>("step");

		if (step <= 0)
		{
			throw runtime_error("step size must be > 0");
		}

		conf->itsForecastStep = step;

		HPTimeResolution stepResolution = kHourResolution;

		vector<forecast_time> theTimes;

		for (const auto& originDateTime : originDateTimes)
		{
			int curtime = start;

			do
			{
				forecast_time theTime(originDateTime, originDateTime);

				theTime.ValidDateTime().Adjust(stepResolution, curtime);

				theTime.StepResolution(stepResolution);

				theTimes.push_back(theTime);

				curtime += step;

			} while (curtime <= stop);
		}

		anInfo->Times(theTimes);

		return;
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing time information from 'start_hour': ") +
		                    e.what());
	}

	try
	{
		// try start_minute/stop_minute

		int start = pt.get<int>("start_minute");
		int stop = pt.get<int>("stop_minute");
		int step = pt.get<int>("step");

		conf->itsForecastStep = step;

		HPTimeResolution stepResolution = kMinuteResolution;

		int curtime = start;

		vector<forecast_time> theTimes;

		for (const auto& originDateTime : originDateTimes)
		{
			do
			{
				forecast_time theTime(originDateTime, originDateTime);

				theTime.ValidDateTime().Adjust(stepResolution, curtime);

				theTime.StepResolution(stepResolution);

				theTimes.push_back(theTime);

				curtime += step;

			} while (curtime <= stop);
		}

		anInfo->Times(theTimes);
	}
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing time information: ") + e.what());
	}
}

unique_ptr<grid> ParseAreaAndGridFromDatabase(configuration& conf, const boost::property_tree::ptree& pt)
{
	using himan::kTopLeft;
	using himan::kBottomLeft;

	unique_ptr<grid> g;

	try
	{
		string geom = pt.get<string>("target_geom_name");

		conf.TargetGeomName(geom);

		HPDatabaseType dbtype = conf.DatabaseType();

		map<string, string> geominfo;

		double scale = 1;

		if (dbtype == kNeons || dbtype == kNeonsAndRadon)
		{
			auto n = GET_PLUGIN(neons);

			geominfo = n->NeonsDB().GetGeometryDefinition(geom);
			scale = 0.001;
		}

		if (geominfo.empty() && (dbtype == kRadon || dbtype == kNeonsAndRadon))
		{
			auto r = GET_PLUGIN(radon);

			geominfo = r->RadonDB().GetGeometryDefinition(geom);
		}

		if (geominfo.empty())
		{
			throw runtime_error("Fatal::json_parser Unknown geometry '" + geom + "' found");
		}

		if ((geominfo["prjn_name"] == "latlon" && geominfo["geom_parm_1"] == "0") || geominfo["prjn_id"] == "1")
		{
			g = unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid);
			latitude_longitude_grid* const llg = dynamic_cast<latitude_longitude_grid*>(g.get());

			const double di = scale * boost::lexical_cast<double>(geominfo["pas_longitude"]);
			const double dj = scale * boost::lexical_cast<double>(geominfo["pas_latitude"]);

			llg->Ni(boost::lexical_cast<size_t>(geominfo["col_cnt"]));
			llg->Nj(boost::lexical_cast<size_t>(geominfo["row_cnt"]));
			llg->Di(di);
			llg->Dj(dj);

			if (geominfo["stor_desc"] == "+x-y")
			{
				llg->ScanningMode(kTopLeft);
			}
			else if (geominfo["stor_desc"] == "+x+y")
			{
				llg->ScanningMode(kBottomLeft);
			}
			else
			{
				throw runtime_error("Fatal::json_parser scanning mode " + geominfo["stor_desc"] + " not supported yet");
			}

			const double X0 = boost::lexical_cast<double>(geominfo["long_orig"]) * scale;
			const double Y0 = boost::lexical_cast<double>(geominfo["lat_orig"]) * scale;

			const double X1 = fmod(X0 + (llg->Ni() - 1) * di, 360);

			double Y1 = kHPMissingValue;

			switch (llg->ScanningMode())
			{
				case kTopLeft:
					Y1 = Y0 - (llg->Nj() - 1) * dj;
					break;
				case kBottomLeft:
					Y1 = Y0 + (llg->Nj() - 1) * dj;
					break;
				default:
					break;
			}

			llg->FirstPoint(point(X0, Y0));
			llg->LastPoint(point(X1, Y1));
		}
		else if ((geominfo["prjn_name"] == "latlon" &&
		          (geominfo["geom_parm_1"] != "0" || geominfo["geom_parm_2"] != "0"))  // neons
		         || (geominfo["prjn_id"] == "4"))                                      // radon
		{
			g = unique_ptr<rotated_latitude_longitude_grid>(new rotated_latitude_longitude_grid);
			rotated_latitude_longitude_grid* const rllg = dynamic_cast<rotated_latitude_longitude_grid*>(g.get());

			const double di = scale * boost::lexical_cast<double>(geominfo["pas_longitude"]);
			const double dj = scale * boost::lexical_cast<double>(geominfo["pas_latitude"]);

			rllg->Ni(boost::lexical_cast<size_t>(geominfo["col_cnt"]));
			rllg->Nj(boost::lexical_cast<size_t>(geominfo["row_cnt"]));
			rllg->Di(di);
			rllg->Dj(dj);

			if (geominfo["stor_desc"] == "+x-y")
			{
				rllg->ScanningMode(kTopLeft);
			}
			else if (geominfo["stor_desc"] == "+x+y")
			{
				rllg->ScanningMode(kBottomLeft);
			}
			else
			{
				throw runtime_error("Fatal::json_parser scanning mode " + geominfo["stor_desc"] + " not supported yet");
			}

			rllg->SouthPole(point(boost::lexical_cast<double>(geominfo["geom_parm_2"]) * scale,
			                      boost::lexical_cast<double>(geominfo["geom_parm_1"]) * scale));

			const double X0 = boost::lexical_cast<double>(geominfo["long_orig"]) * scale;
			const double Y0 = boost::lexical_cast<double>(geominfo["lat_orig"]) * scale;

			const double X1 = fmod(X0 + (rllg->Ni() - 1) * di, 360);

			double Y1 = kHPMissingValue;

			switch (rllg->ScanningMode())
			{
				case kTopLeft:
					Y1 = Y0 - (rllg->Nj() - 1) * dj;
					break;
				case kBottomLeft:
					Y1 = Y0 + (rllg->Nj() - 1) * dj;
					break;
				default:
					break;
			}

			rllg->FirstPoint(point(X0, Y0));
			rllg->LastPoint(point(X1, Y1));
		}
		else if (geominfo["prjn_name"] == "polster" || geominfo["prjn_name"] == "polarstereo" ||
		         geominfo["prjn_id"] == "2")
		{
			g = unique_ptr<stereographic_grid>(new stereographic_grid);
			stereographic_grid* const sg = dynamic_cast<stereographic_grid*>(g.get());

			const double di = boost::lexical_cast<double>(geominfo["pas_longitude"]);
			const double dj = boost::lexical_cast<double>(geominfo["pas_latitude"]);

			sg->Orientation(boost::lexical_cast<double>(geominfo["geom_parm_1"]) * scale);
			sg->Di(di);
			sg->Dj(dj);

			sg->Ni(boost::lexical_cast<size_t>(geominfo["col_cnt"]));
			sg->Nj(boost::lexical_cast<size_t>(geominfo["row_cnt"]));

			if (geominfo["stor_desc"] == "+x+y")
			{
				sg->ScanningMode(kBottomLeft);
			}
			else
			{
				throw runtime_error("Fatal::json_parser scanning mode " + geominfo["stor_desc"] +
				                    " not supported yet for stereographic grid");
			}

			const double X0 = boost::lexical_cast<double>(geominfo["long_orig"]) * scale;
			const double Y0 = boost::lexical_cast<double>(geominfo["lat_orig"]) * scale;

			const auto coordinates = himan::util::CoordinatesFromFirstGridPoint(point(X0, Y0), sg->Orientation(),
			                                                                    sg->Ni(), sg->Nj(), di, dj);

			sg->BottomLeft(coordinates.first);
			sg->TopRight(coordinates.second);
		}
		else if (geominfo["prjn_id"] == "6")
		{
			g = unique_ptr<reduced_gaussian_grid>(new reduced_gaussian_grid);
			reduced_gaussian_grid* const gg = dynamic_cast<reduced_gaussian_grid*>(g.get());

			gg->N(boost::lexical_cast<int>(geominfo["n"]));
			gg->Nj(boost::lexical_cast<int>(geominfo["nj"]));

			auto strlongitudes = himan::util::Split(geominfo["longitudes_along_parallels"], ",", false);
			vector<int> longitudes;

			for (auto& l : strlongitudes)
			{
				longitudes.push_back(boost::lexical_cast<int>(l));
			}

			gg->NumberOfPointsAlongParallels(longitudes);

			assert(boost::lexical_cast<size_t>(geominfo["n"]) * 2 == longitudes.size());

			const point first(boost::lexical_cast<double>(geominfo["first_point_lon"]),
			                  boost::lexical_cast<double>(geominfo["first_point_lat"]));
			const point last(boost::lexical_cast<double>(geominfo["last_point_lon"]),
			                 boost::lexical_cast<double>(geominfo["last_point_lat"]));

			if (geominfo["scanning_mode"] == "+x-y")
			{
				gg->ScanningMode(kTopLeft);
				gg->BottomLeft(point(first.X(), last.Y()));
				gg->TopRight(point(last.X(), first.Y()));
				gg->TopLeft(first);
				gg->BottomRight(last);
			}
			else if (geominfo["scanning_mode"] == "+x+y")
			{
				gg->ScanningMode(kBottomLeft);
				gg->BottomLeft(first);
				gg->TopRight(last);
				gg->TopLeft(point(first.X(), last.Y()));
				gg->BottomRight(point(last.X(), first.Y()));
			}
			else
			{
				throw runtime_error("Fatal::json_parser scanning mode " + geominfo["stor_desc"] + " not supported yet");
			}
		}
		else if (geominfo["prjn_id"] == "5")
		{
			g = unique_ptr<lambert_conformal_grid>(new lambert_conformal_grid);
			lambert_conformal_grid* const lcg = dynamic_cast<lambert_conformal_grid*>(g.get());

			lcg->Ni(boost::lexical_cast<int>(geominfo["ni"]));
			lcg->Nj(boost::lexical_cast<int>(geominfo["nj"]));

			lcg->Di(boost::lexical_cast<double>(geominfo["di"]));
			lcg->Dj(boost::lexical_cast<double>(geominfo["dj"]));

			lcg->Orientation(boost::lexical_cast<double>(geominfo["orientation"]));

			lcg->StandardParallel1(boost::lexical_cast<double>(geominfo["latin1"]));

			if (!geominfo["latin2"].empty())
			{
				lcg->StandardParallel1(boost::lexical_cast<double>(geominfo["latin2"]));
			}

			if (!geominfo["south_pole_lon"].empty())
			{
				const point sp(boost::lexical_cast<double>(geominfo["south_pole_lon"]),
				               boost::lexical_cast<double>(geominfo["south_pole_lat"]));

				lcg->SouthPole(sp);
			}

			const point first(boost::lexical_cast<double>(geominfo["first_point_lon"]),
			                  boost::lexical_cast<double>(geominfo["first_point_lat"]));

			if (geominfo["scanning_mode"] == "+x-y")
			{
				lcg->ScanningMode(kTopLeft);
				lcg->TopLeft(first);
			}
			else if (geominfo["scanning_mode"] == "+x+y")
			{
				lcg->ScanningMode(kBottomLeft);
				lcg->BottomLeft(first);
			}
			else
			{
				throw runtime_error("Fatal::json_parser scanning mode " + geominfo["stor_desc"] + " not supported yet");
			}
		}
		else
		{
			throw runtime_error("Fatal::json_parser Unknown projection: " + geominfo["prjn_name"]);
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Fatal::json_parser Error parsing area information: ") + e.what());
	}

	return g;
}

unique_ptr<grid> ParseAreaAndGridFromPoints(configuration& conf, const boost::property_tree::ptree& pt)
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

			theStations.push_back(station(i, "point_" + boost::lexical_cast<string>(i),
			                              boost::lexical_cast<double>(lon), boost::lexical_cast<double>(lat)));

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
				fmisid = boost::lexical_cast<unsigned long>(str);
			}
			catch (boost::bad_lexical_cast& e)
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

			theStations.push_back(station(fmisid, stationinfo["station_name"],
			                              boost::lexical_cast<double>(stationinfo["longitude"]),
			                              boost::lexical_cast<double>(stationinfo["latitude"])));
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

unique_ptr<grid> json_parser::ParseAreaAndGrid(shared_ptr<configuration> conf, const boost::property_tree::ptree& pt)
{
	/*
	 * Parse area and grid from different possible options.
	 * Order or parsing:
	 *
	 * 1. 'source_geom_name': this is used in fetching data, it's not used to create an area instance
	 * 2. neons style geom name: 'target_geom_name'
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

	// 2. neons-style geom_name

	auto g = ParseAreaAndGridFromDatabase(*conf, pt);

	if (g)
	{
		return g;
	}

	// 3. Points

	auto ig = ParseAreaAndGridFromPoints(*conf, pt);

	if (ig)
	{
		// Disable cuda interpolation (too inefficienct for single points)
		itsLogger.Trace("Disabling cuda interpolation for single point data");
		conf->UseCudaForInterpolation(false);
		return ig;
	}

	// 4. Target geometry is still not set, check for bbox

	unique_ptr<grid> rg;

	try
	{
		vector<string> coordinates = himan::util::Split(pt.get<string>("bbox"), ",", false);

		rg = unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid);

		dynamic_cast<latitude_longitude_grid*>(rg.get())->BottomLeft(
		    point(boost::lexical_cast<double>(coordinates[0]), boost::lexical_cast<double>(coordinates[1])));
		dynamic_cast<latitude_longitude_grid*>(rg.get())->TopRight(
		    point(boost::lexical_cast<double>(coordinates[2]), boost::lexical_cast<double>(coordinates[3])));

		dynamic_cast<latitude_longitude_grid*>(rg.get())->Ni(pt.get<size_t>("ni"));
		dynamic_cast<latitude_longitude_grid*>(rg.get())->Nj(pt.get<size_t>("nj"));

		rg->ScanningMode(HPScanningModeFromString.at(pt.get<string>("scanning_mode")));

		return rg;
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
			throw runtime_error(ClassName() + ": scanning mode " + HPScanningModeToString.at(mode) +
			                    " not supported (ever)");
		}

		string projection = pt.get<string>("projection");

		if (projection == "latlon")
		{
			rg = unique_ptr<latitude_longitude_grid>(new latitude_longitude_grid);
			dynamic_cast<latitude_longitude_grid*>(rg.get())->BottomLeft(
			    point(pt.get<double>("bottom_left_longitude"), pt.get<double>("bottom_left_latitude")));
			dynamic_cast<latitude_longitude_grid*>(rg.get())->TopRight(
			    point(pt.get<double>("top_right_longitude"), pt.get<double>("top_right_latitude")));
			dynamic_cast<latitude_longitude_grid*>(rg.get())->Ni(pt.get<size_t>("ni"));
			dynamic_cast<latitude_longitude_grid*>(rg.get())->Nj(pt.get<size_t>("nj"));
		}
		else if (projection == "rotated_latlon")
		{
			rg = unique_ptr<rotated_latitude_longitude_grid>(new rotated_latitude_longitude_grid);
			dynamic_cast<rotated_latitude_longitude_grid*>(rg.get())->BottomLeft(
			    point(pt.get<double>("bottom_left_longitude"), pt.get<double>("bottom_left_latitude")));
			dynamic_cast<rotated_latitude_longitude_grid*>(rg.get())->TopRight(
			    point(pt.get<double>("top_right_longitude"), pt.get<double>("top_right_latitude")));
			dynamic_cast<rotated_latitude_longitude_grid*>(rg.get())->SouthPole(
			    point(pt.get<double>("south_pole_longitude"), pt.get<double>("south_pole_latitude")));
			dynamic_cast<rotated_latitude_longitude_grid*>(rg.get())->Ni(pt.get<size_t>("ni"));
			dynamic_cast<rotated_latitude_longitude_grid*>(rg.get())->Nj(pt.get<size_t>("nj"));
		}
		else if (projection == "stereographic")
		{
			rg = unique_ptr<stereographic_grid>(new stereographic_grid);
			dynamic_cast<stereographic_grid*>(rg.get())->BottomLeft(
			    point(pt.get<double>("bottom_left_longitude"), pt.get<double>("bottom_left_latitude")));
			dynamic_cast<stereographic_grid*>(rg.get())->TopRight(
			    point(pt.get<double>("top_right_longitude"), pt.get<double>("top_right_latitude")));
			dynamic_cast<stereographic_grid*>(rg.get())->Orientation(pt.get<double>("orientation"));
			dynamic_cast<stereographic_grid*>(rg.get())->Ni(pt.get<size_t>("ni"));
			dynamic_cast<stereographic_grid*>(rg.get())->Nj(pt.get<size_t>("nj"));
		}
		else
		{
			throw runtime_error(ClassName() + ": Unknown type: " + projection);
		}

		rg->ScanningMode(mode);
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		throw runtime_error(e.what());
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing area: ") + e.what());
	}

	return rg;
}

void ParseProducers(shared_ptr<configuration> conf, shared_ptr<info> anInfo, const boost::property_tree::ptree& pt)
{
	try
	{
		std::vector<producer> sourceProducers;
		vector<string> sourceProducersStr = himan::util::Split(pt.get<string>("source_producer"), ",", false);

		HPDatabaseType dbtype = conf->DatabaseType();

		if (dbtype == kNeons || dbtype == kNeonsAndRadon)
		{
			auto n = GET_PLUGIN(neons);

			for (const auto& prodstr : sourceProducersStr)
			{
				long pid = stol(prodstr);

				producer prod(pid);

				map<string, string> prodInfo = n->NeonsDB().GetGridModelDefinition(static_cast<unsigned long>(pid));

				if (!prodInfo.empty())
				{
					prod.Centre(boost::lexical_cast<long>(prodInfo["ident_id"]));
					prod.Name(prodInfo["ref_prod"]);
					prod.TableVersion(boost::lexical_cast<long>(prodInfo["no_vers"]));
					prod.Process(boost::lexical_cast<long>(prodInfo["model_id"]));
					prod.Class(kGridClass);
					sourceProducers.push_back(prod);
				}
				else
				{
					itsLogger.Warning("Failed to find source producer from Neons: " + prodstr);
				}
			}
		}

		if (sourceProducers.size() == 0 && (dbtype == kRadon || dbtype == kNeonsAndRadon))
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
						prod.Centre(boost::lexical_cast<long>(prodInfo["ident_id"]));
						prod.Process(boost::lexical_cast<long>(prodInfo["model_id"]));
					}

					prod.Class(static_cast<HPProducerClass>(stoi(prodInfo["producer_class"])));

					if (dbtype == kNeonsAndRadon)
					{
						itsLogger.Info("Forcing database type to radon");
						conf->DatabaseType(kRadon);
					}
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
			abort();
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

	catch (boost::property_tree::ptree_bad_path& e)
	{
		itsLogger.Fatal("Source producer definitions not found: " + string(e.what()));
		abort();
	}
	catch (exception& e)
	{
		itsLogger.Fatal("Error parsing source producer information: " + string(e.what()));
		abort();
	}

	try
	{
		/*
		 * Target producer is also set to target info; source infos (and producers) are created
		 * as data is fetched from files.
		 */

		HPDatabaseType dbtype = conf->DatabaseType();

		long pid = boost::lexical_cast<long>(pt.get<string>("target_producer"));
		producer prod(pid);

		map<string, string> prodInfo;

		if (dbtype == kNeons || dbtype == kNeonsAndRadon)
		{
			auto n = GET_PLUGIN(neons);
			prodInfo = n->NeonsDB().GetGridModelDefinition(static_cast<unsigned long>(pid));
		}

		if (prodInfo.empty() && (dbtype == kRadon || dbtype == kNeonsAndRadon))
		{
			auto r = GET_PLUGIN(radon);
			prodInfo = r->RadonDB().GetProducerDefinition(static_cast<unsigned long>(pid));
		}

		if (!prodInfo.empty())
		{
			if (prodInfo["ident_id"].empty() || prodInfo["model_id"].empty())
			{
				itsLogger.Warning("Centre or ident information not found for producer " + prodInfo["ref_prod"]);
			}
			else
			{
				prod.Centre(boost::lexical_cast<long>(prodInfo["ident_id"]));
				prod.Process(boost::lexical_cast<long>(prodInfo["model_id"]));
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

		conf->TargetProducer(prod);
		anInfo->Producer(prod);
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		itsLogger.Fatal("Target producer definitions not found: " + string(e.what()));
		abort();
	}
	catch (exception& e)
	{
		itsLogger.Fatal("Error parsing target producer information: " + string(e.what()));
		abort();
	}
}

void ParseLevels(shared_ptr<info> anInfo, const boost::property_tree::ptree& pt)
{
	try
	{
		string levelTypeStr = pt.get<string>("leveltype");
		string levelValuesStr = pt.get<string>("levels");

		vector<level> levels = LevelsFromString(levelTypeStr, levelValuesStr);

		anInfo->Levels(levels);
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

	const vector<string> levelsStr = himan::util::Split(levelValues, ",", true);

	if (theLevelType == kHeightLayer || theLevelType == kGroundDepth)
	{
		for (size_t i = 0; i < levelsStr.size(); i++)
		{
			const vector<string> levelIntervals = himan::util::Split(levelsStr[i], "_", false);

			if (levelIntervals.size() != 2)
			{
				throw runtime_error(
				    "height_layer and ground_depth requires two level values per definition (lx1_ly1, lx2_ly2, ..., "
				    "lxN_lyN)");
			}

			levels.push_back(level(theLevelType, boost::lexical_cast<float>(levelIntervals[0]),
			                       boost::lexical_cast<float>(levelIntervals[1])));
		}
	}
	else
	{
		for (size_t i = 0; i < levelsStr.size(); i++)
		{
			levels.push_back(level(theLevelType, boost::lexical_cast<float>(levelsStr[i]), levelType));
		}
	}

	assert(!levels.empty());

	return levels;
}

/*
 * ParseBoolean()
 *
 * Will check if given argument is a boolean value or not.
 * Note: will change argument to lower case.
 */

bool ParseBoolean(string booleanValue)
{
	bool ret;

	boost::algorithm::to_lower(booleanValue);

	if (booleanValue == "yes" || booleanValue == "true" || booleanValue == "1")
	{
		ret = true;
	}

	else if (booleanValue == "no" || booleanValue == "false" || booleanValue == "0")
	{
		ret = false;
	}

	else
	{
		itsLogger.Fatal("Invalid boolean value: " + booleanValue);
		abort();
	}

	return ret;
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

			if (boost::regex_search(type, boost::regex("pf")))
			{
				forecastType = kEpsPerturbation;
				string list = "";
				for (size_t i = 2; i < type.size(); i++) list += type[i];

				vector<string> range = himan::util::Split(list, "-", false);

				if (range.size() == 1)
				{
					forecastTypes.push_back(forecast_type(forecastType, boost::lexical_cast<double>(range[0])));
				}
				else
				{
					assert(range.size() == 2);

					int start = boost::lexical_cast<int>(range[0]);
					int stop = boost::lexical_cast<int>(range[1]);

					while (start <= stop)
					{
						forecastTypes.push_back(forecast_type(forecastType, boost::lexical_cast<double>(start)));
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
				else if (type == "deterministic")
				{
					forecastTypes.push_back(forecast_type(kDeterministic));
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
