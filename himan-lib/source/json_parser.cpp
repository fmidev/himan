/**
 * @file json_parser.cpp
 *
 * @date Nov 19, 2012
 * @author partio, revised aalto
 */

#include "json_parser.h"
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <stdexcept>
#include "plugin_factory.h"
#include "logger_factory.h"
#include "util.h"
#include <map>
#include "point.h"
#include "regular_grid.h"
#include "irregular_grid.h"

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"
#include "radon.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace std;
//using namespace boost::property_tree;

unique_ptr<irregular_grid> ParseAreaAndGridFromPoints(configuration& conf, const boost::property_tree::ptree& pt);
unique_ptr<regular_grid> ParseAreaAndGridFromDatabase(configuration& conf, const boost::property_tree::ptree& pt);

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

unique_ptr<json_parser> json_parser::itsInstance;

json_parser* json_parser::Instance()
{

	if (!itsInstance)
	{
		itsInstance = unique_ptr<json_parser> (new json_parser);
	}

	return itsInstance.get();
}

json_parser::json_parser()
{
	itsLogger = logger_factory::Instance()->GetLog("json_parser");
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

	itsLogger->Trace("Parsing configuration file '" + conf->ConfigurationFile() + "'");

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

	auto baseInfo = make_shared<info> ();

	/* Check producers */

	ParseProducers(conf, baseInfo, pt);

	/* Check area definitions */

	auto g = ParseAreaAndGrid(conf, pt);

	baseInfo->itsBaseGrid = move(g);
	
	/* Check time definitions */

	conf->FirstSourceProducer();
	ParseTime(conf, baseInfo, pt);

	/* Check levels */

	//ParseLevels(baseInfo, pt);

	/* Check file_write */

	try
	{

		string theFileWriteOption = pt.get<string>("file_write");

		if (theFileWriteOption == "neons")
		{
			itsLogger->Warning("file_write_option value 'neons' has been deprecated, use 'database' instead");
			conf->FileWriteOption(kDatabase);
		}
		else if (theFileWriteOption == "database")
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

	/* Check read_data_from_database */

	try
	{
		string theReadDataFromDatabase = pt.get<string>("read_data_from_database");

		if (!ParseBoolean(theReadDataFromDatabase))
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
		throw runtime_error(string("Error parsing meta information: ") + e.what());
	}

	/* Check file_wait_timeout */

	try
	{
		string theFileWaitTimeout = pt.get<string>("file_wait_timeout");

		conf->itsFileWaitTimeout = boost::lexical_cast<unsigned short> (theFileWaitTimeout);

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing meta information: ") + e.what());
	}

	/* Check leading_dimension */

	try
	{
		string theLeadingDimensionStr = pt.get<string>("leading_dimension");

		HPDimensionType theLeadingDimension = kUnknownDimension;

		if (theLeadingDimensionStr == "time")
		{
			theLeadingDimension = kTimeDimension;
		}
		else if (theLeadingDimensionStr == "level")
		{
			theLeadingDimension = kLevelDimension;
		}
		else
		{
			throw runtime_error(ClassName() + ": unsupported leading dimension: " + theLeadingDimensionStr);
		}

		conf->itsLeadingDimension = theLeadingDimension;

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing meta information: ") + e.what());
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
		throw runtime_error(string("Error parsing meta information: ") + e.what());
	}

	// Check global file_type option

	try
	{
		string theFileType = boost::to_upper_copy(pt.get<string>("file_type"));

		if (theFileType == "GRIB")
		{
			conf->itsOutputFileType = kGRIB;
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
		throw runtime_error(string("Error parsing meta information: ") + e.what());
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
	
	BOOST_FOREACH(boost::property_tree::ptree::value_type &element, pq)
	{
		std::shared_ptr<info> anInfo (new info(*baseInfo));

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
				
		try
		{
			string theFileType = boost::to_upper_copy(element.second.get<string>("file_type"));

			if (theFileType == "GRIB")
			{
				delayedFileType = kGRIB;
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

			if (theFileWriteOption == "neons")
			{
				itsLogger->Warning("file_write_option value 'neons' has been deprecated, use 'database' instead");
				delayedFileWrite = kDatabase;
			}
			else if (theFileWriteOption == "database")
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

		boost::property_tree::ptree& plugins = element.second.get_child("plugins");

		if (plugins.empty())
		{
			throw runtime_error(ClassName() + ": plugin definitions not found");
		}


		BOOST_FOREACH(boost::property_tree::ptree::value_type &plugin, plugins)
		{
			shared_ptr<plugin_configuration> pc = make_shared<plugin_configuration> (*conf);

			pc->UseCache(delayedUseCache);
			pc->itsOutputFileType = delayedFileType;
			pc->FileWriteOption(delayedFileWrite);
			
			if (plugin.second.empty())
			{
				throw runtime_error(ClassName() + ": plugin definition is empty");
			}

			BOOST_FOREACH(boost::property_tree::ptree::value_type& kv, plugin.second)
			{
				string key = kv.first;
				string value;

				try
				{
					value = kv.second.get<string> ("");
				}
				catch (...)
				{
					continue;
				}

				if (key == "name")
				{
					pc->Name(value);
				}
				else
				{
					if (value.empty())
					{
						BOOST_FOREACH(boost::property_tree::ptree::value_type& listval, kv.second)
						{

							//pc->AddOption(key, value);
							pc->AddOption(key, listval.second.get<string> (""));
						}
					}
					else
					{
						pc->AddOption(key, value);
					}
				}
			}

			if (pc->Name().empty())
			{
				throw runtime_error(ClassName() + ": plugin name not found from configuration");
			}

			pc->Info(make_shared<info> (*anInfo));	// We have to have a copy for all configs.
													// Each plugin will later on create a data backend.
			pluginContainer.push_back(pc);
			
		}

	} // END BOOST_FOREACH

	return pluginContainer;

}

void json_parser::ParseTime(shared_ptr<configuration> conf,
								std::shared_ptr<info> anInfo,
								const boost::property_tree::ptree& pt)
{
	/* Check origin time */

	string originDateTime;
	string mask;

	const producer sourceProducer = conf->SourceProducer();
	
	try
	{
		originDateTime = pt.get<string>("origintime");
		mask = "%Y-%m-%d %H:%M:%S";

		boost::algorithm::to_lower(originDateTime);

		if (originDateTime == "latest")
		{
			HPDatabaseType dbtype = conf->DatabaseType();
			
			map<string,string> prod;
			
			if (dbtype == kNeons || dbtype == kNeonsAndRadon)
			{
				auto n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

				prod = n->NeonsDB().GetProducerDefinition(static_cast<unsigned long> (sourceProducer.Id()));

				if (!prod.empty())
				{
					originDateTime = n->NeonsDB().GetLatestTime(prod["ref_prod"]);
					
					if (!originDateTime.empty())
					{
						mask = "%Y%m%d%H%M";
						anInfo->OriginDateTime(originDateTime, mask);
					}
				}
					
			}
			
			if (prod.empty() && (dbtype == kRadon || dbtype == kNeonsAndRadon))
			{
				auto r = dynamic_pointer_cast<plugin::radon> (plugin_factory::Instance()->Plugin("radon"));

				prod = r->RadonDB().GetProducerDefinition(static_cast<unsigned long> (sourceProducer.Id()));

				if (!prod.empty())
				{
					originDateTime = r->RadonDB().GetLatestTime(prod["ref_prod"]);
					
					if (!originDateTime.empty())
					{
						mask = "%Y-%m-%d %H:%M:00";
						anInfo->OriginDateTime(originDateTime, mask);
					}
				}
				
			}
			
			if (prod.empty())
			{
				throw runtime_error("Producer definition not found for procucer id " + boost::lexical_cast<string> (sourceProducer.Id()));
			}
			else if (originDateTime.size() == 0)
			{
				throw runtime_error("Latest time not found from Neons for producer '" + prod["ref_prod"] + "'");
			}
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		if (anInfo->OriginDateTime().Empty())
		{
			throw runtime_error(ClassName() + ": origintime not found");
		}
	}
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing time information: ") + e.what());
	}

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
		vector<string> timesStr = util::Split(hours, ",", true);

		vector<int> times ;

		for (size_t i = 0; i < timesStr.size(); i++)
		{
			times.push_back(boost::lexical_cast<int> (timesStr[i]));
		}

		sort (times.begin(), times.end());

		vector<forecast_time> theTimes;

		for (size_t i = 0; i < times.size(); i++)
		{

			// Create forecast_time with both times origintime, then adjust the validtime

			forecast_time theTime (originDateTime, originDateTime, mask);
			
			theTime.ValidDateTime().Adjust(kHourResolution, times[i]);

			theTimes.push_back(theTime);
		}

		anInfo->Times(theTimes);

		return;

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing time information: ") + e.what());
	}

	// hours was not specified
	// check if start/stop times are

	// First check step_unit which is deprecated and issue warning
	try
	{
		string stepUnit = pt.get<string>("step_unit");

		if (!stepUnit.empty())
		{
			itsLogger->Warning("Key 'step_unit' is deprecated");
		}
	}
	catch (exception& e)
	{}


	try
	{
		int start = pt.get<int>("start_hour");
		int stop = pt.get<int>("stop_hour");
		int step = pt.get<int>("step");

		conf->itsForecastStep = step;
		
		HPTimeResolution stepResolution = kHourResolution;

		if (stop > 1<<8)
		{
			anInfo->StepSizeOverOneByte(true);
		}

		int curtime = start;

		vector<forecast_time> theTimes;

		do
		{

			forecast_time theTime (originDateTime, originDateTime, mask);

			theTime.ValidDateTime().Adjust(stepResolution, curtime);

			theTime.StepResolution(stepResolution);

			theTimes.push_back(theTime);

			curtime += step;

		} while (curtime <= stop);

		anInfo->Times(theTimes);

		return;

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
	}
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing time information: ") + e.what());
	}
	
	try
	{
		// try start_minute/stop_minute

		int start = pt.get<int>("start_minute");
		int stop = pt.get<int>("stop_minute");
		int step = pt.get<int>("step");

		conf->itsForecastStep = step;

		HPTimeResolution stepResolution = kMinuteResolution;

		if (stop > 1<<8)
		{
			anInfo->StepSizeOverOneByte(true);
		}

		int curtime = start;

		vector<forecast_time> theTimes;

		do
		{

			forecast_time theTime (originDateTime, originDateTime, mask);

			theTime.ValidDateTime().Adjust(stepResolution, curtime);

			theTime.StepResolution(stepResolution);

			theTimes.push_back(theTime);

			curtime += step;

		} while (curtime <= stop);

		anInfo->Times(theTimes);
	}	
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing time information: ") + e.what());
	}

}

unique_ptr<regular_grid> ParseAreaAndGridFromDatabase(configuration& conf, const boost::property_tree::ptree& pt)
{
	unique_ptr<regular_grid> g;
		
	try
	{
		string geom = pt.get<string>("target_geom_name");

		g = unique_ptr<regular_grid> (new regular_grid);
		
		conf.TargetGeomName(geom);

		HPDatabaseType dbtype = conf.DatabaseType();
		
		map<string, string> geominfo;
		
		if (dbtype == kNeons || dbtype == kNeonsAndRadon)
		{
			auto n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

			geominfo = n->NeonsDB().GetGeometryDefinition(geom);
		}
		
		if (geominfo.empty() && (dbtype == kRadon || dbtype == kNeonsAndRadon))
		{
			auto r = dynamic_pointer_cast<plugin::radon> (plugin_factory::Instance()->Plugin("radon"));

			geominfo = r->RadonDB().GetGeometryDefinition(geom);
		}

		if (geominfo.empty())
		{
			throw runtime_error("Fatal::json_parser Unknown geometry '" + geom + "' found");	
		}
			
		//regular_grid* _g = dynamic_cast<regular_grid*> (g.get()); // shortcut to avoid a million dynamic casts

		/*
		 *  In Neons we don't have rotlatlon projection used separately, instead we have
		 *  to check if geom_parm_1 and geom_parm_2 specify the regular rotated location
		 *  if south pole (0,30).
		 */

		double di = boost::lexical_cast<double>(geominfo["pas_longitude"]);
		double dj = boost::lexical_cast<double>(geominfo["pas_latitude"]);

		if (
				(geominfo["prjn_name"] == "latlon" && (geominfo["geom_parm_1"] != "0" || geominfo["geom_parm_2"] != "0")) // neons
				||
				(geominfo["prjn_id"] == "4")) // radon
		{
			g->Projection(kRotatedLatLonProjection);
			g->SouthPole(point(boost::lexical_cast<double>(geominfo["geom_parm_2"]) / 1e3, boost::lexical_cast<double>(geominfo["geom_parm_1"]) / 1e3));
			di /= 1e3;
			dj /= 1e3;
		}
		else if (geominfo["prjn_name"] == "latlon" || geominfo["prjn_id"] == "1")
		{
			g->Projection(kLatLonProjection);
			di /= 1e3;
			dj /= 1e3;
		}
		else if (geominfo["prjn_name"] == "polster" || geominfo["prjn_name"] == "polarstereo" || geominfo["prjn_id"] == "2")
		{
			g->Projection(kStereographicProjection);
			g->Orientation(boost::lexical_cast<double>(geominfo["geom_parm_1"]) / 1e3);
			g->Di(di);
			g->Dj(dj);
		}
		else
		{
			throw runtime_error("Fatal::json_parser Unknown projection: " + geominfo["prjn_name"]);
		}

		g->Ni(boost::lexical_cast<size_t> (geominfo["col_cnt"]));
		g->Nj(boost::lexical_cast<size_t> (geominfo["row_cnt"]));

		if (geominfo["stor_desc"] == "+x-y")
		{
			g->ScanningMode(kTopLeft);
		}
		else if (geominfo["stor_desc"] == "+x+y")
		{
			g->ScanningMode(kBottomLeft);
		}
		else
		{
			throw runtime_error("Fatal::json_parser scanning mode " + geominfo["stor_desc"] + " not supported yet");
		}

		double X0 = boost::lexical_cast<double>(geominfo["long_orig"]) / 1e3;
		double Y0 = boost::lexical_cast<double>(geominfo["lat_orig"]) / 1e3;

		std::pair<point, point> coordinates;

		if (g->Projection() == kStereographicProjection)
		{
			assert(g->ScanningMode() == kBottomLeft);
			coordinates = util::CoordinatesFromFirstGridPoint(point(X0, Y0), g->Orientation(), g->Ni(), g->Nj(), di, dj);
		}
		else
		{
			coordinates = util::CoordinatesFromFirstGridPoint(point(X0, Y0), g->Ni(), g->Nj(), di, dj, g->ScanningMode());
		}

		g->BottomLeft(coordinates.first);
		g->TopRight(coordinates.second);

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

unique_ptr<irregular_grid> ParseAreaAndGridFromPoints(configuration& conf, const boost::property_tree::ptree& pt)
{
	unique_ptr<irregular_grid> g;
	
	// check points
	
	try
	{
		vector<string> stations = util::Split(pt.get<string>("points"), ",", false);

		g = unique_ptr<irregular_grid> (new irregular_grid());
	
		// hard coded projection to latlon
		
		g->Projection(kLatLonProjection);

		vector<station> theStations;
		
		int i = 1;
		
		BOOST_FOREACH(const string& line, stations)
		{
			vector<string> point = util::Split(line, "/", false);
			
			if (point.size() != 2)
			{
				cout << "Error::json_parser Line " + line + " is invalid" << endl;
				continue;
			}
			
			theStations.push_back(
				station(i, 
					"point_" + boost::lexical_cast<string> (i), 
					boost::lexical_cast<double>(point[0]), 
					boost::lexical_cast<double>(point[1]))
			);
			
			i++;
		}
		
		g->Stations(theStations);

		return g;
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
		vector<string> stations = util::Split(pt.get<string>("stations"), ";", false);

		g = unique_ptr<irregular_grid> (new irregular_grid());
	
		// hard coded projection to latlon
		
		g->Projection(kLatLonProjection);

		vector<station> theStations;
		
		auto r = dynamic_pointer_cast<plugin::radon> (plugin_factory::Instance()->Plugin("radon"));
		
		BOOST_FOREACH(const string& str, stations)
		{		
			throw runtime_error("Not ready yet");
			/*unsigned long fmisid;
			
			try
			{
				fmisid = boost::lexical_cast<unsigned long> (str);
			}
			catch (boost::bad_lexical_cast& e)
			{
				cout << "Error::json_parser Invalid fmisid: " << str << endl;
				continue;
			}
			
			auto stationinfo = r->RadonDB().GetStationInfoFromFmiSID(fmisid);
			
			if (stationinfo.empty())
			{
				cout << "Error::json_parser Station " << str << " not found from database" << endl;
				continue;
			}
			theStations.push_back(
				station(fmisid,
					stationinfo["station_name"],
					boost::lexical_cast<double>(stationinfo["longitude"]),
					boost::lexical_cast<double>(stationinfo["latitude"])
				)
			);*/
		}
		
		g->Stations(theStations);

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (const exception& e)
	{
		throw runtime_error(string("Fatal::json_parser Error parsing stations: ") + e.what());
	}
	
	if (g->Stations().empty())
	{
		throw runtime_error("Fatal::json_parser No valid points or stations found");
	}
	
	return g;
}

unique_ptr<grid> json_parser::ParseAreaAndGrid(shared_ptr<configuration> conf, const boost::property_tree::ptree& pt)
{

	unique_ptr<grid> g;
	
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

	bool haveArea = false;
	
	// 1. Check for source geom name

	try
	{
		conf->itsSourceGeomNames = util::Split(pt.get<string>("source_geom_name"), ",", false);
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
	
	g = ParseAreaAndGridFromDatabase(*conf, pt);
	
	if (g)
	{
		return g;
	}

	// 3. Points
	
	g = ParseAreaAndGridFromPoints(*conf, pt);
	
	if (g)
	{
		return g;
	}
	
	// 4. Target geometry is still not set, check for bbox

	// From this point forward we only support regular grids
	
	g = unique_ptr<regular_grid> (new regular_grid());

	try
	{

		vector<string> coordinates = util::Split(pt.get<string>("bbox"), ",", false);

		// hard coded projection to latlon
		
		g->Projection(kLatLonProjection);

		g->BottomLeft(point(boost::lexical_cast<double>(coordinates[0]), boost::lexical_cast<double>(coordinates[1])));
		g->TopRight(point(boost::lexical_cast<double>(coordinates[2]), boost::lexical_cast<double>(coordinates[3])));

		haveArea = true;
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing bbox: ") + e.what());
	}

	// Check for manual definition of area
	
	if (!haveArea)
	{

		try
		{
			string projection = pt.get<string>("projection");

			if (projection == "latlon")
			{
				g->Projection(kLatLonProjection);
			}
			else if (projection == "rotated_latlon")
			{
				g->Projection(kRotatedLatLonProjection);
			}
			else if (projection == "stereographic")
			{
				g->Projection(kStereographicProjection);
			}
			else
			{
				throw runtime_error(ClassName() + ": Unknown projection: " + projection);
			}

		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			throw runtime_error(string("Projection definition not found: ") + e.what());
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing projection: ") + e.what());
		}

		try
		{
			g->BottomLeft(point(pt.get<double>("bottom_left_longitude"), pt.get<double>("bottom_left_latitude")));
			g->TopRight(point(pt.get<double>("top_right_longitude"), pt.get<double>("top_right_latitude")));
		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			throw runtime_error(string("Area corner definitions not found: ") + e.what());
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing area corners: ") + e.what());
		}

		/* Check orientation */

		try
		{
			g->Orientation(pt.get<double>("orientation"));
		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			if (g->Projection() == kStereographicProjection)
			{
				throw runtime_error(string("Orientation not found for stereographic projection: ") + e.what());
			}
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing area orientation: ") + e.what());
		}

		/* Check south pole coordinates */

		try
		{
			g->SouthPole(point(pt.get<double>("south_pole_longitude"), pt.get<double>("south_pole_latitude")));
		}
		catch (boost::property_tree::ptree_bad_path& e)
		{
			if (g->Projection() == kRotatedLatLonProjection)
			{
				throw runtime_error(string("South pole coordinates not found for rotated latlon projection: ") + e.what());
			}
		}
		catch (exception& e)
		{
			throw runtime_error(string("Error parsing south pole location: ") + e.what());
		}
	}

	/* Finally check grid definitions */

	regular_grid* _g = dynamic_cast<regular_grid*> (g.get()); // shortcut to avoid a million dynamic casts

	try
	{
		_g->Ni(pt.get<size_t>("ni"));
		_g->Nj(pt.get<size_t>("nj"));

		
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		throw runtime_error(string("Grid definitions not found: ") + e.what());

	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing grid dimensions: ") + e.what());
	}

	// Default scanningmode to +x+y
	
	_g->ScanningMode(kBottomLeft);
	
	try
	{
		string mode = pt.get<string> ("scanning_mode");

		if (mode == "+x-y")
		{
			_g->ScanningMode(kTopLeft);
		}
		else if (mode == "+x+y")
		{
			_g->ScanningMode(kBottomLeft);
		}
		else
		{
			throw runtime_error(ClassName() + ": scanning mode " + mode + " not supported (yet)");
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing scanning mode: ") + e.what());
	}
	
	return g;

}

void json_parser::ParseProducers(shared_ptr<configuration> conf, shared_ptr<info> anInfo, const boost::property_tree::ptree& pt)
{
	try
	{

		std::vector<producer> sourceProducers;
		vector<string> sourceProducersStr = util::Split(pt.get<string>("source_producer"), ",", false);

		HPDatabaseType dbtype = conf->DatabaseType();
		
		if (dbtype == kNeons || dbtype == kNeonsAndRadon)
		{
			auto n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

			for (size_t i = 0; i < sourceProducersStr.size(); i++)
			{
				long pid = boost::lexical_cast<long> (sourceProducersStr[i]);

				producer prod(pid);

				map<string,string> prodInfo = n->NeonsDB().GetGridModelDefinition(static_cast<unsigned long> (pid));

				if (!prodInfo.empty())
				{
					prod.Centre(boost::lexical_cast<long> (prodInfo["ident_id"]));
					prod.Name(prodInfo["ref_prod"]);
					prod.TableVersion(boost::lexical_cast<long> (prodInfo["no_vers"]));
					prod.Process(boost::lexical_cast<long> (prodInfo["model_id"]));
				}
				else
				{
					itsLogger->Warning("Unknown source producer: " + sourceProducersStr[i]);
				}

				sourceProducers.push_back(prod);
			}
		}
		
		if (sourceProducers.size() == 0 && (dbtype == kRadon || dbtype == kNeonsAndRadon))
		{
			auto r = dynamic_pointer_cast<plugin::radon> (plugin_factory::Instance()->Plugin("radon"));

			for (size_t i = 0; i < sourceProducersStr.size(); i++)
			{
				long pid = boost::lexical_cast<long> (sourceProducersStr[i]);

				producer prod(pid);

				map<string,string> prodInfo = r->RadonDB().GetProducerDefinition(static_cast<unsigned long> (pid));

				if (!prodInfo.empty())
				{
					prod.Centre(boost::lexical_cast<long> (prodInfo["ident_id"]));
					prod.Name(prodInfo["ref_prod"]);
					prod.Process(boost::lexical_cast<long> (prodInfo["model_id"]));
				}
				else
				{
					itsLogger->Warning("Unknown source producer: " + sourceProducersStr[i]);
				}

				sourceProducers.push_back(prod);
			}
		}
		
		
		if (sourceProducers.size() == 0)
		{
			throw runtime_error(ClassName() + ": Source producer information was not found from database");
		}
		
		conf->SourceProducers(sourceProducers);

		/*
		 * Target producer is also set to target info; source infos (and producers) are created
		 * as data is fetched from files.
		 */

		long pid = boost::lexical_cast<long> (pt.get<string>("target_producer"));
		producer prod (pid);
		
		map<string,string> prodInfo;
		
		if (dbtype == kNeons || dbtype == kNeonsAndRadon)
		{
			auto n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));
			prodInfo = n->NeonsDB().GetGridModelDefinition(static_cast<unsigned long> (pid));
		}
		
		if (prodInfo.empty() && (dbtype == kRadon || dbtype == kNeonsAndRadon))
		{
			auto r = dynamic_pointer_cast<plugin::radon> (plugin_factory::Instance()->Plugin("radon"));
			prodInfo = r->RadonDB().GetProducerDefinition(static_cast<unsigned long> (pid));
		}
		
		if (!prodInfo.empty())
		{
			prod.Centre(boost::lexical_cast<long> (prodInfo["ident_id"]));
			prod.Name(prodInfo["ref_prod"]);
			//prod.TableVersion(boost::lexical_cast<long> (prodInfo["no_vers"]));
			prod.Process(boost::lexical_cast<long> (prodInfo["model_id"]));
		}
		else
		{
			itsLogger->Warning("Unknown target producer: " + pt.get<string>("target_producer"));
		}

		anInfo->itsProducer = prod;
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		throw runtime_error(ClassName() + string(": Producer definitions not found: ") + e.what());
	}
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing producer information: ") + e.what());
	}

}

void json_parser::ParseLevels(shared_ptr<info> anInfo, const boost::property_tree::ptree& pt)
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

vector<level> json_parser::LevelsFromString(const string& levelType, const string& levelValues) const
{
	HPLevelType theLevelType = HPStringToLevelType.at(boost::to_lower_copy(levelType));

	vector<string> levelsStr = util::Split(levelValues, ",", true);

	vector<level> levels;

	for (size_t i = 0; i < levelsStr.size(); i++)
	{
		levels.push_back(level(theLevelType, boost::lexical_cast<float> (levelsStr[i]), levelType));
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

bool json_parser::ParseBoolean(string& booleanValue)
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
		throw runtime_error(ClassName() + ": Invalid boolean value: " + booleanValue);
	}

	return ret;
}
