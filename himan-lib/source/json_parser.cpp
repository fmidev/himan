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

#define HIMAN_AUXILIARY_INCLUDE

#include "neons.h"

#undef HIMAN_AUXILIARY_INCLUDE

using namespace himan;
using namespace std;

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

json_parser* json_parser::itsInstance = NULL;

json_parser* json_parser::Instance()
{

	if (!itsInstance)
	{
		itsInstance = new json_parser;
	}

	return itsInstance;
}

json_parser::json_parser()
{
	itsLogger = std::unique_ptr<logger> (logger_factory::Instance()->GetLog("json_parser"));
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

	itsLogger->Debug("Parsing configuration file");

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

	shared_ptr<info> baseInfo(new info());

	/* Check producers */

	ParseProducers(conf, baseInfo, pt);

	/* Check area definitions */

	ParseAreaAndGrid(conf, baseInfo, pt);

	/* Check time definitions */

	ParseTime(conf->SourceProducers()[0], baseInfo, pt);

	/* Check levels */

	//ParseLevels(baseInfo, pt);

	/* Check whole_file_write */

	try
	{

		string theFileWriteOption = pt.get<string>("file_write");

		if (theFileWriteOption == "neons")
		{
			conf->FileWriteOption(kNeons);
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
	/* Check processqueue */

	std::vector<std::shared_ptr<info> > infoQueue;

	boost::property_tree::ptree& pq = pt.get_child("processqueue");

	BOOST_FOREACH(boost::property_tree::ptree::value_type &element, pq)
	{
		std::shared_ptr<info> anInfo (new info(*baseInfo));
		anInfo->Create(); // Reset data backend

		try
		{
			ParseAreaAndGrid(conf, anInfo, element.second);
		}
		catch (...)
		{
			// do nothing
		}

		try
		{
			ParseLevels(anInfo, element.second);
		}
		catch (...)
		{
			throw runtime_error(ClassName() + ": Unable to proceed");
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
			throw runtime_error(string("Error parsing meta information: ") + e.what());
		}

		boost::property_tree::ptree& plugins = element.second.get_child("plugins");

		if (plugins.empty())
		{
			throw runtime_error(ClassName() + ": plugin definitions not found");
		}

		BOOST_FOREACH(boost::property_tree::ptree::value_type &plugin, plugins)
		{
			shared_ptr<plugin_configuration> pc(new plugin_configuration(conf));

			pc->UseCache(delayedUseCache);
			
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
					pc->AddOption(key, value);
				}

				if (pc->Name().empty())
				{
					throw runtime_error(ClassName() + ": plugin name not found from configuration");
				}

			}

			pc->Info(anInfo);
			pluginContainer.push_back(pc);
			
		}

	} // END BOOST_FOREACH

	return pluginContainer;

}

void json_parser::ParseTime(const producer& sourceProducer,
								std::shared_ptr<info> anInfo,
								const boost::property_tree::ptree& pt)
{
	/* Check origin time */

	string originDateTime;
	string mask;

	try
	{
		originDateTime = pt.get<string>("origintime");
		mask = "%Y-%m-%d %H:%M:%S";

		boost::algorithm::to_lower(originDateTime);

		if (originDateTime == "latest")
		{
			shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

			map<string,string> prod = n->NeonsDB().GetProducerDefinition(sourceProducer.Id());

			if (prod.empty())
			{
				throw runtime_error("Producer definition not found for procucer id " + boost::lexical_cast<string> (sourceProducer.Id()));
			}

			originDateTime = n->NeonsDB().GetLatestTime(prod["ref_prod"]);

			if (originDateTime.size() == 0)
			{
				throw runtime_error("Latest time not found from Neons for producer '" + prod["ref_prod"] + "'");
			}

			mask = "%Y%m%d%H%M";
		}

		anInfo->OriginDateTime(originDateTime, mask);

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

			forecast_time theTime (shared_ptr<raw_time> (new raw_time(originDateTime, mask)),
								   shared_ptr<raw_time> (new raw_time(originDateTime, mask)));

			theTime.ValidDateTime()->Adjust(kHourResolution, times[i]);

			theTimes.push_back(theTime);
		}

		anInfo->Times(theTimes);

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// hours was not specified
		// check if start/stop times are

		int start = boost::lexical_cast<int> (pt.get<string>("start_hour"));
		int stop = boost::lexical_cast<int> (pt.get<string>("stop_hour"));
		int step = boost::lexical_cast<int> (pt.get<string>("step"));

		string unit = pt.get<string>("step_unit");

		if (unit != "hour" && unit != "minute")
		{
			throw runtime_error("Step unit '" + unit + "' not supported");
		}

		HPTimeResolution stepResolution = kHourResolution;

		if (unit == "minute")
		{
			start *= 60;
			stop *= 60;
			stepResolution = kMinuteResolution;
		}

		if (stop > 1<<8)
		{
			anInfo->StepSizeOverOneByte(true);
		}

		int curtime = start;

		vector<forecast_time> theTimes;

		do
		{

			forecast_time theTime (shared_ptr<raw_time> (new raw_time(originDateTime, mask)),
								   shared_ptr<raw_time> (new raw_time(originDateTime, mask)));

			theTime.ValidDateTime()->Adjust(stepResolution, curtime);

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

void json_parser::ParseAreaAndGrid(shared_ptr<configuration> conf, std::shared_ptr<info> anInfo, const boost::property_tree::ptree& pt)
{

	/* First check for neons style geom name */

	try
	{
		string geom = pt.get<string>("target_geom_name");

		conf->itsTargetGeomName = geom;

		shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

		map<string, string> geominfo = n->NeonsDB().GetGeometryDefinition(geom);

		if (!geominfo.empty())
		{
			/*
			 *  In Neons we don't have rotlatlon projection used separately, instead we have
			 *  to check if geom_parm_1 and geom_parm_2 specify the regular rotated location
			 *  if south pole (0,30).
			 */

			double di = boost::lexical_cast<double>(geominfo["pas_longitude"]);
			double dj = boost::lexical_cast<double>(geominfo["pas_latitude"]);

			if (geominfo["prjn_name"] == "latlon" && (geominfo["geom_parm_1"] != "0" || geominfo["geom_parm_2"] != "0"))
			{
				anInfo->itsProjection = kRotatedLatLonProjection;
				anInfo->itsSouthPole = point(boost::lexical_cast<double>(geominfo["geom_parm_2"]) / 1e3, boost::lexical_cast<double>(geominfo["geom_parm_1"]) / 1e3);
				di /= 1e3;
				dj /= 1e3;
			}
			else if (geominfo["prjn_name"] == "latlon")
			{
				anInfo->itsProjection = kLatLonProjection;
				di /= 1e3;
				dj /= 1e3;
			}
			else if (geominfo["prjn_name"] == "polster" || geominfo["prjn_name"] == "polarstereo")
			{
				anInfo->itsProjection = kStereographicProjection;
				anInfo->itsOrientation = boost::lexical_cast<double>(geominfo["geom_parm_1"]) / 1e3;
				anInfo->itsDi = di;
				anInfo->itsDj = dj;
			}
			else
			{
				throw runtime_error(ClassName() + ": Unknown projection: " + geominfo["prjn_name"]);
			}

			anInfo->itsNi = boost::lexical_cast<size_t> (geominfo["col_cnt"]);
			anInfo->itsNj = boost::lexical_cast<size_t> (geominfo["row_cnt"]);

			if (geominfo["stor_desc"] == "+x-y")
			{
				anInfo->itsScanningMode = kTopLeft;
			}
			else if (geominfo["stor_desc"] == "+x+y")
			{
				anInfo->itsScanningMode = kBottomLeft;
			}
			else
			{
				throw runtime_error(ClassName() + ": scanning mode " + geominfo["stor_desc"] + " not supported yet");
			}

			double X0 = boost::lexical_cast<double>(geominfo["long_orig"]) / 1e3;
			double Y0 = boost::lexical_cast<double>(geominfo["lat_orig"]) / 1e3;

			std::pair<point, point> coordinates;

			if (anInfo->itsProjection == kStereographicProjection)
			{
				assert(anInfo->itsScanningMode == kBottomLeft);
				coordinates = util::CoordinatesFromFirstGridPoint(point(X0, Y0), anInfo->itsOrientation, anInfo->itsNi, anInfo->itsNj, di, dj);
			}
			else
			{
				coordinates = util::CoordinatesFromFirstGridPoint(point(X0, Y0), anInfo->itsNi, anInfo->itsNj, di, dj, anInfo->itsScanningMode);
			}

			anInfo->itsBottomLeft = coordinates.first;
			anInfo->itsTopRight = coordinates.second;

		}
		else
		{
			throw runtime_error(ClassName() + ": Unknown geometry '" + geom + "' found");
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing area information: ") + e.what());
	}

	// Check for source geom name

	try
	{
		string geom = pt.get<string>("source_geom_name");

		conf->itsSourceGeomName = geom;
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing area information: ") + e.what());
	}

	if (!conf->itsTargetGeomName.empty())
	{
		return;
	}
	
	// Check for manual definition of area

	try
	{
		string projection = pt.get<string>("projection");

		if (projection == "latlon")
		{
			anInfo->itsProjection = kLatLonProjection;
		}
		else if (projection == "rotated_latlon")
		{
			anInfo->itsProjection = kRotatedLatLonProjection;
		}
		else if (projection == "stereographic")
		{
			anInfo->itsProjection = kStereographicProjection;
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
		anInfo->itsBottomLeft = point(pt.get<double>("bottom_left_longitude"), pt.get<double>("bottom_left_latitude"));
		anInfo->itsTopRight = point(pt.get<double>("top_right_longitude"), pt.get<double>("top_right_latitude"));
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
		anInfo->itsOrientation = pt.get<double>("orientation");
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		if (anInfo->itsProjection == kStereographicProjection)
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
		anInfo->itsSouthPole = point(pt.get<double>("south_pole_longitude"), pt.get<double>("south_pole_latitude"));
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		if (anInfo->itsProjection == kRotatedLatLonProjection)
		{
			throw runtime_error(string("South pole coordinates not found for rotated latlon projection: ") + e.what());
		}
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing south pole location: ") + e.what());
	}

	/* Check grid definitions */

	try
	{
		anInfo->itsNi = pt.get<size_t>("ni");
		anInfo->itsNj = pt.get<size_t>("nj");

		string mode = pt.get<string> ("scanning_mode");

		if (mode == "+x-y")
		{
			anInfo->itsScanningMode = kTopLeft;
		}
		else if (mode == "+x+y")
		{
			anInfo->itsScanningMode = kBottomLeft;
		}
		else
		{
			throw runtime_error(ClassName() + ": scanning mode " + mode + " not supported (yet)");
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		throw runtime_error(string("Grid definitions not found: ") + e.what());

	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing grid dimensions: ") + e.what());
	}

}

void json_parser::ParseProducers(shared_ptr<configuration> conf, shared_ptr<info> anInfo, const boost::property_tree::ptree& pt)
{
	try
	{

		std::vector<producer> sourceProducers;
		vector<string> sourceProducersStr = util::Split(pt.get<string>("source_producer"), ",", false);

		shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

		for (size_t i = 0; i < sourceProducersStr.size(); i++)
		{
			long pid = boost::lexical_cast<long> (sourceProducersStr[i]);

			producer prod(pid);

			map<string,string> prodInfo = n->NeonsDB().GetGridModelDefinition(pid);

			if (!prodInfo.empty())
			{
				prod.Centre(boost::lexical_cast<long> (prodInfo["ident_id"]));
				prod.Name(prodInfo["ref_prod"]);
				prod.TableVersion(boost::lexical_cast<long> (prodInfo["no_vers"]));
				prod.Process(boost::lexical_cast<long> (prodInfo["model_id"]));
			}

			sourceProducers.push_back(prod);
		}

		conf->SourceProducers(sourceProducers);

		/*
		 * Target producer is also set to target info; source infos (and producers) are created
		 * as data is fetched from files.
		 */

		long pid = boost::lexical_cast<long> (pt.get<string>("target_producer"));

		map<string,string> prodInfo = n->NeonsDB().GetGridModelDefinition(pid);

		producer prod (pid);

		if (!prodInfo.empty())
		{
                        prod.Centre(boost::lexical_cast<long> (prodInfo["ident_id"]));
                        prod.Name(prodInfo["ref_prod"]);
                        prod.TableVersion(boost::lexical_cast<long> (prodInfo["no_vers"]));
                        prod.Process(boost::lexical_cast<long> (prodInfo["model_id"]));
		}
		// conf->TargetProducer(prod);

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

		string levelTypeStr = pt.get<std::string>("leveltype");
		string levelValuesStr = pt.get<std::string>("levels");

		vector<level> levels = LevelsFromString(levelTypeStr, levelValuesStr);

		anInfo->Levels(levels);

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		throw runtime_error(string("Level definition not found: ") + e.what());
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing level information: ") + e.what());
	}
}

vector<level> json_parser::LevelsFromString(const string& levelType, const string& levelValues) const
{
	string levelTypeUpper = levelType;
	boost::to_upper(levelTypeUpper);

	HPLevelType theLevelType;

	if (levelTypeUpper == "HEIGHT")
	{
		theLevelType = kHeight;
	}
	else if (levelTypeUpper == "PRESSURE")
	{
		theLevelType = kPressure;
	}
	else if (levelTypeUpper == "HYBRID")
	{
		theLevelType = kHybrid;
	}
	else if (levelTypeUpper == "GROUND")
	{
		theLevelType = kGround;
	}
	else if (levelTypeUpper == "MEANSEA")
	{
		theLevelType = kMeanSea;
	}
	else
	{
		throw runtime_error("Unknown level type: " + levelType);
	}

	// can cause exception, what will happen then ?

	vector<string> levelsStr = util::Split(levelValues, ",", true);

	vector<level> levels;

	for (size_t i = 0; i < levelsStr.size(); i++)
	{
		levels.push_back(level(theLevelType, boost::lexical_cast<float> (levelsStr[i]), levelType));
	}

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
