/*
 * ini_parser.cpp
 *
 *  Created on: Nov 19, 2012
 *	  Author: partio
 */

#include "json_parser.h"
#include <boost/program_options.hpp>
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
	itsConfiguration = shared_ptr<configuration> (new configuration());
}

shared_ptr<configuration> json_parser::Parse(int argc, char** argv)
{

	ParseAndCreateInfo(argc, argv);

	return itsConfiguration;

}


void json_parser::ParseAndCreateInfo(int argc, char** argv)
{

	ParseCommandLine(argc, argv);

	// Check that we have configuration file defined in command line

	if (itsConfiguration->itsConfigurationFile.empty())
	{
		throw runtime_error("Configuration file not defined");
	}

	ParseConfigurationFile(itsConfiguration->itsConfigurationFile);

	// Check requested plugins

	if (itsConfiguration->itsPlugins.size() == 0)
	{
		throw runtime_error("No requested plugins defined");
	}

}

void json_parser::ParseCommandLine(int argc, char* argv[])
{
	itsLogger->Debug("Parsing command line");

	namespace po = boost::program_options;

	po::options_description desc("Allowed options");

	// Can't use required() since it doesn't allow us to use --list-plugins

	string outfileType;

	himan::HPDebugState debugState = himan::kDebugMsg;

	int logLevel = 0;

	desc.add_options()
	("help,h", "print out help message")
	("type,t", po::value(&outfileType), "output file type, one of: grib, grib2, netcdf, querydata")
	("version,v", "display version number")
	("configuration-file,f", po::value(&(itsConfiguration->itsConfigurationFile)), "configuration file")
	("auxiliary-files,a", po::value<vector<string> > (&(itsConfiguration->itsAuxiliaryFiles)), "auxiliary (helper) file(s)")
	("threads,j", po::value(&itsConfiguration->itsThreadCount), "number of started threads")
	("list-plugins,l", "list all defined plugins")
	("dedbug-level,d", po::value(&logLevel), "set log level: 0(fatal) 1(error) 2(warning) 3(info) 4(debug) 5(trace)")
	("no-cuda", "disable cuda extensions")
	;

	po::positional_options_description p;
	p.add("auxiliary-files", -1);

	po::variables_map opt;
	po::store(po::command_line_parser(argc, argv)
			  .options(desc)
			  .positional(p)
			  .run(),
			  opt);

	po::notify(opt);

	if (logLevel)
	{
		switch (logLevel)
		{
		case 0:
			debugState = kFatalMsg;
			break;
		case 1:
			debugState = kErrorMsg;
			break;
		case 2:
			debugState = kWarningMsg;
			break;
		case 3:
			debugState = kInfoMsg;
			break;
		case 4:
			debugState = kDebugMsg;
			break;
		case 5:
			debugState = kTraceMsg;
			break;
		}

	}

	logger_factory::Instance()->DebugState(debugState);

	if (opt.count("version"))
	{
		cout << "himan-lib version ???" << endl;
		exit(1);
	}

	if (opt.count("no-cuda"))
	{
		itsConfiguration->itsUseCuda = false;
	}

	if (!outfileType.empty())
	{
		if (outfileType == "grib")
		{
			itsConfiguration->itsOutputFileType = kGRIB1;
		}
		else if (outfileType == "grib2")
		{
			itsConfiguration->itsOutputFileType = kGRIB2;
		}
		else if (outfileType == "netcdf")
		{
			itsConfiguration->itsOutputFileType = kNetCDF;
		}
		else if (outfileType == "querydata")
		{
			itsConfiguration->itsOutputFileType = kQueryData;
		}
		else
		{
			throw runtime_error("Invalid file type: " + outfileType);
		}
	}

	if (opt.count("help"))
	{
		cout << "usage: himan [ options ]" << endl;
		cout << desc;
		cout << endl << "Examples:" << endl;
		cout << "  himan -f etc/tpot.json" << endl;
		cout << "  himan -f etc/vvmms.json -a file.grib -t querydata" << endl << endl;
		exit(1);
	}

	if (opt.count("list-plugins"))
	{

		vector<shared_ptr<plugin::himan_plugin> > thePlugins = plugin_factory::Instance()->CompiledPlugins();

		cout << "Compiled plugins" << endl;

		for (size_t i = 0; i < thePlugins.size(); i++)
		{
			cout << "\t" << thePlugins[i]->ClassName() << "\t(version " << thePlugins[i]->Version() << ")" << endl;
		}

		thePlugins = plugin_factory::Instance()->InterpretedPlugins();

		cout << "Interpreted plugins" << endl;

		for (size_t i = 0; i < thePlugins.size(); i++)
		{
			cout << "\t" << thePlugins[i]->ClassName() << "\t(version " << thePlugins[i]->Version() << ")" << endl;
		}

		cout << "Auxiliary plugins" << endl;

		thePlugins = plugin_factory::Instance()->AuxiliaryPlugins();

		for (size_t i = 0; i < thePlugins.size(); i++)
		{
			cout << "\t" << thePlugins[i]->ClassName() << "\t(version " << thePlugins[i]->Version() << ")" << endl;
		}

		exit(1);
	}

}


void json_parser::ParseConfigurationFile(const string& theConfigurationFile)
{

	itsLogger->Debug("Parsing configuration file");

	boost::property_tree::ptree pt;

	try
	{
		boost::property_tree::json_parser::read_json(theConfigurationFile, pt);
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error reading configuration file: ") + e.what());
	}

	/* Check area definitions */

	ParseAreaAndGrid(pt);

	/* Check plugins */

	try
	{
		BOOST_FOREACH(boost::property_tree::ptree::value_type &node, pt.get_child("processqueue"))
                {
                        itsConfiguration->Plugins(util::Split(node.second.get<std::string>("plugins"), ",", true));
                }
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing plugins: ") + e.what());
	}

	/* Check producer */

	try
	{

		itsConfiguration->SourceProducer(boost::lexical_cast<unsigned int> (pt.get<string>("source_producer")));
		itsConfiguration->TargetProducer(boost::lexical_cast<unsigned int> (pt.get<string>("target_producer")));

		/*
		 * Target producer is also set to target info; source infos (and producers) are created
		 * as data is fetched from files.
		 */

		itsConfiguration->Info()->Producer(itsConfiguration->TargetProducer());

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing producer information: ") + e.what());
	}

	ParseTime(pt);

	/* Check level */

	try
	{

		BOOST_FOREACH(boost::property_tree::ptree::value_type &node, pt.get_child("processqueue"))
                {
        	string theLevelTypeStr = node.second.get<std::string>("leveltype");
  	
		boost::to_upper(theLevelTypeStr);

		HPLevelType theLevelType;

		if (theLevelTypeStr == "HEIGHT")
		{
			theLevelType = kHeight;
		}
		else if (theLevelTypeStr == "PRESSURE")
		{
			theLevelType = kPressure;
		}
		else if (theLevelTypeStr == "HYBRID")
		{
			theLevelType = kHybrid;
		}
		else if (theLevelTypeStr == "GROUND")
		{
			theLevelType = kGround;
		}
		else if (theLevelTypeStr == "MEANSEA")
		{
			theLevelType = kMeanSea;
		}

		else
		{
			throw runtime_error("Unknown level type: " + theLevelTypeStr);	// not good practice; constructing string
		}

		// can cause exception, what will happen then ?

		vector<string> levelsStr = util::Split(node.second.get<std::string>("levels"), ",", true);

		vector<float> levels ;

		for (size_t i = 0; i < levelsStr.size(); i++)
		{
			levels.push_back(boost::lexical_cast<float> (levelsStr[i]));
		}

		sort (levels.begin(), levels.end());

		vector<level> theLevels;

		for (size_t i = 0; i < levels.size(); i++)
		{
			theLevels.push_back(level(theLevelType, levels[i], theLevelTypeStr));
		}

		itsConfiguration->Info()->Levels(theLevels);

                } // END BOOST_FOREACH
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing level information: ") + e.what());
	}

	/* Check whole_file_write */

	try
	{

		string theWholeFileWrite = pt.get<string>("whole_file_write");

		if (ParseBoolean(theWholeFileWrite))
		{
			itsConfiguration->WholeFileWrite(true);
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
			itsConfiguration->ReadDataFromDatabase(false);
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

		itsConfiguration->itsFileWaitTimeout = boost::lexical_cast<unsigned short> (theFileWaitTimeout);

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

		itsConfiguration->itsLeadingDimension = theLeadingDimension;

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing meta information: ") + e.what());
	}
}

void json_parser::ParseTime(const boost::property_tree::ptree& pt)
{
	/* Check origin time */

	try
	{
		string theOriginDateTime = pt.get<string>("origintime");
		string mask = "%Y-%m-%d %H:%M:%S";

		boost::algorithm::to_lower(theOriginDateTime);

		if (theOriginDateTime == "latest")
		{
			shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

			producer p(itsConfiguration->SourceProducer());

			map<string,string> prod = n->NeonsDB().GetProducerDefinition(p.Id());

			p.Name(prod["ref_prod"]);

			theOriginDateTime = n->LatestTime(p);

			if (theOriginDateTime.size() == 0)
			{
				throw runtime_error("Unable to proceed");
			}

			mask = "%Y%m%d%H%M";
		}

		itsConfiguration->Info()->OriginDateTime(theOriginDateTime, mask);

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		throw runtime_error(ClassName() + ": origintime not found");
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

			forecast_time theTime (shared_ptr<raw_time> (new raw_time(itsConfiguration->Info()->OriginDateTime())),
								   shared_ptr<raw_time> (new raw_time(itsConfiguration->Info()->OriginDateTime())));

			theTime.ValidDateTime()->Adjust(kHour, times[i]);

			theTimes.push_back(theTime);
		}

		itsConfiguration->Info()->Times(theTimes);

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

		vector<forecast_time> theTimes;

		int curtime = start;
		int curstep = 0;

		HPTimeResolution stepResolution = kHour;

		if (unit == "minute")
		{
			stop *= 60;
			stepResolution = kMinute;
		}

		do
		{

			forecast_time theTime (shared_ptr<raw_time> (new raw_time(itsConfiguration->Info()->OriginDateTime())),
								   shared_ptr<raw_time> (new raw_time(itsConfiguration->Info()->OriginDateTime())));

			theTime.ValidDateTime()->Adjust(stepResolution, curstep);

			theTime.StepResolution(stepResolution);

			theTimes.push_back(theTime);

			curtime += step;
			curstep = curtime;

		} while (curtime <= stop);

		itsConfiguration->Info()->Times(theTimes);

	}
	catch (exception& e)
	{
		throw runtime_error(ClassName() + ": " + string("Error parsing time information: ") + e.what());
	}

}

void json_parser::ParseAreaAndGrid(const boost::property_tree::ptree& pt)
{

	/* First check for neons style geom name */

	try
	{
		string geom = pt.get<string>("geom_name");

		itsConfiguration->itsGeomName = geom;

		shared_ptr<plugin::neons> n = dynamic_pointer_cast<plugin::neons> (plugin_factory::Instance()->Plugin("neons"));

		map<string, string> geominfo = n->GeometryDefinition(geom);

		if (!geominfo.empty())
		{
			/*
			 *  In Neons we don't have rotlatlon projection used separately, instead we have
			 *  to check if geom_parm_1 and geom_parm_2 specify the regular rotated location
			 *  if south pole (0,30).
			 */

			if (geominfo["prjn_name"] == "latlon" && (geominfo["geom_parm_1"] != "0" || geominfo["geom_parm_2"] != "0"))
			{
				itsConfiguration->itsInfo->Projection(kRotatedLatLonProjection);
				itsConfiguration->itsInfo->SouthPole(point(boost::lexical_cast<double>(geominfo["geom_parm_2"]) / 1e3, boost::lexical_cast<double>(geominfo["geom_parm_1"]) / 1e3));

			}
			else if (geominfo["prjn_name"] == "latlon")
			{
				itsConfiguration->itsInfo->Projection(kLatLonProjection);
			}
			else if (geominfo["prjn_name"] == "polster" || geominfo["prjn_name"] == "polarstereo")
			{
				itsConfiguration->itsInfo->Projection(kStereographicProjection);
				itsConfiguration->itsInfo->Orientation(boost::lexical_cast<double>(geominfo["geom_parm_1"]) / 1e3);
			}
			else
			{
				throw runtime_error(ClassName() + ": Unknown projection: " + geominfo["prjn_name"]);
			}

			itsConfiguration->Ni(boost::lexical_cast<size_t> (geominfo["col_cnt"]));
			itsConfiguration->Nj(boost::lexical_cast<size_t> (geominfo["row_cnt"]));

			if (geominfo["stor_desc"] == "+x-y")
			{
				itsConfiguration->itsScanningMode = kTopLeft;
			}
			else if (geominfo["stor_desc"] == "+x+y")
			{
				itsConfiguration->itsScanningMode = kBottomLeft;
			}
			else
			{
				throw runtime_error(ClassName() + ": scanning mode " + geominfo["stor_desc"] + " not supported yet");
			}

			double X0 = boost::lexical_cast<double>(geominfo["long_orig"]) / 1e3;
			double Y0 = boost::lexical_cast<double>(geominfo["lat_orig"]) / 1e3;

			double di = boost::lexical_cast<double>(geominfo["pas_longitude"])/1e3;
			double dj = boost::lexical_cast<double>(geominfo["pas_latitude"])/1e3;

			std::pair<point, point> coordinates = util::CoordinatesFromFirstGridPoint(point(X0, Y0), itsConfiguration->Ni(), itsConfiguration->Nj(), di, dj, itsConfiguration->itsScanningMode);
//			itsConfiguration->Info()->SetCoordinatesFromFirstGridPoint(point(X0, Y0), itsConfiguration->Ni(), itsConfiguration->Nj(), di, dj);

			itsConfiguration->itsInfo->BottomLeft(coordinates.first);
			itsConfiguration->itsInfo->TopRight(coordinates.second);
			return;
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

	try
	{
		string theProjection = pt.get<string>("projection");

		if (itsConfiguration->itsInfo->Projection() == kUnknownProjection)
		{

			if (theProjection == "latlon")
			{
				itsConfiguration->itsInfo->Projection(kLatLonProjection);
			}
			else if (theProjection == "rotated_latlon")
			{
				itsConfiguration->itsInfo->Projection(kRotatedLatLonProjection);
			}
			else if (theProjection == "stereographic")
			{
				itsConfiguration->itsInfo->Projection(kStereographicProjection);
			}
			else
			{
				itsLogger->Warning("Unknown projection: " + theProjection);
			}
		}
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing projection: ") + e.what());
	}

	try
	{
		itsConfiguration->Info()->BottomLeft(point(pt.get<double>("bottom_left_longitude"), pt.get<double>("bottom_left_latitude")));
		itsConfiguration->Info()->TopRight(point(pt.get<double>("top_right_longitude"), pt.get<double>("top_right_latitude")));
		itsConfiguration->Info()->Orientation(pt.get<double>("orientation"));

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing area corners: ") + e.what());
	}

	/* Check orientation */

	try
	{
		itsConfiguration->Info()->Orientation(pt.get<double>("orientation"));

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing area corners: ") + e.what());
	}

	/* Check south pole coordinates */

	try
	{
		itsConfiguration->Info()->BottomLeft(point(pt.get<double>("south_pole_longitude"), pt.get<double>("south_pole_latitude")));
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing area corners: ") + e.what());
	}

	/* Check grid definitions */

	try
	{
		itsConfiguration->Ni(pt.get<size_t>("ni"));
		itsConfiguration->Nj(pt.get<size_t>("nj"));

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (exception& e)
	{
		throw runtime_error(string("Error parsing grid dimensions: ") + e.what());
	}
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
