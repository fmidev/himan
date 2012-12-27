/*
 * ini_parser.cpp
 *
 *  Created on: Nov 19, 2012
 *      Author: partio
 */

#include "ini_parser.h"
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <stdexcept>
#include "plugin_factory.h"
#include "logger_factory.h"

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
 *    override those in the conf file.
 *
 * 2) Read configuration file (if specified).
 *
 * 3) Create configuration instance.
 *
 * Some of the required information is missing, this function will not
 * behave nicely and will throw an error.
 *
 */

ini_parser* ini_parser::itsInstance = NULL;

ini_parser* ini_parser::Instance()
{

	if (!itsInstance)
	{
		itsInstance = new ini_parser;
	}

	return itsInstance;
}

ini_parser::ini_parser()
{
	itsLogger = logger_factory::Instance()->GetLog("ini_parser");
	itsConfiguration = shared_ptr<configuration> (new configuration());
}

shared_ptr<configuration> ini_parser::Parse(int argc, char** argv)
{

	ParseAndCreateInfo(argc, argv);

	return itsConfiguration;

}


void ini_parser::ParseAndCreateInfo(int argc, char** argv)
{

	ParseCommandLine(argc, argv);

	// Check that we have configuration file defined in command line

	if (itsConfiguration->itsConfigurationFile.empty())
	{
		throw std::runtime_error("Configuration file not defined");
	}

	ParseConfigurationFile(itsConfiguration->itsConfigurationFile);

	// Check requested plugins

	if (itsConfiguration->itsPlugins.size() == 0)
	{
		throw std::runtime_error("No requested plugins defined");
	}

	if (itsConfiguration->itsOutputFileType == kUnknownFile)
	{
		itsConfiguration->itsOutputFileType = kQueryData;    // Default output file is qd
	}

}

void ini_parser::ParseCommandLine(int argc, char* argv[])
{
	itsLogger->Info("Parsing command line");

	namespace po = boost::program_options;

	po::options_description desc("Allowed options");

	// Can't use required() since it doesn't allow us to use --list-plugins

	std::string outfileType;

	himan::HPDebugState debugState = himan::kDebugMsg;

	int logLevel = 0;

	desc.add_options()
	("help,h", "print out help message")
	("type,t", po::value(&outfileType), "output file type, one of: grib, grib2, netcdf, querydata")
	("version,v", "display version number")
	("configuration-file,f", po::value(&(itsConfiguration->itsConfigurationFile)), "configuration file")
	("aux-files,a", po::value<std::vector<std::string> > (&(itsConfiguration->itsAuxiliaryFiles)), "auxiliary (helper) file(s)")
	("plugins,p", po::value<std::vector<std::string> > (&(itsConfiguration->itsPlugins)), "calculated plugins")
	("list-plugins,l", "list all defined plugins")
	("d,debug-level", po::value(&logLevel), "set log level: 0(fatal) 1(error) 2(warning) 3(info) 4(debug) 5(trace)")
	;

	po::positional_options_description p;
	p.add("aux-files", -1);

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
		std::cout << "himan-lib version ???" << std::endl;
		exit(1);
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
			throw std::runtime_error("Invalid file type: " + outfileType);
		}
	}

	if (opt.count("help"))
	{
		std::cout << "usage: himan [ options ]" << std::endl;
		std::cout << desc;
		std::cout << std::endl << "Examples:" << std::endl;
		std::cout << "  himan -f etc/tpot.ini" << std::endl;
		std::cout << "  himan -f etc/vvmms.ini -a file.grib -t querydata" << std::endl << std::endl;
		exit(1);
	}

	if (opt.count("list-plugins"))
	{

		std::vector<shared_ptr<plugin::himan_plugin> > thePlugins = plugin_factory::Instance()->CompiledPlugins();

		std::cout << "Compiled plugins" << std::endl;

		for (size_t i = 0; i < thePlugins.size(); i++)
		{
			std::cout << "\t" << thePlugins[i]->ClassName() << "\t(version " << thePlugins[i]->Version() << ")" << std::endl;
		}

		thePlugins = plugin_factory::Instance()->InterpretedPlugins();

		std::cout << "Interpreted plugins" << std::endl;

		for (size_t i = 0; i < thePlugins.size(); i++)
		{
			std::cout << "\t" << thePlugins[i]->ClassName() << "\t(version " << thePlugins[i]->Version() << ")" << std::endl;
		}

		std::cout << "Auxiliary plugins" << std::endl;

		thePlugins = plugin_factory::Instance()->AuxiliaryPlugins();

		for (size_t i = 0; i < thePlugins.size(); i++)
		{
			std::cout << "\t" << thePlugins[i]->ClassName() << "\t(version " << thePlugins[i]->Version() << ")" << std::endl;
		}

		exit(1);
	}

}


void ini_parser::ParseConfigurationFile(const std::string& theConfigurationFile)
{

	itsLogger->Debug("Parsing configuration file");

	boost::property_tree::ptree pt;

	try
	{
		boost::property_tree::ini_parser::read_ini(theConfigurationFile, pt);
	}
	catch (std::exception& e)
	{
		throw std::runtime_error(std::string("Error reading configuration file: ") + e.what());
	}

	/* Check area definitions */

	try
	{
		std::string theProjection = pt.get<std::string>("area.projection");

		if (itsConfiguration->itsInfo->Projection() == kUnknownProjection)
		{

			if (theProjection == "latlon")
			{
				itsConfiguration->itsInfo->Projection(kLatLonProjection);
			}
			else if (theProjection == "rotlatlon")
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
	catch (std::exception& e)
	{
		throw std::runtime_error(std::string("Error parsing projection: ") + e.what());
	}

	try
	{
		itsConfiguration->Info()->BottomLeftLatitude(pt.get<double>("area.bottom_left_latitude"));
		itsConfiguration->Info()->BottomLeftLongitude(pt.get<double>("area.bottom_left_longitude"));
		itsConfiguration->Info()->TopRightLatitude(pt.get<double>("area.top_right_latitude"));
		itsConfiguration->Info()->TopRightLongitude(pt.get<double>("area.top_right_longitude"));
		itsConfiguration->Info()->Orientation(pt.get<double>("area.orientation"));

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (std::exception& e)
	{
		throw std::runtime_error(std::string("Error parsing area corners: ") + e.what());
	}

	/* Check grid definitions */

	try
	{
		itsConfiguration->Ni(pt.get<size_t>("grid.ni"));
		itsConfiguration->Nj(pt.get<size_t>("grid.nj"));

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (std::exception& e)
	{
		throw std::runtime_error(std::string("Error parsing grid dimensions: ") + e.what());
	}

	/* Check plugins (parameters) */

	try
	{
		itsConfiguration->Plugins(Split(pt.get<std::string>("plugins.plugins"), ','));
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (std::exception& e)
	{
		throw std::runtime_error(std::string("Error parsing plugins: ") + e.what());
	}

	/* Check time */

	try
	{

		const std::string theOriginDateTime = pt.get<std::string>("time.origintime");

		itsConfiguration->Info()->OriginDateTime(theOriginDateTime);

		std::vector<std::string> timesStr = Split(pt.get<std::string>("time.hours"), ',', true);

		std::vector<int> times ;

		for (size_t i = 0; i < timesStr.size(); i++)
		{
			times.push_back(boost::lexical_cast<int> (timesStr[i]));
		}

		sort (times.begin(), times.end());

		std::vector<shared_ptr<forecast_time> > theTimes;

		for (size_t i = 0; i < times.size(); i++)
		{

			// Create forecast_time with both times origintime, then adjust the validtime

			shared_ptr<forecast_time> theTime (new forecast_time(shared_ptr<raw_time> (new raw_time (itsConfiguration->Info()->OriginDateTime())),
			                                   shared_ptr<raw_time> (new raw_time(itsConfiguration->Info()->OriginDateTime()))));

			theTime->ValidDateTime()->Adjust("hours", times[i]);

			theTimes.push_back(theTime);
		}

		itsConfiguration->Info()->Times(theTimes);

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (std::exception& e)
	{
		throw std::runtime_error(ClassName() + ": " + std::string("Error parsing time information: ") + e.what());
	}

	/* Check producer */

	try
	{

		itsConfiguration->SourceProducer(boost::lexical_cast<unsigned int> (pt.get<std::string>("producer.source_producer")));
		itsConfiguration->TargetProducer(boost::lexical_cast<unsigned int> (pt.get<std::string>("producer.target_producer")));

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
	catch (std::exception& e)
	{
		throw std::runtime_error(ClassName() + ": " + std::string("Error parsing producer information: ") + e.what());
	}

	/* Check level */

	try
	{

		std::string theLevelTypeStr = pt.get<std::string>("level.leveltype");

		HPLevelType theLevelType;

		if (theLevelTypeStr == "height")
		{
			theLevelType = kHeight;
		}
		else if (theLevelTypeStr == "pressure")
		{
			theLevelType = kPressure;
		}
		else if (theLevelTypeStr == "hybrid")
		{
			theLevelType = kHybrid;
		}

		else
		{
			throw std::runtime_error("Unknown level type: " + theLevelTypeStr);    // not good practice; constructing string
		}

		// can cause exception, what will happen then ?

		std::vector<std::string> levelsStr = Split(pt.get<std::string>("level.levels"), ',', true);

		std::vector<float> levels ;

		for (size_t i = 0; i < levelsStr.size(); i++)
		{
			levels.push_back(boost::lexical_cast<float> (levelsStr[i]));
		}

		sort (levels.begin(), levels.end());

		std::vector<shared_ptr<level> > theLevels;

		for (size_t i = 0; i < levels.size(); i++)
		{
			theLevels.push_back(shared_ptr<level> (new level(theLevelType, levels[i])));
		}

		itsConfiguration->Info()->Levels(theLevels);
	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (std::exception& e)
	{
		throw std::runtime_error(std::string("Error parsing level information: ") + e.what());
	}

	/* Check whole_file_write */

	try
	{

		std::string theWholeFileWrite = pt.get<std::string>("meta.whole_file_write");

		if (ParseBoolean(theWholeFileWrite))
		{
			itsConfiguration->WholeFileWrite(true);
		}

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (std::exception& e)
	{
		throw std::runtime_error(std::string("Error parsing meta information: ") + e.what());
	}

	/* Check read_data_from_database */

	try
	{
		std::string theReadDataFromDatabase = pt.get<std::string>("meta.read_data_from_database");

		if (!ParseBoolean(theReadDataFromDatabase))
		{
			itsConfiguration->ReadDataFromDatabase(false);
		}

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (std::exception& e)
	{
		throw std::runtime_error(std::string("Error parsing meta information: ") + e.what());
	}

	/* Check file_wait_timeout */

	try
	{
		std::string theReadDataFromDatabase = pt.get<std::string>("meta.read_data_from_database");

		if (!ParseBoolean(theReadDataFromDatabase))
		{
			itsConfiguration->ReadDataFromDatabase(false);
		}

	}
	catch (boost::property_tree::ptree_bad_path& e)
	{
		// Something was not found; do nothing
	}
	catch (std::exception& e)
	{
		throw std::runtime_error(std::string("Error parsing meta information: ") + e.what());
	}
}

// copied from http://stackoverflow.com/questions/236129/splitting-a-string-in-c and modified a bit

std::vector<std::string> ini_parser::Split(const std::string& s, char delim, bool fill)
{

	std::string item;

	std::vector<std::string> orig_elems;

	boost::split(orig_elems, s, boost::is_any_of(","));

	if (!fill || orig_elems.size() == 0)
	{
		return orig_elems;
	}

	std::vector<std::string> filled_elems;
	std::vector<std::string> splitted_elems;

	std::vector<std::string>::iterator it;

	for (it = orig_elems.begin(); it != orig_elems.end(); )
	{

		boost::split(splitted_elems, *it, boost::is_any_of("-"));

		if (splitted_elems.size() == 2)
		{
			it = orig_elems.erase(it);

			for (int i = boost::lexical_cast<int> (splitted_elems[0]); i <= boost::lexical_cast<int> (splitted_elems[1]); i++)
			{
				filled_elems.push_back(boost::lexical_cast<std::string> (i));
			}
		}
		else
		{
			++it;
		}
	}

	std::vector<std::string> all_elems;

	all_elems.reserve(orig_elems.size() + filled_elems.size());

	all_elems.insert(all_elems.end(), orig_elems.begin(), orig_elems.end());
	all_elems.insert(all_elems.end(), filled_elems.begin(), filled_elems.end());

	return all_elems;
}

/*
 * ParseBoolean()
 *
 * Will check if given argument is a boolean value or not.
 * Note: will change argument to lower case.
 */

bool ini_parser::ParseBoolean(std::string& booleanValue)
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
		throw std::runtime_error(ClassName() + ": Invalid boolean value: " + booleanValue);
	}

	return ret;
}
