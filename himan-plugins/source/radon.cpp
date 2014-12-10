/**
 * @file radon.cpp
 *
 * @date Oct 28, 2012
 * @author tack
 */

#include "radon.h"
#include "logger_factory.h"
#include "plugin_factory.h"
#include <thread>
#include <sstream>
#include "util.h"
#include "unistd.h" // getuid())

using namespace std;
using namespace himan::plugin;

const int MAX_WORKERS = 16;
once_flag oflag;

radon::radon() : itsInit(false), itsRadonDB()
{
	itsLogger = unique_ptr<logger> (logger_factory::Instance()->GetLog("radon"));

	// no lambda functions for gcc 4.4 :(
	// call_once(oflag, [](){ NFmiNeonsDBPool::MaxWorkers(MAX_WORKERS); });

	call_once(oflag, &himan::plugin::radon::InitPool, this);
}

void radon::InitPool()
{
	NFmiRadonDBPool::Instance()->MaxWorkers(MAX_WORKERS);

	uid_t uid = getuid();
	
	if (uid == 1459) // weto
	{
		char* base = getenv("MASALA_BASE");

		if (string(base) == "/masala")
		{
			NFmiRadonDBPool::Instance()->ReadWriteTransaction(true);
			NFmiRadonDBPool::Instance()->Username("wetodb");
			NFmiRadonDBPool::Instance()->Password("3loHRgdio");
		}
		else
		{
			itsLogger->Warning("Program executed as uid 1459 ('weto') but MASALA_BASE not set");
		}
	}
}

vector<string> radon::Files(const search_options& options)
{

	Init();

	vector<string> files;

	string analtime = options.time.OriginDateTime()->String("%Y%m%d%H%M%S");
	string levelvalue = boost::lexical_cast<string> (options.level.Value());

	string ref_prod = options.prod.Name();
	// long no_vers = options.prod.TableVersion();

	string level_name = HPLevelTypeToString.at(options.level.Type());

	vector<vector<string> > gridgeoms;
	vector<string> sourceGeoms = options.configuration->SourceGeomNames();

	if (sourceGeoms.empty())
	{
		// Get all geometries
		gridgeoms = itsRadonDB->GetGridGeoms(ref_prod, analtime);
	}
	else
	{
		for (size_t i = 0; i < sourceGeoms.size(); i++)
		{
			vector<vector<string>> geoms = itsRadonDB->GetGridGeoms(ref_prod, analtime, sourceGeoms[i]);
			gridgeoms.insert(gridgeoms.end(), geoms.begin(), geoms.end());
		}
	}

	if (gridgeoms.empty())
	{
		itsLogger->Warning("No geometries found for producer " + ref_prod +
		", analysistime " + analtime + ", source geom name(s) '" + util::Join(sourceGeoms, ",") +"', param " + options.param.Name());
		
		return files;
	}

	for (size_t i = 0; i < gridgeoms.size(); i++)
	{
		string tablename = gridgeoms[i][1];

		string parm_name = options.param.Name();

		string query = "SELECT param_id, level_id, level_value, forecast_period, file_location, file_server "
				   "FROM "+tablename+"_v "
				   "WHERE analysis_time = "+analtime+" "
				   "AND param_name = "+parm_name+" "
				   "AND level_name = "+level_name+" "
				   "AND level_value = " +levelvalue+" "
				   "AND forecast_period = "+boost::lexical_cast<string> (options.time.Step())+" "
				   "ORDER BY forecast_period, level_id, level_value";

		itsRadonDB->Query(query);

		vector<string> values = itsRadonDB->FetchRow();

		if (values.empty())
		{
			continue;
		}

		itsLogger->Trace("Found data for parameter " + parm_name + " from radon geometry " + gridgeoms[i][0]);
		
		files.push_back(values[4]);

		break; // discontinue loop on first positive match

	}

	return files;

}

bool radon::Save(shared_ptr<const info> resultInfo, const string& theFileName)
{
	Init();

	stringstream query;

	/*
	 * 1. Get grid information
	 * 2. Get model information
	 * 3. Get data set information (ie model run)
	 * 4. Insert or update
	 */

	himan::point firstGridPoint = resultInfo->Grid()->FirstGridPoint();

	/*
	 * pas_latitude and pas_longitude cannot be checked programmatically
	 * since f.ex. in the case for GFS in radon we have value 500 and
	 * by calculating we have value 498. But not check these columns should
	 * not matter as long as row_cnt, col_cnt, lat_orig and lon_orig match
	 * (since pas_latitude and pas_longitude are derived from these anyway)
	 */

	query 	<< "SELECT id "
			<< "FROM geom "
			<< "WHERE row_cnt = " << resultInfo->Nj()
			<< " AND col_cnt = " << resultInfo->Ni()
			<< " AND lat_orig = " << (firstGridPoint.Y() * 1e3)
			<< " AND long_orig = " << (firstGridPoint.X() * 1e3);

	itsRadonDB->Query(query.str());

	vector<string> row;

	row = itsRadonDB->FetchRow();

	if (row.empty())
	{
		itsLogger->Warning("Grid geometry not found from radon");
		return false;
	}

	string geom_id = row[0];

	query.str("");

	query	<< "SELECT "
			<< "id, table_name"
			<< "FROM as_grid "
			<< "WHERE geometry_id = '" << geom_id << "'"
			<< " AND analysis_time = '" << resultInfo->OriginDateTime().String("%Y%m%d%H%M") << "'";

	itsRadonDB->Query(query.str());

	row = itsRadonDB->FetchRow();

	if (row.empty())
	{
		itsLogger->Warning("Data set definition not found from radon");
		return false;
	}

	string table_name = row[1];
	string dset_id = row[0];

	string eps_specifier = "0";

	query.str("");

	string host = "undetermined host";

	char* hostname = getenv("HOSTNAME");

	if (hostname != NULL)
	{
		host = string(hostname);
	}

	/*
	 * We have our own error loggings for unique key violations
	 */
	
	// itsRadonDB->Verbose(false);
	
	query  << "INSERT INTO " << table_name
		   << " (param_id, level_id, level_value, forecast_period, file_location, file_server) "
		   << "SELECT param.id, level.id, "
		   << resultInfo->Level().Value() << ", "
		   << resultInfo->Time().Step() << ", "
		   << "'" << theFileName << "', "
		   << "'" << host << "'"
		   << "FROM param, level "
		   << "WHERE param.name = '" << resultInfo->Param().Name() << "', "
		   << "AND level,name = upper('" << HPLevelTypeToString.at(resultInfo->Level().Type()) << "')";
		  
	try
	{
		itsRadonDB->Execute(query.str());
		itsRadonDB->Commit();
	}
	catch (int e)
	{
		itsLogger->Error("Error code: " + boost::lexical_cast<string> (e));
		itsLogger->Error("Query: " + query.str());

		itsRadonDB->Rollback();

		return false;
	}

	itsLogger->Trace("Saved information on file '" + theFileName + "' to radon");

	return true;
}

map<string,string> radon::Grib1ParameterName(long producer, long fmiParameterId, long codeTableVersion, long timeRangeIndicator, long levelId, double level_value)
{	
	Init();
	
	map<string,string> paramName = itsRadonDB->ParameterFromGrib1(producer, codeTableVersion, fmiParameterId, timeRangeIndicator, levelId, level_value);
	return paramName; 
}

map<string,string> radon::Grib2ParameterName(long fmiParameterId, long category, long discipline, long producer, long levelId, double level_value)
{
	Init();
	
	map<string,string> paramName = itsRadonDB->ParameterFromGrib2(producer, discipline, category, fmiParameterId, levelId, level_value);
	return paramName;
}

string radon::ProducerMetaData(long producerId, const string& attribute) const
{
	string ret;

	if (attribute == "last hybrid level number")
	{
		switch (producerId)
		{
			case 1:
			case 199:
			case 210:
			case 230:
				ret = "65";
			break;

			case 131:
			case 240:
				ret = "137";
				break;

			default:
				throw runtime_error(ClassName() + ": Producer not supported");
				break;

		}
	}
	else if (attribute == "first hybrid level number")
	{
		switch (producerId)
		{
			case 1:
			case 199:
			case 210:
			case 230:
				ret = "1";
				break;

			case 131:
			case 240:
				ret = "24";
				break;

			default:
				throw runtime_error(ClassName() + ": Producer not supported");
				break;

		}
	}
	else
	{
		throw runtime_error(ClassName() + ": Attribute not recognized");
	}

	return ret;

	
	// In the future maybe something like this:

	//Init();
	
	//string query = "SELECT value FROM producers_eav WHERE producer_id = " + boost::lexical_cast<string> (producerId) + " AND attribute = '" + attribute + "'";
}
