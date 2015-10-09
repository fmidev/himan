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
#include "regular_grid.h"

using namespace std;
using namespace himan::plugin;

const int MAX_WORKERS = 16;
static once_flag oflag;

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
		NFmiRadonDBPool::Instance()->Username("wetodb");
		NFmiRadonDBPool::Instance()->Password("3loHRgdio");
	}
}

vector<string> radon::Files(search_options& options)
{

	Init();

	vector<string> files;

	string analtime = options.time.OriginDateTime().String("%Y-%m-%d %H:%M:%S+00");
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
	
	string forecastTypeValue = (options.ftype.Type() == kEpsPerturbation) ? boost::lexical_cast<string> (options.ftype.Value()) : "-1";
			
	for (size_t i = 0; i < gridgeoms.size(); i++)
	{
		string tablename = gridgeoms[i][1];
		string geomid = gridgeoms[i][0];

		string parm_name = options.param.Name();

		string query = "SELECT param_id, level_id, level_value, forecast_period, file_location, file_server "
				   "FROM "+tablename+"_v "
				   "WHERE analysis_time = '"+analtime+"' "
				   "AND param_name = '"+parm_name+"' "
				   "AND level_name = upper('"+level_name+"') "
				   "AND level_value = "+levelvalue+" "
				   "AND forecast_period = '"+util::MakeSQLInterval(options.time)+"' "
				   "AND geometry_id = "+geomid+" "
				   "AND forecast_type_id = "+boost::lexical_cast<string> (options.ftype.Type())+" "
				   "AND forecast_type_value = "+forecastTypeValue+" "
				   "ORDER BY forecast_period, level_id, level_value";

		itsRadonDB->Query(query);

		vector<string> values = itsRadonDB->FetchRow();

		if (values.empty())
		{
			continue;
		}

		itsLogger->Trace("Found data for parameter " + parm_name + " from radon geometry " + gridgeoms[i][3]);
		
		files.push_back(values[4]);

		break; // discontinue loop on first positive match

	}

	return files;

}

bool radon::Save(const info& resultInfo, const string& theFileName)
{
	
	Init();
	
	stringstream query;

	if (resultInfo.Grid()->Type() != kRegularGrid)
	{
		itsLogger->Error("Only grid data can be stored to radon for now");
		return false;
	}

	const regular_grid* g = dynamic_cast<regular_grid*> (resultInfo.Grid());

	/*
	 * 1. Get grid information
	 * 2. Get model information
	 * 3. Get data set information (ie model run)
	 * 4. Insert or update
	 */

	himan::point firstGridPoint = g->FirstGridPoint();

	// get grib1 gridType

	int gridType = -1;

	switch (g->Projection()) {
		case 10:
			gridType = 0; // latlon
			break;
		case 11:
			gridType = 10; // rot latlon
			break;
		case 13:
			gridType = 5; // polster
			break;
		default:	
			throw runtime_error("Unsupported projection: " + HPProjectionTypeToString.at(g->Projection()));
	}

        auto geominfo = itsRadonDB->GetGeometryDefinition(g->Ni(), g->Nj(), firstGridPoint.Y(), firstGridPoint.X(), g->Di(), g->Dj(), 1, gridType);

	if (geominfo.empty())
	{
		itsLogger->Warning("Grid geometry not found from radon");
		return false;
	}

	string geom_id = geominfo["id"];

	query.str("");

	query	<< "SELECT "
			<< "id, table_name "
			<< "FROM as_grid "
			<< "WHERE geometry_id = '" << geom_id << "'"
			<< " AND analysis_time = '" << resultInfo.OriginDateTime().String("%Y-%m-%d %H:%M:%S+00") << "'"
			<< " AND producer_id = " << resultInfo.Producer().Id();

	itsRadonDB->Query(query.str());

	auto row = itsRadonDB->FetchRow();

	if (row.empty())
	{
		itsLogger->Warning("Data set definition not found from radon");
		return false;
	}

	string table_name = row[1];
	string dset_id = row[0];

	query.str("");

	char host[255];
	gethostname(host, 255);
	
	auto paraminfo = itsRadonDB->GetParameterFromDatabaseName(resultInfo.Producer().Id(), resultInfo.Param().Name());

	if (paraminfo.empty())
	{
		itsLogger->Error("Parameter information not found from radon for parameter " + resultInfo.Param().Name());
		return false;
	}

	auto levelinfo = itsRadonDB->GetLevelFromGrib(resultInfo.Producer().Id(), resultInfo.Level().Type(), 1);

	if (levelinfo.empty())
	{
		itsLogger->Error("Level information not found from radon for level " + resultInfo.Level().Name());
		return false;
	}

	/*
	 * We have our own error logging for unique key violations
	 */
	
	// itsRadonDB->Verbose(false);
	
	double forecastTypeValue = (resultInfo.ForecastType().Type() == kEpsPerturbation) ? resultInfo.ForecastType().Value() : -1.;
	
	string analysisTime = resultInfo.OriginDateTime().String("%Y-%m-%d %H:%M:%S+00");
			
	query  << "INSERT INTO data." << table_name
		   << " (producer_id, analysis_time, geometry_id, param_id, level_id, level_value, forecast_period, forecast_type_id, forecast_type_value, file_location, file_server) VALUES ("
		   << resultInfo.Producer().Id() << ", "
		   << "'" << analysisTime << "', "
		   << geom_id << ", "
		   << paraminfo["id"] << ", "
		   << levelinfo["id"] << ", "
		   << resultInfo.Level().Value() << ", "
		   << "'" << util::MakeSQLInterval(resultInfo.Time()) << "', "
		   << resultInfo.ForecastType().Type() << ", "
		   << forecastTypeValue << ","
		   << "'" << theFileName << "', "
		   << "'" << host << "')"
		   ;
	
	try
	{
		itsRadonDB->Execute(query.str());
		
		query.str("");
		
		query << "UPDATE as_grid SET record_count = record_count+1 WHERE producer_id = " << resultInfo.Producer().Id()
				<< " AND geometry_id = " << geom_id
				<< " AND analysis_time = '" << analysisTime << "'";

		itsRadonDB->Execute(query.str());
		itsRadonDB->Commit();
	}
	catch (int e)
	{
		itsRadonDB->Rollback();
		
		if (e == 7)
		{
			query.str("");
			query << "UPDATE data." << table_name << " SET "
					<< "file_location = '" << theFileName << "', "
					<< "file_server = '" << host << "' WHERE "
					<< "producer_id = " << resultInfo.Producer().Id() << " AND "
					<< "analysis_time = '" << analysisTime << "' AND "
					<< "geometry_id = " << geom_id << " AND "
					<< "param_id = " << paraminfo["id"] << " AND "
					<< "level_id = " << levelinfo["id"] << " AND "
					<< "level_value = " << resultInfo.Level().Value() << " AND "
					<< "forecast_period = " << "'" << util::MakeSQLInterval(resultInfo.Time()) << "' AND "
					<< "forecast_type_id = " << resultInfo.ForecastType().Type() << " AND "
					<< "forecast_type_value = " << forecastTypeValue;
					
			itsRadonDB->Execute(query.str());
			itsRadonDB->Commit();
		}
		else
		{
			itsLogger->Error("Error code: " + boost::lexical_cast<string> (e));
			itsLogger->Error("Query: " + query.str());
		}

		return false;
	}

	itsLogger->Trace("Saved information on file '" + theFileName + "' to radon");

	return true;
}

map<string,string> radon::Grib1ParameterName(long producer, long fmiParameterId, long codeTableVersion, long timeRangeIndicator, long levelId, double level_value)
{	
	Init();
	
	map<string,string> paramName = itsRadonDB->GetParameterFromGrib1(producer, codeTableVersion, fmiParameterId, timeRangeIndicator, levelId, level_value);
	return paramName; 
}

map<string,string> radon::Grib2ParameterName(long fmiParameterId, long category, long discipline, long producer, long levelId, double level_value)
{
	Init();
	
	map<string,string> paramName = itsRadonDB->GetParameterFromGrib2(producer, discipline, category, fmiParameterId, levelId, level_value);
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

