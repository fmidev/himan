/**
 * @file radon.cpp
 *
 */

#include "radon.h"
#include "logger.h"
#include "plugin_factory.h"
#include "util.h"
#include <sstream>
#include <thread>

using namespace std;
using namespace himan::plugin;

const int MAX_WORKERS = 32;
static once_flag oflag;
static map<string, string> tableNameCache;
static mutex tableNameMutex;

void radon::Init()
{
	if (!itsInit)
	{
		try
		{
			call_once(oflag, [&]() {
				NFmiRadonDBPool::Instance()->Username("wetodb");
				NFmiRadonDBPool::Instance()->Password(util::GetEnv("RADON_WETODB_PASSWORD"));
				NFmiRadonDBPool::Instance()->Database("radon");
				NFmiRadonDBPool::Instance()->Hostname("vorlon");

				if (NFmiRadonDBPool::Instance()->MaxWorkers() < MAX_WORKERS)
				{
					NFmiRadonDBPool::Instance()->MaxWorkers(MAX_WORKERS);
				}
			});

			itsRadonDB = std::unique_ptr<NFmiRadonDB>(NFmiRadonDBPool::Instance()->GetConnection());
		}
		catch (int e)
		{
			itsLogger.Fatal("Failed to get connection");
			abort();
		}

		itsInit = true;
	}
}

radon::radon() : itsInit(false), itsRadonDB() { itsLogger = logger("radon"); }
void radon::PoolMaxWorkers(int maxWorkers)
{
	itsLogger.Warning("Switching worker pool size to " + std::to_string(maxWorkers));
	NFmiRadonDBPool::Instance()->MaxWorkers(maxWorkers);
}

vector<std::string> radon::CSV(search_options& options)
{
	Init();

	vector<string> csv;

	if (options.prod.Class() != kPreviClass)
	{
		itsLogger.Error("Grid producer does not have csv based data");
		return csv;
	}

	stringstream query;

	const auto analtime = options.time.OriginDateTime().String();
	const auto key = to_string(options.prod.Id()) + "_" + analtime;
	string tableName;

	const auto mapValue = tableNameCache.find(key);

	if (mapValue == tableNameCache.end())
	{
		query << "SELECT table_name FROM as_previ WHERE producer_id = " << options.prod.Id()
		      << " AND (min_analysis_time, max_analysis_time) OVERLAPS ('" << analtime << "', '" << analtime << "')";

		itsRadonDB->Query(query.str());

		const auto row = itsRadonDB->FetchRow();

		if (row.empty())
		{
			itsLogger.Error("No tables found from as_previ for producer " + options.prod.Name());
			return csv;
		}

		tableName = row[0];

		lock_guard<mutex> lock(tableNameMutex);
		tableNameCache[key] = tableName;
	}
	else
	{
		tableName = (*mapValue).second;
	}

	string levelValue2 = "-1";

	if (!IsKHPMissingValue(options.level.Value2()))
	{
		levelValue2 = boost::lexical_cast<string>(options.level.Value2());
	}

	const string period = util::MakeSQLInterval(options.time);

	query.str("");

	query << "SELECT "
	      << "t.station_id,"
	      << "s.name AS station_name,"
	      << "st_x(s.position) AS longitude,"
	      << "st_y(s.position) AS latitude,"
	      << "t.value "
	      << "FROM " << tableName << "_v t, station s "
	      << "WHERE "
	      << "t.station_id = s.id "
	      << "AND t.analysis_time = '" << analtime << "' "
	      << "AND t.param_name = '" + options.param.Name() << "' "
	      << "AND t.level_name = upper('" + HPLevelTypeToString.at(options.level.Type()) << "') "
	      << "AND t.level_value = " << options.level.Value() << " "
	      << "AND (t.level_value2 = " << options.level.Value2() << " OR t.level_value2 = -1) "
	      << "AND t.forecast_period = '" << period << "' "
	      << "AND t.forecast_type_id = " << options.ftype.Type() << " "
	      << "AND t.forecast_type_value = " << options.ftype.Value() << " "
	      << "AND t.station_id IN (";

	auto localInfo = make_shared<info>(*options.configuration->Info());

	for (localInfo->ResetLocation(); localInfo->NextLocation();)
	{
		const auto station = localInfo->Station();
		query << station.Id() << ",";
	}

	query.seekp(-1, std::ios_base::end);
	query << ")";

	itsRadonDB->Query(query.str());

	while (true)
	{
		query.str("");

		const auto row = itsRadonDB->FetchRow();

		if (row.empty())
		{
			break;
		}

		// producer_id,origintime,station_id,station_name,longitude,latitude,param_name,level_name,level_value,level_value2,forecast_period,forecast_type_id,forecast_type_value,value

		query << options.prod.Id() << "," << analtime << "," << row[0] << "," << row[1] << "," << row[2] << ","
		      << row[3] << "," << options.param.Name() << "," << HPLevelTypeToString.at(options.level.Type()) << ","
		      << options.level.Value() << "," << options.level.Value2() << "," << period << "," << options.ftype.Type()
		      << "," << options.ftype.Value() << "," << row[4] << endl;

		csv.push_back(query.str());
	}

	return csv;
}

vector<vector<string>> GetGridGeoms(himan::plugin::search_options& options, unique_ptr<NFmiRadonDB>& itsRadonDB)
{
	vector<vector<string>> gridgeoms;
	vector<string> sourceGeoms = options.configuration->SourceGeomNames();

	const string ref_prod = options.prod.Name();
	const string analtime = options.time.OriginDateTime().String("%Y-%m-%d %H:%M:%S+00");

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

	return gridgeoms;
}

string CreateFileSQLQuery(himan::plugin::search_options& options, const vector<vector<string>>& gridgeoms)
{
	const string analtime = options.time.OriginDateTime().String("%Y-%m-%d %H:%M:%S+00");
	string levelValue = boost::lexical_cast<string>(options.level.Value());
	string levelValue2 = "-1";

	if (options.level.Value2() != himan::kHPMissingValue)
	{
		levelValue2 = boost::lexical_cast<string>(options.level.Value2());
	}

	if (options.prod.Class() != himan::kGridClass)
	{
		throw runtime_error("Previ producer does not have file data");
	}

	const string level_name = himan::HPLevelTypeToString.at(options.level.Type());

	string forecastTypeValue = "-1";  // default, deterministic/analysis

	if (options.ftype.Type() > 2)
	{
		forecastTypeValue = boost::lexical_cast<string>(options.ftype.Value());
	}

	string forecastTypeId = boost::lexical_cast<string>(options.ftype.Type());

	if (options.time.Step() == 0 && options.ftype.Type() == 1)
	{
		// ECMWF (and maybe others) use forecast type id == 2 for analysis hour
		forecastTypeId += ",2";
	}

	const string parm_name = options.param.Name();

	// HIMAN-172: Reducing radon query count
	// Coalesce multiple-geometry queries into one if all are
	// using the same source table.
	// If the source table is not the same, make a union query.

	bool sameTableForAllGeometries = true;

	const string firstTable = gridgeoms[0][1];

	for (size_t i = 1; i < gridgeoms.size(); i++)
	{
		if (firstTable != gridgeoms[i][1])
		{
			sameTableForAllGeometries = false;
		}
	}

	stringstream query;

	if (sameTableForAllGeometries)
	{
		query << "SELECT file_location, geometry_id FROM " << firstTable << "_v "
		      << "WHERE analysis_time = '" << analtime << "'"
		      << " AND param_name = '" << parm_name << "'"
		      << " AND level_name = upper('" << level_name << "')"
		      << " AND level_value = " << levelValue << " AND level_value2 = " << levelValue2
		      << " AND forecast_period = '" << himan::util::MakeSQLInterval(options.time) << "'"
		      << "AND geometry_id IN (";

		for (const auto& geom : gridgeoms)
		{
			query << geom[0] << ",";
		}

		query.seekp(-1, ios_base::end);

		query << ") AND forecast_type_id IN (" << forecastTypeId << ")"
		      << " AND forecast_type_value = " << forecastTypeValue
		      << " ORDER BY forecast_period, level_id, level_value";
	}
	else
	{
		for (size_t i = 0; i < gridgeoms.size(); i++)
		{
			string tablename = gridgeoms[i][1];
			string geomid = gridgeoms[i][0];

			query << "SELECT file_location, geometry_id "
			      << "FROM " << tablename << "_v "
			      << "WHERE analysis_time = '" << analtime << "'"
			      << " AND param_name = '" << parm_name << "'"
			      << " AND level_name = upper('" << level_name << "') "
			      << " AND level_value = " << levelValue << " AND level_value2 = " << levelValue2
			      << " AND forecast_period = '" << himan::util::MakeSQLInterval(options.time) << "'"
			      << " AND geometry_id = " << geomid << " AND forecast_type_id IN (" << forecastTypeId << ")"
			      << " AND forecast_type_value = " << forecastTypeValue << " UNION ALL";
		}

		query.seekp(-9, ios_base::end);
		query << " ORDER BY forecast_period, level_id, level_value";
	}

	return query.str();
}

vector<string> radon::Files(search_options& options)
{
	Init();

	vector<string> files, values;

	const auto gridgeoms = GetGridGeoms(options, itsRadonDB);

	if (gridgeoms.empty())
	{
		return files;
	}

	const auto query = CreateFileSQLQuery(options, gridgeoms);

	if (query.empty())
	{
		return files;
	}

	try
	{
		itsRadonDB->Query(query);
		values = itsRadonDB->FetchRow();
	}
	catch (const pqxx::sql_error& e)
	{
		// Sometimes we get errors like:
		// ERROR:  deadlock detected
		// DETAIL:  Process 23465 waits for AccessShareLock on relation 35841462 of database 32027825; blocked by
		// process 23477.
		//
		// This is caused when table partitions are dropped while himan is trying to query the table.
		// As a workaround, re-execute the query.

		itsLogger.Warning("Caught database error: " + string(e.what()));
		sleep(1);
		itsRadonDB->Query(query);
		values = itsRadonDB->FetchRow();
	}

	if (values.empty())
	{
		return files;
	}

	itsLogger.Trace("Found data for parameter " + options.param.Name() + " from radon geometry " + values[1]);

	files.push_back(values[0]);

	return files;
}

bool radon::Save(const info& resultInfo, const string& theFileName)
{
	Init();

	if (resultInfo.Producer().Class() == kGridClass)
	{
		return SaveGrid(resultInfo, theFileName);
	}
	else if (resultInfo.Producer().Class() == kPreviClass)
	{
		return SavePrevi(resultInfo);
	}

	return false;
}

bool radon::SavePrevi(const info& resultInfo)
{
	stringstream query;

	auto analysisTime = resultInfo.Time().OriginDateTime().String("%Y-%m-%d %H:%M:%S+00");

	query << "SELECT id,table_name FROM as_previ WHERE producer_id = " << resultInfo.Producer().Id()
	      << " AND (min_analysis_time, max_analysis_time) OVERLAPS ('" << analysisTime << "', '" << analysisTime
	      << "')";

	itsRadonDB->Query(query.str());

	auto row = itsRadonDB->FetchRow();

	if (row.empty())
	{
		itsLogger.Warning("Data set definition not found from radon");
		return false;
	}

	string table_name = row[1];

	auto levelinfo = itsRadonDB->GetLevelFromDatabaseName(HPLevelTypeToString.at(resultInfo.Level().Type()));

	if (levelinfo.empty())
	{
		itsLogger.Error("Level information not found from radon for level " +
		                HPLevelTypeToString.at(resultInfo.Level().Type()) + ", producer " +
		                to_string(resultInfo.Producer().Id()));
		return false;
	}

	auto paraminfo = itsRadonDB->GetParameterFromDatabaseName(resultInfo.Producer().Id(), resultInfo.Param().Name(),
	                                                          stoi(levelinfo["id"]), resultInfo.Level().Value());

	if (paraminfo.empty())
	{
		itsLogger.Error("Parameter information not found from radon for parameter " + resultInfo.Param().Name() +
		                ", producer " + to_string(resultInfo.Producer().Id()));
		return false;
	}

	int forecastTypeValue = -1;  // default, deterministic/analysis

	if (resultInfo.ForecastType().Type() > 2)
	{
		forecastTypeValue = static_cast<int>(resultInfo.ForecastType().Value());
	}

	double levelValue2 = IsKHPMissingValue(resultInfo.Level().Value2()) ? -1 : resultInfo.Level().Value2();

	auto localInfo = resultInfo;

	for (localInfo.ResetLocation(); localInfo.NextLocation();)
	{
		query.str("");

		query << "INSERT INTO data." << table_name << " (producer_id, station_id, analysis_time, param_id, level_id, "
		                                              "level_value, level_value2, forecast_period, "
		                                              "forecast_type_id, forecast_type_value, value) VALUES ("
		      << localInfo.Producer().Id() << ", " << localInfo.Station().Id() << ", "
		      << "'" << analysisTime << "', " << paraminfo["id"] << ", " << levelinfo["id"] << ", "
		      << localInfo.Level().Value() << ", " << levelValue2 << ", "
		      << "'" << util::MakeSQLInterval(localInfo.Time()) << "', "
		      << static_cast<int>(localInfo.ForecastType().Type()) << ", " << forecastTypeValue << ","
		      << localInfo.Value() << ")";

		try
		{
			itsRadonDB->Execute(query.str());
			itsRadonDB->Commit();
		}
		catch (const pqxx::unique_violation& e)
		{
			itsRadonDB->Rollback();

			query.str("");
			query << "UPDATE data." << table_name << " SET "
			      << "value = " << localInfo.Value() << " WHERE "
			      << "producer_id = " << resultInfo.Producer().Id() << " AND "
			      << "station_id = " << localInfo.Station().Id() << " AND "
			      << "analysis_time = '" << analysisTime << "' AND "
			      << "param_id = " << paraminfo["id"] << " AND "
			      << "level_id = " << levelinfo["id"] << " AND "
			      << "level_value = " << resultInfo.Level().Value() << " AND "
			      << "level_value2 = " << levelValue2 << " AND "
			      << "forecast_period = "
			      << "'" << util::MakeSQLInterval(resultInfo.Time()) << "' AND "
			      << "forecast_type_id = " << static_cast<int>(resultInfo.ForecastType().Type()) << " AND "
			      << "forecast_type_value = " << forecastTypeValue;

			itsRadonDB->Execute(query.str());
			itsRadonDB->Commit();
		}
	}

	return true;
}

bool radon::SaveGrid(const info& resultInfo, const string& theFileName)
{
	stringstream query;

	if (resultInfo.Grid()->Class() != kRegularGrid)
	{
		itsLogger.Error("Only regular grid data can be stored to radon for now");
		return false;
	}

	/*
	 * 1. Get grid information
	 * 2. Get model information
	 * 3. Get data set information (ie model run)
	 * 4. Insert or update
	 */

	himan::point firstGridPoint = resultInfo.Grid()->FirstPoint();

	// get grib1 gridType

	int gribVersion = 1;
	int gridType = -1;

	switch (resultInfo.Grid()->Type())
	{
		case kLatitudeLongitude:
			gridType = 0;
			break;
		case kRotatedLatitudeLongitude:
			gridType = 10;
			break;
		case kStereographic:
			gridType = 5;
			break;
		case kReducedGaussian:
			gridType = 24;  // "stretched" gaussian
			break;
		case kAzimuthalEquidistant:
			gribVersion = 2;
			gridType = 110;
			break;
		case kLambertConformalConic:
			gridType = 3;
			break;
		default:
			throw runtime_error("Unsupported projection: " + to_string(resultInfo.Grid()->Type()) + " " +
			                    HPGridTypeToString.at(resultInfo.Grid()->Type()));
	}

	map<string, string> geominfo = itsRadonDB->GetGeometryDefinition(
	    resultInfo.Grid()->Ni(), resultInfo.Grid()->Nj(), firstGridPoint.Y(), firstGridPoint.X(),
	    resultInfo.Grid()->Di(), resultInfo.Grid()->Dj(), gribVersion, gridType);

	if (geominfo.empty())
	{
		itsLogger.Warning("Grid geometry not found from radon");
		return false;
	}

	const string geom_id = geominfo["id"];
	const string geom_name = geominfo["name"];
	auto analysisTime = resultInfo.Time().OriginDateTime().String("%Y-%m-%d %H:%M:%S+00");

	query.str("");

	query << "SELECT "
	      << "id, schema_name, partition_name, record_count "
	      << "FROM as_grid_v "
	      << "WHERE geometry_name = '" << geom_name << "'"
	      << " AND (min_analysis_time, max_analysis_time) OVERLAPS ('" << analysisTime << "'"
	      << ", '" << analysisTime << "')"
	      << " AND producer_id = " << resultInfo.Producer().Id();

	itsRadonDB->Query(query.str());

	auto row = itsRadonDB->FetchRow();

	if (row.empty())
	{
		itsLogger.Warning("Data set definition not found from radon");
		return false;
	}

	const string schema_name = row[1];
	const string table_name = row[2];
	const string record_count = row[3];

	query.str("");

	char host[255];
	gethostname(host, 255);

	auto levelinfo = itsRadonDB->GetLevelFromDatabaseName(HPLevelTypeToString.at(resultInfo.Level().Type()));

	if (levelinfo.empty())
	{
		itsLogger.Error("Level information not found from radon for level " +
		                HPLevelTypeToString.at(resultInfo.Level().Type()) + ", producer " +
		                to_string(resultInfo.Producer().Id()));
		return false;
	}

	auto paraminfo = itsRadonDB->GetParameterFromDatabaseName(resultInfo.Producer().Id(), resultInfo.Param().Name(),
	                                                          stoi(levelinfo["id"]), resultInfo.Level().Value());

	if (paraminfo.empty())
	{
		itsLogger.Error("Parameter information not found from radon for parameter " + resultInfo.Param().Name() +
		                ", producer " + to_string(resultInfo.Producer().Id()));
		return false;
	}

	/*
	 * We have our own error logging for unique key violations
	 */

	// itsRadonDB->Verbose(false);
	int forecastTypeValue = -1;  // default, deterministic/analysis

	if (resultInfo.ForecastType().Type() > 2)
	{
		forecastTypeValue = static_cast<int>(resultInfo.ForecastType().Value());
	}

	double levelValue2 = IsKHPMissingValue(resultInfo.Level().Value2()) ? -1 : resultInfo.Level().Value2();
	const string fullTableName = schema_name + "." + table_name;

	query
	    << "INSERT INTO " << fullTableName
	    << " (producer_id, analysis_time, geometry_id, param_id, level_id, level_value, level_value2, forecast_period, "
	       "forecast_type_id, forecast_type_value, file_location, file_server) VALUES ("
	    << resultInfo.Producer().Id() << ", "
	    << "'" << analysisTime << "', " << geom_id << ", " << paraminfo["id"] << ", " << levelinfo["id"] << ", "
	    << resultInfo.Level().Value() << ", " << levelValue2 << ", "
	    << "'" << util::MakeSQLInterval(resultInfo.Time()) << "', "
	    << static_cast<int>(resultInfo.ForecastType().Type()) << ", " << forecastTypeValue << ","
	    << "'" << theFileName << "', "
	    << "'" << host << "')";

	try
	{
		itsRadonDB->Execute(query.str());
		itsRadonDB->Commit();

		// After first insert we have to analyze table manually. Otherwise Himan might not be
		// able to fetch this inserted data in subsequent plugin calls, because it checks the
		// record_count column from as_grid_v which is only updated by database ANALYZE calls.
		// The database DOES do this automatically, but only after a certain threshold has been
		// passed (by default 50 changed rows).
		//
		// In some cases this implementation might lead to multiple ANALYZE calls being made, when
		// the first fields are insterted from multiple parallel threads. This does not matter,
		// ANALYZE on a near-empty table should be fast enough.

		if (record_count == "0")
		{
			itsLogger.Trace("Analyzing table " + fullTableName + " due to first insert");

			query.str("");
			query << "ANALYZE " << fullTableName;
			itsRadonDB->Execute(query.str());
		}
	}
	catch (const pqxx::unique_violation& e)
	{
		itsRadonDB->Rollback();

		query.str("");
		query << "UPDATE " << fullTableName << " SET "
		      << "file_location = '" << theFileName << "', "
		      << "file_server = '" << host << "' WHERE "
		      << "producer_id = " << resultInfo.Producer().Id() << " AND "
		      << "analysis_time = '" << analysisTime << "' AND "
		      << "geometry_id = " << geom_id << " AND "
		      << "param_id = " << paraminfo["id"] << " AND "
		      << "level_id = " << levelinfo["id"] << " AND "
		      << "level_value = " << resultInfo.Level().Value() << " AND "
		      << "level_value2 = " << levelValue2 << " AND "
		      << "forecast_period = "
		      << "'" << util::MakeSQLInterval(resultInfo.Time()) << "' AND "
		      << "forecast_type_id = " << static_cast<int>(resultInfo.ForecastType().Type()) << " AND "
		      << "forecast_type_value = " << forecastTypeValue;

		itsRadonDB->Execute(query.str());
		itsRadonDB->Commit();
	}

	itsLogger.Trace("Saved information on file '" + theFileName + "' to radon");

	return true;
}
