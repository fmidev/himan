/**
 * @file radon.cpp
 *
 */

#include "radon.h"
#include "logger.h"
#include "plugin_factory.h"
#include "point_list.h"
#include "util.h"
#include <boost/filesystem.hpp>
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

				string radonHost = "vorlon";

				try
				{
					radonHost = util::GetEnv("RADON_HOSTNAME");
				}
				catch (...)
				{
				}

				string radonName = "radon";

				try
				{
					radonName = util::GetEnv("RADON_DATABASENAME");
				}
				catch (...)
				{
				}

				int radonPort = 5432;

				try
				{
					radonPort = stoi(util::GetEnv("RADON_PORT"));
				}
				catch (...)
				{
				}

				NFmiRadonDBPool::Instance()->Username("wetodb");
				NFmiRadonDBPool::Instance()->Password(util::GetEnv("RADON_WETODB_PASSWORD"));
				NFmiRadonDBPool::Instance()->Database(radonName);
				NFmiRadonDBPool::Instance()->Hostname(radonHost);
				NFmiRadonDBPool::Instance()->Port(radonPort);

				if (NFmiRadonDBPool::Instance()->MaxWorkers() < MAX_WORKERS)
				{
					NFmiRadonDBPool::Instance()->MaxWorkers(MAX_WORKERS);
				}
				itsLogger.Info("Connected to radon (db=" + radonName + ", host=" + radonHost + ":" +
				               std::to_string(radonPort) + ")");
			});

			itsRadonDB = std::unique_ptr<NFmiRadonDB>(NFmiRadonDBPool::Instance()->GetConnection());
		}
		catch (int e)
		{
			itsLogger.Fatal("Failed to get connection");
			himan::Abort();
		}

		itsInit = true;
	}
}

radon::radon() : itsInit(false), itsRadonDB()
{
	itsLogger = logger("radon");
}
void radon::PoolMaxWorkers(int maxWorkers)
{
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
	string forecastTypeValue = "-1";  // default, deterministic/analysis

	if (options.ftype.Type() >= 3 && options.ftype.Type() <= 4)
	{
		forecastTypeValue = boost::lexical_cast<string>(options.ftype.Value());
	}

	query.str("");

	query << "SELECT "
	      << "t.station_id,"
	      << "s.name AS station_name,"
	      << "st_x(s.position) AS longitude,"
	      << "st_y(s.position) AS latitude,"
	      << "t.value "
	      << "FROM " << tableName << " t, station s, param p "
	      << "WHERE "
	      << "t.station_id = s.id "
	      << "AND t.analysis_time = '" << analtime << "' "
	      << "AND t.param_id = p.id "
	      << "AND p.name = '" << options.param.Name() << "' "
	      << "AND t.level_id = " << options.level.Type() << " "
	      << "AND t.level_value = " << options.level.Value() << " "
	      << "AND (t.level_value2 = " << options.level.Value2() << " OR t.level_value2 = -1) "
	      << "AND t.forecast_period = '" << period << "' "
	      << "AND t.forecast_type_id = " << options.ftype.Type() << " "
	      << "AND t.forecast_type_value = " << forecastTypeValue << " "
	      << "AND t.station_id IN (";

	const point_list* list = dynamic_cast<const point_list*>(options.configuration->BaseGrid());

	if (!list)
	{
		throw std::runtime_error("Input grid type is not point_list");
	}

	const auto stations = list->Stations();

	for (const auto& station : stations)
	{
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
		// TODO: This set of queries could be coalesced into just one query, but in that case we need
		// to make sure that the order of the returned geometries (if there are more than one) is the
		// same as what was given to us (i.e. what was specified in the configuration file).
		// See CreateFileSQLQuery() for guide how to do it.

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

	if (options.ftype.Type() >= 3 && options.ftype.Type() <= 4)
	{
		forecastTypeValue = boost::lexical_cast<string>(options.ftype.Value());
	}

	string forecastTypeId = boost::lexical_cast<string>(options.ftype.Type());

	if (options.time.Step().Hours() == 0 && options.ftype.Type() == 1)
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
		const std::string partition = gridgeoms[0][5];
		const std::string schema = gridgeoms[0][4];

		// clang-format off

		query << "SELECT t.file_location, g.name, byte_offset, byte_length, file_format_id, file_protocol_id, message_no, t.file_server "
		      << "FROM " << schema << "." << partition << " t, geom g, param p, level l"
		      << " WHERE t.geometry_id = g.id"
		      << " AND t.producer_id = " << options.prod.Id()
		      << " AND t.param_id = p.id"
		      << " AND l.id = t.level_id"
		      << " AND t.analysis_time = '" << analtime << "'"
		      << " AND p.name = '" << parm_name << "'"
		      << " AND l.name = upper('" << level_name << "')"
		      << " AND t.level_value = " << levelValue << " AND t.level_value2 = " << levelValue2
		      << " AND t.forecast_period = '" << himan::util::MakeSQLInterval(options.time) << "'"
		      << " AND forecast_type_id IN (" << forecastTypeId << ")"
		      << " AND forecast_type_value = " << forecastTypeValue << " AND g.id IN (";

		// clang-format on

		for (const auto& geom : gridgeoms)
		{
			query << geom[0] << ",";
		}

		query.seekp(-1, ios_base::end);

		// Add custom sort order, as we want to preserve to order from conf file

		query << ") ORDER BY forecast_period, level_id, level_value"
		      << ", array_position(array[";

		for (const auto& geom : gridgeoms)
		{
			query << "'" << geom[0] << "',";
		}

		query.seekp(-1, ios_base::end);
		query << "], geometry_id::text)";
	}
	else
	{
		for (size_t i = 0; i < gridgeoms.size(); i++)
		{
			string tablename = gridgeoms[i][1];
			string geomid = gridgeoms[i][0];

			query << "SELECT file_location, geometry_name, byte_offset, byte_length, file_format_id, file_protocol_id, "
			         "message_no "
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

vector<himan::file_information> radon::Files(search_options& options)
{
	Init();

	vector<string> values;
	vector<file_information> ret;

	const auto gridgeoms = GetGridGeoms(options, itsRadonDB);

	if (gridgeoms.empty())
	{
		return ret;
	}

	const auto query = CreateFileSQLQuery(options, gridgeoms);

	if (query.empty())
	{
		return ret;
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
		return ret;
	}

	itsLogger.Trace("Found data for parameter " + options.param.Name() + " from radon geometry " + values[1]);

	file_information finfo;
	finfo.file_location = values[0];
	finfo.file_server = values[7];
	finfo.file_type = static_cast<HPFileType>(stoi(values[4]));  // 1 = GRIB1, 2=GRIB2
	finfo.storage_type = static_cast<HPFileStorageType>(stoi(values[5]));

	try
	{
		finfo.offset = static_cast<unsigned long>(stoul(values[2]));
		finfo.length = static_cast<unsigned long>(stoul(values[3]));
		finfo.message_no = static_cast<unsigned long>(stoul(values[6]));
	}
	catch (const invalid_argument& e)
	{
		finfo.offset = boost::none;
		finfo.length = boost::none;
		finfo.message_no = boost::none;
	}

	return {finfo};
}

bool radon::Save(const info<double>& resultInfo, const file_information& finfo, const string& targetGeomName)
{
	return Save<double>(resultInfo, finfo, targetGeomName);
}

template <typename T>
bool radon::Save(const info<T>& resultInfo, const file_information& finfo, const string& targetGeomName)
{
	Init();

	if (resultInfo.Producer().Class() == kGridClass)
	{
		return SaveGrid(resultInfo, finfo, targetGeomName);
	}
	else if (resultInfo.Producer().Class() == kPreviClass)
	{
		return SavePrevi(resultInfo);
	}

	return false;
}

template bool radon::Save<double>(const info<double>&, const file_information&, const string&);
template bool radon::Save<float>(const info<float>&, const file_information&, const string&);

template <typename T>
bool radon::SavePrevi(const info<T>& resultInfo)
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

	if (resultInfo.ForecastType().Type() >= 3 && resultInfo.ForecastType().Type() <= 4)
	{
		forecastTypeValue = static_cast<int>(resultInfo.ForecastType().Value());
	}

	double levelValue2 = IsKHPMissingValue(resultInfo.Level().Value2()) ? -1 : resultInfo.Level().Value2();

	auto localInfo = resultInfo;

	for (localInfo.ResetLocation(); localInfo.NextLocation();)
	{
		query.str("");

		query << "INSERT INTO data." << table_name
		      << " (producer_id, station_id, analysis_time, param_id, level_id, "
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

template bool radon::SavePrevi<double>(const info<double>&);
template bool radon::SavePrevi<float>(const info<float>&);

template <typename T>
bool radon::SaveGrid(const info<T>& resultInfo, const file_information& finfo, const string& targetGeomName)
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

	// Start by trying to search with the geometry name. If there are duplicates, the 'wrong' geometry maybe returned if
	// we search with the geometry definition first.
	map<string, string> geominfo;

	if (!targetGeomName.empty())
	{
		geominfo = itsRadonDB->GetGeometryDefinition(targetGeomName);
	}

	if (geominfo.empty())
	{
		if (resultInfo.Grid()->Class() == kRegularGrid)
		{
			auto gr = dynamic_pointer_cast<regular_grid>(resultInfo.Grid());

			geominfo = itsRadonDB->GetGeometryDefinition(gr->Ni(), gr->Nj(), firstGridPoint.Y(), firstGridPoint.X(),
			                                             gr->Di(), gr->Dj(), gribVersion, gridType);
		}
	}

	if (geominfo.empty())
	{
		itsLogger.Warning("Grid geometry not found from radon");
		return false;
	}

	const string geom_id = geominfo["id"];
	const string geom_name = geominfo["name"];
	auto analysisTime = resultInfo.Time().OriginDateTime().String("%Y-%m-%d %H:%M:%S+00");

	auto tableinfo = itsRadonDB->GetTableName(resultInfo.Producer().Id(), analysisTime, geom_name);

	if (tableinfo.empty())
	{
		itsLogger.Warning("Data set definition not found from radon");
		return false;
	}

	const string schema_name = tableinfo["schema_name"];
	const string table_name = tableinfo["partition_name"];
	const string record_count = tableinfo["record_count"];

	query.str("");

	string host;

	switch (finfo.storage_type)
	{
		case kLocalFileSystem:
		{
			char host_[128];
			gethostname(host_, 128);
			host = string(host_);
		}
		break;
		case kS3ObjectStorageSystem:
		{
			const char* host_ = getenv("S3_HOSTNAME");
			host = string(host_);
		}
		break;
		default:
			break;
	}

	if (host.empty())
	{
		itsLogger.Error("Hostname could not be determined");
		himan::Abort();
	}

	auto levelinfo = itsRadonDB->GetLevelFromDatabaseName(HPLevelTypeToString.at(resultInfo.Level().Type()));

	if (levelinfo.empty())
	{
		itsLogger.Error("Level information not found from radon for level " +
		                HPLevelTypeToString.at(resultInfo.Level().Type()) + ", producer " +
		                to_string(resultInfo.Producer().Id()));
		return false;
	}

	if (resultInfo.Param().Id() == kHPMissingInt)
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

	if (resultInfo.ForecastType().Type() >= 3 && resultInfo.ForecastType().Type() <= 4)
	{
		forecastTypeValue = static_cast<int>(resultInfo.ForecastType().Value());
	}

	double levelValue2 = IsKHPMissingValue(resultInfo.Level().Value2()) ? -1 : resultInfo.Level().Value2();
	const string fullTableName = schema_name + "." + table_name;

	auto FormatToSQL = [](const boost::optional<unsigned long>& opt) -> string {
		if (opt)
		{
			return to_string(opt.get());
		}

		return "NULL";
	};

	query
	    << "INSERT INTO " << fullTableName
	    << " (producer_id, analysis_time, geometry_id, param_id, level_id, level_value, level_value2, forecast_period, "
	    << "forecast_type_id, forecast_type_value, file_location, file_server, file_format_id, file_protocol_id, "
	    << "message_no, byte_offset, byte_length) VALUES (" << resultInfo.Producer().Id() << ", "
	    << "'" << analysisTime << "', " << geom_id << ", " << resultInfo.Param().Id() << ", " << levelinfo["id"] << ", "
	    << resultInfo.Level().Value() << ", " << levelValue2 << ", "
	    << "'" << util::MakeSQLInterval(resultInfo.Time()) << "', "
	    << static_cast<int>(resultInfo.ForecastType().Type()) << ", " << forecastTypeValue << ","
	    << "'" << finfo.file_location << "', "
	    << "'" << host << "', " << finfo.file_type << ", " << finfo.storage_type << ", "
	    << FormatToSQL(finfo.message_no) << ", " << FormatToSQL(finfo.offset) << ", " << FormatToSQL(finfo.length)
	    << ")";

	try
	{
		itsRadonDB->Execute(query.str());
		itsRadonDB->Commit();

		// After first insert set record_count to 1, to mark that this partition has data

		if (record_count == "0")
		{
			itsLogger.Trace("Updating as_grid record_count column for " + fullTableName);

			query.str("");
			query << "UPDATE as_grid SET record_count = 1 WHERE schema_name = '" << schema_name
			      << "' AND partition_name = '" << table_name << "'";

			itsRadonDB->Execute(query.str());
		}
	}
	catch (const pqxx::unique_violation& e)
	{
		itsRadonDB->Rollback();

		query.str("");
		query << "UPDATE " << fullTableName << " SET "
		      << "file_location = '" << finfo.file_location << "', "
		      << "file_server = '" << host << "', "
		      << "file_format_id = " << finfo.file_type << ", "
		      << "file_protocol_id = " << finfo.storage_type << ", "
		      << "message_no = " << FormatToSQL(finfo.message_no) << ", "
		      << "byte_offset = " << FormatToSQL(finfo.offset) << ", "
		      << "byte_length = " << FormatToSQL(finfo.length) << " WHERE "
		      << "producer_id = " << resultInfo.Producer().Id() << " AND "
		      << "analysis_time = '" << analysisTime << "' AND "
		      << "geometry_id = " << geom_id << " AND "
		      << "param_id = " << resultInfo.Param().Id() << " AND "
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

	query.str("");
	query << "Saved information on file '" << finfo.file_location << "'";

	if (finfo.message_no)
	{
		query << " message no " << finfo.message_no.get();
	}

	itsLogger.Trace(query.str());

	return true;
}

template bool radon::SaveGrid<double>(const info<double>&, const file_information&, const string&);
template bool radon::SaveGrid<float>(const info<float>&, const file_information&, const string&);
