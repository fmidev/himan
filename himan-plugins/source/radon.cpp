/**
 * @file radon.cpp
 *
 */

#include "radon.h"
#include "logger.h"
#include "plugin_factory.h"
#include "point_list.h"
#include "util.h"
#include <sstream>
#include <thread>

using namespace std;
using namespace himan::plugin;

static int radonVersion = -1;
const int RADON_MIN_REQUIRED_VERSION = -1;
const int RADON_MIN_RECOMMENDED_VERSION = 20241119;
const int MAX_WORKERS = 32;
static once_flag oflag;
static map<string, string> tableNameCache;
static mutex tableNameMutex;

namespace
{
template <typename T>
string HandleNULL(const T& value);

template <>
string HandleNULL(const std::optional<double>& opt)
{
	if (opt)
	{
		return fmt::format("{}", opt.value());
	}

	return "NULL";
}

template <>
string HandleNULL(const std::optional<int>& opt)
{
	if (opt)
	{
		return to_string(opt.value());
	}

	return "NULL";
}
template <>
string HandleNULL(const std::optional<unsigned long>& opt)
{
	if (opt)
	{
		return to_string(opt.value());
	}

	return "NULL";
}

template <>
string HandleNULL(const himan::time_duration& value)
{
	if (value.Empty())
	{
		return "NULL";
	}

	return fmt::format("'{}'", static_cast<string>(value));
}

std::optional<int> HimanAggregationToRadonAggregation(himan::HPAggregationType agg)
{
	using namespace himan;
	switch (agg)
	{
		case kUnknownAggregationType:
		case kAverage:
		case kAccumulation:
		case kMaximum:
		case kMinimum:
			return static_cast<int>(agg) + 1;
		case kDifference:
		default:
			std::cerr << fmt::format("Unsupported Himan aggregation type: {}", HPAggregationTypeToString.at(agg))
			          << std::endl;
			return std::nullopt;
	}
}

std::optional<int> HimanProcessingTypeToRadonProcessingType(himan::HPProcessingType pt)
{
	using namespace himan;
	switch (pt)
	{
		case kUnknownProcessingType:
		case kProbabilityGreaterThanOrEqual:
		case kProbabilityGreaterThan:
		case kProbabilityLessThanOrEqual:
		case kProbabilityLessThan:
		case kProbabilityEquals:
		case kProbabilityEqualsIn:
			return static_cast<int>(pt) + 1;
		case kFractile:
			return 8;
		case kAreaProbabilityGreaterThanOrEqual:
			return 9;
		case kAreaProbabilityGreaterThan:
			return 10;
		case kAreaProbabilityLessThanOrEqual:
			return 11;
		case kAreaProbabilityLessThan:
			return 12;
		case kAreaProbabilityEquals:
			return 13;
		case kAreaProbabilityEqualsIn:
			return 14;
		case kBiasCorrection:
			return 15;
		case kMean:
			return 16;
		case kStandardDeviation:
			return 17;
		case kFiltered:
			return 18;
		case kDetrend:
			return 19;
		case kAnomaly:
			return 20;
		case kNormalized:
			return 21;
		case kClimatology:
			return 22;
		case kCategorized:
			return 23;
		case kPercentChange:
			return 24;
		case kEFI:
			return 25;
		case kProbabilityBetween:
			return 26;
		case kProbabilityNotEquals:
			return 27;
		case kProbability:
			return 28;
		case kAreaProbabilityBetween:
			return 29;
		case kAreaProbabilityNotEquals:
			return 30;
		case kSpread:
		default:
			std::cerr << fmt::format("Unsupported Himan processing type: {}", HPProcessingTypeToString.at(pt))
			          << std::endl;
			return std::nullopt;
	}
}
}  // namespace

void radon::Init()
{
	if (itsInit)
	{
		return;
	}
	try
	{
		call_once(
		    oflag,
		    [&]()
		    {
			    string radonHost = util::GetEnv("RADON_HOSTNAME");

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

			    itsLogger.Info(fmt::format("Connected to radon (db={}, host={}:{})", radonName, radonHost, radonPort));

			    try
			    {
				    itsRadonDB = std::unique_ptr<NFmiRadonDB>(NFmiRadonDBPool::Instance()->GetConnection());
				    const std::string version = GetVersion();

				    radonVersion = std::stoi(version);

				    if (radonVersion < RADON_MIN_REQUIRED_VERSION)
				    {
					    itsLogger.Fatal(fmt::format("Radon version '{}' is too old, at least '{}' is required",
					                                radonVersion, RADON_MIN_REQUIRED_VERSION));
					    himan::Abort();
				    }
				    else if (radonVersion < RADON_MIN_RECOMMENDED_VERSION)
				    {
					    itsLogger.Warning(fmt::format("Radon version '{}' found, at least '{}' is recommended",
					                                  radonVersion, RADON_MIN_RECOMMENDED_VERSION));
				    }
				    else
				    {
					    itsLogger.Debug(fmt::format("Radon version '{}' found", radonVersion));
				    }
			    }
			    catch (const std::invalid_argument& e)
			    {
				    itsLogger.Debug(fmt::format("Unable to determine radon version: {}", e.what()));
			    }
			    catch (const pqxx::failure& e)
			    {
				    itsLogger.Trace(fmt::format("pqxx error fetching radon version"));
			    }
			    catch (const std::exception& e)
			    {
				    itsLogger.Trace(fmt::format("Unknown error fetching radon version: {}", e.what()));
			    }
			    catch (...)
			    {
			    }
		    });

		if (!itsRadonDB)
		{
			itsRadonDB = std::unique_ptr<NFmiRadonDB>(NFmiRadonDBPool::Instance()->GetConnection());
		}
	}
	catch (int e)
	{
		itsLogger.Fatal("Failed to get connection");
		himan::Abort();
	}
	catch (const std::exception& e)
	{
		itsLogger.Fatal(e.what());
		himan::Abort();
	}

	itsInit = true;
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

	long int producer_id = options.prod.Id();
	const string analtime = options.time.OriginDateTime().String("%Y-%m-%d %H:%M:%S+00");

	if (sourceGeoms.empty())
	{
		// Get all geometries
		gridgeoms = itsRadonDB->GetGridGeoms(producer_id, analtime);
	}
	else
	{
		// TODO: This set of queries could be coalesced into just one query, but in that case we need
		// to make sure that the order of the returned geometries (if there are more than one) is the
		// same as what was given to us (i.e. what was specified in the configuration file).
		// See CreateFileSQLQuery() for guide how to do it.

		for (size_t i = 0; i < sourceGeoms.size(); i++)
		{
			vector<vector<string>> geoms = itsRadonDB->GetGridGeoms(producer_id, analtime, sourceGeoms[i]);
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
		      << " AND t.file_format_id NOT IN (3,4)" // no netcdf
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
			      << " AND t.file_format_id NOT IN (3,4)"  // no netcdf
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

	itsLogger.Trace(fmt::format("Found data for parameter {} from radon geometry {}, file name '{}' position {}/{}:{}",
	                            options.param.Name(), values[1], values[0], values[6], values[2], values[3]));

	file_information finfo;
	finfo.file_location = values[0];
	finfo.file_server = values[7];
	switch (stoi(values[4]))
	{
		case 1:
		case 2:
			finfo.file_type = static_cast<HPFileType>(stoi(values[4]));
			break;
		case 3:
			finfo.file_type = kNetCDF;
			break;
		case 4:
			finfo.file_type = kNetCDFv4;
			break;
		case 5:
			finfo.file_type = kGeoTIFF;
			break;
		default:
			itsLogger.Error("Unknown file type: " + values[4]);
			return ret;
	}

	finfo.storage_type = static_cast<HPFileStorageType>(stoi(values[5]));

	try
	{
		finfo.offset = static_cast<unsigned long>(stoul(values[2]));
		finfo.length = static_cast<unsigned long>(stoul(values[3]));
	}
	catch (const invalid_argument& e)
	{
		finfo.offset = std::nullopt;
		finfo.length = std::nullopt;
	}

	try
	{
		finfo.message_no = static_cast<unsigned long>(stoul(values[6]));
	}
	catch (const invalid_argument& e)
	{
		finfo.message_no = std::nullopt;
	}

	return {finfo};
}

pair<bool, radon_record> radon::Save(const info<double>& resultInfo, const file_information& finfo,
                                     const string& targetGeomName, bool dryRun)
{
	return Save<double>(resultInfo, finfo, targetGeomName, dryRun);
}

template <typename T>
pair<bool, radon_record> radon::Save(const info<T>& resultInfo, const file_information& finfo,
                                     const string& targetGeomName, bool dryRun)
{
	Init();

	if (resultInfo.Producer().Class() == kGridClass)
	{
		return SaveGrid(resultInfo, finfo, targetGeomName, dryRun);
	}
	else if (resultInfo.Producer().Class() == kPreviClass)
	{
		return SavePrevi(resultInfo, dryRun);
	}

	himan::Abort();
}

template pair<bool, radon_record> radon::Save<double>(const info<double>&, const file_information&, const string&,
                                                      bool);
template pair<bool, radon_record> radon::Save<float>(const info<float>&, const file_information&, const string&, bool);
template pair<bool, radon_record> radon::Save<short>(const info<short>&, const file_information&, const string&, bool);
template pair<bool, radon_record> radon::Save<unsigned char>(const info<unsigned char>&, const file_information&,
                                                             const string&, bool);

template <typename T>
pair<bool, radon_record> radon::SavePrevi(const info<T>& resultInfo, bool dryRun)
{
	stringstream query;

	auto analysisTime = resultInfo.Time().OriginDateTime().String("%Y-%m-%d %H:%M:%S+00");

	query << "SELECT id,schema_name,table_name,partition_name,record_count FROM as_previ WHERE producer_id = "
	      << resultInfo.Producer().Id() << " AND (min_analysis_time, max_analysis_time) OVERLAPS ('" << analysisTime
	      << "', '" << analysisTime << "')";

	itsRadonDB->Query(query.str());

	auto row = itsRadonDB->FetchRow();

	if (row.empty())
	{
		itsLogger.Warning(fmt::format("Dataset definition not found from radon for producer {}, analysis time '{}'",
		                              resultInfo.Producer().Id(), analysisTime));
		return make_pair(false, radon_record());
	}

	const string& schema_name = row[1];
	const string& table_name = row[2];
	const string& partition_name = row[3];
	const string& record_count = row[4];

	auto levelinfo = itsRadonDB->GetLevelFromDatabaseName(HPLevelTypeToString.at(resultInfo.Level().Type()));

	if (levelinfo.empty())
	{
		itsLogger.Error("Level information not found from radon for level " +
		                HPLevelTypeToString.at(resultInfo.Level().Type()) + ", producer " +
		                to_string(resultInfo.Producer().Id()));
		return make_pair(false, radon_record());
	}

	auto paraminfo = itsRadonDB->GetParameterFromDatabaseName(resultInfo.Producer().Id(), resultInfo.Param().Name(),
	                                                          stoi(levelinfo["id"]), resultInfo.Level().Value());

	if (paraminfo.empty())
	{
		itsLogger.Error("Parameter information not found from radon for parameter " + resultInfo.Param().Name() +
		                ", producer " + to_string(resultInfo.Producer().Id()));
		return make_pair(false, radon_record());
	}

	int forecastTypeValue = -1;  // default, deterministic/analysis

	if (resultInfo.ForecastType().Type() >= 3 && resultInfo.ForecastType().Type() <= 4)
	{
		forecastTypeValue = static_cast<int>(resultInfo.ForecastType().Value());
	}

	double levelValue2 = IsKHPMissingValue(resultInfo.Level().Value2()) ? -1 : resultInfo.Level().Value2();

	auto localInfo = resultInfo;

	const string leadTime = util::MakeSQLInterval(localInfo.Time());

	auto NaNToNull = [](const T& value) -> string
	{
		if (IsMissing(value))
		{
			return "NULL";
		}
		else
		{
			return fmt::format("{}", value);
		}
	};

	for (localInfo.ResetLocation(); localInfo.NextLocation();)
	{
		const std::string value = NaNToNull(localInfo.Value());

		if (value == "NULL")
		{
			itsLogger.Trace(
			    fmt::format("Inserting NULL value to {} for station {}", table_name, localInfo.Station().Id()));
		}

		query.str("");

		query << "INSERT INTO data." << table_name
		      << " (producer_id, station_id, analysis_time, param_id, level_id, "
		         "level_value, level_value2, forecast_period, "
		         "forecast_type_id, forecast_type_value, value) VALUES ("
		      << localInfo.Producer().Id() << ", " << localInfo.Station().Id() << ", "
		      << "'" << analysisTime << "', " << paraminfo["id"] << ", " << levelinfo["id"] << ", "
		      << localInfo.Level().Value() << ", " << levelValue2 << ", "
		      << "'" << leadTime << "', " << static_cast<int>(localInfo.ForecastType().Type()) << ", "
		      << forecastTypeValue << "," << value << ")";

		if (dryRun)
		{
			itsLogger.Trace(query.str());
			continue;
		}

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
			      << "value = " << NaNToNull(localInfo.Value()) << " WHERE "
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

	if (record_count.empty() || record_count == "0")
	{
		itsLogger.Trace("Updating as_previ record_count column for " + table_name);

		query.str("");
		query << "UPDATE as_previ SET record_count = 1 WHERE schema_name = '" << schema_name
		      << "' AND partition_name = '" << partition_name << "' AND analysis_time = '" << analysisTime << "'"
		      << " AND producer_id = " << resultInfo.Producer().Id();

		itsRadonDB->Execute(query.str());
	}
	return make_pair(true, radon_record(schema_name, table_name, partition_name, "", -1));
}

template pair<bool, radon_record> radon::SavePrevi<double>(const info<double>&, bool);
template pair<bool, radon_record> radon::SavePrevi<float>(const info<float>&, bool);
template pair<bool, radon_record> radon::SavePrevi<short>(const info<short>&, bool);
template pair<bool, radon_record> radon::SavePrevi<unsigned char>(const info<unsigned char>&, bool);

template <typename T>
pair<bool, radon_record> radon::SaveGrid(const info<T>& resultInfo, const file_information& finfo,
                                         const string& targetGeomName, bool dryRun)
{
	if (resultInfo.Grid()->Class() != kRegularGrid)
	{
		itsLogger.Error("Only regular grid data can be stored to radon for now");
		return make_pair(false, radon_record());
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

			// rotated coordinates as the first point for rotated_latitude_longitude

			const himan::point fp =
			    (gr->Type() == himan::kRotatedLatitudeLongitude)
			        ? dynamic_pointer_cast<himan::rotated_latitude_longitude_grid>(gr)->Rotate(gr->FirstPoint())
			        : gr->FirstPoint();

			geominfo =
			    itsRadonDB->GetGeometryDefinition(gr->Ni(), gr->Nj(), fp.Y(), fp.X(), gr->Di(), gr->Dj(), gr->Type());
		}
	}

	if (geominfo.empty())
	{
		itsLogger.Error("Grid geometry not found from radon");
		return make_pair(false, radon_record());
	}

	const string geom_id = geominfo["id"];
	const string geom_name = geominfo["name"];
	auto analysisTime = resultInfo.Time().OriginDateTime().String("%Y-%m-%d %H:%M:%S+00");

	auto tableinfo = itsRadonDB->GetTableName(resultInfo.Producer().Id(), analysisTime, geom_name);

	if (tableinfo.empty())
	{
		itsLogger.Error(
		    fmt::format("Dataset definition not found from radon for producer {}, analysis time '{}', geometry '{}'",
		                resultInfo.Producer().Id(), analysisTime, geom_name));
		return make_pair(false, radon_record());
	}

	const string schema_name = tableinfo["schema_name"];
	const string table_name = tableinfo["table_name"];
	const string partition_name = tableinfo["partition_name"];
	const string record_count = tableinfo["record_count"];

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

			if (host_ == nullptr)
			{
				itsLogger.Error("Env variable S3_HOSTNAME missing");
				return make_pair(false, radon_record());
			}
			host = string(host_);
			if (host.find("http") == string::npos)
			{
				itsLogger.Trace("S3_HOSTNAME missing protocol -- adding 'https://'");
				host = "https://" + host;
			}
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
		return make_pair(false, radon_record());
	}

	if (resultInfo.Param().Id() == kHPMissingInt)
	{
		itsLogger.Error("Parameter information not found from radon for parameter " + resultInfo.Param().Name() +
		                ", producer " + to_string(resultInfo.Producer().Id()));
		return make_pair(false, radon_record());
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
	const string fullTableName = fmt::format("{}.{}", schema_name, partition_name);

	int radonFileFormat = 0;

	switch (finfo.file_type)
	{
		case kGRIB1:
			radonFileFormat = 1;
			break;
		case kGRIB2:
			radonFileFormat = 2;
			break;
		case kNetCDF:
			radonFileFormat = 3;
			break;
		case kNetCDFv4:
			radonFileFormat = 4;
			break;
		case kGeoTIFF:
			radonFileFormat = 5;
			break;
		default:
			itsLogger.Error("Unknown file type: " + to_string(finfo.file_type));
			himan::Abort();
	}

	string query;

	if (radonVersion >= 20241119)
	{
		const auto agg = resultInfo.Param().Aggregation();
		const auto pt = resultInfo.Param().ProcessingType();

		auto agg_str = HandleNULL(HimanAggregationToRadonAggregation(agg.Type()));
		auto agg_period_str = HandleNULL(agg.TimeDuration());
		auto pt_str = HandleNULL(HimanProcessingTypeToRadonProcessingType(pt.Type()));
		auto pt_value_str = HandleNULL(pt.Value());
		auto pt_value2_str = HandleNULL(pt.Value2());

		query = fmt::format(
		    "INSERT INTO {} (producer_id, analysis_time, geometry_id, param_id, level_id, "
		    "level_value, level_value2, forecast_period, forecast_type_id, forecast_type_value, "
		    "file_location, file_server, file_format_id, file_protocol_id, message_no, byte_offset, "
		    "byte_length, aggregation_id, aggregation_period, processing_type_id, processing_type_value, "
		    "processing_type_value2) VALUES ({}, '{}', {}, {}, {}, {}, {}, '{}', {}, {}, '{}', '{}', {}, {}, {}, {}, "
		    "{}, {}, {}, {}, {}, {})",
		    fullTableName, resultInfo.Producer().Id(), analysisTime, geom_id, resultInfo.Param().Id(), levelinfo["id"],
		    resultInfo.Level().Value(), levelValue2, util::MakeSQLInterval(resultInfo.Time()),
		    static_cast<int>(resultInfo.ForecastType().Type()), forecastTypeValue, finfo.file_location, host,
		    radonFileFormat, static_cast<int>(finfo.storage_type), HandleNULL(finfo.message_no),
		    HandleNULL(finfo.offset), HandleNULL(finfo.length), agg_str, agg_period_str, pt_str, pt_value_str,
		    pt_value2_str);
	}
	else
	{
		query = fmt::format(
		    "INSERT INTO {} (producer_id, analysis_time, geometry_id, param_id, level_id, "
		    "level_value, level_value2, forecast_period, forecast_type_id, forecast_type_value, "
		    "file_location, file_server, file_format_id, file_protocol_id, message_no, byte_offset, "
		    "byte_length) VALUES ({}, '{}', {}, {}, {}, {}, {}, '{}', {}, {}, '{}', '{}', {}, {}, {}, {}, "
		    "{})",
		    fullTableName, resultInfo.Producer().Id(), analysisTime, geom_id, resultInfo.Param().Id(), levelinfo["id"],
		    resultInfo.Level().Value(), levelValue2, util::MakeSQLInterval(resultInfo.Time()),
		    static_cast<int>(resultInfo.ForecastType().Type()), forecastTypeValue, finfo.file_location, host,
		    radonFileFormat, static_cast<int>(finfo.storage_type), HandleNULL(finfo.message_no),
		    HandleNULL(finfo.offset), HandleNULL(finfo.length));
	}

	if (dryRun)
	{
		itsLogger.Trace(query);
		return make_pair(true, radon_record(schema_name, table_name, partition_name, geom_name, stoi(geom_id)));
	}

	try
	{
		itsRadonDB->Execute(query);
		itsRadonDB->Commit();

		// After first insert set record_count to 1, to mark that this partition has data

		if (record_count == "0")
		{
			itsLogger.Trace("Updating as_grid record_count column for " + fullTableName);

			query = fmt::format(
			    "UPDATE as_grid SET record_count = 1 WHERE schema_name = '{}' AND partition_name = '{}' AND "
			    "analysis_time = '{}' AND producer_id = {} AND geometry_id = {}",
			    schema_name, partition_name, analysisTime, resultInfo.Producer().Id(), geom_id);

			itsRadonDB->Execute(query);
		}
	}
	catch (const pqxx::unique_violation& e)
	{
		itsRadonDB->Rollback();

		if (radonVersion >= 20241119)
		{
			const auto agg = resultInfo.Param().Aggregation();
			const auto pt = resultInfo.Param().ProcessingType();

			auto agg_str = HandleNULL(HimanAggregationToRadonAggregation(agg.Type()));
			auto agg_period_str = HandleNULL(agg.TimeDuration());
			auto pt_str = HandleNULL(HimanProcessingTypeToRadonProcessingType(pt.Type()));
			auto pt_value_str = HandleNULL(pt.Value());
			auto pt_value2_str = HandleNULL(pt.Value2());

			query = fmt::format(
			    "UPDATE {} SET "
			    "file_location = '{}', "
			    "file_server = '{}', "
			    "file_format_id = {}, "
			    "file_protocol_id = {}, "
			    "message_no = {}, "
			    "byte_offset = {}, "
			    "byte_length = {}, "
			    "aggregation_id = {}, "
			    "aggregation_period = {}, "
			    "processing_type_id = {}, "
			    "processing_type_value = {}, "
			    "processing_type_value2 = {} WHERE "
			    "producer_id = {} AND "
			    "analysis_time = '{}' AND "
			    "geometry_id = {} AND "
			    "param_id = {} AND "
			    "level_id = {} AND "
			    "level_value = {} AND "
			    "level_value2 = {} AND "
			    "forecast_period = '{}' AND "
			    "forecast_type_id = {} AND "
			    "forecast_type_value = {}",
			    fullTableName, finfo.file_location, host, radonFileFormat, static_cast<int>(finfo.storage_type),
			    HandleNULL(finfo.message_no), HandleNULL(finfo.offset), HandleNULL(finfo.length), agg_str,
			    agg_period_str, pt_str, pt_value_str, pt_value2_str, resultInfo.Producer().Id(), analysisTime, geom_id,
			    resultInfo.Param().Id(), levelinfo["id"], resultInfo.Level().Value(), levelValue2,
			    util::MakeSQLInterval(resultInfo.Time()), static_cast<int>(resultInfo.ForecastType().Type()),
			    forecastTypeValue);
		}
		else
		{
			query = fmt::format(
			    "UPDATE {} SET "
			    "file_location = '{}', "
			    "file_server = '{}', "
			    "file_format_id = {}, "
			    "file_protocol_id = {}, "
			    "message_no = {}, "
			    "byte_offset = {}, "
			    "byte_length = {} WHERE "
			    "producer_id = {} AND "
			    "analysis_time = '{}' AND "
			    "geometry_id = {} AND "
			    "param_id = {} AND "
			    "level_id = {} AND "
			    "level_value = {} AND "
			    "level_value2 = {} AND "
			    "forecast_period = '{}' AND "
			    "forecast_type_id = {} AND "
			    "forecast_type_value = {}",
			    fullTableName, finfo.file_location, host, radonFileFormat, static_cast<int>(finfo.storage_type),
			    HandleNULL(finfo.message_no), HandleNULL(finfo.offset), HandleNULL(finfo.length),
			    resultInfo.Producer().Id(), analysisTime, geom_id, resultInfo.Param().Id(), levelinfo["id"],
			    resultInfo.Level().Value(), levelValue2, util::MakeSQLInterval(resultInfo.Time()),
			    static_cast<int>(resultInfo.ForecastType().Type()), forecastTypeValue);
		}

		itsRadonDB->Execute(query);
		itsRadonDB->Commit();
	}

	return make_pair(true, radon_record(schema_name, table_name, partition_name, geom_name, stoi(geom_id)));
}

template pair<bool, radon_record> radon::SaveGrid<double>(const info<double>&, const file_information&, const string&,
                                                          bool);
template pair<bool, radon_record> radon::SaveGrid<float>(const info<float>&, const file_information&, const string&,
                                                         bool);
template pair<bool, radon_record> radon::SaveGrid<short>(const info<short>&, const file_information&, const string&,
                                                         bool);
template pair<bool, radon_record> radon::SaveGrid<unsigned char>(const info<unsigned char>&, const file_information&,
                                                                 const string&, bool);

std::string radon::GetVersion() const
{
	const std::string query("SELECT radon_version_f()");
	itsRadonDB->Query(query);

	const std::vector<std::string> row = itsRadonDB->FetchRow();

	if (row.empty())
	{
		throw std::out_of_range("radon version not defined");
	}

	return row[0];
}
